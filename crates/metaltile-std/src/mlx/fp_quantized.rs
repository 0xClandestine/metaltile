//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! FP quantized benchmark — #[kernel] DSL vs MLX metal/fp_quantized.metal

use metaltile::kernel;

#[kernel]
pub fn mt_fp4_quant_dequant(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let x = load(inp[gid]);
    let ax = abs(x);
    let group_max = simd_max(ax);
    let inv_scale = select(group_max > 0.0f32, 6.0f32 / group_max, 0.0f32);
    let norm = ax * inv_scale;
    let q = select(
        norm < 0.25f32,
        0.0f32,
        select(
            norm < 0.75f32,
            0.5f32,
            select(
                norm < 1.25f32,
                1.0f32,
                select(
                    norm < 1.75f32,
                    1.5f32,
                    select(
                        norm < 2.5f32,
                        2.0f32,
                        select(norm < 3.5f32, 3.0f32, select(norm < 5.0f32, 4.0f32, 6.0f32)),
                    ),
                ),
            ),
        ),
    );
    let sign = select(x < 0.0f32, -1.0f32, 1.0f32);
    let result = sign * q * (group_max / 6.0f32);
    store(out[gid], result);
}

// ─── mt_fp8_quant_dequant — fp8 (e4m3 / e5m2) quant + dequant ─────────────
//
// The fp8 counterpart of `mt_fp4_quant_dequant` above — closes the fp8
// gap in the `fp_quantized` audit row. fp8 is the standard inference
// activation/KV format on Hopper / Blackwell-class hardware and the
// MLX `fp_quantized.metal` family ships both variants:
//
//   - **e4m3** — 1 sign · 4 exponent · 3 mantissa. Bias 7, max ±448.
//     No infinities (the all-ones exponent is reused for finite
//     values); higher precision, narrower range. The default for
//     weights / activations.
//   - **e5m2** — 1 sign · 5 exponent · 2 mantissa. Bias 15, max ±57344.
//     Wider dynamic range, coarser mantissa; used where range matters
//     (gradients, some KV-cache layouts).
//
// No new DSL dtype is needed: fp8 quantize-dequantize is a pure
// arithmetic transform expressible with `floor` / `log2` / `exp2` /
// `round`, all already in the DSL. The round-trip emulates fp8
// rounding directly on f32 — **round each value's mantissa to the
// format's mantissa-bit count**:
//
//   1. Per-group max-scale the magnitude into the fp8 range
//      (`group_max → fp8_max`), as `mt_fp4_quant_dequant` does for fp4.
//   2. `e = clamp(floor(log2(norm)), e_min, e_max)` — the binade.
//   3. `quantum = exp2(e - mantissa_bits)` — the representable step at
//      that binade. Clamping `e` at `e_min` gives correct subnormal
//      behaviour (fixed quantum below the smallest normal); clamping at
//      `e_max` saturates large values to `fp8_max`.
//   4. `q = round(norm / quantum) * quantum` — the fp8 grid point.
//   5. Rescale by `group_max / fp8_max` and reapply the sign.
//
// This is exact for every normal and subnormal fp8 value; it saturates
// (rather than producing NaN/Inf) out-of-range inputs — matching MLX's
// `mxfp8` / `nvfp8` quantize-dequantize, which has no inf either.
//
// Constexpr layout — identical to `mt_fp4_quant_dequant`:
//   inp / out — [n], f32. group = one simdgroup (32 lanes), `simd_max`
//   gives the per-group amax.
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D**, but each group of 32 consecutive elements is one
//   simdgroup — `simd_max` reduces the group amax. Dispatch
//   `grid = [n, 1, 1]`, `tpg = [32, 1, 1]` (matching the fp4 kernel's
//   `tpg = 32`); `n` must be a multiple of 32.
// - **`mantissa_bits`, `e_min`, `e_max`, `fp8_max`** are baked per
//   format by the `fp8_kernel!` macro — a wrong set silently rounds wrong.
macro_rules! fp8_kernel {
    ($name:ident, $subop:literal, $mant:literal, $emin:literal, $emax:literal, $fp8max:literal) => {
        // `#[kernel(bench(...))]` registers both the kernel and BenchSpec
        // for this non-generic kernel (the attribute handles the
        // no-DType `kernel_ir_for` signature) — so each fp8 format gets
        // its own bench row, like `mt_fp4_quant_dequant`. No `mlx=` /
        // `metal_file=`: fp8 has no MLX side-by-side counterpart.
        // Single-line `#[kernel(bench(...))]` — rustfmt's indent tracking inside
        // `macro_rules!` bodies is non-idempotent for multi-line attributes
        // (it adds 8 spaces every `fmt` run); a single line is stable.
        #[kernel]
        pub fn $name(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
            let gid = program_id::<0>();
            let x = load(inp[gid]);
            let ax = abs(x);
            let group_max = simd_max(ax);

            // Scale the magnitude so the group's largest value maps to
            // the format's max representable magnitude.
            let inv_scale = select(group_max > 0.0f32, $fp8max / group_max, 0.0f32);
            let norm = ax * inv_scale;

            // Round `norm` to the fp8 grid: find its binade, clamp to
            // the representable exponent range, snap the mantissa to
            // `$mant` bits via the per-binade quantum.
            let raw_e = floor(log2(norm));
            // Clamp the exponent: at `e_min` the quantum is fixed
            // (subnormals); at `e_max` large values saturate.
            let e_lo = select(raw_e < $emin, $emin, raw_e);
            let e = select(e_lo > $emax, $emax, e_lo);
            let quantum = exp2(e - $mant);
            let snapped = round(norm / quantum) * quantum;
            // norm == 0 → log2 is -inf → e clamps to e_min, round(0)=0,
            // so `snapped` is already 0; the select keeps it explicit.
            let q = select(norm > 0.0f32, snapped, 0.0f32);
            // Saturate anything that still exceeds the format max.
            let q_clamped = select(q > $fp8max, $fp8max, q);

            let sign = select(x < 0.0f32, -1.0f32, 1.0f32);
            let result = sign * q_clamped * (group_max / $fp8max);
            store(out[gid], result);
        }
    };
}

// e4m3 — 3 mantissa bits, exponent range [-6, 8] (bias 7; e4m3 reuses
// the all-ones exponent for finite values so the top binade is 8),
// max magnitude 448.
fp8_kernel!(mt_fp8_e4m3_quant_dequant, "fp8_e4m3", 3.0f32, -6.0f32, 8.0f32, 448.0f32);
// e5m2 — 2 mantissa bits, exponent range [-14, 15] (bias 15), max
// magnitude 57344.
fp8_kernel!(mt_fp8_e5m2_quant_dequant, "fp8_e5m2", 2.0f32, -14.0f32, 15.0f32, 57344.0f32);

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    // ── fp4 oracle helpers ───────────────────────────────────────────────────

    const FP4_CODEBOOK: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    fn fp4_snap(norm: f32) -> f32 {
        if norm < 0.25 {
            0.0
        } else if norm < 0.75 {
            0.5
        } else if norm < 1.25 {
            1.0
        } else if norm < 1.75 {
            1.5
        } else if norm < 2.5 {
            2.0
        } else if norm < 3.5 {
            3.0
        } else if norm < 5.0 {
            4.0
        } else {
            6.0
        }
    }

    fn oracle_fp4(inp: &[f32]) -> Vec<f32> {
        assert!(inp.len().is_multiple_of(32), "input length must be a multiple of 32");
        let mut out = vec![0.0f32; inp.len()];
        for (gi, group) in inp.chunks_exact(32).enumerate() {
            let group_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let inv_scale = if group_max > 0.0 { 6.0 / group_max } else { 0.0 };
            let rescale = group_max / 6.0;
            for (i, &x) in group.iter().enumerate() {
                let norm = x.abs() * inv_scale;
                let q = fp4_snap(norm);
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                out[gi * 32 + i] = sign * q * rescale;
            }
        }
        out
    }

    fn synthetic_group_fp4(seed: usize) -> Vec<f32> {
        (0..32)
            .map(|i| {
                let v = ((i * 7 + seed * 11) % 33) as f32 * 0.03 - 0.46;
                match i % 4 {
                    0 => v * 10.0,
                    1 => v * 0.05,
                    2 => 0.0,
                    _ => v,
                }
            })
            .collect()
    }

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }

    // ── fp4 tests ────────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/fp4_quant_dequant/round_trip", dtypes = [f32], tol = 1e-3)]
    fn test_fp4_round_trip(dt: DType) -> TestSetup {
        let inp: Vec<f32> = (0..4).flat_map(synthetic_group_fp4).collect();
        let n = inp.len() as u32;
        let expected = oracle_fp4(&inp);
        let kernel = mt_fp4_quant_dequant::kernel_ir_for();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp), DType::F32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; inp.len()]), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("n", n)
            .grid_3d((inp.len() / 32) as u32, 1, 1, [32, 1, 1])
    }

    #[test_kernel(name = "mlx/fp4_quant_dequant/codebook_roundtrip", dtypes = [f32], tol = 1e-4)]
    fn test_fp4_codebook_roundtrip(dt: DType) -> TestSetup {
        let scale = 4.0f32;
        let inp: Vec<f32> = (0..32)
            .map(|i| {
                let mag = FP4_CODEBOOK[i % 8] * scale;
                if i % 2 == 0 { mag } else { -mag }
            })
            .collect();
        let n = inp.len() as u32;
        let expected = oracle_fp4(&inp);
        let kernel = mt_fp4_quant_dequant::kernel_ir_for();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp), DType::F32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; inp.len()]), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("n", n)
            .grid_3d(1, 1, 1, [32, 1, 1])
    }

    #[test_kernel(name = "mlx/fp4_quant_dequant/zero_group", dtypes = [f32], tol = 0.0)]
    fn test_fp4_zero_group(dt: DType) -> TestSetup {
        let inp = vec![0.0f32; 32];
        let n = inp.len() as u32;
        let expected = oracle_fp4(&inp);
        let kernel = mt_fp4_quant_dequant::kernel_ir_for();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp), DType::F32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; inp.len()]), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("n", n)
            .grid_3d(1, 1, 1, [32, 1, 1])
    }

    // ── fp8 oracle helpers ───────────────────────────────────────────────────

    struct Fp8Fmt {
        mantissa_bits: f32,
        e_min: f32,
        e_max: f32,
        fp8_max: f32,
    }
    const E4M3: Fp8Fmt = Fp8Fmt { mantissa_bits: 3.0, e_min: -6.0, e_max: 8.0, fp8_max: 448.0 };
    const E5M2: Fp8Fmt = Fp8Fmt { mantissa_bits: 2.0, e_min: -14.0, e_max: 15.0, fp8_max: 57344.0 };

    fn oracle_fp8_round_trip(inp: &[f32], fmt: &Fp8Fmt) -> Vec<f32> {
        assert!(inp.len().is_multiple_of(32), "input length must be a multiple of 32");
        let mut out = vec![0.0f32; inp.len()];
        for (gi, group) in inp.chunks_exact(32).enumerate() {
            let group_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let inv_scale = if group_max > 0.0 { fmt.fp8_max / group_max } else { 0.0 };
            let rescale = group_max / fmt.fp8_max;
            for (i, &x) in group.iter().enumerate() {
                let norm = x.abs() * inv_scale;
                let q = if norm > 0.0 {
                    let raw_e = norm.log2().floor();
                    let e = raw_e.clamp(fmt.e_min, fmt.e_max);
                    let quantum = (e - fmt.mantissa_bits).exp2();
                    (norm / quantum).round() * quantum
                } else {
                    0.0
                };
                let q_clamped = q.min(fmt.fp8_max);
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                out[gi * 32 + i] = sign * q_clamped * rescale;
            }
        }
        out
    }

    fn synthetic_group_fp8(seed: usize) -> Vec<f32> {
        (0..32)
            .map(|i| {
                let v = ((i * 37 + seed * 13) % 100) as f32 * 0.01 - 0.5;
                match i % 4 {
                    0 => v * 100.0,
                    1 => v * 0.001,
                    2 => 0.0,
                    _ => v,
                }
            })
            .collect()
    }

    // ── fp8 e4m3 tests ───────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/fp8_e4m3_quant_dequant/round_trip", dtypes = [f32], tol = 1e-2)]
    fn test_fp8_e4m3_round_trip(dt: DType) -> TestSetup {
        let inp: Vec<f32> = (0..4).flat_map(synthetic_group_fp8).collect();
        let n = inp.len() as u32;
        let expected = oracle_fp8_round_trip(&inp, &E4M3);
        let kernel = mt_fp8_e4m3_quant_dequant::kernel_ir_for();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp), DType::F32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; inp.len()]), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("n", n)
            .grid_3d(inp.len() as u32, 1, 1, [32, 1, 1])
    }

    // ── fp8 e5m2 tests ───────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/fp8_e5m2_quant_dequant/round_trip", dtypes = [f32], tol = 1e-1)]
    fn test_fp8_e5m2_round_trip(dt: DType) -> TestSetup {
        let inp: Vec<f32> = (0..4).flat_map(synthetic_group_fp8).collect();
        let n = inp.len() as u32;
        let expected = oracle_fp8_round_trip(&inp, &E5M2);
        let kernel = mt_fp8_e5m2_quant_dequant::kernel_ir_for();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp), DType::F32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; inp.len()]), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("n", n)
            .grid_3d(inp.len() as u32, 1, 1, [32, 1, 1])
    }
}

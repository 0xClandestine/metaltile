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
        // `#[kernel]` registers both the kernel and BenchSpec
        // for this non-generic kernel (the attribute handles the
        // no-DType `kernel_ir_for` signature) — so each fp8 format gets
        // its own bench row, like `mt_fp4_quant_dequant`. No `mlx=` /
        // `metal_file=`: fp8 has no MLX side-by-side counterpart.
        // Single-line `#[kernel]` — rustfmt's indent tracking inside
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

/// New-syntax correctness for `mt_fp4_quant_dequant` (Grid3D, one 32-lane
/// simdgroup per group; per-group amax → fp4-codebook snap → rescale). The
/// oracle replays the exact codebook; inputs are kept clear of codebook
/// decision boundaries so an f32 ULP can't flip a cell. The fp8 e4m3/e5m2
/// variants are bench-only (their parameterised codebooks would need a
/// separate oracle — covered by their legacy tests).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_fp4_quant_dequant, mt_fp8_e4m3_quant_dequant, mt_fp8_e5m2_quant_dequant};

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

    fn synthetic_group(seed: usize) -> Vec<f32> {
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

    #[test_kernel(dtypes = [f32], tol = 1e-4)]
    fn test_mt_fp4_quant_dequant(_dt: DType) -> TestSetup {
        let inp: Vec<f32> = (0..4).flat_map(synthetic_group).collect();
        let n = inp.len();
        // Per-32-element-simdgroup amax-scale → codebook snap → rescale.
        let mut expected = vec![0.0f32; n];
        for (gi, group) in inp.chunks_exact(32).enumerate() {
            let group_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let inv_scale = if group_max > 0.0 { 6.0 / group_max } else { 0.0 };
            let rescale = group_max / 6.0;
            for (i, &x) in group.iter().enumerate() {
                let q = fp4_snap(x.abs() * inv_scale);
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                expected[gi * 32 + i] = sign * q * rescale;
            }
        }
        TestSetup::new(mt_fp4_quant_dequant::kernel_ir_for())
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec(
                "inp",
                inp.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", n, DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec(
                "out",
                expected.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F32,
            ))
            .grid_3d((n / 32) as u32, 1, 1, [32, 1, 1])
    }

    /// CPU oracle for the fp8 round-trip: per-32-element amax-scale, then snap
    /// each magnitude to the `mant`-bit-mantissa fp8 grid (binade via
    /// `floor(log2)`, exponent clamped to `[emin, emax]`, `round` to the
    /// per-binade quantum, saturate at `fp8max`), then rescale. Mirrors the
    /// `fp8_kernel!` body exactly.
    fn fp8_oracle(inp: &[f32], mant: f32, emin: f32, emax: f32, fp8max: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; inp.len()];
        for (gi, group) in inp.chunks_exact(32).enumerate() {
            let group_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let inv_scale = if group_max > 0.0 { fp8max / group_max } else { 0.0 };
            let rescale = group_max / fp8max;
            for (i, &x) in group.iter().enumerate() {
                let norm = x.abs() * inv_scale;
                let q = if norm > 0.0 {
                    let e = norm.log2().floor().clamp(emin, emax);
                    let quantum = (e - mant).exp2();
                    ((norm / quantum).round() * quantum).min(fp8max)
                } else {
                    0.0
                };
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                out[gi * 32 + i] = sign * q * rescale;
            }
        }
        out
    }

    fn fp8_setup(
        kernel: metaltile::core::ir::Kernel,
        mant: f32,
        emin: f32,
        emax: f32,
        fp8max: f32,
    ) -> TestSetup {
        let inp: Vec<f32> = (0..4).flat_map(synthetic_group).collect();
        let n = inp.len();
        let expected = fp8_oracle(&inp, mant, emin, emax, fp8max);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec(
                "inp",
                inp.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", n, DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec(
                "out",
                expected.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F32,
            ))
            .grid_3d((n / 32) as u32, 1, 1, [32, 1, 1])
    }

    // e4m3: 3 mantissa bits, exponent [-6, 8], max ±448.
    #[test_kernel(dtypes = [f32], tol = 1e-3)]
    fn test_mt_fp8_e4m3_quant_dequant(_dt: DType) -> TestSetup {
        fp8_setup(mt_fp8_e4m3_quant_dequant::kernel_ir_for(), 3.0, -6.0, 8.0, 448.0)
    }
    // e5m2: 2 mantissa bits, exponent [-14, 15], max ±57344.
    #[test_kernel(dtypes = [f32], tol = 1e-3)]
    fn test_mt_fp8_e5m2_quant_dequant(_dt: DType) -> TestSetup {
        fp8_setup(mt_fp8_e5m2_quant_dequant::kernel_ir_for(), 2.0, -14.0, 15.0, 57344.0)
    }
}

/// New-syntax benchmarks for the fp-quantize round-trip kernels.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::{mt_fp4_quant_dequant, mt_fp8_e4m3_quant_dequant, mt_fp8_e5m2_quant_dequant};
    use crate::bench_types::{InputDomain, input_buffer};

    const QUANT_N: usize = 64 * 1024 * 1024;

    fn qb(kernel: Kernel) -> BenchSetup {
        let n = QUANT_N;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", n, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, DType::F32).output())
            .constexpr("n", n as u32)
            .grid_3d((n / 32) as u32, 1, 1, [32, 1, 1])
            .bytes_moved((2 * n * 4) as u64)
    }

    // fp4 carries the MLX `metal/fp_quantized.metal`
    // `nvfp4_quantize_dequantize_float_gs_16_b_4` reference. The MLX kernel is
    // 2-buffer (`w`[[0]] input, `out`[[1]] output, both f32) with no scalars and
    // no function constants. It is dispatched 2D: `index = tidx.x + grid_dim.x *
    // tidx.y`, so the legacy `[1, n/32, 1]` threadgroups × `[32,1,1]` tpg gives
    // `grid_dim.x = 32` and each 32-lane threadgroup is one simdgroup covering 32
    // consecutive elements — the same element-to-simdgroup grouping the MT Grid3D
    // dispatch uses. `inp` is shared by name with the MT input below.
    //
    // The input is seeded `Signed` (period-8 pattern `[-3..3]`) rather than
    // `qb`'s raw `BenchBuffer::random` (random f32 *bytes* alias to inf/nan, which
    // would poison the quantize round-trip and the A/B). The pattern's period (8)
    // divides every group boundary, so the per-group amax is a uniform 3.0 — which
    // also neutralises the gs16-vs-gs32 scale split described next.
    //
    // NOTE (semantic divergence): MLX `nvfp4` quantises at **group_size 16**
    // (`use_mx_scale = group_size == 32` is false → each 32-lane simdgroup is
    // split into two 16-lane amax groups), whereas `mt_fp4_quant_dequant` takes a
    // full **32-lane** `simd_max` (group_size 32). With a non-uniform input the
    // two would pick different per-group scales near a 16-boundary and disagree by
    // up to a codebook step; the `Signed` pattern's uniform amax avoids that, so
    // the legacy tol=0.5 dequant-band floor holds for the A/B.
    #[bench(name = "mlx/fp_quantized/fp4", dtypes = [f32])]
    fn bench_fp4(_dt: DType) -> BenchSetup {
        let n = QUANT_N;
        BenchSetup::new(mt_fp4_quant_dequant::kernel_ir_for())
            .mode(KernelMode::Grid3D)
            .buffer(input_buffer("inp", n, DType::F32, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", n, DType::F32).output())
            .constexpr("n", n as u32)
            .grid_3d((n / 32) as u32, 1, 1, [32, 1, 1])
            .bytes_moved((2 * n * 4) as u64)
            .with_reference(
                RefKernel::new(
                    "nvfp4_quantize_dequantize_float_gs_16_b_4".to_string(),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/fp_quantized.metal")),
                )
                // w[[0]] shared by name with the MT `inp`; out[[1]] fresh.
                .buffer(BenchBuffer::zeros("inp", n, DType::F32))
                .buffer(BenchBuffer::zeros("out", n, DType::F32).output())
                // 2D: [1, n/32, 1] threadgroups × [32,1,1] → grid_dim.x = 32.
                .grid(Grid::new_3d(1, (n / 32) as u32, 1, [32, 1, 1]))
                .tol(0.5),
            )
    }
    #[bench(name = "mlx/fp_quantized/fp8_e4m3", dtypes = [f32])]
    fn bench_fp8_e4m3(_dt: DType) -> BenchSetup { qb(mt_fp8_e4m3_quant_dequant::kernel_ir_for()) }
    #[bench(name = "mlx/fp_quantized/fp8_e5m2", dtypes = [f32])]
    fn bench_fp8_e5m2(_dt: DType) -> BenchSetup { qb(mt_fp8_e5m2_quant_dequant::kernel_ir_for()) }
}

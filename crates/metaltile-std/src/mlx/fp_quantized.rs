//! FP quantized benchmark — #[kernel] DSL vs MLX metal/fp_quantized.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="fp_quantized",
    subop="fp4_quant_dequant",
    class=FpQuantized,
    n=1048576,
    tpg=32,
    tol=0.5,
    mlx="nvfp4_quantize_dequantize_float_gs_16_b_4",
    metal_file="fp_quantized.metal",
    dtypes=crate::spec::F32_ONLY,
)]
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
        // `#[bench_kernel]` placed before `#[kernel]` registers a BenchSpec
        // for this non-generic kernel (the attribute handles the
        // no-DType `kernel_ir_for` signature) — so each fp8 format gets
        // its own bench row, like `mt_fp4_quant_dequant`. No `mlx=` /
        // `metal_file=`: fp8 has no MLX side-by-side counterpart.
        // Single-line `#[bench_kernel]` — rustfmt's indent tracking inside
        // `macro_rules!` bodies is non-idempotent for multi-line attributes
        // (it adds 8 spaces every `fmt` run); a single line is stable.
        #[bench_kernel(op = "fp_quantized", subop = $subop, class = FpQuantized, n = 1048576, tpg = 32, tol = 0.05, dtypes = crate::spec::F32_ONLY)]
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

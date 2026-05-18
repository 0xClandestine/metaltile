//! Quantized MatVec benchmark — #[kernel] DSL vs MLX metal/quantized.metal

use metaltile::{bench_kernel, kernel};
// (out_dim, in_dim) pairs. 4096² = baseline reference. Other rows are
// production hot-paths in Qwen3-class inference:
//   - 5120²       Qwen3-8B/14B attention proj (Q/K/V/O), MLP.gate/up at hidden
//   - 14336×5120  Qwen3-8B/14B MLP up_proj
//   - 5120×14336  Qwen3-8B/14B MLP down_proj
//   - 27648×5120  Qwen3-coder-30B MoE expert up_proj
static QUANTIZED_SHAPES: &[(usize, usize)] =
    &[(4096, 4096), (5120, 5120), (14336, 5120), (5120, 14336), (27648, 5120)];

#[bench_kernel(
    op="quantized",
    subop="qmv",
    class=QuantizedMatVec,
    shapes=&QUANTIZED_SHAPES,
    group_size=64,
    // tpg=64 = 2 simdgroups × 32 lanes. Kernel processes 8 output rows
    // per TG (each simdgroup handles 4 rows independently, indexed by
    // simd_id). Dispatcher grid is `m/8` TGs — matches MLX qmv_fast.
    tpg=64,
    tol=1e-3,
    mlx="affine_qmv_fast_float16_t_gs_64_b_4_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16],
)]
#[kernel]
pub fn mt_qmv<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] gs_per_row: u32,
) {
    // Multi-row tile: 8 output rows per TG, 2 simdgroups × 32 lanes.
    // Each simdgroup independently handles 4 rows (indexed by simd_id).
    // Each lane caches 16 X values in registers per outer block and
    // reuses them across all 4 rows' qdot accumulation — 4× reduction
    // in X bandwidth + 8× fewer TGs vs the previous 1-row-per-TG layout.
    // Matches MLX qmv_fast geometry (`quantized.h:749`) exactly.
    //
    // Per outer iter: 16 X loads (once per simdgroup) + per-row (2
    // weight packs + 16 int4 extracts + 16 FMAs into q_dot + 1 add into
    // x_sum + 1 scale + 1 bias + 1 partial accumulation). Block = 16 X
    // × 32 lanes = 512 K elements.
    //
    // Math: result_row = sum_g (scale_g * sum_{i in g} q_i*x_i
    //                          + bias_g * sum_{i in g} x_i)
    // The bias hoist (algebraic split) eliminates one FMA per int4 in
    // the hot loop — matches MLX `qdot` in quantized.h:235.
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let packs_per_row = k / 8u32;
    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    for _b in range(0u32, k, 512u32) {
        // 16 X loads — consecutive in IR for vectorize fusion (4× float4).
        let xb = _b + lane_x_off;
        let xi0 = xb;
        let xi1 = xb + 1u32;
        let xi2 = xb + 2u32;
        let xi3 = xb + 3u32;
        let xi4 = xb + 4u32;
        let xi5 = xb + 5u32;
        let xi6 = xb + 6u32;
        let xi7 = xb + 7u32;
        let xi8 = xb + 8u32;
        let xi9 = xb + 9u32;
        let xi10 = xb + 10u32;
        let xi11 = xb + 11u32;
        let xi12 = xb + 12u32;
        let xi13 = xb + 13u32;
        let xi14 = xb + 14u32;
        let xi15 = xb + 15u32;
        // Mask-without-shift: X pre-scaled by inverse nibble position, weight
        // mask returns nibble × position-power. Saves 7 shifts per pack × 2
        // packs × 4 rows = 56 shifts per outer iter. Mirrors MLX `qdot` for
        // bits=4 (`quantized.h:235-244`).
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
        // Incremental xs accumulator from raw loads — saves 12 muls vs the
        // reconstruction-from-scaled approach. Raw x dies right after the
        // scale + xs accumulator both consume it.
        // Cast T-typed X to f32 at load time for the inner FMA chain;
        // accumulators stay in f32 regardless of T. Identity for T=f32.
        let x0 = load(x[xi0]).cast::<f32>();
        let x1_raw = load(x[xi1]).cast::<f32>();
        let x2_raw = load(x[xi2]).cast::<f32>();
        let x3_raw = load(x[xi3]).cast::<f32>();
        let x4 = load(x[xi4]).cast::<f32>();
        let x5_raw = load(x[xi5]).cast::<f32>();
        let x6_raw = load(x[xi6]).cast::<f32>();
        let x7_raw = load(x[xi7]).cast::<f32>();
        let x8 = load(x[xi8]).cast::<f32>();
        let x9_raw = load(x[xi9]).cast::<f32>();
        let x10_raw = load(x[xi10]).cast::<f32>();
        let x11_raw = load(x[xi11]).cast::<f32>();
        let x12 = load(x[xi12]).cast::<f32>();
        let x13_raw = load(x[xi13]).cast::<f32>();
        let x14_raw = load(x[xi14]).cast::<f32>();
        let x15_raw = load(x[xi15]).cast::<f32>();
        let xs = x0
            + x1_raw
            + x2_raw
            + x3_raw
            + x4
            + x5_raw
            + x6_raw
            + x7_raw
            + x8
            + x9_raw
            + x10_raw
            + x11_raw
            + x12
            + x13_raw
            + x14_raw
            + x15_raw;
        let x1 = x1_raw * s_16;
        let x2 = x2_raw * s_256;
        let x3 = x3_raw * s_4096;
        let x5 = x5_raw * s_16;
        let x6 = x6_raw * s_256;
        let x7 = x7_raw * s_4096;
        let x9 = x9_raw * s_16;
        let x10 = x10_raw * s_256;
        let x11 = x11_raw * s_4096;
        let x13 = x13_raw * s_16;
        let x14 = x14_raw * s_256;
        let x15 = x15_raw * s_4096;

        // Each lane covers 16 X values within a single gs=64 group.
        // 4 lanes per group, 8 groups per block (32 lanes × 16 = 512).
        let g = xb / 64u32;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 ──
        let p00 = load(w[w_base0 + pack_off]);
        let p01 = load(w[w_base0 + pack_off + 1u32]);
        let p00_hi = p00 >> 16u32;
        let p01_hi = p01 >> 16u32;
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        // Lo half (nibbles 0-3): mask 0xf, 0xf0, 0xf00, 0xf000 — values *1, *16, *256, *4096.
        // Multiplied against pre-scaled x[0..3] (*1, *1/16, *1/256, *1/4096) → q*x.
        let q00 = (p00 & 15u32).cast::<f32>();
        let q01 = (p00 & 240u32).cast::<f32>();
        let q02 = (p00 & 3840u32).cast::<f32>();
        let q03 = (p00 & 61440u32).cast::<f32>();
        let q04 = (p00_hi & 15u32).cast::<f32>();
        let q05 = (p00_hi & 240u32).cast::<f32>();
        let q06 = (p00_hi & 3840u32).cast::<f32>();
        let q07 = (p00_hi & 61440u32).cast::<f32>();
        let q08 = (p01 & 15u32).cast::<f32>();
        let q09 = (p01 & 240u32).cast::<f32>();
        let q010 = (p01 & 3840u32).cast::<f32>();
        let q011 = (p01 & 61440u32).cast::<f32>();
        let q012 = (p01_hi & 15u32).cast::<f32>();
        let q013 = (p01_hi & 240u32).cast::<f32>();
        let q014 = (p01_hi & 3840u32).cast::<f32>();
        let q015 = (p01_hi & 61440u32).cast::<f32>();
        let qd0 = q00 * x0
            + q01 * x1
            + q02 * x2
            + q03 * x3
            + q04 * x4
            + q05 * x5
            + q06 * x6
            + q07 * x7
            + q08 * x8
            + q09 * x9
            + q010 * x10
            + q011 * x11
            + q012 * x12
            + q013 * x13
            + q014 * x14
            + q015 * x15;
        acc0 = acc0 + s0 * qd0 + bi0 * xs;

        // ── Row 1 ──
        let p10 = load(w[w_base1 + pack_off]);
        let p11 = load(w[w_base1 + pack_off + 1u32]);
        let p10_hi = p10 >> 16u32;
        let p11_hi = p11 >> 16u32;
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let q10 = (p10 & 15u32).cast::<f32>();
        let q11 = (p10 & 240u32).cast::<f32>();
        let q12 = (p10 & 3840u32).cast::<f32>();
        let q13 = (p10 & 61440u32).cast::<f32>();
        let q14 = (p10_hi & 15u32).cast::<f32>();
        let q15 = (p10_hi & 240u32).cast::<f32>();
        let q16 = (p10_hi & 3840u32).cast::<f32>();
        let q17 = (p10_hi & 61440u32).cast::<f32>();
        let q18 = (p11 & 15u32).cast::<f32>();
        let q19 = (p11 & 240u32).cast::<f32>();
        let q110 = (p11 & 3840u32).cast::<f32>();
        let q111 = (p11 & 61440u32).cast::<f32>();
        let q112 = (p11_hi & 15u32).cast::<f32>();
        let q113 = (p11_hi & 240u32).cast::<f32>();
        let q114 = (p11_hi & 3840u32).cast::<f32>();
        let q115 = (p11_hi & 61440u32).cast::<f32>();
        let qd1 = q10 * x0
            + q11 * x1
            + q12 * x2
            + q13 * x3
            + q14 * x4
            + q15 * x5
            + q16 * x6
            + q17 * x7
            + q18 * x8
            + q19 * x9
            + q110 * x10
            + q111 * x11
            + q112 * x12
            + q113 * x13
            + q114 * x14
            + q115 * x15;
        acc1 = acc1 + s1 * qd1 + bi1 * xs;

        // ── Row 2 ──
        let p20 = load(w[w_base2 + pack_off]);
        let p21 = load(w[w_base2 + pack_off + 1u32]);
        let p20_hi = p20 >> 16u32;
        let p21_hi = p21 >> 16u32;
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let q20 = (p20 & 15u32).cast::<f32>();
        let q21 = (p20 & 240u32).cast::<f32>();
        let q22 = (p20 & 3840u32).cast::<f32>();
        let q23 = (p20 & 61440u32).cast::<f32>();
        let q24 = (p20_hi & 15u32).cast::<f32>();
        let q25 = (p20_hi & 240u32).cast::<f32>();
        let q26 = (p20_hi & 3840u32).cast::<f32>();
        let q27 = (p20_hi & 61440u32).cast::<f32>();
        let q28 = (p21 & 15u32).cast::<f32>();
        let q29 = (p21 & 240u32).cast::<f32>();
        let q210 = (p21 & 3840u32).cast::<f32>();
        let q211 = (p21 & 61440u32).cast::<f32>();
        let q212 = (p21_hi & 15u32).cast::<f32>();
        let q213 = (p21_hi & 240u32).cast::<f32>();
        let q214 = (p21_hi & 3840u32).cast::<f32>();
        let q215 = (p21_hi & 61440u32).cast::<f32>();
        let qd2 = q20 * x0
            + q21 * x1
            + q22 * x2
            + q23 * x3
            + q24 * x4
            + q25 * x5
            + q26 * x6
            + q27 * x7
            + q28 * x8
            + q29 * x9
            + q210 * x10
            + q211 * x11
            + q212 * x12
            + q213 * x13
            + q214 * x14
            + q215 * x15;
        acc2 = acc2 + s2 * qd2 + bi2 * xs;

        // ── Row 3 ──
        let p30 = load(w[w_base3 + pack_off]);
        let p31 = load(w[w_base3 + pack_off + 1u32]);
        let p30_hi = p30 >> 16u32;
        let p31_hi = p31 >> 16u32;
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let q30 = (p30 & 15u32).cast::<f32>();
        let q31 = (p30 & 240u32).cast::<f32>();
        let q32 = (p30 & 3840u32).cast::<f32>();
        let q33 = (p30 & 61440u32).cast::<f32>();
        let q34 = (p30_hi & 15u32).cast::<f32>();
        let q35 = (p30_hi & 240u32).cast::<f32>();
        let q36 = (p30_hi & 3840u32).cast::<f32>();
        let q37 = (p30_hi & 61440u32).cast::<f32>();
        let q38 = (p31 & 15u32).cast::<f32>();
        let q39 = (p31 & 240u32).cast::<f32>();
        let q310 = (p31 & 3840u32).cast::<f32>();
        let q311 = (p31 & 61440u32).cast::<f32>();
        let q312 = (p31_hi & 15u32).cast::<f32>();
        let q313 = (p31_hi & 240u32).cast::<f32>();
        let q314 = (p31_hi & 3840u32).cast::<f32>();
        let q315 = (p31_hi & 61440u32).cast::<f32>();
        let qd3 = q30 * x0
            + q31 * x1
            + q32 * x2
            + q33 * x3
            + q34 * x4
            + q35 * x5
            + q36 * x6
            + q37 * x7
            + q38 * x8
            + q39 * x9
            + q310 * x10
            + q311 * x11
            + q312 * x12
            + q313 * x13
            + q314 * x14
            + q315 * x15;
        acc3 = acc3 + s3 * qd3 + bi3 * xs;
    }

    // Cross-lane reduction: each row's partial → single value, lane 0 stores.
    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(out[row0], r0.cast::<T>());
        store(out[row1], r1.cast::<T>());
        store(out[row2], r2.cast::<T>());
        store(out[row3], r3.cast::<T>());
    }
}

// ─── mt_affine_dequantize_int4 ─────────────────────────────────────────
//
// One thread per pack (8 nibbles in one uint32). For each output i in
// 0..8: `q = (val >> (i*4)) & 0xf`, then `out[oindex+i] = scale * q + bias`
// where scale/bias are looked up by group index `oindex / group_size`.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 4>` from
// `quantized.h`. Both kernels read the same byte stream and produce the
// same output (MLX views weights as `uint8_t*`, ours as `Tensor<u32>` —
// same bits, different lens).
#[bench_kernel(
    op="affine",
    subop="dequantize_int4",
    class=AffineDequantize,
    bits=4,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=1e-2 — bf16 round-trip error scales with max_q (= 15). At
    // n_groups=4096 the worst-case absolute drift is ~3e-3.
    tol=1e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int4<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    let q0 = (val >> 0u32) & 15u32;
    let q1 = (val >> 4u32) & 15u32;
    let q2 = (val >> 8u32) & 15u32;
    let q3 = (val >> 12u32) & 15u32;
    let q4 = (val >> 16u32) & 15u32;
    let q5 = (val >> 20u32) & 15u32;
    let q6 = (val >> 24u32) & 15u32;
    let q7 = (val >> 28u32) & 15u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

// ─── mt_affine_quantize_int4 ───────────────────────────────────────────
//
// Inverse of dequantize: one threadgroup per group, finds min/max over
// the group, computes scale/bias, then packs 8 nibbles per uint32. The
// per-group nature means no cross-threadgroup sync is needed.
//
// MLX's `affine_quantize` uses a 32-thread simd-group cooperative reduce
// across `group_size` elements; we use the same shape (one threadgroup
// of 32 threads per group) and reduce via `simd_min` / `simd_max`.
//
// Packing: `packs_per_group = group_size / pack_factor = 64 / 8 = 8`
// nibble-packs per group. Lanes 0..7 each pack one uint32 in parallel
// — they re-read the 8 input values for their pack from device memory
// (cheap; the data is already cached after the min/max reduction's
// first load). Eliminating the lane-0 serial loop is the main perf
// difference vs the original implementation.
//
// Restriction: hardcodes group_size=64 and bits=4 in the unrolling
// (`group_size / 32 = 2` values per thread, 8 nibbles per uint32).
// Bigger group sizes or other bit widths follow the same template with
// different constants.
#[bench_kernel(
    op="affine",
    subop="quantize_int4",
    class=AffineQuantize,
    bits=4,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int4<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 15.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());
    }

    // Packs in parallel: lanes 0..packs_per_group each pack one uint32.
    // For group_size=64 → packs_per_group=8, so 8 lanes work in parallel
    // vs the previous lane-0 serial loop over all 8 packs.
    let packs_per_group = group_size / 8u32;
    if lane < packs_per_group {
        let pack_in_base = in_base + lane * 8u32;
        let mut acc = 0u32;
        for k in range(0u32, 8u32, 1u32) {
            let v = load(w[pack_in_base + k]).cast::<f32>();
            let q_f = (v - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 15.0f32, 15.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>();
            acc = acc | (q << (k * 4u32));
        }
        store(out[g_idx * packs_per_group + lane], acc);
    }
}

// ─── mt_affine_dequantize_int8 ─────────────────────────────────────────
//
// One thread per pack (4 bytes in one uint32). Same shape as int4 but
// each pack covers 4 output values instead of 8, and bit-extraction
// shifts by multiples of 8 instead of 4.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 8>`.
#[bench_kernel(
    op="affine",
    subop="dequantize_int8",
    class=AffineDequantize,
    bits=8,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=1e-1 — int8 max_q=255 amplifies bf16 round-trip drift; the
    // worst case at n_groups=4096 is ~5e-2.
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int8<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 4u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    let q0 = (val >> 0u32) & 255u32;
    let q1 = (val >> 8u32) & 255u32;
    let q2 = (val >> 16u32) & 255u32;
    let q3 = (val >> 24u32) & 255u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
}

// ─── mt_affine_quantize_int8 ───────────────────────────────────────────
#[bench_kernel(
    op="affine",
    subop="quantize_int8",
    class=AffineQuantize,
    bits=8,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int8<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 255.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());
    }

    // Packs in parallel: lanes 0..packs_per_group each pack one uint32.
    // For group_size=64, pack_factor=4 → packs_per_group=16, so 16
    // lanes pack in parallel vs the previous lane-0 serial loop.
    let packs_per_group = group_size / 4u32;
    if lane < packs_per_group {
        let pack_in_base = in_base + lane * 4u32;
        let mut acc = 0u32;
        for k in range(0u32, 4u32, 1u32) {
            let v = load(w[pack_in_base + k]).cast::<f32>();
            let q_f = (v - bias) * inv_scale + 0.5f32;
            let q_c = select(q_f > 255.0f32, 255.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_c.cast::<u32>();
            acc = acc | (q << (k * 8u32));
        }
        store(out[g_idx * packs_per_group + lane], acc);
    }
}

// ─── Byte-stream dequant variants (int3 / int5 / int6) ───────────────
//
// Non-power-of-2 bit widths can't pack cleanly into a uint32, so each
// pack spans `bytes_per_pack` bytes that may cross a uint32 boundary.
// The runner allocates a one-uint32 sentinel past the end so the always-
// on `w[u_idx0 + 1]` load is safe even for the last pack.
//
// Bit layouts match MLX `affine_dequantize<T, group_size, {3,5,6}>`
// exactly.

#[bench_kernel(
    op="affine",
    subop="dequantize_int3",
    class=AffineDequantize,
    bits=3,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    // tol=5e-3 — int3 max_q=7; worst-case bf16 drift at n_groups=4096
    // is ~1e-3.
    tol=5e-3,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int3<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let bytes_per_pack = 3u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let q0 = b0 & 7u32;
    let q1 = (b0 >> 3u32) & 7u32;
    let q2 = ((b0 >> 6u32) & 3u32) | ((b1 & 1u32) << 2u32);
    let q3 = (b1 >> 1u32) & 7u32;
    let q4 = (b1 >> 4u32) & 7u32;
    let q5 = ((b1 >> 7u32) & 1u32) | ((b2 & 3u32) << 1u32);
    let q6 = (b2 >> 2u32) & 7u32;
    let q7 = (b2 >> 5u32) & 7u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

#[bench_kernel(
    op="affine",
    subop="dequantize_int5",
    class=AffineDequantize,
    bits=5,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    tol=1e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int5<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let bytes_per_pack = 5u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let s3 = (byte_off + 3u32) & 3u32;
    let s4 = (byte_off + 4u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let in0_3 = (byte_off + 3u32) / 4u32 == u_idx0;
    let in0_4 = (byte_off + 4u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;
    let b3 = (select(in0_3, u0, u1) >> (s3 * 8u32)) & 255u32;
    let b4 = (select(in0_4, u0, u1) >> (s4 * 8u32)) & 255u32;

    let q0 = b0 & 31u32;
    let q1 = ((b0 >> 5u32) & 7u32) | ((b1 & 3u32) << 3u32);
    let q2 = (b1 >> 2u32) & 31u32;
    let q3 = ((b1 >> 7u32) & 1u32) | ((b2 & 15u32) << 1u32);
    let q4 = ((b2 >> 4u32) & 15u32) | ((b3 & 1u32) << 4u32);
    let q5 = (b3 >> 1u32) & 31u32;
    let q6 = ((b3 >> 6u32) & 3u32) | ((b4 & 7u32) << 2u32);
    let q7 = (b4 >> 3u32) & 31u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

#[bench_kernel(
    op="affine",
    subop="dequantize_int6",
    class=AffineDequantize,
    bits=6,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    // tol=5e-2 — int6 max_q=63; worst-case bf16 drift at n_groups=4096
    // is ~1.3e-2.
    tol=5e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int6<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 4u32;
    let bytes_per_pack = 3u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let q0 = b0 & 63u32;
    let q1 = ((b0 >> 6u32) & 3u32) | ((b1 & 15u32) << 2u32);
    let q2 = ((b1 >> 4u32) & 15u32) | ((b2 & 3u32) << 4u32);
    let q3 = (b2 >> 2u32) & 63u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
}

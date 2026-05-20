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

// ─── mt_qmm ─────────────────────────────────────────────────────────────
//
// Quantized matmul (B>1 / prefill). Same int4 weight layout as `mt_qmv`
// extended along the M axis (token count). Each threadgroup owns 8
// consecutive output columns at one M-row — `mt_qmv`'s 2 SG × 4 N-row
// tile lifted into M via an outer grid axis (`tgid_y = m_row`). The
// inner K-walk is bit-identical to `mt_qmv`: each lane caches 16 X
// values per 512-wide K-block and reuses them across all 4 N-rows in
// its simdgroup, using the same mask-without-shift trick (X
// pre-scaled by inverse nibble position, weight mask returns
// nibble × position-power) + algebraic-split accumulator
// `acc += s_g · Σ q·x + bias_g · Σ x` that mirrors MLX `qdot` in
// `quantized.h:235-244`.
//
// Geometry:
//   tpg = 64 = 2 simdgroups × 32 lanes
//   8 outputs per TG (each SG owns 4 N-rows, indexed by simd_id)
//   Block = 16 X × 32 lanes = 512 K elements per outer iter
//   Grid: [n / 8, m, 1]
//
// Layouts:
//   w       [n, k/8]               u32   — int4 nibbles (8 per uint32)
//   scales  [n, gs_per_row]        T
//   biases  [n, gs_per_row]        T
//   x       [m, k]                 T
//   out     [m, n]                 T
//
// At M = 1 this is byte-identical to `mt_qmv`. At M > 1 each M-row
// runs as a fully independent threadgroup grid axis — no W reuse
// across M-rows (W is loaded fresh per (M-row, N-tile) pair). The
// natural v3 step is a BM × BN output tile with W cached in TG
// memory and amortised across BM M-rows.
#[bench_kernel(
    op="quantized",
    subop="qmm",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=4 = canonical small-batch prefill token count (covers
    // single-prompt prefill chunks + small batched serving). Larger
    // M values exposed via the #[ignore] `mt_qmm_perf_bench_*` test.
    m=4,
    group_size=64,
    // tpg=64 same as mt_qmv (2 SG × 32 lanes). Each TG produces 8
    // outputs at one (m_row, n_tile).
    tpg=64,
    tol=1e-3,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16],
)]
#[kernel]
pub fn mt_qmm<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_row = tgid_y;
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

    let x_row_base = m_row * k;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    for _b in range(0u32, k, 512u32) {
        // 16 X loads — consecutive in IR for vectorize fusion.
        let xb = x_row_base + _b + lane_x_off;
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
        // Mask-without-shift constants. Same as mt_qmv.
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
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

        // Group index within this row's K dimension. mt_qmv uses
        // `xb / 64` because there `xb` is already a K-position; here
        // `xb` includes the `x_row_base = m_row * k` offset, so we
        // recompute against the K-local base.
        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 ──
        let p00 = load(w[w_base0 + pack_off]);
        let p01 = load(w[w_base0 + pack_off + 1u32]);
        let p00_hi = p00 >> 16u32;
        let p01_hi = p01 >> 16u32;
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
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

    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(out[m_row * n + row0], r0.cast::<T>());
        store(out[m_row * n + row1], r1.cast::<T>());
        store(out[m_row * n + row2], r2.cast::<T>());
        store(out[m_row * n + row3], r3.cast::<T>());
    }
}

// ─── mt_qmm_bm2 ─────────────────────────────────────────────────────────
//
// Quantized matmul v3 — BM × BN output tile with TG-memory-free W reuse.
//
// Same int4 weight layout + 8-output 2 SG × 4 N-row geometry as
// `mt_qmm`, but lifts BM=2 M-rows into the same threadgroup so the W
// packs + nibble extractions are loaded ONCE per K-block per N-row and
// reused across both M-rows. Per K-block per TG: 8 W loads (unchanged
// from v2) producing 16 outputs (vs 8). W bandwidth per output halves.
//
// Geometry:
//   tpg = 64 = 2 SG × 32 lanes
//   BM = 2 (M-rows per TG)
//   BN = 8 (N-rows per TG, each SG owns 4)
//   16 outputs per TG (BM × BN)
//   Grid: [n / 8, m / 2, 1]
//
// Register footprint per lane (f32):
//   32 X values (16 per M-row × 2 M-rows) = 128 bytes
//   8 accumulators (4 N-rows × 2 M-rows)   =  32 bytes
//   16 W nibble extracts (shared)          =  64 bytes
//   ≈ 240 bytes — well inside Apple GPU's ~1024 byte/lane register file.
//
// At M < 2 the caller should dispatch `mt_qmm` (BM=1) instead — this
// kernel asserts `m % 2 == 0` via the grid dim. v4 BM=4 is the next
// step if M=32 still doesn't beat MLX after this lands; see #55.
#[bench_kernel(
    op="quantized",
    subop="qmm_bm2",
    class=QuantizedMatMul,
    shapes=&QUANTIZED_SHAPES,
    // M=8 = larger-batch prefill where W-reuse matters most. M=2 / 4
    // also benefit (W reload halved); M=1 should keep dispatching
    // mt_qmm (v2) since the BM=2 tile would burn TG slots on unused
    // outputs.
    // M=8 is a representative mid-M cell. Clean median-of-5 head-to-head
    // bm2/v2 (25 cells per M, both rigs): bm2 wins 350/350 across
    // M ∈ {2,4,6,8,12,16,32}. Speedups grow with M: 1.09× at M=2
    // → 1.24× M5 / 1.30× M2 at M=32. vs MLX `affine_qmm_t`, the M=8
    // bench cell measures 1.7-2.5× M5 / 1.4-1.7× f16 M2 (3-run M5
    // drift ≤3pt). Selector `mt_qmm_for` routes every even M ≥ 2
    // to bm2. Neither kernel beats MLX at M ≥ 16 (MLX's BM=BN=32
    // simdgroup-matrix tile dominates large-M); closing that gap is
    // the BM=4/BM=8 follow-up.
    m=8,
    group_size=64,
    tpg=64,
    tol=1e-3,
    mlx="affine_qmm_t_{tn}_gs_64_b_4_alN_true_batch_0",
    metal_file="quantized.metal",
    dtypes=&[metaltile_core::dtype::DType::F32, metaltile_core::dtype::DType::F16],
)]
#[kernel]
pub fn mt_qmm_bm2<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let tg = tgid_x;
    let m_tile = tgid_y;
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

    // BM=2 M-rows per TG.
    let m_row_a = m_tile * 2u32;
    let m_row_b = m_row_a + 1u32;
    let x_base_a = m_row_a * k;
    let x_base_b = m_row_b * k;

    // 8 accumulators: 4 N-rows × 2 M-rows.
    let mut acc0_a = 0.0f32;
    let mut acc0_b = 0.0f32;
    let mut acc1_a = 0.0f32;
    let mut acc1_b = 0.0f32;
    let mut acc2_a = 0.0f32;
    let mut acc2_b = 0.0f32;
    let mut acc3_a = 0.0f32;
    let mut acc3_b = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    for _b in range(0u32, k, 512u32) {
        // ── Load 16 X values for M-row A ──
        let xb_a = x_base_a + _b + lane_x_off;
        let s_16 = 0.0625f32;
        let s_256 = 0.00390625f32;
        let s_4096 = 0.000244140625f32;
        let x0_a = load(x[xb_a]).cast::<f32>();
        let x1_a_raw = load(x[xb_a + 1u32]).cast::<f32>();
        let x2_a_raw = load(x[xb_a + 2u32]).cast::<f32>();
        let x3_a_raw = load(x[xb_a + 3u32]).cast::<f32>();
        let x4_a = load(x[xb_a + 4u32]).cast::<f32>();
        let x5_a_raw = load(x[xb_a + 5u32]).cast::<f32>();
        let x6_a_raw = load(x[xb_a + 6u32]).cast::<f32>();
        let x7_a_raw = load(x[xb_a + 7u32]).cast::<f32>();
        let x8_a = load(x[xb_a + 8u32]).cast::<f32>();
        let x9_a_raw = load(x[xb_a + 9u32]).cast::<f32>();
        let x10_a_raw = load(x[xb_a + 10u32]).cast::<f32>();
        let x11_a_raw = load(x[xb_a + 11u32]).cast::<f32>();
        let x12_a = load(x[xb_a + 12u32]).cast::<f32>();
        let x13_a_raw = load(x[xb_a + 13u32]).cast::<f32>();
        let x14_a_raw = load(x[xb_a + 14u32]).cast::<f32>();
        let x15_a_raw = load(x[xb_a + 15u32]).cast::<f32>();
        let xs_a = x0_a
            + x1_a_raw
            + x2_a_raw
            + x3_a_raw
            + x4_a
            + x5_a_raw
            + x6_a_raw
            + x7_a_raw
            + x8_a
            + x9_a_raw
            + x10_a_raw
            + x11_a_raw
            + x12_a
            + x13_a_raw
            + x14_a_raw
            + x15_a_raw;
        let x1_a = x1_a_raw * s_16;
        let x2_a = x2_a_raw * s_256;
        let x3_a = x3_a_raw * s_4096;
        let x5_a = x5_a_raw * s_16;
        let x6_a = x6_a_raw * s_256;
        let x7_a = x7_a_raw * s_4096;
        let x9_a = x9_a_raw * s_16;
        let x10_a = x10_a_raw * s_256;
        let x11_a = x11_a_raw * s_4096;
        let x13_a = x13_a_raw * s_16;
        let x14_a = x14_a_raw * s_256;
        let x15_a = x15_a_raw * s_4096;

        // ── Load 16 X values for M-row B ──
        let xb_b = x_base_b + _b + lane_x_off;
        let x0_b = load(x[xb_b]).cast::<f32>();
        let x1_b_raw = load(x[xb_b + 1u32]).cast::<f32>();
        let x2_b_raw = load(x[xb_b + 2u32]).cast::<f32>();
        let x3_b_raw = load(x[xb_b + 3u32]).cast::<f32>();
        let x4_b = load(x[xb_b + 4u32]).cast::<f32>();
        let x5_b_raw = load(x[xb_b + 5u32]).cast::<f32>();
        let x6_b_raw = load(x[xb_b + 6u32]).cast::<f32>();
        let x7_b_raw = load(x[xb_b + 7u32]).cast::<f32>();
        let x8_b = load(x[xb_b + 8u32]).cast::<f32>();
        let x9_b_raw = load(x[xb_b + 9u32]).cast::<f32>();
        let x10_b_raw = load(x[xb_b + 10u32]).cast::<f32>();
        let x11_b_raw = load(x[xb_b + 11u32]).cast::<f32>();
        let x12_b = load(x[xb_b + 12u32]).cast::<f32>();
        let x13_b_raw = load(x[xb_b + 13u32]).cast::<f32>();
        let x14_b_raw = load(x[xb_b + 14u32]).cast::<f32>();
        let x15_b_raw = load(x[xb_b + 15u32]).cast::<f32>();
        let xs_b = x0_b
            + x1_b_raw
            + x2_b_raw
            + x3_b_raw
            + x4_b
            + x5_b_raw
            + x6_b_raw
            + x7_b_raw
            + x8_b
            + x9_b_raw
            + x10_b_raw
            + x11_b_raw
            + x12_b
            + x13_b_raw
            + x14_b_raw
            + x15_b_raw;
        let x1_b = x1_b_raw * s_16;
        let x2_b = x2_b_raw * s_256;
        let x3_b = x3_b_raw * s_4096;
        let x5_b = x5_b_raw * s_16;
        let x6_b = x6_b_raw * s_256;
        let x7_b = x7_b_raw * s_4096;
        let x9_b = x9_b_raw * s_16;
        let x10_b = x10_b_raw * s_256;
        let x11_b = x11_b_raw * s_4096;
        let x13_b = x13_b_raw * s_16;
        let x14_b = x14_b_raw * s_256;
        let x15_b = x15_b_raw * s_4096;

        let g = (_b + lane_x_off) / 64u32;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 (shared W extracts, dual qdots) ──
        let p00 = load(w[w_base0 + pack_off]);
        let p01 = load(w[w_base0 + pack_off + 1u32]);
        let p00_hi = p00 >> 16u32;
        let p01_hi = p01 >> 16u32;
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
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
        let qd0_a = q00 * x0_a
            + q01 * x1_a
            + q02 * x2_a
            + q03 * x3_a
            + q04 * x4_a
            + q05 * x5_a
            + q06 * x6_a
            + q07 * x7_a
            + q08 * x8_a
            + q09 * x9_a
            + q010 * x10_a
            + q011 * x11_a
            + q012 * x12_a
            + q013 * x13_a
            + q014 * x14_a
            + q015 * x15_a;
        let qd0_b = q00 * x0_b
            + q01 * x1_b
            + q02 * x2_b
            + q03 * x3_b
            + q04 * x4_b
            + q05 * x5_b
            + q06 * x6_b
            + q07 * x7_b
            + q08 * x8_b
            + q09 * x9_b
            + q010 * x10_b
            + q011 * x11_b
            + q012 * x12_b
            + q013 * x13_b
            + q014 * x14_b
            + q015 * x15_b;
        acc0_a = acc0_a + s0 * qd0_a + bi0 * xs_a;
        acc0_b = acc0_b + s0 * qd0_b + bi0 * xs_b;

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
        let qd1_a = q10 * x0_a
            + q11 * x1_a
            + q12 * x2_a
            + q13 * x3_a
            + q14 * x4_a
            + q15 * x5_a
            + q16 * x6_a
            + q17 * x7_a
            + q18 * x8_a
            + q19 * x9_a
            + q110 * x10_a
            + q111 * x11_a
            + q112 * x12_a
            + q113 * x13_a
            + q114 * x14_a
            + q115 * x15_a;
        let qd1_b = q10 * x0_b
            + q11 * x1_b
            + q12 * x2_b
            + q13 * x3_b
            + q14 * x4_b
            + q15 * x5_b
            + q16 * x6_b
            + q17 * x7_b
            + q18 * x8_b
            + q19 * x9_b
            + q110 * x10_b
            + q111 * x11_b
            + q112 * x12_b
            + q113 * x13_b
            + q114 * x14_b
            + q115 * x15_b;
        acc1_a = acc1_a + s1 * qd1_a + bi1 * xs_a;
        acc1_b = acc1_b + s1 * qd1_b + bi1 * xs_b;

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
        let qd2_a = q20 * x0_a
            + q21 * x1_a
            + q22 * x2_a
            + q23 * x3_a
            + q24 * x4_a
            + q25 * x5_a
            + q26 * x6_a
            + q27 * x7_a
            + q28 * x8_a
            + q29 * x9_a
            + q210 * x10_a
            + q211 * x11_a
            + q212 * x12_a
            + q213 * x13_a
            + q214 * x14_a
            + q215 * x15_a;
        let qd2_b = q20 * x0_b
            + q21 * x1_b
            + q22 * x2_b
            + q23 * x3_b
            + q24 * x4_b
            + q25 * x5_b
            + q26 * x6_b
            + q27 * x7_b
            + q28 * x8_b
            + q29 * x9_b
            + q210 * x10_b
            + q211 * x11_b
            + q212 * x12_b
            + q213 * x13_b
            + q214 * x14_b
            + q215 * x15_b;
        acc2_a = acc2_a + s2 * qd2_a + bi2 * xs_a;
        acc2_b = acc2_b + s2 * qd2_b + bi2 * xs_b;

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
        let qd3_a = q30 * x0_a
            + q31 * x1_a
            + q32 * x2_a
            + q33 * x3_a
            + q34 * x4_a
            + q35 * x5_a
            + q36 * x6_a
            + q37 * x7_a
            + q38 * x8_a
            + q39 * x9_a
            + q310 * x10_a
            + q311 * x11_a
            + q312 * x12_a
            + q313 * x13_a
            + q314 * x14_a
            + q315 * x15_a;
        let qd3_b = q30 * x0_b
            + q31 * x1_b
            + q32 * x2_b
            + q33 * x3_b
            + q34 * x4_b
            + q35 * x5_b
            + q36 * x6_b
            + q37 * x7_b
            + q38 * x8_b
            + q39 * x9_b
            + q310 * x10_b
            + q311 * x11_b
            + q312 * x12_b
            + q313 * x13_b
            + q314 * x14_b
            + q315 * x15_b;
        acc3_a = acc3_a + s3 * qd3_a + bi3 * xs_a;
        acc3_b = acc3_b + s3 * qd3_b + bi3 * xs_b;
    }

    // Cross-lane reduce + lane-0 stores. 8 outputs per TG.
    let r0_a = simd_sum(acc0_a);
    let r0_b = simd_sum(acc0_b);
    let r1_a = simd_sum(acc1_a);
    let r1_b = simd_sum(acc1_b);
    let r2_a = simd_sum(acc2_a);
    let r2_b = simd_sum(acc2_b);
    let r3_a = simd_sum(acc3_a);
    let r3_b = simd_sum(acc3_b);
    if lane == 0u32 {
        store(out[m_row_a * n + row0], r0_a.cast::<T>());
        store(out[m_row_a * n + row1], r1_a.cast::<T>());
        store(out[m_row_a * n + row2], r2_a.cast::<T>());
        store(out[m_row_a * n + row3], r3_a.cast::<T>());
        store(out[m_row_b * n + row0], r0_b.cast::<T>());
        store(out[m_row_b * n + row1], r1_b.cast::<T>());
        store(out[m_row_b * n + row2], r2_b.cast::<T>());
        store(out[m_row_b * n + row3], r3_b.cast::<T>());
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

/// Auto-select the best `mt_qmm*` kernel for a given dtype + M
/// (number of tokens / batched rows in this prefill). Returns the
/// kernel IR ready to dispatch. Caller still owns grid sizing — see
/// the table in the docstring for the per-route grid shape.
///
/// Routing — anywhere `m % 2 == 0 && m >= 2`, `mt_qmm_bm2` wins. The
/// W-bandwidth-halving from BM=2 W-reuse is the dominant factor
/// across the whole M-range we tested (1 ≤ M ≤ 32); the only
/// constraint is that bm2's hand-unrolled BM=2 tile requires even M.
/// Odd M and M=1 fall back to v2 (where bm2 is undefined / wastes
/// half the output slots).
///
/// Head-to-head bm2/v2 speedup (median of 5 reruns × 5 Qwen3
/// production shapes = 25 cells per M; WARMUP=20 + ITERS=50 per
/// kernel per cell; resident-buffer harness; clean shell sessions;
/// `mt_qmm_v2_vs_bm2_head_to_head_f16_m_sweep`):
///
/// | M  | M5 Max speedup | M5 wins | M2 mini speedup | M2 wins |
/// |---:|---------------:|:-------:|----------------:|:-------:|
/// |  2 | 1.09×          | 25/25   | 1.11×           | 25/25   |
/// |  4 | 1.21×          | 25/25   | 1.26×           | 25/25   |
/// |  6 | 1.22×          | 25/25   | 1.27×           | 25/25   |
/// |  8 | 1.22×          | 25/25   | 1.28×           | 25/25   |
/// | 12 | 1.23×          | 25/25   | 1.29×           | 25/25   |
/// | 16 | 1.23×          | 25/25   | 1.29×           | 25/25   |
/// | 32 | 1.24×          | 25/25   | 1.30×           | 25/25   |
///
/// 350/350 cells went to bm2 (175 per rig). No shape or M cell on
/// either rig has v2 beating bm2. Cell-to-cell variance is tight
/// (median min/max within ~2% on M ≥ 8; ~10-20% on M ≤ 6 where
/// individual cells occasionally see lucky outliers).
///
/// Both kernels still trail MLX `affine_qmm_t` at M ≥ 16 (MLX's
/// BM=BN=32 simdgroup-matrix tile dominates at large M); the bm2
/// route just gives us the better of our two kernels everywhere.
/// Closing the MLX gap at M ≥ 16 needs the BM=4/BM=8 follow-up.
pub fn mt_qmm_for(dtype: metaltile_core::dtype::DType, m: u32) -> metaltile_core::ir::Kernel {
    use metaltile_core::ir::KernelMode;
    let mut k = if m >= 2 && m.is_multiple_of(2) {
        mt_qmm_bm2::kernel_ir_for(dtype)
    } else {
        mt_qmm::kernel_ir_for(dtype)
    };
    // Reduction mode required for the `tgid_x`/`tgid_y` aliases
    // both kernels reference. Same dispatch contract as `mt_qmv`.
    k.mode = KernelMode::Reduction;
    k
}

#[cfg(test)]
mod qmm_selector_tests {
    use metaltile_core::dtype::DType;

    use super::*;

    #[test]
    fn selector_picks_bm2_at_even_m_ge_2() {
        // Median-of-5 head-to-head bench (2026-05-19) on M5 + M2 mini
        // showed bm2 wins v2 at every even M from 2 to 32 inclusive.
        // The W-bandwidth halving from BM=2 W-reuse is the dominant
        // factor across the whole M range we tested.
        for m in [2u32, 4, 6, 8, 12, 16, 24, 32, 64] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(k.name, "mt_qmm_bm2", "m={m}: even ≥ 2 should route to bm2");
        }
    }

    #[test]
    fn selector_picks_v2_at_m_1() {
        // bm2 requires m % 2 == 0 (BM=2 tile). M=1 falls back to v2.
        let k = mt_qmm_for(DType::F32, 1);
        assert_eq!(k.name, "mt_qmm");
    }

    #[test]
    fn selector_picks_v2_at_odd_m() {
        // bm2 requires m % 2 == 0. Odd M dispatches v2 which has
        // unit BM and handles any M.
        for m in [3u32, 5, 7, 9, 15, 31] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(k.name, "mt_qmm", "m={m}: odd M should route to v2");
        }
    }

    #[test]
    fn selector_picks_bm2_across_dtypes_at_m_8() {
        for dt in [DType::F32, DType::F16] {
            let k = mt_qmm_for(dt, 8);
            assert_eq!(k.name, "mt_qmm_bm2", "dt={dt:?}");
        }
    }

    #[test]
    fn selector_kernels_carry_reduction_mode() {
        for m in [1u32, 4, 8, 16, 32] {
            let k = mt_qmm_for(DType::F32, m);
            assert_eq!(
                k.mode,
                metaltile_core::ir::KernelMode::Reduction,
                "m={m}: missing Reduction mode",
            );
        }
    }
}

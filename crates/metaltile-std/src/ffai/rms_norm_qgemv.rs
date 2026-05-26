//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused RMSNorm + 4-bit quantized GEMV for decode (single-token).
//!
//! Computes `y = qmatmul(rms_norm(x) * norm_weight, W_q)` in one
//! dispatch, eliminating the global-memory round-trip of the normalized
//! activation between a standalone `rms_norm` and a quantized matvec.
//!
//! Two variants:
//!
//! **`ffai_rms_norm_qgemv`** — one output row per TG (the original port).
//! Reduction-mode: one threadgroup per output row. Phase 1 reduces
//! `sum(x²)` across the threadgroup → `inv_rms`; phase 2 is a
//! pack-strided int4 GEMV that feeds on
//! `normed[i] = x[i] * norm_weight[i] * inv_rms` instead of raw `x`, so
//! the normalized activation never leaves registers. Grid: `[out_dim, 1, 1]`,
//! TPG ≥ 32.
//!
//! **`ffai_rms_norm_qgemv_fast`** — 8 output rows per TG, mirroring
//! `mt_qmv`'s geometry. Phase 1 (SSQ → `inv_rms`) is shared across all
//! 8 rows — the TG-wide reduce amortizes the RMSNorm over 8 outputs.
//! Phase 2 uses the `mt_qmv` mask-without-shift trick (X pre-scaled by
//! inverse nibble position, weight mask returns nibble × position-power)
//! plus the algebraic-split accumulator (`s*Σq·normed + b*Σnormed`),
//! exactly as in MLX `rms_norm_qmm`. Grid: `[out_dim/8, 1, 1]`,
//! TPG = 64 (2 simdgroups × 32 lanes).
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   weight  [out_dim, N/8]    uint32  (8 int4 values per u32)
//!   scales  [out_dim, N/G]    T
//!   biases  [out_dim, N/G]    T
//!   x, norm_weight  [N]       T
//!   y               [out_dim] T
//!
//! ## DISPATCH INVARIANTS (fast variant)
//!
//! - **Grid: `[out_dim/8, 1, 1]`** — one TG per 8-row tile.
//! - **TPG = 64** (2 simdgroups × 32 lanes).
//! - `in_dim` a multiple of 512 (block size = 512 K elements per outer
//!   iter; equivalently `in_dim` must be a multiple of 8 and 64 and ≥ 512).
//! - `out_dim` must be a multiple of 8.
//! - `group_size` must be 64 (one group per 512-K block / 4 lanes).
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_qgemv_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

/// `y[row] = Σ_i (q[row,i]·scale + bias) · (x[i]·norm_weight[i]·inv_rms)`,
/// with `inv_rms = rsqrt(mean(x²) + eps)`, weights int4-packed.
/// One output row per threadgroup (original correctness-first variant).
#[bench_kernel(
    op="rms_norm_qgemv",
    subop="rms_norm_qgemv",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    // Phase 1: RMSNorm — per-thread partial sum of squares, then cross-kernel
    // call to mt_rms_inv_scalar for the threadgroup reduce + rsqrt.
    // ssq is a Value arg; eps_buf and in_dim are Tensor args.
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: pack-strided int4 GEMV over the normalized activation.
    let vals_per_pack = 8u32; // 32 / 4 bits
    let mask = 15u32;
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;
    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for _p in range(0u32, p_iters, 1u32) {
        let pack_idx = _p * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                let normed = xi * nw * inv_rms;
                acc = acc + (q.cast::<f32>() * scale + bias) * normed;
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Perf-tuned fused RMSNorm + int4 GEMV — 8 output rows per TG.
///
/// Geometry: tpg = 64 = 2 simdgroups × 32 lanes. `simd_id` selects the
/// simdgroup (0 or 1); each simdgroup independently computes 4 output rows.
/// Phase 1 (SSQ for RMSNorm) is shared — `mt_rms_inv_scalar` performs a
/// TG-wide reduce, so the same `inv_rms` is broadcast to all 8 rows.
/// Phase 2 reuses `mt_qmv`'s two-pass algebraic split:
///   `acc = scale * q_dot + bias * normed_xs`
/// where `q_dot = Σ q_i * normed_i` and `normed_xs = Σ normed_i`.
/// The mask-without-shift trick eliminates per-nibble shifts — identical
/// to `mt_qmv`. Block = 16 X × 32 lanes = 512 K elements per outer iter.
/// The `range(0, 4)` row loops use `stack_alloc` accumulators; the DSL
/// unrolls constexpr-bounded loops at codegen so the emitted MSL is
/// identical to the hand-unrolled form.
///
/// Grid: `[out_dim/8, 1, 1]`. out_dim must be a multiple of 8;
/// in_dim must be a multiple of 512; group_size must be 64.
#[bench_kernel(
    op="rms_norm_qgemv",
    subop="rms_norm_qgemv_fast",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_rms_norm_qgemv_fast<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // Each TG covers 8 output rows: simdgroup 0 → rows 0-3, sg 1 → rows 4-7.
    let base_row = tg * 8u32 + sg * 4u32;
    // Phase 1: TG-wide SSQ for RMSNorm.
    // All 64 threads cooperate — `mt_rms_inv_scalar` performs the full
    // TG reduce + rsqrt + broadcast, identical to the single-row variant.
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: 4-row int4 GEMV per simdgroup, mirroring `mt_qmv`.
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Mask-without-shift constants — eliminates 56 shifts per block.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // thread-private array; DSL unrolls range(0,4) loops at codegen.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    for _b in range(0u32, in_dim, 512u32) {
        // Load 16 X values and apply RMSNorm + norm_weight in registers.
        let xb = _b + lane_x_off;
        // Fuse RMSNorm: normed[i] = x[i] * norm_weight[i] * inv_rms.
        // Raw values needed at nibble positions 1/2/3 for mask-without-shift.
        let n0_raw  = load(x[xb]).cast::<f32>()        * load(norm_weight[xb]).cast::<f32>()        * inv_rms;
        let n1_raw  = load(x[xb +  1u32]).cast::<f32>() * load(norm_weight[xb +  1u32]).cast::<f32>() * inv_rms;
        let n2_raw  = load(x[xb +  2u32]).cast::<f32>() * load(norm_weight[xb +  2u32]).cast::<f32>() * inv_rms;
        let n3_raw  = load(x[xb +  3u32]).cast::<f32>() * load(norm_weight[xb +  3u32]).cast::<f32>() * inv_rms;
        let n4_raw  = load(x[xb +  4u32]).cast::<f32>() * load(norm_weight[xb +  4u32]).cast::<f32>() * inv_rms;
        let n5_raw  = load(x[xb +  5u32]).cast::<f32>() * load(norm_weight[xb +  5u32]).cast::<f32>() * inv_rms;
        let n6_raw  = load(x[xb +  6u32]).cast::<f32>() * load(norm_weight[xb +  6u32]).cast::<f32>() * inv_rms;
        let n7_raw  = load(x[xb +  7u32]).cast::<f32>() * load(norm_weight[xb +  7u32]).cast::<f32>() * inv_rms;
        let n8_raw  = load(x[xb +  8u32]).cast::<f32>() * load(norm_weight[xb +  8u32]).cast::<f32>() * inv_rms;
        let n9_raw  = load(x[xb +  9u32]).cast::<f32>() * load(norm_weight[xb +  9u32]).cast::<f32>() * inv_rms;
        let n10_raw = load(x[xb + 10u32]).cast::<f32>() * load(norm_weight[xb + 10u32]).cast::<f32>() * inv_rms;
        let n11_raw = load(x[xb + 11u32]).cast::<f32>() * load(norm_weight[xb + 11u32]).cast::<f32>() * inv_rms;
        let n12_raw = load(x[xb + 12u32]).cast::<f32>() * load(norm_weight[xb + 12u32]).cast::<f32>() * inv_rms;
        let n13_raw = load(x[xb + 13u32]).cast::<f32>() * load(norm_weight[xb + 13u32]).cast::<f32>() * inv_rms;
        let n14_raw = load(x[xb + 14u32]).cast::<f32>() * load(norm_weight[xb + 14u32]).cast::<f32>() * inv_rms;
        let n15_raw = load(x[xb + 15u32]).cast::<f32>() * load(norm_weight[xb + 15u32]).cast::<f32>() * inv_rms;
        // Sum of normed activations for the bias term of the algebraic split.
        let ns = n0_raw + n1_raw + n2_raw  + n3_raw  + n4_raw  + n5_raw  + n6_raw  + n7_raw
               + n8_raw + n9_raw + n10_raw + n11_raw + n12_raw + n13_raw + n14_raw + n15_raw;
        // Pre-scale normed values at nibble positions 1/2/3 for
        // mask-without-shift. Position 0 stays unscaled (*1).
        let n1  = n1_raw  * s_16;  let n2  = n2_raw  * s_256;  let n3  = n3_raw  * s_4096;
        let n5  = n5_raw  * s_16;  let n6  = n6_raw  * s_256;  let n7  = n7_raw  * s_4096;
        let n9  = n9_raw  * s_16;  let n10 = n10_raw * s_256;  let n11 = n11_raw * s_4096;
        let n13 = n13_raw * s_16;  let n14 = n14_raw * s_256;  let n15 = n15_raw * s_4096;
        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;
        // 4 rows × identical work, looped — DSL unrolls at codegen.
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let wb = row * packs_per_row;
            let sb = row * gs_per_row;
            let p_lo = load(weight[wb + pack_off]);
            let p_hi = load(weight[wb + pack_off + 1u32]);
            let p_lo_hi = p_lo >> 16u32;
            let p_hi_hi = p_hi >> 16u32;
            let s  = load(scales[sb + g]).cast::<f32>();
            let bi = load(biases[sb + g]).cast::<f32>();
            let qd = (p_lo    & 15u32).cast::<f32>() * n0_raw
                   + (p_lo    & 240u32).cast::<f32>() * n1
                   + (p_lo    & 3840u32).cast::<f32>() * n2
                   + (p_lo    & 61440u32).cast::<f32>() * n3
                   + (p_lo_hi & 15u32).cast::<f32>() * n4_raw
                   + (p_lo_hi & 240u32).cast::<f32>() * n5
                   + (p_lo_hi & 3840u32).cast::<f32>() * n6
                   + (p_lo_hi & 61440u32).cast::<f32>() * n7
                   + (p_hi    & 15u32).cast::<f32>() * n8_raw
                   + (p_hi    & 240u32).cast::<f32>() * n9
                   + (p_hi    & 3840u32).cast::<f32>() * n10
                   + (p_hi    & 61440u32).cast::<f32>() * n11
                   + (p_hi_hi & 15u32).cast::<f32>() * n12_raw
                   + (p_hi_hi & 240u32).cast::<f32>() * n13
                   + (p_hi_hi & 3840u32).cast::<f32>() * n14
                   + (p_hi_hi & 61440u32).cast::<f32>() * n15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * ns);
        }
    }
    // Cross-lane reduce: each row → one value per simdgroup.
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(output[base_row + _r], r.cast::<T>());
        }
    }
}

// ─── ffai_rms_norm_qgemv_int8_fast ───────────────────────────────────────────
//
// Fused RMSNorm + int8-quantized GEMV — 8-row-per-TG perf variant.
//
// Mirrors `ffai_rms_norm_qgemv_fast` (int4, 8-row-per-TG, 2 SG × 32 lanes)
// but replaces the int4 nibble-unpack with int8 byte-extract:
//   - 4 bytes per u32 (vals_per_pack = 4 vs 8 for int4)
//   - mask = 0xFF, shifts = 0 / 8 / 16 / 24
//   - packs_per_row = in_dim / 4; lane covers 4 consecutive K positions per pack
//
// Phase 1 (TG-wide SSQ → inv_rms via `mt_rms_inv_scalar`) is identical to
// the int4 fast variant — the RMSNorm is independent of the quantization format.
//
// Phase 2 uses the same algebraic-split accumulator (`s*q_dot + b*normed_xs`)
// that the int4 fast variant uses.
//
// ## DISPATCH INVARIANTS
//
// - **Grid: `[out_dim/8, 1, 1]`** — one TG per 8-row tile.
// - **TPG = 64** (2 simdgroups × 32 lanes).
// - `in_dim` must be a multiple of 512.
// - `out_dim` must be a multiple of 8.
// - `group_size` must be 64.

/// Perf-tuned fused RMSNorm + int8 GEMV — 8 output rows per TG.
///
/// int8 variant of `ffai_rms_norm_qgemv_fast`. Byte-extract (4 vals/pack),
/// algebraic-split accumulator. Grid: `[out_dim/8, 1, 1]`. The `range(0, 4)`
/// row loops use `stack_alloc` accumulators; the DSL unrolls at codegen.
#[bench_kernel(
    op="rms_norm_qgemv",
    subop="rms_norm_qgemv_int8_fast",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_rms_norm_qgemv_int8_fast<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // Each TG covers 8 output rows: sg 0 → rows 0-3, sg 1 → rows 4-7.
    let base_row = tg * 8u32 + sg * 4u32;
    // Phase 1: TG-wide SSQ for RMSNorm (same as int4 fast variant).
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: 4-row int8 GEMV per simdgroup, algebraic-split accumulator.
    // int8: 4 bytes per u32, packs_per_row = in_dim / 4.
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 4u32;
    // Each lane covers 16 K values per block (512 K / 32 lanes).
    // int8: 4 bytes/pack → 4 packs per lane per block.
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 4u32;
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // thread-private array; DSL unrolls range(0,4) loops at codegen.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    for _b in range(0u32, in_dim, 512u32) {
        // Load 16 X values, fuse RMSNorm.
        let xb = _b + lane_x_off;
        let n0  = load(x[xb]).cast::<f32>()        * load(norm_weight[xb]).cast::<f32>()        * inv_rms;
        let n1  = load(x[xb +  1u32]).cast::<f32>() * load(norm_weight[xb +  1u32]).cast::<f32>() * inv_rms;
        let n2  = load(x[xb +  2u32]).cast::<f32>() * load(norm_weight[xb +  2u32]).cast::<f32>() * inv_rms;
        let n3  = load(x[xb +  3u32]).cast::<f32>() * load(norm_weight[xb +  3u32]).cast::<f32>() * inv_rms;
        let n4  = load(x[xb +  4u32]).cast::<f32>() * load(norm_weight[xb +  4u32]).cast::<f32>() * inv_rms;
        let n5  = load(x[xb +  5u32]).cast::<f32>() * load(norm_weight[xb +  5u32]).cast::<f32>() * inv_rms;
        let n6  = load(x[xb +  6u32]).cast::<f32>() * load(norm_weight[xb +  6u32]).cast::<f32>() * inv_rms;
        let n7  = load(x[xb +  7u32]).cast::<f32>() * load(norm_weight[xb +  7u32]).cast::<f32>() * inv_rms;
        let n8  = load(x[xb +  8u32]).cast::<f32>() * load(norm_weight[xb +  8u32]).cast::<f32>() * inv_rms;
        let n9  = load(x[xb +  9u32]).cast::<f32>() * load(norm_weight[xb +  9u32]).cast::<f32>() * inv_rms;
        let n10 = load(x[xb + 10u32]).cast::<f32>() * load(norm_weight[xb + 10u32]).cast::<f32>() * inv_rms;
        let n11 = load(x[xb + 11u32]).cast::<f32>() * load(norm_weight[xb + 11u32]).cast::<f32>() * inv_rms;
        let n12 = load(x[xb + 12u32]).cast::<f32>() * load(norm_weight[xb + 12u32]).cast::<f32>() * inv_rms;
        let n13 = load(x[xb + 13u32]).cast::<f32>() * load(norm_weight[xb + 13u32]).cast::<f32>() * inv_rms;
        let n14 = load(x[xb + 14u32]).cast::<f32>() * load(norm_weight[xb + 14u32]).cast::<f32>() * inv_rms;
        let n15 = load(x[xb + 15u32]).cast::<f32>() * load(norm_weight[xb + 15u32]).cast::<f32>() * inv_rms;
        // Bias accumulation sum for algebraic split.
        let ns = n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7
               + n8 + n9 + n10 + n11 + n12 + n13 + n14 + n15;
        let g = xb / group_size;
        let pack_off = _b / 4u32 + lane_pack_off;
        // 4 rows × identical work, looped — DSL unrolls at codegen.
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let wb = row * packs_per_row;
            let sb = row * gs_per_row;
            let p0 = load(weight[wb + pack_off]);
            let p1 = load(weight[wb + pack_off + 1u32]);
            let p2 = load(weight[wb + pack_off + 2u32]);
            let p3 = load(weight[wb + pack_off + 3u32]);
            let s  = load(scales[sb + g]).cast::<f32>();
            let bi = load(biases[sb + g]).cast::<f32>();
            let qd = (p0 & 255u32).cast::<f32>() * n0
                   + ((p0 >> 8u32)  & 255u32).cast::<f32>() * n1
                   + ((p0 >> 16u32) & 255u32).cast::<f32>() * n2
                   + ((p0 >> 24u32) & 255u32).cast::<f32>() * n3
                   + (p1 & 255u32).cast::<f32>() * n4
                   + ((p1 >> 8u32)  & 255u32).cast::<f32>() * n5
                   + ((p1 >> 16u32) & 255u32).cast::<f32>() * n6
                   + ((p1 >> 24u32) & 255u32).cast::<f32>() * n7
                   + (p2 & 255u32).cast::<f32>() * n8
                   + ((p2 >> 8u32)  & 255u32).cast::<f32>() * n9
                   + ((p2 >> 16u32) & 255u32).cast::<f32>() * n10
                   + ((p2 >> 24u32) & 255u32).cast::<f32>() * n11
                   + (p3 & 255u32).cast::<f32>() * n12
                   + ((p3 >> 8u32)  & 255u32).cast::<f32>() * n13
                   + ((p3 >> 16u32) & 255u32).cast::<f32>() * n14
                   + ((p3 >> 24u32) & 255u32).cast::<f32>() * n15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * ns);
        }
    }
    // Cross-lane reduce: each row → one value per simdgroup.
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(output[base_row + _r], r.cast::<T>());
        }
    }
}

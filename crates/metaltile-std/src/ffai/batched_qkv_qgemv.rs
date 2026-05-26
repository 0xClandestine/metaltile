//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Batched Q/K/V 4-bit quantized GEMV — fuses the three independent
//! Q, K, V projection matvecs of a decode step into one dispatch.
//!
//! The `z` grid axis selects the matrix (`program_id::<2>()`:
//! 0 = Q, 1 = K, 2 = V); the `x` grid axis is the output tile. The
//! result lands in a single contiguous `y` of length
//! `out_q + out_k + out_v`, with Q, K, V concatenated in that order.
//!
//! Two variants:
//!
//! **`ffai_batched_qkv_qgemv`** — one output row per TG (original
//! correctness-first variant). Grid: `[max(out_q,out_k,out_v), 1, 3]`;
//! `program_id::<0>()` = output row, `program_id::<2>()` = matrix.
//!
//! **`ffai_batched_qkv_qgemv_fast`** — 8 output rows per TG, mirroring
//! `mt_qmv`'s geometry. Each TG computes 8 output rows of the matrix
//! selected by `program_id::<2>()`. Grid:
//! `[ceil(max(out_q,out_k,out_v)/8), 1, 3]`, TPG = 64 (2 simdgroups ×
//! 32 lanes). Uses `mt_qmv`'s mask-without-shift trick + algebraic-split
//! accumulator (`s*q_dot + b*xs`) — identical inner loop to
//! `ffai_rms_norm_qgemv_fast` but without the RMSNorm phase.
//! out_q, out_k, out_v must each be multiples of 8; in_dim must be a
//! multiple of 512; group_size must be 64.
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   w_*       [out_*, N/8]   uint32
//!   scales_*  [out_*, N/G]   T
//!   biases_*  [out_*, N/G]   T
//!   x         [N]            T
//!   y         [out_q+out_k+out_v] T
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_qkv_qgemv_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

/// Fused Q/K/V int4 quantized GEMV — one output row per TG.
/// `program_id::<2>()` picks the matrix.
#[bench_kernel(
    op="batched_qkv_qgemv",
    subop="batched_qkv_qgemv",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<T>,
    biases_q: Tensor<T>,
    w_k: Tensor<u32>,
    scales_k: Tensor<T>,
    biases_k: Tensor<T>,
    w_v: Tensor<u32>,
    scales_v: Tensor<T>,
    biases_v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let vals_per_pack = 8u32; // 32 / 4 bits
    let mask = 15u32;
    let n_packs = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_group_off = row * n_groups;
    if matrix == 0u32 {
        if row < out_q {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_q[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_q[row_group_off + g]).cast::<f32>();
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_q = reduce_sum(acc);
            if tid == 0u32 {
                store(out[row], total_q.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_k[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_k[row_group_off + g]).cast::<f32>();
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_k = reduce_sum(acc);
            if tid == 0u32 {
                store(out[out_q + row], total_k.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_v[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_v[row_group_off + g]).cast::<f32>();
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_v = reduce_sum(acc);
            if tid == 0u32 {
                store(out[out_q + out_k + row], total_v.cast::<T>());
            }
        }
    }
}

/// Perf-tuned fused Q/K/V int4 quantized GEMV — 8 output rows per TG.
///
/// Geometry: tpg = 64 = 2 simdgroups × 32 lanes. Each TG computes
/// 8 output rows of the matrix chosen by `program_id::<2>()`.
/// Uses `mt_qmv`'s mask-without-shift trick + algebraic-split accumulator
/// — identical inner loop to `ffai_rms_norm_qgemv_fast` but without the
/// RMSNorm phase.
///
/// The x-preload (16 activations per lane per 512-element K-block) is
/// hoisted before the per-matrix dispatch and shared across all three
/// branches. The `range(0, 4)` row loops use `stack_alloc` accumulators;
/// the DSL unrolls constexpr-bounded loops at codegen so the emitted MSL
/// is identical to the hand-unrolled form.
///
/// Grid: `[ceil(max(out_q,out_k,out_v)/8), 1, 3]`.
/// out_q, out_k, out_v must be multiples of 8; in_dim must be a multiple
/// of 512; group_size must be 64. TGs past a matrix's out_* rows no-op.
#[bench_kernel(
    op="batched_qkv_qgemv",
    subop="batched_qkv_qgemv_fast",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_batched_qkv_qgemv_fast<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<T>,
    biases_q: Tensor<T>,
    w_k: Tensor<u32>,
    scales_k: Tensor<T>,
    biases_k: Tensor<T>,
    w_v: Tensor<u32>,
    scales_v: Tensor<T>,
    biases_v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Mask-without-shift constants — eliminates 56 shifts per block.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    // Route the row guard to the output size for this matrix slice.
    let out_limit = select(matrix == 0u32, out_q, select(matrix == 1u32, out_k, out_v));
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // thread-private array; DSL unrolls range(0,4) loops at codegen.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    if base_row < out_limit {
        for _b in range(0u32, in_dim, 512u32) {
            // 16 x loads per K-block, shared across all three matrix branches.
            let xb = _b + lane_x_off;
            let x0 = load(x[xb]).cast::<f32>();
            let x1r = load(x[xb + 1u32]).cast::<f32>();
            let x2r = load(x[xb + 2u32]).cast::<f32>();
            let x3r = load(x[xb + 3u32]).cast::<f32>();
            let x4 = load(x[xb + 4u32]).cast::<f32>();
            let x5r = load(x[xb + 5u32]).cast::<f32>();
            let x6r = load(x[xb + 6u32]).cast::<f32>();
            let x7r = load(x[xb + 7u32]).cast::<f32>();
            let x8 = load(x[xb + 8u32]).cast::<f32>();
            let x9r = load(x[xb + 9u32]).cast::<f32>();
            let x10r = load(x[xb + 10u32]).cast::<f32>();
            let x11r = load(x[xb + 11u32]).cast::<f32>();
            let x12 = load(x[xb + 12u32]).cast::<f32>();
            let x13r = load(x[xb + 13u32]).cast::<f32>();
            let x14r = load(x[xb + 14u32]).cast::<f32>();
            let x15r = load(x[xb + 15u32]).cast::<f32>();
            // xs = Σ x[i] over the 16-element block (bias term).
            let xs = x0
                + x1r
                + x2r
                + x3r
                + x4
                + x5r
                + x6r
                + x7r
                + x8
                + x9r
                + x10r
                + x11r
                + x12
                + x13r
                + x14r
                + x15r;
            // Pre-scale nibble positions 1/2/3 for mask-without-shift.
            let x1 = x1r * s_16;
            let x2 = x2r * s_256;
            let x3 = x3r * s_4096;
            let x5 = x5r * s_16;
            let x6 = x6r * s_256;
            let x7 = x7r * s_4096;
            let x9 = x9r * s_16;
            let x10 = x10r * s_256;
            let x11 = x11r * s_4096;
            let x13 = x13r * s_16;
            let x14 = x14r * s_256;
            let x15 = x15r * s_4096;
            let g = xb / group_size;
            let pack_off = _b / 8u32 + lane_pack_off;
            // Per-matrix dispatch. Only tensor names differ across branches.
            if matrix == 0u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_q[wb + pack_off]);
                    let p_hi = load(w_q[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_q[sb + g]).cast::<f32>();
                    let bi = load(biases_q[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 1u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_k[wb + pack_off]);
                    let p_hi = load(w_k[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_k[sb + g]).cast::<f32>();
                    let bi = load(biases_k[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 2u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_v[wb + pack_off]);
                    let p_hi = load(w_v[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_v[sb + g]).cast::<f32>();
                    let bi = load(biases_v[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
        }
        // Cross-lane reduce + store. out_* are multiples of 8 so all four
        // rows are valid whenever base_row < out_limit.
        for _r in range(0u32, 4u32, 1u32) {
            let v = stack_load("accs", _r);
            let r = simd_sum(v);
            if lane == 0u32 {
                if matrix == 0u32 {
                    store(out[base_row + _r], r.cast::<T>());
                }
                if matrix == 1u32 {
                    store(out[out_q + base_row + _r], r.cast::<T>());
                }
                if matrix == 2u32 {
                    store(out[out_q + out_k + base_row + _r], r.cast::<T>());
                }
            }
        }
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **batched Q/K/V block-scaled dequantizing GEMM** (M>1) for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! Multi-token sibling of `batched_qkv_block_scaled_qgemv`: instead of a
//! single activation row it consumes `x: [M, in_dim]` and produces row `m`
//! of THREE separate output tensors —
//!   q_buf: [M, out_q] T
//!   k_buf: [M, out_k] T
//!   v_buf: [M, out_v] T
//!
//! One dispatch computes `out_X[m, n] = Σ_k dequant(W_X[n, k]) · x[m, k]` for
//! all three projections. The block-scaled weight decode of
//! `mlx/block_scaled_qmm.rs` replaces the int affine; the (matrix, token, row)
//! geometry mirrors the int4 `ffai_batched_qkv_qmm_fast`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [max(out_q,out_k,out_v), M, 3]`,
//!   `tpg = [TPG, 1, 1]`, TPG ≥ 32 & a multiple of 32. One TG per
//!   `(matrix, m_token, out_row)`; rows past a matrix's `out_*` no-op.
//!   * `program_id::<2>()` selects matrix (0→Q, 1→K, 2→V).
//!   * `program_id::<1>()` selects batched token `mr` (0..M).
//!   * `program_id::<0>()` selects the output row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_*, in_dim/8]` u32 (4-bit) or `[out_*, in_dim]` u8 (8-bit);
//!   scales `[out_*, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!   `x` is `[M, in_dim]`, each `*_buf` is `[M, out_*]`, all row-major. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 batched Q/K/V GEMM (M>1) — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u32>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u32>,
    scales_v: Tensor<u8>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_q[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_k[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_v[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

/// nvfp4 batched Q/K/V GEMM (M>1) — E2M1 (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u32>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u32>,
    scales_v: Tensor<u8>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_q[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_k[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_v[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E4M3) batched Q/K/V GEMM (M>1) — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u8>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u8>,
    scales_v: Tensor<u8>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_q[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_q[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_k[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_k[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_v[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_v[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E5M2) batched Q/K/V GEMM (M>1) — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u8>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u8>,
    scales_v: Tensor<u8>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_q[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_q[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_k[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_k[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_v[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_v[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

/// nvfp8 batched Q/K/V GEMM (M>1) — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f32>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 batched Q/K/V GEMMs ─────
// These reuse the block-scaled framework but store a raw per-group FP32 scale
// (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 +
// f32 scale), so it reuses `mt_nvfp8_batched_qkv_qmm` — only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 batched Q/K/V GEMM (M>1) — E2M1 weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp4_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u32>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u32>,
    scales_v: Tensor<f32>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_q[row_block_off + blk]);
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_k[row_block_off + blk]);
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_v[row_block_off + blk]);
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

/// Legacy fp8 (E5M2) batched Q/K/V GEMM (M>1) — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f32>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

/// Symmetric int8 batched Q/K/V GEMM (M>1) — 8-bit codes (group 64), per-group
/// FP32 scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_batched_qkv_qmm<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f32>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = int8_decode(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = int8_decode(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = int8_decode(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Reduction-contract threadgroup width (≥ 32, multiple of 32).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` quantized weights (mixed signs).
    /// `seed` decorrelates the Q/K/V matrices.
    fn weights(out_dim: usize, in_dim: usize, seed: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + ((r as usize + seed) % 5) as f32 * 0.2) * (0.1 + (c % 13.0) * 0.2);
                if (i + seed).is_multiple_of(3) { -mag } else { mag }
            })
            .collect()
    }

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`, row-major `[M, out_dim]`.
    fn qmm_oracle(
        wdq: &[f32],
        x: &[f32],
        m_rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m_rows * out_dim];
        for mr in 0..m_rows {
            for n in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += wdq[n * in_dim + k] * x[mr * in_dim + k];
                }
                out[mr * out_dim + n] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_q: usize,
        out_k: usize,
        out_v: usize,
        in_dim: usize,
        m_rows: usize,
        dt: DType,
    ) -> TestSetup {
        // Pack + dequant each of the three weight matrices (distinct seeds).
        let pack_w = |out_dim: usize, seed: usize| {
            let w = weights(out_dim, in_dim, seed);
            let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
            let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
            (p, wdq)
        };
        let (pq, wdq_q) = pack_w(out_q, 0);
        let (pk, wdq_k) = pack_w(out_k, 1);
        let (pv, wdq_v) = pack_w(out_v, 2);
        // Build x as [m_rows, in_dim] and round it through the storage dtype.
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let eq = qmm_oracle(&wdq_q, &x, m_rows, in_dim, out_q);
        let ek = qmm_oracle(&wdq_k, &x, m_rows, in_dim, out_k);
        let ev = qmm_oracle(&wdq_v, &x, m_rows, in_dim, out_v);
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let max_rows = out_q.max(out_k).max(out_v);
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w_q", pq.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_q", pq.scales, scales_dt))
            .input(TestBuffer::from_vec("w_k", pk.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_k", pk.scales, scales_dt))
            .input(TestBuffer::from_vec("w_v", pv.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_v", pv.scales, scales_dt))
            .input(TestBuffer::zeros("q_buf", m_rows * out_q, dt))
            .input(TestBuffer::zeros("k_buf", m_rows * out_k, dt))
            .input(TestBuffer::zeros("v_buf", m_rows * out_v, dt))
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", pq.global.max(pk.global).max(pv.global));
        }
        s.expect(TestBuffer::from_vec("q_buf", pack_f32(&eq, dt), dt))
            .expect(TestBuffer::from_vec("k_buf", pack_f32(&ek, dt), dt))
            .expect(TestBuffer::from_vec("v_buf", pack_f32(&ev, dt), dt))
            .grid_3d(max_rows as u32, m_rows as u32, 3, [TPG, 1, 1])
    }

    // out_q 16, out_k/out_v 4, in_dim 256 (÷ 16/32), m 2.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(mt_mxfp4_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 16, 4, 4, 256, 2, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(mt_nvfp4_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 16, 4, 4, 256, 2, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxfp8_e4m3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxfp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(mt_nvfp8_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 16, 4, 4, 256, 2, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(mt_fp4_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Fp4, 16, 4, 4, 256, 2, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            mt_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            mt_fp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(mt_int8_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int8, 16, 4, 4, 256, 2, dt)
    }
}

/// Small-batch prefill (M=8) benches at a Qwen3-class fused-QKV shape
/// (out_q=4096, out_k=out_v=1024, in_dim=4096). Throughput is
/// data-independent → random packed buffers.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn qkv_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_q: usize,
        out_k: usize,
        out_v: usize,
        in_dim: usize,
        m: usize,
        dt: DType,
    ) -> BenchSetup {
        let bs = fmt.block_size();
        let (codes_dt, code_div) =
            if fmt.element_bits() == 4 { (DType::U32, 8) } else { (DType::U8, 1) };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let max_rows = out_q.max(out_k).max(out_v);
        let sz = dt.size_bytes();
        let codes = |o: usize| o * in_dim / code_div;
        let scl = |o: usize| o * (in_dim / bs);
        let bytes = (codes(out_q) + codes(out_k) + codes(out_v)) * codes_dt.size_bytes()
            + (scl(out_q) + scl(out_k) + scl(out_v)) * scales_dt.size_bytes()
            + m * in_dim * sz
            + m * (out_q + out_k + out_v) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::random("w_q", codes(out_q), codes_dt))
            .buffer(BenchBuffer::random("scales_q", scl(out_q), scales_dt))
            .buffer(BenchBuffer::random("w_k", codes(out_k), codes_dt))
            .buffer(BenchBuffer::random("scales_k", scl(out_k), scales_dt))
            .buffer(BenchBuffer::random("w_v", codes(out_v), codes_dt))
            .buffer(BenchBuffer::random("scales_v", scl(out_v), scales_dt))
            .buffer(BenchBuffer::zeros("q_buf", m * out_q, dt).output())
            .buffer(BenchBuffer::zeros("k_buf", m * out_k, dt).output())
            .buffer(BenchBuffer::zeros("v_buf", m * out_v, dt).output())
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", bs as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(max_rows as u32, m as u32, 3, [64, 1, 1])
            .bytes_moved(bytes as u64)
            // 3 fused qmms: 2 * m * (out_q + out_k + out_v) * in_dim
            .flops(2 * m as u64 * (out_q + out_k + out_v) as u64 * in_dim as u64)
            .with_shape_label(format!(
                "{} m={m} q={out_q} k={out_k} v={out_v} in={in_dim}",
                fmt.name()
            ))
    }

    #[bench(name = "ffai/batched_qkv_block_qmm/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxfp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxfp8_e4m3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxfp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_fp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_fp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(name = "ffai/batched_qkv_block_qmm/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int8,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
}

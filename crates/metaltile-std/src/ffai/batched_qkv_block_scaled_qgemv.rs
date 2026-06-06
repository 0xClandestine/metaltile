//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **batched Q/K/V block-scaled dequantizing GEMV** for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! One dispatch computes the three projections `out = [Wq·x | Wk·x | Wv·x]`
//! over a shared single-token activation `x`. Mirrors the int4 scalar variant
//! `ffai_batched_qkv_qgemv`: the grid z-dimension `program_id::<2>()` selects
//! the matrix (0→Q, 1→K, 2→V), `program_id::<0>()` the output row; the block-
//! scaled weight decode of `mlx/block_scaled_matmul.rs` replaces the int affine.
//!
//! ## DISPATCH INVARIANTS (identical to the int4 scalar variant)
//!
//! - **Mode: Reduction**, `grid = [max(out_q,out_k,out_v), 1, 3]`,
//!   `tpg = [TPG, 1, 1]`, TPG ≥ 32 & a multiple of 32. One TG per (matrix,row);
//!   rows past a matrix's `out_*` no-op.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_*, in_dim/8]` u32 (4-bit) or `[out_*, in_dim]` u8 (8-bit);
//!   scales `[out_*, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!   Output is the `out_q+out_k+out_v` concatenation. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 batched Q/K/V GEMV — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u32>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u32>,
    scales_v: Tensor<u8>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
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
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// nvfp4 batched Q/K/V GEMV — E2M1 (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u32>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u32>,
    scales_v: Tensor<u8>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        mt_decode_e4m3(load(scales_q[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                        mt_decode_e4m3(load(scales_k[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                        mt_decode_e4m3(load(scales_v[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E4M3) batched Q/K/V GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u8>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u8>,
    scales_v: Tensor<u8>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_q[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_q[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_k[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_k[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_v[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_v[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E5M2) batched Q/K/V GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u8>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u8>,
    scales_v: Tensor<u8>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_q[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_q[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_k[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_k[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_v[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_v[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// nvfp8 batched Q/K/V GEMV — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f32>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 batched Q/K/V GEMVs ─────
// These share the block-scaled framework but store a raw per-group FP32 scale
// (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 +
// f32 scale), so it reuses `mt_nvfp8_batched_qkv_qgemv` — only fp4 (4-bit
// E2M1), fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) decode here.

/// Legacy fp4 batched Q/K/V GEMV — E2M1 weights (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp4_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u32>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u32>,
    scales_v: Tensor<f32>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
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
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// Legacy fp8 (E5M2) batched Q/K/V GEMV — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f32>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// Symmetric int8 batched Q/K/V GEMV — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f32>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f32>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f32>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

// ── Symmetric sub-byte integer batched Q/K/V GEMVs (int2/3/4/5/6 + MXINT2..6)
// + MXINT8 ──────────────────────────────────────────────────────────────────
// Each element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words (per-row word-aligned; element `c` at bit `c·bits`
// within that row's bit-stream). Decode mirrors the proven `int_qgemv_*` macros
// of `mlx/block_scaled_matmul.rs` exactly: extract the low N bits with a
// straddle-aware two-word read, sign-extend in float (subtract 2^N when the top
// bit is set; `$half`/`$full` are 2^(N-1) / 2^N), then multiply by the block
// scale and the matching activation. The three weight matrices (Q/K/V) are
// separate buffers indexed by `program_id::<2>()`, exactly as the existing
// `int8`/E8M0 formats here lay them out — only the per-element decode changes,
// applied identically to each of the three weight rows. `$half`/`$full` are
// passed as literals to keep the constexpr math out of the DSL shift operands.
// The dispatch geometry is unchanged from the rest of the family (Reduction,
// `grid = [max(out_q,out_k,out_v), 1, 3]`, `tpg = [TPG, 1, 1]`).

/// FP32-scaled symmetric int batched Q/K/V GEMV (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP32 scale, dotted with the shared activation.
/// Each matrix's `row_word_off` indexes its own tight bit-stream
/// (`in_dim · bits / 32` u32 words per row); the contraction dim is a multiple
/// of 32 so every row starts word-aligned.
macro_rules! int_batched_qkv_qgemv_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            w_q: Tensor<u32>,
            scales_q: Tensor<f32>,
            w_k: Tensor<u32>,
            scales_k: Tensor<f32>,
            w_v: Tensor<u32>,
            scales_v: Tensor<f32>,
            output: Tensor<T>,
            #[constexpr] out_q: u32,
            #[constexpr] out_k: u32,
            #[constexpr] out_v: u32,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let matrix = program_id::<2>();
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;
            let iters = (in_dim + lsize - 1u32) / lsize;
            let mut acc = 0.0f32;
            if matrix == 0u32 {
                if row < out_q {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_q[row_word_off + word_idx]);
                            let w1 = load(
                                w_q[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_q[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 1u32 {
                if row < out_k {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_k[row_word_off + word_idx]);
                            let w1 = load(
                                w_k[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_k[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 2u32 {
                if row < out_v {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_v[row_word_off + word_idx]);
                            let w1 = load(
                                w_v[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_v[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                if matrix == 0u32 {
                    if row < out_q {
                        store(output[row], total.cast::<T>());
                    }
                }
                if matrix == 1u32 {
                    if row < out_k {
                        store(output[out_q + row], total.cast::<T>());
                    }
                }
                if matrix == 2u32 {
                    if row < out_v {
                        store(output[out_q + out_k + row], total.cast::<T>());
                    }
                }
            }
        }
    };
}
int_batched_qkv_qgemv_f32!(mt_int2_batched_qkv_qgemv, 2u32, 2u32, 4.0f32);
int_batched_qkv_qgemv_f32!(mt_int3_batched_qkv_qgemv, 3u32, 4u32, 8.0f32);
int_batched_qkv_qgemv_f32!(mt_int4_batched_qkv_qgemv, 4u32, 8u32, 16.0f32);
int_batched_qkv_qgemv_f32!(mt_int5_batched_qkv_qgemv, 5u32, 16u32, 32.0f32);
int_batched_qkv_qgemv_f32!(mt_int6_batched_qkv_qgemv, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int batched Q/K/V GEMV (MXINT2/3/4/5/6): per-element
/// bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`, dotted with the
/// shared activation. Same straddle-aware decode and per-matrix layout as
/// `int_batched_qkv_qgemv_f32`; only the scale axis differs (one u8 exponent
/// per block instead of a raw f32).
macro_rules! int_batched_qkv_qgemv_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            w_q: Tensor<u32>,
            scales_q: Tensor<u8>,
            w_k: Tensor<u32>,
            scales_k: Tensor<u8>,
            w_v: Tensor<u32>,
            scales_v: Tensor<u8>,
            output: Tensor<T>,
            #[constexpr] out_q: u32,
            #[constexpr] out_k: u32,
            #[constexpr] out_v: u32,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let matrix = program_id::<2>();
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;
            let iters = (in_dim + lsize - 1u32) / lsize;
            let mut acc = 0.0f32;
            if matrix == 0u32 {
                if row < out_q {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_q[row_word_off + word_idx]);
                            let w1 = load(
                                w_q[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_q[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 1u32 {
                if row < out_k {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_k[row_word_off + word_idx]);
                            let w1 = load(
                                w_k[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_k[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 2u32 {
                if row < out_v {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_v[row_word_off + word_idx]);
                            let w1 = load(
                                w_v[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_v[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                if matrix == 0u32 {
                    if row < out_q {
                        store(output[row], total.cast::<T>());
                    }
                }
                if matrix == 1u32 {
                    if row < out_k {
                        store(output[out_q + row], total.cast::<T>());
                    }
                }
                if matrix == 2u32 {
                    if row < out_v {
                        store(output[out_q + out_k + row], total.cast::<T>());
                    }
                }
            }
        }
    };
}
int_batched_qkv_qgemv_e8m0!(mt_mxint2_batched_qkv_qgemv, 2u32, 2u32, 4.0f32);
int_batched_qkv_qgemv_e8m0!(mt_mxint3_batched_qkv_qgemv, 3u32, 4u32, 8.0f32);
int_batched_qkv_qgemv_e8m0!(mt_mxint4_batched_qkv_qgemv, 4u32, 8u32, 16.0f32);
int_batched_qkv_qgemv_e8m0!(mt_mxint5_batched_qkv_qgemv, 5u32, 16u32, 32.0f32);
int_batched_qkv_qgemv_e8m0!(mt_mxint6_batched_qkv_qgemv, 6u32, 32u32, 64.0f32);

/// MXINT8 batched Q/K/V GEMV — 8-bit symmetric codes (byte layout, block 32),
/// E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the 8-bit float
/// formats (one byte per code), decode is `mt_decode_int8 → val · scale`. The
/// three weight matrices are separate u8 buffers selected by `program_id::<2>()`.
#[kernel]
pub fn mt_mxint8_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<u8>,
    w_k: Tensor<u8>,
    scales_k: Tensor<u8>,
    w_v: Tensor<u8>,
    scales_v: Tensor<u8>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_q[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_q[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_k[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_k[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_v[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_v[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

// ── FP16-scale twins of the FP32-scaled batched Q/K/V GEMVs ─────────────────
// These are near-clones of the legacy float-scale + symmetric-int kernels above;
// the ONLY change is the per-group scale tensor type — `Tensor<f16>` instead of
// `Tensor<f32>` — and the scale read, which appends `.cast::<f32>()` to widen the
// half scale before the decode math. Element decode (E2M1 / E4M3 / E5M2 / the int
// bit-stream + sign-extend), weight layout, dispatch geometry, staging, and the
// reduction are byte-for-byte identical to the FP32 twin. Applied identically to
// all three Q/K/V scale buffers. `fp8_e4m3_f16` (8-bit E4M3 + f16 scale) reuses
// `mt_nvfp8_f16_batched_qkv_qgemv`, exactly as `fp8_e4m3` reuses the nvfp8 kernel.

/// nvfp8 (f16 scale) batched Q/K/V GEMV — E4M3 weights (block 16), per-block FP16
/// scale. Clone of `mt_nvfp8_batched_qkv_qgemv`; scale tensor is `Tensor<f16>`,
/// read widened with `.cast::<f32>()`. Also serves `Fp8E4m3F16` (same shape).
#[kernel]
pub fn mt_nvfp8_f16_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f16>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f16>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f16>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// fp4 (f16 scale) batched Q/K/V GEMV — E2M1 weights (group 32), per-group FP16
/// scale. Clone of `mt_fp4_batched_qkv_qgemv`; scale tensor is `Tensor<f16>`,
/// read widened with `.cast::<f32>()`.
#[kernel]
pub fn mt_fp4_f16_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<f16>,
    w_k: Tensor<u32>,
    scales_k: Tensor<f16>,
    w_v: Tensor<u32>,
    scales_v: Tensor<f16>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_q[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                    let scale = load(scales_k[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
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
                    let scale = load(scales_v[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// fp8 (E5M2, f16 scale) batched Q/K/V GEMV — 8-bit weights (group 32), per-group
/// FP16 scale. Clone of `mt_fp8_e5m2_batched_qkv_qgemv`; scale tensor is
/// `Tensor<f16>`, read widened with `.cast::<f32>()`.
#[kernel]
pub fn mt_fp8_e5m2_f16_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f16>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f16>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f16>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
            }
        }
    }
}

/// FP16-scaled symmetric int batched Q/K/V GEMV (int2/3/4/5/6): identical to
/// `int_batched_qkv_qgemv_f32` (same straddle-aware bit-stream decode + sign
/// extend + per-matrix layout); only the scale axis differs — one `f16` per group
/// widened to f32 with `.cast::<f32>()` instead of a raw f32.
macro_rules! int_batched_qkv_qgemv_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            w_q: Tensor<u32>,
            scales_q: Tensor<f16>,
            w_k: Tensor<u32>,
            scales_k: Tensor<f16>,
            w_v: Tensor<u32>,
            scales_v: Tensor<f16>,
            output: Tensor<T>,
            #[constexpr] out_q: u32,
            #[constexpr] out_k: u32,
            #[constexpr] out_v: u32,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let matrix = program_id::<2>();
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;
            let iters = (in_dim + lsize - 1u32) / lsize;
            let mut acc = 0.0f32;
            if matrix == 0u32 {
                if row < out_q {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_q[row_word_off + word_idx]);
                            let w1 = load(
                                w_q[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_q[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 1u32 {
                if row < out_k {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_k[row_word_off + word_idx]);
                            let w1 = load(
                                w_k[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_k[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 2u32 {
                if row < out_v {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_v[row_word_off + word_idx]);
                            let w1 = load(
                                w_v[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_v[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                        }
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                if matrix == 0u32 {
                    if row < out_q {
                        store(output[row], total.cast::<T>());
                    }
                }
                if matrix == 1u32 {
                    if row < out_k {
                        store(output[out_q + row], total.cast::<T>());
                    }
                }
                if matrix == 2u32 {
                    if row < out_v {
                        store(output[out_q + out_k + row], total.cast::<T>());
                    }
                }
            }
        }
    };
}
int_batched_qkv_qgemv_f16!(mt_int2_f16_batched_qkv_qgemv, 2u32, 2u32, 4.0f32);
int_batched_qkv_qgemv_f16!(mt_int3_f16_batched_qkv_qgemv, 3u32, 4u32, 8.0f32);
int_batched_qkv_qgemv_f16!(mt_int4_f16_batched_qkv_qgemv, 4u32, 8u32, 16.0f32);
int_batched_qkv_qgemv_f16!(mt_int5_f16_batched_qkv_qgemv, 5u32, 16u32, 32.0f32);
int_batched_qkv_qgemv_f16!(mt_int6_f16_batched_qkv_qgemv, 6u32, 32u32, 64.0f32);

/// int8 (f16 scale) batched Q/K/V GEMV — 8-bit symmetric codes (byte layout,
/// group 64), per-group FP16 scale. Clone of `mt_int8_batched_qkv_qgemv`; scale
/// tensor is `Tensor<f16>`, read widened with `.cast::<f32>()`.
#[kernel]
pub fn mt_int8_f16_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u8>,
    scales_q: Tensor<f16>,
    w_k: Tensor<u8>,
    scales_k: Tensor<f16>,
    w_v: Tensor<u8>,
    scales_v: Tensor<f16>,
    output: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_q {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_q[row_off + c]).cast::<u32>());
                    let scale = load(scales_q[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_k[row_off + c]).cast::<u32>());
                    let scale = load(scales_k[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_v[row_off + c]).cast::<u32>());
                    let scale = load(scales_v[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(output[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(output[out_q + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(output[out_q + out_k + row], total.cast::<T>());
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

    const TPG: u32 = 64;

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

    fn gemv(wdq: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim).map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * x[c]).sum()).collect()
    }

    fn qkv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_q: usize,
        out_k: usize,
        out_v: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let pack_w = |out_dim: usize, seed: usize| {
            let w = weights(out_dim, in_dim, seed);
            let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
            let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
            (p, wdq)
        };
        let (pq, wdq_q) = pack_w(out_q, 0);
        let (pk, wdq_k) = pack_w(out_k, 1);
        let (pv, wdq_v) = pack_w(out_v, 2);
        let x_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let mut expected = gemv(&wdq_q, &x, out_q, in_dim);
        expected.extend(gemv(&wdq_k, &x, out_k, in_dim));
        expected.extend(gemv(&wdq_v, &x, out_v, in_dim));
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new integer/fp16 formats pick up
        // the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
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
            .input(TestBuffer::zeros("output", out_q + out_k + out_v, dt))
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", pq.global.max(pk.global).max(pv.global));
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            max_rows as u32,
            1,
            3,
            [TPG, 1, 1],
        )
    }

    // out_q 16, out_k/out_v 4, in_dim 256 (÷ 16/32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_mxfp4_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 16, 4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_nvfp4_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 16, 4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxfp8_e4m3_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxfp8_e5m2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_nvfp8_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 16, 4, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_fp4_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Fp4, 16, 4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_nvfp8_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_fp8_e5m2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_int8_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Int8, 16, 4, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 is a multiple of 32,
    // so `in_dim*bits % 32 == 0` for every width — each weight row's bit-stream
    // is word-aligned. The kernels and oracle share the codec, so the GPU
    // output tracks the dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_int2_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Int2, 16, 4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_int3_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Int3, 16, 4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_int4_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Int4, 16, 4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_int5_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Int5, 16, 4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(mt_int6_batched_qkv_qgemv::kernel_ir_for(dt), QFormat::Int6, 16, 4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxint2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxint3_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxint4_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxint5_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxint6_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_mxint8_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            16,
            4,
            4,
            256,
            dt,
        )
    }

    // FP16-scale twins: same element packing + geometry as their FP32 twins, only
    // the per-group scale binds as f16 (handled by the `scales_dt` match). in_dim
    // 256 stays a multiple of 32/64, so every weight row's bit-stream is
    // word-aligned. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel (same 8-bit-E4M3
    // + f16-scale shape), exactly as `fp8_e4m3` reuses the nvfp8 kernel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_nvfp8_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_nvfp8_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_fp4_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_fp8_e5m2_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_int2_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_int3_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_int4_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_int5_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_int6_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_batched_qkv_qgemv(dt: DType) -> TestSetup {
        qkv_setup(
            mt_int8_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            16,
            4,
            4,
            256,
            dt,
        )
    }
}

/// Decode-shape benches at a Qwen3-class fused-QKV shape (out_q=4096,
/// out_k=out_v=1024, in_dim=4096). Throughput is data-independent → random.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn qkv_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_q: usize,
        out_k: usize,
        out_v: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let bs = fmt.block_size();
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let codes_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let max_rows = out_q.max(out_k).max(out_v);
        let sz = dt.size_bytes();
        // 8-bit codes stay `out·in_dim` bytes; sub-byte widths pack each row's
        // `out·in_dim` elements into `bitstream_words` u32 words (4-bit collapses
        // to the old `n/8`). in_dim is a multiple of 32 so each row is word-aligned.
        let codes = |o: usize| {
            if fmt.element_bits() == 8 {
                o * in_dim
            } else {
                crate::quant::format::bitstream_words(o * in_dim, fmt.element_bits())
            }
        };
        let scl = |o: usize| o * (in_dim / bs);
        let bytes = (codes(out_q) + codes(out_k) + codes(out_v)) * codes_dt.size_bytes()
            + (scl(out_q) + scl(out_k) + scl(out_v)) * scales_dt.size_bytes()
            + in_dim * sz
            + (out_q + out_k + out_v) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("w_q", codes(out_q), codes_dt))
            .buffer(BenchBuffer::random("scales_q", scl(out_q), scales_dt))
            .buffer(BenchBuffer::random("w_k", codes(out_k), codes_dt))
            .buffer(BenchBuffer::random("scales_k", scl(out_k), scales_dt))
            .buffer(BenchBuffer::random("w_v", codes(out_v), codes_dt))
            .buffer(BenchBuffer::random("scales_v", scl(out_v), scales_dt))
            .buffer(BenchBuffer::zeros("output", out_q + out_k + out_v, dt).output())
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", bs as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(max_rows as u32, 1, 3, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * (out_q + out_k + out_v) as u64 * in_dim as u64)
            .with_shape_label(format!("{} q={out_q} k={out_k} v={out_v} in={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxfp4_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp4_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxfp8_e4m3_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxfp8_e5m2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp8_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_fp4_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp4,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp8_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_fp8_e5m2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int8_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int8,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int2,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int3_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int3,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int4_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int4,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int5_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int5,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int6_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int6,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxint2_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxint3_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxint4_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxint5_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxint6_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_mxint8_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3
    // + f16-scale shape); the others mirror their FP32 twin's bench one-for-one.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp8_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_nvfp8_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_fp4_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_fp8_e5m2_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int2_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int3_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int4_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int5_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int6_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            mt_int8_f16_batched_qkv_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            4096,
            1024,
            1024,
            4096,
            dt,
        )
    }
}

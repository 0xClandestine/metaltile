//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **batched 4-output block-scaled dequantizing GEMM** (M>1) for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! Multi-token sibling of `batched_4_block_scaled_qgemv`: instead of a single
//! activation row it consumes `x: [M, in_dim]` and produces row `m` of FOUR
//! independent projections that share the same activation —
//!   a_buf: [M, out_a] T
//!   b_buf: [M, out_b] T
//!   c_buf: [M, out_c] T
//!   d_buf: [M, out_d] T
//!
//! One dispatch computes `out_X[m, n] = Σ_k dequant(W_X[n, k]) · x[m, k]` for
//! all four projections. The block-scaled weight decode of
//! `mlx/block_scaled_qmm.rs` replaces the int affine; the (matrix, token, row)
//! geometry mirrors the int4 `ffai_batched_4_qmm_fast`.
//!
//! Unlike the Q/K/V multi-token variant — which writes THREE buffers — the four
//! projections here write to FOUR SEPARATE output buffers (`a_buf`, `b_buf`,
//! `c_buf`, `d_buf`), each indexed by `mr * out_X + row`. Callers may alias all
//! four into one backing allocation; the kernel only sees four base pointers.
//! This matches `mt_*_batched_4_qgemv`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [max(out_a,out_b,out_c,out_d), M, 4]`,
//!   `tpg = [TPG, 1, 1]`, TPG >= 32 & a multiple of 32. One TG per
//!   `(matrix, m_token, out_row)`; rows past a matrix's `out_*` no-op.
//!   * `program_id::<2>()` selects matrix (0=A, 1=B, 2=C, 3=D).
//!   * `program_id::<1>()` selects batched token `mr` (0..M).
//!   * `program_id::<0>()` selects the output row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_*, in_dim/8]` u32 (4-bit) or `[out_*, in_dim]` u8 (8-bit);
//!   scales `[out_*, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!   `x` is `[M, in_dim]`, each `*_buf` is `[M, out_*]`, all row-major. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 batched 4-output GEMM (M>1) — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u32>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u32>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u32>,
    scales_d: Tensor<u8>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_a[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_a[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_b[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_b[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_c[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_c[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_d[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_d[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// nvfp4 batched 4-output GEMM (M>1) — E2M1 (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u32>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u32>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u32>,
    scales_d: Tensor<u8>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        mt_decode_e4m3(load(scales_a[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_a[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        mt_decode_e4m3(load(scales_b[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_b[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        mt_decode_e4m3(load(scales_c[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_c[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        mt_decode_e4m3(load(scales_d[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_d[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E4M3) batched 4-output GEMM (M>1) — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u8>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u8>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u8>,
    scales_d: Tensor<u8>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_a[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_a[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_b[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_b[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_c[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_c[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_d[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_d[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E5M2) batched 4-output GEMM (M>1) — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u8>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u8>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u8>,
    scales_d: Tensor<u8>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_a[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_a[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_b[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_b[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_c[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_c[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_d[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_d[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// nvfp8 batched 4-output GEMM (M>1) — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f32>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f32>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f32>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f32>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 batched-4 GEMMs ─────────
// These share the block-scaled framework but store a raw per-group FP32 scale
// (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 +
// f32 scale), so it reuses `mt_nvfp8_batched_4_qmm` — only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 batched 4-output GEMM (M>1) — E2M1 weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp4_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<f32>,
    w_b: Tensor<u32>,
    scales_b: Tensor<f32>,
    w_c: Tensor<u32>,
    scales_c: Tensor<f32>,
    w_d: Tensor<u32>,
    scales_d: Tensor<f32>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_a[row_block_off + blk]);
                    let packed = load(w_a[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_b[row_block_off + blk]);
                    let packed = load(w_b[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_c[row_block_off + blk]);
                    let packed = load(w_c[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_d[row_block_off + blk]);
                    let packed = load(w_d[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// Legacy fp8 (E5M2) batched 4-output GEMM (M>1) — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f32>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f32>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f32>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f32>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// Symmetric int8 batched 4-output GEMM (M>1) — 8-bit codes (group 64), per-group
/// FP32 scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f32>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f32>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f32>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f32>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

// ── Symmetric sub-byte integer batched-4 GEMMs (int2/3/4/5/6 + MXINT2..6) ────
// The element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words (per-row word-aligned; element `c` of a row begins at
// bit `c·bits` within that row's bit-stream). Each weight row holds
// `in_dim·bits / 32` u32 words, so row `row`'s words start at
// `row · words_per_row` — the sub-byte analogue of the 8-bit `row * in_dim`
// byte base used by `mt_int8_batched_4_qmm`. Decode mirrors the GPU-verified
// `block_scaled_matmul::int_qgemv_*` / `block_scaled_dequant::int_dequant_*`
// macros exactly: a straddle-aware two-word read extracts the low N bits, then
// the value is sign-extended in float (subtract 2^N when the top bit is set;
// `$half`/`$full` are 2^(N-1) / 2^N) and multiplied by the block scale and the
// matching activation. `$half`/`$full` are passed as literals to keep the
// constexpr math out of the DSL shift operands.
//
// The (matrix, token, row) geometry, the four `if matrix == N` branches, and the
// `reduce_sum` + `tid == 0` store are IDENTICAL to the 8-bit `int8` kernel —
// only the per-element decode changes — so no new dispatch shape is introduced.

/// FP32-scaled symmetric sub-byte int batched-4 GEMM (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP32 scale, dotted with the shared activation.
/// `row_word_off` indexes each weight's tight bit-stream
/// (`in_dim · bits / 32` u32 words per row), mirroring the int8 `row * in_dim`.
macro_rules! int_batched_4_qmm_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            w_a: Tensor<u32>,
            scales_a: Tensor<f32>,
            w_b: Tensor<u32>,
            scales_b: Tensor<f32>,
            w_c: Tensor<u32>,
            scales_c: Tensor<f32>,
            w_d: Tensor<u32>,
            scales_d: Tensor<f32>,
            mut a_buf: Tensor<T>,
            mut b_buf: Tensor<T>,
            mut c_buf: Tensor<T>,
            mut d_buf: Tensor<T>,
            #[constexpr] out_a: u32,
            #[constexpr] out_b: u32,
            #[constexpr] out_c: u32,
            #[constexpr] out_d: u32,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let matrix = program_id::<2>();
            let mr = program_id::<1>();
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;
            let x_row_off = mr * in_dim;
            let iters = (in_dim + lsize - 1u32) / lsize;
            let mut acc = 0.0f32;
            if matrix == 0u32 {
                if row < out_a {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_a[row_word_off + word_idx]);
                            let w1 = load(
                                w_a[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_a[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 1u32 {
                if row < out_b {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_b[row_word_off + word_idx]);
                            let w1 = load(
                                w_b[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_b[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 2u32 {
                if row < out_c {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_c[row_word_off + word_idx]);
                            let w1 = load(
                                w_c[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_c[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 3u32 {
                if row < out_d {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_d[row_word_off + word_idx]);
                            let w1 = load(
                                w_d[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale = load(scales_d[row_block_off + c / block_size]);
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                if matrix == 0u32 {
                    if row < out_a {
                        store(a_buf[mr * out_a + row], total.cast::<T>());
                    }
                }
                if matrix == 1u32 {
                    if row < out_b {
                        store(b_buf[mr * out_b + row], total.cast::<T>());
                    }
                }
                if matrix == 2u32 {
                    if row < out_c {
                        store(c_buf[mr * out_c + row], total.cast::<T>());
                    }
                }
                if matrix == 3u32 {
                    if row < out_d {
                        store(d_buf[mr * out_d + row], total.cast::<T>());
                    }
                }
            }
        }
    };
}
int_batched_4_qmm_f32!(mt_int2_batched_4_qmm, 2u32, 2u32, 4.0f32);
int_batched_4_qmm_f32!(mt_int3_batched_4_qmm, 3u32, 4u32, 8.0f32);
int_batched_4_qmm_f32!(mt_int4_batched_4_qmm, 4u32, 8u32, 16.0f32);
int_batched_4_qmm_f32!(mt_int5_batched_4_qmm, 5u32, 16u32, 32.0f32);
int_batched_4_qmm_f32!(mt_int6_batched_4_qmm, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric sub-byte int batched-4 GEMM (MXINT2/3/4/5/6):
/// per-element bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`, dotted
/// with the shared activation. Same straddle-aware decode and per-matrix
/// reduction as `int_batched_4_qmm_f32`; only the scale axis differs (one u8
/// exponent per block instead of a raw f32).
macro_rules! int_batched_4_qmm_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            w_a: Tensor<u32>,
            scales_a: Tensor<u8>,
            w_b: Tensor<u32>,
            scales_b: Tensor<u8>,
            w_c: Tensor<u32>,
            scales_c: Tensor<u8>,
            w_d: Tensor<u32>,
            scales_d: Tensor<u8>,
            mut a_buf: Tensor<T>,
            mut b_buf: Tensor<T>,
            mut c_buf: Tensor<T>,
            mut d_buf: Tensor<T>,
            #[constexpr] out_a: u32,
            #[constexpr] out_b: u32,
            #[constexpr] out_c: u32,
            #[constexpr] out_d: u32,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let matrix = program_id::<2>();
            let mr = program_id::<1>();
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;
            let x_row_off = mr * in_dim;
            let iters = (in_dim + lsize - 1u32) / lsize;
            let mut acc = 0.0f32;
            if matrix == 0u32 {
                if row < out_a {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_a[row_word_off + word_idx]);
                            let w1 = load(
                                w_a[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_a[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 1u32 {
                if row < out_b {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_b[row_word_off + word_idx]);
                            let w1 = load(
                                w_b[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_b[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 2u32 {
                if row < out_c {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_c[row_word_off + word_idx]);
                            let w1 = load(
                                w_c[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_c[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 3u32 {
                if row < out_d {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_d[row_word_off + word_idx]);
                            let w1 = load(
                                w_d[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let sbits =
                                load(scales_d[row_block_off + c / block_size]).cast::<f32>();
                            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                if matrix == 0u32 {
                    if row < out_a {
                        store(a_buf[mr * out_a + row], total.cast::<T>());
                    }
                }
                if matrix == 1u32 {
                    if row < out_b {
                        store(b_buf[mr * out_b + row], total.cast::<T>());
                    }
                }
                if matrix == 2u32 {
                    if row < out_c {
                        store(c_buf[mr * out_c + row], total.cast::<T>());
                    }
                }
                if matrix == 3u32 {
                    if row < out_d {
                        store(d_buf[mr * out_d + row], total.cast::<T>());
                    }
                }
            }
        }
    };
}
int_batched_4_qmm_e8m0!(mt_mxint2_batched_4_qmm, 2u32, 2u32, 4.0f32);
int_batched_4_qmm_e8m0!(mt_mxint3_batched_4_qmm, 3u32, 4u32, 8.0f32);
int_batched_4_qmm_e8m0!(mt_mxint4_batched_4_qmm, 4u32, 8u32, 16.0f32);
int_batched_4_qmm_e8m0!(mt_mxint5_batched_4_qmm, 5u32, 16u32, 32.0f32);
int_batched_4_qmm_e8m0!(mt_mxint6_batched_4_qmm, 6u32, 32u32, 64.0f32);

/// MXINT8 batched-4 GEMM (M>1) — 8-bit symmetric codes (byte layout, block 32),
/// E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the 8-bit float
/// formats (one byte per code); decode is `mt_decode_int8 → val · scale`. Same
/// (matrix, token, row) geometry as `mt_int8_batched_4_qmm`.
#[kernel]
pub fn mt_mxint8_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u8>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u8>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u8>,
    scales_d: Tensor<u8>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_a[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_a[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_b[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_b[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_c[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_c[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_d[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_d[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

// ── FP16-scale twins of the FP32-scaled batched-4 GEMMs ─────────────────────
// Identical element decode, weight indexing, (matrix, token, row) geometry,
// staging, and reduction as their FP32 twins above — only the scale tensor type
// changes from `Tensor<f32>` to `Tensor<f16>`, read as a native half and cast to
// f32 (`load(scales[...]).cast::<f32>()`). The GPU half load matches the host
// `f16_scale_decode`, so the dequant-then-matmul oracle still holds exactly. No
// new dispatch shape is introduced. `nvfp8_f16` also serves `fp8_e4m3_f16` (same
// 8-bit-E4M3 + scale shape), exactly as `nvfp8` serves `fp8_e4m3` today.

/// nvfp8 (FP16 scale) batched 4-output GEMM (M>1) — E4M3 weights (block 16),
/// per-block FP16 scale. FP16-scale twin of `mt_nvfp8_batched_4_qmm`; also serves
/// `fp8_e4m3_f16` (same 8-bit-E4M3 + scale shape).
#[kernel]
pub fn mt_nvfp8_f16_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f16>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f16>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f16>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f16>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e4m3(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// fp4 (FP16 scale) batched 4-output GEMM (M>1) — E2M1 weights (group 32),
/// per-group FP16 scale. FP16-scale twin of `mt_fp4_batched_4_qmm`.
#[kernel]
pub fn mt_fp4_f16_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<f16>,
    w_b: Tensor<u32>,
    scales_b: Tensor<f16>,
    w_c: Tensor<u32>,
    scales_c: Tensor<f16>,
    w_d: Tensor<u32>,
    scales_d: Tensor<f16>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_a[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_a[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_b[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_b[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_c[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_c[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = load(scales_d[row_block_off + blk]).cast::<f32>();
                    let packed = load(w_d[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// fp8 (E5M2, FP16 scale) batched 4-output GEMM (M>1) — 8-bit weights (group 32),
/// per-group FP16 scale. FP16-scale twin of `mt_fp8_e5m2_batched_4_qmm`.
#[kernel]
pub fn mt_fp8_e5m2_f16_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f16>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f16>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f16>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f16>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_e5m2(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
            }
        }
    }
}

/// FP16-scaled symmetric sub-byte int batched-4 GEMM (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP16 scale, dotted with the shared activation.
/// FP16-scale twin of `int_batched_4_qmm_f32` — identical straddle-aware decode,
/// weight indexing, and per-matrix reduction; only the scale tensor type changes
/// to `Tensor<f16>` (read as half, cast to f32).
macro_rules! int_batched_4_qmm_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            w_a: Tensor<u32>,
            scales_a: Tensor<f16>,
            w_b: Tensor<u32>,
            scales_b: Tensor<f16>,
            w_c: Tensor<u32>,
            scales_c: Tensor<f16>,
            w_d: Tensor<u32>,
            scales_d: Tensor<f16>,
            mut a_buf: Tensor<T>,
            mut b_buf: Tensor<T>,
            mut c_buf: Tensor<T>,
            mut d_buf: Tensor<T>,
            #[constexpr] out_a: u32,
            #[constexpr] out_b: u32,
            #[constexpr] out_c: u32,
            #[constexpr] out_d: u32,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let matrix = program_id::<2>();
            let mr = program_id::<1>();
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;
            let x_row_off = mr * in_dim;
            let iters = (in_dim + lsize - 1u32) / lsize;
            let mut acc = 0.0f32;
            if matrix == 0u32 {
                if row < out_a {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_a[row_word_off + word_idx]);
                            let w1 = load(
                                w_a[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_a[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 1u32 {
                if row < out_b {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_b[row_word_off + word_idx]);
                            let w1 = load(
                                w_b[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_b[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 2u32 {
                if row < out_c {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_c[row_word_off + word_idx]);
                            let w1 = load(
                                w_c[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_c[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            if matrix == 3u32 {
                if row < out_d {
                    for it in range(0u32, iters, 1u32) {
                        let c = it * lsize + tid;
                        if c < in_dim {
                            let bit_off = c * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w_d[row_word_off + word_idx]);
                            let w1 = load(
                                w_d[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            let scale =
                                load(scales_d[row_block_off + c / block_size]).cast::<f32>();
                            acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                        }
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                if matrix == 0u32 {
                    if row < out_a {
                        store(a_buf[mr * out_a + row], total.cast::<T>());
                    }
                }
                if matrix == 1u32 {
                    if row < out_b {
                        store(b_buf[mr * out_b + row], total.cast::<T>());
                    }
                }
                if matrix == 2u32 {
                    if row < out_c {
                        store(c_buf[mr * out_c + row], total.cast::<T>());
                    }
                }
                if matrix == 3u32 {
                    if row < out_d {
                        store(d_buf[mr * out_d + row], total.cast::<T>());
                    }
                }
            }
        }
    };
}
int_batched_4_qmm_f16!(mt_int2_f16_batched_4_qmm, 2u32, 2u32, 4.0f32);
int_batched_4_qmm_f16!(mt_int3_f16_batched_4_qmm, 3u32, 4u32, 8.0f32);
int_batched_4_qmm_f16!(mt_int4_f16_batched_4_qmm, 4u32, 8u32, 16.0f32);
int_batched_4_qmm_f16!(mt_int5_f16_batched_4_qmm, 5u32, 16u32, 32.0f32);
int_batched_4_qmm_f16!(mt_int6_f16_batched_4_qmm, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) batched-4 GEMM (M>1) — 8-bit symmetric codes (byte layout,
/// group 64), per-group FP16 scale. FP16-scale twin of `mt_int8_batched_4_qmm`;
/// decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_f16_batched_4_qmm<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f16>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f16>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f16>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f16>,
    mut a_buf: Tensor<T>,
    mut b_buf: Tensor<T>,
    mut c_buf: Tensor<T>,
    mut d_buf: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
        if row < out_a {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = mt_decode_int8(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_buf[mr * out_a + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_buf[mr * out_b + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_buf[mr * out_c + row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_buf[mr * out_d + row], total.cast::<T>());
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

    /// Reduction-contract threadgroup width (>= 32, multiple of 32).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` quantized weights (mixed signs).
    /// `seed` decorrelates the four matrices.
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
    fn batched_4_qmm_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_a: usize,
        out_b: usize,
        out_c: usize,
        out_d: usize,
        in_dim: usize,
        m_rows: usize,
        dt: DType,
    ) -> TestSetup {
        // Pack + dequant each of the four weight matrices (distinct seeds).
        let pack_w = |out_dim: usize, seed: usize| {
            let w = weights(out_dim, in_dim, seed);
            let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
            let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
            (p, wdq)
        };
        let (pa, wdq_a) = pack_w(out_a, 0);
        let (pb, wdq_b) = pack_w(out_b, 1);
        let (pc, wdq_c) = pack_w(out_c, 2);
        let (pd, wdq_d) = pack_w(out_d, 3);
        // Build x as [m_rows, in_dim] and round it through the storage dtype.
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        // Four SEPARATE expected outputs — each matrix writes its own buffer.
        let ea = qmm_oracle(&wdq_a, &x, m_rows, in_dim, out_a);
        let eb = qmm_oracle(&wdq_b, &x, m_rows, in_dim, out_b);
        let ec = qmm_oracle(&wdq_c, &x, m_rows, in_dim, out_c);
        let ed = qmm_oracle(&wdq_d, &x, m_rows, in_dim, out_d);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new formats pick up the right
        // buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let max_rows = out_a.max(out_b).max(out_c).max(out_d);
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w_a", pa.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_a", pa.scales, scales_dt))
            .input(TestBuffer::from_vec("w_b", pb.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_b", pb.scales, scales_dt))
            .input(TestBuffer::from_vec("w_c", pc.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_c", pc.scales, scales_dt))
            .input(TestBuffer::from_vec("w_d", pd.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_d", pd.scales, scales_dt))
            .input(TestBuffer::zeros("a_buf", m_rows * out_a, dt))
            .input(TestBuffer::zeros("b_buf", m_rows * out_b, dt))
            .input(TestBuffer::zeros("c_buf", m_rows * out_c, dt))
            .input(TestBuffer::zeros("d_buf", m_rows * out_d, dt))
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", pa.global.max(pb.global).max(pc.global).max(pd.global));
        }
        s.expect(TestBuffer::from_vec("a_buf", pack_f32(&ea, dt), dt))
            .expect(TestBuffer::from_vec("b_buf", pack_f32(&eb, dt), dt))
            .expect(TestBuffer::from_vec("c_buf", pack_f32(&ec, dt), dt))
            .expect(TestBuffer::from_vec("d_buf", pack_f32(&ed, dt), dt))
            .grid_3d(max_rows as u32, m_rows as u32, 4, [TPG, 1, 1])
    }

    // out_a 16, out_b/out_c/out_d 4, in_dim 256 (÷ 16/32), m 2.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxfp4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxfp4,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_nvfp4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Nvfp4,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxfp8_e4m3_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxfp8_e5m2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_nvfp8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_fp4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp4,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_nvfp8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_fp8_e5m2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int8,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each weight row's bit-stream is
    // word-aligned, and is divisible by every block/group size (32 / 64). The
    // kernel and oracle share the codec, so the GPU output tracks the
    // dequant-then-matmul reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int2,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int3_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int3,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int4,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int5_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int5,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int6_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int6,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxint2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint2,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxint3_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint3,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxint4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint4,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxint5_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint5,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxint6_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint6,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_mxint8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint8,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode in
    // their own kernel. in_dim 256 stays divisible by every block/group size
    // (16 / 32 / 64) and keeps each weight row's bit-stream word-aligned.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_nvfp8_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_nvfp8_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_fp4_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp4F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_fp8_e5m2_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int2_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int2F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int3_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int3F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int4_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int4F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int5_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int5F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int6_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int6F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_batched_4_qmm(dt: DType) -> TestSetup {
        batched_4_qmm_setup(
            mt_int8_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int8F16,
            16,
            4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
}

/// Small-batch prefill (M=8) benches at a fused 4-projection shape
/// (out_a=out_b=out_c=out_d=4096, in_dim=4096). Throughput is
/// data-independent → random packed buffers.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn batched_4_qmm_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_a: usize,
        out_b: usize,
        out_c: usize,
        out_d: usize,
        in_dim: usize,
        m: usize,
        dt: DType,
    ) -> BenchSetup {
        let bs = fmt.block_size();
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        // Both axes are driven off the format so new integer formats pick up the
        // right buffer types.
        let codes_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let max_rows = out_a.max(out_b).max(out_c).max(out_d);
        let sz = dt.size_bytes();
        // 8-bit: one byte per code; sub-byte: `bitstream_words` u32 words for the
        // matrix's `o * in_dim` elements (4-bit collapses to the old `n / 8`).
        let codes = |o: usize| {
            if fmt.element_bits() == 8 {
                o * in_dim
            } else {
                crate::quant::format::bitstream_words(o * in_dim, fmt.element_bits())
            }
        };
        let scl = |o: usize| o * (in_dim / bs);
        let bytes = (codes(out_a) + codes(out_b) + codes(out_c) + codes(out_d))
            * codes_dt.size_bytes()
            + (scl(out_a) + scl(out_b) + scl(out_c) + scl(out_d)) * scales_dt.size_bytes()
            + m * in_dim * sz
            + m * (out_a + out_b + out_c + out_d) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::random("w_a", codes(out_a), codes_dt))
            .buffer(BenchBuffer::random("scales_a", scl(out_a), scales_dt))
            .buffer(BenchBuffer::random("w_b", codes(out_b), codes_dt))
            .buffer(BenchBuffer::random("scales_b", scl(out_b), scales_dt))
            .buffer(BenchBuffer::random("w_c", codes(out_c), codes_dt))
            .buffer(BenchBuffer::random("scales_c", scl(out_c), scales_dt))
            .buffer(BenchBuffer::random("w_d", codes(out_d), codes_dt))
            .buffer(BenchBuffer::random("scales_d", scl(out_d), scales_dt))
            .buffer(BenchBuffer::zeros("a_buf", m * out_a, dt).output())
            .buffer(BenchBuffer::zeros("b_buf", m * out_b, dt).output())
            .buffer(BenchBuffer::zeros("c_buf", m * out_c, dt).output())
            .buffer(BenchBuffer::zeros("d_buf", m * out_d, dt).output())
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", bs as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(max_rows as u32, m as u32, 4, [64, 1, 1])
            .bytes_moved(bytes as u64)
            // 4 fused qmms: 2 * m * (out_a + out_b + out_c + out_d) * in_dim
            .flops(2 * m as u64 * (out_a + out_b + out_c + out_d) as u64 * in_dim as u64)
            .with_shape_label(format!(
                "{} m={m} a={out_a} b={out_b} c={out_c} d={out_d} in={in_dim}",
                fmt.name()
            ))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxfp4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_nvfp4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxfp8_e4m3_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxfp8_e5m2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_nvfp8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_fp4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_nvfp8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_fp8_e5m2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int8,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int2,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int3_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int3,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int4,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int5_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int5,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int6_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int6,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxint2_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint2,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxint3_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint3,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxint4_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint4,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxint5_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint5,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxint6_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint6,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_mxint8_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Mxint8,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_nvfp8_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_nvfp8_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_fp4_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_fp8_e5m2_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int2_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int2F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int3_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int3F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int4_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int4F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int5_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int5F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int6_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int6F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_qmm_bench(
            mt_int8_f16_batched_4_qmm::kernel_ir_for(dt),
            QFormat::Int8F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            8,
            dt,
        )
    }
}

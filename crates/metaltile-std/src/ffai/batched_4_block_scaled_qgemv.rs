//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **batched 4-output block-scaled dequantizing GEMV** for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! One dispatch computes FOUR independent projections that share the same
//! single-token activation `x`: `out = [Wa·x, Wb·x, Wc·x, Wd·x]`. Mirrors the
//! int4 fused variant `ffai_batched_4_qgemv_fast`, but with the simple
//! one-output-row-per-TG Reduction geometry of the block-scaled Q/K/V variant
//! `mt_*_batched_qkv_qgemv` rather than the perf-tuned mask-without-shift path.
//!
//! Unlike the Q/K/V variant — which writes a single concatenated `output`
//! buffer — the four projections here write to FOUR SEPARATE output buffers
//! (`a_out`, `b_out`, `c_out`, `d_out`), each indexed directly by `row` (no
//! offset). Callers may alias all four into one backing allocation; the kernel
//! only sees four base pointers. This matches `ffai_batched_4_qgemv_fast`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [max(out_a,out_b,out_c,out_d), 1, 4]`,
//!   `tpg = [TPG, 1, 1]`, TPG >= 32 & a multiple of 32. One TG per
//!   (matrix, row); rows past a matrix's `out_*` no-op. The grid z-dimension
//!   `program_id::<2>()` selects the matrix (0=A, 1=B, 2=C, 3=D),
//!   `program_id::<0>()` the output row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_*, in_dim/8]` u32 (4-bit) or `[out_*, in_dim]` u8 (8-bit);
//!   scales `[out_*, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!   Each matrix writes its own `out_*`-length output buffer. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 batched 4-output GEMV — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_batched_4_qgemv<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u32>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u32>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u32>,
    scales_d: Tensor<u8>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
    if matrix == 0u32 {
        if row < out_a {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_a[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_a[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(a_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_b[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_b[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(b_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_c[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_c[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(c_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale = exp2(load(scales_d[row_block_off + blk]).cast::<f32>() - 127.0f32);
                    let packed = load(w_d[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(d_out[row], total.cast::<T>());
            }
        }
    }
}

/// nvfp4 batched 4-output GEMV — E2M1 (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_batched_4_qgemv<T>(
    x: Tensor<T>,
    w_a: Tensor<u32>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u32>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u32>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u32>,
    scales_d: Tensor<u8>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
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
    if matrix == 0u32 {
        if row < out_a {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_a[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_a[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(a_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_b[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_b[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(b_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_c[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_c[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(c_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let blk = pack_idx / packs_per_block;
                    let scale =
                        e4m3_decode(load(scales_d[row_block_off + blk]).cast::<u32>()) * global;
                    let packed = load(w_d[row_pack_off + pack_idx]);
                    let p_off = pack_idx * 8u32;
                    for i in range(0u32, 8u32, 1u32) {
                        let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                        acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(d_out[row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E4M3) batched 4-output GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_batched_4_qgemv<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u8>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u8>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u8>,
    scales_d: Tensor<u8>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    if matrix == 0u32 {
        if row < out_a {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_a[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_a[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(a_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_b[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_b[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(b_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_c[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_c[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(c_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_d[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_d[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(d_out[row], total.cast::<T>());
            }
        }
    }
}

/// mxfp8 (E5M2) batched 4-output GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_batched_4_qgemv<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<u8>,
    w_b: Tensor<u8>,
    scales_b: Tensor<u8>,
    w_c: Tensor<u8>,
    scales_c: Tensor<u8>,
    w_d: Tensor<u8>,
    scales_d: Tensor<u8>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    if matrix == 0u32 {
        if row < out_a {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_a[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_a[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(a_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_b[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_b[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(b_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_c[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_c[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(c_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e5m2_decode(load(w_d[row_off + c]).cast::<u32>());
                    let scale = exp2(
                        load(scales_d[row_block_off + c / block_size]).cast::<f32>() - 127.0f32,
                    );
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(d_out[row], total.cast::<T>());
            }
        }
    }
}

/// nvfp8 batched 4-output GEMV — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_batched_4_qgemv<T>(
    x: Tensor<T>,
    w_a: Tensor<u8>,
    scales_a: Tensor<f32>,
    w_b: Tensor<u8>,
    scales_b: Tensor<f32>,
    w_c: Tensor<u8>,
    scales_c: Tensor<f32>,
    w_d: Tensor<u8>,
    scales_d: Tensor<f32>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let iters = (in_dim + lsize - 1u32) / lsize;
    let row_off = row * in_dim;
    let row_block_off = row * n_blocks;
    if matrix == 0u32 {
        if row < out_a {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_a[row_off + c]).cast::<u32>());
                    let scale = load(scales_a[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(a_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_b[row_off + c]).cast::<u32>());
                    let scale = load(scales_b[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(b_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_c[row_off + c]).cast::<u32>());
                    let scale = load(scales_c[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(c_out[row], total.cast::<T>());
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            let mut acc = 0.0f32;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let elem = e4m3_decode(load(w_d[row_off + c]).cast::<u32>());
                    let scale = load(scales_d[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(d_out[row], total.cast::<T>());
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

    #[allow(clippy::too_many_arguments)]
    fn batched_4_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_a: usize,
        out_b: usize,
        out_c: usize,
        out_d: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
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
        let x_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        // Four SEPARATE expected outputs — each matrix writes its own buffer.
        let ea = gemv(&wdq_a, &x, out_a, in_dim);
        let eb = gemv(&wdq_b, &x, out_b, in_dim);
        let ec = gemv(&wdq_c, &x, out_c, in_dim);
        let ed = gemv(&wdq_d, &x, out_d, in_dim);
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
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
            .input(TestBuffer::zeros("a_out", out_a, dt))
            .input(TestBuffer::zeros("b_out", out_b, dt))
            .input(TestBuffer::zeros("c_out", out_c, dt))
            .input(TestBuffer::zeros("d_out", out_d, dt))
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", pa.global.max(pb.global).max(pc.global).max(pd.global));
        }
        s.expect(TestBuffer::from_vec("a_out", pack_f32(&ea, dt), dt))
            .expect(TestBuffer::from_vec("b_out", pack_f32(&eb, dt), dt))
            .expect(TestBuffer::from_vec("c_out", pack_f32(&ec, dt), dt))
            .expect(TestBuffer::from_vec("d_out", pack_f32(&ed, dt), dt))
            .grid_3d(max_rows as u32, 1, 4, [TPG, 1, 1])
    }

    // out_a 16, out_b/out_c/out_d 4, in_dim 256 (÷ 16/32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxfp8_e4m3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxfp8_e5m2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
}

/// Decode-shape benches at a fused 4-projection shape (out_a=out_b=out_c=out_d
/// =4096, in_dim=4096). Throughput is data-independent → random buffers.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn batched_4_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_a: usize,
        out_b: usize,
        out_c: usize,
        out_d: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let bs = fmt.block_size();
        let (codes_dt, code_div) =
            if fmt.element_bits() == 4 { (DType::U32, 8) } else { (DType::U8, 1) };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
        let max_rows = out_a.max(out_b).max(out_c).max(out_d);
        let sz = dt.size_bytes();
        let codes = |o: usize| o * in_dim / code_div;
        let scl = |o: usize| o * (in_dim / bs);
        let bytes = (codes(out_a) + codes(out_b) + codes(out_c) + codes(out_d))
            * codes_dt.size_bytes()
            + (scl(out_a) + scl(out_b) + scl(out_c) + scl(out_d)) * scales_dt.size_bytes()
            + in_dim * sz
            + (out_a + out_b + out_c + out_d) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("w_a", codes(out_a), codes_dt))
            .buffer(BenchBuffer::random("scales_a", scl(out_a), scales_dt))
            .buffer(BenchBuffer::random("w_b", codes(out_b), codes_dt))
            .buffer(BenchBuffer::random("scales_b", scl(out_b), scales_dt))
            .buffer(BenchBuffer::random("w_c", codes(out_c), codes_dt))
            .buffer(BenchBuffer::random("scales_c", scl(out_c), scales_dt))
            .buffer(BenchBuffer::random("w_d", codes(out_d), codes_dt))
            .buffer(BenchBuffer::random("scales_d", scl(out_d), scales_dt))
            .buffer(BenchBuffer::zeros("a_out", out_a, dt).output())
            .buffer(BenchBuffer::zeros("b_out", out_b, dt).output())
            .buffer(BenchBuffer::zeros("c_out", out_c, dt).output())
            .buffer(BenchBuffer::zeros("d_out", out_d, dt).output())
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", bs as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(max_rows as u32, 1, 4, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * (out_a + out_b + out_c + out_d) as u64 * in_dim as u64)
            .with_shape_label(format!(
                "{} a={out_a} b={out_b} c={out_c} d={out_d} in={in_dim}",
                fmt.name()
            ))
    }

    #[bench(name = "ffai/batched_4_block_qgemv/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/batched_4_block_qgemv/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/batched_4_block_qgemv/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxfp8_e4m3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/batched_4_block_qgemv/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxfp8_e5m2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/batched_4_block_qgemv/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **RMSNorm + block-scaled dequantizing GEMV** for decode (single-token),
//! for the spec-conformant float formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 /
//! nvfp8), the legacy float-scale fp4 / fp8 + symmetric int8, and the symmetric
//! integer family (int2/3/4/5/6 + MXINT2..6 + MXINT8).
//!
//! `y = qmatmul(rms_norm(x) · norm_weight, W_q)` in one dispatch — the int4
//! fusion of `ffai/rms_norm_qgemv.rs` (`ffai_rms_norm_qgemv`, the simple
//! one-row-per-TG variant) with the block-scaled weight decode of
//! `mlx/block_scaled_matmul.rs`. The normalized activation never leaves
//! registers between the RMSNorm reduce and the matvec.
//!
//! ## DISPATCH INVARIANTS (identical to the proven int4 + block-scaled GEMVs)
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [TPG, 1, 1]` with
//!   TPG ≥ 32 and a multiple of 32 (tests/benches use 64). One TG per row.
//! - Phase 1: per-thread Σx², `mt_rms_inv_scalar` does the TG reduce + rsqrt.
//! - Phase 2: the pack-/element-strided block-scaled GEMV of
//!   `block_scaled_matmul.rs`, feeding on `normed[i] = x[i]·norm_weight[i]·inv_rms`.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8,
//!   and `in_dim · bits` a multiple of 32 for the sub-byte int widths so each
//!   row's tight bit-stream is u32-word-aligned.
//! - weight `[out_dim, in_dim]` u8 (8-bit) or, for every sub-byte width
//!   (4-bit nibble packs + int2/3/5/6 tight bit-streams), `bitstream_words` u32
//!   words per the row's LSB-first bit-stream; scales
//!   `[out_dim, in_dim/block_size]` (u8 E8M0/E4M3, or f32 for nvfp8 / int*).
//!
//! Block-scaled formats carry **no bias** (the int affine scale+bias path lives
//! in `rms_norm_qgemv.rs`); the accumulation is `dequant(W)·normed`.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 fused RMSNorm + GEMV — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    // Phase 1: RMSNorm — per-thread Σx², TG reduce + rsqrt via cross-kernel call.
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
    // Phase 2: pack-strided E2M1 GEMV over the normalized activation.
    let n_packs_per_row = in_dim / 8u32; // 8 nibbles per u32
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let sbits = load(scales[row_block_off + blk]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = mt_decode_e2m1(nib);
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                acc = acc + (val * scale) * (xi * nw * inv_rms);
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// nvfp4 fused RMSNorm + GEMV — E2M1 weights (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let row = program_id::<0>();
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
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = mt_decode_e4m3(load(scales[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = mt_decode_e2m1(nib);
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                acc = acc + (val * scale) * (xi * nw * inv_rms);
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) fused RMSNorm + GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e4m3(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// mxfp8 (E5M2) fused RMSNorm + GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e5m2(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// nvfp8 fused RMSNorm + GEMV — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e4m3(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 fused RMSNorm GEMVs ─────
// These share the fused framework but carry a raw per-group FP32 scale (no
// E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 + f32
// scale), so it reuses `mt_nvfp8_rms_norm_qgemv` — only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode.

/// Legacy fp4 fused RMSNorm + GEMV — E2M1 weights (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp4_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = load(scales[row_block_off + blk]);
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = mt_decode_e2m1(nib);
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                acc = acc + (val * scale) * (xi * nw * inv_rms);
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Legacy fp8 (E5M2) fused RMSNorm + GEMV — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e5m2(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Symmetric int8 fused RMSNorm + GEMV — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_int8(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── Symmetric sub-byte integer fused RMSNorm GEMVs (int2/3/4/5/6 + MXINT2..6) ─
// The weight element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words (per-row word-aligned; element `c` at bit `c·bits`
// within the row's bit-stream). The decode mirrors `block_scaled_matmul.rs`'s
// proven `int_qgemv_*` macros exactly: extract the low N bits with a straddle-
// aware two-word read, sign-extend in float (subtract 2^N when the top bit is
// set; `$half`/`$full` are 2^(N-1) / 2^N), then multiply by the block scale.
// The shared RMSNorm prologue (Phase 1 Σx² + `mt_rms_inv_scalar`) is identical
// to the other fused kernels; only the Phase-2 weight decode + scale read change.
// Element-strided like `mt_int8_rms_norm_qgemv` — one TG per output row, threads
// stride over the row's elements, `reduce_sum` folds the partials. `$half`/`$full`
// are passed as literals to keep the constexpr math out of the DSL shift operands.
// The dispatch geometry is unchanged from the rest of the family (Reduction,
// `grid = [out_dim, 1, 1]`, `tpg = [64, 1, 1]`).

/// FP32-scaled symmetric int fused RMSNorm + GEMV (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP32 scale, dotted with the normalized activation
/// `x[c]·nw[c]·inv_rms`. `row_word_off` indexes the row's tight bit-stream
/// (`in_dim · bits / 32` u32 words per row).
macro_rules! int_rms_qgemv_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            norm_weight: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f32>,
            output: Tensor<T>,
            eps_buf: Tensor<f32>,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
            // Phase 1: RMSNorm — per-thread Σx², TG reduce + rsqrt (shared decode).
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
            // Phase 2: element-strided bit-stream int GEMV over the normalized x.
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;

            let mut acc = 0.0f32;
            let iters = (in_dim + lsize - 1u32) / lsize;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let bit_off = c * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[row_block_off + c / block_size]);
                    let xi = load(x[c]).cast::<f32>();
                    let nw = load(norm_weight[c]).cast::<f32>();
                    acc = acc + (val * scale) * (xi * nw * inv_rms);
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}
int_rms_qgemv_f32!(mt_int2_rms_norm_qgemv, 2u32, 2u32, 4.0f32);
int_rms_qgemv_f32!(mt_int3_rms_norm_qgemv, 3u32, 4u32, 8.0f32);
int_rms_qgemv_f32!(mt_int4_rms_norm_qgemv, 4u32, 8u32, 16.0f32);
int_rms_qgemv_f32!(mt_int5_rms_norm_qgemv, 5u32, 16u32, 32.0f32);
int_rms_qgemv_f32!(mt_int6_rms_norm_qgemv, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int fused RMSNorm + GEMV (MXINT2/3/4/5/6): per-element
/// bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`, dotted with the
/// normalized activation. Same straddle-aware decode and element-strided
/// reduction as `int_rms_qgemv_f32`; only the scale axis differs (one u8
/// exponent per block instead of a raw f32).
macro_rules! int_rms_qgemv_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            norm_weight: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<u8>,
            output: Tensor<T>,
            eps_buf: Tensor<f32>,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
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
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;

            let mut acc = 0.0f32;
            let iters = (in_dim + lsize - 1u32) / lsize;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let bit_off = c * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    let xi = load(x[c]).cast::<f32>();
                    let nw = load(norm_weight[c]).cast::<f32>();
                    acc = acc + (val * scale) * (xi * nw * inv_rms);
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}
int_rms_qgemv_e8m0!(mt_mxint2_rms_norm_qgemv, 2u32, 2u32, 4.0f32);
int_rms_qgemv_e8m0!(mt_mxint3_rms_norm_qgemv, 3u32, 4u32, 8.0f32);
int_rms_qgemv_e8m0!(mt_mxint4_rms_norm_qgemv, 4u32, 8u32, 16.0f32);
int_rms_qgemv_e8m0!(mt_mxint5_rms_norm_qgemv, 5u32, 16u32, 32.0f32);
int_rms_qgemv_e8m0!(mt_mxint6_rms_norm_qgemv, 6u32, 32u32, 64.0f32);

/// MXINT8 fused RMSNorm + GEMV — 8-bit symmetric codes (byte layout, block 32),
/// E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the 8-bit float
/// formats (one byte per code); decode is `mt_decode_int8 → val · scale`.
#[kernel]
pub fn mt_mxint8_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_int8(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── FP16-scale twins of the FP32-scaled fused RMSNorm GEMVs ─────────────────
// Each kernel below is a verbatim clone of its FP32-scaled twin above, changing
// ONLY the scale axis: `scales` binds as `Tensor<f16>` (native `half`) and the
// per-block scale read becomes `load(...).cast::<f32>()`. The element decode
// (E2M1 / E4M3 / E5M2 / sub-byte int bit-stream + sign-extend), weight indexing,
// shared RMSNorm prologue, reduction geometry, and grid/tpg are all IDENTICAL to
// the twin — only the scale storage type differs. The half load matches the host
// `f16_scale_decode`, so the dequant-then-dot oracle still holds exactly. The
// GPU-verified decode + f16 scale read mirror `mlx/block_scaled_dequant.rs`'s
// `mt_*_f16_dequant` family. `fp8_e4m3_f16` reuses `mt_nvfp8_f16_rms_norm_qgemv`
// (same 8-bit-E4M3 + scale shape), exactly as `fp8_e4m3` reuses the nvfp8 kernel.

/// nvfp8 (FP16 scale) fused RMSNorm + GEMV — E4M3 weights (block 16), per-block
/// FP16 scale. Twin of `mt_nvfp8_rms_norm_qgemv`; also serves `fp8_e4m3_f16`.
#[kernel]
pub fn mt_nvfp8_f16_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e4m3(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// fp4 (FP16 scale) fused RMSNorm + GEMV — E2M1 weights (group 32), per-group
/// FP16 scale. Twin of `mt_fp4_rms_norm_qgemv`.
#[kernel]
pub fn mt_fp4_f16_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = load(scales[row_block_off + blk]).cast::<f32>();
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = mt_decode_e2m1(nib);
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                acc = acc + (val * scale) * (xi * nw * inv_rms);
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// fp8 (E5M2, FP16 scale) fused RMSNorm + GEMV — 8-bit weights (group 32),
/// per-group FP16 scale. Twin of `mt_fp8_e5m2_rms_norm_qgemv`.
#[kernel]
pub fn mt_fp8_e5m2_f16_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e5m2(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// FP16-scaled symmetric int fused RMSNorm + GEMV (int2/3/4/5/6): identical
/// straddle-aware bit-stream decode + element-strided reduction as
/// `int_rms_qgemv_f32`; only the scale axis differs (native `half` per group,
/// read as `load(scales[...]).cast::<f32>()`).
macro_rules! int_rms_qgemv_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            norm_weight: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f16>,
            output: Tensor<T>,
            eps_buf: Tensor<f32>,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
            // Phase 1: RMSNorm — per-thread Σx², TG reduce + rsqrt (shared decode).
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
            // Phase 2: element-strided bit-stream int GEMV over the normalized x.
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let row_word_off = row * words_per_row;
            let row_block_off = row * n_blocks;

            let mut acc = 0.0f32;
            let iters = (in_dim + lsize - 1u32) / lsize;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let bit_off = c * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
                    let xi = load(x[c]).cast::<f32>();
                    let nw = load(norm_weight[c]).cast::<f32>();
                    acc = acc + (val * scale) * (xi * nw * inv_rms);
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}
int_rms_qgemv_f16!(mt_int2_f16_rms_norm_qgemv, 2u32, 2u32, 4.0f32);
int_rms_qgemv_f16!(mt_int3_f16_rms_norm_qgemv, 3u32, 4u32, 8.0f32);
int_rms_qgemv_f16!(mt_int4_f16_rms_norm_qgemv, 4u32, 8u32, 16.0f32);
int_rms_qgemv_f16!(mt_int5_f16_rms_norm_qgemv, 5u32, 16u32, 32.0f32);
int_rms_qgemv_f16!(mt_int6_f16_rms_norm_qgemv, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) fused RMSNorm + GEMV — 8-bit symmetric codes (byte layout,
/// group 64), per-group FP16 scale. Twin of `mt_int8_rms_norm_qgemv`.
#[kernel]
pub fn mt_int8_f16_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_int8(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let xi = load(x[c]).cast::<f32>();
            let nw = load(norm_weight[c]).cast::<f32>();
            acc = acc + (elem * scale) * (xi * nw * inv_rms);
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// One TG-row's lanes; ≥ 32 and a multiple of 32 (the Reduction contract).
    const TPG: u32 = 64;
    const EPS: f32 = 1e-5;

    /// Deterministic `[out_dim, in_dim]` weights with mixed signs + per-block
    /// magnitude variation (same generator as `block_scaled_matmul.rs`).
    fn weights(out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + r * 0.25) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// Fused RMSNorm + dequant-dot reference:
    /// `inv_rms = 1/√(mean(x²)+eps)`, `out[r] = Σ_c W_dq[r,c]·(x[c]·nw[c]·inv_rms)`.
    fn rms_qgemv_oracle(
        wdq: &[f32],
        x: &[f32],
        nw: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let ssq: f32 = x.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / (ssq / in_dim as f32 + EPS).sqrt();
        (0..out_dim)
            .map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * (x[c] * nw[c] * inv_rms)).sum())
            .collect()
    }

    fn rms_qgemv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(out_dim, in_dim);
        let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
        // Round x / norm_weight through `dt` so the oracle sees what the GPU sees.
        let x_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.05 + 0.1).collect();
        let nw_f: Vec<f32> = (0..in_dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let nw = unpack_f32(&pack_f32(&nw_f, dt), dt);
        let expected = rms_qgemv_oracle(&wdq, &x, &nw, out_dim, in_dim);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as native half; E8M0/E4M3 scales as one
        // byte. Both axes are driven off the format so new integer / fp16 formats
        // pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&nw_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .input(TestBuffer::from_vec("eps_buf", EPS.to_le_bytes().to_vec(), DType::F32))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            out_dim as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // out_dim 4, in_dim 256 (divisible by both block sizes 16 / 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_nvfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_mxfp8_e4m3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_mxfp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // in_dim 256 is a multiple of int8's block_size 64 (= 4 × 64).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_fp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_fp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each row's bit-stream is
    // word-aligned. The kernel and oracle share the codec, so the fused output
    // tracks the RMSNorm + dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int3_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int5_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int6_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint3_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint5_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint6_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint8, 4, 256, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode in
    // their own clone. Same dims (in_dim 256) satisfy every block / bit-stream
    // alignment, and the codec round-trip keeps the fused output on the oracle.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_fp4_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_fp8_e5m2_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int2_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int3_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int4_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int5_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int6_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int8_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8F16, 4, 256, dt)
    }
}

/// Decode-shape (single-token) benches at the canonical hidden = out = 4096 so
/// the GFLOP/s + roofline columns rank precisions side by side. Throughput is
/// data-independent, so packed weight/scale buffers are random bytes.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn rms_qgemv_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = out_dim * (in_dim / fmt.block_size());
        let n = out_dim * in_dim;
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + in_dim * sz   // x
            + in_dim * sz   // norm_weight
            + out_dim * sz; // output
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", in_dim, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .buffer(BenchBuffer::random("eps_buf", 1, DType::F32))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64) // GEMV (B=1): 2·N·K
            .with_shape_label(format!("{} m={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_mxfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_nvfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxfp8_e4m3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxfp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_fp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_fp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int3_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int5_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int6_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint4_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint5_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint6_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint8_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            4096,
            4096,
            dt,
        )
    }
    // FP16-scale twins (fp8_e4m3_f16 reuses the nvfp8_f16 kernel).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_fp4_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_fp8_e5m2_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int2_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int3_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int4_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int5_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int6_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            4096,
            4096,
            dt,
        )
    }
}

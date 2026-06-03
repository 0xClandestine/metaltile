//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **dequantizing GEMV** kernels (Phase B of the precision
//! roadmap, `specs/BENCH_METRICS_SPEC.md` Appendix B): `output[row] =
//! Σ_k dequant(weight[row, k]) · input[k]` for the spec-conformant formats.
//!
//! The dispatch geometry is the **proven pack-strided reduction** from
//! `ffai/dequant_gemv.rs` — one threadgroup per output row, threads stride over
//! the row's packed words, `reduce_sum` folds the partials. Only the per-element
//! *decode* differs (block-scaled E2M1/E4M3/… instead of int-affine), so no new
//! dispatch shape is introduced (and the reduction freeze hazard — TPG ≥ 32 &
//! multiple of 32 — is handled exactly as the int kernels handle it).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [TPG, 1, 1]` with
//!   TPG ≥ 32 and a multiple of 32 (tests/benches use 64). One TG per row.
//! - `in_dim` a multiple of `block_size`; `block_size` a multiple of 8 (so a
//!   u32 pack of 8 nibbles lies wholly inside one block — one scale load/pack).
//! - weight `[out_dim, in_dim/8]` u32 (8 E2M1 nibbles/word, little-endian);
//!   scales `[out_dim, in_dim/block_size]` u8 (E8M0); input `[in_dim]`,
//!   output `[out_dim]`.

use metaltile::kernel;

/// mxfp4 dequantizing GEMV — E2M1 weights (block 32) with an E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_qgemv<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
            // All 8 nibbles of a pack lie in one block → one scale load.
            let blk = pack_idx / packs_per_block;
            let sbits = load(scales[row_block_off + blk]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// nvfp4 dequantizing GEMV — E2M1 weights (block 16), E4M3 micro-scale ×
/// a global FP32. Pack-strided like mxfp4 (block 16 ⇒ 2 packs/block).
#[kernel]
pub fn mt_nvfp4_qgemv<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let row = program_id::<0>();
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
            // E4M3 micro-scale × global.
            let scale = e4m3_decode(load(scales[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) dequantizing GEMV — 8-bit weights (block 32), E8M0 pow-2 scale.
/// Element-strided: one byte per code, so threads stride over elements.
#[kernel]
pub fn mt_mxfp8_e4m3_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// mxfp8 (E5M2) dequantizing GEMV — 8-bit weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// nvfp8 dequantizing GEMV — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 GEMVs ──────────────────
// These share the block-scaled framework but store a raw per-group FP32 scale
// (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 +
// f32 scale), so it reuses `mt_nvfp8_qgemv` — only fp4 (4-bit E2M1), fp8_e5m2
// (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 dequantizing GEMV — E2M1 weights (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp4_qgemv<T>(
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
            let scale = load(scales[row_block_off + pack_idx / packs_per_block]);
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Legacy fp8 (E5M2) dequantizing GEMV — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Symmetric int8 dequantizing GEMV — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── Symmetric sub-byte integer GEMVs (int2/3/4/5/6 + MXINT2..6) ─────────────
// The element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words (per-row word-aligned; element `c` at bit `c·bits`
// within the row's bit-stream). Decode mirrors `block_scaled_dequant`'s proven
// `int_dequant_*` macros exactly: extract the low N bits with a straddle-aware
// two-word read, sign-extend in float (subtract 2^N when the top bit is set;
// `$half`/`$full` are 2^(N-1) / 2^N), then multiply by the block scale and the
// matching input. Element-strided like `mt_int8_qgemv` — one TG per output row,
// threads stride over the row's elements, `reduce_sum` folds the partials.
// `$half`/`$full` are passed as literals to keep the constexpr math out of the
// DSL shift operands. The dispatch geometry is unchanged from the rest of the
// family (Reduction, `grid = [out_dim, 1, 1]`, `tpg = [64, 1, 1]`).

/// FP32-scaled symmetric int GEMV (int2/3/4/5/6): per-element bit-stream code
/// × per-group FP32 scale, dotted with the input. `row_word_off` indexes the
/// row's tight bit-stream (`in_dim · bits / 32` u32 words per row).
macro_rules! int_qgemv_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<f32>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
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
                    acc = acc + (val * scale) * load(input[c]).cast::<f32>();
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}
int_qgemv_f32!(mt_int2_qgemv, 2u32, 2u32, 4.0f32);
int_qgemv_f32!(mt_int3_qgemv, 3u32, 4u32, 8.0f32);
int_qgemv_f32!(mt_int4_qgemv, 4u32, 8u32, 16.0f32);
int_qgemv_f32!(mt_int5_qgemv, 5u32, 16u32, 32.0f32);
int_qgemv_f32!(mt_int6_qgemv, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int GEMV (MXINT2/3/4/5/6): per-element bit-stream code
/// × pow-2 (E8M0) block scale `2^(bits-127)`, dotted with the input. Same
/// straddle-aware decode and element-strided reduction as `int_qgemv_f32`; only
/// the scale axis differs (one u8 exponent per block instead of a raw f32).
macro_rules! int_qgemv_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<u8>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
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
                    acc = acc + (val * scale) * load(input[c]).cast::<f32>();
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}
int_qgemv_e8m0!(mt_mxint2_qgemv, 2u32, 2u32, 4.0f32);
int_qgemv_e8m0!(mt_mxint3_qgemv, 3u32, 4u32, 8.0f32);
int_qgemv_e8m0!(mt_mxint4_qgemv, 4u32, 8u32, 16.0f32);
int_qgemv_e8m0!(mt_mxint5_qgemv, 5u32, 16u32, 32.0f32);
int_qgemv_e8m0!(mt_mxint6_qgemv, 6u32, 32u32, 64.0f32);

/// MXINT8 dequantizing GEMV — 8-bit symmetric codes (byte layout, block 32),
/// E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the 8-bit float
/// formats (one byte per code), decode is `int8_decode → val · scale`.
#[kernel]
pub fn mt_mxint8_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── FP16-scale twins of the FP32-scaled formats ─────────────────────────────
// Identical element decode + reduction geometry as their FP32 twins above; the
// only change is the per-block scale is stored as native `half` (`Tensor<f16>`)
// and cast to f32 on load (matching the host `f16_scale_decode`, so the oracle
// still holds exactly). `mt_nvfp8_f16_qgemv` serves both `nvfp8_f16` and
// `fp8_e4m3_f16` (same 8-bit-E4M3 + scale shape), exactly as `mt_nvfp8_qgemv`
// serves `fp8_e4m3` today. No new dispatch shape is introduced.

/// nvfp8 (FP16-scale) dequantizing GEMV — E4M3 weights (block 16), per-block
/// FP16 scale. Clone of `mt_nvfp8_qgemv` with the scale tensor in `half`. Also
/// serves `fp8_e4m3_f16` (same 8-bit-E4M3 + scale shape).
#[kernel]
pub fn mt_nvfp8_f16_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// fp4 (FP16-scale) dequantizing GEMV — E2M1 weights (group 32), per-group
/// FP16 scale. Clone of `mt_fp4_qgemv` with the scale tensor in `half`.
#[kernel]
pub fn mt_fp4_f16_qgemv<T>(
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
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
            let scale = load(scales[row_block_off + pack_idx / packs_per_block]).cast::<f32>();
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let val = e2m1_decode((packed >> (i * 4u32)) & 0xFu32);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// fp8 (E5M2, FP16-scale) dequantizing GEMV — 8-bit weights (group 32), FP16
/// scale. Clone of `mt_fp8_e5m2_qgemv` with the scale tensor in `half`.
#[kernel]
pub fn mt_fp8_e5m2_f16_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// FP16-scaled symmetric int GEMV (int2/3/4/5/6): identical straddle-aware
/// bit-stream decode + element-strided reduction as `int_qgemv_f32`; the only
/// change is the per-group scale is stored as `half` and cast to f32 on load.
macro_rules! int_qgemv_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<f16>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
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
                    acc = acc + (val * scale) * load(input[c]).cast::<f32>();
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }
    };
}
int_qgemv_f16!(mt_int2_f16_qgemv, 2u32, 2u32, 4.0f32);
int_qgemv_f16!(mt_int3_f16_qgemv, 3u32, 4u32, 8.0f32);
int_qgemv_f16!(mt_int4_f16_qgemv, 4u32, 8u32, 16.0f32);
int_qgemv_f16!(mt_int5_f16_qgemv, 5u32, 16u32, 32.0f32);
int_qgemv_f16!(mt_int6_f16_qgemv, 6u32, 32u32, 64.0f32);

/// int8 (FP16-scale) dequantizing GEMV — 8-bit symmetric codes (byte layout,
/// group 64), per-group FP16 scale. Clone of `mt_int8_qgemv` with the scale
/// tensor in `half`. Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_f16_qgemv<T>(
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
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

    /// Deterministic `[out_dim, in_dim]` weights with mixed signs + per-block
    /// magnitude variation.
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

    /// Dequant-then-dot reference: `out[r] = Σ_c dequant(W)[r,c] · input[c]`.
    fn qgemv_oracle(wdq: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim).map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * input[c]).sum()).collect()
    }

    fn qgemv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(out_dim, in_dim);
        let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        // Round-trip the input through `dt` so the oracle sees what the GPU sees.
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = qgemv_oracle(&wdq, &x, out_dim, in_dim);
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
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
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

    // out_dim 4, in_dim 256 (divisible by both block sizes) — mirrors the int
    // dequant_gemv test shape.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxfp4_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_nvfp4_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxfp8_e4m3_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxfp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E5, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_fp4_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_fp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int8_qgemv::kernel_ir_for(dt), QFormat::Int8, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each row's bit-stream is
    // word-aligned. The kernel and oracle share the codec, so the GPU output
    // tracks the dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int2_qgemv::kernel_ir_for(dt), QFormat::Int2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int3_qgemv::kernel_ir_for(dt), QFormat::Int3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int4_qgemv::kernel_ir_for(dt), QFormat::Int4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int5_qgemv::kernel_ir_for(dt), QFormat::Int5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int6_qgemv::kernel_ir_for(dt), QFormat::Int6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxint2_qgemv::kernel_ir_for(dt), QFormat::Mxint2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxint3_qgemv::kernel_ir_for(dt), QFormat::Mxint3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxint4_qgemv::kernel_ir_for(dt), QFormat::Mxint4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxint5_qgemv::kernel_ir_for(dt), QFormat::Mxint5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxint6_qgemv::kernel_ir_for(dt), QFormat::Mxint6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_mxint8_qgemv::kernel_ir_for(dt), QFormat::Mxint8, 4, 256, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Nvfp8F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_fp4_f16_qgemv::kernel_ir_for(dt), QFormat::Fp4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_fp8_e5m2_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int2_f16_qgemv::kernel_ir_for(dt), QFormat::Int2F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int3_f16_qgemv::kernel_ir_for(dt), QFormat::Int3F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int4_f16_qgemv::kernel_ir_for(dt), QFormat::Int4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int5_f16_qgemv::kernel_ir_for(dt), QFormat::Int5F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int6_f16_qgemv::kernel_ir_for(dt), QFormat::Int6F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(mt_int8_f16_qgemv::kernel_ir_for(dt), QFormat::Int8F16, 4, 256, dt)
    }
}

/// Decode-shape (single-token GEMV) benches at the canonical N=K=4096 so the
/// GFLOP/s + roofline columns rank the precisions side by side (the spec's
/// "which precision is fastest" goal). Throughput is data-independent, so the
/// packed weight/scale buffers are random bytes.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn qgemv_bench(
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
            + in_dim * sz
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
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
    fn bench_mxfp4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxfp4_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_nvfp4_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxfp8_e4m3_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxfp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_fp4_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_fp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int8_qgemv::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int2_qgemv::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int3_qgemv::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int4_qgemv::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int5_qgemv::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int6_qgemv::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxint2_qgemv::kernel_ir_for(dt), QFormat::Mxint2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxint3_qgemv::kernel_ir_for(dt), QFormat::Mxint3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxint4_qgemv::kernel_ir_for(dt), QFormat::Mxint4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxint5_qgemv::kernel_ir_for(dt), QFormat::Mxint5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxint6_qgemv::kernel_ir_for(dt), QFormat::Mxint6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_mxint8_qgemv::kernel_ir_for(dt), QFormat::Mxint8, 4096, 4096, dt)
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Nvfp8F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_fp4_f16_qgemv::kernel_ir_for(dt), QFormat::Fp4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_fp8_e5m2_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int2_f16_qgemv::kernel_ir_for(dt), QFormat::Int2F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int3_f16_qgemv::kernel_ir_for(dt), QFormat::Int3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int4_f16_qgemv::kernel_ir_for(dt), QFormat::Int4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int5_f16_qgemv::kernel_ir_for(dt), QFormat::Int5F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int6_f16_qgemv::kernel_ir_for(dt), QFormat::Int6F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(mt_int8_f16_qgemv::kernel_ir_for(dt), QFormat::Int8F16, 4096, 4096, dt)
    }
}

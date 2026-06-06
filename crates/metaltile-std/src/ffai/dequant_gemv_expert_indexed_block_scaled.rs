//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Per-expert-indexed **block-scaled / legacy-fp / int8 dequantizing GEMV**.
//!
//! Block-scaled counterpart of `ffai/dequant_gemv_expert_indexed.rs` (which
//! handles the int4-affine case). For the eight non-int4 quantization formats
//! — mxfp4 / nvfp4 / mxfp8_e4m3 / mxfp8_e5m2 / nvfp8 + legacy fp4 / fp8_e5m2 /
//! int8 — the weight + scale tensors are **stacked across experts** and the
//! kernel reads which expert to index from a GPU-resident
//! `expert_index: Tensor<u32>` at runtime.
//!
//! Each kernel body is the `mlx/block_scaled_matmul.rs` qgemv for that format
//! (same one-TG-per-output-row pack-/element-strided reduction), with two extra
//! per-row offsets — exactly like the int4 expert-indexed kernel:
//!
//!   weight_expert_off = expert · out_dim · n_packs_per_row   (4-bit)
//!   weight_expert_off = expert · out_dim · in_dim            (8-bit)
//!   scale_expert_off  = expert · out_dim · n_blocks
//!
//! computed from `expert_index[0]` loaded once per threadgroup, then folded
//! into the row pack/element/block base offsets. There is no int affine bias —
//! block-scaled / fp / symmetric-int8 carry a scale only.
//!
//! `fp8_e4m3` is **not** a separate kernel: its layout (8-bit E4M3 codes + a
//! per-group FP32 scale) is identical to `nvfp8`, so the `fp8_e4m3` test + bench
//! dispatch `mt_nvfp8_dequant_gemv_expert_indexed` with `QFormat::Fp8E4m3`.
//!
//! ## Memory layout
//!
//! For `n_experts` experts each a `[out_dim, in_dim]` block-scaled slab:
//!
//!   weights_stacked  [n_experts, out_dim, in_dim/8]  u32   (4-bit, 8 nibbles/word)
//!   weights_stacked  [n_experts, out_dim, in_dim]    u8    (8-bit, 1 code/byte)
//!   scales_stacked   [n_experts, out_dim, in_dim/B]  u8|f32 (E8M0/E4M3 byte, or FP32)
//!   input            [in_dim]                         T
//!   expert_index     [1]                              u32
//!   output           [out_dim]                        T
//!
//! ## Dispatch
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [TPG, 1, 1]` with
//!   TPG ≥ 32 and a multiple of 32 (tests/benches use 64). One TG per row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.

use metaltile::kernel;

/// mxfp4 expert-indexed dequantizing GEMV — E2M1 weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp4_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<u8>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 8u32; // 8 nibbles per u32
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    // expert_index[0] ∈ [0, n_experts): stride the row bases by the per-expert span.
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            // All 8 nibbles of a pack lie in one block → one scale load.
            let blk = pack_idx / packs_per_block;
            let sbits = load(scales_stacked[row_block_off + blk]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = mt_decode_e2m1(nib);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// nvfp4 expert-indexed dequantizing GEMV — E2M1 weights (block 16), E4M3
/// micro-scale × a global FP32. Pack-strided like mxfp4 (block 16 ⇒ 2 packs/block).
#[kernel]
pub fn mt_nvfp4_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<u8>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            // E4M3 micro-scale × global.
            let scale =
                mt_decode_e4m3(load(scales_stacked[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = mt_decode_e2m1(nib);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) expert-indexed dequantizing GEMV — 8-bit weights (block 32),
/// E8M0 pow-2 scale. Element-strided: one byte per code.
#[kernel]
pub fn mt_mxfp8_e4m3_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<u8>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e4m3(load(weights_stacked[row_off + c]).cast::<u32>());
            let sbits = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// mxfp8 (E5M2) expert-indexed dequantizing GEMV — 8-bit weights (block 32),
/// E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<u8>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e5m2(load(weights_stacked[row_off + c]).cast::<u32>());
            let sbits = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// nvfp8 expert-indexed dequantizing GEMV — E4M3 weights (block 16), per-block
/// FP32 scale. Also serves the legacy `fp8_e4m3` format (identical layout).
#[kernel]
pub fn mt_nvfp8_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<f32>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e4m3(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 expert-indexed GEMVs ─────
// These share the framework but store a raw per-group FP32 scale (no E8M0/E4M3/
// global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 + f32 scale), so it
// reuses `mt_nvfp8_dequant_gemv_expert_indexed`; only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 expert-indexed dequantizing GEMV — E2M1 weights (group 32),
/// per-group FP32 scale.
#[kernel]
pub fn mt_fp4_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<f32>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let scale = load(scales_stacked[row_block_off + pack_idx / packs_per_block]);
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Legacy fp8 (E5M2) expert-indexed dequantizing GEMV — 8-bit weights
/// (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<f32>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e5m2(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Symmetric int8 expert-indexed dequantizing GEMV — 8-bit codes (group 64),
/// per-group FP32 scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<f32>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_int8(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── Symmetric sub-byte integer expert-indexed GEMVs (int2/3/4/5/6 + MXINT2..6 +
//    MXINT8) ───────────────────────────────────────────────────────────────────
// Mirror `mlx/block_scaled_matmul.rs`'s `int_qgemv_f32!` / `int_qgemv_e8m0!` /
// `mt_mxint8_qgemv` element decode, but fold in the per-expert row bases exactly
// like the int8 kernel above. The element is a signed N-bit two's-complement code
// tight-bit-packed LSB-first into u32 words. The whole `[n_experts·out_dim, in_dim]`
// stack is one bit-stream, so for global row `g_row = expert·out_dim + row` the
// row's word base is `g_row · (in_dim · bits / 32)` (per-row word-aligned because
// `in_dim` is a multiple of 32 for every width). The scale base folds the expert
// stride the same way the float kernels do: `(expert·out_dim + row)·n_blocks`.
// Decode = straddle-aware two-word read + float sign-extend (subtract 2^N when the
// top bit is set; `$half`/`$full` are 2^(N-1) / 2^N), then × block scale × input.
// Element-strided like `mt_int8_dequant_gemv_expert_indexed`. The dispatch geometry
// is unchanged from the rest of the family (Reduction, `grid = [out_dim, 1, 1]`,
// `tpg = [64, 1, 1]`).

/// FP32-scaled symmetric int expert-indexed GEMV (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP32 scale, dotted with the input. The row's tight
/// bit-stream word base folds the expert stride via `g_row = expert·out_dim + row`.
macro_rules! int_qgemv_f32_expert_indexed {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weights_stacked: Tensor<u32>,
            scales_stacked: Tensor<f32>,
            input: Tensor<T>,
            expert_index: Tensor<u32>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            // g_row = expert·out_dim + row → fold the expert stride into both bases.
            let expert = load(expert_index[0u32]);
            let g_row = expert * out_dim + row;
            let row_word_off = g_row * words_per_row;
            let row_block_off = g_row * n_blocks;

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
                    let w0 = load(weights_stacked[row_word_off + word_idx]);
                    let w1 = load(
                        weights_stacked
                            [row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales_stacked[row_block_off + c / block_size]);
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
int_qgemv_f32_expert_indexed!(mt_int2_dequant_gemv_expert_indexed, 2u32, 2u32, 4.0f32);
int_qgemv_f32_expert_indexed!(mt_int3_dequant_gemv_expert_indexed, 3u32, 4u32, 8.0f32);
int_qgemv_f32_expert_indexed!(mt_int4_dequant_gemv_expert_indexed, 4u32, 8u32, 16.0f32);
int_qgemv_f32_expert_indexed!(mt_int5_dequant_gemv_expert_indexed, 5u32, 16u32, 32.0f32);
int_qgemv_f32_expert_indexed!(mt_int6_dequant_gemv_expert_indexed, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int expert-indexed GEMV (MXINT2/3/4/5/6): per-element
/// bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`, dotted with the
/// input. Same straddle-aware decode + expert-folded bases as the FP32 variant;
/// only the scale axis differs (one u8 exponent per block instead of a raw f32).
macro_rules! int_qgemv_e8m0_expert_indexed {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weights_stacked: Tensor<u32>,
            scales_stacked: Tensor<u8>,
            input: Tensor<T>,
            expert_index: Tensor<u32>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            let expert = load(expert_index[0u32]);
            let g_row = expert * out_dim + row;
            let row_word_off = g_row * words_per_row;
            let row_block_off = g_row * n_blocks;

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
                    let w0 = load(weights_stacked[row_word_off + word_idx]);
                    let w1 = load(
                        weights_stacked
                            [row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let sbits = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
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
int_qgemv_e8m0_expert_indexed!(mt_mxint2_dequant_gemv_expert_indexed, 2u32, 2u32, 4.0f32);
int_qgemv_e8m0_expert_indexed!(mt_mxint3_dequant_gemv_expert_indexed, 3u32, 4u32, 8.0f32);
int_qgemv_e8m0_expert_indexed!(mt_mxint4_dequant_gemv_expert_indexed, 4u32, 8u32, 16.0f32);
int_qgemv_e8m0_expert_indexed!(mt_mxint5_dequant_gemv_expert_indexed, 5u32, 16u32, 32.0f32);
int_qgemv_e8m0_expert_indexed!(mt_mxint6_dequant_gemv_expert_indexed, 6u32, 32u32, 64.0f32);

/// MXINT8 expert-indexed dequantizing GEMV — 8-bit symmetric codes (byte layout,
/// block 32), E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the
/// 8-bit float formats (one byte per code), decode is `mt_decode_int8 → val · scale`,
/// with the per-expert row bases folded in exactly like the int8 kernel.
#[kernel]
pub fn mt_mxint8_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<u8>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_int8(load(weights_stacked[row_off + c]).cast::<u32>());
            let sbits = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// ── FP16-scale twins (nvfp8_f16 / fp4_f16 / fp8_e5m2_f16 + int2..int6/int8 f16) ─
// Near-clones of the FP32-scaled kernels above for the same element layout; only
// the scale tensor changes from `Tensor<f32>` to `Tensor<f16>` and the scale read
// gains a `.cast::<f32>()`. Element decode (E2M1 / E4M3 / E5M2 / int bit-stream +
// sign-extend), weight indexing, the per-expert row bases, and the dispatch
// geometry are IDENTICAL to the FP32 twin. The GPU-verified scale-read pattern is
// `mlx/block_scaled_dequant.rs` (`mt_nvfp8_f16_dequant` et al.): native half load
// then cast to f32. `fp8_e4m3_f16` reuses `mt_nvfp8_f16_dequant_gemv_expert_indexed`
// (same 8-bit E4M3 + f16-scale shape), exactly as `fp8_e4m3` reuses the nvfp8 kernel.

/// nvfp8 (FP16-scale) expert-indexed dequantizing GEMV — E4M3 weights (block 16),
/// per-block FP16 scale. Also serves the `fp8_e4m3_f16` format (identical layout).
/// Clone of `mt_nvfp8_dequant_gemv_expert_indexed` with the scale as half.
#[kernel]
pub fn mt_nvfp8_f16_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<f16>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e4m3(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// fp4 (FP16-scale) expert-indexed dequantizing GEMV — E2M1 weights (group 32),
/// per-group FP16 scale. Clone of `mt_fp4_dequant_gemv_expert_indexed`, scale → half.
#[kernel]
pub fn mt_fp4_f16_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<f16>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let scale =
                load(scales_stacked[row_block_off + pack_idx / packs_per_block]).cast::<f32>();
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// fp8 (E5M2, FP16-scale) expert-indexed dequantizing GEMV — 8-bit weights
/// (group 32), per-group FP16 scale. Clone of
/// `mt_fp8_e5m2_dequant_gemv_expert_indexed`, scale → half.
#[kernel]
pub fn mt_fp8_e5m2_f16_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<f16>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_e5m2(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// FP16-scaled symmetric int expert-indexed GEMV (int2/3/4/5/6 F16): identical
/// straddle-aware bit-stream decode + expert-folded row bases as
/// `int_qgemv_f32_expert_indexed!`; only the scale tensor is half (read with a
/// trailing `.cast::<f32>()`).
macro_rules! int_qgemv_f16_expert_indexed {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weights_stacked: Tensor<u32>,
            scales_stacked: Tensor<f16>,
            input: Tensor<T>,
            expert_index: Tensor<u32>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let row = program_id::<0>();
            let words_per_row = in_dim * $bits / 32u32;
            let n_blocks = in_dim / block_size;
            // g_row = expert·out_dim + row → fold the expert stride into both bases.
            let expert = load(expert_index[0u32]);
            let g_row = expert * out_dim + row;
            let row_word_off = g_row * words_per_row;
            let row_block_off = g_row * n_blocks;

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
                    let w0 = load(weights_stacked[row_word_off + word_idx]);
                    let w1 = load(
                        weights_stacked
                            [row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
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
int_qgemv_f16_expert_indexed!(mt_int2_f16_dequant_gemv_expert_indexed, 2u32, 2u32, 4.0f32);
int_qgemv_f16_expert_indexed!(mt_int3_f16_dequant_gemv_expert_indexed, 3u32, 4u32, 8.0f32);
int_qgemv_f16_expert_indexed!(mt_int4_f16_dequant_gemv_expert_indexed, 4u32, 8u32, 16.0f32);
int_qgemv_f16_expert_indexed!(mt_int5_f16_dequant_gemv_expert_indexed, 5u32, 16u32, 32.0f32);
int_qgemv_f16_expert_indexed!(mt_int6_f16_dequant_gemv_expert_indexed, 6u32, 32u32, 64.0f32);

/// int8 (FP16-scale) expert-indexed dequantizing GEMV — 8-bit symmetric codes
/// (byte layout, group 64), per-group FP16 scale. Clone of
/// `mt_int8_dequant_gemv_expert_indexed`, scale → half.
#[kernel]
pub fn mt_int8_f16_dequant_gemv_expert_indexed<T>(
    weights_stacked: Tensor<u8>,
    scales_stacked: Tensor<f16>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * in_dim;
    let scale_expert_off = expert * out_dim * n_blocks;
    let row_off = weight_expert_off + row * in_dim;
    let row_block_off = scale_expert_off + row * n_blocks;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = mt_decode_int8(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(input[c]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// Correctness tests for the per-expert-indexed block-scaled dequant GEMVs.
///
/// Oracle: build the FULL `[n_experts·out_dim, in_dim]` stacked weight and pack
/// it **once** via `crate::quant::format::pack` (so the resulting `codes`/`scales`
/// buffers are a single contiguous, correctly-aligned bit-stream — concatenating
/// per-expert packs would misalign experts after the first for the straddling
/// sub-byte widths 3/5/6, which append a guard word). Pick a non-zero expert,
/// dequant the full stack, slice the selected expert's `[out_dim, in_dim]` rows,
/// and replay `out[row] = Σ_i wdq[row,i]·x[i]` in f32. Verifies the expert-stride
/// offset math on both the weight + scale row bases. Inputs are dtype-rounded so
/// the GPU sees exactly what the oracle does.
///
/// Grid: `grid_3d(out_dim, 1, 1, [TPG, 1, 1])` — one TG per output row, TPG = 64.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// 64 lanes per output row (≥ 32, multiple of 32 — Reduction contract).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` weights for expert `e` — mixed signs +
    /// per-expert + per-row + along-K magnitude variation so the per-block scale
    /// (and the expert stride) are genuinely exercised.
    fn weights(e: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + e as f32 * 0.3 + r * 0.25) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// Dequant-then-dot reference for the selected expert's dequantized slab.
    fn oracle(wdq: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim).map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * input[c]).sum()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn expert_setup(
        kernel: Kernel,
        fmt: QFormat,
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        expert: usize,
        dt: DType,
    ) -> TestSetup {
        // Build the FULL `[n_experts·out_dim, in_dim]` stacked weight and pack it
        // ONCE — `pack` then produces a single contiguous, per-row word-aligned
        // bit-stream for `codes` (and a contiguous `scales` axis). Packing each
        // expert independently and concatenating would misalign experts after the
        // first for the straddling sub-byte widths (3/5/6), which append a guard
        // word; a single stacked pack is byte-identical for the 4/8-bit formats
        // (no regression) and correct for every width. `p.codes`/`p.scales` bind
        // directly with no per-expert concat.
        let stacked_rows = n_experts * out_dim;
        let mut stacked_w: Vec<f32> = Vec::with_capacity(stacked_rows * in_dim);
        for e in 0..n_experts {
            stacked_w.extend_from_slice(&weights(e, out_dim, in_dim));
        }
        let p = crate::quant::format::pack(fmt, &stacked_w, stacked_rows, in_dim);
        let sel_global = p.global;
        // Dequant the full stack, then slice the selected expert's row band for
        // the oracle (rows `[expert·out_dim, (expert+1)·out_dim)`).
        let wdq_all = crate::quant::format::dequant(fmt, &p, stacked_rows, in_dim);
        let wdq = &wdq_all[expert * out_dim * in_dim..(expert + 1) * out_dim * in_dim];

        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        // Round-trip the input through `dt` so the oracle sees what the GPU sees.
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = oracle(wdq, &x, out_dim, in_dim);

        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weights_stacked", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_stacked", p.scales, scales_dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("expert_index", u32_bytes(&[expert as u32]), DType::U32))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", sel_global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            out_dim as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // n_experts 4, out_dim 4, in_dim 256 (divisible by every block/group size —
    // 16/32/64), expert 2 to exercise a non-zero expert stride.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxfp8_e4m3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxfp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode in their own.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // int8 group is 64 → in_dim 256 = 4×64 divides evenly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width (and divides every group/block
    // size), so the single stacked pack's per-row bit-stream is word-aligned and
    // the kernel + oracle share the codec to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // ── FP16-scale twins ────────────────────────────────────────────────────
    // Same geometry as the FP32-scaled formats; only the scale axis is half.
    // `fp8_e4m3_f16` dispatches the nvfp8_f16 kernel (identical 8-bit-E4M3 +
    // f16-scale shape), exactly as `fp8_e4m3` reuses the nvfp8 kernel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp8_e5m2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int3_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int5_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int6_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
}

/// Decode-shape benches: per-expert-indexed dequant GEMV over an 8-expert stack
/// at the canonical out_dim=in_dim=4096 so the GFLOP/s + roofline columns rank
/// the precisions side by side. Active stream = one expert's slab + its scales +
/// input + output. One TG per output row. Throughput is data-independent, so the
/// packed weight/scale buffers are random bytes.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn expert_bench(
        kernel: Kernel,
        fmt: QFormat,
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let blocks_per_expert = out_dim * (in_dim / fmt.block_size());
        // The whole stack is one bit-stream packed once, so the `weights_stacked`
        // buffer is `bitstream_words(n_experts·out_dim·in_dim, bits)` u32 words for
        // every sub-byte width (4-bit collapses to the old `total/8`), or one byte
        // per code at 8-bit. Per-expert lengths drive only the active-stream byte
        // accounting (one expert's slab is read per dispatch). Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let codes_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let total_codes = if fmt.element_bits() == 8 {
            n_experts * out_dim * in_dim
        } else {
            crate::quant::format::bitstream_words(n_experts * out_dim * in_dim, fmt.element_bits())
        };
        let codes_per_expert = if fmt.element_bits() == 8 {
            out_dim * in_dim
        } else {
            crate::quant::format::bitstream_words(out_dim * in_dim, fmt.element_bits())
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        // Active stream: one expert's weight slab + its scales + input + output.
        let bytes = codes_per_expert * codes_dt.size_bytes()
            + blocks_per_expert * scales_dt.size_bytes()
            + in_dim * sz
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weights_stacked", total_codes, codes_dt))
            .buffer(BenchBuffer::random("scales_stacked", n_experts * blocks_per_expert, scales_dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("expert_index", 1, DType::U32))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64) // qgemv expert-indexed (B=1): 2·N·K
            .with_shape_label(format!("{} m={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxfp8_e4m3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxfp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8,
            8,
            4096,
            4096,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint2,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint3,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint5,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint6,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint8,
            8,
            4096,
            4096,
            dt,
        )
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3
    // + f16-scale shape); the rest decode in their own per-element kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp8_e5m2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int3_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int5_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int6_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8F16,
            8,
            4096,
            4096,
            dt,
        )
    }
}

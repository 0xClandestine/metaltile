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
                e4m3_decode(load(scales_stacked[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
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
            let elem = e4m3_decode(load(weights_stacked[row_off + c]).cast::<u32>());
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
            let elem = e5m2_decode(load(weights_stacked[row_off + c]).cast::<u32>());
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
            let elem = e4m3_decode(load(weights_stacked[row_off + c]).cast::<u32>());
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
            let elem = e5m2_decode(load(weights_stacked[row_off + c]).cast::<u32>());
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
            let elem = int8_decode(load(weights_stacked[row_off + c]).cast::<u32>());
            let scale = load(scales_stacked[row_block_off + c / block_size]);
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
/// Oracle: stack `n_experts` block-scaled `[out_dim, in_dim]` weight slabs
/// (each packed independently via `crate::quant::format::pack`, then their
/// `codes`/`scales` byte buffers concatenated in expert order), pick a non-zero
/// expert, dequant **only** the selected expert's slab via
/// `crate::quant::format::dequant`, and replay `out[row] = Σ_i wdq[row,i]·x[i]`
/// in f32. Verifies the expert-stride offset math on both the weight + scale
/// row bases. Inputs are dtype-rounded so the GPU sees exactly what the oracle does.
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
        // Pack each expert's slab independently, then concatenate the per-expert
        // code + scale byte buffers in expert order (each slab's bytes already
        // carry the right per-slab layout — 4-bit codes are 8 nibbles/u32-word,
        // 8-bit codes 1/byte; scales are E8M0/E4M3 bytes or LE f32). The selected
        // expert's `PackedTensor` is dequantized for the oracle.
        let mut codes_stacked: Vec<u8> = Vec::new();
        let mut scales_stacked: Vec<u8> = Vec::new();
        let mut sel_packed = None;
        let mut sel_global = 1.0f32;
        for e in 0..n_experts {
            let w = weights(e, out_dim, in_dim);
            let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
            codes_stacked.extend_from_slice(&p.codes);
            scales_stacked.extend_from_slice(&p.scales);
            if e == expert {
                sel_global = p.global;
                sel_packed = Some(p);
            }
        }
        let p_sel = sel_packed.expect("expert index in range");
        let wdq = crate::quant::format::dequant(fmt, &p_sel, out_dim, in_dim);

        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        // Round-trip the input through `dt` so the oracle sees what the GPU sees.
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = oracle(&wdq, &x, out_dim, in_dim);

        // 4-bit codes bind as packed u32; 8-bit as one uchar each. FP32 (nvfp8 /
        // legacy fp / int8) scales bind as f32; all others are one byte (E8M0/E4M3).
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weights_stacked", codes_stacked, weight_dt))
            .input(TestBuffer::from_vec("scales_stacked", scales_stacked, scales_dt))
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
        let (codes_per_expert, codes_dt) = if fmt.element_bits() == 4 {
            (out_dim * in_dim / 8, DType::U32)
        } else {
            (out_dim * in_dim, DType::U8)
        };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let sz = dt.size_bytes();
        // Active stream: one expert's weight slab + its scales + input + output.
        let bytes = codes_per_expert * codes_dt.size_bytes()
            + blocks_per_expert * scales_dt.size_bytes()
            + in_dim * sz
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weights_stacked", n_experts * codes_per_expert, codes_dt))
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

    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/mxfp4", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/nvfp4", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/nvfp8", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/fp4", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/fp8_e4m3", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/fp8_e5m2", dtypes = [f32, f16, bf16])]
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
    #[bench(name = "ffai/dequant_gemv_expert_indexed_block/int8", dtypes = [f32, f16, bf16])]
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
}

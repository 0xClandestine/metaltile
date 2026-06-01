//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **MoE gather-GEMM** kernels — per-token expert-routed matmul:
//! `output[m, n] = Σ_k dequant(weight[expert_ids[m], n, k]) · x[m, k]` for the
//! spec-conformant formats (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! Identical to [`super::block_scaled_qmm`] except the weight row is selected by
//! the per-token expert id: the expert stack is one `[E·out_dim, in_dim]` packed
//! tensor, so row `expert_ids[m]·out_dim + n` addresses expert `e`'s output row
//! `n`. Packing the whole stack in one call keeps nvfp4's single global FP32
//! valid across experts (no per-expert scale bookkeeping).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim·m_rows, 1, 1]`, `tpg = [TPG, 1, 1]`
//!   (TPG ≥ 32 & multiple of 32) — same as qmm; only the weight/scale row offset
//!   gains the `expert·out_dim` term.
//! - `weight` is the `[E·out_dim, …]` packed stack; `scales` likewise; layouts +
//!   the `block_size | 8` rule match the GEMV/GEMM kernels. `expert_ids` is
//!   `[m_rows]` u32, `x` is `[m_rows, in_dim]`, `output` is `[m_rows, out_dim]`.

use metaltile::kernel;

/// mxfp4 MoE gather-GEMM — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_gather_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    expert_ids: Tensor<u32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let wrow = load(expert_ids[mr]) * out_dim + n;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = wrow * n_packs_per_row;
    let row_block_off = wrow * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let sbits = load(scales[row_block_off + blk]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// nvfp4 MoE gather-GEMM — E2M1 weights (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_gather_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    expert_ids: Tensor<u32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let wrow = load(expert_ids[mr]) * out_dim + n;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = wrow * n_packs_per_row;
    let row_block_off = wrow * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = e4m3_decode(load(scales[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) MoE gather-GEMM — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_gather_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    expert_ids: Tensor<u32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let wrow = load(expert_ids[mr]) * out_dim + n;
    let row_off = wrow * in_dim;
    let row_block_off = wrow * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// mxfp8 (E5M2) MoE gather-GEMM — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_gather_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    expert_ids: Tensor<u32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let wrow = load(expert_ids[mr]) * out_dim + n;
    let row_off = wrow * in_dim;
    let row_block_off = wrow * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// nvfp8 MoE gather-GEMM — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_gather_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    expert_ids: Tensor<u32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let wrow = load(expert_ids[mr]) * out_dim + n;
    let row_off = wrow * in_dim;
    let row_block_off = wrow * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
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

    /// Deterministic `[E·out_dim, in_dim]` expert-stacked weights.
    fn weights(stack_rows: usize, in_dim: usize) -> Vec<f32> {
        (0..stack_rows * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.4 + (r % 7.0) * 0.2) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// `out[m, n] = Σ_k dequant(W)[expert_ids[m]·out_dim + n, k] · x[m, k]`.
    #[allow(clippy::too_many_arguments)]
    fn gather_oracle(
        wdq: &[f32],
        x: &[f32],
        eids: &[u32],
        m_rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m_rows * out_dim];
        for mr in 0..m_rows {
            let base = eids[mr] as usize * out_dim;
            for n in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += wdq[(base + n) * in_dim + k] * x[mr * in_dim + k];
                }
                out[mr * out_dim + n] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_setup(
        kernel: Kernel,
        fmt: QFormat,
        n_experts: usize,
        m_rows: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let stack_rows = n_experts * out_dim;
        let w = weights(stack_rows, in_dim);
        let p = crate::quant::format::pack(fmt, &w, stack_rows, in_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, stack_rows, in_dim);
        // Deterministic per-token expert routing.
        let eids: Vec<u32> = (0..m_rows).map(|m| (m * 2 + 1) as u32 % n_experts as u32).collect();
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = gather_oracle(&wdq, &x, &eids, m_rows, in_dim, out_dim);
        let eid_bytes: Vec<u8> = eids.iter().flat_map(|e| e.to_le_bytes()).collect();
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("expert_ids", eid_bytes, DType::U32))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("output", m_rows * out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            (out_dim * m_rows) as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // 4 experts, 3 routed tokens, out_dim 4, in_dim 256.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(mt_mxfp4_gather_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(mt_nvfp4_gather_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            mt_mxfp8_e4m3_gather_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            3,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            mt_mxfp8_e5m2_gather_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            3,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(mt_nvfp8_gather_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 4, 3, 4, 256, dt)
    }
}

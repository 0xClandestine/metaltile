//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **dequantizing GEMM** (multi-row matmul) kernels — the qmm
//! counterpart of the GEMVs in [`super::block_scaled_matmul`]:
//! `output[m, n] = Σ_k dequant(weight[n, k]) · x[m, k]` for the spec-conformant
//! formats (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! Each `(m, n)` output element is one threadgroup that reduces over K — the
//! same proven Reduction geometry as the GEMVs, just flattened into a 1-D grid
//! of `out_dim · m_rows` threadgroups so it depends only on `program_id::<0>()`
//! (no 2-D grid assumptions). `tg → (mr = tg / out_dim, n = tg − mr·out_dim)`,
//! and `output[tg]` is exactly `output[mr·out_dim + n]`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim·m_rows, 1, 1]`, `tpg = [TPG, 1, 1]`
//!   with TPG ≥ 32 & a multiple of 32. One TG per output element.
//! - Weight/scale layouts + the `block_size | 8` packing rule are identical to
//!   the GEMVs (see [`super::block_scaled_matmul`]). `x` is `[m_rows, in_dim]`,
//!   `output` is `[m_rows, out_dim]`, both row-major.

use metaltile::kernel;

/// mxfp4 dequantizing GEMM — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = n * n_packs_per_row;
    let row_block_off = n * (in_dim / block_size);
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
                let m = nib & 0x7u32;
                let mag = select(
                    m < 1u32,
                    0.0f32,
                    select(
                        m < 2u32,
                        0.5f32,
                        select(
                            m < 3u32,
                            1.0f32,
                            select(
                                m < 4u32,
                                1.5f32,
                                select(
                                    m < 5u32,
                                    2.0f32,
                                    select(m < 6u32, 3.0f32, select(m < 7u32, 4.0f32, 6.0f32)),
                                ),
                            ),
                        ),
                    ),
                );
                let val = select((nib & 0x8u32) > 0u32, -mag, mag);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// nvfp4 dequantizing GEMM — E2M1 weights (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
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
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = n * n_packs_per_row;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let sb = load(scales[row_block_off + blk]).cast::<u32>();
            let se = (sb >> 3u32) & 0xFu32;
            let sm = sb & 0x7u32;
            let smag = select(
                se < 1u32,
                sm.cast::<f32>() * 0.001953125f32,
                (1.0f32 + sm.cast::<f32>() * 0.125f32) * exp2(se.cast::<f32>() - 7.0f32),
            );
            let scale = select((sb >> 7u32) > 0u32, -smag, smag) * global;
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let m = nib & 0x7u32;
                let mag = select(
                    m < 1u32,
                    0.0f32,
                    select(
                        m < 2u32,
                        0.5f32,
                        select(
                            m < 3u32,
                            1.0f32,
                            select(
                                m < 4u32,
                                1.5f32,
                                select(
                                    m < 5u32,
                                    2.0f32,
                                    select(m < 6u32, 3.0f32, select(m < 7u32, 4.0f32, 6.0f32)),
                                ),
                            ),
                        ),
                    ),
                );
                let val = select((nib & 0x8u32) > 0u32, -mag, mag);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) dequantizing GEMM — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let bits = load(weight[row_off + c]).cast::<u32>();
            let exp = (bits >> 3u32) & 0xFu32;
            let mant = bits & 0x7u32;
            let mag = select(
                exp < 1u32,
                mant.cast::<f32>() * 0.001953125f32,
                (1.0f32 + mant.cast::<f32>() * 0.125f32) * exp2(exp.cast::<f32>() - 7.0f32),
            );
            let elem = select((bits >> 7u32) > 0u32, -mag, mag);
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

/// mxfp8 (E5M2) dequantizing GEMM — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let bits = load(weight[row_off + c]).cast::<u32>();
            let exp = (bits >> 2u32) & 0x1Fu32;
            let mant = bits & 0x3u32;
            let mag = select(
                exp < 1u32,
                mant.cast::<f32>() * 0.0000152587890625f32,
                (1.0f32 + mant.cast::<f32>() * 0.25f32) * exp2(exp.cast::<f32>() - 15.0f32),
            );
            let elem = select((bits >> 7u32) > 0u32, -mag, mag);
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

/// nvfp8 dequantizing GEMM — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let bits = load(weight[row_off + c]).cast::<u32>();
            let exp = (bits >> 3u32) & 0xFu32;
            let mant = bits & 0x7u32;
            let mag = select(
                exp < 1u32,
                mant.cast::<f32>() * 0.001953125f32,
                (1.0f32 + mant.cast::<f32>() * 0.125f32) * exp2(exp.cast::<f32>() - 7.0f32),
            );
            let elem = select((bits >> 7u32) > 0u32, -mag, mag);
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

    /// Reduction-contract threadgroup width (≥ 32, multiple of 32).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` quantized weights (mixed signs).
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

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`.
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

    fn qmm_setup(
        kernel: Kernel,
        fmt: QFormat,
        m_rows: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(out_dim, in_dim);
        let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = qmm_oracle(&wdq, &x, m_rows, in_dim, out_dim);
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
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

    // m_rows 3, out_dim 4, in_dim 256 (divisible by both block sizes).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp4_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp4_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp8_e4m3_qmm::kernel_ir_for(dt), QFormat::Mxfp8E4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Mxfp8E5, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 3, 4, 256, dt)
    }
}

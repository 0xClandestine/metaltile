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

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 GEMMs ──────────────────
// These share the block-scaled framework but store a raw per-group FP32 scale
// (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 +
// f32 scale), so it reuses `mt_nvfp8_qmm` — only fp4 (4-bit E2M1), fp8_e5m2
// (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 dequantizing GEMM — E2M1 weights (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp4_qmm<T>(
    weight: Tensor<u32>,
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
            let scale = load(scales[row_block_off + blk]);
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

/// Legacy fp8 (E5M2) dequantizing GEMM — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_qmm<T>(
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
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// Symmetric int8 dequantizing GEMM — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_qmm<T>(
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
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
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

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // in_dim 256 is a multiple of int8's block_size (64), so all formats fit.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp4_qmm::kernel_ir_for(dt), QFormat::Fp4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int8_qmm::kernel_ir_for(dt), QFormat::Int8, 3, 4, 256, dt)
    }
}

/// Batched-decode (m=32) GEMM benches at N=K=4096 — the compute-throughput
/// precision ranking. Random packed buffers (throughput is data-independent).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn qmm_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = out_dim * (in_dim / fmt.block_size());
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
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
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m * in_dim * sz
            + m * out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::zeros("output", m * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_dim * m) as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m as u64 * out_dim as u64 * in_dim as u64) // GEMM: 2·M·N·K
            .with_shape_label(format!("{} m={m} n={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(name = "ffai/block_scaled_qmm/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp4_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp4_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp8_e4m3_qmm::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp4_qmm::kernel_ir_for(dt), QFormat::Fp4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int8_qmm::kernel_ir_for(dt), QFormat::Int8, 32, 4096, 4096, dt)
    }
}

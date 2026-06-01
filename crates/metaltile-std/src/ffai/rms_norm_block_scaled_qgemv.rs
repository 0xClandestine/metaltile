//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **RMSNorm + block-scaled dequantizing GEMV** for decode (single-token),
//! for the spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
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
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_dim, in_dim/8]` u32 (4-bit) or `[out_dim, in_dim]` u8 (8-bit);
//!   scales `[out_dim, in_dim/block_size]` (u8 E8M0/E4M3, or f32 for nvfp8).
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
                let val = e2m1_decode(nib);
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
            let scale = e4m3_decode(load(scales[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
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
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
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
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
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
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
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
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
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
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (out_dim * in_dim / 8, DType::U32)
        } else {
            (out_dim * in_dim, DType::U8)
        };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
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

    #[bench(name = "ffai/rms_norm_block_qgemv/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_mxfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(name = "ffai/rms_norm_block_qgemv/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_nvfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(name = "ffai/rms_norm_block_qgemv/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxfp8_e4m3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/rms_norm_block_qgemv/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxfp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/rms_norm_block_qgemv/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
}

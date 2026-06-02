//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight** 1D convolution — the weight-quantized
//! counterpart of `ffai/audio_conv1d.rs` (the STT audio patch-embedding conv).
//!
//! The dense conv projects every output element over the `in_ch × k` receptive
//! field of an NCL input with an OIK filter `[out_ch, in_ch, k]`. That filter is
//! a genuine quantizable parameter, so we flatten its contraction axis to
//! `C = in_ch * k` and quantize each output channel row `[out_ch, C]` block-wise
//! along `C` in the spec formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8 + legacy
//! fp4/fp8 + symmetric int8).
//!
//! Filter-tap mapping: dense `w_idx = (oc * in_ch + ic) * k + kx = oc*C + col`
//! with `col = ic*k + kx`. So the dense filter load is replaced by a decode of
//! the packed code at logical `(row = oc, col = ic*k + kx)`:
//!   * 4-bit: `weight` is `[out_ch, C/8]` u32 — nibble at word `oc*(C/8)+col/8`,
//!     shift `(col%8)*4`.
//!   * 8-bit: `weight` is `[out_ch, C]` u8 — byte at `oc*C+col`.
//!
//! The decoded element is scaled by `scales[oc*(C/block_size) + col/block_size]`.
//!
//! Only the filter is quantized; the per-channel `bias` stays `T` (tiny and
//! precision-sensitive). Geometry, stride/pad guards and accumulation match the
//! dense kernel **verbatim**: **Grid3D**, one thread per output element
//! (`program_id::<0>()` = flat `(n, oc, op)`); indices stay in the *padded*
//! frame so every value is a non-negative u32 and padding taps mask to zero.
//! `C = in_ch*k` is a multiple of `block_size` (4-bit `block_size` a multiple of
//! 8). fp8_e4m3 reuses the nvfp8 kernel. Codegen-only; correctness pinned by the
//! in-source `#[test_kernel]`s vs a `quant::format::dequant` oracle.

use metaltile::kernel;

/// mxfp4 quantized-weight conv1d — E2M1 filter (block 32), E8M0 pow-2 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let w_packs_per_row = c_dim / 8u32;
    let n_blocks = c_dim / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp4 quantized-weight conv1d — E2M1 filter (block 16), E4M3 micro-scale × global.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let w_packs_per_row = c_dim / 8u32;
    let n_blocks = c_dim / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale =
                e4m3_decode(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global;
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp4 quantized-weight conv1d — E2M1 filter (group 32), per-group FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let w_packs_per_row = c_dim / 8u32;
    let n_blocks = c_dim / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E4M3) quantized-weight conv1d — 8-bit filter (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let n_blocks = c_dim / block_size;
    let w_row = oc * c_dim;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E5M2) quantized-weight conv1d — 8-bit filter (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let n_blocks = c_dim / block_size;
    let w_row = oc * c_dim;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp8 (E5M2) quantized-weight conv1d — 8-bit filter (group 32), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let n_blocks = c_dim / block_size;
    let w_row = oc * c_dim;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp8 quantized-weight conv1d — E4M3 filter (block 16), per-block FP32 scale.
/// Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let n_blocks = c_dim / block_size;
    let w_row = oc * c_dim;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Symmetric int8 quantized-weight conv1d — 8-bit codes (group 64), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let n_blocks = c_dim / block_size;
    let w_row = oc * c_dim;
    let w_row_blk = oc * n_blocks;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 1D conv oracle (NCL input, OIK filter) over a *dequantized* filter.
    /// Padding taps zero. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv1d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            let in_idx = (n * in_ch + ic) * in_len + ix;
                            // Quantized filter flattens [out_ch, in_ch, k] to
                            // [out_ch, C] with C = in_ch*k, col = ic*k + kx.
                            let col = ic * k + kx;
                            let w_idx = oc * (in_ch * k) + col;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * out_ch * out_len;
        // Flatten the filter contraction to C = in_ch*k and quantize [out_ch, C].
        let c_dim = in_ch * k;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let weight_f = ramp(out_ch * c_dim, 11, 4.0);
        let p = crate::quant::format::pack(fmt, &weight_f, out_ch, c_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, c_dim);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected =
            naive_conv1d(&input, &wdq, &bias, batch, in_ch, in_len, out_ch, k, stride, pad);
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
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // in_ch=8, k=8 → C=64 (÷ 16/32/64); out_ch=8, in_len=32, stride 1, pad 1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxfp4_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxfp4,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp4_audio_conv1d::kernel_ir_for(dt),
            QFormat::Nvfp4,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(mt_fp4_audio_conv1d::kernel_ir_for(dt), QFormat::Fp4, 1, 8, 32, 8, 8, 1, 1, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxfp8_e4m3_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxfp8_e5m2_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp8_e5m2_audio_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_audio_conv1d::kernel_ir_for(dt),
            QFormat::Nvfp8,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_audio_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int8_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int8,
            1,
            8,
            32,
            8,
            8,
            1,
            1,
            dt,
        )
    }
}

/// Decode-shape benches: realistic STT stem conv (in_ch=128, out_ch=128, k=8 →
/// C=1024 divisible by all block sizes; in_len=1024, stride 2, pad 1). Grid3D,
/// one thread per output element.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn conv1d_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let c_dim = in_ch * k;
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (out_ch * c_dim / 8, DType::U32)
        } else {
            (out_ch * c_dim, DType::U8)
        };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let n_blocks = out_ch * (c_dim / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + batch * in_ch * in_len * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_len, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            // 2 * Co * Lo * C (groups=1, C = in_ch*k)
            .flops(2 * out_ch as u64 * out_len as u64 * c_dim as u64)
            .with_shape_label(format!("{} oc={out_ch} lo={out_len} c={c_dim}", fmt.name()))
    }

    macro_rules! conv1d_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr, $name:literal) => {
            #[bench(name = $name, dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                conv1d_bench($kernel(dt), $fmt, 1, 128, 1024, 128, 8, 2, 1, dt)
            }
        };
    }
    conv1d_bench_fmt!(
        bench_mxfp4,
        mt_mxfp4_audio_conv1d::kernel_ir_for,
        QFormat::Mxfp4,
        "ffai/audio_conv1d_block/mxfp4"
    );
    conv1d_bench_fmt!(
        bench_nvfp4,
        mt_nvfp4_audio_conv1d::kernel_ir_for,
        QFormat::Nvfp4,
        "ffai/audio_conv1d_block/nvfp4"
    );
    conv1d_bench_fmt!(
        bench_fp4,
        mt_fp4_audio_conv1d::kernel_ir_for,
        QFormat::Fp4,
        "ffai/audio_conv1d_block/fp4"
    );
    conv1d_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_audio_conv1d::kernel_ir_for,
        QFormat::Mxfp8E4,
        "ffai/audio_conv1d_block/mxfp8_e4m3"
    );
    conv1d_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_audio_conv1d::kernel_ir_for,
        QFormat::Mxfp8E5,
        "ffai/audio_conv1d_block/mxfp8_e5m2"
    );
    conv1d_bench_fmt!(
        bench_fp8_e5m2,
        mt_fp8_e5m2_audio_conv1d::kernel_ir_for,
        QFormat::Fp8E5m2,
        "ffai/audio_conv1d_block/fp8_e5m2"
    );
    conv1d_bench_fmt!(
        bench_nvfp8,
        mt_nvfp8_audio_conv1d::kernel_ir_for,
        QFormat::Nvfp8,
        "ffai/audio_conv1d_block/nvfp8"
    );
    conv1d_bench_fmt!(
        bench_int8,
        mt_int8_audio_conv1d::kernel_ir_for,
        QFormat::Int8,
        "ffai/audio_conv1d_block/int8"
    );
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight depthwise 2D convolution** — the
//! weight-quantized counterpart of `ffai/depthwise_conv2d.rs`.
//!
//! Depthwise conv applies, per channel `c`, a single `k × k` filter to that
//! channel's own input plane (no cross-channel mixing). The filter is a genuine
//! quantizable parameter: the per-channel `[ch, k, k]` weight squeezes to a
//! `[ch, C]` matrix with `C = k*k`, block-scaled along the `C` contraction in
//! the spec formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8 + legacy fp4/fp8 +
//! symmetric int8). For channel `c`, tap `(ky, kx)` maps to `col = ky*k + kx`;
//! the packed code at `(row = c, col)` decodes against
//! `scales[c*(C/block_size) + col/block_size]`.
//!
//! Only the weight is quantized — the input plane and the per-channel `bias`
//! stay `T` (the bias is tiny and precision-sensitive). Geometry / loops / grid
//! / padding / dilation / stride are **identical** to the dense
//! `depthwise_conv2d`: **Grid3D**, one thread per output element
//! (`program_id::<0>()` = flat `(n, c, oh, ow)`), `grid_1d(n_out, 256)`. The
//! per-tap weight decode reuses the DSL decode intrinsics. fp8_e4m3 reuses the
//! nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
//!
//! ## Block-size vs. real depthwise filters
//!
//! `C = k*k` must be a multiple of `block_size` (16 / 32 / 64; 4-bit packs 8
//! codes per `u32`). A real `3 × 3` depthwise filter (`C = 9`) is far smaller
//! than any spec block, so it would need a sub-block-size group or padding. For
//! this matrix-coverage kernel we use **`k = 8 → C = 64`** in the tests so all
//! nine formats fit the generic codec; the kernel body itself is general over
//! `k`. Codegen-only; correctness pinned by the in-source `#[test_kernel]`s vs a
//! `quant::format::dequant` oracle.
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with `grid_1d(n_out, 256)`.
//! `out_h` / `out_w` must match `(in + 2*pad - dilation*(k-1) - 1)/stride + 1`
//! for the given `(k, stride, pad, dilation)`, and `bias` must have `ch`
//! elements. Weight is `[ch, C]` (4-bit: `[ch, C/8]` u32; 8-bit: `[ch, C]` u8),
//! `C = k*k` a multiple of `block_size`.

use metaltile::kernel;

/// mxfp4 quantized depthwise conv2d — E2M1 weight (block 32), E8M0 pow-2 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let w_packs_per_row = cols / 8u32;
    let n_blocks = cols / block_size;
    let w_row_pack = c * w_packs_per_row;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp4 quantized depthwise conv2d — E2M1 weight (block 16), E4M3 micro-scale × global.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let w_packs_per_row = cols / 8u32;
    let n_blocks = cols / block_size;
    let w_row_pack = c * w_packs_per_row;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale =
                e4m3_decode(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global;
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp4 quantized depthwise conv2d — E2M1 weight (group 32), per-group FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let w_packs_per_row = cols / 8u32;
    let n_blocks = cols / block_size;
    let w_row_pack = c * w_packs_per_row;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E4M3) quantized depthwise conv2d — 8-bit weight (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let n_blocks = cols / block_size;
    let w_row = c * cols;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E5M2) quantized depthwise conv2d — 8-bit weight (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let n_blocks = cols / block_size;
    let w_row = c * cols;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp8 (E5M2) quantized depthwise conv2d — 8-bit weight (group 32), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let n_blocks = cols / block_size;
    let w_row = c * cols;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp8 quantized depthwise conv2d — E4M3 weight (block 16), per-block FP32 scale.
/// Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let n_blocks = cols / block_size;
    let w_row = c * cols;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
            let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Symmetric int8 quantized depthwise conv2d — 8-bit codes (group 64), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let c = t2 % ch;
    let n = t2 / ch;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let in_c_base = (n * ch + c) * in_h * in_w;
    let cols = k * k;
    let n_blocks = cols / block_size;
    let w_row = c * cols;
    let w_row_blk = c * n_blocks;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let x = load(input[in_c_base + ih * in_w + iw]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = ky * k + kx;
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

    fn out_dim(in_d: usize, k: usize, stride: usize, pad: usize, dilation: usize) -> usize {
        (in_d + 2 * pad - dilation * (k - 1) - 1) / stride + 1
    }

    #[allow(clippy::too_many_arguments)]
    fn dw_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = out_dim(in_h, k, stride, pad, dilation);
        let out_w = out_dim(in_w, k, stride, pad, dilation);
        let n_out = batch * ch * out_h * out_w;
        // Filter squeezed to [ch, C] with C = k*k, block-scaled along C.
        let cols = k * k;
        let input_f = ramp(batch * ch * in_h * in_w, 13, 6.0);
        let bias_f = ramp(ch, 5, 2.0);
        let w_f = ramp(ch * cols, 11, 4.0);
        let p = crate::quant::format::pack(fmt, &w_f, ch, cols);
        let wdq = crate::quant::format::dequant(fmt, &p, ch, cols);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        // Oracle: dense depthwise math over the dequantized [ch, C] filter,
        // filter tap (ky, kx) → col = ky*k + kx (row = c).
        let mut expected = vec![0.0f32; n_out];
        for n in 0..batch {
            for c in 0..ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[c];
                        for ky in 0..k {
                            let ph = oh * stride + ky * dilation;
                            if ph < pad || ph >= pad + in_h {
                                continue;
                            }
                            let ih = ph - pad;
                            for kx in 0..k {
                                let pw = ow * stride + kx * dilation;
                                if pw < pad || pw >= pad + in_w {
                                    continue;
                                }
                                let iw = pw - pad;
                                let in_idx = ((n * ch + c) * in_h + ih) * in_w + iw;
                                let col = ky * k + kx;
                                acc += input[in_idx] * wdq[c * cols + col];
                            }
                        }
                        expected[((n * ch + c) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
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
            .constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // ch=8, k=8 → C=64 (÷ 16/32/64); 16×16 input, stride 1, pad 0.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxfp4_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxfp4,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_nvfp4_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Nvfp4,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_fp4_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Fp4,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxfp8_e4m3_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxfp8_e5m2_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_fp8_e5m2_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_nvfp8_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Nvfp8,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_nvfp8_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int8_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int8,
            1,
            8,
            16,
            16,
            8,
            1,
            0,
            1,
            dt,
        )
    }
}

/// Decode-shape benches: realistic depthwise stage (256 channels, 64×64 feature
/// map, k=8 → C=64 quantized filter, stride 1, pad 0). Grid3D, one thread per
/// output element.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn dw_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * ch * out_h * out_w;
        let cols = k * k;
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (ch * cols / 8, DType::U32)
        } else {
            (ch * cols, DType::U8)
        };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let n_blocks = ch * (cols / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + batch * ch * in_h * in_w * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            // 2 * n_out * C (one MAC per filter tap per output element).
            .flops(2 * n_out as u64 * cols as u64)
            .with_shape_label(format!("{} ch={ch} k={k} C={cols}", fmt.name()))
    }

    macro_rules! dw_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr, $name:literal) => {
            #[bench(name = $name, dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                dw_bench($kernel(dt), $fmt, 1, 256, 64, 64, 8, 1, 0, 1, dt)
            }
        };
    }
    dw_bench_fmt!(
        bench_mxfp4,
        mt_mxfp4_depthwise_conv2d::kernel_ir_for,
        QFormat::Mxfp4,
        "ffai/depthwise_conv2d_block/mxfp4"
    );
    dw_bench_fmt!(
        bench_nvfp4,
        mt_nvfp4_depthwise_conv2d::kernel_ir_for,
        QFormat::Nvfp4,
        "ffai/depthwise_conv2d_block/nvfp4"
    );
    dw_bench_fmt!(
        bench_fp4,
        mt_fp4_depthwise_conv2d::kernel_ir_for,
        QFormat::Fp4,
        "ffai/depthwise_conv2d_block/fp4"
    );
    dw_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_depthwise_conv2d::kernel_ir_for,
        QFormat::Mxfp8E4,
        "ffai/depthwise_conv2d_block/mxfp8_e4m3"
    );
    dw_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_depthwise_conv2d::kernel_ir_for,
        QFormat::Mxfp8E5,
        "ffai/depthwise_conv2d_block/mxfp8_e5m2"
    );
    dw_bench_fmt!(
        bench_fp8_e5m2,
        mt_fp8_e5m2_depthwise_conv2d::kernel_ir_for,
        QFormat::Fp8E5m2,
        "ffai/depthwise_conv2d_block/fp8_e5m2"
    );
    dw_bench_fmt!(
        bench_nvfp8,
        mt_nvfp8_depthwise_conv2d::kernel_ir_for,
        QFormat::Nvfp8,
        "ffai/depthwise_conv2d_block/nvfp8"
    );
    dw_bench_fmt!(
        bench_int8,
        mt_int8_depthwise_conv2d::kernel_ir_for,
        QFormat::Int8,
        "ffai/depthwise_conv2d_block/int8"
    );
}

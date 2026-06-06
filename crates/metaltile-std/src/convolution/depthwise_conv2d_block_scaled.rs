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
            let wt = mt_decode_e2m1(nib) * scale;
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
                mt_decode_e4m3(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global;
            let wt = mt_decode_e2m1(nib) * scale;
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
            let wt = mt_decode_e2m1(nib) * scale;
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
            let elem = mt_decode_e4m3(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_e5m2(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_e5m2(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_e4m3(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_int8(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ── Symmetric sub-byte integer depthwise conv (int2/3/4/5/6 + MXINT2..6) ─────
// The filter element is a signed N-bit two's-complement code. Unlike the
// per-row word-aligned GEMV (`mlx/block_scaled_matmul.rs`, where `in_dim` is a
// multiple of 32), the depthwise filter squeezes to `[ch, C]` with `C = k*k` —
// which is generally NOT a multiple of 32, so **per-row bit-streams are not
// word-aligned**. `quant::format::pack` packs the whole `[ch, C]` matrix as ONE
// flat LSB-first bit-stream keyed on the *global* flat element index
// `flat = c * C + col` (it never re-bases per row), so the decode here must use
// that same global index: `bit_off = (c·C + col)·bits`, then a straddle-aware
// two-word read into the flat `weight` buffer (no per-row word base). This is
// exact whether or not `C` is a multiple of 32. Everything else — the
// (n, c, oh, ow) flattening, the padding/dilation/stride loop, the per-block
// scale index `c·n_blocks + col/block_size`, the Grid3D geometry — is identical
// to the existing 8-bit int kernel. `$half`/`$full` are passed as literals
// (2^(N-1) / 2^N) to keep the constexpr math out of the DSL shift operands.

/// FP32-scaled symmetric int depthwise conv (int2/3/4/5/6): per-tap bit-stream
/// code × per-group FP32 scale. The filter is one flat `[ch, C]` bit-stream, so
/// tap `(c, col)` decodes at the GLOBAL bit offset `(c·C + col)·bits`.
macro_rules! int_dw_conv_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
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
            let n_blocks = cols / block_size;
            // Flat global element base for this channel's filter row (codes are a
            // single `[ch, C]` bit-stream keyed on `c·C + col`, never per-row
            // word-aligned), and the per-channel block base for the scales.
            let w_row_elem = c * cols;
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
                    // Straddle-aware two-word read at the GLOBAL bit offset.
                    let bit_off = (w_row_elem + col) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[word_idx]);
                    let w1 = load(weight[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let elem = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[w_row_blk + col / block_size]);
                    let wt = elem * scale;
                    acc = acc + x_m * wt;
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_dw_conv_f32!(mt_int2_depthwise_conv2d, 2u32, 2u32, 4.0f32);
int_dw_conv_f32!(mt_int3_depthwise_conv2d, 3u32, 4u32, 8.0f32);
int_dw_conv_f32!(mt_int4_depthwise_conv2d, 4u32, 8u32, 16.0f32);
int_dw_conv_f32!(mt_int5_depthwise_conv2d, 5u32, 16u32, 32.0f32);
int_dw_conv_f32!(mt_int6_depthwise_conv2d, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int depthwise conv (MXINT2/3/4/5/6): per-tap bit-stream
/// code × pow-2 (E8M0) block scale `2^(bits-127)`. Same flat-bit-stream decode as
/// `int_dw_conv_f32`; only the scale axis differs (one u8 exponent per block).
macro_rules! int_dw_conv_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
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
            let n_blocks = cols / block_size;
            let w_row_elem = c * cols;
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
                    let bit_off = (w_row_elem + col) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[word_idx]);
                    let w1 = load(weight[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let elem = select(q >= $half, qf - $full, qf); // sign-extend
                    let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    let wt = elem * scale;
                    acc = acc + x_m * wt;
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_dw_conv_e8m0!(mt_mxint2_depthwise_conv2d, 2u32, 2u32, 4.0f32);
int_dw_conv_e8m0!(mt_mxint3_depthwise_conv2d, 3u32, 4u32, 8.0f32);
int_dw_conv_e8m0!(mt_mxint4_depthwise_conv2d, 4u32, 8u32, 16.0f32);
int_dw_conv_e8m0!(mt_mxint5_depthwise_conv2d, 5u32, 16u32, 32.0f32);
int_dw_conv_e8m0!(mt_mxint6_depthwise_conv2d, 6u32, 32u32, 64.0f32);

/// MXINT8 quantized depthwise conv2d — 8-bit symmetric codes (byte layout,
/// block 32), E8M0 pow-2 block scale `2^(bits-127)`. Byte-per-code layout
/// (one `uchar` each) identical to the 8-bit float formats, so it indexes
/// `weight[c·C + col]` like `mt_int8`; only the decode + E8M0 scale differ.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_depthwise_conv2d<T>(
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
            let elem = mt_decode_int8(load(weight[w_row + col]).cast::<u32>());
            let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ── FP16-scale twins (Track-1 fp16-scale formats) ───────────────────────────
// Each kernel below is a verbatim clone of its FP32-scaled twin above; the ONLY
// change is the scale tensor (`Tensor<f16>` instead of `Tensor<f32>`) and the
// scale read (`load(scales[...]).cast::<f32>()` instead of `load(scales[...])`).
// Element decode, weight indexing (including the sub-byte GLOBAL flat bit-offset
// `(c·C + col)·bits`), Grid3D geometry, and the conv loop are all IDENTICAL to
// the twin. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale
// shape), exactly as fp8_e4m3 reuses nvfp8 today.

/// fp4 (FP16 scale) quantized depthwise conv2d — E2M1 weight (group 32),
/// per-group FP16 scale. Clone of `mt_fp4_depthwise_conv2d`, scale → f16.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
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
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = mt_decode_e2m1(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// fp8 (E5M2, FP16 scale) quantized depthwise conv2d — 8-bit weight (group 32),
/// per-group FP16 scale. Clone of `mt_fp8_e5m2_depthwise_conv2d`, scale → f16.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
            let elem = mt_decode_e5m2(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp8 (FP16 scale) quantized depthwise conv2d — E4M3 weight (block 16),
/// per-block FP16 scale. Clone of `mt_nvfp8_depthwise_conv2d`, scale → f16.
/// Also serves **fp8_e4m3_f16** (same 8-bit-E4M3 + f16-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
            let elem = mt_decode_e4m3(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// FP16-scaled symmetric int depthwise conv (int2/3/4/5/6, FP16-scale twins of
/// `int_dw_conv_f32`): per-tap bit-stream code × per-group FP16 scale. The filter
/// is one flat `[ch, C]` bit-stream, so tap `(c, col)` decodes at the GLOBAL bit
/// offset `(c·C + col)·bits` — identical to the FP32 twin; only the scale differs.
macro_rules! int_dw_conv_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f16>,
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
            // Flat global element base for this channel's filter row (codes are a
            // single `[ch, C]` bit-stream keyed on `c·C + col`, never per-row
            // word-aligned), and the per-channel block base for the scales.
            let w_row_elem = c * cols;
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
                    // Straddle-aware two-word read at the GLOBAL bit offset.
                    let bit_off = (w_row_elem + col) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[word_idx]);
                    let w1 = load(weight[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let elem = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                    let wt = elem * scale;
                    acc = acc + x_m * wt;
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_dw_conv_f16!(mt_int2_f16_depthwise_conv2d, 2u32, 2u32, 4.0f32);
int_dw_conv_f16!(mt_int3_f16_depthwise_conv2d, 3u32, 4u32, 8.0f32);
int_dw_conv_f16!(mt_int4_f16_depthwise_conv2d, 4u32, 8u32, 16.0f32);
int_dw_conv_f16!(mt_int5_f16_depthwise_conv2d, 5u32, 16u32, 32.0f32);
int_dw_conv_f16!(mt_int6_f16_depthwise_conv2d, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) quantized depthwise conv2d — 8-bit codes (byte layout,
/// group 64), per-group FP16 scale. Clone of `mt_int8_depthwise_conv2d`,
/// scale → f16.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
            let elem = mt_decode_int8(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
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

    // Symmetric sub-byte ints (int2-6, FP32 group scale 64) + MXINT (mxint2-6,
    // E8M0 block scale 32) + MXINT8 (8-bit, E8M0). k=8 → C=64 is divisible by
    // every group/block (64 and 32), so each channel's flat-bit-stream filter row
    // lands on a u32 boundary and the per-block scale index is exact. The kernel
    // and oracle share the codec, so the GPU output tracks the dequant-then-conv
    // reference to float precision regardless of how coarse the quantization is.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int2_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int2,
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
    fn test_int3_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int3_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int3,
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
    fn test_int4_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int4_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int4,
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
    fn test_int5_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int5_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int5,
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
    fn test_int6_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int6_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int6,
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
    fn test_mxint2_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxint2_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxint2,
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
    fn test_mxint3_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxint3_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxint3,
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
    fn test_mxint4_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxint4_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxint4,
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
    fn test_mxint5_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxint5_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxint5,
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
    fn test_mxint6_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxint6_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxint6,
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
    fn test_mxint8_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_mxint8_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Mxint8,
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

    // FP16-scale twins: same element packing + geometry as their FP32 twins, with
    // an FP16 scale tensor. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 +
    // f16 scale). Tolerances match the other formats since the per-tap math is
    // identical apart from the half-precision scale read.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_nvfp8_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
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
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_nvfp8_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
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
    fn test_fp4_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_fp4_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Fp4F16,
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
    fn test_fp8_e5m2_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_fp8_e5m2_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
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
    fn test_int2_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int2_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int2F16,
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
    fn test_int3_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int3_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int3F16,
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
    fn test_int4_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int4_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int4F16,
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
    fn test_int5_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int5_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int5F16,
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
    fn test_int6_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int6_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int6F16,
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
    fn test_int8_f16_depthwise_conv2d(dt: DType) -> TestSetup {
        dw_setup(
            mt_int8_f16_depthwise_conv2d::kernel_ir_for(dt),
            QFormat::Int8F16,
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
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words
        // (+ guard word for straddling reads). Both axes are driven off the
        // format so new integer formats pick up the right buffer geometry.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (ch * cols, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(ch * cols, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
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
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                dw_bench($kernel(dt), $fmt, 1, 256, 64, 64, 8, 1, 0, 1, dt)
            }
        };
    }
    dw_bench_fmt!(bench_mxfp4, mt_mxfp4_depthwise_conv2d::kernel_ir_for, QFormat::Mxfp4);
    dw_bench_fmt!(bench_nvfp4, mt_nvfp4_depthwise_conv2d::kernel_ir_for, QFormat::Nvfp4);
    dw_bench_fmt!(bench_fp4, mt_fp4_depthwise_conv2d::kernel_ir_for, QFormat::Fp4);
    dw_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_depthwise_conv2d::kernel_ir_for,
        QFormat::Mxfp8E4
    );
    dw_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_depthwise_conv2d::kernel_ir_for,
        QFormat::Mxfp8E5
    );
    dw_bench_fmt!(bench_fp8_e5m2, mt_fp8_e5m2_depthwise_conv2d::kernel_ir_for, QFormat::Fp8E5m2);
    dw_bench_fmt!(bench_nvfp8, mt_nvfp8_depthwise_conv2d::kernel_ir_for, QFormat::Nvfp8);
    dw_bench_fmt!(bench_int8, mt_int8_depthwise_conv2d::kernel_ir_for, QFormat::Int8);
    // Symmetric sub-byte ints (int2-6, FP32 group scale) + MXINT (mxint2-6, E8M0
    // block scale) + MXINT8 (8-bit, E8M0).
    dw_bench_fmt!(bench_int2, mt_int2_depthwise_conv2d::kernel_ir_for, QFormat::Int2);
    dw_bench_fmt!(bench_int3, mt_int3_depthwise_conv2d::kernel_ir_for, QFormat::Int3);
    dw_bench_fmt!(bench_int4, mt_int4_depthwise_conv2d::kernel_ir_for, QFormat::Int4);
    dw_bench_fmt!(bench_int5, mt_int5_depthwise_conv2d::kernel_ir_for, QFormat::Int5);
    dw_bench_fmt!(bench_int6, mt_int6_depthwise_conv2d::kernel_ir_for, QFormat::Int6);
    dw_bench_fmt!(bench_mxint2, mt_mxint2_depthwise_conv2d::kernel_ir_for, QFormat::Mxint2);
    dw_bench_fmt!(bench_mxint3, mt_mxint3_depthwise_conv2d::kernel_ir_for, QFormat::Mxint3);
    dw_bench_fmt!(bench_mxint4, mt_mxint4_depthwise_conv2d::kernel_ir_for, QFormat::Mxint4);
    dw_bench_fmt!(bench_mxint5, mt_mxint5_depthwise_conv2d::kernel_ir_for, QFormat::Mxint5);
    dw_bench_fmt!(bench_mxint6, mt_mxint6_depthwise_conv2d::kernel_ir_for, QFormat::Mxint6);
    dw_bench_fmt!(bench_mxint8, mt_mxint8_depthwise_conv2d::kernel_ir_for, QFormat::Mxint8);
    // FP16-scale twins (fp8_e4m3_f16 reuses the nvfp8_f16 kernel).
    dw_bench_fmt!(bench_nvfp8_f16, mt_nvfp8_f16_depthwise_conv2d::kernel_ir_for, QFormat::Nvfp8F16);
    dw_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_depthwise_conv2d::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    dw_bench_fmt!(bench_fp4_f16, mt_fp4_f16_depthwise_conv2d::kernel_ir_for, QFormat::Fp4F16);
    dw_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_depthwise_conv2d::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    dw_bench_fmt!(bench_int2_f16, mt_int2_f16_depthwise_conv2d::kernel_ir_for, QFormat::Int2F16);
    dw_bench_fmt!(bench_int3_f16, mt_int3_f16_depthwise_conv2d::kernel_ir_for, QFormat::Int3F16);
    dw_bench_fmt!(bench_int4_f16, mt_int4_f16_depthwise_conv2d::kernel_ir_for, QFormat::Int4F16);
    dw_bench_fmt!(bench_int5_f16, mt_int5_f16_depthwise_conv2d::kernel_ir_for, QFormat::Int5F16);
    dw_bench_fmt!(bench_int6_f16, mt_int6_f16_depthwise_conv2d::kernel_ir_for, QFormat::Int6F16);
    dw_bench_fmt!(bench_int8_f16, mt_int8_f16_depthwise_conv2d::kernel_ir_for, QFormat::Int8F16);
}

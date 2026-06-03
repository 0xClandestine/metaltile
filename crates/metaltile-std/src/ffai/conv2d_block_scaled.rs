//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight 2D convolution** — the weight-quantized
//! counterpart of `ffai/conv2d.rs`.
//!
//! A conv2d output element is a dot product over the `in_ch × kh × kw`
//! receptive field against a filter row, so the filter `[out_ch, in_ch, kh,
//! kw]` is a genuine quantizable parameter. We treat it as a 2-D matrix
//! `[out_ch, C]` with `C = in_ch · kh · kw` — the per-output-channel
//! contraction — block-scaled along `C` in the spec formats (mxfp4 / nvfp4 /
//! mxfp8 e4m3+e5m2 / nvfp8 + legacy fp4/fp8 + symmetric int8 + sub-byte
//! int2..int6 / mxint2..mxint6 + mxint8).
//!
//! For an output channel `oc` and a tap `(ic, ky, kx)` the contraction index
//! is `col = (ic·kh + ky)·kw + kx = ic·kh·kw + ky·kw + kx`. The dense filter
//! value `weight[((oc·in_ch+ic)·kh+ky)·kw+kx]` becomes
//! `element_decode(code[oc, col]) · block_scale[oc, col/block_size]` (× global
//! for nvfp4). 4-bit codes are packed `[out_ch, C/8]` u32 (8 nibbles/word, code
//! at word `oc·(C/8)+col/8`, shift `(col%8)·4`); 8-bit codes are `[out_ch, C]`
//! u8 (byte at `oc·C+col`). Sub-byte integers (int2..6 / mxint2..6) are a tight
//! LSB-first bit-stream, per output-channel row word-aligned: row `oc` starts at
//! word `oc·(C·bits/32)`, code `col` at bit `col·bits` (straddle-aware read).
//! Only the filter is quantized — the per-channel `bias` stays `T`.
//!
//! Geometry is **identical** to the dense `conv2d_generic`: **Grid3D**, one
//! thread per output element (`program_id::<0>()` = flat
//! `((n·out_ch+oc)·out_h+oh)·out_w+ow`), the same stride / padding / dilation
//! receptive-field walk in the padded input frame, fp32 accumulation, padding
//! taps clamped to contribute zero. `C` is a multiple of `block_size` (4-bit
//! `block_size` a multiple of 8). fp8_e4m3 reuses the nvfp8 kernel. Codegen-only;
//! correctness pinned by the in-source `#[test_kernel]`s vs a
//! `quant::format::dequant` oracle running the dense conv2d math.

use metaltile::kernel;

/// mxfp4 quantized-weight conv2d — E2M1 filter (block 32), E8M0 pow-2 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    // Flat output index → (n, oc, oh, ow). One thread per output.
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    // Receptive-field anchors in the *padded* input frame (see conv2d.rs) —
    // a real pixel at padded row `ph` sits at unpadded row `ph - pad_h`,
    // valid iff `pad_h <= ph < pad_h + in_h`.
    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    // Quantized-filter layout: filter as [out_ch, C], C = in_ch*kh*kw,
    // block-scaled along C. 4-bit codes pack 8 nibbles per u32 word.
    let contraction = in_ch * kh * kw;
    let w_packs_per_row = contraction / 8u32;
    let n_blocks = contraction / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    // Walk the in_ch × kh × kw receptive field. Padding pixels (row/col
    // outside the real input) contribute zero — the load is clamped to a
    // valid index and masked out. `col` is the contraction index into the
    // quantized filter row.
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale =
                    exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
                acc = acc + pix_m * (e2m1_decode(nib) * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// nvfp4 quantized-weight conv2d — E2M1 filter (block 16), E4M3 micro-scale × global.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let w_packs_per_row = contraction / 8u32;
    let n_blocks = contraction / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale =
                    e4m3_decode(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global;
                acc = acc + pix_m * (e2m1_decode(nib) * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// Legacy fp4 quantized-weight conv2d — E2M1 filter (group 32), per-group FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let w_packs_per_row = contraction / 8u32;
    let n_blocks = contraction / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + pix_m * (e2m1_decode(nib) * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E4M3) quantized-weight conv2d — 8-bit filter (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
                let scale =
                    exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E5M2) quantized-weight conv2d — 8-bit filter (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
                let scale =
                    exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// Legacy fp8 (E5M2) quantized-weight conv2d — 8-bit filter (group 32), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// nvfp8 quantized-weight conv2d — E4M3 filter (block 16), per-block FP32 scale.
/// Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// Symmetric int8 quantized-weight conv2d — 8-bit codes (group 64), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

// ── Symmetric sub-byte integer conv2d (int2/3/4/5/6 + MXINT2..6) ────────────
// The filter element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words. The filter is the 2-D matrix [out_ch, C] with
// C = in_ch·kh·kw, so each output-channel row is a contiguous bit-stream of
// `C` codes; row `oc` begins at word `oc · (C·bits / 32)` (per-row word-aligned
// because `C·bits` is a multiple of 32). For contraction index `col` the code
// sits at bit `col·bits` within that row's stream. Decode mirrors the GEMV
// family's proven `int_qgemv_*` macros (and `block_scaled_dequant`'s
// `int_dequant_*`): a straddle-aware two-word read of the low N bits, then
// sign-extend in float (subtract 2^N when the top bit is set; `$half`/`$full`
// are 2^(N-1) / 2^N), then multiply by the block scale. Only the per-element
// decode differs from `mt_int8_conv2d`; the receptive-field walk, padding clamp,
// fp32 accumulation, and Grid3D geometry are identical to the rest of the
// family. `$half`/`$full` are passed as literals to keep the constexpr math out
// of the DSL shift operands.

/// FP32-scaled symmetric int conv2d (int2/3/4/5/6): per-element bit-stream
/// filter code × per-group FP32 scale, contracted over the in_ch×kh×kw
/// receptive field. `w_row_word` indexes the filter row's tight bit-stream
/// (`C · bits / 32` u32 words per output channel).
macro_rules! int_conv2d_f32 {
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
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] stride_h: u32,
            #[constexpr] stride_w: u32,
            #[constexpr] pad_h: u32,
            #[constexpr] pad_w: u32,
            #[constexpr] block_size: u32,
        ) {
            let idx = program_id::<0>();
            let ow = idx % out_w;
            let t1 = idx / out_w;
            let oh = t1 % out_h;
            let t2 = t1 / out_h;
            let oc = t2 % out_ch;
            let n = t2 / out_ch;

            let ph0 = oh * stride_h;
            let pw0 = ow * stride_w;

            let input_plane = in_h * in_w;
            let in_n_stride = in_ch * input_plane;

            // Filter as [out_ch, C], C = in_ch*kh*kw, tight bit-stream along C.
            let contraction = in_ch * kh * kw;
            let words_per_row = contraction * $bits / 32u32;
            let n_blocks = contraction / block_size;
            let w_row_word = oc * words_per_row;
            let w_row_blk = oc * n_blocks;

            let mut acc = load(bias[oc]).cast::<f32>();

            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * input_plane;
                let col_ic = ic * kh * kw;
                for ky in range(0u32, kh, 1u32) {
                    let ph = ph0 + ky;
                    let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
                    let ih = select(row_ok, ph - pad_h, 0u32);
                    for kx in range(0u32, kw, 1u32) {
                        let pw = pw0 + kx;
                        let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                        let valid = row_ok & col_ok;
                        let iw = select(col_ok, pw - pad_w, 0u32);

                        let in_idx = in_ic_base + ih * in_w + iw;
                        let pix = load(input[in_idx]).cast::<f32>();
                        let pix_m = select(valid, pix, 0.0f32);

                        // Straddle-aware decode of the N-bit code at bit col*bits.
                        let col = col_ic + ky * kw + kx;
                        let bit_off = col * $bits;
                        let word_idx = bit_off / 32u32;
                        let bit_in_w = bit_off & 31u32;
                        let bits_in_w0 = 32u32 - bit_in_w;
                        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                        let spill = $bits - lo_bits;
                        let w0 = load(weight[w_row_word + word_idx]);
                        let w1 = load(
                            weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                        );
                        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                        let q = lo | hi;
                        let qf = q.cast::<f32>();
                        let elem = select(q >= $half, qf - $full, qf); // sign-extend
                        let scale = load(scales[w_row_blk + col / block_size]);
                        acc = acc + pix_m * (elem * scale);
                    }
                }
            }

            store(out[idx], acc.cast::<T>());
        }
    };
}
int_conv2d_f32!(mt_int2_conv2d, 2u32, 2u32, 4.0f32);
int_conv2d_f32!(mt_int3_conv2d, 3u32, 4u32, 8.0f32);
int_conv2d_f32!(mt_int4_conv2d, 4u32, 8u32, 16.0f32);
int_conv2d_f32!(mt_int5_conv2d, 5u32, 16u32, 32.0f32);
int_conv2d_f32!(mt_int6_conv2d, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int conv2d (MXINT2/3/4/5/6): per-element bit-stream
/// filter code × pow-2 (E8M0) block scale `2^(bits-127)`, contracted over the
/// receptive field. Same straddle-aware decode and geometry as `int_conv2d_f32`;
/// only the scale axis differs (one u8 exponent per block instead of a raw f32).
macro_rules! int_conv2d_e8m0 {
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
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] stride_h: u32,
            #[constexpr] stride_w: u32,
            #[constexpr] pad_h: u32,
            #[constexpr] pad_w: u32,
            #[constexpr] block_size: u32,
        ) {
            let idx = program_id::<0>();
            let ow = idx % out_w;
            let t1 = idx / out_w;
            let oh = t1 % out_h;
            let t2 = t1 / out_h;
            let oc = t2 % out_ch;
            let n = t2 / out_ch;

            let ph0 = oh * stride_h;
            let pw0 = ow * stride_w;

            let input_plane = in_h * in_w;
            let in_n_stride = in_ch * input_plane;

            let contraction = in_ch * kh * kw;
            let words_per_row = contraction * $bits / 32u32;
            let n_blocks = contraction / block_size;
            let w_row_word = oc * words_per_row;
            let w_row_blk = oc * n_blocks;

            let mut acc = load(bias[oc]).cast::<f32>();

            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * input_plane;
                let col_ic = ic * kh * kw;
                for ky in range(0u32, kh, 1u32) {
                    let ph = ph0 + ky;
                    let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
                    let ih = select(row_ok, ph - pad_h, 0u32);
                    for kx in range(0u32, kw, 1u32) {
                        let pw = pw0 + kx;
                        let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                        let valid = row_ok & col_ok;
                        let iw = select(col_ok, pw - pad_w, 0u32);

                        let in_idx = in_ic_base + ih * in_w + iw;
                        let pix = load(input[in_idx]).cast::<f32>();
                        let pix_m = select(valid, pix, 0.0f32);

                        let col = col_ic + ky * kw + kx;
                        let bit_off = col * $bits;
                        let word_idx = bit_off / 32u32;
                        let bit_in_w = bit_off & 31u32;
                        let bits_in_w0 = 32u32 - bit_in_w;
                        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                        let spill = $bits - lo_bits;
                        let w0 = load(weight[w_row_word + word_idx]);
                        let w1 = load(
                            weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                        );
                        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                        let q = lo | hi;
                        let qf = q.cast::<f32>();
                        let elem = select(q >= $half, qf - $full, qf); // sign-extend
                        let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                        let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                        acc = acc + pix_m * (elem * scale);
                    }
                }
            }

            store(out[idx], acc.cast::<T>());
        }
    };
}
int_conv2d_e8m0!(mt_mxint2_conv2d, 2u32, 2u32, 4.0f32);
int_conv2d_e8m0!(mt_mxint3_conv2d, 3u32, 4u32, 8.0f32);
int_conv2d_e8m0!(mt_mxint4_conv2d, 4u32, 8u32, 16.0f32);
int_conv2d_e8m0!(mt_mxint5_conv2d, 5u32, 16u32, 32.0f32);
int_conv2d_e8m0!(mt_mxint6_conv2d, 6u32, 32u32, 64.0f32);

/// MXINT8 quantized-weight conv2d — 8-bit symmetric codes (byte layout, block
/// 32), E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the 8-bit
/// float formats (one byte per code); decode is `int8_decode → elem · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
                let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

// ── FP16-scale twins ─────────────────────────────────────────────────────────
// Near-clones of the FP32-scaled conv2d kernels above: the per-element decode
// (E2M1 / E4M3 / E5M2 / sub-byte int bit-stream + sign-extend / int8 byte),
// weight indexing, receptive-field walk, padding clamp, fp32 accumulation, and
// Grid3D geometry are IDENTICAL to the FP32 twin. Only the scale changes — the
// `scales` tensor is a native `half` (`Tensor<f16>`) read with `.cast::<f32>()`
// instead of a raw `Tensor<f32>`. The GPU half load matches the host
// `f16_scale_decode`, so the dequant-then-conv oracle still holds exactly. See
// `mlx/block_scaled_dequant.rs`'s `mt_*_f16_dequant` for the verified scale read.

/// nvfp8 (FP16 scale) quantized-weight conv2d — E4M3 filter (block 16), per-block
/// FP16 scale. Also serves **fp8_e4m3_f16** (same 8-bit-E4M3 + scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// fp4 (FP16 scale) quantized-weight conv2d — E2M1 filter (group 32), per-group
/// FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let w_packs_per_row = contraction / 8u32;
    let n_blocks = contraction / block_size;
    let w_row_pack = oc * w_packs_per_row;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + pix_m * (e2m1_decode(nib) * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// fp8 (E5M2, FP16 scale) quantized-weight conv2d — E5M2 filter (group 32),
/// per-group FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// Symmetric int8 (FP16 scale) quantized-weight conv2d — 8-bit codes (group 64),
/// per-group FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;

    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;

    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;

    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row = oc * contraction;
    let w_row_blk = oc * n_blocks;

    let mut acc = load(bias[oc]).cast::<f32>();

    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);

                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);

                let col = col_ic + ky * kw + kx;
                let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + pix_m * (elem * scale);
            }
        }
    }

    store(out[idx], acc.cast::<T>());
}

/// FP16-scaled symmetric int conv2d (int2/3/4/5/6, FP16 scale): clone of
/// `int_conv2d_f32` with the `scales` tensor read as a native `half`. The
/// straddle-aware bit-stream decode, weight indexing, receptive-field walk, and
/// Grid3D geometry are identical to the FP32 twin; only the scale axis differs.
macro_rules! int_conv2d_f16 {
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
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] stride_h: u32,
            #[constexpr] stride_w: u32,
            #[constexpr] pad_h: u32,
            #[constexpr] pad_w: u32,
            #[constexpr] block_size: u32,
        ) {
            let idx = program_id::<0>();
            let ow = idx % out_w;
            let t1 = idx / out_w;
            let oh = t1 % out_h;
            let t2 = t1 / out_h;
            let oc = t2 % out_ch;
            let n = t2 / out_ch;

            let ph0 = oh * stride_h;
            let pw0 = ow * stride_w;

            let input_plane = in_h * in_w;
            let in_n_stride = in_ch * input_plane;

            // Filter as [out_ch, C], C = in_ch*kh*kw, tight bit-stream along C.
            let contraction = in_ch * kh * kw;
            let words_per_row = contraction * $bits / 32u32;
            let n_blocks = contraction / block_size;
            let w_row_word = oc * words_per_row;
            let w_row_blk = oc * n_blocks;

            let mut acc = load(bias[oc]).cast::<f32>();

            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * input_plane;
                let col_ic = ic * kh * kw;
                for ky in range(0u32, kh, 1u32) {
                    let ph = ph0 + ky;
                    let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
                    let ih = select(row_ok, ph - pad_h, 0u32);
                    for kx in range(0u32, kw, 1u32) {
                        let pw = pw0 + kx;
                        let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                        let valid = row_ok & col_ok;
                        let iw = select(col_ok, pw - pad_w, 0u32);

                        let in_idx = in_ic_base + ih * in_w + iw;
                        let pix = load(input[in_idx]).cast::<f32>();
                        let pix_m = select(valid, pix, 0.0f32);

                        // Straddle-aware decode of the N-bit code at bit col*bits.
                        let col = col_ic + ky * kw + kx;
                        let bit_off = col * $bits;
                        let word_idx = bit_off / 32u32;
                        let bit_in_w = bit_off & 31u32;
                        let bits_in_w0 = 32u32 - bit_in_w;
                        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                        let spill = $bits - lo_bits;
                        let w0 = load(weight[w_row_word + word_idx]);
                        let w1 = load(
                            weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                        );
                        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                        let q = lo | hi;
                        let qf = q.cast::<f32>();
                        let elem = select(q >= $half, qf - $full, qf); // sign-extend
                        let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                        acc = acc + pix_m * (elem * scale);
                    }
                }
            }

            store(out[idx], acc.cast::<T>());
        }
    };
}
int_conv2d_f16!(mt_int2_f16_conv2d, 2u32, 2u32, 4.0f32);
int_conv2d_f16!(mt_int3_f16_conv2d, 3u32, 4u32, 8.0f32);
int_conv2d_f16!(mt_int4_f16_conv2d, 4u32, 8u32, 16.0f32);
int_conv2d_f16!(mt_int5_f16_conv2d, 5u32, 16u32, 32.0f32);
int_conv2d_f16!(mt_int6_f16_conv2d, 6u32, 32u32, 64.0f32);

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Deterministic ramp identical to the dense conv2d helper: a bounded
    /// zig-zag so f16/bf16 stay in range.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 2D conv oracle (NCHW input, OIHW weight), groups=1, dilation=1.
    /// Padding taps contribute zero — the SAME dense math as conv2d.rs's
    /// `naive_conv2d`, run over the *dequantized* filter. All f32.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn naive_conv2d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        // Quantized filter is laid out as the 2-D matrix [out_ch, C] with
        // C = in_ch*kh*kw and col = (ic*kh + ky)*kw + kx, so the dequantized
        // weight row `oc` is contiguous over `col`.
        let contraction = in_ch * kh * kw;
        let mut out = vec![0.0f32; batch * out_ch * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[oc];
                        for ic in 0..in_ch {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ph = oh * stride_h + ky;
                                    let pw = ow * stride_w + kx;
                                    if ph < pad_h
                                        || ph >= pad_h + in_h
                                        || pw < pad_w
                                        || pw >= pad_w + in_w
                                    {
                                        continue;
                                    }
                                    let ih = ph - pad_h;
                                    let iw = pw - pad_w;
                                    let in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                                    let col = (ic * kh + ky) * kw + kx;
                                    let w_idx = oc * contraction + col;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        let o_idx = ((n * out_ch + oc) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the [out_ch, C] filter via the
    /// shared codec, dequantize for the oracle, and run the dense conv2d math.
    /// Mirrors conv2d.rs's `conv2d_setup` grid + KernelMode exactly.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        // Contraction C = in_ch*kh*kw — the quantized filter is [out_ch, C].
        let contraction = in_ch * kh * kw;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 6.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        // Quantize the [out_ch, C] filter via the shared codec.
        let w_f = ramp(out_ch * contraction, 11, 4.0);
        let p = crate::quant::format::pack(fmt, &w_f, out_ch, contraction);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, contraction);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        // Oracle: dense conv2d over the dequantized filter row [out_ch, C].
        let expected = naive_conv2d(
            &input, &wdq, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w,
            pad_h, pad_w,
        );
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
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", pad_h as u32)
            .constexpr("pad_w", pad_w as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // in_ch=4, kh=kw=4 → C = 64 (÷ 16/32/64); 8×8 image, stride 1, pad 1,
    // dilation 1; out_ch=8. Exercises the in-kernel padding clamp.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxfp4_conv2d::kernel_ir_for(dt),
            QFormat::Mxfp4,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_nvfp4_conv2d::kernel_ir_for(dt),
            QFormat::Nvfp4,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_fp4_conv2d::kernel_ir_for(dt),
            QFormat::Fp4,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxfp8_e4m3_conv2d::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxfp8_e5m2_conv2d::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_fp8_e5m2_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_nvfp8_conv2d::kernel_ir_for(dt),
            QFormat::Nvfp8,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_nvfp8_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int8_conv2d::kernel_ir_for(dt),
            QFormat::Int8,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_ch=4, kh=kw=4 → C = 64, so
    // `C*bits % 32 == 0` for every width and each filter row's bit-stream is
    // word-aligned. The kernel and oracle share the codec, so the GPU output
    // tracks the dequant-then-conv reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int2_conv2d::kernel_ir_for(dt),
            QFormat::Int2,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int3_conv2d::kernel_ir_for(dt),
            QFormat::Int3,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int4_conv2d::kernel_ir_for(dt),
            QFormat::Int4,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int5_conv2d::kernel_ir_for(dt),
            QFormat::Int5,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int6_conv2d::kernel_ir_for(dt),
            QFormat::Int6,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxint2_conv2d::kernel_ir_for(dt),
            QFormat::Mxint2,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxint3_conv2d::kernel_ir_for(dt),
            QFormat::Mxint3,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxint4_conv2d::kernel_ir_for(dt),
            QFormat::Mxint4,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxint5_conv2d::kernel_ir_for(dt),
            QFormat::Mxint5,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxint6_conv2d::kernel_ir_for(dt),
            QFormat::Mxint6,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_mxint8_conv2d::kernel_ir_for(dt),
            QFormat::Mxint8,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }

    // FP16-scale twins: same element packing + geometry as their FP32-scaled
    // twin, only the per-group scale is read as a native half. C = 64 again, so
    // every sub-byte filter row's bit-stream stays word-aligned.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_nvfp8_f16_conv2d::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_nvfp8_f16_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_fp4_f16_conv2d::kernel_ir_for(dt),
            QFormat::Fp4F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_fp8_e5m2_f16_conv2d::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int2_f16_conv2d::kernel_ir_for(dt),
            QFormat::Int2F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int3_f16_conv2d::kernel_ir_for(dt),
            QFormat::Int3F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int4_f16_conv2d::kernel_ir_for(dt),
            QFormat::Int4F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int5_f16_conv2d::kernel_ir_for(dt),
            QFormat::Int5F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int6_f16_conv2d::kernel_ir_for(dt),
            QFormat::Int6F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_conv2d(dt: DType) -> TestSetup {
        conv2d_setup(
            mt_int8_f16_conv2d::kernel_ir_for(dt),
            QFormat::Int8F16,
            1,
            4,
            8,
            8,
            8,
            4,
            4,
            1,
            1,
            1,
            1,
            dt,
        )
    }
}

/// Decode-shape benches: a realistic conv (in_ch=64, out_ch=128, 4×4 kernel →
/// C = 1024, divisible by all block sizes). Grid3D, one thread per output
/// element; bytes_moved counts weight + scales + input + output streams.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn conv2d_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_h = (in_h - kh) / stride_h + 1;
        let out_w = (in_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        let contraction = in_ch * kh * kw;
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let n_codes = out_ch * contraction;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_codes, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n_codes, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = out_ch * (contraction / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + batch * in_ch * in_h * in_w * sz
            + out_ch * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", 0u32)
            .constexpr("pad_w", 0u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            // 2 * n_out * C; C = in_ch*kh*kw is the per-output contraction.
            .flops(2 * n_out as u64 * contraction as u64)
            .with_shape_label(format!(
                "{} co={out_ch} ho={out_h} wo={out_w} C={contraction}",
                fmt.name()
            ))
    }

    macro_rules! conv2d_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr, $name:literal) => {
            #[bench(name = $name, dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                // in_ch=64, out_ch=128, 4×4 kernel → C=1024 (÷ 16/32/64).
                conv2d_bench($kernel(dt), $fmt, 1, 64, 56, 56, 128, 4, 4, 1, 1, dt)
            }
        };
    }
    conv2d_bench_fmt!(
        bench_mxfp4,
        mt_mxfp4_conv2d::kernel_ir_for,
        QFormat::Mxfp4,
        "ffai/conv2d_block/mxfp4"
    );
    conv2d_bench_fmt!(
        bench_nvfp4,
        mt_nvfp4_conv2d::kernel_ir_for,
        QFormat::Nvfp4,
        "ffai/conv2d_block/nvfp4"
    );
    conv2d_bench_fmt!(
        bench_fp4,
        mt_fp4_conv2d::kernel_ir_for,
        QFormat::Fp4,
        "ffai/conv2d_block/fp4"
    );
    conv2d_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_conv2d::kernel_ir_for,
        QFormat::Mxfp8E4,
        "ffai/conv2d_block/mxfp8_e4m3"
    );
    conv2d_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_conv2d::kernel_ir_for,
        QFormat::Mxfp8E5,
        "ffai/conv2d_block/mxfp8_e5m2"
    );
    conv2d_bench_fmt!(
        bench_fp8_e5m2,
        mt_fp8_e5m2_conv2d::kernel_ir_for,
        QFormat::Fp8E5m2,
        "ffai/conv2d_block/fp8_e5m2"
    );
    conv2d_bench_fmt!(
        bench_nvfp8,
        mt_nvfp8_conv2d::kernel_ir_for,
        QFormat::Nvfp8,
        "ffai/conv2d_block/nvfp8"
    );
    conv2d_bench_fmt!(
        bench_int8,
        mt_int8_conv2d::kernel_ir_for,
        QFormat::Int8,
        "ffai/conv2d_block/int8"
    );
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). C=1024 is a multiple of 32, so every filter row's
    // bit-stream is word-aligned for all widths.
    conv2d_bench_fmt!(
        bench_int2,
        mt_int2_conv2d::kernel_ir_for,
        QFormat::Int2,
        "ffai/conv2d_block/int2"
    );
    conv2d_bench_fmt!(
        bench_int3,
        mt_int3_conv2d::kernel_ir_for,
        QFormat::Int3,
        "ffai/conv2d_block/int3"
    );
    conv2d_bench_fmt!(
        bench_int4,
        mt_int4_conv2d::kernel_ir_for,
        QFormat::Int4,
        "ffai/conv2d_block/int4"
    );
    conv2d_bench_fmt!(
        bench_int5,
        mt_int5_conv2d::kernel_ir_for,
        QFormat::Int5,
        "ffai/conv2d_block/int5"
    );
    conv2d_bench_fmt!(
        bench_int6,
        mt_int6_conv2d::kernel_ir_for,
        QFormat::Int6,
        "ffai/conv2d_block/int6"
    );
    conv2d_bench_fmt!(
        bench_mxint2,
        mt_mxint2_conv2d::kernel_ir_for,
        QFormat::Mxint2,
        "ffai/conv2d_block/mxint2"
    );
    conv2d_bench_fmt!(
        bench_mxint3,
        mt_mxint3_conv2d::kernel_ir_for,
        QFormat::Mxint3,
        "ffai/conv2d_block/mxint3"
    );
    conv2d_bench_fmt!(
        bench_mxint4,
        mt_mxint4_conv2d::kernel_ir_for,
        QFormat::Mxint4,
        "ffai/conv2d_block/mxint4"
    );
    conv2d_bench_fmt!(
        bench_mxint5,
        mt_mxint5_conv2d::kernel_ir_for,
        QFormat::Mxint5,
        "ffai/conv2d_block/mxint5"
    );
    conv2d_bench_fmt!(
        bench_mxint6,
        mt_mxint6_conv2d::kernel_ir_for,
        QFormat::Mxint6,
        "ffai/conv2d_block/mxint6"
    );
    conv2d_bench_fmt!(
        bench_mxint8,
        mt_mxint8_conv2d::kernel_ir_for,
        QFormat::Mxint8,
        "ffai/conv2d_block/mxint8"
    );
    // FP16-scale twins: same element packing as their FP32-scaled twin, scale
    // read as a native half. fp8_e4m3_f16 reuses the nvfp8_f16 kernel.
    conv2d_bench_fmt!(
        bench_nvfp8_f16,
        mt_nvfp8_f16_conv2d::kernel_ir_for,
        QFormat::Nvfp8F16,
        "ffai/conv2d_block/nvfp8_f16"
    );
    conv2d_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_conv2d::kernel_ir_for,
        QFormat::Fp8E4m3F16,
        "ffai/conv2d_block/fp8_e4m3_f16"
    );
    conv2d_bench_fmt!(
        bench_fp4_f16,
        mt_fp4_f16_conv2d::kernel_ir_for,
        QFormat::Fp4F16,
        "ffai/conv2d_block/fp4_f16"
    );
    conv2d_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_conv2d::kernel_ir_for,
        QFormat::Fp8E5m2F16,
        "ffai/conv2d_block/fp8_e5m2_f16"
    );
    conv2d_bench_fmt!(
        bench_int2_f16,
        mt_int2_f16_conv2d::kernel_ir_for,
        QFormat::Int2F16,
        "ffai/conv2d_block/int2_f16"
    );
    conv2d_bench_fmt!(
        bench_int3_f16,
        mt_int3_f16_conv2d::kernel_ir_for,
        QFormat::Int3F16,
        "ffai/conv2d_block/int3_f16"
    );
    conv2d_bench_fmt!(
        bench_int4_f16,
        mt_int4_f16_conv2d::kernel_ir_for,
        QFormat::Int4F16,
        "ffai/conv2d_block/int4_f16"
    );
    conv2d_bench_fmt!(
        bench_int5_f16,
        mt_int5_f16_conv2d::kernel_ir_for,
        QFormat::Int5F16,
        "ffai/conv2d_block/int5_f16"
    );
    conv2d_bench_fmt!(
        bench_int6_f16,
        mt_int6_f16_conv2d::kernel_ir_for,
        QFormat::Int6F16,
        "ffai/conv2d_block/int6_f16"
    );
    conv2d_bench_fmt!(
        bench_int8_f16,
        mt_int8_f16_conv2d::kernel_ir_for,
        QFormat::Int8F16,
        "ffai/conv2d_block/int8_f16"
    );
}

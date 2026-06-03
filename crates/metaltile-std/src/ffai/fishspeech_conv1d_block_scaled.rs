//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight** dilated 1D convolution — the
//! weight-quantized counterpart of the FishSpeech ResBlock dilated conv
//! `conv1d_dilated` in `ffai/fishspeech_conv1d.rs`.
//!
//! The dense dilated conv projects every output element over the `in_ch × k`
//! receptive field of an NCL input with an OIK filter `[out_ch, in_ch, k]`,
//! reading padded input index `op*stride + kx*dilation` per tap. That filter is
//! a genuine quantizable parameter, so we flatten its contraction axis to
//! `C = in_ch * k` and quantize each output-channel row `[out_ch, C]` block-wise
//! along `C` in the spec formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8 + legacy
//! fp4/fp8 + symmetric int8 + symmetric sub-byte ints int2/3/4/5/6 & their E8M0
//! MXINT2/3/4/5/6 + 8-bit MXINT8).
//!
//! Filter-tap mapping: dense `w_idx = (oc * in_ch + ic) * k + kx = oc*C + col`
//! with `col = ic*k + kx`. So the dense filter load is replaced by a decode of
//! the packed code at the global flat filter element `g = oc*C + col`:
//!
//!   * 4-bit: `weight` is `[out_ch, C/8]` u32 — nibble at word `oc*(C/8)+col/8`,
//!     shift `(col%8)*4`.
//!   * 8-bit: `weight` is `[out_ch, C]` u8 — byte at `oc*C+col`.
//!   * sub-byte int2/3/5/6 (+ E8M0 MXINT): `weight` is a single flat LSB-first
//!     u32 bit-stream over the whole `[out_ch, C]` filter (`quant::format::pack`'s
//!     layout — *no* per-row word padding). Element `g = oc*C + col` lives at bit
//!     `g*bits`; a straddle-aware two-word read extracts the signed N-bit code.
//!
//! The decoded element is scaled by `scales[oc*(C/block_size) + col/block_size]`.
//!
//! **Dilation lives on the input tap, not the filter column.** The filter column
//! is the dense flat tap `col = ic*k + kx`; the dilation only widens the input
//! sample index `p = op*stride + kx*dilation`. So the decode mapping is identical
//! to the non-dilated audio conv; only the input gather guard changes.
//!
//! Only the filter is quantized; the per-channel `bias` stays `T` (tiny and
//! precision-sensitive). Geometry, dilation/stride/pad guards and accumulation
//! match the dense kernel **verbatim**: **Grid3D**, one thread per output element
//! (`program_id::<0>()` = flat `(n, oc, op)`); indices stay in the *padded* frame
//! so every value is a non-negative u32 and padding taps mask to zero.
//! `C = in_ch*k` is a multiple of `block_size` (4-bit `block_size` a multiple of
//! 8). fp8_e4m3 reuses the nvfp8 kernel. Codegen-only; correctness pinned by the
//! in-source `#[test_kernel]`s vs a `quant::format::dequant` oracle.

use metaltile::kernel;

/// mxfp4 quantized-weight dilated conv1d — E2M1 filter (block 32), E8M0 pow-2 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// nvfp4 quantized-weight dilated conv1d — E2M1 filter (block 16), E4M3 micro-scale × global.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// Legacy fp4 quantized-weight dilated conv1d — E2M1 filter (group 32), per-group FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// mxfp8 (E4M3) quantized-weight dilated conv1d — 8-bit filter (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// mxfp8 (E5M2) quantized-weight dilated conv1d — 8-bit filter (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// Legacy fp8 (E5M2) quantized-weight dilated conv1d — 8-bit filter (group 32), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// nvfp8 quantized-weight dilated conv1d — E4M3 filter (block 16), per-block FP32 scale.
/// Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

/// Symmetric int8 quantized-weight dilated conv1d — 8-bit codes (group 64), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
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

// ── FP16-scale twins of the FP32-scaled formats ─────────────────────────────
// Identical decode + geometry to their FP32 twins; only the scale axis is a
// native `half` (`Tensor<f16>`) read and cast to f32. The GPU half load matches
// the host `f16_scale_decode`, so each oracle still holds exactly. fp8_e4m3_f16
// reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).

/// nvfp8 (FP16 scale) quantized-weight dilated conv1d — E4M3 filter (block 16),
/// per-block FP16 scale. Clone of `mt_nvfp8_fishspeech_conv1d`, scale → f16. Also
/// serves **fp8_e4m3_f16** (same 8-bit-E4M3 + f16-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_fishspeech_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// fp4 (FP16 scale) quantized-weight dilated conv1d — E2M1 filter (group 32),
/// per-group FP16 scale. Clone of `mt_fp4_fishspeech_conv1d`, scale → f16.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_fishspeech_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = e2m1_decode(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// fp8 (E5M2, FP16 scale) quantized-weight dilated conv1d — 8-bit filter (group
/// 32), per-group FP16 scale. Clone of `mt_fp8_e5m2_fishspeech_conv1d`, scale → f16.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_fishspeech_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Symmetric int8 (FP16 scale) quantized-weight dilated conv1d — 8-bit codes
/// (group 64), per-group FP16 scale. Clone of `mt_int8_fishspeech_conv1d`,
/// scale → f16.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_fishspeech_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ── Symmetric sub-byte integer dilated conv1d (int2/3/4/5/6 + MXINT2..6) ─────
// The filter element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into a **single flat u32 bit-stream** over the whole `[out_ch, C]`
// filter (`C = in_ch*k`). `quant::format::pack` writes element `oc*C + col` at
// bit `(oc*C + col) * bits` of that global stream — there is *no* per-row word
// padding (unlike the per-row layout of the GEMV family, which only coincides
// with the global stream because its `in_dim*bits` is a multiple of 32). So the
// decode addresses the bit-stream by the **global filter element index**
// `g = oc*C + col` (the same flat index the dense `w_idx = oc*C + col` uses, and
// the same index `block_scaled_dequant`'s `int_dequant_*` kernels use): extract
// the low N bits with a straddle-aware two-word read, sign-extend in float
// (subtract 2^N when the top bit is set; `$half`/`$full` are 2^(N-1) / 2^N), then
// multiply by the block scale and the (masked) input tap. `$half`/`$full` are
// literals to keep the constexpr math out of the DSL shift operands. Geometry,
// dilation/stride/pad guards and accumulation match the dense kernel verbatim:
// Grid3D, one thread per output element. (For 4-bit and any width where `C*bits`
// is a multiple of 32 the per-row form `oc*(C*bits/32)` is byte-identical; the
// global form is correct unconditionally and matches `pack`'s flat layout.)

/// FP32-scaled symmetric int dilated conv1d (int2/3/4/5/6) — per-element
/// bit-stream filter code × per-group FP32 scale. `weight` is the flat
/// `bitstream_words(out_ch*C, bits)` u32 stream; element `oc*C + col` lives at
/// bit `(oc*C + col) * bits`.
macro_rules! int_conv1d_f32 {
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
            #[constexpr] in_len: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_len: u32,
            #[constexpr] k: u32,
            #[constexpr] stride: u32,
            #[constexpr] pad: u32,
            #[constexpr] dilation: u32,
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
            let w_row_elem = oc * c_dim; // global filter element base for this row
            let w_row_blk = oc * n_blocks;
            let mut acc = load(bias[oc]).cast::<f32>();
            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * in_len;
                let col_ic = ic * k;
                for kx in range(0u32, k, 1u32) {
                    let p = p0 + kx * dilation;
                    let valid = (p >= pad) & (p < pad + in_len);
                    let ix = select(valid, p - pad, 0u32);
                    let x = load(input[in_ic_base + ix]).cast::<f32>();
                    let x_m = select(valid, x, 0.0f32);
                    let col = col_ic + kx;
                    // Global flat bit offset into the whole-filter stream.
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
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[w_row_blk + col / block_size]);
                    let wt = val * scale;
                    acc = acc + x_m * wt;
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_conv1d_f32!(mt_int2_fishspeech_conv1d, 2u32, 2u32, 4.0f32);
int_conv1d_f32!(mt_int3_fishspeech_conv1d, 3u32, 4u32, 8.0f32);
int_conv1d_f32!(mt_int4_fishspeech_conv1d, 4u32, 8u32, 16.0f32);
int_conv1d_f32!(mt_int5_fishspeech_conv1d, 5u32, 16u32, 32.0f32);
int_conv1d_f32!(mt_int6_fishspeech_conv1d, 6u32, 32u32, 64.0f32);

/// FP16-scaled symmetric int dilated conv1d (int2/3/4/5/6) — clone of
/// `int_conv1d_f32` with the per-group scale read from a native `half`
/// (`Tensor<f16>`) and cast to f32. Same straddle-aware global-stream decode;
/// only the scale axis differs.
macro_rules! int_conv1d_f16 {
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
            #[constexpr] in_len: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_len: u32,
            #[constexpr] k: u32,
            #[constexpr] stride: u32,
            #[constexpr] pad: u32,
            #[constexpr] dilation: u32,
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
            let w_row_elem = oc * c_dim; // global filter element base for this row
            let w_row_blk = oc * n_blocks;
            let mut acc = load(bias[oc]).cast::<f32>();
            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * in_len;
                let col_ic = ic * k;
                for kx in range(0u32, k, 1u32) {
                    let p = p0 + kx * dilation;
                    let valid = (p >= pad) & (p < pad + in_len);
                    let ix = select(valid, p - pad, 0u32);
                    let x = load(input[in_ic_base + ix]).cast::<f32>();
                    let x_m = select(valid, x, 0.0f32);
                    let col = col_ic + kx;
                    // Global flat bit offset into the whole-filter stream.
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
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                    let wt = val * scale;
                    acc = acc + x_m * wt;
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_conv1d_f16!(mt_int2_f16_fishspeech_conv1d, 2u32, 2u32, 4.0f32);
int_conv1d_f16!(mt_int3_f16_fishspeech_conv1d, 3u32, 4u32, 8.0f32);
int_conv1d_f16!(mt_int4_f16_fishspeech_conv1d, 4u32, 8u32, 16.0f32);
int_conv1d_f16!(mt_int5_f16_fishspeech_conv1d, 5u32, 16u32, 32.0f32);
int_conv1d_f16!(mt_int6_f16_fishspeech_conv1d, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int dilated conv1d (MXINT2/3/4/5/6) — per-element
/// bit-stream filter code × pow-2 (E8M0) block scale `2^(bits-127)`. Same
/// straddle-aware global-stream decode as `int_conv1d_f32`; only the scale axis
/// differs (one u8 exponent per block instead of a raw f32).
macro_rules! int_conv1d_e8m0 {
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
            #[constexpr] in_len: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_len: u32,
            #[constexpr] k: u32,
            #[constexpr] stride: u32,
            #[constexpr] pad: u32,
            #[constexpr] dilation: u32,
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
            let w_row_elem = oc * c_dim; // global filter element base for this row
            let w_row_blk = oc * n_blocks;
            let mut acc = load(bias[oc]).cast::<f32>();
            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * in_len;
                let col_ic = ic * k;
                for kx in range(0u32, k, 1u32) {
                    let p = p0 + kx * dilation;
                    let valid = (p >= pad) & (p < pad + in_len);
                    let ix = select(valid, p - pad, 0u32);
                    let x = load(input[in_ic_base + ix]).cast::<f32>();
                    let x_m = select(valid, x, 0.0f32);
                    let col = col_ic + kx;
                    // Global flat bit offset into the whole-filter stream.
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
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    let wt = val * scale;
                    acc = acc + x_m * wt;
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_conv1d_e8m0!(mt_mxint2_fishspeech_conv1d, 2u32, 2u32, 4.0f32);
int_conv1d_e8m0!(mt_mxint3_fishspeech_conv1d, 3u32, 4u32, 8.0f32);
int_conv1d_e8m0!(mt_mxint4_fishspeech_conv1d, 4u32, 8u32, 16.0f32);
int_conv1d_e8m0!(mt_mxint5_fishspeech_conv1d, 5u32, 16u32, 32.0f32);
int_conv1d_e8m0!(mt_mxint6_fishspeech_conv1d, 6u32, 32u32, 64.0f32);

/// MXINT8 quantized-weight dilated conv1d — 8-bit symmetric codes (byte layout,
/// block 32), E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the
/// 8-bit float formats (one byte per code at `oc*C + col`); decode is
/// `int8_decode → val · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_fishspeech_conv1d<T>(
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
    #[constexpr] dilation: u32,
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
            let p = p0 + kx * dilation;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;
            let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
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

    /// Dilated 1D conv oracle (NCL input, OIK filter) over a *dequantized* filter.
    /// Tap `kx` reads padded input index `op*stride + kx*dilation`; padding taps
    /// zero. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_dilated(
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
        dilation: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx * dilation;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            let in_idx = (n * in_ch + ic) * in_len + ix;
                            // Quantized filter flattens [out_ch, in_ch, k] to
                            // [out_ch, C] with C = in_ch*k, col = ic*k + kx.
                            // Dilation lives on the input tap, not the column.
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
        dilation: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
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
        let expected = naive_dilated(
            &input, &wdq, &bias, batch, in_ch, in_len, out_ch, k, stride, pad, dilation,
        );
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as a flat u32 bit-stream.
        // FP32 scales bind as f32; FP16 scales as native half; E8M0/E4M3 scales as
        // one byte. Both axes are driven off the format so new quant formats pick
        // up the right buffers.
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
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
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

    // in_ch=8, k=8 → C=64 (÷ 16/32/64); out_ch=8, in_len=32, stride 1.
    // MRF ResBlock dilation: pad = dilation = 3 (same padding), k=8.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxfp4_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxfp4,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp4_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Nvfp4,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp4_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Fp4,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxfp8_e4m3_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxfp8_e5m2_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp8_e5m2_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Nvfp8,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int8_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int8,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). C = in_ch*k = 64 satisfies
    // `C*bits % 32 == 0` for every width, so the per-row stream is word-aligned
    // (here the global-element decode is byte-identical to a per-row form). The
    // kernel and oracle share the codec, so the GPU output tracks the
    // dequant-then-conv reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int2_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int2,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int3_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int3,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int4_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int4,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int5_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int5,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int6_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int6,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint2_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxint2,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint3_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxint3,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint4_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxint4,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint5_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxint5,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint6_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxint6,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint8_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Mxint8,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + f16-scale shape); the rest are
    // per-element clones of their FP32 twin with the scale read as a native half.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp4_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Fp4F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp8_e5m2_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int2_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int2F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int3_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int3F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int4_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int4F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int5_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int5F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int6_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int6F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_fishspeech_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int8_f16_fishspeech_conv1d::kernel_ir_for(dt),
            QFormat::Int8F16,
            1,
            8,
            32,
            8,
            8,
            1,
            3,
            3,
            dt,
        )
    }
}

/// Decode-shape benches: realistic FishSpeech ResBlock dilated conv (in_ch=128,
/// out_ch=128, k=8 → C=1024 divisible by all block sizes; in_len=1024, stride 1,
/// dilation 3, same pad=dilation*(k-1)/2 truncated; here pad=3). Grid3D, one
/// thread per output element.
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
        dilation: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let c_dim = in_ch * k;
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble packs
        // + int2/3/5/6 tight bit-streams) tight-bit-packs into a flat u32 stream
        // over the whole `[out_ch, C]` filter (`bitstream_words` collapses to the
        // old `n/8` for 4-bit, so no regression). Scale axis driven off the format.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (out_ch * c_dim, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(out_ch * c_dim, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
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
            .constexpr("dilation", dilation as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            // 2 * Co * Lo * C (groups=1, C = in_ch*k; dilated conv same MAC count as dense)
            .flops(2 * out_ch as u64 * out_len as u64 * c_dim as u64)
            .with_shape_label(format!("{} oc={out_ch} lo={out_len} c={c_dim}", fmt.name()))
    }

    macro_rules! conv1d_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr, $name:literal) => {
            #[bench(name = $name, dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                conv1d_bench($kernel(dt), $fmt, 1, 128, 1024, 128, 8, 1, 3, 3, dt)
            }
        };
    }
    conv1d_bench_fmt!(
        bench_mxfp4,
        mt_mxfp4_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxfp4,
        "ffai/fishspeech_conv1d_block/mxfp4"
    );
    conv1d_bench_fmt!(
        bench_nvfp4,
        mt_nvfp4_fishspeech_conv1d::kernel_ir_for,
        QFormat::Nvfp4,
        "ffai/fishspeech_conv1d_block/nvfp4"
    );
    conv1d_bench_fmt!(
        bench_fp4,
        mt_fp4_fishspeech_conv1d::kernel_ir_for,
        QFormat::Fp4,
        "ffai/fishspeech_conv1d_block/fp4"
    );
    conv1d_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxfp8E4,
        "ffai/fishspeech_conv1d_block/mxfp8_e4m3"
    );
    conv1d_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxfp8E5,
        "ffai/fishspeech_conv1d_block/mxfp8_e5m2"
    );
    conv1d_bench_fmt!(
        bench_fp8_e5m2,
        mt_fp8_e5m2_fishspeech_conv1d::kernel_ir_for,
        QFormat::Fp8E5m2,
        "ffai/fishspeech_conv1d_block/fp8_e5m2"
    );
    conv1d_bench_fmt!(
        bench_nvfp8,
        mt_nvfp8_fishspeech_conv1d::kernel_ir_for,
        QFormat::Nvfp8,
        "ffai/fishspeech_conv1d_block/nvfp8"
    );
    conv1d_bench_fmt!(
        bench_int8,
        mt_int8_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int8,
        "ffai/fishspeech_conv1d_block/int8"
    );
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    conv1d_bench_fmt!(
        bench_int2,
        mt_int2_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int2,
        "ffai/fishspeech_conv1d_block/int2"
    );
    conv1d_bench_fmt!(
        bench_int3,
        mt_int3_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int3,
        "ffai/fishspeech_conv1d_block/int3"
    );
    conv1d_bench_fmt!(
        bench_int4,
        mt_int4_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int4,
        "ffai/fishspeech_conv1d_block/int4"
    );
    conv1d_bench_fmt!(
        bench_int5,
        mt_int5_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int5,
        "ffai/fishspeech_conv1d_block/int5"
    );
    conv1d_bench_fmt!(
        bench_int6,
        mt_int6_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int6,
        "ffai/fishspeech_conv1d_block/int6"
    );
    conv1d_bench_fmt!(
        bench_mxint2,
        mt_mxint2_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxint2,
        "ffai/fishspeech_conv1d_block/mxint2"
    );
    conv1d_bench_fmt!(
        bench_mxint3,
        mt_mxint3_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxint3,
        "ffai/fishspeech_conv1d_block/mxint3"
    );
    conv1d_bench_fmt!(
        bench_mxint4,
        mt_mxint4_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxint4,
        "ffai/fishspeech_conv1d_block/mxint4"
    );
    conv1d_bench_fmt!(
        bench_mxint5,
        mt_mxint5_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxint5,
        "ffai/fishspeech_conv1d_block/mxint5"
    );
    conv1d_bench_fmt!(
        bench_mxint6,
        mt_mxint6_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxint6,
        "ffai/fishspeech_conv1d_block/mxint6"
    );
    conv1d_bench_fmt!(
        bench_mxint8,
        mt_mxint8_fishspeech_conv1d::kernel_ir_for,
        QFormat::Mxint8,
        "ffai/fishspeech_conv1d_block/mxint8"
    );
    // FP16-scale twins. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel.
    conv1d_bench_fmt!(
        bench_nvfp8_f16,
        mt_nvfp8_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Nvfp8F16,
        "ffai/fishspeech_conv1d_block/nvfp8_f16"
    );
    conv1d_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Fp8E4m3F16,
        "ffai/fishspeech_conv1d_block/fp8_e4m3_f16"
    );
    conv1d_bench_fmt!(
        bench_fp4_f16,
        mt_fp4_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Fp4F16,
        "ffai/fishspeech_conv1d_block/fp4_f16"
    );
    conv1d_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Fp8E5m2F16,
        "ffai/fishspeech_conv1d_block/fp8_e5m2_f16"
    );
    conv1d_bench_fmt!(
        bench_int2_f16,
        mt_int2_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int2F16,
        "ffai/fishspeech_conv1d_block/int2_f16"
    );
    conv1d_bench_fmt!(
        bench_int3_f16,
        mt_int3_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int3F16,
        "ffai/fishspeech_conv1d_block/int3_f16"
    );
    conv1d_bench_fmt!(
        bench_int4_f16,
        mt_int4_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int4F16,
        "ffai/fishspeech_conv1d_block/int4_f16"
    );
    conv1d_bench_fmt!(
        bench_int5_f16,
        mt_int5_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int5F16,
        "ffai/fishspeech_conv1d_block/int5_f16"
    );
    conv1d_bench_fmt!(
        bench_int6_f16,
        mt_int6_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int6F16,
        "ffai/fishspeech_conv1d_block/int6_f16"
    );
    conv1d_bench_fmt!(
        bench_int8_f16,
        mt_int8_f16_fishspeech_conv1d::kernel_ir_for,
        QFormat::Int8F16,
        "ffai/fishspeech_conv1d_block/int8_f16"
    );
}

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
            let wt = mt_decode_e2m1(nib) * scale;
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
                mt_decode_e4m3(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global;
            let wt = mt_decode_e2m1(nib) * scale;
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
            let wt = mt_decode_e2m1(nib) * scale;
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
            let elem = mt_decode_e4m3(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_e5m2(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_e5m2(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_e4m3(load(weight[w_row + col]).cast::<u32>());
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
            let elem = mt_decode_int8(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ── Symmetric sub-byte integer conv1d (int2/3/4/5/6 + MXINT2..6) ─────────────
// The filter element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words. **Layout note:** `quant::format::pack` writes every
// element at bit `global_elem · bits` of ONE flat bit-stream over the whole
// `[out_ch, C]` filter (`global_elem = oc·C + col`, `col = ic·k + kx`) — it is
// *not* a per-row word-aligned stream. So the bit offset is computed from the
// **global** element index, mirroring `pack` exactly (this is also why the 4-bit
// nibble path's `oc·(C/8) + col/8` works only because C is a multiple of 8). The
// decode then matches `block_scaled_dequant`'s proven `int_dequant_*` macros:
// extract the low N bits with a straddle-aware two-word read, sign-extend in
// float (`$half`/`$full` = 2^(N-1)/2^N), and multiply by the per-block scale. The
// block scale still indexes `[out_ch, C]` as `w_row_blk + col/block_size` (same
// as int8). Geometry (Grid3D, one thread per output element) is unchanged.

/// FP32-scaled symmetric int conv1d (int2/3/4/5/6): per-element bit-stream filter
/// code × per-group FP32 scale, accumulated over the `in_ch·k` receptive field.
/// `g_off = oc*c_dim + col` is the global element index into the flat bit-stream.
macro_rules! int_audio_conv1d_f32 {
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
            let w_row_elem = oc * c_dim; // global element base of this output row
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
                    // Flat global bit-stream: element `oc*C + col` at bit `·$bits`.
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
                    acc = acc + x_m * (elem * scale);
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_audio_conv1d_f32!(mt_int2_audio_conv1d, 2u32, 2u32, 4.0f32);
int_audio_conv1d_f32!(mt_int3_audio_conv1d, 3u32, 4u32, 8.0f32);
int_audio_conv1d_f32!(mt_int4_audio_conv1d, 4u32, 8u32, 16.0f32);
int_audio_conv1d_f32!(mt_int5_audio_conv1d, 5u32, 16u32, 32.0f32);
int_audio_conv1d_f32!(mt_int6_audio_conv1d, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int conv1d (MXINT2/3/4/5/6): per-element bit-stream
/// filter code × pow-2 (E8M0) block scale `2^(bits-127)`, over the `in_ch·k`
/// receptive field. Same flat-global-bit-stream decode as `int_audio_conv1d_f32`;
/// only the scale axis differs (one u8 exponent per block instead of a raw f32).
macro_rules! int_audio_conv1d_e8m0 {
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
            let w_row_elem = oc * c_dim; // global element base of this output row
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
                    // Flat global bit-stream: element `oc*C + col` at bit `·$bits`.
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
                    acc = acc + x_m * (elem * scale);
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_audio_conv1d_e8m0!(mt_mxint2_audio_conv1d, 2u32, 2u32, 4.0f32);
int_audio_conv1d_e8m0!(mt_mxint3_audio_conv1d, 3u32, 4u32, 8.0f32);
int_audio_conv1d_e8m0!(mt_mxint4_audio_conv1d, 4u32, 8u32, 16.0f32);
int_audio_conv1d_e8m0!(mt_mxint5_audio_conv1d, 5u32, 16u32, 32.0f32);
int_audio_conv1d_e8m0!(mt_mxint6_audio_conv1d, 6u32, 32u32, 64.0f32);

/// MXINT8 quantized-weight conv1d — 8-bit symmetric codes (byte layout, block
/// 32), E8M0 pow-2 block scale `2^(bits-127)`. Byte-addressed like the int8 /
/// mxfp8 kernels (one code per byte), decode is `mt_decode_int8 → elem · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_audio_conv1d<T>(
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
            let elem = mt_decode_int8(load(weight[w_row + col]).cast::<u32>());
            let scale = exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ── FP16-scale twins ─────────────────────────────────────────────────────────
// Near-clones of the FP32-scaled kernels above for the same element formats:
// only the per-block scale changes from a raw `Tensor<f32>` to a native
// `Tensor<f16>` read + `.cast::<f32>()`. Element decode (E2M1 / E4M3 / E5M2 /
// sub-byte int bit-stream + sign-extend / byte-layout int8), weight indexing,
// dispatch geometry and accumulation are IDENTICAL to the FP32 twin. The GPU
// half load matches the host `f16_scale_decode`, so the dequant-then-conv oracle
// still holds exactly. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel (same
// 8-bit-E4M3 + FP16-scale shape), exactly as `fp8_e4m3` reuses `nvfp8` above.

/// nvfp8 (FP16 scale) quantized-weight conv1d — E4M3 filter (block 16), per-block
/// FP16 scale. Also serves **fp8_e4m3_f16** (same 8-bit-E4M3 + FP16-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_audio_conv1d<T>(
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
            let elem = mt_decode_e4m3(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp4 (FP16 scale) quantized-weight conv1d — E2M1 filter (group 32),
/// per-group FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_audio_conv1d<T>(
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
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = mt_decode_e2m1(nib) * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp8 (E5M2, FP16 scale) quantized-weight conv1d — 8-bit filter (group
/// 32), per-group FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_audio_conv1d<T>(
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
            let elem = mt_decode_e5m2(load(weight[w_row + col]).cast::<u32>());
            let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
            let wt = elem * scale;
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// FP16-scaled symmetric int conv1d (int2/3/4/5/6): per-element bit-stream filter
/// code × per-group FP16 scale, accumulated over the `in_ch·k` receptive field.
/// Identical flat-global-bit-stream decode as `int_audio_conv1d_f32`; only the
/// scale axis differs (a native `Tensor<f16>` read + `.cast::<f32>()`).
/// `g_off = oc*c_dim + col` is the global element index into the flat bit-stream.
macro_rules! int_audio_conv1d_f16 {
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
            let w_row_elem = oc * c_dim; // global element base of this output row
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
                    // Flat global bit-stream: element `oc*C + col` at bit `·$bits`.
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
                    acc = acc + x_m * (elem * scale);
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_audio_conv1d_f16!(mt_int2_f16_audio_conv1d, 2u32, 2u32, 4.0f32);
int_audio_conv1d_f16!(mt_int3_f16_audio_conv1d, 3u32, 4u32, 8.0f32);
int_audio_conv1d_f16!(mt_int4_f16_audio_conv1d, 4u32, 8u32, 16.0f32);
int_audio_conv1d_f16!(mt_int5_f16_audio_conv1d, 5u32, 16u32, 32.0f32);
int_audio_conv1d_f16!(mt_int6_f16_audio_conv1d, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) quantized-weight conv1d — 8-bit symmetric codes (byte
/// layout, group 64), per-group FP16 scale. Byte-addressed like the int8 kernel;
/// decode is `mt_decode_int8 → elem · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_audio_conv1d<T>(
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
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new integer / fp16 formats pick
        // up the right buffer types.
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

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). The flattened contraction
    // C = in_ch*k = 64 satisfies `C*bits % 32 == 0` for every width AND is a
    // multiple of every block/group size, so the per-output-channel bit-stream
    // is word-aligned within the flat global stream. Kernel + oracle share the
    // codec, so the GPU output tracks the dequant-then-conv reference to float
    // precision regardless of how coarse the quantization is.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int2_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int2,
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
    fn test_int3_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int3_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int3,
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
    fn test_int4_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int4_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int4,
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
    fn test_int5_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int5_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int5,
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
    fn test_int6_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int6_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int6,
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
    fn test_mxint2_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint2_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxint2,
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
    fn test_mxint3_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint3_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxint3,
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
    fn test_mxint4_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint4_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxint4,
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
    fn test_mxint5_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint5_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxint5,
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
    fn test_mxint6_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint6_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxint6,
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
    fn test_mxint8_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_mxint8_audio_conv1d::kernel_ir_for(dt),
            QFormat::Mxint8,
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

    // FP16-scale twins of the FP32-scaled formats. Same geometry / dims; only the
    // scale tensor binds as f16 (driven off `scale_kind()` in `conv1d_setup`).
    // `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel (8-bit E4M3 + FP16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
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
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_nvfp8_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
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
    fn test_fp4_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp4_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Fp4F16,
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
    fn test_fp8_e5m2_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_fp8_e5m2_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
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
    fn test_int2_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int2_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int2F16,
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
    fn test_int3_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int3_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int3F16,
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
    fn test_int4_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int4_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int4F16,
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
    fn test_int5_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int5_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int5F16,
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
    fn test_int6_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int6_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int6F16,
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
    fn test_int8_f16_audio_conv1d(dt: DType) -> TestSetup {
        conv1d_setup(
            mt_int8_f16_audio_conv1d::kernel_ir_for(dt),
            QFormat::Int8F16,
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
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words
        // (with a guard word for straddling 3/5/6-bit reads).
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
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                conv1d_bench($kernel(dt), $fmt, 1, 128, 1024, 128, 8, 2, 1, dt)
            }
        };
    }
    conv1d_bench_fmt!(bench_mxfp4, mt_mxfp4_audio_conv1d::kernel_ir_for, QFormat::Mxfp4);
    conv1d_bench_fmt!(bench_nvfp4, mt_nvfp4_audio_conv1d::kernel_ir_for, QFormat::Nvfp4);
    conv1d_bench_fmt!(bench_fp4, mt_fp4_audio_conv1d::kernel_ir_for, QFormat::Fp4);
    conv1d_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_audio_conv1d::kernel_ir_for,
        QFormat::Mxfp8E4
    );
    conv1d_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_audio_conv1d::kernel_ir_for,
        QFormat::Mxfp8E5
    );
    conv1d_bench_fmt!(bench_fp8_e5m2, mt_fp8_e5m2_audio_conv1d::kernel_ir_for, QFormat::Fp8E5m2);
    conv1d_bench_fmt!(bench_nvfp8, mt_nvfp8_audio_conv1d::kernel_ir_for, QFormat::Nvfp8);
    conv1d_bench_fmt!(bench_int8, mt_int8_audio_conv1d::kernel_ir_for, QFormat::Int8);
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    conv1d_bench_fmt!(bench_int2, mt_int2_audio_conv1d::kernel_ir_for, QFormat::Int2);
    conv1d_bench_fmt!(bench_int3, mt_int3_audio_conv1d::kernel_ir_for, QFormat::Int3);
    conv1d_bench_fmt!(bench_int4, mt_int4_audio_conv1d::kernel_ir_for, QFormat::Int4);
    conv1d_bench_fmt!(bench_int5, mt_int5_audio_conv1d::kernel_ir_for, QFormat::Int5);
    conv1d_bench_fmt!(bench_int6, mt_int6_audio_conv1d::kernel_ir_for, QFormat::Int6);
    conv1d_bench_fmt!(bench_mxint2, mt_mxint2_audio_conv1d::kernel_ir_for, QFormat::Mxint2);
    conv1d_bench_fmt!(bench_mxint3, mt_mxint3_audio_conv1d::kernel_ir_for, QFormat::Mxint3);
    conv1d_bench_fmt!(bench_mxint4, mt_mxint4_audio_conv1d::kernel_ir_for, QFormat::Mxint4);
    conv1d_bench_fmt!(bench_mxint5, mt_mxint5_audio_conv1d::kernel_ir_for, QFormat::Mxint5);
    conv1d_bench_fmt!(bench_mxint6, mt_mxint6_audio_conv1d::kernel_ir_for, QFormat::Mxint6);
    conv1d_bench_fmt!(bench_mxint8, mt_mxint8_audio_conv1d::kernel_ir_for, QFormat::Mxint8);
    // FP16-scale twins (f16 group/block scale). `fp8_e4m3_f16` reuses nvfp8_f16.
    conv1d_bench_fmt!(bench_nvfp8_f16, mt_nvfp8_f16_audio_conv1d::kernel_ir_for, QFormat::Nvfp8F16);
    conv1d_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_audio_conv1d::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    conv1d_bench_fmt!(bench_fp4_f16, mt_fp4_f16_audio_conv1d::kernel_ir_for, QFormat::Fp4F16);
    conv1d_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_audio_conv1d::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    conv1d_bench_fmt!(bench_int2_f16, mt_int2_f16_audio_conv1d::kernel_ir_for, QFormat::Int2F16);
    conv1d_bench_fmt!(bench_int3_f16, mt_int3_f16_audio_conv1d::kernel_ir_for, QFormat::Int3F16);
    conv1d_bench_fmt!(bench_int4_f16, mt_int4_f16_audio_conv1d::kernel_ir_for, QFormat::Int4F16);
    conv1d_bench_fmt!(bench_int5_f16, mt_int5_f16_audio_conv1d::kernel_ir_for, QFormat::Int5F16);
    conv1d_bench_fmt!(bench_int6_f16, mt_int6_f16_audio_conv1d::kernel_ir_for, QFormat::Int6F16);
    conv1d_bench_fmt!(bench_int8_f16, mt_int8_f16_audio_conv1d::kernel_ir_for, QFormat::Int8F16);
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Consolidated 2D convolution — see `../PLAN.md` for the full migration plan.
//!
//! Four `#[kernel]` / `#[kernel(variants(...))]` blocks cover all standard 2D
//! conv kernels in this crate:
//!
//! * **Dense direct** — `conv2d` (patch14/16 variants), `conv2d_generic`,
//!   `conv2d_grouped` (dilation + groups). Ported from `conv2d.rs`.
//!
//! * **Block-scaled direct conv2d** — `mt_conv2d_quant` (19 formats, same
//!   FMT table as `mt_conv1d_quant`). The body is the 2D analogue of the
//!   1D quant body: `c_dim = in_ch·kh·kw`, loops are `for ic { for ky { for
//!   kx { } } }`, `col = col_ic + ky·kw + kx`. No DILATED cross axis —
//!   there is no dilated block-scaled direct conv2d variant.
//!
//! * **MMA-tiled implicit-GEMM** — `conv2d_mma` (stride=1, pad=0, tpg=128,
//!   32×32 output tile). Ported from `conv2d_mma.rs`.
//!
//! * **Dense depthwise** — `depthwise_conv2d` (NCHW) and
//!   `depthwise_conv2d_nhwc` (channel-last). Ported from their respective
//!   files.
//!
//! * **Block-scaled depthwise** — `mt_dw_conv2d_quant` (19 formats). The
//!   depthwise sub-byte decode uses a GLOBAL flat bit offset
//!   `(c·C + col)·bits` (not per-row word-aligned) because `C = k*k` is
//!   not always a multiple of 32. All other decode / scale logic is
//!   identical to `mt_conv2d_quant`.
//!
//! ## Format map (block-scaled only)
//!
//! | FMT | Format         | WT  | ST  |
//! |----:|:---------------|:----|:----|
//! |   0 | mxfp4          | u32 | u8  |
//! |   1 | nvfp4          | u32 | u8  |
//! | 2-6 | mxint{2..6}    | u32 | u8  |
//! |   7 | fp4            | u32 | f32 |
//! |8-12 | int{2..6}      | u32 | f32 |
//! |  13 | mxfp8\_e4m3    | u8  | u8  |
//! |  14 | mxfp8\_e5m2    | u8  | u8  |
//! |  15 | mxint8         | u8  | u8  |
//! |  16 | fp8\_e5m2      | u8  | f32 |
//! |  17 | nvfp8          | u8  | f32 |
//! |  18 | int8           | u8  | f32 |

use metaltile::kernel;

// ─── § Dense direct ───────────────────────────────────────────────────────────

#[kernel(variants(
    KH = [14u32, 16u32],
    KW = [14u32, 16u32],
    SH = [14u32, 16u32],
    SW = [14u32, 16u32],
    suffix = "patch{KH}"
))]
pub fn conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
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
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;
    let ph0 = oh * SH;
    let pw0 = ow * SW;
    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;
    let w_in_stride = KH * KW;
    let w_oc_stride = in_ch * w_in_stride;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let w_ic_base = oc * w_oc_stride + ic * w_in_stride;
        for ky in range(0u32, KH, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, KW, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);
                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);
                let w_idx = w_ic_base + ky * KW + kx;
                let wt = load(weight[w_idx]).cast::<f32>();
                acc = acc + pix_m * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

#[kernel]
pub fn conv2d_generic<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
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
    let w_in_stride = kh * kw;
    let w_oc_stride = in_ch * w_in_stride;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let w_ic_base = oc * w_oc_stride + ic * w_in_stride;
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
                let w_idx = w_ic_base + ky * kw + kx;
                let wt = load(weight[w_idx]).cast::<f32>();
                acc = acc + pix_m * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv2d_grouped<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
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
    #[constexpr] dilation_h: u32,
    #[constexpr] dilation_w: u32,
    #[constexpr] icpg: u32,
    #[constexpr] ocpg: u32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;
    let group = oc / ocpg;
    let ic_base = group * icpg;
    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;
    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;
    let w_in_stride = kh * kw;
    let w_oc_stride = icpg * w_in_stride;
    let mut acc = load(bias[oc]).cast::<f32>();
    for wic in range(0u32, icpg, 1u32) {
        let real_ic = ic_base + wic;
        let in_ic_base = n * in_n_stride + real_ic * input_plane;
        let w_ic_base = oc * w_oc_stride + wic * w_in_stride;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky * dilation_h;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx * dilation_w;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);
                let in_idx = in_ic_base + ih * in_w + iw;
                let pix = load(input[in_idx]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);
                let w_idx = w_ic_base + ky * kw + kx;
                let wt = load(weight[w_idx]).cast::<f32>();
                acc = acc + pix_m * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ─── § Block-scaled direct conv2d ─────────────────────────────────────────────
// Same 19-format FMT table as mt_conv1d_quant. No DILATED cross axis (no
// dilated block-scaled direct conv2d variant exists). The body is the 2D
// analogue: c_dim = in_ch*kh*kw, loops for ic { for ky { for kx { } } },
// col = col_ic + ky*kw + kx, per-row word-base decode for sub-byte.

#[kernel(variants(
    (FMT,         BITS,  WT,  ST ) = [
        (mxfp4,      4u32, u32, u8 ),
        (nvfp4,      4u32, u32, u8 ),
        (mxint2,     2u32, u32, u8 ),
        (mxint3,     3u32, u32, u8 ),
        (mxint4,     4u32, u32, u8 ),
        (mxint5,     5u32, u32, u8 ),
        (mxint6,     6u32, u32, u8 ),
        (fp4,        4u32, u32, f32),
        (int2,       2u32, u32, f32),
        (int3,       3u32, u32, f32),
        (int4,       4u32, u32, f32),
        (int5,       5u32, u32, f32),
        (int6,       6u32, u32, f32),
        (mxfp8,      8u32, u8,  u8 ),
        (mxfp8_e5m2, 8u32, u8,  u8 ),
        (mxint8,     8u32, u8,  u8 ),
        (fp8_e5m2,   8u32, u8,  f32),
        (nvfp8,      8u32, u8,  f32),
        (int8,       8u32, u8,  f32),
    ],
    suffix = "{FMT}",
))]
/// Block-scaled quantized-weight 2D convolution — 19 formats. The body
/// dispatches on `FMT` to select the per-tap element and scale decode;
/// only the body branch matching the variant's FMT value is live.
#[allow(clippy::too_many_arguments)]
pub fn mt_conv2d_quant<T>(
    input: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
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
    #[constexpr(only_when = "FMT == 1u32")] global: f32,
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
    let c_dim = in_ch * kh * kw;
    let w_row_word_base = oc * (c_dim * BITS / 32u32);
    let w_row_blk = oc * (c_dim / block_size);
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
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
                let elem = if FMT <= 12u32 {
                    let bit_off = col * BITS;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let lo_bits = select(32u32 - bit_in_w >= BITS, BITS, 32u32 - bit_in_w);
                    let spill = BITS - lo_bits;
                    let w0 = load(weight[w_row_word_base + word_idx]);
                    let w1 = load(weight[w_row_word_base + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                    if FMT <= 1u32 || FMT == 7u32 {
                        mt_decode_e2m1(q)
                    } else {
                        let qf = q.cast::<f32>();
                        select(q >= half, qf - full, qf)
                    }
                } else {
                    let raw = load(weight[oc * c_dim + col]).cast::<u32>();
                    if FMT == 13u32 || FMT == 17u32 {
                        mt_decode_e4m3(raw)
                    } else if FMT == 14u32 || FMT == 16u32 {
                        mt_decode_e5m2(raw)
                    } else {
                        mt_decode_int8(raw)
                    }
                };
                let scale = if FMT <= 6u32 {
                    if FMT == 1u32 {
                        mt_decode_e4m3(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global
                    } else {
                        mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
                    }
                } else if FMT >= 13u32 && FMT <= 15u32 {
                    mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
                } else {
                    load(scales[w_row_blk + col / block_size])
                };
                acc = acc + pix_m * (elem * scale);
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ─── § MMA-tiled implicit-GEMM conv2d ─────────────────────────────────────────

/// MMA-tiled 2D convolution (stride=1, dilation=1, pad=0).
///
/// Grid `[out_ch/32, (batch*out_h*out_w)/32, 1]`, tpg = 128.
/// Each TG computes a 32×32 tile of `out[pixels, out_channels]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
) {
    let oc_tile = tgid_x;
    let px_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T);
    threadgroup_alloc("bs", 1152, T);
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    let kk = kh * kw;
    let total_k = in_ch * kk;
    let out_hw = out_h * out_w;
    let a_px_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_px = px_tile * 32u32 + a_px_row;
    let n_px = global_px / out_hw;
    let rem_px = global_px - n_px * out_hw;
    let oh_px = rem_px / out_w;
    let ow_px = rem_px - oh_px * out_w;
    let in_n_stride = in_ch * in_h * in_w;
    let px_in_base = n_px * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let w_oc_base = global_oc * total_k;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kk;
            let rem_kt = kt_safe - ic * kk;
            let ky = rem_kt / kw;
            let kx = rem_kt - ky * kw;
            let ih = oh_px + ky;
            let iw = ow_px + kx;
            let in_idx = px_in_base + ic * in_h * in_w + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_px_row * stride + a_k_base + i, val);
        }
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let w_idx = w_oc_base + kt_safe;
            let raw = load(weight[w_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    let out_px_base = px_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(out[(out_px_base + fm) * out_ch + out_oc_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_px_base + fm) * out_ch + out_oc_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    store(out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0], simdgroup_elem_load(c_f01, 0).cast::<T>());
    store(out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1], simdgroup_elem_load(c_f01, 1).cast::<T>());
    store(out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0], simdgroup_elem_load(c_f10, 0).cast::<T>());
    store(out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1], simdgroup_elem_load(c_f10, 1).cast::<T>());
    store(out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0], simdgroup_elem_load(c_f11, 0).cast::<T>());
    store(out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1], simdgroup_elem_load(c_f11, 1).cast::<T>());
}

// ─── § Dense depthwise ────────────────────────────────────────────────────────

#[kernel]
pub fn depthwise_conv2d<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
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
    let w_c_base = c * k * k;
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
            let wt = load(weight[w_c_base + ky * k + kx]).cast::<f32>();
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

#[kernel]
pub fn depthwise_conv2d_nhwc<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k_h: u32,
    #[constexpr] k_w: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
) {
    let idx = program_id::<0>();
    let c = idx % ch;
    let t1 = idx / ch;
    let ow = t1 % out_w;
    let t2 = t1 / out_w;
    let oh = t2 % out_h;
    let n = t2 / out_h;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let n_base = n * in_h * in_w;
    let w_c_base = c * k_h * k_w;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k_h, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k_w, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let in_idx = (n_base + ih * in_w + iw) * ch + c;
            let x = load(input[in_idx]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let wt = load(weight[w_c_base + ky * k_w + kx]).cast::<f32>();
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ─── § Block-scaled depthwise conv2d ──────────────────────────────────────────
// Same 19-format FMT table. Sub-byte (FMT 2-6, 8-12) uses the GLOBAL flat
// bit offset `(c·C + col)·BITS` (not per-row word-aligned) because C = k*k
// is not always a multiple of 32. The `w_row_elem = c * c_dim` declared
// outside the loops provides the flat element base for all FMT ≤ 12.
// Scale decode is identical to mt_conv2d_quant.

#[kernel(variants(
    (FMT,         BITS,  WT,  ST ) = [
        (mxfp4,      4u32, u32, u8 ),
        (nvfp4,      4u32, u32, u8 ),
        (mxint2,     2u32, u32, u8 ),
        (mxint3,     3u32, u32, u8 ),
        (mxint4,     4u32, u32, u8 ),
        (mxint5,     5u32, u32, u8 ),
        (mxint6,     6u32, u32, u8 ),
        (fp4,        4u32, u32, f32),
        (int2,       2u32, u32, f32),
        (int3,       3u32, u32, f32),
        (int4,       4u32, u32, f32),
        (int5,       5u32, u32, f32),
        (int6,       6u32, u32, f32),
        (mxfp8,      8u32, u8,  u8 ),
        (mxfp8_e5m2, 8u32, u8,  u8 ),
        (mxint8,     8u32, u8,  u8 ),
        (fp8_e5m2,   8u32, u8,  f32),
        (nvfp8,      8u32, u8,  f32),
        (int8,       8u32, u8,  f32),
    ],
    suffix = "{FMT}",
))]
/// Block-scaled quantized-weight depthwise 2D convolution — 19 formats.
/// Sub-byte formats use a GLOBAL flat bit offset (not per-row word-aligned)
/// because `C = k*k` need not be a multiple of 32.
#[allow(clippy::too_many_arguments)]
pub fn mt_dw_conv2d_quant<T>(
    input: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
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
    #[constexpr(only_when = "FMT == 1u32")] global: f32,
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
    let c_dim = k * k;
    // Global flat element base for this channel (sub-byte decode uses it;
    // for nibble formats it equals c*c_dim/8 which is the per-row pack base).
    let w_row_elem = c * c_dim;
    let w_row_blk = c * (c_dim / block_size);
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
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
            let elem = if FMT <= 12u32 {
                // Global flat bit offset: (c*C + col)*BITS.
                let bit_off = (w_row_elem + col) * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let lo_bits = select(32u32 - bit_in_w >= BITS, BITS, 32u32 - bit_in_w);
                let spill = BITS - lo_bits;
                let w0 = load(weight[word_idx]);
                let w1 = load(weight[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                if FMT <= 1u32 || FMT == 7u32 {
                    mt_decode_e2m1(q)
                } else {
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf)
                }
            } else {
                let raw = load(weight[w_row_elem + col]).cast::<u32>();
                if FMT == 13u32 || FMT == 17u32 {
                    mt_decode_e4m3(raw)
                } else if FMT == 14u32 || FMT == 16u32 {
                    mt_decode_e5m2(raw)
                } else {
                    mt_decode_int8(raw)
                }
            };
            let scale = if FMT <= 6u32 {
                if FMT == 1u32 {
                    mt_decode_e4m3(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global
                } else {
                    mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
                }
            } else if FMT >= 13u32 && FMT <= 15u32 {
                mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
            } else {
                load(scales[w_row_blk + col / block_size])
            };
            acc = acc + x_m * (elem * scale);
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ─── § Tests ──────────────────────────────────────────────────────────────────

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    fn out_dim(in_d: usize, k: usize, stride: usize, pad: usize, dilation: usize) -> usize {
        (in_d + 2 * pad - dilation * (k - 1) - 1) / stride + 1
    }

    // ── Dense direct oracles ─────────────────────────────────────────────────

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
        dilation_h: usize,
        dilation_w: usize,
        icpg: usize,
        ocpg: usize,
    ) -> Vec<f32> {
        let out_h = (in_h + 2 * pad_h - ((kh - 1) * dilation_h + 1)) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - ((kw - 1) * dilation_w + 1)) / stride_w + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_ch {
                let group = oc / ocpg;
                let ic_base = group * icpg;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[oc];
                        for wic in 0..icpg {
                            let real_ic = ic_base + wic;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ph = oh * stride_h + ky * dilation_h;
                                    let pw = ow * stride_w + kx * dilation_w;
                                    if ph < pad_h || ph >= pad_h + in_h || pw < pad_w || pw >= pad_w + in_w {
                                        continue;
                                    }
                                    let ih = ph - pad_h;
                                    let iw = pw - pad_w;
                                    let in_idx = ((n * in_ch + real_ic) * in_h + ih) * in_w + iw;
                                    let w_idx = ((oc * icpg + wic) * kh + ky) * kw + kx;
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

    #[allow(clippy::too_many_arguments)]
    fn conv2d_setup(
        kernel: Kernel,
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
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * kh * kw, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv2d(&input, &weight, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1, in_ch, out_ch);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
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
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_setup(
        batch: usize, in_ch: usize, in_h: usize, in_w: usize,
        out_ch: usize, kh: usize, kw: usize,
        stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize,
        dilation_h: usize, dilation_w: usize, groups: usize, dt: DType,
    ) -> TestSetup {
        let (icpg, ocpg) = (in_ch / groups, out_ch / groups);
        let eff_kh = (kh - 1) * dilation_h + 1;
        let eff_kw = (kw - 1) * dilation_w + 1;
        let out_h = (in_h + 2 * pad_h - eff_kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - eff_kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(out_ch * icpg * kh * kw, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv2d(&input, &weight, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, icpg, ocpg);
        TestSetup::new(conv2d_grouped::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
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
            .constexpr("dilation_h", dilation_h as u32)
            .constexpr("dilation_w", dilation_w as u32)
            .constexpr("icpg", icpg as u32)
            .constexpr("ocpg", ocpg as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_patch14(dt: DType) -> TestSetup {
        conv2d_setup(conv2d_patch14::kernel_ir_for(dt), 1, 3, 28, 42, 8, 14, 14, 14, 14, 0, 0, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_patch16(dt: DType) -> TestSetup {
        conv2d_setup(conv2d_patch16::kernel_ir_for(dt), 1, 3, 32, 48, 6, 16, 16, 16, 16, 0, 0, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_generic(dt: DType) -> TestSetup {
        conv2d_setup(conv2d_generic::kernel_ir_for(dt), 2, 4, 9, 11, 5, 3, 3, 1, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_grouped_depthwise(dt: DType) -> TestSetup {
        grouped_setup(2, 8, 12, 14, 8, 3, 3, 1, 1, 1, 1, 1, 1, 8, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_grouped_full(dt: DType) -> TestSetup {
        grouped_setup(1, 6, 20, 22, 8, 3, 3, 2, 2, 2, 2, 2, 2, 2, dt)
    }

    // ── Block-scaled direct conv2d setup ─────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn blockscaled_conv2d_setup(
        kernel: metaltile::core::ir::Kernel,
        fmt: crate::quant::format::QFormat,
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
        dt: metaltile::core::DType,
    ) -> TestSetup {
        use metaltile::core::ir::KernelMode;
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        let contraction = in_ch * kh * kw;
        let input_f: Vec<f32> = (0..(batch * in_ch * in_h * in_w)).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 6.0).collect();
        let bias_f: Vec<f32> = (0..out_ch).map(|i| ((i % 5) as f32 / 5.0 - 0.5) * 2.0).collect();
        let w_f: Vec<f32> = (0..(out_ch * contraction)).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 4.0).collect();
        let p = crate::quant::format::pack(fmt, &w_f, out_ch, contraction);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, contraction);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_conv2d(&input, &wdq, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1, in_ch, out_ch);
        let weight_dt = if fmt.element_bits() == 8 { metaltile::core::DType::U8 } else { metaltile::core::DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => metaltile::core::DType::F32,
            crate::quant::format::ScaleKind::F16 => metaltile::core::DType::F16,
            _ => metaltile::core::DType::U8,
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
        if matches!(fmt, crate::quant::format::QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // in_ch=4, kh=kw=4 → C=64; 8×8 image, stride 1, pad 1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_conv2d_quant_mxfp4(dt: DType) -> TestSetup {
        blockscaled_conv2d_setup(mt_conv2d_quant_mxfp4::kernel_ir_for(dt), crate::quant::format::QFormat::Mxfp4, 1, 4, 8, 8, 8, 4, 4, 1, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_conv2d_quant_nvfp4(dt: DType) -> TestSetup {
        blockscaled_conv2d_setup(mt_conv2d_quant_nvfp4::kernel_ir_for(dt), crate::quant::format::QFormat::Nvfp4, 1, 4, 8, 8, 8, 4, 4, 1, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_conv2d_quant_int8(dt: DType) -> TestSetup {
        blockscaled_conv2d_setup(mt_conv2d_quant_int8::kernel_ir_for(dt), crate::quant::format::QFormat::Int8, 1, 4, 8, 8, 8, 4, 4, 1, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_conv2d_quant_mxint4(dt: DType) -> TestSetup {
        blockscaled_conv2d_setup(mt_conv2d_quant_mxint4::kernel_ir_for(dt), crate::quant::format::QFormat::Mxint4, 1, 4, 8, 8, 8, 4, 4, 1, 1, 1, 1, dt)
    }

    // ── Dense depthwise tests ─────────────────────────────────────────────────

    fn naive_depthwise_conv2d(
        input: &[f32], weight: &[f32], bias: &[f32],
        batch: usize, ch: usize, in_h: usize, in_w: usize,
        k: usize, stride: usize, pad: usize, dilation: usize,
    ) -> Vec<f32> {
        let out_h = out_dim(in_h, k, stride, pad, dilation);
        let out_w = out_dim(in_w, k, stride, pad, dilation);
        let mut out = vec![0.0f32; batch * ch * out_h * out_w];
        for n in 0..batch {
            for c in 0..ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[c];
                        for ky in 0..k {
                            let ph = oh * stride + ky * dilation;
                            if ph < pad || ph >= pad + in_h { continue; }
                            let ih = ph - pad;
                            for kx in 0..k {
                                let pw = ow * stride + kx * dilation;
                                if pw < pad || pw >= pad + in_w { continue; }
                                let iw = pw - pad;
                                acc += input[((n * ch + c) * in_h + ih) * in_w + iw] * weight[(c * k + ky) * k + kx];
                            }
                        }
                        out[((n * ch + c) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn dw_setup(kernel: Kernel, batch: usize, ch: usize, in_h: usize, in_w: usize, k: usize, stride: usize, pad: usize, dilation: usize, dt: DType) -> TestSetup {
        let out_h = out_dim(in_h, k, stride, pad, dilation);
        let out_w = out_dim(in_w, k, stride, pad, dilation);
        let n_out = batch * ch * out_h * out_w;
        let input_f = ramp(batch * ch * in_h * in_w, 13, 6.0);
        let weight_f = ramp(ch * k * k, 11, 4.0);
        let bias_f = ramp(ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_depthwise_conv2d(&input, &weight, &bias, batch, ch, in_h, in_w, k, stride, pad, dilation);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32).constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32).constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32).constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_3x3_s1(dt: DType) -> TestSetup {
        dw_setup(depthwise_conv2d::kernel_ir_for(dt), 1, 8, 16, 16, 3, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_3x3_s2(dt: DType) -> TestSetup {
        dw_setup(depthwise_conv2d::kernel_ir_for(dt), 2, 6, 24, 24, 3, 2, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_patch(dt: DType) -> TestSetup {
        dw_setup(depthwise_conv2d::kernel_ir_for(dt), 1, 4, 28, 28, 14, 14, 0, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_dilated(dt: DType) -> TestSetup {
        dw_setup(depthwise_conv2d::kernel_ir_for(dt), 1, 5, 20, 20, 3, 1, 2, 2, dt)
    }

    fn naive_depthwise_conv2d_nhwc(
        input: &[f32], weight: &[f32], bias: &[f32],
        batch: usize, ch: usize, in_h: usize, in_w: usize,
        k_h: usize, k_w: usize, stride: usize, pad: usize, dilation: usize,
    ) -> Vec<f32> {
        let out_h = out_dim(in_h, k_h, stride, pad, dilation);
        let out_w = out_dim(in_w, k_w, stride, pad, dilation);
        let mut out = vec![0.0f32; batch * out_h * out_w * ch];
        for n in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for c in 0..ch {
                        let mut acc = bias[c];
                        for ky in 0..k_h {
                            let ph = oh * stride + ky * dilation;
                            if ph < pad || ph >= pad + in_h { continue; }
                            let ih = ph - pad;
                            for kx in 0..k_w {
                                let pw = ow * stride + kx * dilation;
                                if pw < pad || pw >= pad + in_w { continue; }
                                let iw = pw - pad;
                                acc += input[((n * in_h + ih) * in_w + iw) * ch + c] * weight[(c * k_h + ky) * k_w + kx];
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * ch + c] = acc;
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn nhwc_setup(kernel: Kernel, batch: usize, ch: usize, in_h: usize, in_w: usize, k_h: usize, k_w: usize, stride: usize, pad: usize, dilation: usize, dt: DType) -> TestSetup {
        let out_h = out_dim(in_h, k_h, stride, pad, dilation);
        let out_w = out_dim(in_w, k_w, stride, pad, dilation);
        let n_out = batch * out_h * out_w * ch;
        let input_f = ramp(batch * in_h * in_w * ch, 13, 6.0);
        let weight_f = ramp(ch * k_h * k_w, 11, 4.0);
        let bias_f = ramp(ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_depthwise_conv2d_nhwc(&input, &weight, &bias, batch, ch, in_h, in_w, k_h, k_w, stride, pad, dilation);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32).constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("k_h", k_h as u32).constexpr("k_w", k_w as u32)
            .constexpr("stride", stride as u32).constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_nhwc_3x3_s1(dt: DType) -> TestSetup {
        nhwc_setup(depthwise_conv2d_nhwc::kernel_ir_for(dt), 1, 8, 16, 16, 3, 3, 1, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_nhwc_3x3_s2(dt: DType) -> TestSetup {
        nhwc_setup(depthwise_conv2d_nhwc::kernel_ir_for(dt), 2, 6, 24, 24, 3, 3, 2, 1, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_nhwc_7x7(dt: DType) -> TestSetup {
        nhwc_setup(depthwise_conv2d_nhwc::kernel_ir_for(dt), 1, 5, 20, 20, 7, 7, 1, 3, 1, dt)
    }

    // ── MMA tests ────────────────────────────────────────────────────────────

    fn naive_conv2d_mma(input: &[f32], weight: &[f32], batch: usize, in_ch: usize, in_h: usize, in_w: usize, out_ch: usize, kh: usize, kw: usize) -> Vec<f32> {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let out_hw = out_h * out_w;
        let n_pixels = batch * out_hw;
        let mut out = vec![0.0f32; n_pixels * out_ch];
        for n in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let pixel = n * out_hw + oh * out_w + ow;
                    for oc in 0..out_ch {
                        let mut acc = 0.0f32;
                        for ic in 0..in_ch {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    acc += input[((n * in_ch + ic) * in_h + oh + ky) * in_w + ow + kx] * weight[((oc * in_ch + ic) * kh + ky) * kw + kx];
                                }
                            }
                        }
                        out[pixel * out_ch + oc] = acc;
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn mma_setup(batch: usize, in_ch: usize, in_h: usize, in_w: usize, out_ch: usize, kh: usize, kw: usize, dt: DType) -> TestSetup {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        assert_eq!(out_ch % 32, 0);
        assert_eq!(n_pixels % 32, 0);
        let n_out = n_pixels * out_ch;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 2.0);
        let weight_f = ramp(out_ch * in_ch * kh * kw, 11, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let expected = naive_conv2d_mma(&input, &weight, batch, in_ch, in_h, in_w, out_ch, kh, kw);
        TestSetup::new(conv2d_mma::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32).constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32).constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32).constexpr("kw", kw as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_ch / 32) as u32, (n_pixels / 32) as u32, 1, [128, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_mma_3x3(dt: DType) -> TestSetup { mma_setup(1, 4, 10, 10, 32, 3, 3, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_conv2d_mma_multi_tile(dt: DType) -> TestSetup { mma_setup(4, 4, 8, 8, 32, 1, 1, dt) }

    // ── Block-scaled depthwise tests ──────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn blockscaled_dw_setup(
        kernel: metaltile::core::ir::Kernel,
        fmt: crate::quant::format::QFormat,
        batch: usize, ch: usize, in_h: usize, in_w: usize,
        k: usize, stride: usize, pad: usize, dilation: usize,
        dt: metaltile::core::DType,
    ) -> TestSetup {
        use metaltile::core::ir::KernelMode;
        let out_h = out_dim(in_h, k, stride, pad, dilation);
        let out_w = out_dim(in_w, k, stride, pad, dilation);
        let n_out = batch * ch * out_h * out_w;
        let c_dim = k * k;
        let input_f: Vec<f32> = (0..(batch * ch * in_h * in_w)).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 6.0).collect();
        let bias_f: Vec<f32> = (0..ch).map(|i| ((i % 5) as f32 / 5.0 - 0.5) * 2.0).collect();
        let w_f: Vec<f32> = (0..(ch * c_dim)).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 4.0).collect();
        let p = crate::quant::format::pack(fmt, &w_f, ch, c_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, ch, c_dim);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = {
            let mut out = vec![0.0f32; batch * ch * out_h * out_w];
            for n in 0..batch {
                for c in 0..ch {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut acc = bias[c];
                            for ky in 0..k {
                                let ph = oh * stride + ky * dilation;
                                if ph < pad || ph >= pad + in_h { continue; }
                                let ih = ph - pad;
                                for kx in 0..k {
                                    let pw = ow * stride + kx * dilation;
                                    if pw < pad || pw >= pad + in_w { continue; }
                                    let iw = pw - pad;
                                    let col = ky * k + kx;
                                    acc += input[((n * ch + c) * in_h + ih) * in_w + iw] * wdq[c * c_dim + col];
                                }
                            }
                            out[((n * ch + c) * out_h + oh) * out_w + ow] = acc;
                        }
                    }
                }
            }
            out
        };
        let weight_dt = if fmt.element_bits() == 8 { metaltile::core::DType::U8 } else { metaltile::core::DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => metaltile::core::DType::F32,
            crate::quant::format::ScaleKind::F16 => metaltile::core::DType::F16,
            _ => metaltile::core::DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32).constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32).constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32).constexpr("dilation", dilation as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, crate::quant::format::QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // k=8 → C=64 so all block sizes (16/32/64) divide evenly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_dw_conv2d_quant_mxfp4(dt: DType) -> TestSetup {
        blockscaled_dw_setup(mt_dw_conv2d_quant_mxfp4::kernel_ir_for(dt), crate::quant::format::QFormat::Mxfp4, 1, 4, 16, 16, 8, 1, 0, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_dw_conv2d_quant_int8(dt: DType) -> TestSetup {
        blockscaled_dw_setup(mt_dw_conv2d_quant_int8::kernel_ir_for(dt), crate::quant::format::QFormat::Int8, 1, 4, 16, 16, 8, 1, 0, 1, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_dw_conv2d_quant_mxint4(dt: DType) -> TestSetup {
        blockscaled_dw_setup(mt_dw_conv2d_quant_mxint4::kernel_ir_for(dt), crate::quant::format::QFormat::Mxint4, 1, 4, 16, 16, 8, 1, 0, 1, dt)
    }
}

// ─── § Benches ────────────────────────────────────────────────────────────────

pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    fn conv2d_bench(kernel: Kernel, batch: usize, in_ch: usize, in_h: usize, in_w: usize, out_ch: usize, kh: usize, kw: usize, stride_h: usize, stride_w: usize, dt: DType) -> BenchSetup {
        let out_h = (in_h - kh) / stride_h + 1;
        let out_w = (in_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * kh * kw, dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32).constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32).constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32).constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32).constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", 0u32).constexpr("pad_w", 0u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            .flops(2 * (batch as u64) * (out_ch as u64) * (out_h as u64) * (out_w as u64) * (in_ch as u64) * (kh as u64) * (kw as u64))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv2d_patch14(dt: DType) -> BenchSetup {
        conv2d_bench(conv2d_patch14::kernel_ir_for(dt), 1, 3, 224, 224, 1024, 14, 14, 14, 14, dt)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv2d_patch16(dt: DType) -> BenchSetup {
        conv2d_bench(conv2d_patch16::kernel_ir_for(dt), 1, 3, 224, 224, 768, 16, 16, 16, 16, dt)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv2d_generic(dt: DType) -> BenchSetup {
        conv2d_bench(conv2d_generic::kernel_ir_for(dt), 1, 32, 56, 56, 64, 3, 3, 1, 1, dt)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv2d_grouped(dt: DType) -> BenchSetup {
        let (batch, ch, in_h, in_w, kh, kw) = (1usize, 64usize, 56usize, 56usize, 3usize, 3usize);
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_out = batch * ch * out_h * out_w;
        BenchSetup::new(conv2d_grouped::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", ch * kh * kw, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", ch as u32).constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_ch", ch as u32).constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32).constexpr("kw", kw as u32)
            .constexpr("stride_h", 1u32).constexpr("stride_w", 1u32)
            .constexpr("pad_h", 0u32).constexpr("pad_w", 0u32)
            .constexpr("dilation_h", 1u32).constexpr("dilation_w", 1u32)
            .constexpr("icpg", 1u32).constexpr("ocpg", 1u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            .flops(2 * (batch as u64) * (ch as u64) * (out_h as u64) * (out_w as u64) * (kh as u64) * (kw as u64))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv2d_mma(dt: DType) -> BenchSetup {
        let (batch, in_ch, in_h, in_w, out_ch, kh, kw) = (1usize, 256usize, 32usize, 32usize, 1024usize, 1usize, 1usize);
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        let n_out = n_pixels * out_ch;
        BenchSetup::new(conv2d_mma::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", out_ch * in_ch * kh * kw, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32).constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32).constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32).constexpr("kw", kw as u32)
            .grid_3d((out_ch / 32) as u32, (n_pixels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            .flops(2 * (batch as u64) * (out_ch as u64) * (out_h as u64) * (out_w as u64) * (in_ch as u64) * (kh as u64) * (kw as u64))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_depthwise_conv2d(dt: DType) -> BenchSetup {
        let (batch, ch, in_h, in_w, k, stride, pad, dilation) = (1usize, 64usize, 112usize, 112usize, 3usize, 2usize, 1usize, 1usize);
        let out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * ch * out_h * out_w;
        BenchSetup::new(depthwise_conv2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", ch * k * k, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32).constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("k", k as u32).constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32).constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            .flops(2 * (batch as u64) * (ch as u64) * (out_h as u64) * (out_w as u64) * (k as u64) * (k as u64))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_depthwise_conv2d_nhwc(dt: DType) -> BenchSetup {
        let (batch, ch, in_h, in_w, k, stride, pad, dilation) = (1usize, 64usize, 112usize, 112usize, 3usize, 2usize, 1usize, 1usize);
        let out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * out_h * out_w * ch;
        BenchSetup::new(depthwise_conv2d_nhwc::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_h * in_w * ch, dt))
            .buffer(BenchBuffer::random("weight", ch * k * k, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32).constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32).constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32).constexpr("out_w", out_w as u32)
            .constexpr("k_h", k as u32).constexpr("k_w", k as u32)
            .constexpr("stride", stride as u32).constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            .flops(2 * (batch as u64) * (ch as u64) * (out_h as u64) * (out_w as u64) * (k as u64) * (k as u64))
    }
}

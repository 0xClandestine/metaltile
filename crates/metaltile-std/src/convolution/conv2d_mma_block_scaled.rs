//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantized-weight variants of the cooperative (simdgroup-matrix
//! MMA) 2D convolution. These are the M≥32 ALU-throughput conv path for the
//! spec-conformant quantized formats — a direct fusion of two GPU-verified
//! templates:
//!
//! - `ffai/conv2d_mma.rs` (dense cooperative conv2d) supplies the **entire
//!   geometry verbatim**: the implicit-im2col A-load (input gather with
//!   in-bounds masking), the threadgroup/simdgroup allocation sizes, the
//!   4-frag × 4-k-inner `simdgroup_matmul` inner loop, the C-store, and the
//!   dispatch grid.
//!
//! - `mlx/block_scaled_mma.rs` (block-scaled MMA qmm) supplies the per-format
//!   **weight dequant staging** — the ONLY thing that changes vs the dense
//!   conv2d_mma is the cooperative B-load into `bs`.
//!
//! ## Implicit im2col as a quantized-weight GEMM
//!
//! Treat the convolution as a GEMM where the B (weight) operand is quantized:
//!
//!   out[BN_pixels, BM_oc] = A[BN_pixels, BK] × dequant(B)[BK, BM_oc]
//!
//! where:
//!
//!   - `BK  = in_ch * kh * kw` (filter taps per output position = contraction)
//!   - `BN  = batch * out_h * out_w` (output positions = "pixels")
//!   - `BM  = out_ch` (output channels)
//!
//! The A matrix is never materialised — each lane computes its `(kh, kw, ic)`
//! → `(h_in, w_in, ic)` gather index on-the-fly (copied verbatim from the
//! dense conv2d_mma). The B matrix is the quantized filter laid out as the
//! 2-D matrix `[out_ch, BK]` (row `oc`, column `kt = ic*kh*kw + ky*kw + kx`).
//!
//! ## Quantized B-load (the only change vs dense conv2d_mma)
//!
//! Each lane stages 8 contiguous taps `kt` for its oc-row `b_oc_row` into
//! `bs`. The dense load reads `weight[w_oc_base + kt]` directly; here the
//! same `(oc, kt)` element is dequantized instead:
//!
//!   - 4-bit (E2M1): weight is `[out_ch, BK/8]` u32 (8 nibbles/word). The
//!     nibble for tap `kt` is word `oc*(BK/8) + kt/8`, shift `(kt%8)*4`.
//!
//!   - 8-bit (E4M3 / E5M2 / int8): weight is `[out_ch, BK]` u8 (1 byte/tap).
//!     The byte for tap `kt` is `oc*BK + kt`.
//!
//!   - The block scale for tap `kt` is `scales[oc*(BK/block_size) + kt/block_size]`
//!     (E8M0 `exp2(b-127)`, E4M3 micro-scale × global FP32, or raw FP32).
//!
//! Decoding is done tap-by-tap so arbitrary `kt` alignment is handled (the
//! K-loop steps by 32 but `BK` need only be a multiple of 32 and of the
//! format's block_size). The dense in-bounds masking is preserved:
//! `select(kt < total_k, decoded, 0.0)`, and the safe (clamped) index keeps
//! the gather in range.
//!
//! ## Dispatch invariants (identical to dense conv2d_mma)
//!
//! - **Mode: Reduction**, grid `[out_ch/32, (batch*out_h*out_w)/32, 1]`,
//!   tpg = 128 (4 simdgroups × 32 lanes, 2×2 warp grid).
//! - BM = BN = 32, output tile 32×32. `out_ch` and `n_pixels` multiples of 32.
//! - `BK = in_ch*kh*kw` a multiple of 32 (the MMA K-tile) and of block_size.
//! - stride = 1, dilation = 1, padding = 0 (vision patch-conv style).
//! - NCHW input, quantized OIHW-flattened weight, pixel-major out. No bias.
//!
//! Codegen-only. Correctness validated by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 (E2M1 weights, block 32, E8M0 pow-2 scale) cooperative conv2d.
///
/// Grid `[out_ch/32, (batch*out_h*out_w)/32, 1]`, tpg = 128.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    // ── Geometry copied verbatim from dense conv2d_mma ──
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
    // Quantized weight layout: 4-bit packs 8 nibbles/u32 word → row stride
    // total_k/8 words; one E8M0 scale byte per block → row stride
    // total_k/block_size.
    let packs_per_row = total_k / 8u32;
    let n_blocks = total_k / block_size;
    let w_pack_row_base = global_oc * packs_per_row;
    let sb_base = global_oc * n_blocks;
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit im2col gather) — verbatim from conv2d_mma ─
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
        // ─ 2. Coop B load (mxfp4 dequant) — E2M1 nibble × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = mt_decode_e2m1(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        // ─ 3. MMA inner loop — copied verbatim from conv2d_mma ─
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
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
        // k_inner = 1
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
        // k_inner = 2
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
        // k_inner = 3
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
    // ── 4. Write 4 C frags to global out — verbatim from conv2d_mma ──
    let out_px_base = px_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// nvfp4 (E2M1 weights, block 16, E4M3 micro-scale × global FP32) conv2d.
///
/// `global` is the LAST constexpr (two-level scaling).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
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
    let packs_per_row = total_k / 8u32;
    let n_blocks = total_k / block_size;
    let w_pack_row_base = global_oc * packs_per_row;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (nvfp4 dequant) — E2M1 nibble × E4M3 micro × global ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let micro = mt_decode_e4m3(load(scales[sb_base + kt_safe / block_size]).cast::<u32>());
            let scale = micro * global;
            let decoded = mt_decode_e2m1(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// fp4 (E2M1 weights, group 32, raw per-group FP32 scale) conv2d.
///
/// Verified on f32/f16/bf16 against the `quant::format` oracle.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let packs_per_row = total_k / 8u32;
    let n_blocks = total_k / block_size;
    let w_pack_row_base = global_oc * packs_per_row;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (fp4 dequant) — E2M1 nibble × raw per-group FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = mt_decode_e2m1(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// mxfp8 (E4M3 weights, block 32, E8M0 pow-2 scale) cooperative conv2d.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    // 8-bit: weight is [out_ch, total_k] u8 (1 byte/tap); one E8M0 scale byte
    // per block → row stride total_k/block_size.
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (mxfp8 e4m3 dequant) — E4M3 byte × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = mt_decode_e4m3(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// mxfp8 (E5M2 weights, block 32, E8M0 pow-2 scale) cooperative conv2d.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (mxfp8 e5m2 dequant) — E5M2 byte × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = mt_decode_e5m2(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// nvfp8 (E4M3 weights, block 16, raw per-block FP32 scale) cooperative conv2d.
///
/// fp8_e4m3 reuses this kernel (same 8-bit-E4M3 + FP32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (nvfp8 dequant) — E4M3 byte × raw per-block FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = mt_decode_e4m3(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// Legacy fp8 (E5M2 weights, group 32, raw per-group FP32 scale) conv2d.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (fp8 e5m2 dequant) — E5M2 byte × raw per-group FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = mt_decode_e5m2(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// Symmetric int8 (8-bit codes, group 64, raw per-group FP32 scale) conv2d.
///
/// Decode is sign-extend → `code · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (int8 dequant) — sign-extended code × raw per-group FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = mt_decode_int8(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ── Symmetric integer (2/3/4/5/6-bit) cooperative conv2d ───────────────────
// These mirror `mt_int8_conv2d_mma` verbatim for the entire dispatch geometry
// (implicit-im2col A-load, 4-frag × 4-k-inner MMA inner loop, C-store, grid).
// The ONLY change is the cooperative B-load: instead of one byte per tap, the
// quantized filter is a tight LSB-first bit-stream packed into `Tensor<u32>`.
//
// ## Sub-byte filter staging (the only change vs `mt_int8_conv2d_mma`)
//
// The quantized filter is the 2-D matrix `[out_ch, BK]` (BK = total_k =
// in_ch*kh*kw, the per-output-channel contraction length C). Element
// `(oc, col)` — output channel `oc`, contraction index `col` over
// in_ch·kh·kw — lives at the GLOBAL flat bit offset
//
//   bit_off = (oc * C + col) * BITS
//
// in the row-major LSB-first u32 code stream produced by `quant::format::pack`.
// Using the global offset (rather than a per-row word base) is robust whether
// or not C is a multiple of 32 — it matches the flat layout exactly, and is
// the same straddle-aware two-word read used by `mlx/block_scaled_mma.rs`'s
// `int_qmm_mma_f32!` / `int_qmm_mma_e8m0!`.
//
// `w_global_row_base = global_oc * total_k` is each lane's row-start element
// index; the per-tap element index is `w_global_row_base + kt_safe`. Decode is
// a straddle-aware two-word read + float sign-extend, identical to the codec.
macro_rules! int_conv2d_mma_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] block_size: u32,
        ) {
            // ── Geometry copied verbatim from `mt_int8_conv2d_mma` ──
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
            let n_blocks = total_k / block_size;
            // Global element index of this lane's filter row start (flat
            // bit-stream is row-major: element (oc, col) at bit (oc·total_k+col)·bits).
            let w_global_row_base = global_oc * total_k;
            let sb_base = global_oc * n_blocks;
            for kb in range(0u32, total_k, 32u32) {
                // ─ 1. Coop A load (implicit im2col gather) — verbatim from int8 ─
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
                // ─ 2. Coop B load (intN dequant) — bit-stream code × raw per-group FP32 ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let scale = load(scales[sb_base + kt_safe / block_size]);
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_global_row_base + kt_safe) * $bits;
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
                    let qv = select(q >= $half, qf - $full, qf); // sign-extend
                    let decoded = qv * scale;
                    let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
                }
                threadgroup_barrier();
                // ─ 3. MMA inner loop — copied verbatim from int8 ─
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;
                // k_inner = 0
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
                // k_inner = 1
                simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 8u32 + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 8u32 + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 8u32 + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 8u32 + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 8u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                // k_inner = 2
                simdgroup_elem_store(
                    a_f0,
                    0,
                    threadgroup_load("as", row_a0 * stride + 16u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f0,
                    1,
                    threadgroup_load("as", row_a0 * stride + 16u32 + fn1),
                );
                simdgroup_elem_store(
                    a_f1,
                    0,
                    threadgroup_load("as", row_a1 * stride + 16u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f1,
                    1,
                    threadgroup_load("as", row_a1 * stride + 16u32 + fn1),
                );
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 16u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                // k_inner = 3
                simdgroup_elem_store(
                    a_f0,
                    0,
                    threadgroup_load("as", row_a0 * stride + 24u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f0,
                    1,
                    threadgroup_load("as", row_a0 * stride + 24u32 + fn1),
                );
                simdgroup_elem_store(
                    a_f1,
                    0,
                    threadgroup_load("as", row_a1 * stride + 24u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f1,
                    1,
                    threadgroup_load("as", row_a1 * stride + 24u32 + fn1),
                );
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                threadgroup_barrier();
            }
            // ── 4. Write 4 C frags to global out — verbatim from int8 ──
            let out_px_base = px_tile * 32u32 + sm * 16u32;
            let out_oc_base = oc_tile * 32u32 + sn * 16u32;
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }
    };
}

// int2 (2-bit symmetric, group 64, raw per-group FP32 scale) cooperative conv2d.
int_conv2d_mma_f32!(mt_int2_conv2d_mma, 2u32, 2u32, 4.0f32);
// int3 (3-bit symmetric, group 64, raw per-group FP32 scale) cooperative conv2d.
int_conv2d_mma_f32!(mt_int3_conv2d_mma, 3u32, 4u32, 8.0f32);
// int4 (4-bit symmetric, group 64, raw per-group FP32 scale) cooperative conv2d.
int_conv2d_mma_f32!(mt_int4_conv2d_mma, 4u32, 8u32, 16.0f32);
// int5 (5-bit symmetric, group 64, raw per-group FP32 scale) cooperative conv2d.
int_conv2d_mma_f32!(mt_int5_conv2d_mma, 5u32, 16u32, 32.0f32);
// int6 (6-bit symmetric, group 64, raw per-group FP32 scale) cooperative conv2d.
int_conv2d_mma_f32!(mt_int6_conv2d_mma, 6u32, 32u32, 64.0f32);

// ── E8M0-scaled symmetric integer (2/3/4/5/6-bit) cooperative conv2d ───────
// Identical body to `int_conv2d_mma_f32!` (same straddle-aware global-bit-offset
// sub-byte decode and dispatch geometry); only the scale axis differs — one
// u8 E8M0 exponent per block, dequantized as `2^(bits-127)`.
macro_rules! int_conv2d_mma_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<u8>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] block_size: u32,
        ) {
            // ── Geometry copied verbatim from `mt_int8_conv2d_mma` ──
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
            let n_blocks = total_k / block_size;
            // Global element index of this lane's filter row start (flat
            // bit-stream is row-major: element (oc, col) at bit (oc·total_k+col)·bits).
            let w_global_row_base = global_oc * total_k;
            let sb_base = global_oc * n_blocks;
            for kb in range(0u32, total_k, 32u32) {
                // ─ 1. Coop A load (implicit im2col gather) — verbatim from int8 ─
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
                // ─ 2. Coop B load (mxintN dequant) — bit-stream code × E8M0 pow-2 scale ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_global_row_base + kt_safe) * $bits;
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
                    let qv = select(q >= $half, qf - $full, qf); // sign-extend
                    let decoded = qv * scale;
                    let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
                }
                threadgroup_barrier();
                // ─ 3. MMA inner loop — copied verbatim from int8 ─
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;
                // k_inner = 0
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
                // k_inner = 1
                simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 8u32 + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 8u32 + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 8u32 + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 8u32 + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 8u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                // k_inner = 2
                simdgroup_elem_store(
                    a_f0,
                    0,
                    threadgroup_load("as", row_a0 * stride + 16u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f0,
                    1,
                    threadgroup_load("as", row_a0 * stride + 16u32 + fn1),
                );
                simdgroup_elem_store(
                    a_f1,
                    0,
                    threadgroup_load("as", row_a1 * stride + 16u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f1,
                    1,
                    threadgroup_load("as", row_a1 * stride + 16u32 + fn1),
                );
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 16u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                // k_inner = 3
                simdgroup_elem_store(
                    a_f0,
                    0,
                    threadgroup_load("as", row_a0 * stride + 24u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f0,
                    1,
                    threadgroup_load("as", row_a0 * stride + 24u32 + fn1),
                );
                simdgroup_elem_store(
                    a_f1,
                    0,
                    threadgroup_load("as", row_a1 * stride + 24u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f1,
                    1,
                    threadgroup_load("as", row_a1 * stride + 24u32 + fn1),
                );
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                threadgroup_barrier();
            }
            // ── 4. Write 4 C frags to global out — verbatim from int8 ──
            let out_px_base = px_tile * 32u32 + sm * 16u32;
            let out_oc_base = oc_tile * 32u32 + sn * 16u32;
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }
    };
}

// mxint2 (2-bit symmetric, block 32, E8M0 pow-2 scale) cooperative conv2d.
int_conv2d_mma_e8m0!(mt_mxint2_conv2d_mma, 2u32, 2u32, 4.0f32);
// mxint3 (3-bit symmetric, block 32, E8M0 pow-2 scale) cooperative conv2d.
int_conv2d_mma_e8m0!(mt_mxint3_conv2d_mma, 3u32, 4u32, 8.0f32);
// mxint4 (4-bit symmetric, block 32, E8M0 pow-2 scale) cooperative conv2d.
int_conv2d_mma_e8m0!(mt_mxint4_conv2d_mma, 4u32, 8u32, 16.0f32);
// mxint5 (5-bit symmetric, block 32, E8M0 pow-2 scale) cooperative conv2d.
int_conv2d_mma_e8m0!(mt_mxint5_conv2d_mma, 5u32, 16u32, 32.0f32);
// mxint6 (6-bit symmetric, block 32, E8M0 pow-2 scale) cooperative conv2d.
int_conv2d_mma_e8m0!(mt_mxint6_conv2d_mma, 6u32, 32u32, 64.0f32);

/// mxint8 (8-bit symmetric codes, byte layout, block 32, E8M0 pow-2 scale)
/// cooperative conv2d. Identical geometry and B-load mapping to
/// `mt_int8_conv2d_mma` (one byte per code); only the scale axis is E8M0
/// (`2^(bits-127)`) instead of a raw per-group FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    // 8-bit: weight is [out_ch, total_k] u8 (1 byte/tap); one E8M0 scale byte
    // per block → row stride total_k/block_size.
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (mxint8 dequant) — sign-extended byte × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
            let decoded = mt_decode_int8(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ── FP16-scale twins of the FP32-scaled block-scaled conv2d kernels ────────
// These are byte-for-byte clones of the FP32-scaled kernels above; the ONLY
// change is the scale axis: the `scales` tensor is `Tensor<f16>` (was
// `Tensor<f32>`) and the per-block scale read is `load(...).cast::<f32>()`
// (was a bare `load(...)`). The element decode (E2M1 / E4M3 / E5M2 / int
// bit-stream + sign-extend), weight indexing, threadgroup staging, dispatch
// geometry, and the 4-frag × 4-k-inner MMA inner loop are IDENTICAL to the
// FP32 twin. The GPU-verified reference for the f16 scale read is
// `mlx/block_scaled_dequant.rs` (`mt_*_f16_dequant`).

/// fp4 (E2M1 weights, group 32, raw per-group FP16 scale) conv2d. FP16-scale
/// twin of `mt_fp4_conv2d_mma`; only the scale axis differs.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let packs_per_row = total_k / 8u32;
    let n_blocks = total_k / block_size;
    let w_pack_row_base = global_oc * packs_per_row;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (fp4 dequant) — E2M1 nibble × raw per-group FP16 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = mt_decode_e2m1(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// nvfp8 (E4M3 weights, block 16, raw per-block FP16 scale) cooperative conv2d.
/// FP16-scale twin of `mt_nvfp8_conv2d_mma`; only the scale axis differs.
/// fp8_e4m3_f16 reuses this kernel (same 8-bit-E4M3 + FP16-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (nvfp8 dequant) — E4M3 byte × raw per-block FP16 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = mt_decode_e4m3(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// Legacy fp8 (E5M2 weights, group 32, raw per-group FP16 scale) conv2d.
/// FP16-scale twin of `mt_fp8_e5m2_conv2d_mma`; only the scale axis differs.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (fp8 e5m2 dequant) — E5M2 byte × raw per-group FP16 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = mt_decode_e5m2(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ── FP16-scaled symmetric integer (2/3/4/5/6-bit) cooperative conv2d ───────
// Byte-for-byte clone of `int_conv2d_mma_f32!` (same straddle-aware
// global-bit-offset sub-byte decode, sign-extend, and dispatch geometry);
// the ONLY change is the scale axis — the `scales` tensor is `Tensor<f16>`
// (was `Tensor<f32>`) and the per-group scale read is `load(...).cast::<f32>()`.
macro_rules! int_conv2d_mma_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f16>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] block_size: u32,
        ) {
            // ── Geometry copied verbatim from `mt_int8_conv2d_mma` ──
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
            let n_blocks = total_k / block_size;
            // Global element index of this lane's filter row start (flat
            // bit-stream is row-major: element (oc, col) at bit (oc·total_k+col)·bits).
            let w_global_row_base = global_oc * total_k;
            let sb_base = global_oc * n_blocks;
            for kb in range(0u32, total_k, 32u32) {
                // ─ 1. Coop A load (implicit im2col gather) — verbatim from int8 ─
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
                // ─ 2. Coop B load (intN dequant) — bit-stream code × raw per-group FP16 ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_global_row_base + kt_safe) * $bits;
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
                    let qv = select(q >= $half, qf - $full, qf); // sign-extend
                    let decoded = qv * scale;
                    let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
                }
                threadgroup_barrier();
                // ─ 3. MMA inner loop — copied verbatim from int8 ─
                let row_a0 = sm * 16u32 + fm;
                let row_a1 = sm * 16u32 + 8u32 + fm;
                let col_b0 = sn * 16u32;
                let col_b1 = sn * 16u32 + 8u32;
                // k_inner = 0
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
                // k_inner = 1
                simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 8u32 + fn0));
                simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 8u32 + fn1));
                simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 8u32 + fn0));
                simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 8u32 + fn1));
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 8u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                // k_inner = 2
                simdgroup_elem_store(
                    a_f0,
                    0,
                    threadgroup_load("as", row_a0 * stride + 16u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f0,
                    1,
                    threadgroup_load("as", row_a0 * stride + 16u32 + fn1),
                );
                simdgroup_elem_store(
                    a_f1,
                    0,
                    threadgroup_load("as", row_a1 * stride + 16u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f1,
                    1,
                    threadgroup_load("as", row_a1 * stride + 16u32 + fn1),
                );
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 16u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                // k_inner = 3
                simdgroup_elem_store(
                    a_f0,
                    0,
                    threadgroup_load("as", row_a0 * stride + 24u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f0,
                    1,
                    threadgroup_load("as", row_a0 * stride + 24u32 + fn1),
                );
                simdgroup_elem_store(
                    a_f1,
                    0,
                    threadgroup_load("as", row_a1 * stride + 24u32 + fn0),
                );
                simdgroup_elem_store(
                    a_f1,
                    1,
                    threadgroup_load("as", row_a1 * stride + 24u32 + fn1),
                );
                simdgroup_barrier_mem_none();
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("bs", (col_b0 + fn0) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("bs", (col_b0 + fn1) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("bs", (col_b1 + fn0) * stride + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("bs", (col_b1 + fn1) * stride + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                threadgroup_barrier();
            }
            // ── 4. Write 4 C frags to global out — verbatim from int8 ──
            let out_px_base = px_tile * 32u32 + sm * 16u32;
            let out_oc_base = oc_tile * 32u32 + sn * 16u32;
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }
    };
}

// int2_f16 (2-bit symmetric, group 64, raw per-group FP16 scale) cooperative conv2d.
int_conv2d_mma_f16!(mt_int2_f16_conv2d_mma, 2u32, 2u32, 4.0f32);
// int3_f16 (3-bit symmetric, group 64, raw per-group FP16 scale) cooperative conv2d.
int_conv2d_mma_f16!(mt_int3_f16_conv2d_mma, 3u32, 4u32, 8.0f32);
// int4_f16 (4-bit symmetric, group 64, raw per-group FP16 scale) cooperative conv2d.
int_conv2d_mma_f16!(mt_int4_f16_conv2d_mma, 4u32, 8u32, 16.0f32);
// int5_f16 (5-bit symmetric, group 64, raw per-group FP16 scale) cooperative conv2d.
int_conv2d_mma_f16!(mt_int5_f16_conv2d_mma, 5u32, 16u32, 32.0f32);
// int6_f16 (6-bit symmetric, group 64, raw per-group FP16 scale) cooperative conv2d.
int_conv2d_mma_f16!(mt_int6_f16_conv2d_mma, 6u32, 32u32, 64.0f32);

/// Symmetric int8 (8-bit codes, group 64, raw per-group FP16 scale) conv2d.
/// FP16-scale twin of `mt_int8_conv2d_mma`; only the scale axis differs.
/// Decode is sign-extend → `code · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_conv2d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
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
    let n_blocks = total_k / block_size;
    let w_row_base = global_oc * total_k;
    let sb_base = global_oc * n_blocks;
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
        // ─ 2. Coop B load (int8 dequant) — sign-extended code × raw per-group FP16 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = mt_decode_int8(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
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
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::{QFormat, ScaleKind},
        utils::{pack_f32, unpack_f32},
    };

    /// Bounded zig-zag ramp (keeps f16/bf16 in range), identical to the dense
    /// conv2d_mma helper.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 2D conv oracle, pixel-major output `[n_pixels, out_ch]`, run over
    /// the *dequantized* filter `[out_ch, BK]` (BK = in_ch*kh*kw, col =
    /// (ic*kh + ky)*kw + kx). stride=1, dilation=1, pad=0, no bias. The SAME
    /// dense math as conv2d_mma.rs's `naive_conv2d_mma`.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv2d_mma(
        input: &[f32],
        weight: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
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
                                    let ih = oh + ky;
                                    let iw = ow + kx;
                                    let in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                                    // Quantized filter is the 2-D matrix
                                    // [out_ch, BK]; col = (ic*kh+ky)*kw+kx.
                                    let col = (ic * kh + ky) * kw + kx;
                                    let w_idx = oc * (in_ch * kh * kw) + col;
                                    acc += input[in_idx] * weight[w_idx];
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

    /// QFormat-parametrized setup: quantize the `[out_ch, BK]` filter via the
    /// shared codec, dequantize for the oracle, and run the dense conv2d_mma
    /// math over the dequantized filter. Mirrors conv2d_mma.rs's `mma_setup`
    /// grid + KernelMode, just swapping the dense filter for a quantized one.
    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        assert_eq!(out_ch % 32, 0, "out_ch must be a multiple of 32 for the MMA tile");
        assert_eq!(n_pixels % 32, 0, "n_pixels must be a multiple of 32 for the MMA tile");
        // BK = in_ch*kh*kw — the quantized filter is [out_ch, BK]. Must be a
        // multiple of the format's block_size and of the 32 MMA K-tile.
        let bk = in_ch * kh * kw;
        assert_eq!(bk % 32, 0, "BK (in_ch*kh*kw) must be a multiple of the 32 MMA K-tile");
        assert_eq!(bk % fmt.block_size(), 0, "BK must be a multiple of the format block_size");
        let n_out = n_pixels * out_ch;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 2.0);
        // Quantize the [out_ch, BK] filter via the shared codec.
        let filter_f = ramp(out_ch * bk, 11, 2.0);
        let p = crate::quant::format::pack(fmt, &filter_f, out_ch, bk);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, bk);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = naive_conv2d_mma(&input, &wdq, batch, in_ch, in_h, in_w, out_ch, kh, kw);
        // Axis-driven binding (robust across all element widths/scale kinds):
        //   - 8-bit codes (E4M3/E5M2/int8/mxint8) bind one uchar each; every
        //     sub-byte width (4-bit E2M1 + the 2/3/5/6-bit int bit-streams)
        //     binds as a packed u32 code stream.
        //   - FP32-scaled formats bind raw f32 scales; E8M0/E4M3-scaled formats
        //     bind one byte each.
        // For the pre-existing formats this is identical to the old `== 4` /
        // `matches!` logic (4-bit → U32, 8-bit → U8; the float-scale list maps
        // exactly to ScaleKind::F32), so there is no regression.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            ScaleKind::F32 => DType::F32,
            ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (out_ch / 32) as u32,
            (n_pixels / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // Dims: in_ch=4, kh=kw=4 → BK=64 (2 K-blocks of 32; divisible by 16/32/64).
    // 7×7 input → out 4×4, batch=2 → n_pixels=32; out_ch=32. One 32×32 tile.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp4_conv2d_mma::kernel_ir_for(dt), QFormat::Mxfp4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp4_conv2d_mma::kernel_ir_for(dt), QFormat::Nvfp4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_conv2d_mma::kernel_ir_for(dt), QFormat::Fp4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e4m3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e5m2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_conv2d_mma::kernel_ir_for(dt), QFormat::Nvfp8, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    // fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int8_conv2d_mma::kernel_ir_for(dt), QFormat::Int8, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    // ── Symmetric integer formats (FP32 group scale, group 64) ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int2_conv2d_mma::kernel_ir_for(dt), QFormat::Int2, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int3_conv2d_mma::kernel_ir_for(dt), QFormat::Int3, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int4_conv2d_mma::kernel_ir_for(dt), QFormat::Int4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int5_conv2d_mma::kernel_ir_for(dt), QFormat::Int5, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int6_conv2d_mma::kernel_ir_for(dt), QFormat::Int6, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    // ── E8M0-scaled symmetric integer formats (block 32) ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint4_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint5_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint6_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    // ── FP16-scale twins of the FP32-scaled formats ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp4_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }
}

/// Realistic vision-encoder benches — the M≥32 simdgroup-matrix throughput
/// path for quantized-weight conv2d. Random packed buffers (throughput is
/// data-independent). Shape: 1×1 conv on a 32×32 feature map with in_ch=64
/// → BK = 64·1·1 = 64 (divisible by 16/32/64 block sizes and the 32 K-tile),
/// n_pixels = 1024, out_ch = 1024.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::{QFormat, ScaleKind};

    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        let n_out = n_pixels * out_ch;
        // BK = in_ch*kh*kw — the quantized filter is [out_ch, BK].
        let bk = in_ch * kh * kw;
        let n_blocks = out_ch * (bk / fmt.block_size());
        // Axis-driven code buffer: 8-bit codes are one byte each; every sub-byte
        // width packs into a tight u32 bit-stream (`bitstream_words` collapses to
        // the old `n/8` for the 4-bit format, so no regression).
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (out_ch * bk, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(out_ch * bk, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            ScaleKind::F32 => DType::F32,
            ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_ch / 32) as u32, (n_pixels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // 2 * N * Co * Ho * Wo * Ci * kh * kw (stride=1, pad=0)
            .flops(
                2 * (batch as u64)
                    * (out_ch as u64)
                    * (out_h as u64)
                    * (out_w as u64)
                    * (in_ch as u64)
                    * (kh as u64)
                    * (kw as u64),
            )
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp4_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp4,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp4_conv2d_mma::kernel_ir_for(dt),
            QFormat::Nvfp4,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp4_conv2d_mma::kernel_ir_for(dt), QFormat::Fp4, 1, 64, 32, 32, 1024, 1, 1, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp8_e4m3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp8_e5m2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Nvfp8,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int8,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int2,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int3,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int4_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int4,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int5_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int5,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int6_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int6,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint4_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint5_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint6_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    // ── FP16-scale twins of the FP32-scaled formats ──
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp4_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int2_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int3_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int4_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int5_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int6_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_conv2d_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            1,
            64,
            32,
            32,
            1024,
            1,
            1,
            dt,
        )
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantized-weight variants of the cooperative (simdgroup-matrix
//! MMA) patch embedding. These are the M≥32 ALU-throughput patch-embed path for
//! the spec-conformant quantized formats — a direct fusion of two GPU-verified
//! templates:
//!
//! - `ffai/patch_embed_mma.rs` (dense cooperative patch embed) supplies the
//!   **entire geometry verbatim**: the implicit patch-unfold A-load (input
//!   gather with in-bounds masking), the threadgroup/simdgroup allocation
//!   sizes, the 4-frag × 4-k-inner `simdgroup_matmul` inner loop, the
//!   bias-added C-store (output layout `[num_patches, hidden]`), and the
//!   dispatch grid.
//!
//! - `ffai/conv2d_mma_block_scaled.rs` (block-scaled MMA conv) supplies the
//!   per-format **weight dequant staging** — the ONLY thing that changes vs the
//!   dense patch_embed_mma is the cooperative B-load into `bs`.
//!
//! ## Patch embed as a quantized-weight GEMM
//!
//! The patch embedding is a linear projection, algebraically a conv2d with
//! `stride = patch`, no overlap, no padding, but a `[num_patches, hidden]`
//! output layout. Treat it as a GEMM where the B (weight) operand is quantized:
//!
//!   out[num_patches, hidden] = A[num_patches, patch_dim] × dequant(B)[patch_dim, hidden]
//!
//! where:
//!
//!   - `patch_dim = in_ch * patch_h * patch_w` (taps per patch = contraction K).
//!   - `A` is the implicit patch unfold — each lane computes its `(ic, py, px)`
//!     → `(ic, py0+py, px0+px)` gather index on-the-fly (copied verbatim from
//!     the dense patch_embed_mma).
//!   - `B` is the quantized projection weight laid out as the 2-D matrix
//!     `[hidden, patch_dim]` (row `h` = hidden unit, column `kt = ic*patch_h*patch_w
//!     + py*patch_w + px`). This is the same row-major `[N, K]` weight the
//!     `quant::format` packer produces.
//!
//! ## Quantized B-load (the only change vs dense patch_embed_mma)
//!
//! Each lane stages 8 contiguous taps `kt` for its hidden-unit row `b_h_row`
//! into `bs`. The dense load reads `weight[w_h_base + kt]` directly; here the
//! same `(h, kt)` element is dequantized instead — identical decode to
//! `conv2d_mma_block_scaled` with the conv's `total_k` replaced by `patch_dim`:
//!
//!   - 4-bit (E2M1): weight is `[hidden, patch_dim/8]` u32 (8 nibbles/word). The
//!     nibble for tap `kt` is word `h*(patch_dim/8) + kt/8`, shift `(kt%8)*4`.
//!
//!   - 8-bit (E4M3 / E5M2 / int8 / MXINT8): weight is `[hidden, patch_dim]` u8
//!     (1 byte/tap). The byte for tap `kt` is `h*patch_dim + kt`.
//!
//!   - sub-byte symmetric int (int2/3/4/5/6 + MXINT2..6): weight is a FLAT
//!     row-major u32 bit-stream, tight-packed LSB-first by `quant::format::pack`.
//!     The N-bit two's-complement code for tap `kt` of row `h` lives at GLOBAL
//!     bit offset `(h*patch_dim + kt)*N`, read straddle-aware across two words
//!     and float-sign-extended (`code - 2^N` when the top bit is set). `patch_dim`
//!     is a multiple of 32, so every row's bit-stream is word-aligned.
//!
//!   - The block scale for tap `kt` is
//!     `scales[h*(patch_dim/block_size) + kt/block_size]` (E8M0 `exp2(b-127)`,
//!     E4M3 micro-scale × global FP32, or raw FP32).
//!
//! The dense in-bounds masking is preserved: `select(kt < patch_dim, decoded,
//! 0.0)`, and the safe (clamped) index keeps the gather in range.
//!
//! ## Dispatch invariants (identical to dense patch_embed_mma)
//!
//! - **Mode: Reduction**, grid `[hidden/32, num_patches/32, 1]`, tpg = 128
//!   (4 simdgroups × 32 lanes, 2×2 warp grid).
//! - BM = BN = 32, output tile 32×32. `hidden` and `num_patches` multiples of 32.
//! - `patch_dim = in_ch*patch_h*patch_w` a multiple of 32 (the MMA K-tile) and
//!   of the format's block_size.
//! - Single image (no batch — matches `patch_embed.rs` layout). Per-channel
//!   `bias` stays `T` (tiny + precision-sensitive); only the weight is quantized.
//!
//! Codegen-only. Correctness validated by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 (E2M1 weights, block 32, E8M0 pow-2 scale) cooperative patch embed.
///
/// Grid `[hidden/32, num_patches/32, 1]`, tpg = 128.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    // ── Geometry copied verbatim from dense patch_embed_mma ──
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    // Quantized weight layout: 4-bit packs 8 nibbles/u32 word → row stride
    // patch_dim/8 words; one E8M0 scale byte per block → row stride
    // patch_dim/block_size.
    let packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_pack_row_base = global_h * packs_per_row;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        // ─ 1. Coop A load (implicit patch unfold gather) — verbatim from patch_embed_mma ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxfp4 dequant) — E2M1 nibble × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = e2m1_decode(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        // ─ 3. MMA inner loop — copied verbatim from patch_embed_mma ─
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
    // ── 4. Add bias and write 4 C frags to global out — verbatim from patch_embed_mma ──
    // out layout: [num_patches, hidden].
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// nvfp4 (E2M1 weights, block 16, E4M3 micro-scale × global FP32) patch embed.
///
/// `global` is the LAST constexpr (two-level scaling).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_pack_row_base = global_h * packs_per_row;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (nvfp4 dequant) — E2M1 nibble × E4M3 micro × global ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let micro = e4m3_decode(load(scales[sb_base + kt_safe / block_size]).cast::<u32>());
            let scale = micro * global;
            let decoded = e2m1_decode(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// fp4 (E2M1 weights, group 32, raw per-group FP32 scale) patch embed.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_pack_row_base = global_h * packs_per_row;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (fp4 dequant) — E2M1 nibble × raw per-group FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = e2m1_decode(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// mxfp8 (E4M3 weights, block 32, E8M0 pow-2 scale) cooperative patch embed.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    // 8-bit: weight is [hidden, patch_dim] u8 (1 byte/tap); one E8M0 scale byte
    // per block → row stride patch_dim/block_size.
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxfp8 e4m3 dequant) — E4M3 byte × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = e4m3_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// mxfp8 (E5M2 weights, block 32, E8M0 pow-2 scale) cooperative patch embed.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxfp8 e5m2 dequant) — E5M2 byte × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = e5m2_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// nvfp8 (E4M3 weights, block 16, raw per-block FP32 scale) cooperative patch embed.
///
/// fp8_e4m3 reuses this kernel (same 8-bit-E4M3 + FP32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (nvfp8 dequant) — E4M3 byte × raw per-block FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = e4m3_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// Legacy fp8 (E5M2 weights, group 32, raw per-group FP32 scale) patch embed.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (fp8 e5m2 dequant) — E5M2 byte × raw per-group FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = e5m2_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// Symmetric int8 (8-bit codes, group 64, raw per-group FP32 scale) patch embed.
///
/// Decode is sign-extend → `code · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (int8 dequant) — sign-extended code × raw per-group FP32 ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]);
            let decoded = int8_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

// ── Symmetric sub-byte integer patch-embed MMA (int2/3/4/5/6 + MXINT2..6) ───
// The projection-weight element is a signed N-bit two's-complement code,
// tight-bit-packed LSB-first into a FLAT global u32 bit-stream by
// `quant::format::pack` (element with global index `idx = h*patch_dim + kt`
// lives at bit `idx · BITS`). These kernels reuse `mt_int8_patch_embed_mma`'s
// dispatch geometry, threadgroup-memory layout, 8×8 frag mapping, the implicit
// patch-unfold A-load, and the MMA inner loop **verbatim** — only the per-tap B
// *dequant* staging changes. The B-load mirrors the 8-bit lane mapping
// (`b_k_base = (lane%4)·8`, each lane stages 8 contiguous taps `kb + b_k_base +
// i`), but instead of reading one byte it decodes each element from the global
// bit-stream with a straddle-aware two-word read + float sign-extend (subtract
// 2^N when the top bit is set; `$half`/`$full` are 2^(N-1) / 2^N), mirroring
// `mlx::block_scaled_mma`'s GPU-verified `int_qmm_mma_*` macros. The in-bounds
// mask (`select(kt < patch_dim, decoded, 0)`) is preserved; on the masked path
// `kt_safe = 0` keeps the bit-stream read in range. `$half`/`$full` are passed
// as literals to keep the constexpr math out of the DSL shift operands.

/// FP32-scaled symmetric int patch-embed MMA (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP32 scale, fed to the simdgroup-matrix matmul.
/// The B bit offset is computed from the tap's GLOBAL index
/// (`(h·patch_dim + kt)·BITS`), matching `quant::format::pack`'s flat LSB-first
/// stream for any `patch_dim`.
macro_rules! int_patch_embed_mma_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            image: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f32>,
            bias: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] patch_h: u32,
            #[constexpr] patch_w: u32,
            #[constexpr] hidden: u32,
            #[constexpr] block_size: u32,
        ) {
            let h_tile = tgid_x;
            let pat_tile = tgid_y;
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
            let phw = patch_h * patch_w;
            let patch_dim = in_ch * phw;
            let patches_w = in_w / patch_w;
            let input_plane = in_h * in_w;
            let a_pat_row = lane_in_tg / 4u32;
            let a_k_quad = lane_in_tg & 3u32;
            let a_k_base = a_k_quad * 8u32;
            let global_pat = pat_tile * 32u32 + a_pat_row;
            let py0 = (global_pat / patches_w) * patch_h;
            let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
            let b_h_row = lane_in_tg / 4u32;
            let b_k_quad = lane_in_tg & 3u32;
            let b_k_base = b_k_quad * 8u32;
            let global_h = h_tile * 32u32 + b_h_row;
            let n_blocks = patch_dim / block_size;
            let sb_base = global_h * n_blocks;
            // Global element index of this row's first tap (flat bit-stream is
            // row-major: element (h, kt) at bit (h·patch_dim + kt)·bits).
            let w_global_row_base = global_h * patch_dim;
            for kb in range(0u32, patch_dim, 32u32) {
                // ─ 1. Coop A load (implicit patch unfold gather) — verbatim ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + a_k_base + i;
                    let in_bounds = kt < patch_dim;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let ic = kt_safe / phw;
                    let rem_kt = kt_safe - ic * phw;
                    let py = rem_kt / patch_w;
                    let px = rem_kt - py * patch_w;
                    let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
                    let raw = load(image[img_idx]).cast::<f32>();
                    let val = select(in_bounds, raw, 0.0f32).cast::<T>();
                    threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
                }
                // ─ 2. Coop B load (int bit-stream dequant) — sign-extended code
                //   × per-group FP32 scale. Same lane→tap mapping as int8. ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < patch_dim;
                    let kt_safe = select(in_bounds, kt, 0u32);
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
                    let code = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[sb_base + kt_safe / block_size]);
                    let decoded = code * scale;
                    let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
            // ── 4. Add bias and write 4 C frags to global out — verbatim ──
            let out_pat_base = pat_tile * 32u32 + sm * 16u32;
            let out_h_base = h_tile * 32u32 + sn * 16u32;
            let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
            let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
            let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
            let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + fn0],
                (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + fn1],
                (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
                (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
                (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
                (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
                (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
                (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
                (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
            );
        }
    };
}
int_patch_embed_mma_f32!(mt_int2_patch_embed_mma, 2u32, 2u32, 4.0f32);
int_patch_embed_mma_f32!(mt_int3_patch_embed_mma, 3u32, 4u32, 8.0f32);
int_patch_embed_mma_f32!(mt_int4_patch_embed_mma, 4u32, 8u32, 16.0f32);
int_patch_embed_mma_f32!(mt_int5_patch_embed_mma, 5u32, 16u32, 32.0f32);
int_patch_embed_mma_f32!(mt_int6_patch_embed_mma, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int patch-embed MMA (MXINT2/3/4/5/6): per-element
/// bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`, fed to the
/// simdgroup-matrix matmul. Same straddle-aware global-bit-offset decode and
/// dispatch geometry as `int_patch_embed_mma_f32`; only the scale axis differs
/// (one u8 exponent per block).
macro_rules! int_patch_embed_mma_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            image: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<u8>,
            bias: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] patch_h: u32,
            #[constexpr] patch_w: u32,
            #[constexpr] hidden: u32,
            #[constexpr] block_size: u32,
        ) {
            let h_tile = tgid_x;
            let pat_tile = tgid_y;
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
            let phw = patch_h * patch_w;
            let patch_dim = in_ch * phw;
            let patches_w = in_w / patch_w;
            let input_plane = in_h * in_w;
            let a_pat_row = lane_in_tg / 4u32;
            let a_k_quad = lane_in_tg & 3u32;
            let a_k_base = a_k_quad * 8u32;
            let global_pat = pat_tile * 32u32 + a_pat_row;
            let py0 = (global_pat / patches_w) * patch_h;
            let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
            let b_h_row = lane_in_tg / 4u32;
            let b_k_quad = lane_in_tg & 3u32;
            let b_k_base = b_k_quad * 8u32;
            let global_h = h_tile * 32u32 + b_h_row;
            let n_blocks = patch_dim / block_size;
            let sb_base = global_h * n_blocks;
            // Global element index of this row's first tap (flat bit-stream is
            // row-major: element (h, kt) at bit (h·patch_dim + kt)·bits).
            let w_global_row_base = global_h * patch_dim;
            for kb in range(0u32, patch_dim, 32u32) {
                // ─ 1. Coop A load (implicit patch unfold gather) — verbatim ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + a_k_base + i;
                    let in_bounds = kt < patch_dim;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let ic = kt_safe / phw;
                    let rem_kt = kt_safe - ic * phw;
                    let py = rem_kt / patch_w;
                    let px = rem_kt - py * patch_w;
                    let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
                    let raw = load(image[img_idx]).cast::<f32>();
                    let val = select(in_bounds, raw, 0.0f32).cast::<T>();
                    threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
                }
                // ─ 2. Coop B load (int bit-stream dequant) — sign-extended code
                //   × E8M0 pow-2 block scale. Same lane→tap mapping as int8. ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < patch_dim;
                    let kt_safe = select(in_bounds, kt, 0u32);
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
                    let code = select(q >= $half, qf - $full, qf); // sign-extend
                    let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    let decoded = code * scale;
                    let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
            // ── 4. Add bias and write 4 C frags to global out — verbatim ──
            let out_pat_base = pat_tile * 32u32 + sm * 16u32;
            let out_h_base = h_tile * 32u32 + sn * 16u32;
            let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
            let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
            let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
            let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + fn0],
                (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + fn1],
                (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
                (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
                (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
                (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
                (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
                (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
                (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
            );
        }
    };
}
int_patch_embed_mma_e8m0!(mt_mxint2_patch_embed_mma, 2u32, 2u32, 4.0f32);
int_patch_embed_mma_e8m0!(mt_mxint3_patch_embed_mma, 3u32, 4u32, 8.0f32);
int_patch_embed_mma_e8m0!(mt_mxint4_patch_embed_mma, 4u32, 8u32, 16.0f32);
int_patch_embed_mma_e8m0!(mt_mxint5_patch_embed_mma, 5u32, 16u32, 32.0f32);
int_patch_embed_mma_e8m0!(mt_mxint6_patch_embed_mma, 6u32, 32u32, 64.0f32);

/// MXINT8 patch embed (8-bit symmetric codes, byte layout, block 32, E8M0 pow-2
/// block scale `2^(bits-127)`). Identical geometry and B-load mapping to
/// `mt_int8_patch_embed_mma` (one byte per code, 8 contiguous bytes per lane);
/// only the scale axis is E8M0 instead of a raw FP32.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxint8 dequant) — sign-extended code × E8M0 pow-2 scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let sbits = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let decoded = int8_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

// ── FP16-scale twins of the FP32-scaled formats ─────────────────────────────
// These mirror their FP32-scaled siblings *byte-for-byte* — identical implicit
// patch-unfold A-load, threadgroup/simdgroup allocation, 4-frag × 4-k-inner MMA
// inner loop, bias-added C-store, dispatch grid (`[hidden/32, num_patches/32,
// 1]`, tpg 128), and per-format element decode. The ONLY difference is the scale
// axis: the `scales` tensor binds as `Tensor<f16>` (was `Tensor<f32>`) and each
// scale read appends `.cast::<f32>()` so the dequant math is unchanged. The GPU-
// verified reference for this scale-read pattern is `mlx::block_scaled_dequant`'s
// `mt_nvfp8_f16_dequant` / `mt_fp4_f16_dequant` / `mt_fp8_e5m2_f16_dequant` /
// `int_dequant_f16!` / `mt_int8_f16_dequant`.

/// nvfp8 with an FP16 per-block scale (E4M3 weights, block 16). FP16-scale twin
/// of `mt_nvfp8_patch_embed_mma`; also serves `QFormat::Fp8E4m3F16` (same 8-bit-
/// E4M3 + scale shape, exactly as `mt_nvfp8_patch_embed_mma` serves `Fp8E4m3`).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (nvfp8 dequant) — E4M3 byte × FP16 per-block scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = e4m3_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// fp4 with an FP16 per-group scale (E2M1 weights, group 32). FP16-scale twin of
/// `mt_fp4_patch_embed_mma`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_pack_row_base = global_h * packs_per_row;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (fp4 dequant) — E2M1 nibble × FP16 per-group scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
            let nib = (pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32;
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = e2m1_decode(nib) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// Legacy fp8 e5m2 with an FP16 per-group scale (E5M2 weights, group 32). FP16-
/// scale twin of `mt_fp8_e5m2_patch_embed_mma`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (fp8 e5m2 dequant) — E5M2 byte × FP16 per-group scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = e5m2_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// Symmetric int8 with an FP16 per-group scale (8-bit codes, group 64). FP16-
/// scale twin of `mt_int8_patch_embed_mma`. Decode is sign-extend → `code · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_patch_embed_mma<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    let n_blocks = patch_dim / block_size;
    let w_row_base = global_h * patch_dim;
    let sb_base = global_h * n_blocks;
    for kb in range(0u32, patch_dim, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (int8 dequant) — sign-extended code × FP16 per-group scale ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let code = load(weight[w_row_base + kt_safe]).cast::<u32>();
            let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
            let decoded = int8_decode(code) * scale;
            let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}

/// FP16-scaled symmetric int patch-embed MMA (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP16 scale, fed to the simdgroup-matrix matmul.
/// Byte-for-byte clone of `int_patch_embed_mma_f32!` — same straddle-aware
/// global-bit-offset decode, dispatch geometry, staging, and MMA inner loop;
/// only the scale binds as `Tensor<f16>` and is `.cast::<f32>()` on read.
macro_rules! int_patch_embed_mma_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            image: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f16>,
            bias: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] patch_h: u32,
            #[constexpr] patch_w: u32,
            #[constexpr] hidden: u32,
            #[constexpr] block_size: u32,
        ) {
            let h_tile = tgid_x;
            let pat_tile = tgid_y;
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
            let phw = patch_h * patch_w;
            let patch_dim = in_ch * phw;
            let patches_w = in_w / patch_w;
            let input_plane = in_h * in_w;
            let a_pat_row = lane_in_tg / 4u32;
            let a_k_quad = lane_in_tg & 3u32;
            let a_k_base = a_k_quad * 8u32;
            let global_pat = pat_tile * 32u32 + a_pat_row;
            let py0 = (global_pat / patches_w) * patch_h;
            let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
            let b_h_row = lane_in_tg / 4u32;
            let b_k_quad = lane_in_tg & 3u32;
            let b_k_base = b_k_quad * 8u32;
            let global_h = h_tile * 32u32 + b_h_row;
            let n_blocks = patch_dim / block_size;
            let sb_base = global_h * n_blocks;
            // Global element index of this row's first tap (flat bit-stream is
            // row-major: element (h, kt) at bit (h·patch_dim + kt)·bits).
            let w_global_row_base = global_h * patch_dim;
            for kb in range(0u32, patch_dim, 32u32) {
                // ─ 1. Coop A load (implicit patch unfold gather) — verbatim ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + a_k_base + i;
                    let in_bounds = kt < patch_dim;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let ic = kt_safe / phw;
                    let rem_kt = kt_safe - ic * phw;
                    let py = rem_kt / patch_w;
                    let px = rem_kt - py * patch_w;
                    let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
                    let raw = load(image[img_idx]).cast::<f32>();
                    let val = select(in_bounds, raw, 0.0f32).cast::<T>();
                    threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
                }
                // ─ 2. Coop B load (int bit-stream dequant) — sign-extended code
                //   × per-group FP16 scale. Same lane→tap mapping as int8. ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < patch_dim;
                    let kt_safe = select(in_bounds, kt, 0u32);
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
                    let code = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[sb_base + kt_safe / block_size]).cast::<f32>();
                    let decoded = code * scale;
                    let val = select(in_bounds, decoded, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
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
            // ── 4. Add bias and write 4 C frags to global out — verbatim ──
            let out_pat_base = pat_tile * 32u32 + sm * 16u32;
            let out_h_base = h_tile * 32u32 + sn * 16u32;
            let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
            let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
            let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
            let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + fn0],
                (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + fn1],
                (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
                (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
            );
            store(
                out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
                (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
                (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
                (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
                (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
            );
            store(
                out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
                (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
            );
        }
    };
}
int_patch_embed_mma_f16!(mt_int2_f16_patch_embed_mma, 2u32, 2u32, 4.0f32);
int_patch_embed_mma_f16!(mt_int3_f16_patch_embed_mma, 3u32, 4u32, 8.0f32);
int_patch_embed_mma_f16!(mt_int4_f16_patch_embed_mma, 4u32, 8u32, 16.0f32);
int_patch_embed_mma_f16!(mt_int5_f16_patch_embed_mma, 5u32, 16u32, 32.0f32);
int_patch_embed_mma_f16!(mt_int6_f16_patch_embed_mma, 6u32, 32u32, 64.0f32);

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Bounded zig-zag ramp (keeps f16/bf16 in range), identical to the dense
    /// patch_embed_mma helper.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Explicit unfold + projection + bias oracle, run over the *dequantized*
    /// weight `[hidden, patch_dim]` (col = ic*patch_h*patch_w + py*patch_w + px).
    /// Output `[num_patches, hidden]`, f32. The SAME dense math as
    /// patch_embed_mma.rs's `naive_patch_embed_mma`.
    #[allow(clippy::too_many_arguments)]
    fn naive_patch_embed(
        image: &[f32],
        weight: &[f32],
        bias: &[f32],
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let patch_dim = in_ch * patch_h * patch_w;
        let input_plane = in_h * in_w;
        let patches_h = in_h / patch_h;
        let patches_w = in_w / patch_w;
        let num_patches = patches_h * patches_w;
        let mut out = vec![0.0f32; num_patches * hidden];
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                let pat = ph * patches_w + pw;
                let py0 = ph * patch_h;
                let px0 = pw * patch_w;
                for h in 0..hidden {
                    let mut acc = bias[h];
                    for ic in 0..in_ch {
                        for py in 0..patch_h {
                            for px in 0..patch_w {
                                let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
                                let w_idx =
                                    h * patch_dim + ic * patch_h * patch_w + py * patch_w + px;
                                acc += image[img_idx] * weight[w_idx];
                            }
                        }
                    }
                    out[pat * hidden + h] = acc;
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the `[hidden, patch_dim]` projection
    /// weight via the shared codec, dequantize for the oracle, and run the dense
    /// patch_embed_mma math over the dequantized weight. Mirrors
    /// patch_embed_mma.rs's `mma_setup` grid + KernelMode, just swapping the
    /// dense weight for a quantized one (the bias stays `T`).
    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> TestSetup {
        let num_patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        assert_eq!(hidden % 32, 0, "hidden must be a multiple of 32 for the MMA tile");
        assert_eq!(num_patches % 32, 0, "num_patches must be a multiple of 32");
        // patch_dim is the contraction K — must be a multiple of the 32 MMA
        // K-tile and of the format's block_size.
        assert_eq!(patch_dim % 32, 0, "patch_dim must be a multiple of the 32 MMA K-tile");
        assert_eq!(patch_dim % fmt.block_size(), 0, "patch_dim must be a multiple of block_size");
        let n_out = num_patches * hidden;
        let image_f = ramp(in_ch * in_h * in_w, 13, 2.0);
        // Quantize the [hidden, patch_dim] projection weight via the shared codec.
        let weight_f = ramp(hidden * patch_dim, 11, 2.0);
        let p = crate::quant::format::pack(fmt, &weight_f, hidden, patch_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, hidden, patch_dim);
        let bias_f = ramp(hidden, 5, 1.0);
        let image = unpack_f32(&pack_f32(&image_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected =
            naive_patch_embed(&image, &wdq, &bias, in_ch, in_h, in_w, patch_h, patch_w, hidden);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types
        // (these are exactly equivalent to the old per-format lists for the
        // pre-existing formats — 4-bit collapses to the u32 branch).
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        // FP32 scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one
        // byte. Driven off the format so each new precision picks the right type.
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("image", pack_f32(&image_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (hidden / 32) as u32,
            (num_patches / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // Dims: in_ch=4, patch 4×4 → patch_dim=64 (2 K-blocks of 32; divisible by
    // 16/32/64). 32×32 image → 64 patches; hidden=32. Two 32×32 patch tiles.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_patch_embed_mma::kernel_ir_for(dt), QFormat::Fp4, 4, 32, 32, 4, 4, 32, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e4m3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). patch_dim=64 is a multiple of 32
    // (the MMA K-tile) and of both block sizes, and `patch_dim*bits % 32 == 0`
    // for every width (64 is a multiple of 32), so each weight row's tight
    // bit-stream is word-aligned. The kernel and oracle share the codec, so the
    // GPU output tracks the dequant-then-projection reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // FP16-scale twins. Same dims / oracle as their FP32 siblings — only the
    // scale tensor binds as f16 (`mma_setup` picks DType::F16 off `scale_kind`).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
}

/// Realistic vision-encoder benches — the M≥32 simdgroup-matrix throughput
/// path for quantized-weight patch embedding. Random packed buffers
/// (throughput is data-independent). Shape: 8×8 patch, 4 channels
/// (patch_dim = 256, divisible by 16/32/64 block sizes and the 32 K-tile),
/// 256×256 image → 32×32 = 1024 patches, hidden = 1024.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> BenchSetup {
        let num_patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        let n_out = num_patches * hidden;
        let n_blocks = hidden * (patch_dim / fmt.block_size());
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words
        // (4-bit `bitstream_words` collapses to the old `n/8`, so no regression).
        let n_weight = hidden * patch_dim;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_weight, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n_weight, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("image", in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", hidden, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((hidden / 32) as u32, (num_patches / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // 2 * num_patches * hidden * in_ch * patch_h * patch_w
            // (conv2d formula: N=1, Co=hidden, Ho*Wo=num_patches)
            .flops(
                2 * (num_patches as u64)
                    * (hidden as u64)
                    * (in_ch as u64)
                    * (patch_h as u64)
                    * (patch_w as u64),
            )
    }

    #[bench(name = "ffai/patch_embed_mma_block/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp8_e4m3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). patch_dim=256 is a multiple of 32 and every block
    // size, and word-aligned for every bit width.
    #[bench(name = "ffai/patch_embed_mma_block/int2", dtypes = [f32, f16, bf16])]
    fn bench_int2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int3", dtypes = [f32, f16, bf16])]
    fn bench_int3_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int4", dtypes = [f32, f16, bf16])]
    fn bench_int4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int5", dtypes = [f32, f16, bf16])]
    fn bench_int5_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int6", dtypes = [f32, f16, bf16])]
    fn bench_int6_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxint2", dtypes = [f32, f16, bf16])]
    fn bench_mxint2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxint3", dtypes = [f32, f16, bf16])]
    fn bench_mxint3_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxint4", dtypes = [f32, f16, bf16])]
    fn bench_mxint4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxint5", dtypes = [f32, f16, bf16])]
    fn bench_mxint5_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxint6", dtypes = [f32, f16, bf16])]
    fn bench_mxint6_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/mxint8", dtypes = [f32, f16, bf16])]
    fn bench_mxint8_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }

    // FP16-scale twins — same shapes / FLOPs as their FP32 siblings; only the
    // scale buffer binds as f16 (`mma_bench` picks DType::F16 off `scale_kind`).
    #[bench(name = "ffai/patch_embed_mma_block/nvfp8_f16", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/fp8_e4m3_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/fp4_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/fp8_e5m2_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int2_f16", dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int3_f16", dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int3_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int4_f16", dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int5_f16", dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int5_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int6_f16", dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int6_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(name = "ffai/patch_embed_mma_block/int8_f16", dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
}

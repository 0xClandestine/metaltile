//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight cooperative (MMA) 3D convolution** — the
//! simdgroup-matrix counterpart of `ffai/conv3d_mma.rs`, with a quantized
//! filter. It is the 3D analogue of `mlx/block_scaled_mma.rs` and the
//! MMA-throughput sibling of `ffai/conv3d_block_scaled.rs`.
//!
//! The dense `conv3d_mma` treats the conv as a GEMM whose A matrix is an
//! implicit im2col over `(kd, kh, kw, ic)` gather indices and whose B matrix
//! is the filter `[out_ch, total_k]` (`total_k = in_ch · kd · kh · kw`).
//! Here only the **B-load (weight staging)** changes: instead of loading a
//! dense `T` filter element, each lane decodes a block-scaled quantized code
//! and multiplies by its per-block scale. The implicit-im2col **A-load**,
//! threadgroup-memory layout, 8×8 frag mapping, MMA inner loop, C-store,
//! grid, and `KernelMode` are **copied verbatim** from `conv3d_mma.rs`.
//!
//! ## Implicit im2col as a matmul (identical to `conv3d_mma`)
//!
//!   out[BN_voxels, BM_oc] = A[BN_voxels, BK] × B[BK, BM_oc]
//!
//! where `BK = in_ch · kd · kh · kw`, `BN = batch · out_d · out_h · out_w`,
//! `BM = out_ch`. Tile geometry: tpg = 128 = 4 SG × 32 lanes, BM = BN = 32,
//! BK = 32, grid `[out_ch/32, (batch·out_d·out_h·out_w)/32, 1]`. Constraints:
//! stride = 1, dilation = 1, padding = 0; `out_ch` and the voxel count both
//! divisible by 32; NCDHW input, OIDHW filter. **No bias** (the dense kernel
//! has none).
//!
//! ## Quantized filter B-load
//!
//! The filter is the 2-D matrix `[out_ch, total_k]`, block-scaled along
//! `total_k`. For output channel `oc` (= `b_oc_row + oc_tile·32`, reused from
//! the dense kernel) and a tap `kt`, the dense filter element is replaced by
//!
//!   element_decode(code[oc, kt]) · scale[oc, kt / block_size]   (× global for nvfp4)
//!
//! with `kt = ((ic·kd + kz)·kh + ky)·kw + kx`. 4-bit codes pack `[out_ch,
//! total_k/8]` u32 (8 nibbles/word — word `oc·(total_k/8) + kt/8`, shift
//! `(kt%8)·4`); 8-bit codes are `[out_ch, total_k]` u8 (byte `oc·total_k +
//! kt`). Decode is per-tap (kt-by-kt), keeping the dense in-bounds masking and
//! the same `bs` store position. `total_k` is a multiple of `block_size`
//! (4-bit `block_size` a multiple of 8) and of the 32-wide MMA K-tile.
//!
//! fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s vs a
//! `quant::format::dequant` oracle running the dense conv3d_mma math.

use metaltile::kernel;

/// mxfp4 quantized-weight cooperative conv3d — E2M1 filter (block 32),
/// E8M0 pow-2 scale. Stride=1, dilation=1, pad=0.
///
/// Grid `[out_ch/32, (batch·out_d·out_h·out_w)/32, 1]`, tpg = 128.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    // BM (oc-axis) tile = tgid_x * 32, BN (voxel-axis) tile = tgid_y * 32.
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // ── 8×8 frag lane mapping (Apple steel_gemm layout) ──────────────────
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // ── TG memory: A and B tiles, skewed stride = 36 ─────────────────────
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T);
    threadgroup_alloc("bs", 1152, T);
    // ── Accumulator frags ─────────────────────────────────────────────────
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
    // ── Precompute K-space extents ────────────────────────────────────────
    let khw = kh * kw;
    let kdhw = kd * khw; // taps per input channel
    let total_k = in_ch * kdhw; // total tap dimension
    // ── Voxel-axis im2col decode for this TG's A rows ────────────────────
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    // Coop A-load lane assignment: lane_in_tg = pv_row * 4 + k_quad.
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    // Base device offset for this voxel's batch + channel-0 position.
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    // Coop B-load (quantized weight). Filter as [out_ch, total_k], block-scaled
    // along total_k; 4-bit codes pack 8 nibbles per u32 word.
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let packs_per_row = total_k / 8u32;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_pack_base = global_oc * packs_per_row;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    // ── K-block loop ──────────────────────────────────────────────────────
    // K-tail handling: `total_k = in_ch * kd * kh * kw` rarely lands on a
    // multiple of 32. The A/B coop loads mask out-of-bound K-taps and clamp
    // the gather/decode index to 0 on OOB so we never read past the buffers;
    // zero contributions leave the partial-K MMA accumulator correct.
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit 5D im2col gather) ───────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            // Decompose kt_safe into (ic, kz, ky, kx).
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            // Gather indices (stride=1, pad=0).
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxfp4 filter: E2M1 nibble × E8M0 pow-2 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            // 4-bit code: word oc*(total_k/8)+kt/8, nibble (kt%8)*4.
            let word_idx = w_oc_pack_base + kt_safe / 8u32;
            let shift = (kt_safe & 7u32) * 4u32;
            let nib = (load(weight[word_idx]) >> shift) & 0xFu32;
            // E8M0 block scale for this tap.
            let g = kt_safe / block_size;
            let sbits = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let dq = e2m1_decode(nib) * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        // ─ 3. MMA inner loop (4 k-inner × 4 frags = 16 MMAs / SG) ──────
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
    // ── 4. Write 4 C frags to global out ─────────────────────────────────
    // out layout: [batch * out_d * out_h * out_w, out_ch].
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// nvfp4 quantized-weight cooperative conv3d — E2M1 filter (block 16),
/// E4M3 micro-scale × global FP32. Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let packs_per_row = total_k / 8u32;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_pack_base = global_oc * packs_per_row;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (nvfp4: E2M1 nibble × E4M3 micro-scale × global) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let word_idx = w_oc_pack_base + kt_safe / 8u32;
            let shift = (kt_safe & 7u32) * 4u32;
            let nib = (load(weight[word_idx]) >> shift) & 0xFu32;
            let g = kt_safe / block_size;
            // global multiplied LAST (two-level scaling).
            let scale = e4m3_decode(load(scales[w_oc_blk_base + g]).cast::<u32>()) * global;
            let dq = e2m1_decode(nib) * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// mxfp8 (E4M3) quantized-weight cooperative conv3d — 8-bit filter (block 32),
/// E8M0 pow-2 scale. Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    // 8-bit codes are one byte each: filter byte = oc*total_k + kt.
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxfp8 e4m3: 8-bit byte × E8M0 pow-2 scale) ────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = e4m3_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let sbits = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// mxfp8 (E5M2) quantized-weight cooperative conv3d — 8-bit filter (block 32),
/// E8M0 pow-2 scale. Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (mxfp8 e5m2: 8-bit byte × E8M0 pow-2 scale) ────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = e5m2_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let sbits = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// Legacy fp8 (E5M2) quantized-weight cooperative conv3d — 8-bit filter
/// (group 32), per-group FP32 scale. Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (legacy fp8 e5m2: 8-bit byte × per-group f32 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = e5m2_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]);
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// nvfp8 quantized-weight cooperative conv3d — E4M3 filter (block 16),
/// per-block FP32 scale. Stride=1, dilation=1, pad=0.
///
/// fp8_e4m3 reuses this kernel (same 8-bit-E4M3 + f32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (nvfp8 e4m3: 8-bit byte × per-block f32 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = e4m3_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]);
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// fp4 quantized-weight cooperative conv3d — E2M1 filter (group 32),
/// per-group FP32 scale. Stride=1, dilation=1, pad=0.
///
/// Verified on f32/f16/bf16 against the `quant::format` oracle.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let packs_per_row = total_k / 8u32;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_pack_base = global_oc * packs_per_row;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (legacy fp4: E2M1 nibble × per-group f32 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let word_idx = w_oc_pack_base + kt_safe / 8u32;
            let shift = (kt_safe & 7u32) * 4u32;
            let nib = (load(weight[word_idx]) >> shift) & 0xFu32;
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]);
            let dq = e2m1_decode(nib) * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// Symmetric int8 quantized-weight cooperative conv3d — 8-bit codes (group 64),
/// per-group FP32 scale. Decode is sign-extend → `code · scale`. Stride=1,
/// dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (symmetric int8: sign-extend × per-group f32 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = int8_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]);
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ── Symmetric sub-byte integer MMA conv3d (int2/3/4/5/6 + MXINT2..6) ─────────
// The filter element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into a FLAT global u32 bit-stream by `quant::format::pack`: for the
// 2-D filter `[out_ch, C]` (`C = in_ch·kd·kh·kw`), the element at output channel
// `oc` and contraction tap `kt` has global index `oc·C + kt` and lives at bit
// `(oc·C + kt)·BITS`. These kernels reuse `mt_int8_conv3d_mma`'s dispatch
// geometry, threadgroup-memory layout, 8×8 frag mapping, MMA inner loop, and the
// implicit-im2col A-load **verbatim** — only the per-tap B (filter) *decode*
// changes. Instead of reading one byte, each lane decodes its tap from the global
// bit-stream with a straddle-aware two-word read + float sign-extend (subtract
// 2^N when the top bit is set; `$half`/`$full` are 2^(N-1) / 2^N), mirroring the
// committed `mlx/block_scaled_mma.rs` `int_qmm_mma_*` macros and the GPU-verified
// `block_scaled_dequant` decode. The global-offset form is robust for any `C`
// (no per-row word alignment is assumed). `$half`/`$full` are passed as literals
// to keep the constexpr math out of the DSL shift operands. The dense kernel's
// in-bounds masking (clamp `kt_safe` to 0 on OOB, zero the staged value) and the
// `bs` store position are preserved exactly.

/// FP32-scaled symmetric int MMA conv3d (int2/3/4/5/6): per-tap bit-stream code
/// × per-group FP32 scale, fed to the simdgroup-matrix matmul over the implicit
/// im2col. The filter bit offset is `(global_oc·total_k + kt)·BITS`, matching
/// `quant::format::pack`'s flat LSB-first stream for any contraction length.
/// Stride=1, dilation=1, pad=0. Grid `[out_ch/32,
/// (batch·out_d·out_h·out_w)/32, 1]`, tpg = 128.
macro_rules! int_conv3d_mma_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_d: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_d: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kd: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] block_size: u32,
        ) {
            let oc_tile = tgid_x;
            let pv_tile = tgid_y;
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
            let khw = kh * kw;
            let kdhw = kd * khw;
            let total_k = in_ch * kdhw;
            let out_hw = out_h * out_w;
            let out_dhw = out_d * out_hw;
            let a_pv_row = lane_in_tg / 4u32;
            let a_k_quad = lane_in_tg & 3u32;
            let a_k_base = a_k_quad * 8u32;
            let global_pv = pv_tile * 32u32 + a_pv_row;
            let n_pv = global_pv / out_dhw;
            let rem_pv = global_pv - n_pv * out_dhw;
            let od_pv = rem_pv / out_hw;
            let rem_hw = rem_pv - od_pv * out_hw;
            let oh_pv = rem_hw / out_w;
            let ow_pv = rem_hw - oh_pv * out_w;
            let in_plane = in_h * in_w;
            let in_vol = in_d * in_plane;
            let in_n_stride = in_ch * in_vol;
            let pv_in_base = n_pv * in_n_stride;
            let b_oc_row = lane_in_tg / 4u32;
            let b_k_quad = lane_in_tg & 3u32;
            let b_k_base = b_k_quad * 8u32;
            let global_oc = oc_tile * 32u32 + b_oc_row;
            let n_blocks_per_row = total_k / block_size;
            // Global element index of this lane's filter row start: element
            // (oc, kt) lives at bit (oc·total_k + kt)·BITS in the flat stream.
            let w_oc_elem_base = global_oc * total_k;
            let w_oc_blk_base = global_oc * n_blocks_per_row;
            for kb in range(0u32, total_k, 32u32) {
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + a_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let ic = kt_safe / kdhw;
                    let rem_kt = kt_safe - ic * kdhw;
                    let kz = rem_kt / khw;
                    let rem_kh = rem_kt - kz * khw;
                    let ky = rem_kh / kw;
                    let kx = rem_kh - ky * kw;
                    let id = od_pv + kz;
                    let ih = oh_pv + ky;
                    let iw = ow_pv + kx;
                    let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
                    let raw = load(input[in_idx]).cast::<f32>();
                    let val = select(in_bounds, raw, 0.0f32).cast::<T>();
                    threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
                }
                // ─ 2. Coop B load (symmetric intN: bit-stream code × f32 scale) ──
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_oc_elem_base + kt_safe) * $bits;
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
                    let g = kt_safe / block_size;
                    let scale = load(scales[w_oc_blk_base + g]);
                    let dq = elem * scale;
                    let val = select(in_bounds, dq, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
                }
                threadgroup_barrier();
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
            let out_pv_base = pv_tile * 32u32 + sm * 16u32;
            let out_oc_base = oc_tile * 32u32 + sn * 16u32;
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }
    };
}
int_conv3d_mma_f32!(mt_int2_conv3d_mma, 2u32, 2u32, 4.0f32);
int_conv3d_mma_f32!(mt_int3_conv3d_mma, 3u32, 4u32, 8.0f32);
int_conv3d_mma_f32!(mt_int4_conv3d_mma, 4u32, 8u32, 16.0f32);
int_conv3d_mma_f32!(mt_int5_conv3d_mma, 5u32, 16u32, 32.0f32);
int_conv3d_mma_f32!(mt_int6_conv3d_mma, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int MMA conv3d (MXINT2/3/4/5/6): per-tap bit-stream
/// code × pow-2 (E8M0) block scale `2^(bits-127)`, fed to the simdgroup-matrix
/// matmul over the implicit im2col. Same straddle-aware global-bit-offset decode
/// and dispatch geometry as `int_conv3d_mma_f32`; only the scale axis differs
/// (one u8 exponent per block). Stride=1, dilation=1, pad=0. Grid
/// `[out_ch/32, (batch·out_d·out_h·out_w)/32, 1]`, tpg = 128.
macro_rules! int_conv3d_mma_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<u8>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_d: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_d: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kd: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] block_size: u32,
        ) {
            let oc_tile = tgid_x;
            let pv_tile = tgid_y;
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
            let khw = kh * kw;
            let kdhw = kd * khw;
            let total_k = in_ch * kdhw;
            let out_hw = out_h * out_w;
            let out_dhw = out_d * out_hw;
            let a_pv_row = lane_in_tg / 4u32;
            let a_k_quad = lane_in_tg & 3u32;
            let a_k_base = a_k_quad * 8u32;
            let global_pv = pv_tile * 32u32 + a_pv_row;
            let n_pv = global_pv / out_dhw;
            let rem_pv = global_pv - n_pv * out_dhw;
            let od_pv = rem_pv / out_hw;
            let rem_hw = rem_pv - od_pv * out_hw;
            let oh_pv = rem_hw / out_w;
            let ow_pv = rem_hw - oh_pv * out_w;
            let in_plane = in_h * in_w;
            let in_vol = in_d * in_plane;
            let in_n_stride = in_ch * in_vol;
            let pv_in_base = n_pv * in_n_stride;
            let b_oc_row = lane_in_tg / 4u32;
            let b_k_quad = lane_in_tg & 3u32;
            let b_k_base = b_k_quad * 8u32;
            let global_oc = oc_tile * 32u32 + b_oc_row;
            let n_blocks_per_row = total_k / block_size;
            // Global element index of this lane's filter row start: element
            // (oc, kt) lives at bit (oc·total_k + kt)·BITS in the flat stream.
            let w_oc_elem_base = global_oc * total_k;
            let w_oc_blk_base = global_oc * n_blocks_per_row;
            for kb in range(0u32, total_k, 32u32) {
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + a_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let ic = kt_safe / kdhw;
                    let rem_kt = kt_safe - ic * kdhw;
                    let kz = rem_kt / khw;
                    let rem_kh = rem_kt - kz * khw;
                    let ky = rem_kh / kw;
                    let kx = rem_kh - ky * kw;
                    let id = od_pv + kz;
                    let ih = oh_pv + ky;
                    let iw = ow_pv + kx;
                    let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
                    let raw = load(input[in_idx]).cast::<f32>();
                    let val = select(in_bounds, raw, 0.0f32).cast::<T>();
                    threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
                }
                // ─ 2. Coop B load (symmetric MXINT: bit-stream code × E8M0 scale) ─
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_oc_elem_base + kt_safe) * $bits;
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
                    let g = kt_safe / block_size;
                    let sbits = load(scales[w_oc_blk_base + g]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    let dq = elem * scale;
                    let val = select(in_bounds, dq, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
                }
                threadgroup_barrier();
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
            let out_pv_base = pv_tile * 32u32 + sm * 16u32;
            let out_oc_base = oc_tile * 32u32 + sn * 16u32;
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }
    };
}
int_conv3d_mma_e8m0!(mt_mxint2_conv3d_mma, 2u32, 2u32, 4.0f32);
int_conv3d_mma_e8m0!(mt_mxint3_conv3d_mma, 3u32, 4u32, 8.0f32);
int_conv3d_mma_e8m0!(mt_mxint4_conv3d_mma, 4u32, 8u32, 16.0f32);
int_conv3d_mma_e8m0!(mt_mxint5_conv3d_mma, 5u32, 16u32, 32.0f32);
int_conv3d_mma_e8m0!(mt_mxint6_conv3d_mma, 6u32, 32u32, 64.0f32);

/// MXINT8 quantized-weight cooperative conv3d — 8-bit symmetric codes (byte
/// layout, block 32), E8M0 pow-2 block scale `2^(bits-127)`. Identical geometry
/// and B-load mapping to `mt_int8_conv3d_mma` (one byte per filter tap); only the
/// scale axis is E8M0 instead of a raw FP32. Stride=1, dilation=1, pad=0.
///
/// Grid `[out_ch/32, (batch·out_d·out_h·out_w)/32, 1]`, tpg = 128.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    // 8-bit codes are one byte each: filter byte = oc*total_k + kt.
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (symmetric mxint8: 8-bit byte × E8M0 pow-2 scale) ──
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = int8_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let sbits = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ── FP16-scale twins (Nvfp8F16 / Fp4F16 / Fp8E5m2F16 / Int2..6F16 / Int8F16) ──
// Byte-for-byte clones of the FP32-scaled kernels above, with the ONLY change
// being the scale tensor (`Tensor<f32>` → `Tensor<f16>`) and the scale read
// (`load(scales[...])` → `load(scales[...]).cast::<f32>()`). The element decode
// (E4M3 / E2M1 / E5M2 / int bit-stream sign-extend), weight indexing, dispatch
// geometry, threadgroup-memory layout (skewed stride 36), 8×8 frag mapping, MMA
// inner loop, C-store, grid, and `KernelMode` are IDENTICAL to their twins.
// `Fp8E4m3F16` reuses the `mt_nvfp8_f16_conv3d_mma` kernel (same 8-bit-E4M3 +
// f16-scale shape), exactly as `Fp8E4m3` reuses `mt_nvfp8_conv3d_mma`. The f16
// scale read mirrors the GPU-verified `block_scaled_dequant` f16 references.

/// nvfp8 (FP16-scale twin) quantized-weight cooperative conv3d — E4M3 filter
/// (block 16), per-block FP16 scale. Clone of `mt_nvfp8_conv3d_mma` with the
/// scale tensor as `Tensor<f16>` and the scale read `.cast::<f32>()`-promoted.
/// Stride=1, dilation=1, pad=0.
///
/// fp8_e4m3_f16 reuses this kernel (same 8-bit-E4M3 + f16-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (nvfp8 e4m3: 8-bit byte × per-block f16 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = e4m3_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// fp4 (FP16-scale twin) quantized-weight cooperative conv3d — E2M1 filter
/// (group 32), per-group FP16 scale. Clone of `mt_fp4_conv3d_mma` with the
/// scale tensor as `Tensor<f16>` and the scale read `.cast::<f32>()`-promoted.
/// Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let packs_per_row = total_k / 8u32;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_pack_base = global_oc * packs_per_row;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (fp4: E2M1 nibble × per-group f16 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let word_idx = w_oc_pack_base + kt_safe / 8u32;
            let shift = (kt_safe & 7u32) * 4u32;
            let nib = (load(weight[word_idx]) >> shift) & 0xFu32;
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let dq = e2m1_decode(nib) * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// Legacy fp8 (E5M2, FP16-scale twin) quantized-weight cooperative conv3d —
/// 8-bit filter (group 32), per-group FP16 scale. Clone of
/// `mt_fp8_e5m2_conv3d_mma` with the scale tensor as `Tensor<f16>` and the
/// scale read `.cast::<f32>()`-promoted. Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (legacy fp8 e5m2: 8-bit byte × per-group f16 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = e5m2_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

/// FP16-scaled symmetric int MMA conv3d (int2/3/4/5/6 FP16-scale twins): clone
/// of `int_conv3d_mma_f32` with the scale tensor as `Tensor<f16>` and the scale
/// read `.cast::<f32>()`-promoted. The per-tap bit-stream code decode (global
/// bit offset `(global_oc·total_k + kt)·BITS`, straddle-aware two-word read +
/// float sign-extend) and dispatch geometry are IDENTICAL to the FP32 twin.
/// Stride=1, dilation=1, pad=0. Grid `[out_ch/32,
/// (batch·out_d·out_h·out_w)/32, 1]`, tpg = 128.
macro_rules! int_conv3d_mma_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            input: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f16>,
            out: Tensor<T>,
            #[constexpr] in_ch: u32,
            #[constexpr] in_d: u32,
            #[constexpr] in_h: u32,
            #[constexpr] in_w: u32,
            #[constexpr] out_ch: u32,
            #[constexpr] out_d: u32,
            #[constexpr] out_h: u32,
            #[constexpr] out_w: u32,
            #[constexpr] kd: u32,
            #[constexpr] kh: u32,
            #[constexpr] kw: u32,
            #[constexpr] block_size: u32,
        ) {
            let oc_tile = tgid_x;
            let pv_tile = tgid_y;
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
            let khw = kh * kw;
            let kdhw = kd * khw;
            let total_k = in_ch * kdhw;
            let out_hw = out_h * out_w;
            let out_dhw = out_d * out_hw;
            let a_pv_row = lane_in_tg / 4u32;
            let a_k_quad = lane_in_tg & 3u32;
            let a_k_base = a_k_quad * 8u32;
            let global_pv = pv_tile * 32u32 + a_pv_row;
            let n_pv = global_pv / out_dhw;
            let rem_pv = global_pv - n_pv * out_dhw;
            let od_pv = rem_pv / out_hw;
            let rem_hw = rem_pv - od_pv * out_hw;
            let oh_pv = rem_hw / out_w;
            let ow_pv = rem_hw - oh_pv * out_w;
            let in_plane = in_h * in_w;
            let in_vol = in_d * in_plane;
            let in_n_stride = in_ch * in_vol;
            let pv_in_base = n_pv * in_n_stride;
            let b_oc_row = lane_in_tg / 4u32;
            let b_k_quad = lane_in_tg & 3u32;
            let b_k_base = b_k_quad * 8u32;
            let global_oc = oc_tile * 32u32 + b_oc_row;
            let n_blocks_per_row = total_k / block_size;
            // Global element index of this lane's filter row start: element
            // (oc, kt) lives at bit (oc·total_k + kt)·BITS in the flat stream.
            let w_oc_elem_base = global_oc * total_k;
            let w_oc_blk_base = global_oc * n_blocks_per_row;
            for kb in range(0u32, total_k, 32u32) {
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + a_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    let ic = kt_safe / kdhw;
                    let rem_kt = kt_safe - ic * kdhw;
                    let kz = rem_kt / khw;
                    let rem_kh = rem_kt - kz * khw;
                    let ky = rem_kh / kw;
                    let kx = rem_kh - ky * kw;
                    let id = od_pv + kz;
                    let ih = oh_pv + ky;
                    let iw = ow_pv + kx;
                    let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
                    let raw = load(input[in_idx]).cast::<f32>();
                    let val = select(in_bounds, raw, 0.0f32).cast::<T>();
                    threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
                }
                // ─ 2. Coop B load (symmetric intN: bit-stream code × f16 scale) ──
                for i in range(0u32, 8u32, 1u32) {
                    let kt = kb + b_k_base + i;
                    let in_bounds = kt < total_k;
                    let kt_safe = select(in_bounds, kt, 0u32);
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_oc_elem_base + kt_safe) * $bits;
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
                    let g = kt_safe / block_size;
                    let scale = load(scales[w_oc_blk_base + g]).cast::<f32>();
                    let dq = elem * scale;
                    let val = select(in_bounds, dq, 0.0f32).cast::<T>();
                    threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
                }
                threadgroup_barrier();
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
            let out_pv_base = pv_tile * 32u32 + sm * 16u32;
            let out_oc_base = oc_tile * 32u32 + sn * 16u32;
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f01, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f01, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
                simdgroup_elem_load(c_f10, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
                simdgroup_elem_load(c_f10, 1).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
                simdgroup_elem_load(c_f11, 0).cast::<T>(),
            );
            store(
                out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
                simdgroup_elem_load(c_f11, 1).cast::<T>(),
            );
        }
    };
}
int_conv3d_mma_f16!(mt_int2_f16_conv3d_mma, 2u32, 2u32, 4.0f32);
int_conv3d_mma_f16!(mt_int3_f16_conv3d_mma, 3u32, 4u32, 8.0f32);
int_conv3d_mma_f16!(mt_int4_f16_conv3d_mma, 4u32, 8u32, 16.0f32);
int_conv3d_mma_f16!(mt_int5_f16_conv3d_mma, 5u32, 16u32, 32.0f32);
int_conv3d_mma_f16!(mt_int6_f16_conv3d_mma, 6u32, 32u32, 64.0f32);

/// int8 (FP16-scale twin) quantized-weight cooperative conv3d — 8-bit codes
/// (group 64), per-group FP16 scale. Clone of `mt_int8_conv3d_mma` with the
/// scale tensor as `Tensor<f16>` and the scale read `.cast::<f32>()`-promoted.
/// Decode is sign-extend → `code · scale`. Stride=1, dilation=1, pad=0.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_conv3d_mma<T>(
    input: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
) {
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
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
    let khw = kh * kw;
    let kdhw = kd * khw;
    let total_k = in_ch * kdhw;
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_byte_base = global_oc * total_k;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    for kb in range(0u32, total_k, 32u32) {
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (symmetric int8: sign-extend × per-group f16 scale) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let elem = int8_decode(load(weight[w_oc_byte_base + kt_safe]).cast::<u32>());
            let g = kt_safe / block_size;
            let scale = load(scales[w_oc_blk_base + g]).cast::<f32>();
            let dq = elem * scale;
            let val = select(in_bounds, dq, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
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
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Bounded zig-zag ramp identical to the dense conv3d_mma helper.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 3D conv oracle, voxel-major output `[n_voxels, out_ch]`.
    /// stride=1, dilation=1, pad=0, no bias. The SAME dense math as
    /// `conv3d_mma.rs::naive_conv3d_mma`, run over the *dequantized* filter
    /// laid out as the 2-D matrix `[out_ch, C]`, `C = in_ch·kd·kh·kw`, with
    /// `col = ((ic·kd + kz)·kh + ky)·kw + kx`. All f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv3d_mma(
        input: &[f32],
        weight: &[f32],
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let out_hw = out_h * out_w;
        let out_dhw = out_d * out_hw;
        let n_voxels = batch * out_dhw;
        let in_plane = in_h * in_w;
        let in_vol = in_d * in_plane;
        let contraction = in_ch * kd * kh * kw;
        let mut out = vec![0.0f32; n_voxels * out_ch];
        for n in 0..batch {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let voxel = n * out_dhw + od * out_hw + oh * out_w + ow;
                        for oc in 0..out_ch {
                            let mut acc = 0.0f32;
                            for ic in 0..in_ch {
                                for kz in 0..kd {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let id = od + kz;
                                            let ih = oh + ky;
                                            let iw = ow + kx;
                                            let in_idx = n * in_ch * in_vol
                                                + ic * in_vol
                                                + id * in_plane
                                                + ih * in_w
                                                + iw;
                                            // Dequantized filter is contiguous over
                                            // col = ((ic*kd+kz)*kh+ky)*kw+kx per oc row.
                                            let col = ((ic * kd + kz) * kh + ky) * kw + kx;
                                            let w_idx = oc * contraction + col;
                                            acc += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                            out[voxel * out_ch + oc] = acc;
                        }
                    }
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the `[out_ch, C]` filter via the
    /// shared codec, dequantize for the oracle, and run the dense conv3d_mma
    /// math. Mirrors conv3d_mma.rs's `mma_setup` grid + KernelMode exactly.
    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> TestSetup {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_voxels = batch * out_d * out_h * out_w;
        assert_eq!(out_ch % 32, 0, "out_ch must be a multiple of 32 for the MMA tile");
        assert_eq!(n_voxels % 32, 0, "n_voxels must be a multiple of 32 for the MMA tile");
        let n_out = n_voxels * out_ch;
        // Contraction C = in_ch*kd*kh*kw — the quantized filter is [out_ch, C].
        let contraction = in_ch * kd * kh * kw;
        let input_f = ramp(batch * in_ch * in_d * in_h * in_w, 13, 2.0);
        // Quantize the [out_ch, C] filter via the shared codec.
        let w_f = ramp(out_ch * contraction, 11, 2.0);
        let p = crate::quant::format::pack(fmt, &w_f, out_ch, contraction);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, contraction);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        // Oracle: dense conv3d_mma over the dequantized filter row [out_ch, C].
        let expected =
            naive_conv3d_mma(&input, &wdq, batch, in_ch, in_d, in_h, in_w, out_ch, kd, kh, kw);
        // 8-bit codes bind as one uchar each; sub-byte codes pack into a u32
        // bit-stream. F32-scaled formats bind raw f32 scales; F16-scaled bind
        // raw f16 scales; E8M0/E4M3 are one byte. Axis-driven so every
        // int/mxint width routes correctly.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (out_ch / 32) as u32,
            (n_voxels / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // in_ch=8, kd=kh=kw=2 → C = 64 (÷ 16/32/64 and a multiple of the 32-wide
    // MMA K-tile); 5×5×5 volume → out 4×4×4, n_voxels=64 (2 tiles); out_ch=32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp4_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxfp4,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp4_conv3d_mma::kernel_ir_for(dt),
            QFormat::Nvfp4,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp4_conv3d_mma::kernel_ir_for(dt),
            QFormat::Fp4,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e4m3_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e5m2_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_conv3d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_conv3d_mma::kernel_ir_for(dt),
            QFormat::Nvfp8,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_conv3d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int8,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    // ── Symmetric sub-byte int (FP32 scale, group 64) + MXINT (E8M0, block 32) ──
    // Same conv shape as int8: in_ch=8, 2×2×2 kernel → C = 64 (a multiple of 32,
    // so each filter row's bit-stream is word-aligned for every width).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int2,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int3,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int4,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int5,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int6,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint2_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint3_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint4_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint5_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint6_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint8_conv3d_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    // ── FP16-scale twins (Nvfp8F16 / Fp4F16 / Fp8E4m3F16 / Fp8E5m2F16 /
    // Int2..6F16 / Int8F16). Same conv shape as the FP32 twins; only the scale
    // tensor is f16. fp8_e4m3_f16 reuses the nvfp8_f16 kernel. ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp4_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_conv3d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_f16_conv3d_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            1,
            8,
            5,
            5,
            5,
            32,
            2,
            2,
            2,
            dt,
        )
    }
}

/// Decode-shape benches: a realistic conv (in_ch=64, out_ch=256, 2×2×2 kernel →
/// C = 512, divisible by all block sizes 16/32/64). Reduction mode,
/// `grid_3d(out_ch/32, n_voxels/32, 1, [128,1,1])` like the dense conv3d_mma.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_voxels = batch * out_d * out_h * out_w;
        let n_out = n_voxels * out_ch;
        let contraction = in_ch * kd * kh * kw;
        // 8-bit codes are one byte each; sub-byte codes pack into a tight u32
        // bit-stream of `bitstream_words(total_elems, bits)` words. Axis-driven
        // so every int/mxint width sizes its code buffer correctly.
        let total_elems = out_ch * contraction;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (total_elems, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(total_elems, fmt.element_bits()), DType::U32)
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
            + batch * in_ch * in_d * in_h * in_w * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_d * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_ch / 32) as u32, (n_voxels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            // 2 * n_out * C; C = in_ch*kd*kh*kw is the per-output contraction.
            .flops(2 * n_out as u64 * contraction as u64)
            .with_shape_label(format!(
                "{} co={out_ch} do={out_d} ho={out_h} wo={out_w} C={contraction}",
                fmt.name()
            ))
    }

    macro_rules! conv3d_mma_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr, $name:literal) => {
            #[bench(name = $name, dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                // in_ch=64, out_ch=256, 2×2×2 kernel on a 9×9×9 volume →
                // out 8×8×8, n_voxels=512; C=512 (÷ 16/32/64).
                mma_bench($kernel(dt), $fmt, 1, 64, 9, 9, 9, 256, 2, 2, 2, dt)
            }
        };
    }
    conv3d_mma_bench_fmt!(
        bench_mxfp4,
        mt_mxfp4_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp4,
        "ffai/conv3d_mma_block/mxfp4"
    );
    conv3d_mma_bench_fmt!(
        bench_nvfp4,
        mt_nvfp4_conv3d_mma::kernel_ir_for,
        QFormat::Nvfp4,
        "ffai/conv3d_mma_block/nvfp4"
    );
    conv3d_mma_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp8E4,
        "ffai/conv3d_mma_block/mxfp8_e4m3"
    );
    conv3d_mma_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp8E5,
        "ffai/conv3d_mma_block/mxfp8_e5m2"
    );
    conv3d_mma_bench_fmt!(
        bench_nvfp8,
        mt_nvfp8_conv3d_mma::kernel_ir_for,
        QFormat::Nvfp8,
        "ffai/conv3d_mma_block/nvfp8"
    );
    conv3d_mma_bench_fmt!(
        bench_fp4,
        mt_fp4_conv3d_mma::kernel_ir_for,
        QFormat::Fp4,
        "ffai/conv3d_mma_block/fp4"
    );
    conv3d_mma_bench_fmt!(
        bench_fp8_e5m2,
        mt_fp8_e5m2_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E5m2,
        "ffai/conv3d_mma_block/fp8_e5m2"
    );
    conv3d_mma_bench_fmt!(
        bench_int8,
        mt_int8_conv3d_mma::kernel_ir_for,
        QFormat::Int8,
        "ffai/conv3d_mma_block/int8"
    );
    conv3d_mma_bench_fmt!(
        bench_int2,
        mt_int2_conv3d_mma::kernel_ir_for,
        QFormat::Int2,
        "ffai/conv3d_mma_block/int2"
    );
    conv3d_mma_bench_fmt!(
        bench_int3,
        mt_int3_conv3d_mma::kernel_ir_for,
        QFormat::Int3,
        "ffai/conv3d_mma_block/int3"
    );
    conv3d_mma_bench_fmt!(
        bench_int4,
        mt_int4_conv3d_mma::kernel_ir_for,
        QFormat::Int4,
        "ffai/conv3d_mma_block/int4"
    );
    conv3d_mma_bench_fmt!(
        bench_int5,
        mt_int5_conv3d_mma::kernel_ir_for,
        QFormat::Int5,
        "ffai/conv3d_mma_block/int5"
    );
    conv3d_mma_bench_fmt!(
        bench_int6,
        mt_int6_conv3d_mma::kernel_ir_for,
        QFormat::Int6,
        "ffai/conv3d_mma_block/int6"
    );
    conv3d_mma_bench_fmt!(
        bench_mxint2,
        mt_mxint2_conv3d_mma::kernel_ir_for,
        QFormat::Mxint2,
        "ffai/conv3d_mma_block/mxint2"
    );
    conv3d_mma_bench_fmt!(
        bench_mxint3,
        mt_mxint3_conv3d_mma::kernel_ir_for,
        QFormat::Mxint3,
        "ffai/conv3d_mma_block/mxint3"
    );
    conv3d_mma_bench_fmt!(
        bench_mxint4,
        mt_mxint4_conv3d_mma::kernel_ir_for,
        QFormat::Mxint4,
        "ffai/conv3d_mma_block/mxint4"
    );
    conv3d_mma_bench_fmt!(
        bench_mxint5,
        mt_mxint5_conv3d_mma::kernel_ir_for,
        QFormat::Mxint5,
        "ffai/conv3d_mma_block/mxint5"
    );
    conv3d_mma_bench_fmt!(
        bench_mxint6,
        mt_mxint6_conv3d_mma::kernel_ir_for,
        QFormat::Mxint6,
        "ffai/conv3d_mma_block/mxint6"
    );
    conv3d_mma_bench_fmt!(
        bench_mxint8,
        mt_mxint8_conv3d_mma::kernel_ir_for,
        QFormat::Mxint8,
        "ffai/conv3d_mma_block/mxint8"
    );
    // ── FP16-scale twins (fp8_e4m3_f16 reuses the nvfp8_f16 kernel) ──
    conv3d_mma_bench_fmt!(
        bench_nvfp8_f16,
        mt_nvfp8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Nvfp8F16,
        "ffai/conv3d_mma_block/nvfp8_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E4m3F16,
        "ffai/conv3d_mma_block/fp8_e4m3_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_fp4_f16,
        mt_fp4_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp4F16,
        "ffai/conv3d_mma_block/fp4_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E5m2F16,
        "ffai/conv3d_mma_block/fp8_e5m2_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_int2_f16,
        mt_int2_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int2F16,
        "ffai/conv3d_mma_block/int2_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_int3_f16,
        mt_int3_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int3F16,
        "ffai/conv3d_mma_block/int3_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_int4_f16,
        mt_int4_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int4F16,
        "ffai/conv3d_mma_block/int4_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_int5_f16,
        mt_int5_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int5F16,
        "ffai/conv3d_mma_block/int5_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_int6_f16,
        mt_int6_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int6F16,
        "ffai/conv3d_mma_block/int6_f16"
    );
    conv3d_mma_bench_fmt!(
        bench_int8_f16,
        mt_int8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int8F16,
        "ffai/conv3d_mma_block/int8_f16"
    );
}

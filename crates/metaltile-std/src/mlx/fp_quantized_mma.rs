//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_fp4_qmm_mma` / `mt_fp8_e4m3_qmm_mma` — fp4/fp8 simdgroup-matrix MMA.
//!
//! Simdgroup-matrix MMA prefill path for fp4 (E2M1) and fp8 (E4M3) quantized
//! dense GEMM — the non-NAX counterpart of `mt_fp_qmm_nax`. Falls back from
//! `mt_fp_qmm_nax` on pre-M4 hardware (no Apple tensor cores).
//!
//! ## fp4 E2M1 codebook
//!
//! Eight fp4 codes pack into one u32 (4 bits each). The E2M1 format
//! `[sign:1][exp:2][mantissa:1]` encodes 8 magnitudes:
//!   `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` — the nvfp4 / MLX `fp4.h` levels.
//! Computed via the `two_m_int` trick (integer arithmetic to avoid f32 LUT):
//!   - `code3 = code & 7` (3-bit magnitude)
//!   - subnormal (exp=0): `two_m_int = mantissa ∈ {0, 1}`
//!   - normal (exp≥1): `two_m_int = (mantissa + 2) * 2^(exp-1) ∈ {2,3,4,6,8,12}`
//!   - sign bit: `1 - 2*(code >> 3)`
//!   - dequant: `scale * sign * two_m_int / 2.0`, **no bias** (fp4 is scale-only).
//!
//! ## fp8 E4M3 dequant
//!
//! Eight fp8 codes pack into two u32s (8 bits each, 4 per u32). E4M3:
//! `[sign:1][exp:4][mantissa:3]`. Dequant follows the `mt_fp8_e4m3_quant_dequant`
//! math from `fp_quantized.rs`: find the binade via `floor/log2`, clamp exponent
//! to `[-6, 8]`, snap mantissa to the fp8 grid, rescale. Here we use the inverse
//! path — given a packed 8-bit code, reconstruct the fp32 value:
//!   `e = (code7 >> 3) - 7` (biased exponent, bias=7), `m = code7 & 7`
//!   normal: `val = 2^e * (1 + m/8)`, subnormal (e_raw=0): `val = 2^(-6) * m/8`
//!   sign: `1 - 2*(code >> 7)`.
//! Scale per group (group_size=32 for fp8, matching `mt_fp8_e4m3_quant_dequant`).
//!
//! ## Geometry (both kernels)
//!
//! Identical to `mt_qmm_mma`:
//!   - tpg = 128 (4 SG × 32 lanes, WM=WN=2)
//!   - BM = BN = BK = 32, output tile 32×32
//!   - Grid: `[N/32, M/32, 1]`
//!   - TG memory: Xs[32×36 T] + Ws[32×36 T]
//!   - KernelMode::Reduction

use metaltile::kernel;

// ─── mt_fp4_qmm_mma — fp4 E2M1 simdgroup-matrix MMA ─────────────────────────
//
// Dense GEMM `Out = X · dequant(W)` with fp4 (E2M1) W packed as 8 codes/u32.
// GROUP_SIZE = 32 (one scale per BK=32 block per N-row), scale-only (no bias).
// W layout: [N, K/8] uint32 (8 fp4 codes per word).
// scales layout: [N, K/group_size] T.

/// fp4 (E2M1) quantized matmul via simdgroup-matrix MMA (pre-M4 fallback).
///
/// `w [n, k/8]` fp4 E2M1 packed (8 codes/u32, MSB = sign),
/// `scales [n, k/group_size]` T (scale-only, group_size=32),
/// `x [m, k]` T, `out [m, n]` T.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
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
    // W coop-dequant: each lane handles one fp4 u32 word (8 codes).
    // lane_in_tg / 4 = w_row (0..32), lane_in_tg & 3 = word_in_row (0..4).
    // 32 N-rows × 4 words = 128 lanes = full TG.
    let w_row = lane_in_tg / 4u32;
    let word_in_row = lane_in_tg & 3u32;
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    // fp4: 8 codes/u32 → packs_per_row = k/8.
    let packs_per_row = k / 8u32;
    let sb_base = (w_n_base + w_row) * gs_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    // group_size = k / gs_per_row (= 32 for the default fp4 layout).
    let group_size = k / gs_per_row;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load ── (identical to mt_qmm_mma)
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        threadgroup_store("xs", x_ws_base, load(x[x_row_dev_base]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 1u32, load(x[x_row_dev_base + 1u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 2u32, load(x[x_row_dev_base + 2u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 3u32, load(x[x_row_dev_base + 3u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 4u32, load(x[x_row_dev_base + 4u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 5u32, load(x[x_row_dev_base + 5u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 6u32, load(x[x_row_dev_base + 6u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 7u32, load(x[x_row_dev_base + 7u32]).cast::<T>());
        // ── 2. Coop W fp4 dequant ──
        // Each lane loads one u32 pack (8 fp4 codes) and dequantizes into Ws.
        let pack_k_off = kb / 8u32 + word_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + word_in_row * 8u32;
        let g = k_off / group_size;
        let s = load(scales[sb_base + g]).cast::<f32>();
        let ws_base = w_row * ws_ld + word_in_row * 8u32;
        // Dequant 8 fp4 codes using the E2M1 two_m_int trick.
        // code3 = 3-bit magnitude (bits 0-2 of each nibble), sign = bit 3.
        for _ci in range(0u32, 8u32, 1u32) {
            let nibble = (pack >> (_ci * 4u32)) & 15u32;
            let sign = 1.0f32 - 2.0f32 * ((nibble >> 3u32) & 1u32).cast::<f32>();
            let code3 = nibble & 7u32;
            let exp = code3 >> 1u32;
            let mantissa = code3 & 1u32;
            // two_m_int: integer value of 2×magnitude.
            // subnormal (exp=0): two_m_int = mantissa ∈ {0, 1}
            // normal (exp≥1): two_m_int = (mantissa + 2) * 2^(exp-1)
            let is_normal = select(exp > 0u32, 1u32, 0u32);
            let two_m_int_sub = mantissa;
            let two_m_int_norm = (mantissa + 2u32) << (exp - 1u32);
            let two_m_int = select(is_normal == 1u32, two_m_int_norm, two_m_int_sub);
            let val = s * sign * two_m_int.cast::<f32>() * 0.5f32;
            threadgroup_store("ws", ws_base + _ci, val.cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — identical to mt_qmm_mma ──
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 1
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 2
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 3
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    // ── 4. Write C frags ──
    let out_m_base = m_tile * 32u32 + sm * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

// ─── mt_fp8_e4m3_qmm_mma — fp8 E4M3 simdgroup-matrix MMA ────────────────────
//
// Dense GEMM with fp8 E4M3 weights. W packed as 4 codes/u32 (8 bits each).
// GROUP_SIZE = 32 (one scale per BK=32 block per N-row), scale-only (no bias).
//
// E4M3 decode: `[sign:1][exp:4][mantissa:3]`, exponent bias = 7.
//   - e_raw = (code7 >> 3) where code7 = lower 7 bits
//   - m = code7 & 7
//   - normal (e_raw > 0): val = 2^(e_raw-7) * (1 + m/8)
//   - subnormal (e_raw = 0): val = 2^(-6) * m/8
//   - sign: 1 - 2*(code >> 7)
// Then rescale by group scale: dequant = scale * sign * magnitude.
//
// W layout: [N, K/4] uint32 (4 fp8 codes per word).
// scales layout: [N, K/group_size] T.

/// fp8 E4M3 quantized matmul via simdgroup-matrix MMA (pre-M4 fallback).
///
/// `w [n, k/4]` fp8 E4M3 packed (4 codes/u32),
/// `scales [n, k/group_size]` T (scale-only, group_size=32),
/// `x [m, k]` T, `out [m, n]` T.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e4m3_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
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
    // W coop-dequant: fp8 has 4 codes/u32 → BK=32 spans 8 words per N-row.
    // 32 N-rows × 8 words = 256 — too many for 128 lanes in one step.
    // Use 2 steps: step 0 covers words 0-3, step 1 covers words 4-7.
    // Within each step: lane_in_tg / 8 = w_row (0..15 per step),
    // lane_in_tg & 7 = word_in_row_step (0..7 within the 4-word span).
    // We split differently: w_row = lane_in_tg / 4, word_in_row = lane_in_tg & 3.
    // This gives 32 N-rows × 4 words = 128 lanes covering words 0..3.
    // Then a second pass covers words 4..7.
    let w_row = lane_in_tg / 4u32;
    let word_in_row = lane_in_tg & 3u32;
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    // fp8: 4 codes/u32 → packs_per_row = k/4.
    let packs_per_row = k / 4u32;
    let sb_base = (w_n_base + w_row) * gs_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let group_size = k / gs_per_row;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        threadgroup_store("xs", x_ws_base, load(x[x_row_dev_base]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 1u32, load(x[x_row_dev_base + 1u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 2u32, load(x[x_row_dev_base + 2u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 3u32, load(x[x_row_dev_base + 3u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 4u32, load(x[x_row_dev_base + 4u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 5u32, load(x[x_row_dev_base + 5u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 6u32, load(x[x_row_dev_base + 6u32]).cast::<T>());
        threadgroup_store("xs", x_ws_base + 7u32, load(x[x_row_dev_base + 7u32]).cast::<T>());
        // ── 2. Coop W fp8 E4M3 dequant — 2 passes (words 0..3, words 4..7) ──
        let k_off = kb + word_in_row * 4u32;
        let g = k_off / group_size;
        let s = load(scales[sb_base + g]).cast::<f32>();
        // Pass A: words 0..3 (word_in_row = 0..3)
        let pack_a = load(w[w_pack_row_base + kb / 4u32 + word_in_row]);
        let ws_base_a = w_row * ws_ld + word_in_row * 4u32;
        for _ci in range(0u32, 4u32, 1u32) {
            let code = (pack_a >> (_ci * 8u32)) & 255u32;
            let sign = 1.0f32 - 2.0f32 * (code >> 7u32).cast::<f32>();
            let code7 = code & 127u32;
            let e_raw = code7 >> 3u32;
            let m = code7 & 7u32;
            // normal (e_raw > 0): 2^(e_raw-7) * (1 + m/8)
            // subnormal (e_raw = 0): 2^(-6) * m/8
            let is_normal = select(e_raw > 0u32, 1u32, 0u32);
            let exp_f = e_raw.cast::<f32>() - 7.0f32;
            let norm_mag = exp2(exp_f) * (1.0f32 + m.cast::<f32>() * 0.125f32);
            let sub_mag = exp2(-6.0f32) * m.cast::<f32>() * 0.125f32;
            let mag = select(is_normal == 1u32, norm_mag, sub_mag);
            let val = s * sign * mag;
            threadgroup_store("ws", ws_base_a + _ci, val.cast::<T>());
        }
        // Pass B: words 4..7 (same lane, offset by 4 in Ws and W packs).
        let k_off_b = kb + (word_in_row + 4u32) * 4u32;
        let g_b = k_off_b / group_size;
        let s_b = load(scales[sb_base + g_b]).cast::<f32>();
        let pack_b = load(w[w_pack_row_base + kb / 4u32 + word_in_row + 4u32]);
        let ws_base_b = w_row * ws_ld + (word_in_row + 4u32) * 4u32;
        for _ci in range(0u32, 4u32, 1u32) {
            let code = (pack_b >> (_ci * 8u32)) & 255u32;
            let sign = 1.0f32 - 2.0f32 * (code >> 7u32).cast::<f32>();
            let code7 = code & 127u32;
            let e_raw = code7 >> 3u32;
            let m = code7 & 7u32;
            let is_normal = select(e_raw > 0u32, 1u32, 0u32);
            let exp_f = e_raw.cast::<f32>() - 7.0f32;
            let norm_mag = exp2(exp_f) * (1.0f32 + m.cast::<f32>() * 0.125f32);
            let sub_mag = exp2(-6.0f32) * m.cast::<f32>() * 0.125f32;
            let mag = select(is_normal == 1u32, norm_mag, sub_mag);
            let val = s_b * sign * mag;
            threadgroup_store("ws", ws_base_b + _ci, val.cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop ── (identical to mt_qmm_mma)
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 1
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 2
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 3
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    // ── 4. Write C frags ──
    let out_m_base = m_tile * 32u32 + sm * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let xf = *x as f64;
            let yf = *y as f64;
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
        (dot / denom) as f32
    }

    fn fp4_decode(nibble: u32) -> f32 {
        let sign = 1.0f32 - 2.0f32 * ((nibble >> 3) & 1) as f32;
        let code3 = nibble & 7;
        let exp = code3 >> 1;
        let mantissa = code3 & 1;
        let two_m_int = if exp > 0 { (mantissa + 2) << (exp - 1) } else { mantissa };
        sign * two_m_int as f32 * 0.5
    }

    fn pack_fp4_row(codes: &[u32]) -> Vec<u32> {
        assert!(codes.len().is_multiple_of(8));
        codes
            .chunks_exact(8)
            .map(|ch| ch.iter().enumerate().fold(0u32, |acc, (i, &c)| acc | ((c & 0xF) << (i * 4))))
            .collect()
    }

    fn build_fp4_inputs(
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>) {
        let codes_flat: Vec<u32> = (0..n * k)
            .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_shr(12) & 0xF)
            .collect();
        let packed: Vec<u32> = codes_flat.chunks_exact(k).flat_map(pack_fp4_row).collect();
        let scales: Vec<f32> =
            (0..n * gs_per_row).map(|i| 0.5 + 0.1 * (i as f32 * 0.07).sin().abs()).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i as f32 * 0.017).sin()).collect();
        (packed, scales, x, codes_flat)
    }

    fn cpu_fp4_qmm_reference(
        codes_flat: &[u32],
        scales: &[f32],
        x: &[f32],
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> Vec<f32> {
        let group_size = k / gs_per_row;
        let mut out = vec![0.0f32; m * n];
        for m_row in 0..m {
            for n_col in 0..n {
                let mut acc = 0.0f32;
                for d in 0..k {
                    let code = codes_flat[n_col * k + d];
                    let g = d / group_size;
                    let s = scales[n_col * gs_per_row + g];
                    acc += s * fp4_decode(code) * x[m_row * k + d];
                }
                out[m_row * n + n_col] = acc;
            }
        }
        out
    }

    fn fp8_e4m3_decode(code: u32) -> f32 {
        let sign = 1.0f32 - 2.0f32 * (code >> 7) as f32;
        let code7 = code & 0x7F;
        let e_raw = code7 >> 3;
        let m = code7 & 7;
        let mag = if e_raw > 0 {
            let exp_f = e_raw as f32 - 7.0;
            exp_f.exp2() * (1.0 + m as f32 * 0.125)
        } else {
            (-6.0f32).exp2() * m as f32 * 0.125
        };
        sign * mag
    }

    fn pack_fp8_row(codes: &[u32]) -> Vec<u32> {
        assert!(codes.len().is_multiple_of(4));
        codes
            .chunks_exact(4)
            .map(|ch| {
                ch.iter().enumerate().fold(0u32, |acc, (i, &c)| acc | ((c & 0xFF) << (i * 8)))
            })
            .collect()
    }

    fn build_fp8_e4m3_inputs(
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>) {
        let codes_flat: Vec<u32> = (0..n * k)
            .map(|i| {
                let c = (i as u32).wrapping_mul(2654435761).wrapping_shr(11) & 0x7F;
                let e = ((c >> 3) & 0xF).max(1);
                let m = c & 7;
                (e << 3) | m
            })
            .collect();
        let packed: Vec<u32> = codes_flat.chunks_exact(k).flat_map(pack_fp8_row).collect();
        let scales: Vec<f32> =
            (0..n * gs_per_row).map(|i| 0.1 + 0.05 * (i as f32 * 0.11).sin().abs()).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.05 * (i as f32 * 0.019).cos()).collect();
        (packed, scales, x, codes_flat)
    }

    fn cpu_fp8_e4m3_qmm_reference(
        codes_flat: &[u32],
        scales: &[f32],
        x: &[f32],
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> Vec<f32> {
        let group_size = k / gs_per_row;
        let mut out = vec![0.0f32; m * n];
        for m_row in 0..m {
            for n_col in 0..n {
                let mut acc = 0.0f32;
                for d in 0..k {
                    let code = codes_flat[n_col * k + d];
                    let g = d / group_size;
                    let s = scales[n_col * gs_per_row + g];
                    acc += s * fp8_e4m3_decode(code) * x[m_row * k + d];
                }
                out[m_row * n + n_col] = acc;
            }
        }
        out
    }

    fn pack_f32_as_dtype(vals: &[f32], dtype: DType) -> Vec<u8> {
        match dtype {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
        }
    }

    fn round_dtype(v: f32, dtype: DType) -> f32 {
        match dtype {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn pack_u32_as_bytes(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    fn unpack_dtype_as_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
        match dtype {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }

    // ── fp4 MMA tests ────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/fp4_qmm_mma/f32_small", dtypes = [f32], tol = 0.001)]
    fn test_fp4_qmm_mma_f32_small(dt: DType) -> TestSetup {
        let (m, n, k, gs_per_row) = (32usize, 32usize, 32usize, 1usize);
        let (packed, scales, x, codes_flat) = build_fp4_inputs(m, n, k, gs_per_row);
        let expected = cpu_fp4_qmm_reference(&codes_flat, &scales, &x, m, n, k, gs_per_row);
        let kernel = mt_fp4_qmm_mma::kernel_ir_for(dt);
        let bytes_per_elem = 4usize;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("w", pack_u32_as_bytes(&packed), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32_as_dtype(&scales, dt), dt))
            .input(TestBuffer::from_vec("x", pack_f32_as_dtype(&x, dt), dt))
            .input(TestBuffer::from_vec("out", vec![0u8; m * n * bytes_per_elem], dt))
            .input(TestBuffer::from_vec("k", (k as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec(
                "gs_per_row",
                (gs_per_row as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .expect(TestBuffer::from_vec("out", pack_f32_as_dtype(&expected, dt), dt))
            .grid_3d(n as u32 / 32, m as u32 / 32, 1, [128, 1, 1])
    }

    #[test_kernel(name = "mlx/fp8_e4m3_qmm_mma/f32_small", dtypes = [f32], tol = 0.001)]
    fn test_fp8_e4m3_qmm_mma_f32_small(dt: DType) -> TestSetup {
        let (m, n, k, gs_per_row) = (32usize, 32usize, 32usize, 1usize);
        let (packed, scales, x, codes_flat) = build_fp8_e4m3_inputs(m, n, k, gs_per_row);
        let expected = cpu_fp8_e4m3_qmm_reference(&codes_flat, &scales, &x, m, n, k, gs_per_row);
        let kernel = mt_fp8_e4m3_qmm_mma::kernel_ir_for(dt);
        let bytes_per_elem = 4usize;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("w", pack_u32_as_bytes(&packed), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32_as_dtype(&scales, dt), dt))
            .input(TestBuffer::from_vec("x", pack_f32_as_dtype(&x, dt), dt))
            .input(TestBuffer::from_vec("out", vec![0u8; m * n * bytes_per_elem], dt))
            .input(TestBuffer::from_vec("k", (k as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec(
                "gs_per_row",
                (gs_per_row as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .expect(TestBuffer::from_vec("out", pack_f32_as_dtype(&expected, dt), dt))
            .grid_3d(n as u32 / 32, m as u32 / 32, 1, [128, 1, 1])
    }
}

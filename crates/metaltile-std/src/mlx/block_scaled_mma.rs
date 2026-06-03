//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled simdgroup-matrix (MMA) dequantizing GEMM — the M ≥ 32
//! ALU-throughput path for the spec-conformant formats. This is a direct
//! adaptation of `mlx/quantized.rs::mt_qmm_mma` (the int4 affine MMA): the
//! **dispatch geometry, threadgroup-memory layout, 8×8 frag mapping, and MMA
//! inner loop are copied verbatim** — only the per-pack weight *dequant*
//! staging changes (E2M1 codebook × E8M0 pow-2 scale instead of int4 affine).
//! Reusing the proven geometry keeps it off the reduction freeze-hazard surface.
//!
//! ## DISPATCH INVARIANTS (identical to `mt_qmm_mma`)
//!
//! - **Mode: Reduction**, `grid = [n/32, m/32, 1]`, `tpg = [128, 1, 1]`
//!   (4 simdgroups × 32 lanes, WM=WN=2). `m`, `n`, `k` all multiples of 32.
//! - BM = BN = BK = 32, output tile 32×32. TG memory `xs`/`ws` are `32×36`
//!   (skew 4 to break bank conflicts; 36 is correct for every dtype).
//! - weight `[n, k/8]` u32 (8 E2M1 nibbles/word); scales `[n, k/block_size]` u8
//!   (E8M0); `block_size` a multiple of 8. x `[m, k]`, out `[m, n]`, row-major.

use metaltile::kernel;

/// mxfp4 simdgroup-matrix dequantizing GEMM (E2M1 weights, E8M0 pow-2 scale).
#[kernel]
pub fn mt_mxfp4_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let packs_per_row = k / 8u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — each lane loads 1 pack → 8 fp T (mxfp4) ──
        let pack_k_off = kb / 8u32 + pack_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + pack_in_row * 8u32;
        let g = k_off / block_size; // E8M0 block index (one per BK for bs=32)
        let sbits = load(scales[sb_base + g]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        let ws_base = w_row * ws_ld + pack_in_row * 8u32;
        for i in range(0u32, 8u32, 1u32) {
            let nib = (pack >> (i * 4u32)) & 0xFu32;
            let val = e2m1_decode(nib);
            threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// nvfp4 simdgroup-matrix dequantizing GEMM (E2M1 weights, E4M3 micro-scale x global).
#[kernel]
pub fn mt_nvfp4_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let packs_per_row = k / 8u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — each lane loads 1 pack → 8 fp T (nvfp4) ──
        let pack_k_off = kb / 8u32 + pack_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + pack_in_row * 8u32;
        let g = k_off / block_size; // E8M0 block index (one per BK for bs=32)
        let scale = e4m3_decode(load(scales[sb_base + g]).cast::<u32>()) * global;
        let ws_base = w_row * ws_ld + pack_in_row * 8u32;
        for i in range(0u32, 8u32, 1u32) {
            let nib = (pack >> (i * 4u32)) & 0xFu32;
            let val = e2m1_decode(nib);
            threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// mxfp8 (E4M3) simdgroup-matrix dequantizing GEMM (8-bit weights, block 32, E8M0 scale).
#[kernel]
pub fn mt_mxfp8_e4m3_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (e4m3) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let sbits = load(scales[sb_base + g]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = e4m3_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// mxfp8 (E5M2) simdgroup-matrix dequantizing GEMM (8-bit weights, block 32, E8M0 scale).
#[kernel]
pub fn mt_mxfp8_e5m2_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (e5m2) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let sbits = load(scales[sb_base + g]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = e5m2_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// nvfp8 simdgroup-matrix dequantizing GEMM (E4M3 weights, block 16, per-block FP32 scale).
#[kernel]
pub fn mt_nvfp8_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (e4m3) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let scale = load(scales[sb_base + g]); // per-block FP32 scale
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = e4m3_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 MMA GEMMs ──────────────
// These share the simdgroup-matrix framework but bind a raw per-group FP32
// scale (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit
// E4M3 + f32 scale), so it reuses `mt_nvfp8_qmm_mma`; only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// fp4 simdgroup-matrix dequantizing GEMM (E2M1 weights, per-group FP32 scale).
///
/// Distinct name from the original `fp_quantized_mma::mt_fp4_qmm_mma`: emitting
/// two kernels under the same MSL function name collided in the pipeline cache,
/// producing order-dependent (and f32-specific) wrong results. Verified correct
/// on f32/f16/bf16 against the `quant::format` oracle.
#[kernel]
pub fn mt_fp4_float_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let packs_per_row = k / 8u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — each lane loads 1 pack → 8 fp T (fp4) ──
        let pack_k_off = kb / 8u32 + pack_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + pack_in_row * 8u32;
        let g = k_off / block_size; // E8M0 block index (one per BK for bs=32)
        let scale = load(scales[sb_base + g]);
        let ws_base = w_row * ws_ld + pack_in_row * 8u32;
        for i in range(0u32, 8u32, 1u32) {
            let nib = (pack >> (i * 4u32)) & 0xFu32;
            let val = e2m1_decode(nib);
            threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// Legacy fp8 (E5M2) simdgroup-matrix dequantizing GEMM (8-bit weights, per-group FP32 scale).
#[kernel]
pub fn mt_fp8_e5m2_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (e5m2) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let scale = load(scales[sb_base + g]);
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = e5m2_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// Symmetric int8 simdgroup-matrix dequantizing GEMM (8-bit codes, block 64,
/// per-group FP32 scale). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (int8) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let scale = load(scales[sb_base + g]);
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = int8_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

// ── Symmetric sub-byte integer MMA GEMMs (int2/3/4/5/6 + MXINT2..6) ─────────
// The element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into a FLAT global u32 bit-stream by `quant::format::pack` (element
// with global index `idx = w_global_row · k + k_col` lives at bit `idx · BITS`).
// These kernels reuse `mt_int8_qmm_mma`'s dispatch geometry, threadgroup-memory
// layout, 8×8 frag mapping, and MMA inner loop **verbatim** — only the per-pack
// W *dequant* staging changes. The W-load mirrors the 4-bit MMA lane mapping
// (`pack_in_row = lane_in_tg & 3`, each lane stages 8 contiguous K columns
// `kb + pack_in_row·8 + i`), but instead of reading one nibble pack it decodes
// each element from the global bit-stream with a straddle-aware two-word read +
// float sign-extend (subtract 2^N when the top bit is set; `$half`/`$full` are
// 2^(N-1) / 2^N), mirroring `block_scaled_dequant`'s proven `int_dequant_*`
// macros. The block index `g = (kb + pack_in_row·8) / block_size` covers all 8
// columns (block_size ∈ {32,64} is a multiple of 8, exactly like the 4-bit
// kernel). `$half`/`$full` are passed as literals to keep the constexpr math out
// of the DSL shift operands.

/// FP32-scaled symmetric int MMA GEMM (int2/3/4/5/6): per-element bit-stream
/// code × per-group FP32 scale, fed to the simdgroup-matrix matmul. The W bit
/// offset is computed from the element's GLOBAL index (`w_global_elem · BITS`),
/// matching `quant::format::pack`'s flat LSB-first stream for any K.
macro_rules! int_qmm_mma_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<f32>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] block_size: u32,
        ) {
            let n_tile = tgid_x;
            let m_tile = tgid_y;
            let lane = simd_lane;
            let sg = simd_group_id();
            let sm = sg / 2u32;
            let sn = sg & 1u32;
            let lane_in_tg = sg * 32u32 + lane;
            // 8×8 frag lane mapping (Apple steel_gemm layout).
            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;
            threadgroup_alloc("xs", 1152, T);
            threadgroup_alloc("ws", 1152, T);
            // 4 output frags per SG, init to 0.
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
            let w_row = lane_in_tg / 4u32;
            let pack_in_row = lane_in_tg & 3u32;
            let x_m_base = m_tile * 32u32;
            let w_n_base = n_tile * 32u32;
            let n_blocks_per_row = k / block_size;
            // Per-lane scale row base (FP32, one per block). Fixed across K-blocks.
            let sb_base = (w_n_base + w_row) * n_blocks_per_row;
            // Global element index of this lane's weight row start (flat
            // bit-stream is row-major: element (row, col) at bit (row·k+col)·bits).
            let w_global_row_base = (w_n_base + w_row) * k;
            let xs_ld = 36u32;
            let ws_ld = 36u32;
            // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
            let x_m_row = lane_in_tg / 4u32;
            let x_k_quad = lane_in_tg & 3u32;
            let x_k_base = x_k_quad * 8u32;
            for kb in range(0u32, k, 32u32) {
                // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
                let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
                let x_ws_base = x_m_row * xs_ld + x_k_base;
                let xv0 = load(x[x_row_dev_base]).cast::<T>();
                let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
                let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
                let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
                let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
                let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
                let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
                let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
                threadgroup_store("xs", x_ws_base, xv0);
                threadgroup_store("xs", x_ws_base + 1u32, xv1);
                threadgroup_store("xs", x_ws_base + 2u32, xv2);
                threadgroup_store("xs", x_ws_base + 3u32, xv3);
                threadgroup_store("xs", x_ws_base + 4u32, xv4);
                threadgroup_store("xs", x_ws_base + 5u32, xv5);
                threadgroup_store("xs", x_ws_base + 6u32, xv6);
                threadgroup_store("xs", x_ws_base + 7u32, xv7);
                // ── 2. Coop W dequant — each lane decodes 8 contiguous K columns ──
                // (int bit-stream). Same lane→column mapping as the 4-bit kernel:
                // pack_in_row picks which 8-column chunk of this BK the lane owns.
                let k_off = kb + pack_in_row * 8u32;
                let g = k_off / block_size; // all 8 columns lie in one block
                let scale = load(scales[sb_base + g]);
                let ws_base = w_row * ws_ld + pack_in_row * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_global_row_base + k_off + i) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(w[word_idx]);
                    let w1 = load(w[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
                }
                threadgroup_barrier();
                // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm),
                );
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm),
                );
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                threadgroup_barrier();
            }
            // ── 4. Write 4 C frags to global out ──
            let out_m_base = m_tile * 32u32 + sm * 16u32;
            let out_n_base = n_tile * 32u32 + sn * 16u32;
            store(
                out[(out_m_base + fm) * n + out_n_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + fm) * n + out_n_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
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
    };
}
int_qmm_mma_f32!(mt_int2_qmm_mma, 2u32, 2u32, 4.0f32);
int_qmm_mma_f32!(mt_int3_qmm_mma, 3u32, 4u32, 8.0f32);
int_qmm_mma_f32!(mt_int4_qmm_mma, 4u32, 8u32, 16.0f32);
int_qmm_mma_f32!(mt_int5_qmm_mma, 5u32, 16u32, 32.0f32);
int_qmm_mma_f32!(mt_int6_qmm_mma, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int MMA GEMM (MXINT2/3/4/5/6): per-element bit-stream
/// code × pow-2 (E8M0) block scale `2^(bits-127)`, fed to the simdgroup-matrix
/// matmul. Same straddle-aware global-bit-offset decode and dispatch geometry as
/// `int_qmm_mma_f32`; only the scale axis differs (one u8 exponent per block).
macro_rules! int_qmm_mma_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<u8>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] block_size: u32,
        ) {
            let n_tile = tgid_x;
            let m_tile = tgid_y;
            let lane = simd_lane;
            let sg = simd_group_id();
            let sm = sg / 2u32;
            let sn = sg & 1u32;
            let lane_in_tg = sg * 32u32 + lane;
            // 8×8 frag lane mapping (Apple steel_gemm layout).
            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;
            threadgroup_alloc("xs", 1152, T);
            threadgroup_alloc("ws", 1152, T);
            // 4 output frags per SG, init to 0.
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
            let w_row = lane_in_tg / 4u32;
            let pack_in_row = lane_in_tg & 3u32;
            let x_m_base = m_tile * 32u32;
            let w_n_base = n_tile * 32u32;
            let n_blocks_per_row = k / block_size;
            // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
            let sb_base = (w_n_base + w_row) * n_blocks_per_row;
            // Global element index of this lane's weight row start (flat
            // bit-stream is row-major: element (row, col) at bit (row·k+col)·bits).
            let w_global_row_base = (w_n_base + w_row) * k;
            let xs_ld = 36u32;
            let ws_ld = 36u32;
            // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
            let x_m_row = lane_in_tg / 4u32;
            let x_k_quad = lane_in_tg & 3u32;
            let x_k_base = x_k_quad * 8u32;
            for kb in range(0u32, k, 32u32) {
                // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
                let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
                let x_ws_base = x_m_row * xs_ld + x_k_base;
                let xv0 = load(x[x_row_dev_base]).cast::<T>();
                let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
                let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
                let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
                let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
                let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
                let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
                let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
                threadgroup_store("xs", x_ws_base, xv0);
                threadgroup_store("xs", x_ws_base + 1u32, xv1);
                threadgroup_store("xs", x_ws_base + 2u32, xv2);
                threadgroup_store("xs", x_ws_base + 3u32, xv3);
                threadgroup_store("xs", x_ws_base + 4u32, xv4);
                threadgroup_store("xs", x_ws_base + 5u32, xv5);
                threadgroup_store("xs", x_ws_base + 6u32, xv6);
                threadgroup_store("xs", x_ws_base + 7u32, xv7);
                // ── 2. Coop W dequant — each lane decodes 8 contiguous K columns ──
                // (int bit-stream). Same lane→column mapping as the 4-bit kernel:
                // pack_in_row picks which 8-column chunk of this BK the lane owns.
                let k_off = kb + pack_in_row * 8u32;
                let g = k_off / block_size; // all 8 columns lie in one block
                let sbits = load(scales[sb_base + g]).cast::<f32>();
                let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                let ws_base = w_row * ws_ld + pack_in_row * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_global_row_base + k_off + i) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(w[word_idx]);
                    let w1 = load(w[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
                }
                threadgroup_barrier();
                // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm),
                );
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm),
                );
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                threadgroup_barrier();
            }
            // ── 4. Write 4 C frags to global out ──
            let out_m_base = m_tile * 32u32 + sm * 16u32;
            let out_n_base = n_tile * 32u32 + sn * 16u32;
            store(
                out[(out_m_base + fm) * n + out_n_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + fm) * n + out_n_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
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
    };
}
int_qmm_mma_e8m0!(mt_mxint2_qmm_mma, 2u32, 2u32, 4.0f32);
int_qmm_mma_e8m0!(mt_mxint3_qmm_mma, 3u32, 4u32, 8.0f32);
int_qmm_mma_e8m0!(mt_mxint4_qmm_mma, 4u32, 8u32, 16.0f32);
int_qmm_mma_e8m0!(mt_mxint5_qmm_mma, 5u32, 16u32, 32.0f32);
int_qmm_mma_e8m0!(mt_mxint6_qmm_mma, 6u32, 32u32, 64.0f32);

/// MXINT8 simdgroup-matrix dequantizing GEMM (8-bit symmetric codes, byte
/// layout, block 32, E8M0 pow-2 block scale `2^(bits-127)`). Identical geometry
/// and W-load mapping to `mt_int8_qmm_mma` (one byte per code, 8 contiguous
/// bytes per lane); only the scale axis is E8M0 instead of a raw FP32.
#[kernel]
pub fn mt_mxint8_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (mxint8) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of sub-byte bit-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let sbits = load(scales[sb_base + g]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = int8_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

// ── FP16-scale twins (fp16 group scale) ─────────────────────────────────────
// Near-clones of the FP32-scaled kernels above for the `*_f16` formats: the only
// change is the scale axis — `scales: Tensor<f16>` instead of `Tensor<f32>`, and
// `load(scales[...]).cast::<f32>()` instead of a raw f32 load. Element decode
// (E2M1 / E4M3 / E5M2 / int bit-stream + sign-extend), weight indexing, dispatch
// geometry, threadgroup-memory layout, 8×8 frag mapping, staging, and the MMA
// inner loop are IDENTICAL to their FP32 twins. The half load + f32 cast matches
// `block_scaled_dequant`'s GPU-verified `*_f16_dequant` references.

/// nvfp8 f16-scale simdgroup-matrix dequantizing GEMM (E4M3 weights, block 16,
/// per-block FP16 scale). Clone of `mt_nvfp8_qmm_mma`, scale → f16. Also serves
/// `Fp8E4m3F16` (same 8-bit-E4M3 + scale shape), exactly as `mt_nvfp8_qmm_mma`
/// serves `Fp8E4m3` today.
#[kernel]
pub fn mt_nvfp8_f16_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (one per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (e4m3) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let scale = load(scales[sb_base + g]).cast::<f32>(); // per-block FP16 scale
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = e4m3_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// fp4 f16-scale simdgroup-matrix dequantizing GEMM (E2M1 weights, per-group
/// FP16 scale). Clone of `mt_fp4_float_qmm_mma`, scale → f16.
#[kernel]
pub fn mt_fp4_f16_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let packs_per_row = k / 8u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (one per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — each lane loads 1 pack → 8 fp T (fp4) ──
        let pack_k_off = kb / 8u32 + pack_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + pack_in_row * 8u32;
        let g = k_off / block_size; // block index (one per BK for bs=32)
        let scale = load(scales[sb_base + g]).cast::<f32>();
        let ws_base = w_row * ws_ld + pack_in_row * 8u32;
        for i in range(0u32, 8u32, 1u32) {
            let nib = (pack >> (i * 4u32)) & 0xFu32;
            let val = e2m1_decode(nib);
            threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// fp8 (E5M2) f16-scale simdgroup-matrix dequantizing GEMM (8-bit weights,
/// per-group FP16 scale). Clone of `mt_fp8_e5m2_qmm_mma`, scale → f16.
#[kernel]
pub fn mt_fp8_e5m2_f16_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (one per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (e5m2) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let scale = load(scales[sb_base + g]).cast::<f32>();
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = e5m2_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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

/// FP16-scaled symmetric int MMA GEMM (int2/3/4/5/6): per-element bit-stream
/// code × per-group FP16 scale, fed to the simdgroup-matrix matmul. Clone of the
/// `int_qmm_mma_f32!` macro, scale → f16 (`load(scales[..]).cast::<f32>()`).
macro_rules! int_qmm_mma_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            w: Tensor<u32>,
            scales: Tensor<f16>,
            x: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] k: u32,
            #[constexpr] n: u32,
            #[constexpr] block_size: u32,
        ) {
            let n_tile = tgid_x;
            let m_tile = tgid_y;
            let lane = simd_lane;
            let sg = simd_group_id();
            let sm = sg / 2u32;
            let sn = sg & 1u32;
            let lane_in_tg = sg * 32u32 + lane;
            // 8×8 frag lane mapping (Apple steel_gemm layout).
            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;
            threadgroup_alloc("xs", 1152, T);
            threadgroup_alloc("ws", 1152, T);
            // 4 output frags per SG, init to 0.
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
            let w_row = lane_in_tg / 4u32;
            let pack_in_row = lane_in_tg & 3u32;
            let x_m_base = m_tile * 32u32;
            let w_n_base = n_tile * 32u32;
            let n_blocks_per_row = k / block_size;
            // Per-lane scale row base (FP16, one per block). Fixed across K-blocks.
            let sb_base = (w_n_base + w_row) * n_blocks_per_row;
            // Global element index of this lane's weight row start (flat
            // bit-stream is row-major: element (row, col) at bit (row·k+col)·bits).
            let w_global_row_base = (w_n_base + w_row) * k;
            let xs_ld = 36u32;
            let ws_ld = 36u32;
            // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
            let x_m_row = lane_in_tg / 4u32;
            let x_k_quad = lane_in_tg & 3u32;
            let x_k_base = x_k_quad * 8u32;
            for kb in range(0u32, k, 32u32) {
                // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
                let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
                let x_ws_base = x_m_row * xs_ld + x_k_base;
                let xv0 = load(x[x_row_dev_base]).cast::<T>();
                let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
                let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
                let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
                let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
                let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
                let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
                let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
                threadgroup_store("xs", x_ws_base, xv0);
                threadgroup_store("xs", x_ws_base + 1u32, xv1);
                threadgroup_store("xs", x_ws_base + 2u32, xv2);
                threadgroup_store("xs", x_ws_base + 3u32, xv3);
                threadgroup_store("xs", x_ws_base + 4u32, xv4);
                threadgroup_store("xs", x_ws_base + 5u32, xv5);
                threadgroup_store("xs", x_ws_base + 6u32, xv6);
                threadgroup_store("xs", x_ws_base + 7u32, xv7);
                // ── 2. Coop W dequant — each lane decodes 8 contiguous K columns ──
                // (int bit-stream). Same lane→column mapping as the 4-bit kernel:
                // pack_in_row picks which 8-column chunk of this BK the lane owns.
                let k_off = kb + pack_in_row * 8u32;
                let g = k_off / block_size; // all 8 columns lie in one block
                let scale = load(scales[sb_base + g]).cast::<f32>();
                let ws_base = w_row * ws_ld + pack_in_row * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    // Global element index → bit offset in the flat LSB-first stream.
                    let bit_off = (w_global_row_base + k_off + i) * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(w[word_idx]);
                    let w1 = load(w[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let val = select(q >= $half, qf - $full, qf); // sign-extend
                    threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
                }
                threadgroup_barrier();
                // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm),
                );
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm),
                );
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
                simdgroup_elem_store(
                    b_f0,
                    0,
                    threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f0,
                    1,
                    threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    0,
                    threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm),
                );
                simdgroup_elem_store(
                    b_f1,
                    1,
                    threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm),
                );
                simdgroup_barrier_mem_none();
                simdgroup_matmul(a_f0, b_f0, c_f00);
                simdgroup_matmul(a_f0, b_f1, c_f01);
                simdgroup_matmul(a_f1, b_f1, c_f11);
                simdgroup_matmul(a_f1, b_f0, c_f10);
                simdgroup_barrier_mem_none();
                threadgroup_barrier();
            }
            // ── 4. Write 4 C frags to global out ──
            let out_m_base = m_tile * 32u32 + sm * 16u32;
            let out_n_base = n_tile * 32u32 + sn * 16u32;
            store(
                out[(out_m_base + fm) * n + out_n_base + fn0],
                simdgroup_elem_load(c_f00, 0).cast::<T>(),
            );
            store(
                out[(out_m_base + fm) * n + out_n_base + fn1],
                simdgroup_elem_load(c_f00, 1).cast::<T>(),
            );
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
    };
}
int_qmm_mma_f16!(mt_int2_f16_qmm_mma, 2u32, 2u32, 4.0f32);
int_qmm_mma_f16!(mt_int3_f16_qmm_mma, 3u32, 4u32, 8.0f32);
int_qmm_mma_f16!(mt_int4_f16_qmm_mma, 4u32, 8u32, 16.0f32);
int_qmm_mma_f16!(mt_int5_f16_qmm_mma, 5u32, 16u32, 32.0f32);
int_qmm_mma_f16!(mt_int6_f16_qmm_mma, 6u32, 32u32, 64.0f32);

/// int8 f16-scale simdgroup-matrix dequantizing GEMM (8-bit symmetric codes,
/// byte layout, per-group FP16 scale). Clone of `mt_int8_qmm_mma`, scale → f16.
#[kernel]
pub fn mt_int8_f16_qmm_mma<T>(
    w: Tensor<u8>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (one per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — 128 lanes × 8 contiguous bytes per lane (int8) ──
        // 8-bit codes are 1 byte each, so the W-load mirrors the X-load mapping
        // (w_row = lane/4, x_k_base = (lane%4)*8) instead of 4-bit pack-striding.
        let w_dev_base = (w_n_base + w_row) * k + kb + x_k_base;
        let g = (kb + x_k_base) / block_size; // all 8 bytes/lane lie in one block
        let scale = load(scales[sb_base + g]).cast::<f32>();
        let ws_base = w_row * ws_ld + x_k_base;
        for i in range(0u32, 8u32, 1u32) {
            let elem = int8_decode(load(w[w_dev_base + i]).cast::<u32>());
            threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
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
    // ── 4. Write 4 C frags to global out ──
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
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Deterministic `[n, k]` weights (mixed signs, per-block magnitude).
    fn weights(n: usize, k: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| {
                let r = (i / k) as f32;
                let c = (i % k) as f32;
                let mag = (0.4 + (r % 7.0) * 0.15) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`.
    fn qmm_oracle(wdq: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for mr in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += wdq[nn * k + kk] * x[mr * k + kk];
                }
                out[mr * n + nn] = acc;
            }
        }
        out
    }

    /// m, n multiples of 32; k a multiple of 32 (and of block_size).
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(n, k);
        let p = crate::quant::format::pack(fmt, &w, n, k);
        let wdq = crate::quant::format::dequant(fmt, &p, n, k);
        let x_f: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = qmm_oracle(&wdq, &x, m, k, n);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new formats pick up the right
        // buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("w", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (n / 32) as u32,
            (m / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // 32×32 output tile, K=64 (2 K-blocks).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp4_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp4_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_nvfp4_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp4_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp8_e4m3_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp8_e4m3_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp8_e5m2_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_nvfp8_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8, 32, 32, 64, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode in their own
    // kernels. int8 has block_size=64, so K=64 is exactly one K-block (and a
    // multiple of the 32 MMA tile) — valid for every variant.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp4_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_float_qmm_mma::kernel_ir_for(dt), QFormat::Fp4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e4m3_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e5m2_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int8_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int8_qmm_mma::kernel_ir_for(dt), QFormat::Int8, 32, 32, 64, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). K=64 is a multiple of 32 (the MMA
    // tile) and of both block sizes, and `K*bits % 32 == 0` for every width, so
    // each weight row's tight bit-stream is word-aligned. The kernel and oracle
    // share the codec, so the GPU output tracks the dequant-then-matmul reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int2_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int2_qmm_mma::kernel_ir_for(dt), QFormat::Int2, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int3_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int3_qmm_mma::kernel_ir_for(dt), QFormat::Int3, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int4_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int4_qmm_mma::kernel_ir_for(dt), QFormat::Int4, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int5_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int5_qmm_mma::kernel_ir_for(dt), QFormat::Int5, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int6_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int6_qmm_mma::kernel_ir_for(dt), QFormat::Int6, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint2_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint2_qmm_mma::kernel_ir_for(dt), QFormat::Mxint2, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint3_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint3_qmm_mma::kernel_ir_for(dt), QFormat::Mxint3, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint4_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint4_qmm_mma::kernel_ir_for(dt), QFormat::Mxint4, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint5_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint5_qmm_mma::kernel_ir_for(dt), QFormat::Mxint5, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint6_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint6_qmm_mma::kernel_ir_for(dt), QFormat::Mxint6, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint8_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint8_qmm_mma::kernel_ir_for(dt), QFormat::Mxint8, 32, 32, 64, dt)
    }

    // FP16-scale twins. Same dims as their FP32 twins (K=64: a multiple of 32 and
    // of every block size; `K*bits % 32 == 0` for every int width, so each weight
    // row's tight bit-stream stays word-aligned). `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape). Tolerances match the
    // FP32 twins — the half-precision scale rounds at pack time, but the kernel
    // and oracle share the codec so the GPU output tracks the dequant reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_nvfp8_f16_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e4m3_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp4_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp4F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e5m2_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp8_e5m2_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int2_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int2_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int2F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int3_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int3_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int3F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int4_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int4F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int5_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int5_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int5F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int6_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int6_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int6F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int8_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int8F16, 32, 32, 64, dt)
    }
}

/// Prefill GEMM (m=n=k=4096) benches — the M≥32 simdgroup-matrix throughput
/// path, where GFLOP/s + %FLOP rank the precisions (and the M5 NA story lives).
/// Random packed buffers (throughput is data-independent).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = n * (k / fmt.block_size());
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n * k, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n * k, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + (m * k + m * n) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("w", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("x", m * k, dt))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((n / 32) as u32, (m / 32) as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m as u64 * n as u64 * k as u64) // GEMM: 2·M·N·K
            .with_shape_label(format!("{} m={m} n={n} k={k}", fmt.name()))
    }

    #[bench(name = "ffai/block_scaled_qmm_mma/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxfp4_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp4_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxfp8_e4m3_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E4, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxfp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E5, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp4_float_qmm_mma::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int8_qmm_mma::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). K=4096 is a multiple of 32 and every block size.
    #[bench(name = "ffai/block_scaled_qmm_mma/int2", dtypes = [f32, f16, bf16])]
    fn bench_int2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int2_qmm_mma::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int3", dtypes = [f32, f16, bf16])]
    fn bench_int3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int3_qmm_mma::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int4", dtypes = [f32, f16, bf16])]
    fn bench_int4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int4_qmm_mma::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int5", dtypes = [f32, f16, bf16])]
    fn bench_int5_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int5_qmm_mma::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int6", dtypes = [f32, f16, bf16])]
    fn bench_int6_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int6_qmm_mma::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxint2", dtypes = [f32, f16, bf16])]
    fn bench_mxint2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint2_qmm_mma::kernel_ir_for(dt), QFormat::Mxint2, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxint3", dtypes = [f32, f16, bf16])]
    fn bench_mxint3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint3_qmm_mma::kernel_ir_for(dt), QFormat::Mxint3, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxint4", dtypes = [f32, f16, bf16])]
    fn bench_mxint4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint4_qmm_mma::kernel_ir_for(dt), QFormat::Mxint4, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxint5", dtypes = [f32, f16, bf16])]
    fn bench_mxint5_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint5_qmm_mma::kernel_ir_for(dt), QFormat::Mxint5, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxint6", dtypes = [f32, f16, bf16])]
    fn bench_mxint6_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint6_qmm_mma::kernel_ir_for(dt), QFormat::Mxint6, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/mxint8", dtypes = [f32, f16, bf16])]
    fn bench_mxint8_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint8_qmm_mma::kernel_ir_for(dt), QFormat::Mxint8, 4096, 4096, 4096, dt)
    }
    // FP16-scale twins. K=4096 is a multiple of 32 and every block size.
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + scale shape).
    #[bench(name = "ffai/block_scaled_qmm_mma/nvfp8_f16", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/fp8_e4m3_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/fp4_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp4F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/fp8_e5m2_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_f16_qmm_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int2_f16", dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int2_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int2F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int3_f16", dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int3_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int3F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int4_f16", dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int4F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int5_f16", dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int5_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int5F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int6_f16", dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int6_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int6F16, 4096, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm_mma/int8_f16", dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int8F16, 4096, 4096, 4096, dt)
    }
}

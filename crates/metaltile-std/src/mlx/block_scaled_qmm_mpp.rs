//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantized matmul via `mpp::tensor_ops::matmul2d`
//! (MetalPerformancePrimitives tensor engine) — the block-scaled counterpart of
//! `mlx/quantized_mpp{,_int8}.rs`. `Out = X · dequant(W)` for the
//! spec-conformant + legacy float-scale + symmetric-int8 formats.
//!
//! The **dispatch geometry and cooperative-matmul tail are byte-identical** to
//! the proven int4/int8 MPP kernels — TPG 128 (4 SG × 32), BM=BN=BK=32, grid
//! `[n/32, m/32, 1]`, the 2×2 warp grid, `Xs`/`Ws`/`OutScratch` threadgroup
//! tiles, and the `coop_tile_*` ops. Only the **W-dequant staging** differs:
//! `element_decode(code) · block_scale` (no bias) instead of the affine
//! `scale·q + bias`. W is dequantized to `coop_stage(T)` as it lands in `Ws`,
//! so the tensor engine sees the same fp16/fp32 tile in every format.
//!
//! Weight layout (per N-row): 4-bit `w [n, k/8] u32` (8 E2M1 nibbles/word),
//! 8-bit `w [n, k] u8` (one E4M3/E5M2/int8 code per byte). Scales
//! `[n, k/block_size]` are u8 (E8M0/E4M3) or f32 (nvfp8 / legacy fp / int8).
//! `block_size` divides the per-lane 8-K-element stripe (8 ≤ block_size, and
//! the stripe is 8-aligned), so one scale load per lane per K-block is exact.
//! `KernelMode::Reduction`. fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 +
//! f32-scale shape). Codegen-only; correctness pinned by the `#[test_kernel]`s.

use metaltile::kernel;

// ── 4-bit (E2M1) MPP kernels — model: mt_qmm_mma_mpp (int4) ────────────────

/// mxfp4 MPP matmul — E2M1 weights, E8M0 pow-2 block scale.
#[kernel]
pub fn mt_mxfp4_qmm_mma_mpp<T>(
    w: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let packs_per_row = k / 8u32;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_pack_row_base = wn_plus_wr * packs_per_row;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let packed = load(w[w_pack_row_base + kb / 8u32 + x_k_quad]);
        let k_off = kb + x_k_quad * 8u32;
        let scale = exp2(load(scales[sb_base + k_off / block_size]).cast::<f32>() - 127.0f32);
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = (packed >> (_ni * 4u32)) & 15u32;
            threadgroup_store("Ws", x_ws_base + _ni, e2m1_decode(nib) * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// nvfp4 MPP matmul — E2M1 weights, E4M3 micro-scale × global FP32.
#[kernel]
pub fn mt_nvfp4_qmm_mma_mpp<T>(
    w: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let packs_per_row = k / 8u32;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_pack_row_base = wn_plus_wr * packs_per_row;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let packed = load(w[w_pack_row_base + kb / 8u32 + x_k_quad]);
        let k_off = kb + x_k_quad * 8u32;
        let scale = e4m3_decode(load(scales[sb_base + k_off / block_size]).cast::<u32>()) * global;
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = (packed >> (_ni * 4u32)) & 15u32;
            threadgroup_store("Ws", x_ws_base + _ni, e2m1_decode(nib) * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// Legacy fp4 MPP matmul — E2M1 weights, per-group FP32 scale.
#[kernel]
pub fn mt_fp4_qmm_mma_mpp<T>(
    w: Tensor<u32>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let packs_per_row = k / 8u32;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_pack_row_base = wn_plus_wr * packs_per_row;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let packed = load(w[w_pack_row_base + kb / 8u32 + x_k_quad]);
        let k_off = kb + x_k_quad * 8u32;
        let scale = load(scales[sb_base + k_off / block_size]);
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = (packed >> (_ni * 4u32)) & 15u32;
            threadgroup_store("Ws", x_ws_base + _ni, e2m1_decode(nib) * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

// ── 8-bit (E4M3 / E5M2 / int8) MPP kernels — u8 byte-strided weight ────────

/// mxfp8 (E4M3) MPP matmul — 8-bit weights, E8M0 pow-2 block scale.
#[kernel]
pub fn mt_mxfp8_e4m3_qmm_mma_mpp<T>(
    w: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_row_base = wn_plus_wr * k;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let k_off = kb + x_k_base;
        let scale = exp2(load(scales[sb_base + k_off / block_size]).cast::<f32>() - 127.0f32);
        for _i in range(0u32, 8u32, 1u32) {
            let elem = e4m3_decode(load(w[w_row_base + k_off + _i]).cast::<u32>());
            threadgroup_store("Ws", x_ws_base + _i, elem * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// mxfp8 (E5M2) MPP matmul — 8-bit weights, E8M0 pow-2 block scale.
#[kernel]
pub fn mt_mxfp8_e5m2_qmm_mma_mpp<T>(
    w: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_row_base = wn_plus_wr * k;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let k_off = kb + x_k_base;
        let scale = exp2(load(scales[sb_base + k_off / block_size]).cast::<f32>() - 127.0f32);
        for _i in range(0u32, 8u32, 1u32) {
            let elem = e5m2_decode(load(w[w_row_base + k_off + _i]).cast::<u32>());
            threadgroup_store("Ws", x_ws_base + _i, elem * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// Legacy fp8 (E5M2) MPP matmul — 8-bit weights, per-group FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_qmm_mma_mpp<T>(
    w: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_row_base = wn_plus_wr * k;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let k_off = kb + x_k_base;
        let scale = load(scales[sb_base + k_off / block_size]);
        for _i in range(0u32, 8u32, 1u32) {
            let elem = e5m2_decode(load(w[w_row_base + k_off + _i]).cast::<u32>());
            threadgroup_store("Ws", x_ws_base + _i, elem * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// nvfp8 MPP matmul — E4M3 weights, per-block FP32 scale.
/// Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape, only block_size).
#[kernel]
pub fn mt_nvfp8_qmm_mma_mpp<T>(
    w: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_row_base = wn_plus_wr * k;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let k_off = kb + x_k_base;
        let scale = load(scales[sb_base + k_off / block_size]);
        for _i in range(0u32, 8u32, 1u32) {
            let elem = e4m3_decode(load(w[w_row_base + k_off + _i]).cast::<u32>());
            threadgroup_store("Ws", x_ws_base + _i, elem * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

/// Symmetric int8 MPP matmul — 8-bit codes, per-group FP32 scale (no bias).
#[kernel]
pub fn mt_int8_qmm_mma_mpp<T>(
    w: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_row_base = wn_plus_wr * k;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let k_off = kb + x_k_base;
        let scale = load(scales[sb_base + k_off / block_size]);
        for _i in range(0u32, 8u32, 1u32) {
            let elem = int8_decode(load(w[w_row_base + k_off + _i]).cast::<u32>());
            threadgroup_store("Ws", x_ws_base + _i, elem * scale);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// `out[mr,nc] = Σ_k x[mr,k] · dequant(W)[nc,k]` — W block-scaled `[n,k]`.
    fn mpp_setup(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> TestSetup {
        let w: Vec<f32> = (0..n * k)
            .map(|i| {
                let r = (i / k) as f32;
                let c = (i % k) as f32;
                let mag = (0.4 + (r % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15);
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::quant::format::pack(fmt, &w, n, k);
        let wdq = crate::quant::format::dequant(fmt, &p, n, k);
        let x_f: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let mut expected = vec![0.0f32; m * n];
        for mr in 0..m {
            for nc in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x[mr * k + kk] * wdq[nc * k + kk];
                }
                expected[mr * n + nc] = acc;
            }
        }
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
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

    // m=32, n=64, k=512 (divisible by 16/32/64) — mirrors the int8 MPP test.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp4_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_mxfp4_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Mxfp4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_nvfp4_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Nvfp4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_fp4_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Fp4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_mxfp8_e4m3_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_mxfp8_e5m2_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_fp8_e5m2_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_nvfp8_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Nvfp8, 32, 64, 512, dt)
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_nvfp8_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_qmm_mma_mpp(dt: DType) -> TestSetup {
        mpp_setup(mt_int8_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Int8, 32, 64, 512, dt)
    }
}

/// MPP tensor-engine matmul benches at a 128×4096×4096 tile shape.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn mpp_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> BenchSetup {
        let (codes_len, codes_dt) =
            if fmt.element_bits() == 4 { (n * k / 8, DType::U32) } else { (n * k, DType::U8) };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let n_blocks = n * (k / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m * k * sz
            + m * n * sz;
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
            .flops(2 * m as u64 * n as u64 * k as u64)
            .with_shape_label(format!("{} m={m} n={n} k={k}", fmt.name()))
    }

    #[bench(name = "mlx/block_scaled_qmm_mpp/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4(dt: DType) -> BenchSetup {
        mpp_bench(mt_mxfp4_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Mxfp4, 128, 4096, 4096, dt)
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4(dt: DType) -> BenchSetup {
        mpp_bench(mt_nvfp4_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Nvfp4, 128, 4096, 4096, dt)
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4(dt: DType) -> BenchSetup {
        mpp_bench(mt_fp4_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Fp4, 128, 4096, 4096, dt)
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3(dt: DType) -> BenchSetup {
        mpp_bench(
            mt_mxfp8_e4m3_qmm_mma_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            128,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2(dt: DType) -> BenchSetup {
        mpp_bench(
            mt_mxfp8_e5m2_qmm_mma_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            128,
            4096,
            4096,
            dt,
        )
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2(dt: DType) -> BenchSetup {
        mpp_bench(mt_fp8_e5m2_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Fp8E5m2, 128, 4096, 4096, dt)
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8(dt: DType) -> BenchSetup {
        mpp_bench(mt_nvfp8_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Nvfp8, 128, 4096, 4096, dt)
    }
    #[bench(name = "mlx/block_scaled_qmm_mpp/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        mpp_bench(mt_int8_qmm_mma_mpp::kernel_ir_for(dt), QFormat::Int8, 128, 4096, 4096, dt)
    }
}

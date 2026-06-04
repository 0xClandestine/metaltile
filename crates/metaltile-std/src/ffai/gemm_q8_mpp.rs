//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Dense Q8_0 GEMM via cooperative-tensor MMA — the MMA replacement for the
//! SCALAR `ffai_gemm_q8` (which does `acc += w*x` on the ALUs, ~12× slower
//! than a simdgroup-MMA Q8 matmul). Used for the attention Q/KV/q_a/q_b
//! projections and the O-LoRA-B / shared-expert dense Q8 matmuls in prefill.
//!
//! Same 64×64×32 coop_tile geometry as `moe_bgemm_iq2xxs_bm64` (4 simdgroups,
//! 2×2 warp grid, 128 threads/tg) but with the Q8_0 dequant and NO expert
//! routing — every output row is the same dense weight, so the bm64
//! sub-run/expert-boundary machinery collapses to a straight K-loop.
//!
//! Weight `[out_dim, k_in]` Q8_0 (qs int8 packed 4/u32 + per-32-block `d`).
//! x `[n_rows, k_in]`; out `[n_rows, out_dim]`. out[r,o]=Σ_k W[o,k]·x[r,k].
//! Name contains `_mpp_` so FFAI's PSOCache LIVE-COMPILES it (the offline
//! metallib lowers cooperative-tensor MMA incorrectly — see PSOCache.isMppKernel).
//!
//! grid (threadgroups) = [ceil(out_dim/64), ceil(n_rows/64), 1], tg [128,1,1].

use metaltile::kernel;

#[kernel]
pub fn ffai_gemm_q8_mpp<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] n_rows: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] k_in: u32,
) {
    let n_tile_base = tgid_x * 64u32; // output-feature tile (N dim)
    let m_tile_base = tgid_y * 64u32; // token tile (M dim)
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    threadgroup_alloc("Xs", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("Ws", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("OutScratch", 4096, f32); // 4 SG × 32 × 32
    coop_tile_setup(
        "gemm",
        32,
        32,
        32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    for kb in range(0u32, k_in, 32u32) {
        // Stage X[m_tile_base..+64, kb..kb+32] → Xs. 128 lanes × 16.
        let gr_x = m_tile_base + x_m_row;
        let in_run_x = gr_x < n_rows;
        let safe_gr_x = select(in_run_x, gr_x, 0u32);
        let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
        let x_ws_base = x_m_row * 32u32 + x_k_base;
        for _i in range(0u32, 16u32, 1u32) {
            let xv = load(x[x_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
        }
        // Dequant W[n_tile_base..+64, kb..kb+32] → Ws via Q8_0. 128 lanes × 16.
        for _i in range(0u32, 16u32, 1u32) {
            let flat = lane_in_tg * 16u32 + _i;
            let w_row = flat / 32u32; // 0..63 (output feature within tile)
            let k_local = flat & 31u32; // 0..31 (BK)
            let global_col = n_tile_base + w_row;
            let k = kb + k_local;
            let vidx = global_col * k_in + k;
            let block = vidx / 32u32;
            let lane = vidx & 31u32;
            let word = load(qs[block * 8u32 + lane / 4u32]);
            let by = (word >> ((lane & 3u32) * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            let w = (load(d_f32[block]) * qf).cast::<T>().cast::<f32>();
            threadgroup_store("Ws", w_row * 32u32 + k_local, w);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
    threadgroup_barrier();
    for _e in range(0u32, 32u32, 1u32) {
        let flat = lane_in_tg * 32u32 + _e;
        let mr = flat / 64u32;
        let nc = flat & 63u32;
        let gr = m_tile_base + mr;
        let gc = n_tile_base + nc;
        let in_run = (gr < n_rows) & (gc < out_dim);
        if in_run {
            let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
            let v = threadgroup_load(
                "OutScratch",
                src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
            );
            store(out[gr * out_dim + gc], v.cast::<T>());
        }
    }
}

/// GROUPED dense Q8 GEMM via cooperative-tensor MMA — the MMA version of
/// `ffai_grouped_gemm_q8` (which is scalar). For prefill O-LoRA-A: output
/// column o belongs to group g = o/rows_per_group, and group g reads input
/// columns [g*k_in, (g+1)*k_in) of an (n_groups*k_in)-wide activation row.
/// A 64-wide output tile is uniform-group (rows_per_group % 64 == 0), so the
/// group offset is computed once per threadgroup. Same 64×64×32 coop_tile
/// MMA as ffai_gemm_q8_mpp. Live-compiled (name has _mpp_).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_grouped_gemm_q8_mpp<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] n_rows: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] k_in: u32,
    #[constexpr] n_groups: u32,
    #[constexpr] rows_per_group: u32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    // Group is uniform per threadgroup (64-wide feature tile within one group).
    let g = n_tile_base / rows_per_group;
    let row_in_stride = n_groups * k_in;
    let in_col_off = g * k_in;
    threadgroup_alloc("Xs", 2048, coop_stage(T));
    threadgroup_alloc("Ws", 2048, coop_stage(T));
    threadgroup_alloc("OutScratch", 4096, f32);
    coop_tile_setup(
        "gemm",
        32,
        32,
        32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    for kb in range(0u32, k_in, 32u32) {
        let gr_x = m_tile_base + x_m_row;
        let in_run_x = gr_x < n_rows;
        let safe_gr_x = select(in_run_x, gr_x, 0u32);
        // GROUPED input read: row is n_groups*k_in wide; this group's slice
        // starts at g*k_in.
        let x_dev_base = safe_gr_x * row_in_stride + in_col_off + kb + x_k_base;
        let x_ws_base = x_m_row * 32u32 + x_k_base;
        for _i in range(0u32, 16u32, 1u32) {
            let xv = load(x[x_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
        }
        for _i in range(0u32, 16u32, 1u32) {
            let flat = lane_in_tg * 16u32 + _i;
            let w_row = flat / 32u32;
            let k_local = flat & 31u32;
            let global_col = n_tile_base + w_row;
            let k = kb + k_local;
            let vidx = global_col * k_in + k;
            let block = vidx / 32u32;
            let lane = vidx & 31u32;
            let word = load(qs[block * 8u32 + lane / 4u32]);
            let by = (word >> ((lane & 3u32) * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            let w = (load(d_f32[block]) * qf).cast::<T>().cast::<f32>();
            threadgroup_store("Ws", w_row * 32u32 + k_local, w);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
    threadgroup_barrier();
    for _e in range(0u32, 32u32, 1u32) {
        let flat = lane_in_tg * 32u32 + _e;
        let mr = flat / 64u32;
        let nc = flat & 63u32;
        let gr = m_tile_base + mr;
        let gc = n_tile_base + nc;
        let in_run = (gr < n_rows) & (gc < out_dim);
        if in_run {
            let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
            let v = threadgroup_load(
                "OutScratch",
                src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
            );
            store(out[gr * out_dim + gc], v.cast::<T>());
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_gemm_q8_mpp, ffai_grouped_gemm_q8_mpp};

    // q_a-like shape: in=4096, out=1024, 256 tokens.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemm_q8_mpp(dt: DType) -> BenchSetup {
        let n_rows = 256usize;
        let out_dim = 1024usize;
        let k_in = 4096usize;
        let n_blocks = out_dim * k_in / 32;
        BenchSetup::new(ffai_gemm_q8_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", n_rows * k_in, dt))
            .buffer(BenchBuffer::random("qs", n_blocks * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_rows * out_dim, dt).output())
            .constexpr("n_rows", n_rows as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("k_in", k_in as u32)
            .grid_3d((out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved((n_blocks * 36 + n_rows * k_in * dt.size_bytes()) as u64)
    }

    // O-LoRA-A shape: 8 groups, per-group in=4096, out=8192, 256 tokens.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemm_q8_mpp(dt: DType) -> BenchSetup {
        let n_rows = 256usize;
        let out_dim = 8192usize;
        let k_in = 4096usize;
        let n_groups = 8usize;
        let rows_per_group = out_dim / n_groups;
        let n_blocks = out_dim * k_in / 32;
        BenchSetup::new(ffai_grouped_gemm_q8_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", n_rows * n_groups * k_in, dt))
            .buffer(BenchBuffer::random("qs", n_blocks * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_rows * out_dim, dt).output())
            .constexpr("n_rows", n_rows as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("n_groups", n_groups as u32)
            .constexpr("rows_per_group", rows_per_group as u32)
            .grid_3d((out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved((n_blocks * 36 + n_rows * n_groups * k_in * dt.size_bytes()) as u64)
    }
}

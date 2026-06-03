//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! HIGH-THROUGHPUT amortized MoE IQ2_XXS grouped BGEMM — the bm64 tiling
//! (64×64×32 tiles, 4 simdgroups / 128 threads, 2×2 warp grid) applied to
//! the IQ2_XXS dequant. The existing moe_bgemm_iq2xxs_mpp uses a tiny
//! 16×32×16 / 1-simdgroup tile (~4-10 GB/s) that is occupancy- and
//! amortization-starved; this mirrors moe_mpp_bm64's fast structure (each
//! 64-row weight tile dequanted once, reused across 64 token rows; 4 SGs for
//! occupancy) to reach high amortized-GEMM throughput.
//!
//! Pool format (slot-major, same as moe_bgemm_iq2xxs_mpp): qs u32
//! [n_experts*nblk*16], d_f32 [n_experts*nblk], nblk = n_out*k_in/256.
//! `indices[row]` = packed expert/slot. x [m_total,k_in], out [m_total,n_out].

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_bgemm_iq2xxs_bm64<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    grid: Tensor<u8>,
    signs: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let nblk_per_expert = n_out * k_in / 256u32;
    // X coop-load: 128 lanes × 16 contiguous K = 2048 = BM(64)×TG_LD(32).
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
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 64u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 64u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 64u32;
        let mut found = 0u32;
        for _ii in range(0u32, 64u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 64u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 64u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 64u32);
        if cur_valid {
            let qs_expert_base = cur_expert * nblk_per_expert * 16u32;
            let d_expert_base = cur_expert * nblk_per_expert;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                // Stage X[m_tile_base..+64, kb..kb+32] → Xs. 128 lanes × 16.
                let gr_x = m_tile_base + x_m_row;
                let in_run_x = (x_m_row >= sub_offset) & (x_m_row < sub_end) & (gr_x < m_total);
                let safe_gr_x = select(in_run_x, gr_x, 0u32);
                let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
                let x_ws_base = x_m_row * 32u32 + x_k_base;
                for _i in range(0u32, 16u32, 1u32) {
                    let xv = load(x[x_dev_base + _i]).cast::<f32>();
                    threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
                }
                // Dequant W[expert, n_tile_base..+64, kb..kb+32] → Ws via the
                // IQ2_XXS formula. 128 lanes × 16 = 2048 = BN(64)×BK(32). The
                // 16 elements a lane fills all share ONE w_row + ONE 32-block
                // (flat=lane*16.._+15 never crosses a 32-boundary; kb is a
                // mult of 32), so aux_idx/aux_sgn/db/block are CONSTANT across
                // them — hoist those global loads out of the inner loop (16→1,
                // the block-at-a-time dequant). Only the octet/sign extraction
                // varies per element.
                let flat0 = lane_in_tg * 16u32;
                let w_row = flat0 / 32u32; // 0..63 (BN row)
                let k_local0 = flat0 & 31u32; // 0 or 16
                let global_row = n_tile_base + w_row;
                let vidx0 = global_row * k_in + kb + k_local0;
                let block = vidx0 / 256u32;
                let group = (vidx0 & 255u32) / 32u32;
                let aux_idx = load(qs[qs_expert_base + block * 16u32 + group * 2u32]);
                let aux_sgn = load(qs[qs_expert_base + block * 16u32 + group * 2u32 + 1u32]);
                let scale_4bit = aux_sgn >> 28u32;
                let db = load(d_f32[d_expert_base + block])
                    * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
                let ws_row_base = w_row * 32u32;
                for _i in range(0u32, 16u32, 1u32) {
                    let k_local = k_local0 + _i; // 0..15 or 16..31
                    let octet_within_index = (k_local & 31u32) / 8u32;
                    let lane_in_octet = k_local & 7u32;
                    let grid_key = (aux_idx >> (octet_within_index * 8u32)) & 0xffu32;
                    let octet = load(grid[grid_key * 8u32 + lane_in_octet])
                        .cast::<u32>()
                        .cast::<i32>()
                        .cast::<f32>();
                    let sign_idx = (aux_sgn >> (octet_within_index * 7u32)) & 0x7fu32;
                    let sign_mask = load(signs[sign_idx]).cast::<u32>();
                    let lane_bit = sign_mask & (1u32 << lane_in_octet);
                    let sign = select(lane_bit != 0u32, -1.0f32, 1.0f32);
                    let w = (db * sign * octet).cast::<T>().cast::<f32>();
                    threadgroup_store("Ws", ws_row_base + k_local, w);
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
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
                    let v = threadgroup_load(
                        "OutScratch",
                        src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
                    );
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_bgemm_iq2xxs_bm64;

    #[bench(name = "ffai/moe/bgemm_iq2xxs_bm64", dtypes = [f32, f16, bf16])]
    fn bench_bgemm_iq2xxs_bm64(dt: DType) -> BenchSetup {
        let n_experts = 4usize;
        let k_in = 4096usize;
        let n_out = 2048usize;
        let t_rows = 256usize;
        let nblk = n_out * k_in / 256;
        BenchSetup::new(ffai_moe_bgemm_iq2xxs_bm64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", t_rows * k_in, dt))
            .buffer(BenchBuffer::random("qs", n_experts * nblk * 16, DType::U32))
            .buffer(BenchBuffer::random("d_f32", n_experts * nblk, DType::F32))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::random("signs", 128, DType::U8))
            .buffer(BenchBuffer::zeros("indices", t_rows, DType::U32))
            .buffer(BenchBuffer::zeros("out", t_rows * n_out, dt).output())
            .constexpr("m_total", t_rows as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .grid_3d(n_out as u32 / 64, (t_rows as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved((n_experts * nblk * 64 + t_rows * k_in * dt.size_bytes()) as u64)
    }
}

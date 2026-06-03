//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Prefill MoE IQ2_XXS WEIGHT-STATIONARY gemv — the amortized fix.
//!
//! The decode `gather_gemv_iq2xxs` runs at ~270 GB/s and `gemv_rows`
//! at ~105 GB/s, but the coop_tile MMA `bgemm_*_bm64` only hits ~21 GB/s
//! (MMA staging + barriers dominate the cheap IQ2 dequant). gemv_rows is
//! fast PER-OP but re-dequants each expert's weight row for EVERY token
//! row (no amortization → loses at large M). This kernel keeps gemv's
//! direct simd_sum dot-product but DEQUANTS each expert's weight row
//! ONCE into threadgroup memory and reuses it across all R rows of that
//! expert in the tile — fast per-op AND amortized.
//!
//! Each threadgroup (32 lanes) owns one output column `m` and a tile of
//! `rows_per_tile` consecutive (token,expert) rows. Rows are pre-permuted
//! by expert (contiguous per expert), so a tile is usually single-expert;
//! the weight row W[expert,m,:] is dequanted into `Ws` once and reused.
//! On an expert boundary inside the tile (rare — ~1 per expert segment)
//! it re-dequants. The expert-change test is threadgroup-uniform
//! (`expert_ids[row]` is the same for all lanes), so the barrier inside
//! the loop is safe. Dequant math identical to gemv_rows / gather_gemv.
//!
//! grid (threadgroups) = [m_out, ceil(m_total/rows_per_tile), 1],
//! threadgroup = [32,1,1].

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gemv_ws_iq2xxs<T>(
    x: Tensor<T>,
    qs_all: Tensor<u32>,
    d_all: Tensor<f32>,
    expert_ids: Tensor<u32>,
    grid: Tensor<u8>,
    signs: Tensor<u8>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] m_total: u32,
    #[constexpr] rows_per_tile: u32,
) {
    let m = tgid_x;
    let tile = tgid_y;
    let lane = tid;
    let row0 = tile * rows_per_tile;

    let blocks_per_row = k_in / 256u32;
    let nblk_per_expert = m_out * blocks_per_row;
    let total_groups = blocks_per_row * 8u32;

    // Ws holds the dequanted weight row W[cur_expert, m, 0..k_in] (f32 in
    // threadgroup so the reuse dot is exact). 4096 = max k_in (gate/up);
    // down (k_in=2048) uses the low half.
    threadgroup_alloc("Ws", 4096, f32);

    let mut cur_expert = 4294967295u32;
    for r in range(0u32, rows_per_tile, 1u32) {
        let row = row0 + r;
        if row < m_total {
            let expert = load(expert_ids[row]);
            // (Re)dequant W[expert, m, :] into Ws on an expert change. The
            // test is threadgroup-uniform → the barriers are safe. A row-dot
            // only READS Ws, so no per-row barrier is needed — only fence Ws
            // before overwriting it on a (rare) expert change.
            if expert != cur_expert {
                if cur_expert != 4294967295u32 {
                    threadgroup_barrier(); // prior rows' reads complete before overwrite
                }
                let qs_row_base = (expert * nblk_per_expert + m * blocks_per_row) * 16u32;
                let d_row_base = expert * nblk_per_expert + m * blocks_per_row;
                for grp in range(lane, total_groups, 32u32) {
                    let b = grp / 8u32;
                    let group = grp & 7u32;
                    let aux_idx = load(qs_all[qs_row_base + b * 16u32 + group * 2u32]);
                    let aux_sgn = load(qs_all[qs_row_base + b * 16u32 + group * 2u32 + 1u32]);
                    let scale_4bit = aux_sgn >> 28u32;
                    let db = load(d_all[d_row_base + b])
                        * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
                    let x_grp = b * 256u32 + group * 32u32;
                    for j in range(0u32, 4u32, 1u32) {
                        let grid_key = (aux_idx >> (j * 8u32)) & 0xffu32;
                        let grid_row_base = grid_key * 8u32;
                        let sign_idx = (aux_sgn >> (j * 7u32)) & 0x7fu32;
                        let sign_mask = load(signs[sign_idx]).cast::<u32>();
                        for l in range(0u32, 8u32, 1u32) {
                            let octet = load(grid[grid_row_base + l])
                                .cast::<u32>()
                                .cast::<i32>()
                                .cast::<f32>();
                            let lane_bit = sign_mask & (1u32 << l);
                            let sign = select(lane_bit != 0u32, -1.0f32, 1.0f32);
                            let w = db * sign * octet;
                            threadgroup_store("Ws", x_grp + j * 8u32 + l, w);
                        }
                    }
                }
                threadgroup_barrier();
                cur_expert = expert;
            }
            // Dot Ws . x[row] over this lane's k-stride, then simd-reduce.
            let x_base = row * k_in;
            let mut acc = 0.0f32;
            for grp in range(lane, total_groups, 32u32) {
                let b = grp / 8u32;
                let group = grp & 7u32;
                let x_grp = b * 256u32 + group * 32u32;
                for e in range(0u32, 32u32, 1u32) {
                    let k = x_grp + e;
                    let w = threadgroup_load("Ws", k);
                    let xv = load(x[x_base + k]).cast::<f32>();
                    acc = acc + w * xv.cast::<f32>();
                }
            }
            let total = simd_sum(acc);
            if lane == 0u32 {
                store(out[row * m_out + m], total.cast::<T>());
            }
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_gemv_ws_iq2xxs;

    // M=256 rows, production gate/up dims, 8 rows/tile.
    #[bench(name = "ffai/moe/gemv_ws_iq2xxs", dtypes = [f32, f16, bf16])]
    fn bench_gemv_ws_iq2xxs(dt: DType) -> BenchSetup {
        let m_total = 256usize;
        let n_experts = 8usize;
        let m_out = 2048usize;
        let k_in = 4096usize;
        let rows_per_tile = 8usize;
        let nblk = m_out * (k_in / 256);
        BenchSetup::new(ffai_moe_gemv_ws_iq2xxs::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("qs_all", n_experts * nblk * 16, DType::U32))
            .buffer(BenchBuffer::random("d_all", n_experts * nblk, DType::F32))
            .buffer(BenchBuffer::zeros("expert_ids", m_total, DType::U32))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::random("signs", 128, DType::U8))
            .buffer(BenchBuffer::zeros("out", m_total * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("m_total", m_total as u32)
            .constexpr("rows_per_tile", rows_per_tile as u32)
            .grid_3d(m_out as u32, (m_total as u32).div_ceil(rows_per_tile as u32), 1, [32, 1, 1])
            .bytes_moved((m_total * nblk * 64 + m_total * k_in * dt.size_bytes()) as u64)
    }
}

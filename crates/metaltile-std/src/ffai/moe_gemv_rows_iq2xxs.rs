//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Prefill MoE IQ2_XXS GEMV-over-rows — the fast decode gemv
//! (ffai_moe_gather_gemv_iq2xxs, ~270 GB/s) applied to a whole batch of
//! M = N*topK (token,expert) rows in ONE dispatch, instead of the
//! coop-tile MMA bgemm (ffai_moe_bgemm_iq2xxs_mpp, ~4-10 GB/s — 15-70x
//! slower). The bgemm's MMA staging + barriers dominate at these quant
//! shapes; the direct simd_sum dot-product the gemv uses is far faster.
//!
//! Each threadgroup (32 lanes) computes one output element
//! out[row, m] = dot(W[expert(row), m, :], x[row, :]), reading the
//! resident split qs/d pool. Rows are pre-permuted by expert (so x[row]
//! is the gathered activation for that (token,expert) pair) and
//! `expert_ids[row]` is the row's expert. NO amortization of the weight
//! dequant across rows of the same expert (each row re-dequants), but the
//! per-op throughput is so much higher than the MMA path that it still
//! wins; a weight-stationary version is the next step if this is the
//! bottleneck. Dequant math is identical to gather_gemv_iq2xxs.
//!
//! grid (threadgroups) = [m_out, m_total, 1], threadgroup = [32,1,1].

use metaltile::kernel;

#[kernel]
pub fn ffai_moe_gemv_rows_iq2xxs<T>(
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
) {
    let m = tgid_x; // output row (0..m_out)
    let row = tgid_y; // (token,expert) pair (0..m_total)
    let lane = tid;
    // grid y-extent is exactly m_total, so row is always in range.

    let blocks_per_row = k_in / 256u32;
    let nblk_per_expert = m_out * blocks_per_row;
    let expert = load(expert_ids[row]);
    let qs_row_base = (expert * nblk_per_expert + m * blocks_per_row) * 16u32;
    let d_row_base = expert * nblk_per_expert + m * blocks_per_row;
    let x_base = row * k_in;

    let total_groups = blocks_per_row * 8u32;
    let mut acc = 0.0f32;
    for grp in range(lane, total_groups, 32u32) {
        let b = grp / 8u32;
        let group = grp & 7u32;
        let aux_idx = load(qs_all[qs_row_base + b * 16u32 + group * 2u32]);
        let aux_sgn = load(qs_all[qs_row_base + b * 16u32 + group * 2u32 + 1u32]);
        let scale_4bit = aux_sgn >> 28u32;
        let db =
            load(d_all[d_row_base + b]) * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
        let x_grp = b * 256u32 + group * 32u32;
        for j in range(0u32, 4u32, 1u32) {
            let grid_key = (aux_idx >> (j * 8u32)) & 0xffu32;
            let grid_row_base = grid_key * 8u32;
            let sign_idx = (aux_sgn >> (j * 7u32)) & 0x7fu32;
            let sign_mask = load(signs[sign_idx]).cast::<u32>();
            for l in range(0u32, 8u32, 1u32) {
                let octet = load(grid[grid_row_base + l]).cast::<u32>().cast::<i32>().cast::<f32>();
                let lane_bit = sign_mask & (1u32 << l);
                let sign = select(lane_bit != 0u32, -1.0f32, 1.0f32);
                let w = (db * sign * octet).cast::<T>().cast::<f32>();
                let xv = load(x[x_base + x_grp + j * 8u32 + l]).cast::<f32>();
                acc = acc + w * xv;
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row * m_out + m], total.cast::<T>());
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_gemv_rows_iq2xxs;

    // M=256 rows, production gate/up dims (m_out=2048, k_in=4096).
    #[bench(name = "ffai/moe/gemv_rows_iq2xxs", dtypes = [f32, f16, bf16])]
    fn bench_gemv_rows_iq2xxs(dt: DType) -> BenchSetup {
        let m_total = 256usize;
        let n_experts = 8usize;
        let m_out = 2048usize;
        let k_in = 4096usize;
        let nblk = m_out * (k_in / 256);
        BenchSetup::new(ffai_moe_gemv_rows_iq2xxs::kernel_ir_for(dt))
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
            .grid_3d(m_out as u32, m_total as u32, 1, [32, 1, 1])
            .bytes_moved((m_total * nblk * 64 + m_total * k_in * dt.size_bytes()) as u64)
    }
}

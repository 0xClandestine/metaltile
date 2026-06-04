//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Prefill MoE Q2_K WEIGHT-STATIONARY gemv (down projection) — the Q2_K
//! twin of ffai_moe_gemv_ws_iq2xxs. Dequants each expert's weight row
//! W_down[expert,m,:] ONCE into threadgroup memory and reuses it across
//! all rows of that expert in the tile (amortized like bm64 but at gemv
//! speed). Rows are pre-permuted by expert (contiguous), so a tile is
//! usually single-expert; on an expert boundary it re-dequants. The
//! expert-change test is threadgroup-uniform → the barrier is safe.
//! Canonical Q2_K layout + dequant identical to gemv_rows_q2k.
//!
//! grid (threadgroups) = [m_out, ceil(m_total/rows_per_tile), 1],
//! threadgroup = [32,1,1].

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gemv_ws_q2k<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    scales: Tensor<u8>,
    d_f32: Tensor<f32>,
    dmin_f32: Tensor<f32>,
    expert_ids: Tensor<u32>,
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

    // Ws holds the dequanted weight row W[cur_expert, m, 0..k_in] (f32).
    // 4096 ≥ k_in (down k_in = intermediate = 2048).
    threadgroup_alloc("Ws", 4096, f32);

    let mut cur_expert = 4294967295u32;
    for r in range(0u32, rows_per_tile, 1u32) {
        let row = row0 + r;
        if row < m_total {
            let expert = load(expert_ids[row]);
            if expert != cur_expert {
                if cur_expert != 4294967295u32 {
                    threadgroup_barrier();
                }
                let qs_expert_base = (expert * nblk_per_expert) * 16u32;
                let sc_expert_base = (expert * nblk_per_expert) * 16u32;
                let blk_expert_base = expert * nblk_per_expert;
                for k in range(lane, k_in, 32u32) {
                    let vidx = m * k_in + k;
                    let block = vidx / 256u32;
                    let in_block = vidx & 255u32;
                    let half = in_block / 128u32;
                    let yh = in_block - half * 128u32;
                    let jg = yh / 32u32;
                    let yg = yh - jg * 32u32;
                    let sub_half = yg / 16u32;
                    let l = yg - sub_half * 16u32;
                    let shift = jg * 2u32;
                    let q_byte = half * 32u32 + sub_half * 16u32 + l;
                    let sub = half * 8u32 + jg * 2u32 + sub_half;
                    let word_idx = q_byte / 4u32;
                    let byte_in_word = q_byte & 3u32;
                    let word = load(qs[qs_expert_base + block * 16u32 + word_idx]);
                    let qs_byte = (word >> (byte_in_word * 8u32)) & 0xffu32;
                    let q_2bit = (qs_byte >> shift) & 0x3u32;
                    let scale_byte =
                        load(scales[sc_expert_base + block * 16u32 + sub]).cast::<u32>();
                    let scale_4bit = scale_byte & 0xfu32;
                    let min_4bit = (scale_byte >> 4u32) & 0xfu32;
                    let d = load(d_f32[blk_expert_base + block]);
                    let dmin = load(dmin_f32[blk_expert_base + block]);
                    let wq = d
                        * scale_4bit.cast::<i32>().cast::<f32>()
                        * q_2bit.cast::<i32>().cast::<f32>()
                        - dmin * min_4bit.cast::<i32>().cast::<f32>();
                    threadgroup_store("Ws", k, wq);
                }
                threadgroup_barrier();
                cur_expert = expert;
            }
            let x_base = row * k_in;
            let mut acc = 0.0f32;
            for k in range(lane, k_in, 32u32) {
                let w = threadgroup_load("Ws", k);
                let xv = load(x[x_base + k]).cast::<f32>();
                acc = acc + w * xv.cast::<f32>();
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

    use super::ffai_moe_gemv_ws_q2k;

    // M=256 rows, production down dims (m_out=4096 hidden, k_in=2048).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_ws_q2k(dt: DType) -> BenchSetup {
        let m_total = 256usize;
        let n_experts = 8usize;
        let m_out = 4096usize;
        let k_in = 2048usize;
        let rows_per_tile = 8usize;
        let nblk = m_out * (k_in / 256);
        BenchSetup::new(ffai_moe_gemv_ws_q2k::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("qs", n_experts * nblk * 16, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * nblk * 16, DType::U8))
            .buffer(BenchBuffer::random("d_f32", n_experts * nblk, DType::F32))
            .buffer(BenchBuffer::random("dmin_f32", n_experts * nblk, DType::F32))
            .buffer(BenchBuffer::zeros("expert_ids", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("m_total", m_total as u32)
            .constexpr("rows_per_tile", rows_per_tile as u32)
            .grid_3d(m_out as u32, (m_total as u32).div_ceil(rows_per_tile as u32), 1, [32, 1, 1])
            .bytes_moved((m_total * nblk * 84 + m_total * k_in * dt.size_bytes()) as u64)
    }
}

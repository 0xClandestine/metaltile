//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! HIGH-THROUGHPUT amortized MoE Q4 grouped BGEMM — bm64 tiling (64×64×32,
//! 4 simdgroups) with the bench's signed-4-bit dequant. The Q4 twin of
//! `ffai_moe_bgemm_q2k_bm64`: processes a 64-row M-tile whose rows are
//! PRE-SORTED by expert id (`indices[row]`), finds contiguous same-expert
//! sub-runs, and runs one MMA GEMM per sub-run against that expert's weights.
//! This replaces the per-token MoE gather loop (which was 72% of prefill time
//! at ~0.1% of the tensor-core peak) with a batched grouped-GEMM over S.
//!
//! Expert weight pool is CONTIGUOUS `[n_experts*n_out, k_in]` Q4:
//!   qs     [n_experts * n_out * (k_in/32) * 4]  u32  (4 words/block, 8 nib/word)
//!   scales [n_experts * n_out * (k_in/32)]      f16  (per-32-block amax/7)
//! Expert e, output row o sits at flat weight row `e*n_out + o`.
//!
//! x `[m_total, k_in]` (token rows sorted by expert); indices `[m_total]`
//! (expert per row); out `[m_total, n_out]`. Name has `_mpp_`-free but the
//! coop_tile ops force the MMA path. The host sorts tokens by expert + scatters
//! the output back to token order.
//!
//! grid (threadgroups) = [n_out/64, ceil(m_total/64), 1], tg [128,1,1].

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_bgemm_q4_bm64<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    scales: Tensor<f16>,
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
    let bpr = k_in / 32u32; // Q4 blocks per weight row
    let nblk_per_expert = n_out * bpr; // Q4 blocks per expert
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
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
            let qs_expert_base = cur_expert * nblk_per_expert * 4u32;
            let sc_expert_base = cur_expert * nblk_per_expert;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                let gr_x = m_tile_base + x_m_row;
                let in_run_x = (x_m_row >= sub_offset) & (x_m_row < sub_end) & (gr_x < m_total);
                let safe_gr_x = select(in_run_x, gr_x, 0u32);
                let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
                let x_ws_base = x_m_row * 32u32 + x_k_base;
                for _i in range(0u32, 16u32, 1u32) {
                    let xv = load(x[x_dev_base + _i]).cast::<f32>();
                    threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
                }
                // Dequant Q4 W → Ws. 128 lanes × 16. Each lane owns 16
                // consecutive k of one w_row; they share the same 32-block when
                // k_local0 is 32-aligned (it is: flat0 = lane*16, 16|32).
                for _i in range(0u32, 16u32, 1u32) {
                    let flat = lane_in_tg * 16u32 + _i;
                    let w_row = flat / 32u32; // 0..63 (output feature within tile)
                    let k_local = flat & 31u32; // 0..31 (BK)
                    let global_row = n_tile_base + w_row; // output feature within expert
                    let k = kb + k_local;
                    let blk = global_row * bpr + k / 32u32; // block within expert
                    let lane = k & 31u32;
                    let word = load(qs[qs_expert_base + blk * 4u32 + lane / 8u32]);
                    let nib = (word >> ((lane & 7u32) * 4u32)) & 0xfu32;
                    let q_signed = select(nib >= 8u32, nib - 16u32, nib);
                    let qf = q_signed.cast::<i32>().cast::<f32>();
                    let sc = load(scales[sc_expert_base + blk]).cast::<f32>();
                    threadgroup_store("Ws", w_row * 32u32 + k_local, (sc * qf).cast::<T>().cast::<f32>());
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

#[cfg(test)]
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_moe_bgemm_q4_bm64;
    use crate::utils::pack_f32;

    fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
        let bpr = k / 32;
        let mut qs = vec![0u32; m * bpr * 4];
        let mut scales = vec![0f32; m * bpr];
        for r in 0..m {
            for b in 0..bpr {
                let base = r * k + b * 32;
                let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
                let d = amax / 7.0;
                scales[r * bpr + b] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for word in 0..4 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                        packed |= ((q as u32) & 0xf) << (i * 4);
                    }
                    qs[r * bpr * 4 + b * 4 + word] = packed;
                }
            }
        }
        (qs, scales)
    }

    fn naive(x: &[f32], qs: &[u32], sc16: &[f32], idx: &[u32], m_total: usize, n_out: usize, k_in: usize) -> Vec<f32> {
        let bpr = k_in / 32;
        let nblk_per_expert = n_out * bpr;
        let mut out = vec![0f32; m_total * n_out];
        for r in 0..m_total {
            let e = idx[r] as usize;
            for o in 0..n_out {
                let mut acc = 0f32;
                for kk in 0..k_in {
                    let blk = e * nblk_per_expert + o * bpr + kk / 32;
                    let lane = kk % 32;
                    let word = qs[blk * 4 + lane / 8];
                    let nib = (word >> ((lane % 8) * 4)) & 0xf;
                    let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                    acc += (q as f32 * sc16[blk]) * x[r * k_in + kk];
                }
                out[r * n_out + o] = acc;
            }
        }
        out
    }

    fn setup(dt: DType) -> TestSetup {
        let (n_exp, m_total, n_out, k_in) = (4usize, 64usize, 64usize, 64usize);
        let bpr = k_in / 32;
        let xv: Vec<f32> = (0..m_total * k_in).map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.1).collect();
        let wv: Vec<f32> = (0..n_exp * n_out * k_in).map(|i| (i as f32 * 0.019 - 0.2).cos() * 0.8).collect();
        let (qs, scales) = quantize_q4(&wv, n_exp * n_out, k_in);
        let sc16: Vec<f32> = scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
        // Sorted-by-expert indices: 16 rows each of expert 0,1,2,3.
        let idx: Vec<u32> = (0..m_total).map(|r| (r / 16) as u32).collect();
        let expected = naive(&xv, &qs, &sc16, &idx, m_total, n_out, k_in);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let idx_bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = bpr;
        TestSetup::new(ffai_moe_bgemm_q4_bm64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&xv, dt), dt))
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F16), DType::F16))
            .input(TestBuffer::from_vec("indices", idx_bytes, DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_out as u32) / 64, (m_total as u32).div_ceil(64), 1, [128, 1, 1])
    }

    #[test_kernel(dtypes = [f16, bf16], tol = [3e-2, 2e-1])]
    fn test_moe_bgemm_q4_bm64(dt: DType) -> TestSetup { setup(dt) }
}

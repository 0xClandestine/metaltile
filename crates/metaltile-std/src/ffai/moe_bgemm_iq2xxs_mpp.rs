//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP MoE IQ2_XXS grouped BGEMM — the prefill counterpart of
//! `moe_gather_gemv_iq2xxs`. Multi-token (M>1) grouped matmul: rows are
//! pre-permuted by expert (`indices[row]` = packed expert slot), and each
//! expert's weight is read+dequanted ONCE and reused across all the tokens
//! routed to it — the amortization that makes MoE prefill fast.
//!
//! Structure is identical to `moe_mpp::mt_moe_gather_qmm_mma_int4_bm16_mpp`
//! (BM=16 / BN=32 / BK=16 coop-tile MMA, contiguous-expert sub-run walk);
//! ONLY the weight-staging loop differs — int4 nibble unpack → IQ2_XXS
//! grid+sign dequant (the same formula as the gemv, see
//! `gguf_dequant_iq2_xxs`).
//!
//! Layout: `qs [n_experts * nblk * 16] u32`, `d_f32 [n_experts * nblk]`,
//! where `nblk = n_out * k_in / 256`; weight value (row r, col k) lives at
//! flat index `r*k_in + k` → block `idx/256`, etc. `x [m_total, k_in]`,
//! `out [m_total, n_out]`. `grid [2048] u8`, `signs [128] u8`. Caller
//! supplies the by-expert-sorted rows + `indices`.

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gather_bgemm_iq2xxs_mpp<T>(
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
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let nblk_per_expert = n_out * k_in / 256u32;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let qs_expert_base = cur_expert * nblk_per_expert * 16u32;
            let d_expert_base = cur_expert * nblk_per_expert;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                // Stage X[m_tile_base..+16, kb..kb+16] → xs. 32 lanes × 8.
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws via the
                // IQ2_XXS grid+sign formula. 32 lanes = 32 BN rows; each lane
                // dequants its row's 16 K-tile values.
                let w_row = lane; // 0..31
                let global_row = n_tile_base + w_row;
                for _kc in range(0u32, 16u32, 1u32) {
                    let k = kb + _kc;
                    let vidx = global_row * k_in + k;
                    let block = vidx / 256u32;
                    let in_block = vidx & 255u32;
                    let group = in_block / 32u32;
                    let in_group = in_block & 31u32;
                    let octet_within_index = in_group / 8u32;
                    let lane_in_octet = in_group & 7u32;
                    let aux_idx = load(qs[qs_expert_base + block * 16u32 + group * 2u32]);
                    let aux_sgn = load(qs[qs_expert_base + block * 16u32 + group * 2u32 + 1u32]);
                    let scale_4bit = aux_sgn >> 28u32;
                    let db = load(d_f32[d_expert_base + block])
                        * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
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
                    threadgroup_store("ws", w_row * 16u32 + _kc, w);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
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

    use super::ffai_moe_gather_bgemm_iq2xxs_mpp;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_bgemm_iq2xxs_mpp(dt: DType) -> BenchSetup {
        let n_experts = 4usize;
        let k_in = 4096usize;
        let n_out = 2048usize;
        let t_rows = 256usize;
        let nblk = n_out * k_in / 256;
        BenchSetup::new(ffai_moe_gather_bgemm_iq2xxs_mpp::kernel_ir_for(dt))
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
            .grid_3d(n_out as u32 / 32, (t_rows as u32).div_ceil(16), 1, [32, 1, 1])
            .bytes_moved((n_experts * nblk * 64 + t_rows * k_in * dt.size_bytes()) as u64)
    }
}

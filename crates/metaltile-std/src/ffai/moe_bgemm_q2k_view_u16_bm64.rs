//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! POOL-FREE amortized MoE Q2_K bgemm — bm64 64×64×32 coop_tile MMA reading RAW
//! resident GGUF Q2_K blocks directly (no CPU deinterleave pool). The down-
//! projection twin of `moe_bgemm_iq2xxs_view_u16_bm64` (which won for gate/up);
//! the existing `moe_bgemm_q2k_view` is the OLD mpp tiling (slower than bm64).
//!
//! Q2_K block = 84 bytes = 42 u16: scales[0..16] u8 (4-bit d-scale low nibble +
//! 4-bit dmin-scale high nibble per 16-value sub-block), qs[16..80] (64 bytes,
//! 2-bit quants), d f16 @80, dmin f16 @82. `indices[row]` = slot/expert id; a
//! block's bytes start at `tensor_byte_off + expert*expert_byte_stride +
//! block*84`. d/dmin read via the f16 view (native half — exact), scales/qs via
//! the u16 view. x [m_total,k_in], out [m_total,n_out]. Live-compiled (bgemm).

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_bgemm_q2k_view_u16_bm64<T>(
    x: Tensor<T>,
    view_u16: Tensor<u16>,
    view_f16: Tensor<f16>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] tensor_byte_off: u32,
    #[constexpr] expert_byte_stride: u32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
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
            // u16 base of this expert's raw Q2_K blocks (84 bytes = 42 u16 each).
            let expert_u16_base = (tensor_byte_off + cur_expert * expert_byte_stride) / 2u32;
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
                // Dequant Q2_K W → Ws. 16 elems/lane share one w_row + 256-block
                // → hoist d/dmin (read once via f16 view at block bytes 80/82 =
                // u16 40/41). scales/qs vary per element.
                let flat0 = lane_in_tg * 16u32;
                let w_row = flat0 / 32u32;
                let k_local0 = flat0 & 31u32;
                let global_row = n_tile_base + w_row;
                let block = (global_row * k_in + kb + k_local0) / 256u32;
                let blk_u16 = expert_u16_base + block * 42u32;
                let d = load(view_f16[blk_u16 + 40u32]).cast::<f32>();
                let dmin = load(view_f16[blk_u16 + 41u32]).cast::<f32>();
                let ws_row_base = w_row * 32u32;
                for _i in range(0u32, 16u32, 1u32) {
                    let k_local = k_local0 + _i;
                    let vidx = global_row * k_in + kb + k_local;
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
                    // qs word (u32) covering this byte: block bytes 16+word_idx*4,
                    // i.e. u16 offset blk_u16 + 8 + word_idx*2 (qs starts at byte 16).
                    let word_idx = q_byte / 4u32;
                    let byte_in_word = q_byte & 3u32;
                    let qw = blk_u16 + 8u32 + word_idx * 2u32;
                    let word = load(view_u16[qw]).cast::<u32>()
                        | (load(view_u16[qw + 1u32]).cast::<u32>() << 16u32);
                    let qs_byte = (word >> (byte_in_word * 8u32)) & 0xffu32;
                    let q_2bit = (qs_byte >> shift) & 0x3u32;
                    // scale byte `sub` is at block byte `sub` = u16 blk_u16 + sub/2.
                    let sc_word = load(view_u16[blk_u16 + sub / 2u32]).cast::<u32>();
                    let scale_byte = (sc_word >> ((sub & 1u32) * 8u32)) & 0xffu32;
                    let scale_4bit = scale_byte & 0xfu32;
                    let min_4bit = (scale_byte >> 4u32) & 0xfu32;
                    let wq = d
                        * scale_4bit.cast::<i32>().cast::<f32>()
                        * q_2bit.cast::<i32>().cast::<f32>()
                        - dmin * min_4bit.cast::<i32>().cast::<f32>();
                    threadgroup_store("Ws", ws_row_base + k_local, wq.cast::<T>().cast::<f32>());
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

    use super::ffai_moe_bgemm_q2k_view_u16_bm64;

    #[bench(name = "ffai/moe/bgemm_q2k_view_u16_bm64", dtypes = [f32, f16, bf16])]
    fn bench(dt: DType) -> BenchSetup {
        let n_experts = 4usize;
        let k_in = 2048usize;
        let n_out = 4096usize;
        let t_rows = 256usize;
        let nblk = n_out * k_in / 256;
        BenchSetup::new(ffai_moe_bgemm_q2k_view_u16_bm64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", t_rows * k_in, dt))
            .buffer(BenchBuffer::random("view_u16", n_experts * nblk * 42, DType::U16))
            .buffer(BenchBuffer::random("view_f16", n_experts * nblk * 42, DType::F16))
            .buffer(BenchBuffer::zeros("indices", t_rows, DType::U32))
            .buffer(BenchBuffer::zeros("out", t_rows * n_out, dt).output())
            .constexpr("m_total", t_rows as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("tensor_byte_off", 0u32)
            .constexpr("expert_byte_stride", (nblk * 84) as u32)
            .grid_3d(n_out as u32 / 64, (t_rows as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved((n_experts * nblk * 84 + t_rows * k_in * dt.size_bytes()) as u64)
    }
}

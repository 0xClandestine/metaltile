//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! ZERO-COPY MoE Q2_K grouped BGEMM — reads raw 84-byte Q2_K blocks straight
//! from a no-copy mmap VIEW buffer (the down-projection counterpart of
//! `moe_bgemm_iq2xxs_view`), instead of the CPU-repacked qs/scales/d/dmin
//! pool. Eliminates the per-layer mmap→pool repack for the down experts.
//!
//! Identical MMA structure to `moe_bgemm_q2k_mpp`; ONLY the weight read
//! differs. Raw Q2_K block (84 bytes, see gguf_dequant_q2_k + the gather
//! repack): scales[0..16] u8, qs[16..80] (64 bytes), d fp16[80..82], dmin
//! fp16[82..84]. A block's bytes start at
//!   `tensor_byte_off + expert*expert_byte_stride + block*84`.
//! `indices[row]` is the EXPERT id. d/dmin are decoded inline from their raw
//! 2 bytes (IEEE binary16), so no buffer aliasing and no CPU extract.

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_bgemm_q2k_view<T>(
    x: Tensor<T>,
    view_u8: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] tensor_byte_off: u32,
    #[constexpr] expert_byte_stride: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    threadgroup_alloc("xs", 256, coop_stage(T));
    threadgroup_alloc("ws", 512, coop_stage(T));
    threadgroup_alloc("out_scratch", 512, f32);
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
            let expert_byte_base = tensor_byte_off + cur_expert * expert_byte_stride;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
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
                // Dequant Q2_K W[expert, n_tile_base+lane, kb..kb+16] → ws,
                // reading raw 84-byte Q2_K blocks straight from the mmap view.
                let w_row = lane;
                let global_row = n_tile_base + w_row;
                for _kc in range(0u32, 16u32, 1u32) {
                    let k = kb + _kc;
                    let vidx = global_row * k_in + k;
                    let block = vidx / 256u32;
                    let in_block = vidx & 255u32;
                    // Canonical Q2_K layout (see gguf_dequant_q2_k.rs).
                    let half = in_block / 128u32;
                    let yh = in_block - half * 128u32;
                    let jg = yh / 32u32;
                    let yg = yh - jg * 32u32;
                    let sub_half = yg / 16u32;
                    let l = yg - sub_half * 16u32;
                    let shift = jg * 2u32;
                    let q_byte = half * 32u32 + sub_half * 16u32 + l;
                    let sub = half * 8u32 + jg * 2u32 + sub_half;
                    let blk_byte = expert_byte_base + block * 84u32;
                    // scales[0..16], qs[16..80], d fp16[80..82], dmin fp16[82..84].
                    let scale_byte = load(view_u8[blk_byte + sub]).cast::<u32>();
                    let scale_4bit = scale_byte & 0xfu32;
                    let min_4bit = (scale_byte >> 4u32) & 0xfu32;
                    let qs_byte = load(view_u8[blk_byte + 16u32 + q_byte]).cast::<u32>();
                    let q_2bit = (qs_byte >> shift) & 0x3u32;
                    // d, dmin decoded inline from raw fp16 (IEEE binary16).
                    let d_lo = load(view_u8[blk_byte + 80u32]).cast::<u32>();
                    let d_hi = load(view_u8[blk_byte + 81u32]).cast::<u32>();
                    let d_bits = d_lo | (d_hi << 8u32);
                    let d_sign = select((d_bits & 0x8000u32) != 0u32, 0.0f32 - 1.0f32, 1.0f32);
                    let d_exp = (d_bits >> 10u32) & 0x1fu32;
                    let d_mant = (d_bits & 0x3ffu32).cast::<i32>().cast::<f32>();
                    let d_norm = (1.0f32 + d_mant / 1024.0f32)
                        * exp2(d_exp.cast::<i32>().cast::<f32>() - 15.0f32);
                    let d_sub = d_mant * exp2(0.0f32 - 24.0f32);
                    let d = d_sign * select(d_exp == 0u32, d_sub, d_norm);
                    let m_lo = load(view_u8[blk_byte + 82u32]).cast::<u32>();
                    let m_hi = load(view_u8[blk_byte + 83u32]).cast::<u32>();
                    let m_bits = m_lo | (m_hi << 8u32);
                    let m_sign = select((m_bits & 0x8000u32) != 0u32, 0.0f32 - 1.0f32, 1.0f32);
                    let m_exp = (m_bits >> 10u32) & 0x1fu32;
                    let m_mant = (m_bits & 0x3ffu32).cast::<i32>().cast::<f32>();
                    let m_norm = (1.0f32 + m_mant / 1024.0f32)
                        * exp2(m_exp.cast::<i32>().cast::<f32>() - 15.0f32);
                    let m_sub = m_mant * exp2(0.0f32 - 24.0f32);
                    let dmin = m_sign * select(m_exp == 0u32, m_sub, m_norm);
                    let wq = d
                        * scale_4bit.cast::<i32>().cast::<f32>()
                        * q_2bit.cast::<i32>().cast::<f32>()
                        - dmin * min_4bit.cast::<i32>().cast::<f32>();
                    threadgroup_store("ws", w_row * 16u32 + _kc, wq.cast::<T>().cast::<f32>());
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

    use super::ffai_moe_bgemm_q2k_view;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_bgemm_q2k_view(dt: DType) -> BenchSetup {
        let n_experts = 4usize;
        let k_in = 2048usize;
        let n_out = 4096usize;
        let t_rows = 256usize;
        let nblk = n_out * k_in / 256;
        let view_bytes = n_experts * nblk * 84;
        BenchSetup::new(ffai_moe_bgemm_q2k_view::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", t_rows * k_in, dt))
            .buffer(BenchBuffer::random("view_u8", view_bytes, DType::U8))
            .buffer(BenchBuffer::zeros("indices", t_rows, DType::U32))
            .buffer(BenchBuffer::zeros("out", t_rows * n_out, dt).output())
            .constexpr("m_total", t_rows as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("tensor_byte_off", 0u32)
            .constexpr("expert_byte_stride", (nblk * 84) as u32)
            .grid_3d(n_out as u32 / 32, (t_rows as u32).div_ceil(16), 1, [32, 1, 1])
            .bytes_moved((view_bytes + t_rows * k_in * dt.size_bytes()) as u64)
    }
}

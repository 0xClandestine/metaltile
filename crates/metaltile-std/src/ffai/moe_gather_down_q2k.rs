//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Multi-expert Q2_K gather down-projection + weighted sum — inline
//! dequant of all `n_slots` routed experts' down weights, each applied
//! to that expert's own SwiGLU inner activation, combined by the router
//! weights into the routed MoE output. ONE dispatch replaces
//! `n_slots` × {dequant, gemv} + the `n_slots`-way weighted accumulate.
//!
//! out[m] = Σ_slot  weights[slot] · ( Σ_k  dequant(downW_slot[m, k]) · inner_slot[k] )
//!
//! ## Inputs (Q2_K split format — produced by the FFAI loader)
//!
//! ```text
//!   inners_all  [n_slots * k_in]                 T   — per-slot SwiGLU inner
//!   qs_all      [n_slots * nblk_per_expert * 16]  u32 — 64 qs bytes / block as 16 LE u32
//!   scales_all  [n_slots * nblk_per_expert * 16]  u8  — 16 packed (4-bit scale|4-bit min) / block
//!   d_all       [n_slots * nblk_per_expert]       f32 — per-block super-scale
//!   dmin_all    [n_slots * nblk_per_expert]       f32 — per-block min super-scale
//!   weights     [n_slots]                         f32 — router combine weights
//!   out         [m_out]                           T   — routed MoE output
//!   k_in / m_out / n_slots  (constexpr)
//! ```
//!
//! `nblk_per_expert = m_out * (k_in / 256)`. Output row `m` of an expert
//! occupies blocks `[m * blocks_per_row .. +blocks_per_row)`.
//!
//! ## Dispatch (Reduction mode)
//!
//! grid (threadgroups) = `[m_out, 1, 1]`, threadgroup = `[32, 1, 1]`.
//! `tgid_x = m`, `tid = lane`. The 32-lane simdgroup strides the k axis
//! across all slots, folding the router weight into each partial; one
//! `simd_sum` then lane-0 store. Q2_K dequant math is identical to
//! `gguf_dequant_q2_k::ffai_gguf_dequant_q2_k`.

use metaltile::kernel;

#[kernel]
pub fn ffai_moe_gather_down_q2k<T>(
    inners_all: Tensor<T>,
    qs_all: Tensor<u32>,
    scales_all: Tensor<u8>,
    d_all: Tensor<f32>,
    dmin_all: Tensor<f32>,
    expert_ids: Tensor<u32>,
    weights: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_slots: u32,
) {
    let m = tgid_x;
    let lane = tid;
    let blocks_per_row = k_in / 256u32;
    let nblk_per_expert = m_out * blocks_per_row;

    let mut acc = 0.0f32;
    for slot in range(0u32, n_slots, 1u32) {
        let w_slot = load(weights[slot]);
        // qs_all/scales_all/d_all/dmin_all hold ALL experts (resident);
        // routed expert for this slot is expert_ids[slot].
        let expert = load(expert_ids[slot]);
        let qs_row_base = (expert * nblk_per_expert + m * blocks_per_row) * 16u32;
        let sc_row_base = (expert * nblk_per_expert + m * blocks_per_row) * 16u32;
        let blk_row_base = expert * nblk_per_expert + m * blocks_per_row;
        let inner_base = slot * k_in;
        for k_iter in range(lane, k_in, 32u32) {
            let b = k_iter / 256u32; // block within row
            let in_block = k_iter & 255u32; // 0..255
            // Canonical Q2_K layout (see gguf_dequant_q2_k.rs): 256 values =
            // 2 halves of 128; each half = 4 j-groups of 32; each j-group is
            // two runs of 16 values over 16 CONSECUTIVE qs bytes at a shared
            // shift (j*2). NOT the naive 4-consecutive-per-byte mapping.
            let half = in_block / 128u32;
            let yh = in_block - half * 128u32;
            let jg = yh / 32u32;
            let yg = yh - jg * 32u32;
            let sub_half = yg / 16u32;
            let l = yg - sub_half * 16u32;
            let shift = jg * 2u32;
            let q_byte = half * 32u32 + sub_half * 16u32 + l;
            let sub = half * 8u32 + jg * 2u32 + sub_half; // scale index 0..15
            let word_idx = q_byte / 4u32;
            let byte_in_word = q_byte & 3u32;
            let word = load(qs_all[qs_row_base + b * 16u32 + word_idx]);
            let qs_byte = (word >> (byte_in_word * 8u32)) & 0xffu32;
            let q_2bit = (qs_byte >> shift) & 0x3u32;
            let scale_byte = load(scales_all[sc_row_base + b * 16u32 + sub]).cast::<u32>();
            let scale_4bit = scale_byte & 0xfu32;
            let min_4bit = (scale_byte >> 4u32) & 0xfu32;
            let d = load(d_all[blk_row_base + b]);
            let dmin = load(dmin_all[blk_row_base + b]);
            let wq =
                d * scale_4bit.cast::<i32>().cast::<f32>() * q_2bit.cast::<i32>().cast::<f32>()
                    - dmin * min_4bit.cast::<i32>().cast::<f32>();
            let wq_t = wq.cast::<T>().cast::<f32>();
            let xv = load(inners_all[inner_base + k_iter]).cast::<f32>();
            acc = acc + w_slot * wq_t * xv;
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[m], total.cast::<T>());
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_gather_down_q2k;

    // n_slots=6; production down dims (m_out=4096, k_in=2048).
    #[bench(name = "ffai/moe/gather_down_q2k", dtypes = [f32, f16, bf16])]
    fn bench_gather_down_q2k(dt: DType) -> BenchSetup {
        let n_slots = 6usize;
        let m_out = 4096usize;
        let k_in = 2048usize;
        let nblk = m_out * (k_in / 256);
        BenchSetup::new(ffai_moe_gather_down_q2k::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inners_all", n_slots * k_in, dt))
            .buffer(BenchBuffer::random("qs_all", n_slots * nblk * 16, DType::U32))
            .buffer(BenchBuffer::random("scales_all", n_slots * nblk * 16, DType::U8))
            .buffer(BenchBuffer::random("d_all", n_slots * nblk, DType::F32))
            .buffer(BenchBuffer::random("dmin_all", n_slots * nblk, DType::F32))
            .buffer(BenchBuffer::zeros("expert_ids", n_slots, DType::U32))
            .buffer(BenchBuffer::random("weights", n_slots, DType::F32))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("n_slots", n_slots as u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((n_slots * nblk * 84 + n_slots * k_in * dt.size_bytes()) as u64)
    }
}

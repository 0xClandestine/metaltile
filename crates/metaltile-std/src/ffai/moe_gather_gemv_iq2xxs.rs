//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Multi-expert IQ2_XXS gather GEMV — inline dequant + matmul-vector for
//! all `n_slots` (=6) routed experts of one MoE role (gate / up) in ONE
//! dispatch.
//!
//! ## Why
//!
//! The DSv4 decode FFN ran one `dequant-to-f16` kernel + one `gemv` per
//! routed expert (6 experts × {gate,up} = 12 dispatches/layer × 43 =
//! ~516 cmd buffers just for gate+up), each dominated by Apple's
//! ~0.13 ms per-kernel dispatch floor rather than real compute. This
//! kernel folds (a) the IQ2_XXS dequant, (b) the gemv, and (c) the
//! 6-expert gather into a SINGLE dispatch per role: `out[slot, m]` for
//! `slot in 0..n_slots`, `m in 0..m_out`. No intermediate f16 weight
//! buffer is ever materialised — the quant bytes are read straight from
//! the (resident) split buffers.
//!
//! ## Inputs (split format — produced by the FFAI loader)
//!
//! ```text
//!   x        [k_in]                              T   — shared activation
//!   qs_all   [n_slots * nblk_per_expert * 16]    u32 — slot-major, the 64
//!                                                       bytes of qs[32] per
//!                                                       block re-laid as 16 LE u32
//!   d_all    [n_slots * nblk_per_expert]         f32 — per-block super-scale
//!   grid     [2048]                              u8  — iq2xxs_grid, 256×8 octets
//!   signs    [128]                               u8  — ksigns_iq2xs sign masks
//!   out      [n_slots * m_out]                   T   — result
//!   k_in     u32 (constexpr)  — input dim (multiple of 256)
//!   m_out    u32 (constexpr)  — output rows per expert
//! ```
//!
//! `nblk_per_expert = m_out * (k_in / 256)`. Output row `m` of an expert
//! occupies blocks `[m * blocks_per_row .. +blocks_per_row)` where
//! `blocks_per_row = k_in / 256`.
//!
//! ## Dispatch (Reduction mode)
//!
//! grid (threadgroups) = `[m_out, n_slots, 1]`, threadgroup = `[32, 1, 1]`.
//! `tgid_x = m`, `tgid_y = slot`, `tid = lane` (0..31). The 32 lanes of
//! the single simdgroup stride over the row's groups-of-32, each lane
//! accumulating a partial dot product; `simd_sum` folds them and lane 0
//! stores. The dequant math is identical to
//! `gguf_dequant_iq2_xxs::ffai_gguf_dequant_iq2_xxs` (the proven,
//! production reference) — see that file for the per-element derivation.

use metaltile::kernel;

#[kernel]
pub fn ffai_moe_gather_gemv_iq2xxs<T>(
    x: Tensor<T>,
    qs_all: Tensor<u32>,
    d_all: Tensor<f32>,
    expert_ids: Tensor<u32>,
    grid: Tensor<u8>,
    signs: Tensor<u8>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
) {
    let m = tgid_x;
    let slot = tgid_y;
    let lane = tid;

    let blocks_per_row = k_in / 256u32;
    let nblk_per_expert = m_out * blocks_per_row;
    // `qs_all` / `d_all` hold ALL experts (resident, pre-split); the
    // routed expert for this slot is `expert_ids[slot]`.
    let expert = load(expert_ids[slot]);
    let qs_row_base = (expert * nblk_per_expert + m * blocks_per_row) * 16u32;
    let d_row_base = expert * nblk_per_expert + m * blocks_per_row;

    // Each block = 8 groups of 32 values. Lanes stride over groups.
    let total_groups = blocks_per_row * 8u32;
    let mut acc = 0.0f32;
    for grp in range(lane, total_groups, 32u32) {
        let b = grp / 8u32; // block within the row
        let group = grp & 7u32; // group within the block (0..7)
        let aux_idx = load(qs_all[qs_row_base + b * 16u32 + group * 2u32]);
        let aux_sgn = load(qs_all[qs_row_base + b * 16u32 + group * 2u32 + 1u32]);
        let scale_4bit = aux_sgn >> 28u32;
        let db =
            load(d_all[d_row_base + b]) * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
        // x indices covered by this 32-value group.
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
                // Round the dequanted weight through the activation dtype
                // `T` before the multiply — the unfused path materialises
                // an f16 weight buffer first, so matching that rounding
                // keeps the fused path bit-identical (greedy-stable).
                let w = (db * sign * octet).cast::<T>().cast::<f32>();
                let xv = load(x[x_grp + j * 8u32 + l]).cast::<f32>();
                acc = acc + w * xv;
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[slot * m_out + m], total.cast::<T>());
    }
}

#[cfg(test)]
pub mod kernel_tests {
    use metaltile::test::*;

    use super::ffai_moe_gather_gemv_iq2xxs;

    #[test]
    fn codegen_gather_gemv_iq2xxs_smoke() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let ir = ffai_moe_gather_gemv_iq2xxs::kernel_ir_for(dt);
            assert!(!ir.body.ops.is_empty(), "no ops for {dt:?}");
            assert!(ir.params.iter().any(|p| p.name == "qs_all"), "missing qs_all");
            assert!(ir.params.iter().any(|p| p.name == "grid"), "missing grid");
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_gather_gemv_iq2xxs;

    // n_slots=6 routed experts; production gate/up dims (m_out=2048, k_in=4096).
    #[bench(name = "ffai/moe/gather_gemv_iq2xxs", dtypes = [f32, f16, bf16])]
    fn bench_gather_gemv_iq2xxs(dt: DType) -> BenchSetup {
        let n_slots = 6usize;
        let m_out = 2048usize;
        let k_in = 4096usize;
        let nblk = m_out * (k_in / 256);
        BenchSetup::new(ffai_moe_gather_gemv_iq2xxs::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", k_in, dt))
            .buffer(BenchBuffer::random("qs_all", n_slots * nblk * 16, DType::U32))
            .buffer(BenchBuffer::random("d_all", n_slots * nblk, DType::F32))
            .buffer(BenchBuffer::zeros("expert_ids", n_slots, DType::U32))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::random("signs", 128, DType::U8))
            .buffer(BenchBuffer::zeros("out", n_slots * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .grid_3d(m_out as u32, n_slots as u32, 1, [32, 1, 1])
            .bytes_moved((n_slots * nblk * 64 + n_slots * nblk * 4 + k_in * dt.size_bytes()) as u64)
    }
}

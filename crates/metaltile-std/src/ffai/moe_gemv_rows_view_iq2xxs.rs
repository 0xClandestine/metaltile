//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! SINGLE-COPY fast prefill MoE IQ2_XXS: the gemv-over-rows kernel
//! (moe_gemv_rows_iq2xxs, ~100 GB/s) reading raw 66-byte IQ2_XXS blocks
//! straight from the resident no-copy mmap VIEW (like moe_bgemm_iq2xxs_view)
//! instead of a repacked pool. This is the combination we want:
//! ONE resident copy of the weights (no 70GB repack pool) + the fast direct
//! simd_sum dot-product (no slow coop-tile MMA). Eliminates both the
//! double-memory and the per-chunk repack/re-read.
//!
//! out[row, m] = dot(W[expert(row), m, :], x[row, :]). Block bytes read
//! inline: d via fp16 decode (2 bytes), aux_idx/aux_sgn via u8-combine.
//! grid (threadgroups) = [m_out, m_total, 1], threadgroup = [32,1,1].

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gemv_rows_view_iq2xxs<T>(
    x: Tensor<T>,
    view_u8: Tensor<u8>,
    grid: Tensor<u8>,
    signs: Tensor<u8>,
    expert_ids: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] m_total: u32,
    #[constexpr] tensor_byte_off: u32,
    #[constexpr] expert_byte_stride: u32,
) {
    let m = tgid_x; // output row (0..m_out)
    let row = tgid_y; // (token,expert) pair (0..m_total)
    let lane = tid;

    let blocks_per_row = k_in / 256u32;
    let expert = load(expert_ids[row]);
    let expert_byte_base = tensor_byte_off + expert * expert_byte_stride;
    // Block index base for output row m within the expert (row-major [m_out,k_in]).
    let row_block0 = m * blocks_per_row;
    let x_base = row * k_in;

    let total_groups = blocks_per_row * 8u32;
    let mut acc = 0.0f32;
    for grp in range(lane, total_groups, 32u32) {
        let b = grp / 8u32; // block within the row
        let group = grp & 7u32; // group within the block (0..7)
        let blk_byte = expert_byte_base + (row_block0 + b) * 66u32;
        // d: leading fp16 (2 bytes), decoded inline (IEEE binary16).
        let d_lo = load(view_u8[blk_byte]).cast::<u32>();
        let d_hi = load(view_u8[blk_byte + 1u32]).cast::<u32>();
        let d_bits = d_lo | (d_hi << 8u32);
        let d_sign = select((d_bits & 0x8000u32) != 0u32, 0.0f32 - 1.0f32, 1.0f32);
        let d_exp = (d_bits >> 10u32) & 0x1fu32;
        let d_mant = (d_bits & 0x3ffu32).cast::<i32>().cast::<f32>();
        let d_norm =
            (1.0f32 + d_mant / 1024.0f32) * exp2(d_exp.cast::<i32>().cast::<f32>() - 15.0f32);
        let d_sub = d_mant * exp2(0.0f32 - 24.0f32);
        let dval = d_sign * select(d_exp == 0u32, d_sub, d_norm);
        // group g: aux_idx u32 + aux_sgn u32 at blk_byte+2+g*8, LE u8-combine.
        let q0 = blk_byte + 2u32 + group * 8u32;
        let aux_idx = load(view_u8[q0]).cast::<u32>()
            | (load(view_u8[q0 + 1u32]).cast::<u32>() << 8u32)
            | (load(view_u8[q0 + 2u32]).cast::<u32>() << 16u32)
            | (load(view_u8[q0 + 3u32]).cast::<u32>() << 24u32);
        let aux_sgn = load(view_u8[q0 + 4u32]).cast::<u32>()
            | (load(view_u8[q0 + 5u32]).cast::<u32>() << 8u32)
            | (load(view_u8[q0 + 6u32]).cast::<u32>() << 16u32)
            | (load(view_u8[q0 + 7u32]).cast::<u32>() << 24u32);
        let scale_4bit = aux_sgn >> 28u32;
        let db = dval * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
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

/// U16 variant — reads the raw IQ2_XXS blocks via ALIGNED u16 loads (the
/// block stride 66 and the qs offset 2 are both even, so every d/aux read is
/// u16-aligned) instead of the u8-recombine. 5 u16 loads/group vs 10 u8 →
/// tests whether the new DType::U16 lets the zero-copy view approach the pool
/// gemv's ~100 GB/s (the u8 path was ~3.5). If so, pool-elimination is viable.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gemv_rows_view_u16_iq2xxs<T>(
    x: Tensor<T>,
    view_u16: Tensor<u16>,
    grid: Tensor<u8>,
    signs: Tensor<u8>,
    expert_ids: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] m_total: u32,
    #[constexpr] tensor_byte_off: u32,
    #[constexpr] expert_byte_stride: u32,
) {
    let m = tgid_x;
    let row = tgid_y;
    let lane = tid;
    let blocks_per_row = k_in / 256u32;
    let expert = load(expert_ids[row]);
    let expert_byte_base = tensor_byte_off + expert * expert_byte_stride;
    let row_block0 = m * blocks_per_row;
    let x_base = row * k_in;
    let total_groups = blocks_per_row * 8u32;
    let mut acc = 0.0f32;
    for grp in range(lane, total_groups, 32u32) {
        let b = grp / 8u32;
        let group = grp & 7u32;
        // u16 index of the block start (blk_byte is even → /2 exact).
        let blk_u16 = (expert_byte_base + (row_block0 + b) * 66u32) / 2u32;
        let d_bits = load(view_u16[blk_u16]).cast::<u32>();
        let d_sign = select((d_bits & 0x8000u32) != 0u32, 0.0f32 - 1.0f32, 1.0f32);
        let d_exp = (d_bits >> 10u32) & 0x1fu32;
        let d_mant = (d_bits & 0x3ffu32).cast::<i32>().cast::<f32>();
        let d_norm =
            (1.0f32 + d_mant / 1024.0f32) * exp2(d_exp.cast::<i32>().cast::<f32>() - 15.0f32);
        let d_sub = d_mant * exp2(0.0f32 - 24.0f32);
        let dval = d_sign * select(d_exp == 0u32, d_sub, d_norm);
        // group g: aux_idx + aux_sgn (2 u32 = 4 u16) at u16 offset blk+1+g*4.
        let q = blk_u16 + 1u32 + group * 4u32;
        let aux_idx =
            load(view_u16[q]).cast::<u32>() | (load(view_u16[q + 1u32]).cast::<u32>() << 16u32);
        let aux_sgn = load(view_u16[q + 2u32]).cast::<u32>()
            | (load(view_u16[q + 3u32]).cast::<u32>() << 16u32);
        let scale_4bit = aux_sgn >> 28u32;
        let db = dval * ((scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25);
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

    use super::ffai_moe_gemv_rows_view_u16_iq2xxs;

    #[bench(name = "ffai/moe/gemv_rows_view_u16_iq2xxs", dtypes = [f32, f16, bf16])]
    fn bench_gemv_rows_view_u16_iq2xxs(dt: DType) -> BenchSetup {
        let m_total = 256usize;
        let n_experts = 8usize;
        let m_out = 2048usize;
        let k_in = 4096usize;
        let nblk = m_out * (k_in / 256);
        let view_bytes = n_experts * nblk * 66;
        BenchSetup::new(ffai_moe_gemv_rows_view_u16_iq2xxs::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("view_u16", view_bytes / 2, DType::U16))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::random("signs", 128, DType::U8))
            .buffer(BenchBuffer::zeros("expert_ids", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("m_total", m_total as u32)
            .constexpr("tensor_byte_off", 0u32)
            .constexpr("expert_byte_stride", (nblk * 66) as u32)
            .grid_3d(m_out as u32, m_total as u32, 1, [32, 1, 1])
            .bytes_moved((m_total * nblk * 64 + m_total * k_in * dt.size_bytes()) as u64)
    }
}

pub mod kernel_benches_old {
    use metaltile::{bench, test::*};

    use super::ffai_moe_gemv_rows_view_iq2xxs;

    #[bench(name = "ffai/moe/gemv_rows_view_iq2xxs", dtypes = [f32, f16, bf16])]
    fn bench_gemv_rows_view_iq2xxs(dt: DType) -> BenchSetup {
        let m_total = 256usize;
        let n_experts = 8usize;
        let m_out = 2048usize;
        let k_in = 4096usize;
        let nblk = m_out * (k_in / 256);
        let view_bytes = n_experts * nblk * 66;
        BenchSetup::new(ffai_moe_gemv_rows_view_iq2xxs::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("view_u8", view_bytes, DType::U8))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::random("signs", 128, DType::U8))
            .buffer(BenchBuffer::zeros("expert_ids", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("m_total", m_total as u32)
            .constexpr("tensor_byte_off", 0u32)
            .constexpr("expert_byte_stride", (nblk * 66) as u32)
            .grid_3d(m_out as u32, m_total as u32, 1, [32, 1, 1])
            .bytes_moved((view_bytes + m_total * k_in * dt.size_bytes()) as u64)
    }
}

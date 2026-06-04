//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Q8_0 multi-row GEMM — `out[r, :] = dequant(weight) · input[r, :]` for a
//! block of `n_rows` rows in one dispatch. The prefill counterpart of
//! `ffai_gemv_q8`: tiles the output into 32×32 blocks and stages a
//! `[32,16]` dequantized-weight tile + `[32,16]` input tile in threadgroup
//! memory, so each Q8 weight is read+dequanted ONCE and reused across all
//! 32 rows of the tile (the amortization that makes prefill fast — a
//! single-row gemv re-streams the weight per token).
//!
//! Weight is the resident Q8 split (`qs` int8 packed 4/u32 + per-32-block
//! `d` scale), laid out as `[out_dim, in_dim]` (row-major over values).
//! Mirrors `ffai_gemm`'s geometry exactly so the dispatch wrapper is
//! identical apart from the two extra weight buffers.
//!
//! ## DISPATCH INVARIANTS (same as ffai_gemm)
//! - TPG = 1024 (32×32). Grid: (out_dim/32) × (n_rows/32) threadgroups.
//! - `in_dim % 16 == 0` (K-tile) AND `in_dim % 32 == 0` (Q8 block).
//! - Row/col edges handled in-kernel (clamp loads to 0, skip OOB stores).

use metaltile::kernel;

#[kernel]
pub fn ffai_gemm_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] n_rows: u32,
) {
    let tid = simd_id * 32u32 + simd_lane;
    let lr = tid / 32u32; // output row within tile (0..31)
    let lo = tid % 32u32; // output col within tile (0..31)
    threadgroup_alloc("gq8_w", 512);
    threadgroup_alloc("gq8_x", 512);
    let mut acc = 0.0f32;
    for k0 in range(0u32, in_dim, 16u32) {
        // threads 0..511 dequant one weight element each into the W tile.
        if tid < 512u32 {
            let s = tid;
            let w_col = tgid_x * 32u32 + s / 16u32;
            let w_valid = w_col < out_dim;
            let w_col_safe = select(w_valid, w_col, 0u32);
            // value index in the [out_dim, in_dim] weight, then Q8 unpack.
            let vidx = w_col_safe * in_dim + k0 + s % 16u32;
            let block = vidx / 32u32;
            let lane = vidx & 31u32;
            let word = load(qs[block * 8u32 + lane / 4u32]);
            let by = (word >> ((lane & 3u32) * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            let w_raw = load(d_f32[block]) * qf;
            threadgroup_store("gq8_w", s, select(w_valid, w_raw, 0.0f32));
        }
        // threads 512..1023 load one input element each into the X tile.
        if tid >= 512u32 {
            let s = tid - 512u32;
            let x_row = tgid_y * 32u32 + s / 16u32;
            let x_valid = x_row < n_rows;
            let x_row_safe = select(x_valid, x_row, 0u32);
            let x_raw = load(input[x_row_safe * in_dim + k0 + s % 16u32]).cast::<f32>();
            threadgroup_store("gq8_x", s, select(x_valid, x_raw, 0.0f32));
        }
        threadgroup_barrier();
        for k in range(0u32, 16u32, 1u32) {
            let w = threadgroup_load("gq8_w", lo * 16u32 + k);
            let x = threadgroup_load("gq8_x", lr * 16u32 + k);
            acc = acc + w * x;
        }
        threadgroup_barrier();
    }
    let r = tgid_y * 32u32 + lr;
    let o = tgid_x * 32u32 + lo;
    if r < n_rows {
        if o < out_dim {
            store(out[r * out_dim + o], acc.cast::<T>());
        }
    }
}

/// GROUPED Q8 GEMM — the amortized fix for prefill O-LoRA-A. Same tiled
/// 32×32 / threadgroup-staged structure as `ffai_gemm_q8` (weight read+
/// dequanted ONCE, reused across 32 rows), but the input is GROUPED: output
/// column `o` belongs to group `g = o / rows_per_group`, and group g reads
/// input columns `[g*in_dim, (g+1)*in_dim)` of a row that is `n_groups*in_dim`
/// wide. Replaces `ffai_grouped_gemv_q8_rows` (per-token gemv, ~47 ms/layer,
/// the #1 prefill attention hotspot) with O-LoRA-B-class amortized speed.
/// A 32-wide output tile lies entirely within one group (rows_per_group %
/// 32 == 0), so `g` is uniform per threadgroup → no per-thread divergence.
/// weight `[out_dim, in_dim]`; input `[n_rows, n_groups*in_dim]`;
/// out `[n_rows, out_dim]`. Grid + TPG identical to ffai_gemm_q8.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_grouped_gemm_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] n_rows: u32,
    #[constexpr] n_groups: u32,
    #[constexpr] rows_per_group: u32,
) {
    let tid = simd_id * 32u32 + simd_lane;
    let lr = tid / 32u32;
    let lo = tid % 32u32;
    // Group is uniform per threadgroup (tile cols all within one group).
    let g = (tgid_x * 32u32) / rows_per_group;
    let row_in_stride = n_groups * in_dim;
    let in_col_off = g * in_dim;
    threadgroup_alloc("gq8_w", 512);
    threadgroup_alloc("gq8_x", 512);
    let mut acc = 0.0f32;
    for k0 in range(0u32, in_dim, 16u32) {
        if tid < 512u32 {
            let s = tid;
            let w_col = tgid_x * 32u32 + s / 16u32;
            let w_valid = w_col < out_dim;
            let w_col_safe = select(w_valid, w_col, 0u32);
            let vidx = w_col_safe * in_dim + k0 + s % 16u32;
            let block = vidx / 32u32;
            let lane = vidx & 31u32;
            let word = load(qs[block * 8u32 + lane / 4u32]);
            let by = (word >> ((lane & 3u32) * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            let w_raw = load(d_f32[block]) * qf;
            threadgroup_store("gq8_w", s, select(w_valid, w_raw, 0.0f32));
        }
        if tid >= 512u32 {
            let s = tid - 512u32;
            let x_row = tgid_y * 32u32 + s / 16u32;
            let x_valid = x_row < n_rows;
            let x_row_safe = select(x_valid, x_row, 0u32);
            // GROUPED input read: row is n_groups*in_dim wide; this group's
            // slice starts at g*in_dim.
            let x_raw =
                load(input[x_row_safe * row_in_stride + in_col_off + k0 + s % 16u32]).cast::<f32>();
            threadgroup_store("gq8_x", s, select(x_valid, x_raw, 0.0f32));
        }
        threadgroup_barrier();
        for k in range(0u32, 16u32, 1u32) {
            let w = threadgroup_load("gq8_w", lo * 16u32 + k);
            let x = threadgroup_load("gq8_x", lr * 16u32 + k);
            acc = acc + w * x;
        }
        threadgroup_barrier();
    }
    let r = tgid_y * 32u32 + lr;
    let o = tgid_x * 32u32 + lo;
    if r < n_rows {
        if o < out_dim {
            store(out[r * out_dim + o], acc.cast::<T>());
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_gemm_q8, ffai_grouped_gemm_q8};

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemm_q8(dt: DType) -> BenchSetup {
        let in_dim = 4096usize;
        let out_dim = 2048usize;
        let n_rows = 256usize;
        let n_blocks = out_dim * in_dim / 32;
        BenchSetup::new(ffai_gemm_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", n_blocks * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("input", n_rows * in_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_rows * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("n_rows", n_rows as u32)
            .grid_3d((out_dim as u32).div_ceil(32), (n_rows as u32).div_ceil(32), 1, [1024, 1, 1])
            .bytes_moved((n_blocks * 36 + n_rows * in_dim * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemm_q8(dt: DType) -> BenchSetup {
        // O-LoRA-A shape: 8 groups, per-group in_dim=4096 (nHeads*headDim/8),
        // out_dim=8192 (8*1024), n_rows=256 tokens.
        let in_dim = 4096usize;
        let out_dim = 8192usize;
        let n_rows = 256usize;
        let n_groups = 8usize;
        let rows_per_group = out_dim / n_groups; // 1024
        let n_blocks = out_dim * in_dim / 32;
        BenchSetup::new(ffai_grouped_gemm_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", n_blocks * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("input", n_rows * n_groups * in_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_rows * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("n_rows", n_rows as u32)
            .constexpr("n_groups", n_groups as u32)
            .constexpr("rows_per_group", rows_per_group as u32)
            .grid_3d((out_dim as u32).div_ceil(32), (n_rows as u32).div_ceil(32), 1, [1024, 1, 1])
            .bytes_moved((n_blocks * 36 + n_rows * n_groups * in_dim * dt.size_bytes()) as u64)
    }
}

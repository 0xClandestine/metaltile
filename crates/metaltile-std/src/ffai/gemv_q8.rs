//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Q8_0 inline-dequant GEMV — `out[r] = Σ_k dequant(W[r,k]) · x[k]`,
//! reading the Q8_0 weight straight from its split buffers so the dense
//! attention/shared-expert projections stay 1 byte/weight resident
//! instead of being pre-expanded to f16 (2 bytes). The DSv4 attention
//! block is bandwidth-bound on these projections (q_b / output_a /
//! output_b are Q8_0 on disk, ~100M weights/layer); halving their bytes
//! roughly halves the attn GPU time.
//!
//! ## Q8_0 block (32 values)
//!   d (f16 scale) + 32 int8 quants;  value[i] = d · q_i8[i]
//!
//! ## Split inputs (loader produces these once, resident)
//!   qs   [m_out * (k_in/32) * 8]  u32  — 32 int8/block packed as 8 LE u32
//!   d    [m_out * (k_in/32)]      f32  — per-block scale (fp16→f32)
//!   x    [k_in]                   T
//!   out  [m_out]                  T
//!
//! Dispatch (Reduction): grid (threadgroups) = [m_out, 1, 1], tg=[32,1,1].

use metaltile::kernel;

#[kernel]
pub fn ffai_gemv_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]);
        let x_base = b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                // sign-extend the byte to int8 range in the float domain
                // (avoids ambiguous integer `select` in MSL).
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                let val = d * qf;
                acc = acc + val * load(x[x_base + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Grouped Q8_0 gemv — `out[r] = Σ_k dequant(W[r,k]) · x[(r/rows_per_group)*k_in + k]`.
/// Each contiguous block of `rows_per_group` output rows reads its own
/// `k_in`-slice of `x`. Fuses the DSv4 grouped O-LoRA (8 groups × a
/// [1024,4096] Q8 slice, each on a different 4096-slice of the attention
/// output) into a SINGLE dispatch instead of 8.
#[kernel]
pub fn ffai_grouped_gemv_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]);
        let x_blk = x_base + b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                acc = acc + d * qf * load(x[x_blk + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// BATCHED grouped Q8_0 gemv — ffai_grouped_gemv_q8 over `n_tokens` rows in
/// ONE dispatch (grid z/y = token). Prefill O-LoRA looped the per-token
/// grouped gemv N times; this folds it. x is [n_tokens, n_groups*k_in],
/// out is [n_tokens, m_out]; n_groups = m_out/rows_per_group.
/// Grid (Reduction): [m_out, n_tokens, 1], tg=[32,1,1].
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_grouped_gemv_q8_rows<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let token = tgid_y;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let n_groups = m_out / rows_per_group;
    let x_base = token * n_groups * k_in + (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]);
        let x_blk = x_base + b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                acc = acc + d * qf * load(x[x_blk + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[token * m_out + row], total.cast::<T>());
    }
}

/// TOKEN-TILED grouped Q8 gemv — the amortized fix for the prefill O-LoRA-A
/// hotspot. `ffai_grouped_gemv_q8_rows` re-reads each weight row from DRAM
/// once PER TOKEN (no amortization); at N=512 that's the single biggest
/// op in the attention block (~47 ms/layer). Here each threadgroup owns one
/// output row and a TILE of `tokens_per_tile` tokens: the Q8 weight block
/// (d + 8 packed = 32 int8) is loaded ONCE and applied to all T tokens, so
/// the weight DRAM traffic drops T-fold. T accumulators in a register stack.
/// grid (threadgroups) = [m_out, ceil(n_tokens/T), 1], threadgroup [32,1,1].
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_grouped_gemv_q8_rows_tiled<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] n_tokens: u32,
) {
    // Tile size is a compile-time LITERAL (stack_alloc can't size on a
    // constexpr param — codegen emits the array decl before the constant
    // is bound). T=8 tokens/tile → 8-fold weight-DRAM amortization.
    let tokens_per_tile = 8u32;
    let row = tgid_x;
    let ttile = tgid_y;
    let lane = tid;
    let tok0 = ttile * tokens_per_tile;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let n_groups = m_out / rows_per_group;
    let group_off = (row / rows_per_group) * k_in;

    stack_alloc("acc", 8, "f32");
    for t in range(0u32, tokens_per_tile, 1u32) {
        stack_store("acc", t, 0.0f32);
    }
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]);
        let blk = b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]); // 4 int8 weights, read ONCE
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                let wv = d * qf; // weight value, reused across the T tokens
                let kk = blk + w * 4u32 + i;
                for t in range(0u32, tokens_per_tile, 1u32) {
                    let tok = tok0 + t;
                    if tok < n_tokens {
                        let xb = tok * n_groups * k_in + group_off + kk;
                        let prev = stack_load("acc", t);
                        stack_store("acc", t, prev + wv * load(x[xb]).cast::<f32>());
                    }
                }
            }
        }
    }
    for t in range(0u32, tokens_per_tile, 1u32) {
        let tok = tok0 + t;
        let total = simd_sum(stack_load("acc", t));
        if (lane == 0u32) & (tok < n_tokens) {
            store(out[tok * m_out + row], total.cast::<T>());
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        ffai_gemv_q8,
        ffai_grouped_gemv_q8,
        ffai_grouped_gemv_q8_rows,
        ffai_grouped_gemv_q8_rows_tiled,
    };

    #[bench(name = "ffai/gemv/q8", dtypes = [f32, f16, bf16])]
    fn bench_gemv_q8(dt: DType) -> BenchSetup {
        let m_out = 4096usize;
        let k_in = 8192usize;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_gemv_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + k_in * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/gemv/grouped_q8", dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_grouped_gemv_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", 8 * k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + 8 * k_in * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/gemv/grouped_q8_rows", dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8_rows(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let n_tokens = 256usize;
        let n_groups = m_out / 1024;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_grouped_gemv_q8_rows::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", n_tokens * n_groups * k_in, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .grid_3d(m_out as u32, n_tokens as u32, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + n_tokens * n_groups * k_in * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/gemv/grouped_q8_rows_tiled", dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8_rows_tiled(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let n_tokens = 256usize;
        let tokens_per_tile = 8usize;
        let n_groups = m_out / 1024;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_grouped_gemv_q8_rows_tiled::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", n_tokens * n_groups * k_in, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .constexpr("n_tokens", n_tokens as u32)
            .grid_3d(m_out as u32, (n_tokens as u32).div_ceil(tokens_per_tile as u32), 1, [
                32, 1, 1,
            ])
            .bytes_moved((m_out * bpr * 36 + n_tokens * n_groups * k_in * dt.size_bytes()) as u64)
    }
}

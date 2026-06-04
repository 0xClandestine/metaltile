//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DeepSeek V4 router top-K selection — GPU-side, so the per-layer MoE
//! routing never round-trips to the CPU.
//!
//! DSv4 selects the top-K experts by the **biased** sqrtsoftplus score
//! and weights them by the **unbiased** score, renormalised to sum to 1.
//! This kernel emits the sum-to-1 weights only. `routed_scaling_factor`
//! is applied by the CALLER (`gpuRoutedFfnTail`) AFTER this renorm —
//! `final_weight_k = scaling * unbiased_k / Σ unbiased` — so it does NOT
//! cancel. Keeping it out of the kernel lets the kernel take just the two
//! score vectors:
//!
//! ```text
//!   chosen      = argmax-k(score_biased)            (indices)
//!   weight_i    = score_unbiased[chosen_i] / Σ_j∈chosen score_unbiased[chosen_j]
//! ```
//!
//! Mirrors the CPU selection in `forwardFfnSubblock` (top-K by biased,
//! weights from unbiased, sum-to-1). Structure adapted from
//! `mt_moe_router_topk` (masked argmax passes, one simdgroup/row).
//!
//! Dispatch: `[n_rows, 1, 1] × [32, 1, 1]` (pins tpg=32; one simdgroup
//! per token row — decode is n_rows=1).

use metaltile::kernel;

#[kernel]
pub fn mt_dsv4_router_topk<T>(
    score_biased: Tensor<T>,
    score_unbiased: Tensor<T>,
    mut indices_out: Tensor<u32>,
    mut weights_out: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] k: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;
    threadgroup_alloc("tg_chosen_idx", 32u32);

    // ── k masked-argmax passes over score_biased ─────────────────────
    for it in range(0u32, k, 1u32) {
        let mut best_val = neg_infinity();
        let mut best_idx = 0u32;
        let n_per_lane = (n_experts + 31u32) / 32u32;
        for r in range(0u32, n_per_lane, 1u32) {
            let j = r * 32u32 + lane;
            if j < n_experts {
                let v = load(score_biased[row_base + j]).cast::<f32>();
                let mut chosen_mask = 0u32;
                for p in range(0u32, it, 1u32) {
                    let cp = threadgroup_load("tg_chosen_idx", p);
                    chosen_mask = chosen_mask | select(j == cp, 1u32, 0u32);
                }
                let candidate = select(chosen_mask > 0u32, neg_infinity(), v);
                let better = candidate > best_val;
                best_val = select(better, candidate, best_val);
                best_idx = select(better, j, best_idx);
            }
        }
        let global_best_val = simd_max(best_val);
        let i_have = best_val == global_best_val;
        let my_idx_or_max = select(i_have, best_idx, 4294967295u32);
        let global_best_idx = simd_min(my_idx_or_max);
        if lane == 0u32 {
            threadgroup_store("tg_chosen_idx", it, global_best_idx);
        }
        simdgroup_barrier_mem_none();
    }

    // ── weights = unbiased[chosen] / Σ unbiased[chosen] ──────────────
    // tg_chosen_idx is f32-typed scratch; cast to u32 for the subscript.
    let my_idx_f = select(lane < k, threadgroup_load("tg_chosen_idx", lane), 0.0f32);
    let my_idx = my_idx_f.cast::<u32>();
    let my_unbiased =
        select(lane < k, load(score_unbiased[row_base + my_idx]).cast::<f32>(), 0.0f32);
    let sum_chosen = simd_sum(my_unbiased);
    let weight = my_unbiased / sum_chosen;
    if lane < k {
        let out_base = row * k + lane;
        store(indices_out[out_base], my_idx);
        store(weights_out[out_base], weight.cast::<T>());
    }
}

/// out[i] = table[idx[i]] — gather a u32 table by u32 indices. Used to
/// remap raw routed expert ids into resident-pool packed slot ids on the
/// GPU (so the gather dispatch's `expert_ids` never touches the CPU).
#[kernel]
pub fn mt_remap_u32(
    table: Tensor<u32>,
    idx: Tensor<u32>,
    mut out: Tensor<u32>,
    #[constexpr] n: u32,
) {
    let i = tid;
    if i < n {
        let e = load(idx[i]);
        store(out[i], load(table[e]));
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_dsv4_router_topk, mt_remap_u32};

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_dsv4_router_topk(dt: DType) -> BenchSetup {
        let n_experts = 256usize;
        let k = 6usize;
        BenchSetup::new(mt_dsv4_router_topk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("score_biased", n_experts, dt))
            .buffer(BenchBuffer::random("score_unbiased", n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .grid_3d(1, 1, 1, [32, 1, 1])
            .bytes_moved((2 * n_experts * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32])]
    fn bench_remap_u32(_dt: DType) -> BenchSetup {
        let n = 6usize;
        BenchSetup::new(mt_remap_u32::kernel_ir())
            .buffer(BenchBuffer::zeros("table", 256, DType::U32))
            .buffer(BenchBuffer::zeros("idx", n, DType::U32))
            .buffer(BenchBuffer::zeros("out", n, DType::U32).output())
            .constexpr("n", n as u32)
            .grid_1d(n, 32)
            .bytes_moved((n * 4 * 2) as u64)
    }
}

//! MoE orchestration kernels — router top-k, permute, unpermute,
//! grouped BGEMM dispatch.
//!
//! Targets Qwen3.6-35B-A3B and Qwen3-Coder-30B-A3B end-to-end serving.
//! The per-expert quantized matmul cell is already served by
//! `mt_qmm_*` (mma / mma_m16 / bm4 / bm2 / v2) — this module adds the
//! routing kernels that go around each expert call.
//!
//! ## Pipeline shape
//!
//! ```text
//!   activations [B*T, hidden]
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │ mt_moe_router_topk│   logits  → [B*T, k] (indices + weights)
//!   └──────────────────┘
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │   mt_moe_permute │   [B*T, hidden]  → [k*B*T, hidden] expert-sorted
//!   └──────────────────┘
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │ per-expert qmm   │   N × mt_qmm_for() calls — already shipped
//!   └──────────────────┘
//!         │
//!         ▼
//!   ┌──────────────────┐
//!   │ mt_moe_unpermute │   [k*B*T, hidden] + weights  → [B*T, hidden]
//!   └──────────────────┘
//! ```

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

// ── mt_moe_router_topk ───────────────────────────────────────────────────
//
// Per-token select top-k experts from `router_logits`, plus softmax
// weights over the chosen k.
//
// Inputs:
//   router_logits — [B*T, n_experts]  (any float dtype, computed in f32)
//   indices_out   — [B*T, k]          (u32)
//   weights_out   — [B*T, k]          (same dtype as router_logits, softmax weights)
//
// Constexpr:
//   n_experts   — typical Qwen3.6-A3B: 128.  Must fit one simdgroup
//                 (≤ 32×32 = 1024) — every reasonable MoE topology.
//   k           — typical 6-8 for production MoE.  Hard cap k ≤ 32.
//
// Geometry:
//   tpg=32  (one simdgroup per token row)
//   grid = [B*T, 1, 1]  (Reduction mode)
//
// Algorithm — k iterations of simd-parallel argmax with mask of
// previously-chosen indices stored in TG memory.  After k passes,
// softmax over the chosen k values in-place on lane 0..k-1.
//
// Bench spec uses BenchDispatch::Generic + shapes: &[] so `tile bench`
// skips it; correctness lives in unit tests + downstream MoE
// integration. Same convention as other ffai/ kernels (gather, sampling).
#[kernel]
pub fn mt_moe_router_topk<T>(
    router_logits: Tensor<T>,
    mut indices_out: Tensor<u32>,
    mut weights_out: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] k: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;

    // TG scratch: chosen indices + values from each of the k argmax passes.
    // 32 slots covers any reasonable k (typical 6-8).
    threadgroup_alloc("tg_chosen_idx", 32u32);
    threadgroup_alloc("tg_chosen_val", 32u32);

    // ── k argmax passes with chosen-mask ─────────────────────────────
    for it in range(0u32, k, 1u32) {
        // Per-lane local argmax over its slice of n_experts.
        // Each lane covers ceil(n_experts/32) experts.
        let mut best_val = neg_infinity();
        let mut best_idx = 0u32;
        let n_per_lane = (n_experts + 31u32) / 32u32;
        for r in range(0u32, n_per_lane, 1u32) {
            let j = r * 32u32 + lane;
            if j < n_experts {
                let v = load(router_logits[row_base + j]).cast::<f32>();
                // Mask: was j picked in a previous iter?
                // Scan tg_chosen_idx[0..it] — k ≤ 8 typically so this
                // is fast even without early exit.
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

        // Cross-lane reduce.  simd_max gives the global best value;
        // ties broken to smaller idx via simd_min on (idx | sentinel).
        let global_best_val = simd_max(best_val);
        let i_have = best_val == global_best_val;
        let my_idx_or_max = select(i_have, best_idx, 4294967295u32); // u32::MAX
        let global_best_idx = simd_min(my_idx_or_max);

        // Lane 0 writes the iter's chosen slot.
        if lane == 0u32 {
            threadgroup_store("tg_chosen_idx", it, global_best_idx);
            threadgroup_store("tg_chosen_val", it, global_best_val);
        }
        simdgroup_barrier_mem_none();
    }

    // ── Softmax over the chosen k values ────────────────────────────
    // Lanes 0..k-1 each hold one chosen value.  Stream + simd reduce.
    let my_val = select(lane < k, threadgroup_load("tg_chosen_val", lane), neg_infinity());
    let row_max = simd_max(my_val);
    let exp_val = exp(my_val - row_max);
    let masked_exp = select(lane < k, exp_val, 0.0f32);
    let sum_exp = simd_sum(masked_exp);
    let weight = masked_exp / sum_exp;

    // ── Write outputs ───────────────────────────────────────────────
    if lane < k {
        let out_base = row * k + lane;
        store(indices_out[out_base], threadgroup_load("tg_chosen_idx", lane));
        store(weights_out[out_base], weight.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "router_topk",
        kernel_name: "mt_moe_router_topk",
        kernel_ir: mt_moe_router_topk::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

// ── mt_moe_unpermute ─────────────────────────────────────────────────────
//
// Combine k expert outputs back into the original token order with
// top-k softmax weights.
//
// Inputs:
//   expert_outputs  — [k*B*T, hidden]   per-expert dense outputs at the
//                                       expert-sorted positions
//   inv_perm        — [B*T, k]          where (token i, slot j) was placed
//                                       in expert_outputs (computed by
//                                       caller's sort step)
//   top_k_weights   — [B*T, k]          softmax weights from
//                                       mt_moe_router_topk
//   out             — [B*T, hidden]     weighted sum across k experts
//
// Constexpr:
//   hidden — model hidden dim (e.g. 2048 for Qwen3-MoE)
//   k      — top-k expert count (e.g. 8)
//
// Geometry:
//   tpg=128  (split hidden across 128 lanes via 4-wide vectorize)
//   grid=[B*T, 1, 1]
//
// Per-token cost: read k * hidden / 128 = (k * hidden) / 128 expert
// values + k weights, do k FMAs per output column, one store per
// column. At hidden=2048, k=8 → ~1k FMAs per token. Bandwidth-bound,
// not ALU-bound.
#[kernel]
pub fn mt_moe_unpermute<T>(
    expert_outputs: Tensor<T>,
    inv_perm: Tensor<u32>,
    top_k_weights: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] k: u32,
) {
    let token = tgid_x;
    let lane = tid;
    let row_base_inv = token * k;
    let row_base_w = token * k;
    let row_base_out = token * hidden;

    let n_per_lane = (hidden + 127u32) / 128u32;
    for r in range(0u32, n_per_lane, 1u32) {
        let h = r * 128u32 + lane;
        if h < hidden {
            let mut acc = 0.0f32;
            for j in range(0u32, k, 1u32) {
                let pos = load(inv_perm[row_base_inv + j]);
                let v = load(expert_outputs[pos * hidden + h]).cast::<f32>();
                let w = load(top_k_weights[row_base_w + j]).cast::<f32>();
                acc = acc + w * v;
            }
            store(out[row_base_out + h], acc.cast::<T>());
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "unpermute",
        kernel_name: "mt_moe_unpermute",
        kernel_ir: mt_moe_unpermute::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

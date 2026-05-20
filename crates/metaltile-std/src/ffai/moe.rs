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
    // 1 = Qwen3-MoE style (softmax over chosen-k, sum-to-1 — `norm_topk_prob=True`)
    // 0 = Qwen3-Next style (softmax over ALL n_experts, return chosen probs
    //     un-renormalized — `norm_topk_prob=False`)
    // Mathematically equivalent at mode 1: softmax-over-chosen-k is the
    // same as (softmax-over-all → renormalize-over-chosen). Mode 0
    // returns probs that sum to < 1 across the chosen k, matching MLX's
    // qwen3_next.py:334-341.
    //
    // INVARIANT: this kernel pins tpg=32 (one simdgroup per token row).
    // The `simdgroup_barrier_mem_none()` below is correct only at tpg=32.
    // Caller must dispatch with `[n_rows, 1, 1] × [32, 1, 1]`.
    #[constexpr] norm_topk_prob: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;

    // TG scratch: chosen indices + values from each of the k argmax passes.
    // 32 slots covers any reasonable k (typical 6-8). Kernel assumes
    // k ≤ 32 — caller MUST enforce this in the host-side dispatcher
    // (no GPU-side check, would silently scribble into adjacent TG mem).
    threadgroup_alloc("tg_chosen_idx", 32u32);
    threadgroup_alloc("tg_chosen_val", 32u32);
    // Cache the all-experts-softmax sum for Qwen3-Next mode (mode 0).
    // 1 slot, written by lane 0 in the prepass.
    threadgroup_alloc("tg_full_sum", 1u32);
    threadgroup_alloc("tg_full_max", 1u32);

    // ── Pre-pass: compute softmax denominator over ALL n_experts ─────
    // Needed only for norm_topk_prob=0 (Qwen3-Next), but the cost is
    // trivial (one simd_max + simd_sum) and emitting it unconditionally
    // keeps the codegen tight (the codegen DCE will drop the dead path
    // when the constexpr branch is unreachable).
    let mut local_max_all = neg_infinity();
    let n_per_lane_pre = (n_experts + 31u32) / 32u32;
    for r in range(0u32, n_per_lane_pre, 1u32) {
        let j = r * 32u32 + lane;
        if j < n_experts {
            let v = load(router_logits[row_base + j]).cast::<f32>();
            let better = v > local_max_all;
            local_max_all = select(better, v, local_max_all);
        }
    }
    let row_max_all = simd_max(local_max_all);
    let mut local_sum_all = 0.0f32;
    for r in range(0u32, n_per_lane_pre, 1u32) {
        let j = r * 32u32 + lane;
        if j < n_experts {
            let v = load(router_logits[row_base + j]).cast::<f32>();
            local_sum_all = local_sum_all + exp(v - row_max_all);
        }
    }
    let row_sum_all = simd_sum(local_sum_all);
    if lane == 0u32 {
        threadgroup_store("tg_full_max", 0u32, row_max_all);
        threadgroup_store("tg_full_sum", 0u32, row_sum_all);
    }
    simdgroup_barrier_mem_none();

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

    // ── Softmax / weight emit per `norm_topk_prob` ──────────────────
    // Mode 1 (Qwen3-MoE, default): softmax over chosen-k (sum-to-1).
    //   numerator   = exp(z_i - max_chosen);  divisor = Σ_j∈chosen
    //   == exp(z_i - max_all) · const / Σ_j∈chosen exp(z_j - max_all) · const
    //   so we can use the SAME numerator as mode 0 (exp(z - max_all)) and
    //   just swap the divisor.  Avoids needing a Rust `if`-expression
    //   which the DSL doesn't unify across arms.
    // Mode 0 (Qwen3-Next): un-normalized chosen probs (sum < 1).
    //   weight_i = exp(z_i - max_all) / Σ_j∈all exp(z_j - max_all)
    let my_val = select(lane < k, threadgroup_load("tg_chosen_val", lane), neg_infinity());
    let row_max_full = threadgroup_load("tg_full_max", 0u32);
    let row_sum_full = threadgroup_load("tg_full_sum", 0u32);
    let exp_val = exp(my_val - row_max_full);
    let masked_exp = select(lane < k, exp_val, 0.0f32);
    let sum_chosen = simd_sum(masked_exp);
    // Pick divisor: chosen-k sum for renormalized (mode 1) or all-experts
    // sum for raw probs (mode 0). select() forces both to be live; codegen
    // const-folds when `norm_topk_prob` bakes in.
    let divisor = select(norm_topk_prob == 1u32, sum_chosen, row_sum_full);
    let weight = masked_exp / divisor;

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

// ── mt_moe_permute ───────────────────────────────────────────────────────
//
// Gather tokens into per-expert contiguous buffers given a pre-computed
// sort permutation. The expensive sort step (argsort over top-k expert
// indices) is done by the caller — typically CPU-side via Rust sort,
// or via a future sort kernel. This kernel is just the data-movement
// half: each output position copies the row indicated by sort_token_idx.
//
// Inputs:
//   tokens          — [B*T, hidden]      activations to gather
//   sort_token_idx  — [k * B*T]          for each permuted position p,
//                                        which original token row sourced it.
//                                        Caller computes via argsort over
//                                        top-k indices flattened to
//                                        (token * k + slot) → token (this is
//                                        the "permute" direction; the inverse
//                                        is `inv_perm` consumed by unpermute).
//   permuted        — [k * B*T, hidden]  expert-sorted output. Each k*B*T
//                                        row corresponds to one (expert, token)
//                                        pair; consecutive rows with the same
//                                        expert form that expert's input slab.
//
// Constexpr:
//   hidden — model hidden dim
//
// Geometry:
//   tpg=128  (split hidden across 128 lanes, ceil(hidden/128) iters)
//   grid=[k*B*T, 1, 1]
//
// Per-permuted-row cost: hidden / 128 = 16 loads + 16 stores (at
// hidden=2048). Bandwidth-bound — no FMAs, just a vector copy.
#[kernel]
pub fn mt_moe_permute<T>(
    tokens: Tensor<T>,
    sort_token_idx: Tensor<u32>,
    mut permuted: Tensor<T>,
    #[constexpr] hidden: u32,
) {
    let permuted_pos = tgid_x;
    let lane = tid;
    let token = load(sort_token_idx[permuted_pos]);
    let src_base = token * hidden;
    let dst_base = permuted_pos * hidden;

    let n_per_lane = (hidden + 127u32) / 128u32;
    for r in range(0u32, n_per_lane, 1u32) {
        let h = r * 128u32 + lane;
        if h < hidden {
            let v = load(tokens[src_base + h]);
            store(permuted[dst_base + h], v);
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "permute",
        kernel_name: "mt_moe_permute",
        kernel_ir: mt_moe_permute::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0, // exact copy — no numerical drift
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

// ── mt_moe_gather_qmm_int4 ────────────────────────────────────────────────
//
// Grouped quantized matmul for MoE. Matches MLX's `gatherQuantizedMM`
// (called by SwitchLinear → SwitchGLU → Qwen35SparseMoeBlock):
//
//     y[t, m] = Σ_k x[t, k] · W[E(t), m, k]
//
// where E(t) is the expert assigned to row t. Pre-permuted layout (caller
// passes `sortedIndices=true` upstream): consecutive rows share an expert,
// and `expert_offsets` is a CSR row-offset array — expert `e` owns rows
// `[expert_offsets[e] .. expert_offsets[e+1])`.
//
// One dispatch → all experts × M_out × T rows. Vs MLX's N separate qmm
// dispatches (128 experts × 40 layers × 3 projections = 15360 launches at
// Qwen3.6-35B-A3B), folding into one kernel saves ~1.5 s of host-side
// launch overhead per forward. Decode benefits most (every step pays it);
// prefill saves a smaller fraction since each per-expert matmul is fatter.
//
// Inputs:
//   x               — [T, K_in]                 f32/f16/bf16 (sorted-by-expert)
//   weight_packed   — [E, M_out, K_in/8]        uint32 (int4 packed, 8 per uint)
//   scales          — [E, M_out, K_in/group]    T  per-group quant scale
//   biases          — [E, M_out, K_in/group]    T  per-group quant bias
//   expert_offsets  — [E + 1]                   uint32  CSR row offsets
//   out             — [T, M_out]                T
//
// Constexpr:
//   k_in       — fused input dim (e.g. 2048 for Qwen3.6-35B-A3B hidden)
//   m_out      — per-expert output dim (e.g. 256 for moe_intermediate)
//   n_experts  — typically 128
//   group_size — quant group size (typically 64)
//
// DISPATCH INVARIANTS
//   - **Mode: Reduction.** Uses `simd_sum` for the per-row dot product.
//   - **Grid: `[m_out, T, 1]`** — one TG per (output column m, row t).
//   - **TG: `[32, 1, 1]`** — one simdgroup. Each lane handles
//     `k_in / 32` packed uint32s × 8 weights each = `k_in / 32` weights.
//   - `k_in` must be a multiple of 32 (every Qwen3 / Qwen3.6 satisfies).
//   - `group_size` must divide `k_in`.
//   - int4 only (MLX's MoE quantization default). Wider precision is a
//     follow-up.
//
// Algorithm — scalar foundation; MMA tiling lands in a follow-up commit.
//
//   1. Resolve expert: linear walk over `expert_offsets`. With N_experts
//      ≤ 256 this is cheap (~256 reads on lane 0 + broadcast via TG mem).
//
//   2. Per-lane dot product over `k_in / 32` packed uint32s. Each uint32
//      packs 8 int4 weights → unpack 8 weights, dequant per-group, FMA.
//
//   3. `simd_sum` reduces 32 partial sums → one output value per TG.
//
// Mirrors the per-thread pattern in `dequant_gemv_int4`.
#[kernel]
pub fn mt_moe_gather_qmm_int4<T>(
    x: Tensor<T>,
    weight_packed: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    expert_offsets: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] group_size: u32,
) {
    let m = tgid_x;
    let row = tgid_y;
    let lane = tid;

    // Resolve expert — linear walk on EVERY lane (cheap, ≤ 256 reads from
    // a small uniform buffer) so the result lives in a per-lane u32 register
    // and never round-trips through float-typed TG memory.
    let mut expert = 0u32;
    let mut found = 0u32;
    for ee in range(0u32, n_experts, 1u32) {
        let end = load(expert_offsets[ee + 1u32]);
        let inside_bool = row < end;
        let inside = select(inside_bool, 1u32, 0u32);
        let take = inside * (1u32 - found);
        expert = select(take == 1u32, ee, expert);
        found = select(take == 1u32, 1u32, found);
    }

    // Stride-by-32 over packs: each lane handles packs at positions
    // lane, lane+32, lane+64, ... up to k_in/8. Correct for both small
    // (k_in=32 → 4 packs, only lanes 0..3 work) and large (k_in=2048 →
    // 256 packs, 8 packs/lane) inputs.
    let total_packs = k_in / 8u32;
    let weight_stride_m = total_packs;
    let weight_row_base = expert * m_out * weight_stride_m + m * weight_stride_m;

    let groups_per_row = k_in / group_size;
    let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;

    let x_row_base = row * k_in;

    let mut acc = 0.0f32;
    for pack_idx in range(lane, total_packs, 32u32) {
        let packed = load(weight_packed[weight_row_base + pack_idx]);
        let k_first = pack_idx * 8u32;
        let g = k_first / group_size;
        let scale = load(scales[scale_row_base + g]).cast::<f32>();
        let bias = load(biases[scale_row_base + g]).cast::<f32>();

        let q0 = (packed >> 0u32) & 15u32;
        let q1 = (packed >> 4u32) & 15u32;
        let q2 = (packed >> 8u32) & 15u32;
        let q3 = (packed >> 12u32) & 15u32;
        let q4 = (packed >> 16u32) & 15u32;
        let q5 = (packed >> 20u32) & 15u32;
        let q6 = (packed >> 24u32) & 15u32;
        let q7 = (packed >> 28u32) & 15u32;

        let w0 = q0.cast::<f32>() * scale + bias;
        let w1 = q1.cast::<f32>() * scale + bias;
        let w2 = q2.cast::<f32>() * scale + bias;
        let w3 = q3.cast::<f32>() * scale + bias;
        let w4 = q4.cast::<f32>() * scale + bias;
        let w5 = q5.cast::<f32>() * scale + bias;
        let w6 = q6.cast::<f32>() * scale + bias;
        let w7 = q7.cast::<f32>() * scale + bias;

        let x0 = load(x[x_row_base + k_first + 0u32]).cast::<f32>();
        let x1 = load(x[x_row_base + k_first + 1u32]).cast::<f32>();
        let x2 = load(x[x_row_base + k_first + 2u32]).cast::<f32>();
        let x3 = load(x[x_row_base + k_first + 3u32]).cast::<f32>();
        let x4 = load(x[x_row_base + k_first + 4u32]).cast::<f32>();
        let x5 = load(x[x_row_base + k_first + 5u32]).cast::<f32>();
        let x6 = load(x[x_row_base + k_first + 6u32]).cast::<f32>();
        let x7 = load(x[x_row_base + k_first + 7u32]).cast::<f32>();

        acc = acc + w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
    }

    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row * m_out + m], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "moe",
        subop: "gather_qmm_int4",
        kernel_name: "mt_moe_gather_qmm_int4",
        kernel_ir: mt_moe_gather_qmm_int4::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 5e-2, // int4 quant — wide tolerance vs full-precision oracle
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

//! Gated DeltaNet (GDN) — single-token decode step.
//!
//! GDN is the recurrent linear-attention variant Qwen3.5 / Qwen3.6 / Qwen3.6-MoE
//! use for their `linear_attention` layers (75% of layers in the hybrid
//! architecture). This kernel covers the **single-token decode** form — the
//! `T = 1` analog of MLX-LM's `gated_delta_kernel`. Chunked prefill (T > 1)
//! lives in a follow-up kernel because the inner T loop changes the dispatch
//! geometry.
//!
//! Recurrence per step (matches MLX-LM `_gated_delta_step_ops`):
//!
//!   state_decayed = state * g            // forget-gate decay
//!   kv_mem        = (state_decayed * k).sum(dk)   // [Dv]
//!   delta         = (v - kv_mem) * beta           // [Dv]
//!   state_new     = state_decayed + outer(delta, k)
//!   y             = (state_new * q).sum(dk)       // [Dv]
//!
//! Layouts (matching MLX-LM):
//!
//!   q, k     : [B, Hk, Dk]
//!   v, y     : [B, Hv, Dv]
//!   g, beta  : [B, Hv]
//!   state    : [B, Hv, Dv, Dk]
//!
//! Hk / Hv may differ (GQA-style key-sharing): each Hk-group serves
//! `Hv / Hk` Hv-heads. State is allocated per Hv-head.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** Each threadgroup is one simdgroup (32 threads).
//! - **Grid: `[dv, B * Hv, 1]`, TG: `[32, 1, 1]`.** `tgid_x = dv_idx`,
//!   `tgid_y = n` (the flattened batch×Hv index), `tid = dk_idx` within
//!   the simdgroup (0..32).
//! - **`dk % 32 == 0`.** Each lane owns `n_per_t = dk / 32` contiguous
//!   state elements via `s_idx = n_per_t * dk_idx + i`. TPG = 32 is the
//!   minimum valid value per `docs/developing.md`.
//! - **Hv must be divisible by Hk** (`Hv / Hk` is the number of Hv-heads
//!   per shared (q, k) Hk-group). The kernel computes `hk_idx = hv_idx /
//!   (Hv / Hk)` and reads (q, k) from the shared Hk slot.
//!
//! State accumulator runs in **f32**: the `g * state + outer(delta, k)`
//! recurrence in bf16 drifts after a few dozen decode steps, same
//! reasoning as `ssm_step`. Activations stay in T.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[kernel]
pub fn mt_gated_delta_step<T>(
    q: Tensor<T>,             // [B, Hk, Dk]   flat: (b * Hk + hk_idx) * Dk + dk_offset
    k: Tensor<T>,             // [B, Hk, Dk]   same layout as q
    v: Tensor<T>,             // [B, Hv, Dv]   flat: n * Dv + dv_idx  where n = b*Hv + hv_idx
    g: Tensor<T>,             // [B, Hv]       flat: n
    beta: Tensor<T>,          // [B, Hv]       flat: n
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]  flat: n * Dv * Dk + dv_idx * Dk + s_idx
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]  same as state_in
    mut y: Tensor<T>,         // [B, Hv, Dv]   same as v
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;

    // GQA decomposition: n = b * Hv + hv_idx; hk_idx = hv_idx / (Hv / Hk)
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;

    let n_per_t = dk / 32u32;

    let g_val = load(g[n]).cast::<f32>();
    let beta_val = load(beta[n]).cast::<f32>();
    let v_val = load(v[n * dv + dv_idx]).cast::<f32>();

    let qk_base = (b * hk + hk_idx) * dk;
    let state_base = n * dv * dk + dv_idx * dk;

    // ─── Phase 1: decay + kv_mem reduction ─────────────────────────────
    //
    // Stash the decayed state into TG memory so phase 2 can read it back
    // for the rank-1 update without recomputing. tg_state is sized at
    // the maximum supported Dk (256 — Qwen3.6's `head_dim`); kernels with
    // smaller Dk under-utilise the TG alloc but stay correct because
    // every read uses `s_idx < dk`. The DSL doesn't yet accept constexpr
    // sizes for `threadgroup_alloc` (PR #88's `stack_alloc` is the
    // future-clean way; until then this is the canonical pattern).
    threadgroup_alloc("tg_state", 256u32);

    let mut kv_mem = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = load(state_in[state_base + s_idx]).cast::<f32>() * g_val;
        threadgroup_store("tg_state", s_idx, s_decayed);
        let k_val = load(k[qk_base + s_idx]).cast::<f32>();
        kv_mem = kv_mem + s_decayed * k_val;
    }
    let kv_mem_sum = simd_sum(kv_mem);

    let delta = (v_val - kv_mem_sum) * beta_val;
    threadgroup_barrier();

    // ─── Phase 2: rank-1 update + output projection ────────────────────
    //
    // For each (dv_idx, dk_idx, i): state_new = state_decayed + k * delta
    // Then accumulate state_new * q into `out` via simd_sum.
    let mut out = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = threadgroup_load("tg_state", s_idx);
        let k_val = load(k[qk_base + s_idx]).cast::<f32>();
        let s_new = s_decayed + k_val * delta;
        store(state_out[state_base + s_idx], s_new.cast::<T>());
        let q_val = load(q[qk_base + s_idx]).cast::<f32>();
        out = out + s_new * q_val;
    }
    let out_sum = simd_sum(out);

    // ─── Phase 3: lane 0 writes the result ────────────────────────────
    if dk_idx == 0u32 {
        store(y[n * dv + dv_idx], out_sum.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "gated_delta",
        subop: "step",
        kernel_name: "mt_gated_delta_step",
        kernel_ir: mt_gated_delta_step::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

//! Multi-query bidirectional SDPA — attends `n_query` query rows
//! against a shared K/V cache in a single dispatch, with every query
//! attending the full `[0, base_kv + n_query)` range (no causal mask).
//!
//! This is the variant needed by Vision-Language tower encoders
//! (SigLIP / CLIP / FastViT) — their attention is bidirectional across
//! all image patches, not causal. `sdpa_multi` already supports the
//! same shape under `causal == 0`, but it is hardcoded to head_dim=128.
//! Common vision tower head_dims are 64 (SigLIP-base/large, CLIP-L)
//! and 32 (FastViT-HD). This file adds the matching variants.
//!
//! ## Naming
//!
//! `ffai_sdpa_bidirectional_dN<T>` — N is the constexpr head_dim,
//! T is the element type (f32 / f16 / bf16). The kernel name is the
//! function name (proc-macro generated); each lane owns `N / 32`
//! consecutive Q/K/V elements at `lane * (N/32) + {0..N/32-1}`.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernels — STRICT threadgroup geometry. Wrappers MUST
//! encode these as preconditions; the same machine-freeze hazard as
//! `ffai_sdpa_decode` / `ffai_sdpa_multi`.
//!
//! - **TPG = 1024 threads** (32 simdgroups × 32 lanes). Hard. A TPG
//!   below 32 makes `n_simd = TPG / 32 = 0`, turning the K walk
//!   `range(sg, n_kv, 0)` into an infinite GPU loop — the freeze.
//! - **`head_dim == N`** (the value baked into the kernel name).
//!   Each lane owns N/32 consecutive Q/K/V elements at
//!   `lane * (N/32) + {0..(N/32-1)}`, indexed unconditionally.
//! - **Grid: 1 threadgroup per (query, q_head).** `tgid_x` ranges
//!   `[0, n_q_heads * n_query)`; decoded `query = tgid / n_q_heads`,
//!   `q_head = tgid % n_q_heads`. Wrapper dispatches
//!   `grid = (n_q_heads * n_query * 1024, 1, 1)`, `tg = (1024, 1, 1)`.
//! - **`n_q_heads % heads_per_group == 0`** for integer GQA fan-out.
//! - **`base_kv + n_query <= kv_stride`** — the kernel never walks
//!   past the cache's allocated depth.
//!
//! Q / `out` layout: `[n_query, n_q_heads, head_dim]` row-major.
//! K / V layout:     `[n_kv_heads, kv_stride, head_dim]` row-major.
//! Online softmax runs in fp32 throughout (storage stays in T).

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

// ─── head_dim = 64 (SigLIP base/large, CLIP-L) ─────────────────────

#[kernel]
pub fn ffai_sdpa_bidirectional_d64<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    // No `causal` branch — every query attends the full block.
    let n_kv = base_kv + n_query;

    // Two tg_out slots at head_dim=64; 2 elements per lane.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);

    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 2u32;

    // Pre-scale this lane's 2-element Q pair once; K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;

    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;

    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv_idx = base + d0;
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let partial = q0 * k0 + q1 * k1;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
    }

    // ── Cross-simdgroup reduction: max + sum_exp ────────────────────
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();

    // ── Cross-simdgroup reduction: outputs ──────────────────────────
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    // Same `+1` padded stride as `sdpa_multi` so adjacent lanes hit
    // distinct threadgroup-memory banks.
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_barrier();

    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "sdpa_bidirectional_d64",
        kernel_name: "ffai_sdpa_bidirectional_d64",
        kernel_ir: ffai_sdpa_bidirectional_d64::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

// ─── head_dim = 32 (FastViT-HD) ────────────────────────────────────

#[kernel]
pub fn ffai_sdpa_bidirectional_d32<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    let n_kv = base_kv + n_query;

    // Single tg_out slot at head_dim=32; 1 element per lane.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);

    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane;

    // Pre-scale this lane's single Q element once; K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;

    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;

    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        let k0_raw = load(k[kv0]);
        let k0 = k0_raw.cast::<f32>();
        let partial = q0 * k0;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v0 = v0_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
    }

    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();

    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_barrier();

    if sg == 0 {
        let mut so0 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "sdpa_bidirectional_d32",
        kernel_name: "ffai_sdpa_bidirectional_d32",
        kernel_ir: ffai_sdpa_bidirectional_d32::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

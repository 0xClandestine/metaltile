//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet (GDN) — decode + chunked-prefill kernels.
//!
//! GDN is the recurrent linear-attention variant Qwen3.5 / Qwen3.6 / Qwen3.6-MoE
//! use for their `linear_attention` layers (75% of layers in the hybrid
//! architecture). Two kernels:
//!
//!   - `mt_gated_delta_step`  — single-token decode (`T = 1`)
//!   - `mt_gated_delta_chunk` — multi-token chunked prefill (`T > 1`); the
//!     kernel that actually unblocks ctx > 2048 (issue #111). State stays
//!     register-resident across the inner T loop so the recurrence runs
//!     once per dispatch instead of N independent decode calls.
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

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    /// CPU oracle: matches `_gated_delta_step_ops` from mlx_lm/models/gated_delta.py.
    fn naive_gated_delta_step(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; b * hv * dv];
        let mut state_out = vec![0.0_f32; b * hv * dv * dk];
        let hk_per_hv = hv / hk;
        for batch in 0..b {
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx / hk_per_hv;
                let g_val = g[n];
                let beta_val = beta[n];
                let qk_base = (batch * hk + hk_idx) * dk;
                for dv_idx in 0..dv {
                    let v_val = v[n * dv + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;
                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    for s_idx in 0..dk {
                        let s = state_in[s_base + s_idx] * g_val;
                        decayed[s_idx] = s;
                        kv_mem += s * k[qk_base + s_idx];
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s_idx in 0..dk {
                        let s_new = decayed[s_idx] + k[qk_base + s_idx] * delta;
                        state_out[s_base + s_idx] = s_new;
                        out += s_new * q[qk_base + s_idx];
                    }
                    y[n * dv + dv_idx] = out;
                }
            }
        }
        (y, state_out)
    }

    #[test_kernel(name = "ffai/gated_delta/step", dtypes = [f32], tol = 1e-4)]
    fn test_gated_delta_step(dt: DType) -> TestSetup {
        use super::mt_gated_delta_step;
        let b = 2usize;
        let hv = 4usize;
        let hk = 2usize;
        let dv = 8usize;
        let dk = 64usize;
        let n_total = b * hv;

        let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0137).sin() * 0.4).collect();
        let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.4).collect();
        let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.019).sin() * 0.3).collect();
        let g: Vec<f32> = (0..n_total).map(|i| 0.8 + 0.1 * ((i as f32) * 0.07).cos()).collect();
        let beta: Vec<f32> = (0..n_total).map(|i| 0.3 + 0.1 * ((i as f32) * 0.05).sin()).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.05).collect();

        let (expected_y, expected_state) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);

        let mut kernel_ir = mt_gated_delta_step::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("q", pack(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack(&v, dt), dt))
            .input(TestBuffer::from_vec("g", pack(&g, dt), dt))
            .input(TestBuffer::from_vec("beta", pack(&beta, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack(&state_in, dt), dt))
            .input(TestBuffer::from_vec("dk", u32_le(dk as u32), DType::U32))
            .input(TestBuffer::from_vec("dv", u32_le(dv as u32), DType::U32))
            .input(TestBuffer::from_vec("hv", u32_le(hv as u32), DType::U32))
            .input(TestBuffer::from_vec("hk", u32_le(hk as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack(&expected_state, dt), dt))
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
    }
}

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
    // Per-lane register cache for the decayed state (`decayed`) and the
    // key slice (`k_cache`) — Metal places small fixed-size local arrays
    // in registers, so the inner loops in phase 1 + phase 2 read from
    // registers, not global memory. Replaces the prior "re-read state_in
    // and re-load k twice" pattern.
    //
    // Cap = 8 (n_per_t at the max supported Dk = 256). Smaller Dk just
    // under-utilises the upper slots.
    stack_alloc("decayed", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");
    let mut kv_mem = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = load(state_in[state_base + s_idx]).cast::<f32>() * g_val;
        let k_val = load(k[qk_base + s_idx]).cast::<f32>();
        stack_store("decayed", i, s_decayed);
        stack_store("k_cache", i, k_val);
        kv_mem = kv_mem + s_decayed * k_val;
    }
    let kv_mem_sum = simd_sum(kv_mem);
    let delta = (v_val - kv_mem_sum) * beta_val;
    // ─── Phase 2: rank-1 update + output projection ────────────────────
    //
    // Read decayed + k from the per-lane register caches (no global
    // load), apply the rank-1 update, store new state, accumulate
    // output against q. Matches MLX-LM's `float state[n_per_t]`
    // register-array pattern from `mlx_lm/models/gated_delta.py`.
    let mut out = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = stack_load("decayed", i);
        let k_val = stack_load("k_cache", i);
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

// ────────────────────────────────────────────────────────────────────
//  Chunked-prefill form (T > 1)
// ────────────────────────────────────────────────────────────────────

/// `mt_gated_delta_chunk` — multi-token GDN forward over `T` tokens.
///
/// Same recurrence math as `mt_gated_delta_step`, wrapped in an inner
/// `for t in 0..T` loop. The recurrent state stays in per-lane
/// stack-allocated registers across the entire T sweep, so a single
/// dispatch handles a full chunk of `T` tokens with one set of
/// load_state / store_state passes — vs `T` independent decode dispatches
/// which would each re-load + re-write the state.
///
/// This is the kernel that unblocks Qwen3.6 ctx > 2048: the hybrid
/// scheduler in mlx-swift-lm calls a chunked GDN kernel for the
/// `linear_attention` layers during prefill. The bug in issue #111 is
/// the scheduler currently emits a single chunk of 2048 with no T-loop
/// to span longer prefills; this kernel + a scheduler patch fix it.
///
/// Layouts (matching MLX-LM `_make_gated_delta_kernel`):
///
///   q, k     : [B, T, Hk, Dk]
///   v, y     : [B, T, Hv, Dv]
///   g, beta  : [B, T, Hv]
///   state    : [B, Hv, Dv, Dk]   (one state per (b, hv) — NO T dim;
///                                 state persists across t)
///
/// ## DISPATCH INVARIANTS
///
/// Same dispatch geometry as `mt_gated_delta_step`:
///
/// - **Mode: Reduction.** Each threadgroup is one simdgroup (32 threads).
/// - **Grid: `[dv, B * Hv, 1]`, TG: `[32, 1, 1]`.**
/// - **`dk % 32 == 0`.** Each lane owns `n_per_t = dk / 32` state
///   elements in a stack-allocated register array (cap 8 — Qwen3.6's
///   Dk=256 / 32). State survives across the entire `T`-loop.
/// - **`t_len` is a runtime u32** (passed as a scalar buffer, not a
///   constexpr) so a single PSO compiles for all chunk sizes the
///   scheduler picks.
#[kernel]
pub fn mt_gated_delta_chunk<T>(
    q: Tensor<T>,             // [B, T, Hk, Dk]
    k: Tensor<T>,             // [B, T, Hk, Dk]
    v: Tensor<T>,             // [B, T, Hv, Dv]
    g: Tensor<T>,             // [B, T, Hv]
    beta: Tensor<T>,          // [B, T, Hv]
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,         // [B, T, Hv, Dv]
    t_len: Tensor<u32>,       // [1] scalar — number of tokens in this chunk
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;
    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);
    let state_base = n * dv * dk + dv_idx * dk;
    // ─── Load state into per-lane registers once ─────────────────────
    //
    // State persists across all `T` recurrence steps in registers.
    // `k_cache` is reloaded per-token (each token has its own k row);
    // we don't carry it across t.
    stack_alloc("state_reg", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let val = load(state_in[state_base + s_idx]).cast::<f32>();
        stack_store("state_reg", i, val);
    }
    // ─── Inner T-loop: GDN recurrence per token ──────────────────────
    //
    // Pointer arithmetic per t:
    //   q[t], k[t]: (b * T + t) * Hk * Dk + hk_idx * Dk + s_idx
    //   v[t], y[t]: (b * T + t) * Hv * Dv + hv_idx * Dv + dv_idx
    //   g[t], beta[t]: (b * T + t) * Hv + hv_idx
    for t in range(0u32, t_total, 1u32) {
        let bt = b * t_total + t;
        let qk_base = (bt * hk + hk_idx) * dk;
        let vy_base = (bt * hv + hv_idx) * dv;
        let gbeta_idx = bt * hv + hv_idx;
        let g_val = load(g[gbeta_idx]).cast::<f32>();
        let beta_val = load(beta[gbeta_idx]).cast::<f32>();
        let v_val = load(v[vy_base + dv_idx]).cast::<f32>();
        // Phase 1: decay state + accumulate kv_mem; cache k.
        let mut kv_mem = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let s_old = stack_load("state_reg", i);
            let s_decayed = s_old * g_val;
            stack_store("state_reg", i, s_decayed);
            let k_val = load(k[qk_base + s_idx]).cast::<f32>();
            stack_store("k_cache", i, k_val);
            kv_mem = kv_mem + s_decayed * k_val;
        }
        let kv_mem_sum = simd_sum(kv_mem);
        let delta = (v_val - kv_mem_sum) * beta_val;
        // Phase 2: rank-1 update + output projection.
        let mut out = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let s_decayed = stack_load("state_reg", i);
            let k_val = stack_load("k_cache", i);
            let s_new = s_decayed + k_val * delta;
            stack_store("state_reg", i, s_new);
            let q_val = load(q[qk_base + s_idx]).cast::<f32>();
            out = out + s_new * q_val;
        }
        let out_sum = simd_sum(out);
        if dk_idx == 0u32 {
            store(y[vy_base + dv_idx], out_sum.cast::<T>());
        }
    }
    // ─── Write final state once at the end ──────────────────────────
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
    }
}

#[cfg(target_os = "macos")]
pub mod tests_support_ctx {
    //! GPU correctness tests for `mt_gated_delta_step` and `mt_gated_delta_chunk`.
    #![allow(clippy::too_many_arguments)]

    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::{mt_gated_delta_chunk, mt_gated_delta_step};

    // ── dtype helpers (mirrors tests/common/mod.rs) ───────────────────────

    #[derive(Clone, Copy, Debug)]
    enum Dt {
        F32,
        F16,
        Bf16,
    }

    impl Dt {
        fn bytes(self) -> usize {
            match self {
                Dt::F32 => 4,
                _ => 2,
            }
        }
        fn to_dtype(self) -> DType {
            match self {
                Dt::F32 => DType::F32,
                Dt::F16 => DType::F16,
                Dt::Bf16 => DType::BF16,
            }
        }
        fn round(self, v: f32) -> f32 {
            match self {
                Dt::F32 => v,
                Dt::F16 => half::f16::from_f32(v).to_f32(),
                Dt::Bf16 => half::bf16::from_f32(v).to_f32(),
            }
        }
    }

    fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
        }
    }

    fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            Dt::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dt::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        }
    }

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    // ── CPU oracle ────────────────────────────────────────────────────────

    fn naive_gated_delta_step(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; b * hv * dv];
        let mut state_out = vec![0.0_f32; b * hv * dv * dk];
        let hk_per_hv = hv / hk;
        for batch in 0..b {
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx / hk_per_hv;
                let g_val = g[n];
                let beta_val = beta[n];
                let qk_base = (batch * hk + hk_idx) * dk;
                for dv_idx in 0..dv {
                    let v_val = v[n * dv + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;
                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    for s_idx in 0..dk {
                        let s = state_in[s_base + s_idx] * g_val;
                        decayed[s_idx] = s;
                        kv_mem += s * k[qk_base + s_idx];
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s_idx in 0..dk {
                        let s_new = decayed[s_idx] + k[qk_base + s_idx] * delta;
                        state_out[s_base + s_idx] = s_new;
                        out += s_new * q[qk_base + s_idx];
                    }
                    y[n * dv + dv_idx] = out;
                }
            }
        }
        (y, state_out)
    }

    fn naive_gated_delta_chunk(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        b: usize,
        t: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = state_in.to_vec();
        let mut y_all = vec![0.0_f32; b * t * hv * dv];
        let hk_per_hv = hv / hk;
        for step in 0..t {
            let q_s = &q[step * b * hk * dk..(step + 1) * b * hk * dk];
            let k_s = &k[step * b * hk * dk..(step + 1) * b * hk * dk];
            let v_s = &v[step * b * hv * dv..(step + 1) * b * hv * dv];
            let g_s = &g[step * b * hv..(step + 1) * b * hv];
            let beta_s = &beta[step * b * hv..(step + 1) * b * hv];
            for batch in 0..b {
                for hv_idx in 0..hv {
                    let n = batch * hv + hv_idx;
                    let hk_idx = hv_idx / hk_per_hv;
                    let g_val = g_s[batch * hv + hv_idx];
                    let beta_val = beta_s[batch * hv + hv_idx];
                    let qk_base = (batch * hk + hk_idx) * dk;
                    for dv_idx in 0..dv {
                        let v_val = v_s[(batch * hv + hv_idx) * dv + dv_idx];
                        let s_base = n * dv * dk + dv_idx * dk;
                        let mut kv_mem = 0.0_f32;
                        let mut decayed = vec![0.0_f32; dk];
                        for s_idx in 0..dk {
                            let s = state[s_base + s_idx] * g_val;
                            decayed[s_idx] = s;
                            kv_mem += s * k_s[qk_base + s_idx];
                        }
                        let delta = (v_val - kv_mem) * beta_val;
                        let mut out = 0.0_f32;
                        for s_idx in 0..dk {
                            let s_new = decayed[s_idx] + k_s[qk_base + s_idx] * delta;
                            state[s_base + s_idx] = s_new;
                            out += s_new * q_s[qk_base + s_idx];
                        }
                        y_all[(step * b * hv + batch * hv + hv_idx) * dv + dv_idx] = out;
                    }
                }
            }
        }
        (y_all, state)
    }

    // ── dispatch helpers ──────────────────────────────────────────────────

    fn run_gated_delta_step(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        dt: Dt,
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_total = b * hv;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("q".into(), pack_bytes(q, dt));
        buffers.insert("k".into(), pack_bytes(k, dt));
        buffers.insert("v".into(), pack_bytes(v, dt));
        buffers.insert("g".into(), pack_bytes(g, dt));
        buffers.insert("beta".into(), pack_bytes(beta, dt));
        buffers.insert("state_in".into(), pack_bytes(state_in, dt));
        buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
        buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; b * hv * dv], dt));
        buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
        buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
        buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
        buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = mt_gated_delta_step::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Reduction;
        assert!(dk.is_multiple_of(32), "mt_gated_delta_step requires dk % 32 == 0");
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
            .expect("mt_gated_delta_step dispatch");
        let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
        let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
        (y, state_out)
    }

    fn run_gated_delta_chunk(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        dt: Dt,
        b: usize,
        t: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_total = b * hv;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("q".into(), pack_bytes(q, dt));
        buffers.insert("k".into(), pack_bytes(k, dt));
        buffers.insert("v".into(), pack_bytes(v, dt));
        buffers.insert("g".into(), pack_bytes(g, dt));
        buffers.insert("beta".into(), pack_bytes(beta, dt));
        buffers.insert("state_in".into(), pack_bytes(state_in, dt));
        buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
        buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; b * t * hv * dv], dt));
        buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());
        buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
        buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
        buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
        buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = mt_gated_delta_chunk::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Reduction;
        assert!(dk.is_multiple_of(32), "mt_gated_delta_chunk requires dk % 32 == 0");
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
            .expect("mt_gated_delta_chunk dispatch");
        let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
        let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
        (y, state_out)
    }

    // ── step tests ────────────────────────────────────────────────────────

    #[test]
    fn gated_delta_step_identity_at_g1_beta0_f32() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 2;
        let hk = 1;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q = vec![1.0_f32; b * hk * dk];
        let k = vec![1.0_f32; b * hk * dk];
        let v: Vec<f32> = (0..b * hv * dv).map(|i| i as f32 * 0.1).collect();
        let g = vec![1.0_f32; b * hv];
        let beta = vec![0.0_f32; b * hv];
        let state_in = vec![0.0_f32; n_total * dv * dk];
        let (y, _) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        for (i, &yv) in y.iter().enumerate() {
            assert!(yv.is_finite(), "y[{i}] non-finite: {yv}");
        }
    }

    #[test]
    fn gated_delta_step_matches_cpu_oracle_f32() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 4;
        let hk = 2;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let v: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> = (0..b * hv).map(|i| 0.9 - (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..b * hv).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
        let (y_expected, state_expected) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_actual, state_actual) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y =
            y_actual.iter().zip(&y_expected).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-5, "y max |diff| = {max_y:.2e}");
        let max_s = state_actual
            .iter()
            .zip(&state_expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_s < 1e-5, "state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_step_gqa_matches_oracle_f32() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 4;
        let hk = 1;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.011).sin() * 0.5).collect();
        let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).cos() * 0.5).collect();
        let v: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.027).sin() * 0.4).collect();
        let g: Vec<f32> = (0..b * hv).map(|i| 0.85 + (i as f32) * 0.02).collect();
        let beta: Vec<f32> = (0..b * hv).map(|i| 0.3 + (i as f32) * 0.02).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.009).cos() * 0.2).collect();
        let (y_exp, s_exp) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, s_act) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-5, "GQA y max |diff| = {max_y:.2e}");
        let max_s = s_act.iter().zip(&s_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_s < 1e-5, "GQA state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_step_v_zero_state_only_decays_f32() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 2;
        let hk = 1;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = vec![0.1_f32; b * hk * dk];
        let k: Vec<f32> = vec![0.1_f32; b * hk * dk];
        let v = vec![0.0_f32; b * hv * dv];
        let g = vec![0.9_f32; b * hv];
        let beta = vec![0.5_f32; b * hv];
        let state_in: Vec<f32> = (0..n_total * dv * dk).map(|i| i as f32 * 0.01).collect();
        let (y_exp, _) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, _) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-5, "v=0 decay y max |diff| = {max_y:.2e}");
    }

    #[test]
    fn gated_delta_step_f16_matches_oracle() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 2;
        let hk = 1;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q_f32: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let k_f32: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let v_f32: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g_f32: Vec<f32> = (0..b * hv).map(|i| 0.9 - (i as f32) * 0.01).collect();
        let beta_f32: Vec<f32> = (0..b * hv).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let state_f32: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
        let round = |v: &[f32]| v.iter().map(|&x| Dt::F16.round(x)).collect::<Vec<_>>();
        let q = round(&q_f32);
        let k = round(&k_f32);
        let v = round(&v_f32);
        let g = round(&g_f32);
        let beta = round(&beta_f32);
        let state_in = round(&state_f32);
        let (y_exp, _) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, _) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F16, b, hv, hk, dv, dk);
        let max_rel = y_act
            .iter()
            .zip(&y_exp)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
            .fold(0.0_f32, f32::max);
        assert!(max_rel < 5e-3, "f16 step max rel = {max_rel:.2e}");
    }

    #[test]
    fn gated_delta_step_qwen36_dk_256_f32() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 4;
        let hk = 2;
        let dv = 4;
        let dk = 256;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0019).sin() * 0.2).collect();
        let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0023).cos() * 0.2).collect();
        let v: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> = vec![0.92_f32; b * hv];
        let beta: Vec<f32> = vec![0.4_f32; b * hv];
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.05).collect();
        let (y_exp, s_exp) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, s_act) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-4, "Dk=256 y max |diff| = {max_y:.2e}");
        let max_s = s_act.iter().zip(&s_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_s < 1e-4, "Dk=256 state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_step_no_gqa_f32() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 4;
        let hk = 4;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let v: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> = (0..b * hv).map(|i| 0.88 + (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..b * hv).map(|i| 0.4 + (i as f32) * 0.02).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.009).cos() * 0.15).collect();
        let (y_exp, s_exp) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, s_act) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-5, "no-GQA y max |diff| = {max_y:.2e}");
        let max_s = s_act.iter().zip(&s_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_s < 1e-5, "no-GQA state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_step_batch_4_f32() {
        let _g = gpu_lock();
        let b = 4;
        let hv = 2;
        let hk = 1;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let v: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> = (0..b * hv).map(|i| 0.9 - (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..b * hv).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
        let (y_exp, s_exp) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, s_act) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-5, "B=4 y max |diff| = {max_y:.2e}");
        let max_s = s_act.iter().zip(&s_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_s < 1e-5, "B=4 state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_step_bf16_matches_oracle() {
        let _g = gpu_lock();
        let b = 1;
        let hv = 2;
        let hk = 1;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<_>>();
        let q =
            round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect::<Vec<_>>());
        let k =
            round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect::<Vec<_>>());
        let v =
            round(&(0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect::<Vec<_>>());
        let g = round(&(0..b * hv).map(|i| 0.9 - (i as f32) * 0.01).collect::<Vec<_>>());
        let beta = round(&(0..b * hv).map(|i| 0.5 + (i as f32) * 0.01).collect::<Vec<_>>());
        let state_in = round(
            &(0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect::<Vec<_>>(),
        );
        let (y_exp, _) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_act, _) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::Bf16, b, hv, hk, dv, dk);
        let max_rel = y_act
            .iter()
            .zip(&y_exp)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
            .fold(0.0_f32, f32::max);
        assert!(max_rel < 5e-2, "bf16 step max rel = {max_rel:.2e}");
    }

    // ── chunk tests ───────────────────────────────────────────────────────

    #[test]
    fn gated_delta_chunk_t1_matches_decode_form_f32() {
        let _g = gpu_lock();
        let b = 1;
        let t = 1;
        let hv = 4;
        let hk = 2;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * t * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let k: Vec<f32> = (0..b * t * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let v: Vec<f32> = (0..b * t * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> = (0..b * t * hv).map(|i| 0.9 - (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..b * t * hv).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
        let (y_chunk, s_chunk) =
            run_gated_delta_chunk(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, t, hv, hk, dv, dk);
        let (y_decode, s_decode) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
        let max_y =
            y_chunk.iter().zip(&y_decode).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-5, "chunk T=1 vs decode y max |diff| = {max_y:.2e}");
        let max_s =
            s_chunk.iter().zip(&s_decode).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_s < 1e-5, "chunk T=1 vs decode state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_chunk_t_64_matches_oracle_f32() {
        let _g = gpu_lock();
        let b = 1;
        let t = 64;
        let hv = 4;
        let hk = 2;
        let dv = 4;
        let dk = 32;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * t * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let k: Vec<f32> = (0..b * t * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let v: Vec<f32> = (0..b * t * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> =
            (0..b * t * hv).map(|i| 0.92 + ((i as f32) * 0.0001).sin() * 0.05).collect();
        let beta: Vec<f32> =
            (0..b * t * hv).map(|i| 0.4 + ((i as f32) * 0.0001).cos() * 0.1).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
        let (y_exp, s_exp) =
            naive_gated_delta_chunk(&q, &k, &v, &g, &beta, &state_in, b, t, hv, hk, dv, dk);
        let (y_act, s_act) =
            run_gated_delta_chunk(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, t, hv, hk, dv, dk);
        let max_y = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-3, "chunk T=64 y max |diff| = {max_y:.2e}");
        let max_s = s_act.iter().zip(&s_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_s < 1e-3, "chunk T=64 state max |diff| = {max_s:.2e}");
    }

    #[test]
    fn gated_delta_chunk_qwen36_dk_256_f32() {
        let _g = gpu_lock();
        let b = 1;
        let t = 8;
        let hv = 2;
        let hk = 1;
        let dv = 2;
        let dk = 256;
        let n_total = b * hv;
        let q: Vec<f32> = (0..b * t * hk * dk).map(|i| ((i as f32) * 0.0019).sin() * 0.2).collect();
        let k: Vec<f32> = (0..b * t * hk * dk).map(|i| ((i as f32) * 0.0023).cos() * 0.2).collect();
        let v: Vec<f32> = (0..b * t * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> = vec![0.92_f32; b * t * hv];
        let beta: Vec<f32> = vec![0.4_f32; b * t * hv];
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.05).collect();
        let (y_exp, _) =
            naive_gated_delta_chunk(&q, &k, &v, &g, &beta, &state_in, b, t, hv, hk, dv, dk);
        let (y_act, _) =
            run_gated_delta_chunk(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, t, hv, hk, dv, dk);
        let max_diff = y_act.iter().zip(&y_exp).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 2e-3, "Dk=256 T=8 y max |diff| = {max_diff:.2e}");
    }
}

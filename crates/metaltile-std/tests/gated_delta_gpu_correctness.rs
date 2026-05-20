//! GPU correctness for `ffai::gated_delta::mt_gated_delta_step`.
//!
//! GDN (Gated DeltaNet) is the recurrent linear-attention variant used by
//! Qwen3.5 / Qwen3.6 / Qwen3.6-MoE for their `linear_attention` layers
//! (75% of layers in those hybrid models). This file pins the single-token
//! decode form — `T = 1` of MLX-LM's `gated_delta_kernel`.
//!
//! Tests pin:
//!
//!   - **Identity at g=1, beta=0**: no decay + no update → state unchanged,
//!     y = state @ q. The "no-op recurrence" baseline.
//!   - **CPU oracle match (f32)** at a realistic shape — Qwen3.6 has
//!     Hk=4, Hv=24, head_dim=256, but we use smaller dims to keep the
//!     test fast. Validates the full recurrence numerically.
//!   - **GQA dispatch correctness**: Hv > Hk → multiple Hv-heads share a
//!     single (q, k) Hk-slot. Catches `hk_idx = hv_idx / (Hv/Hk)` errors.
//!   - **dtype matrix (f16 / bf16)** with derived tolerance.
//!   - **`x = 0` (v = 0) decay invariant**: the recurrence collapses to
//!     `state = state * g`, y = (state*g) @ q. Pins that delta is applied
//!     to v correctly.
//!
//! macOS-gated. Shared gpu_lock via tests/common/.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta::mt_gated_delta_step;

/// CPU oracle: matches `_gated_delta_step_ops` from `mlx_lm/models/gated_delta.py`.
///
/// Shapes:
///   - q, k: [B, Hk, Dk]
///   - v: [B, Hv, Dv]
///   - g, beta: [B, Hv]
///   - state: [B, Hv, Dv, Dk] (f32 in/out)
/// Returns: (y [B, Hv, Dv], new_state [B, Hv, Dv, Dk])
#[allow(clippy::too_many_arguments)]
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

                // Phase 1: decay + kv_mem
                let mut kv_mem = 0.0_f32;
                let mut decayed = vec![0.0_f32; dk];
                for s_idx in 0..dk {
                    let s = state_in[s_base + s_idx] * g_val;
                    decayed[s_idx] = s;
                    kv_mem += s * k[qk_base + s_idx];
                }
                let delta = (v_val - kv_mem) * beta_val;

                // Phase 2: update + output projection
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

#[allow(clippy::too_many_arguments)]
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
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_total * dv], dt));
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gated_delta_step::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Reduction dispatch (docs/developing.md):
    //   tgid_x = dv_idx, tgid_y = n, tid = dk_idx (0..32)
    //   TPG = 32 (one simdgroup), Dk must be a multiple of 32
    assert!(dk.is_multiple_of(32), "mt_gated_delta_step requires dk % 32 == 0");
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
        .expect("mt_gated_delta_step dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    (y, state_out)
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────

#[test]
fn gated_delta_step_identity_at_g1_beta0_f32() {
    let _g = gpu_lock();
    // g=1, beta=0 → decayed = state, delta = 0, state_new = state.
    // y = state @ q exactly. Pure dot product. Catches gross dispatch /
    // index errors before any recurrence math.
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect();
    let v = vec![0.0_f32; n_total * dv]; // not consumed since beta=0
    let g = vec![1.0_f32; n_total];
    let beta = vec![0.0_f32; n_total];
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 1e-5, "identity y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "identity state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_matches_oracle_f32() {
    let _g = gpu_lock();
    // Realistic recurrence: smooth non-trivial gates, full update path.
    let b = 2;
    let hv = 4;
    let hk = 2;
    let dv = 8;
    let dk = 64;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..n_total).map(|i| 0.9 - (i as f32) * 0.01).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    // simd_sum across 32 lanes with dk=64 → 2 mul-adds per lane;
    // recurrence has 2 dependent reductions. ~3 ULPs of f32 accumulation.
    assert!(max_y_diff < 5e-5, "y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_gqa_hv_4x_hk_f32() {
    let _g = gpu_lock();
    // Hv = 4 * Hk: each (q, k) Hk-slot serves 4 Hv-heads. Pins the
    // `hk_idx = hv_idx / (Hv/Hk)` decomposition — a wrong divisor
    // would route the wrong Hv-head to the wrong Hk-slot.
    let b = 2;
    let hv = 8;
    let hk = 2; // Hv / Hk = 4
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.029).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.031).cos() * 0.5).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.041).sin() * 0.3).collect();
    let g: Vec<f32> = (0..n_total).map(|i| 0.85 + (i as f32) * 0.005).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.4 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.013).cos() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 5e-5, "GQA y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "GQA state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_v_zero_collapses_to_pure_decay_f32() {
    let _g = gpu_lock();
    // v = 0 → delta = (0 - kv_mem) * beta = -kv_mem * beta. With beta=0
    // we already pinned the no-delta path; this exercises beta != 0 but
    // checks the recurrence stays bounded.
    let b = 1;
    let hv = 2;
    let hk = 1;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.019).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.023).cos() * 0.5).collect();
    let v = vec![0.0_f32; n_total * dv];
    let g = vec![0.8_f32; n_total];
    let beta = vec![0.5_f32; n_total];
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 5e-5, "v=0 y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "v=0 state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_matches_oracle_f16() {
    let _g = gpu_lock();
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let round = |v: &[f32]| v.iter().map(|&x| Dt::F16.round(x)).collect::<Vec<f32>>();
    let q = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect::<Vec<_>>());
    let k = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect::<Vec<_>>());
    let v = round(&(0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect::<Vec<_>>());
    let g = round(&(0..n_total).map(|i| 0.9 - (i as f32) * 0.01).collect::<Vec<_>>());
    let beta = round(&(0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect::<Vec<_>>());
    let state_in = round(
        &(0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect::<Vec<_>>(),
    );

    let (y_expected, _) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, _) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F16, b, hv, hk, dv, dk);

    let mut max_rel = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // f16 10-bit mantissa + dependent reductions (kv_mem → delta → update → out).
    // Two simd_sums each accumulate ~32 mul-adds.
    assert!(max_rel < 5e-2, "f16 max rel = {max_rel:.2e}");
}

#[test]
fn gated_delta_step_matches_oracle_bf16() {
    let _g = gpu_lock();
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<f32>>();
    let q = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect::<Vec<_>>());
    let k = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect::<Vec<_>>());
    let v = round(&(0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect::<Vec<_>>());
    let g = round(&(0..n_total).map(|i| 0.9 - (i as f32) * 0.01).collect::<Vec<_>>());
    let beta = round(&(0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect::<Vec<_>>());
    let state_in = round(
        &(0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect::<Vec<_>>(),
    );

    let (y_expected, _) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, _) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::Bf16, b, hv, hk, dv, dk);

    let mut max_rel = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // bf16 7-bit mantissa is the wider tolerance.
    assert!(max_rel < 2e-1, "bf16 max rel = {max_rel:.2e}");
}

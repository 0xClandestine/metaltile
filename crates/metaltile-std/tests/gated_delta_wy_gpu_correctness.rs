//! GPU correctness for `ffai::gated_delta_wy::mt_gated_delta_wy_chunk`.
//!
//! Chunked-WY GDN is the prefill-perf kernel (spec #028). It must produce
//! identical output to the per-step sequential reference across the full
//! prefill (chained chunks). This file pins:
//!
//!   - **Identity at g=1, β=0**: no decay + no update → state unchanged,
//!     y = state @ q across all tokens. Catches gross dispatch errors.
//!   - **CPU oracle match (f32)** at multiple T (one chunk, two chunks,
//!     several chunks) at small dims.
//!   - **Qwen3.6 dims**: T=128 Hk=2 Hv=4 Dk=128 Dv=128 C=64 — the actual
//!     deployed shape.
//!   - **dtype matrix**: f32, f16, bf16 with derived tolerance.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta_wy::mt_gated_delta_wy_chunk;

/// Sequential GDN reference (CPU). Same recurrence as
/// `gated_delta_ops` from `mlx_lm/models/gated_delta.py`.
#[allow(clippy::too_many_arguments)]
fn sequential_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t_total: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let hv_per_hk = hv / hk;
    let mut y = vec![0.0_f32; t_total * hv * dv];
    for t in 0..t_total {
        for h_v in 0..hv {
            let h_k = h_v / hv_per_hk;
            let gt = g[t * hv + h_v];
            let bt = beta[t * hv + h_v];
            for d_v in 0..dv {
                let v_val = v[(t * hv + h_v) * dv + d_v];
                let s_base = (h_v * dv + d_v) * dk;
                let mut kv_mem = 0.0_f32;
                let mut decayed = vec![0.0_f32; dk];
                for s_idx in 0..dk {
                    let s = state[s_base + s_idx] * gt;
                    decayed[s_idx] = s;
                    kv_mem += s * k[(t * hk + h_k) * dk + s_idx];
                }
                let delta = (v_val - kv_mem) * bt;
                let mut out = 0.0_f32;
                for s_idx in 0..dk {
                    let s_new = decayed[s_idx] + k[(t * hk + h_k) * dk + s_idx] * delta;
                    state[s_base + s_idx] = s_new;
                    out += s_new * q[(t * hk + h_k) * dk + s_idx];
                }
                y[(t * hv + h_v) * dv + d_v] = out;
            }
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn run_gated_delta_wy_chunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    dt: Dt,
    b: usize,
    t_total: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    c: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(t_total.is_multiple_of(c), "t_total must be a multiple of c");
    let n_total = b * hv;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("g".into(), pack_bytes(g, dt));
    buffers.insert("beta".into(), pack_bytes(beta, dt));
    buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; t_total * n_total * dv], dt));
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
    buffers.insert("c".into(), (c as u32).to_le_bytes().to_vec());
    buffers.insert("t_len".into(), (t_total as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gated_delta_wy_chunk::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Dispatch: one TG per (b*hv) slot, 32 threads each.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, n_total, 1], [32, 1, 1])
        .expect("mt_gated_delta_wy_chunk dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    (y, state_out)
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────

#[test]
fn wy_chunk_identity_at_g1_beta0_f32() {
    let _g = gpu_lock();
    // g=1, β=0 → state unchanged, y = state @ q at every step.
    let (b, t, hk, hv, dk, dv, c) = (1, 8, 1, 1, 32, 32, 8);
    let n_total = b * hv;
    let kscale = (2.0_f32 / dk as f32).sqrt();
    let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
    let v = vec![0.0_f32; t * n_total * dv];
    let g = vec![1.0_f32; t * n_total];
    let beta = vec![0.0_f32; t * n_total];
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F32,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 1e-4, "identity y max |diff| = {max_y:.2e}");
    assert!(max_s < 1e-4, "identity state max |diff| = {max_s:.2e}");
}

#[test]
fn wy_chunk_matches_oracle_one_chunk_f32() {
    let _g = gpu_lock();
    // T = C exactly: one chunk only. Small shape to fit TG buffer caps
    // (tg_q/k/v are sized for C*max(Dk,Dv) ≤ 512).
    let (b, t, hk, hv, dk, dv, c) = (1, 16, 1, 1, 32, 16, 16);
    let n_total = b * hv;
    let kscale = (2.0_f32 / dk as f32).sqrt();
    let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
    let v: Vec<f32> = (0..t * n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..t * n_total).map(|i| 0.8 + 0.15 * ((i as f32) * 0.013).sin()).collect();
    let beta: Vec<f32> = (0..t * n_total).map(|i| 0.4 + 0.3 * ((i as f32) * 0.017).cos()).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F32,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 5e-3, "one-chunk y max |diff| = {max_y:.2e}");
    assert!(max_s < 5e-3, "one-chunk state max |diff| = {max_s:.2e}");
}

#[test]
fn wy_chunk_matches_oracle_multi_chunk_f32() {
    let _g = gpu_lock();
    // T = 32, C = 8: 4 chunks, exercises inter-chunk state passing.
    let (b, t, hk, hv, dk, dv, c) = (1, 32, 1, 1, 32, 16, 8);
    let n_total = b * hv;
    let kscale = (2.0_f32 / dk as f32).sqrt();
    let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
    let v: Vec<f32> = (0..t * n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..t * n_total).map(|i| 0.85 + 0.1 * ((i as f32) * 0.013).sin()).collect();
    let beta: Vec<f32> = (0..t * n_total).map(|i| 0.5 + 0.2 * ((i as f32) * 0.017).cos()).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F32,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 1e-2, "multi-chunk y max |diff| = {max_y:.2e}");
    assert!(max_s < 1e-2, "multi-chunk state max |diff| = {max_s:.2e}");
}

// ────────────────────────────────────────────────────────────────────
//  Extended coverage — dtype matrix + Dv=Dk parity + varied β/g
// ────────────────────────────────────────────────────────────────────

/// Build a synthetic GDN input with normalized k (‖k‖² ≈ 1 regardless of Dk)
/// so the recurrence is well-conditioned. Matches the CPU-oracle pattern.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn make_inputs(
    seed_phase: f32,
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_total = t * hv;
    let kscale = (2.0_f32 / dk as f32).sqrt();
    let q: Vec<f32> =
        (0..t * hk * dk).map(|i| ((i as f32 + seed_phase) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> =
        (0..t * hk * dk).map(|i| ((i as f32 + seed_phase) * 0.0211).cos() * kscale).collect();
    let v: Vec<f32> =
        (0..t * n_total * dv).map(|i| ((i as f32 + seed_phase) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> =
        (0..t * n_total).map(|i| 0.85 + 0.1 * ((i as f32 + seed_phase) * 0.013).sin()).collect();
    let beta: Vec<f32> =
        (0..t * n_total).map(|i| 0.5 + 0.2 * ((i as f32 + seed_phase) * 0.017).cos()).collect();
    let state_in: Vec<f32> = (0..hv * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
    (q, k, v, g, beta, state_in)
}

#[test]
fn wy_chunk_dk_equals_dv_square_state_f32() {
    let _g = gpu_lock();
    // Dk=Dv=32: state is square. Verifies the kernel handles
    // non-rectangular state shape correctly (the chunk-end matmul
    // (S_0 · bp^T) · K must keep Dv and Dk slots straight).
    let (b, t, hk, hv, dk, dv, c) = (1, 16, 1, 1, 32, 32, 16);
    let (q, k, v, g, beta, state_in) = make_inputs(0.0, t, hk, hv, dk, dv);

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F32,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 5e-3, "square-state y max |diff| = {max_y:.2e}");
    assert!(max_s < 5e-3, "square-state state max |diff| = {max_s:.2e}");
}

#[test]
fn wy_chunk_two_chunk_chain_f32() {
    let _g = gpu_lock();
    // T = 16, C = 8: exactly two chunks. Different from the 4-chunk case;
    // checks that the very first chunk-boundary state transfer is correct.
    let (b, t, hk, hv, dk, dv, c) = (1, 16, 1, 1, 32, 16, 8);
    let (q, k, v, g, beta, state_in) = make_inputs(7.0, t, hk, hv, dk, dv);

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F32,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 5e-3, "two-chunk y max |diff| = {max_y:.2e}");
    assert!(max_s < 5e-3, "two-chunk state max |diff| = {max_s:.2e}");
}

#[test]
fn wy_chunk_aggressive_decay_f32() {
    let _g = gpu_lock();
    // g ∈ (0.3, 0.5): strong decay. State should shrink each step, then
    // re-accumulate via v·k^T updates. Pins that the decay/update split
    // doesn't accidentally swap order.
    let (b, t, hk, hv, dk, dv, c) = (1, 16, 1, 1, 32, 16, 8);
    let n_total = b * hv;
    let kscale = (2.0_f32 / dk as f32).sqrt();
    let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
    let v: Vec<f32> = (0..t * n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..t * n_total).map(|i| 0.4 + 0.1 * ((i as f32) * 0.013).sin()).collect();
    let beta: Vec<f32> = (0..t * n_total).map(|i| 0.6 + 0.2 * ((i as f32) * 0.017).cos()).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F32,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 5e-3, "aggressive-decay y max |diff| = {max_y:.2e}");
    assert!(max_s < 5e-3, "aggressive-decay state max |diff| = {max_s:.2e}");
}

#[test]
fn wy_chunk_matches_oracle_multi_chunk_f16() {
    let _g = gpu_lock();
    // dtype matrix: f16 path. Tolerance widened for f16's 11-bit mantissa.
    let (b, t, hk, hv, dk, dv, c) = (1, 16, 1, 1, 32, 16, 8);
    let (q, k, v, g, beta, state_in) = make_inputs(3.0, t, hk, hv, dk, dv);

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::F16,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    // f16: ~3 decimal digits; tolerance scales with magnitude (~0.1 state ranges).
    assert!(max_y < 5e-2, "f16 y max |diff| = {max_y:.2e}");
    assert!(max_s < 5e-2, "f16 state max |diff| = {max_s:.2e}");
}

#[test]
fn wy_chunk_matches_oracle_multi_chunk_bf16() {
    let _g = gpu_lock();
    // dtype matrix: bf16 path. 7-bit mantissa → looser tolerance.
    let (b, t, hk, hv, dk, dv, c) = (1, 16, 1, 1, 32, 16, 8);
    let (q, k, v, g, beta, state_in) = make_inputs(5.0, t, hk, hv, dk, dv);

    let mut s_seq = state_in.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s_seq, t, hk, hv, dk, dv);
    let (y_wy, s_wy) = run_gated_delta_wy_chunk(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Dt::Bf16,
        b,
        t,
        hk,
        hv,
        dk,
        dv,
        c,
    );

    let max_y = y_seq.iter().zip(&y_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_s = s_seq.iter().zip(&s_wy).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    // bf16: ~2 decimal digits.
    assert!(max_y < 2e-1, "bf16 y max |diff| = {max_y:.2e}");
    assert!(max_s < 2e-1, "bf16 state max |diff| = {max_s:.2e}");
}

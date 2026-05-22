//! End-to-end correctness test for `ffai::ssm::ssm_step_a2d` — the
//! Mamba 1 (Jamba) selective-scan single-token decode step with a 2-D
//! per-(channel, state) `A_log`.
//!
//! Grid3D, one thread per `(head, d)` element. Updates the per-head
//! state `h [n_heads, state_dim, head_dim]` in fp32 (state accumulates
//! across many decode steps; bf16 mantissa drifts fast).
//!
//! Recurrence (per (h, d, n) state element):
//!   A     = -exp(A_log[(h*head_dim + d), n])   ← 2-D, varies with n
//!   decay = exp(A * dt[h])
//!   h'    = decay * h[h, n, d] + dt[h] * b[n] * x[h, d]
//!   y[h,d] = Σ_n c[n] * h'[h, n, d]
//!
//! The contrast with the scalar `ssm_step` is the `A_log` index: this
//! variant reads a distinct decay per `(channel, state)` pair, so the
//! decay is no longer constant inside the state loop.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::ssm::ssm_step_a2d;

/// CPU reference matching the kernel's per-(h, d) recurrence with a
/// 2-D `A_log`, the in-place state update, and the y projection. The
/// `dt` callback rounds operands through the kernel's dtype so the
/// reference sees the same load-cast quantisation.
#[allow(clippy::too_many_arguments)]
fn naive_ssm_step_a2d(
    x: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    dt: &[f32],
    h: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    state_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_heads * head_dim];
    for hi in 0..n_heads {
        let dt_val = dt[hi];
        let h_base = hi * state_dim * head_dim;
        for d in 0..head_dim {
            let x_d = x[hi * head_dim + d];
            let channel = hi * head_dim + d;
            let a_log_base = channel * state_dim;
            let mut y_d = 0.0_f32;
            for n in 0..state_dim {
                let a_val = -a_log[a_log_base + n].exp();
                let decay = (a_val * dt_val).exp();
                let h_idx = h_base + n * head_dim + d;
                let h_old = h[h_idx];
                let new_h = decay * h_old + dt_val * b[n] * x_d;
                h[h_idx] = new_h;
                y_d += c[n] * new_h;
            }
            y[hi * head_dim + d] = y_d;
        }
    }
    y
}

/// Run the kernel for one dtype and check `y` + the in-place `h`
/// update against the CPU reference.
fn run_case(dt_kind: Dt) {
    let _g = gpu_lock();
    let n_heads = 4usize;
    let head_dim = 32usize;
    let state_dim = 16usize;

    // Round every operand through the dtype so the oracle matches the
    // kernel's load-cast precision. `h` is genuinely fp32 on both sides.
    let r = |v: f32| dt_kind.round(v);
    let x: Vec<f32> = (0..n_heads * head_dim).map(|i| r(((i % 11) as f32 - 5.0) * 0.02)).collect();
    // A_log is the raw log-param; the kernel applies A = -exp(A_log),
    // so any real A_log yields a stable decay = exp(-exp(A_log)*dt) < 1.
    // Distinct value per (channel, state) — the whole point of the 2-D form.
    let a_log: Vec<f32> =
        (0..n_heads * head_dim * state_dim).map(|i| r(-1.0 + 0.013 * (i as f32 % 19.0))).collect();
    let b: Vec<f32> = (0..state_dim).map(|i| r(((i % 5) as f32 - 2.0) * 0.05)).collect();
    let c: Vec<f32> = (0..state_dim).map(|i| r(((i % 7) as f32 - 3.0) * 0.03)).collect();
    let dt: Vec<f32> = (0..n_heads).map(|i| r(0.1 + 0.05 * i as f32)).collect();
    let h: Vec<f32> =
        (0..n_heads * state_dim * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

    let mut h_cpu = h.clone();
    let expected_y =
        naive_ssm_step_a2d(&x, &a_log, &b, &c, &dt, &mut h_cpu, n_heads, head_dim, state_dim);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, dt_kind));
    buffers.insert("a_log".into(), pack_bytes(&a_log, dt_kind));
    buffers.insert("b".into(), pack_bytes(&b, dt_kind));
    buffers.insert("c".into(), pack_bytes(&c, dt_kind));
    buffers.insert("dt".into(), pack_bytes(&dt, dt_kind));
    buffers.insert("h".into(), pack_bytes(&h, Dt::F32));
    buffers.insert("y".into(), vec![0u8; n_heads * head_dim * dt_kind.bytes()]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("state_dim".into(), (state_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = ssm_step_a2d::kernel_ir_for(dt_kind.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: total threads = grid.x × tg.x. One thread per (head, d).
    let total = n_heads * head_dim;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [total, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let actual_y = unpack_bytes(result.outputs.get("y").expect("y"), dt_kind);
    let actual_h = unpack_bytes(result.outputs.get("h").expect("h"), Dt::F32);

    // f16/bf16 only carry through the activation operands + the y store;
    // the recurrence + state stay fp32, so the tolerance is modest.
    let (y_tol, h_tol, label) = match dt_kind {
        Dt::F32 => (1e-4, 1e-4, "f32"),
        Dt::F16 => (5e-3, 5e-3, "f16"),
        Dt::Bf16 => (3e-2, 3e-2, "bf16"),
    };
    let y_diff = max_abs_diff(&expected_y, &actual_y);
    let h_diff = max_abs_diff(&h_cpu, &actual_h);
    assert!(y_diff < y_tol, "{label}: ssm_step_a2d y max |diff| = {y_diff:.2e}");
    assert!(h_diff < h_tol, "{label}: ssm_step_a2d h max |diff| = {h_diff:.2e}");
}

#[test]
fn ssm_step_a2d_matches_naive_reference_f32() { run_case(Dt::F32); }

#[test]
fn ssm_step_a2d_matches_naive_reference_f16() { run_case(Dt::F16); }

#[test]
fn ssm_step_a2d_matches_naive_reference_bf16() { run_case(Dt::Bf16); }

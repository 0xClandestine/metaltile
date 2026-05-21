//! End-to-end correctness test for `ffai::ssm_step` (Mamba 2 SSD-form
//! single-token decode) on real Metal.
//!
//! Grid3D, one thread per `(head, d)` element. Updates the per-head
//! state `h [n_heads, state_dim, head_dim]` in fp32 (state accumulates
//! across many decode steps; bf16 mantissa drifts fast).
//!
//! Recurrence (per (h, d, n) state element):
//!   decay = exp(a[h] * dt[h])
//!   h'[h, n, d] = decay * h[h, n, d] + dt[h] * b[n] * x[h, d]
//!   y[h, d]    = Σ_n c[n] * h'[h, n, d]
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::ssm::ssm_step;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// CPU reference matching the kernel's per-(h, d) recurrence + the
/// in-place state update + y projection.
#[allow(clippy::too_many_arguments)]
fn naive_ssm_step(
    x: &[f32],
    a: &[f32],
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
        let a_val = a[hi];
        let decay = (a_val * dt_val).exp();
        let h_base = hi * state_dim * head_dim;
        for d in 0..head_dim {
            let x_d = x[hi * head_dim + d];
            let mut y_d = 0.0_f32;
            for n in 0..state_dim {
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

#[test]
fn ssm_step_matches_naive_reference_f32() {
    let n_heads = 4usize;
    let head_dim = 32usize;
    let state_dim = 16usize;

    // Small ramps so the recurrence stays well-behaved (no exp blow-up).
    let x: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
    // a < 0 keeps `decay = exp(a*dt) < 1` (stable recurrence).
    let a: Vec<f32> = (0..n_heads).map(|i| -0.1 * (i + 1) as f32).collect();
    let b: Vec<f32> = (0..state_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let c: Vec<f32> = (0..state_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.03).collect();
    let dt: Vec<f32> = (0..n_heads).map(|i| 0.1 + 0.05 * i as f32).collect();
    let h: Vec<f32> =
        (0..n_heads * state_dim * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let mut h_cpu = h.clone();
    let expected_y = naive_ssm_step(&x, &a, &b, &c, &dt, &mut h_cpu, n_heads, head_dim, state_dim);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), f32_slice_to_bytes(&x));
    buffers.insert("a".into(), f32_slice_to_bytes(&a));
    buffers.insert("b".into(), f32_slice_to_bytes(&b));
    buffers.insert("c".into(), f32_slice_to_bytes(&c));
    buffers.insert("dt".into(), f32_slice_to_bytes(&dt));
    buffers.insert("h".into(), f32_slice_to_bytes(&h));
    buffers.insert("y".into(), vec![0u8; n_heads * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("state_dim".into(), (state_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = ssm_step::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: total threads = grid_groups.x × tg.x. For N legitimate
    // threads dispatch [1,1,1]/[N,1,1] — see conv1d test header.
    let total = n_heads * head_dim;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [total, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let actual_y = bytes_to_f32_vec(result.outputs.get("y").expect("y"));
    let actual_h = bytes_to_f32_vec(result.outputs.get("h").expect("h"));

    let y_diff = max_abs_diff(&expected_y, &actual_y);
    let h_diff = max_abs_diff(&h_cpu, &actual_h);
    assert!(y_diff < 1e-4, "ssm_step y: max |diff| = {y_diff:.2e}");
    assert!(h_diff < 1e-4, "ssm_step h: max |diff| = {h_diff:.2e}");
}

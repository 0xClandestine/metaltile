//! End-to-end correctness test for `ffai::conv1d_causal_step` on real Metal.
//!
//! Mamba 2's depthwise causal conv streaming step. Per-channel one
//! thread; rolling state of `kernel_size - 1` past inputs gets shifted
//! up + the new input appended. Verifies:
//!   - output = bias + Σ_k w[k, d] * (state[k, d] for k < K-1, x[d] for k = K-1)
//!   - state after the call has shifted by one with x[d] at the tail
//!
//! Two iterations to make sure the state mutation lands.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::ssm::conv1d_causal_step;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Naive single-step conv: y[d] = b[d] + Σ_{k<K-1} w[k,d]*state[k,d]
///                                + w[K-1,d]*x[d].
/// Also rolls the state: state[k,d] = state[k+1,d] for k < K-2;
/// state[K-2,d] = x[d].
fn naive_conv_step(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    state: &mut [f32],
    n_channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_channels];
    let k_last = kernel_size - 1;
    for d in 0..n_channels {
        let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
        for k in 0..k_last {
            acc += w[k * n_channels + d] * state[k * n_channels + d];
        }
        y[d] = acc;
    }
    // Shift state, append x at the tail.
    for d in 0..n_channels {
        for k in 0..kernel_size - 2 {
            state[k * n_channels + d] = state[(k + 1) * n_channels + d];
        }
        state[(kernel_size - 2) * n_channels + d] = x[d];
    }
    y
}

fn run_conv_step(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    state: &[f32],
    n_channels: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), f32_slice_to_bytes(x));
    buffers.insert("w".into(), f32_slice_to_bytes(w));
    buffers.insert("b".into(), f32_slice_to_bytes(b));
    buffers.insert("state".into(), f32_slice_to_bytes(state));
    buffers.insert("y".into(), vec![0u8; n_channels * 4]);
    buffers.insert("n_channels".into(), (n_channels as u32).to_le_bytes().to_vec());
    buffers.insert("kernel_size".into(), (kernel_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = conv1d_causal_step::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id<0>() lowers to `gid.x = thread_position_in_grid.x`,
    // and dispatch_with_grid uses dispatchThreadgroups under the hood —
    // total threads = grid_groups.x * tg.x. So for N total threads we want
    // grid_groups=[1,…] tg=[N,…], not [N,…]/[N,…] which spawns N² threads
    // (each illegitimate thread races against legitimate writes via OOB
    // reads). Caught by the state diff = max(x) magnitude failure.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [n_channels, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let y = bytes_to_f32_vec(result.outputs.get("y").expect("y"));
    let state_after = bytes_to_f32_vec(result.outputs.get("state").expect("state"));
    (y, state_after)
}

#[test]
fn conv1d_causal_step_matches_naive_reference_f32() {
    // Mamba 2 production: kernel_size=4, n_channels=1536-ish. We use
    // smaller numbers so the CPU reference runs instantly + the
    // comparison stays eyeball-friendly.
    let n_channels = 64usize;
    let kernel_size = 4usize;

    let x: Vec<f32> = (0..n_channels).map(|i| (i as f32 % 7.0) * 0.1).collect();
    let w: Vec<f32> =
        (0..kernel_size * n_channels).map(|i| (((i * 13) % 11) as f32 - 5.0) * 0.05).collect();
    let b: Vec<f32> = (0..n_channels).map(|i| (i as f32 % 3.0) * 0.01).collect();
    let state: Vec<f32> =
        (0..(kernel_size - 1) * n_channels).map(|i| (((i * 17) % 9) as f32 - 4.0) * 0.02).collect();
    let mut state_cpu = state.clone();

    let expected = naive_conv_step(&x, &w, &b, &mut state_cpu, n_channels, kernel_size);
    let (actual_y, actual_state) = run_conv_step(&x, &w, &b, &state, n_channels, kernel_size);

    let y_diff = max_abs_diff(&expected, &actual_y);
    let state_diff = max_abs_diff(&state_cpu, &actual_state);
    assert!(y_diff < 1e-5, "conv1d y: max |diff| = {y_diff:.2e}");
    assert!(state_diff < 1e-5, "conv1d state: max |diff| = {state_diff:.2e}");
}

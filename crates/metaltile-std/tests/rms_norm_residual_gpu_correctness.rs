//! GPU correctness for `ffai::rms_norm_residual` — fused RMSNorm +
//! residual add: `out = residual + w * x * rsqrt(mean(x²) + eps)`.
//!
//! Pins the same `N = TPG * 4` reduction invariant as `mt_rms_norm`
//! (one threadgroup per row, `TPG = n / 4`, multiple of 32) and that
//! the residual is added *after* normalization — a regression that
//! folds residual into the SSQ, or drops it, only shows as drifting
//! logits in FFAI integration. The naive f32 oracle pins both.
//!
//! macOS-gated: needs a real Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, naive_rms_norm_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rms_norm_residual::ffai_rms_norm_residual;

fn run(
    x: &[f32],
    residual: &[f32],
    w: &[f32],
    dt: Dt,
    n: usize,
    rows: usize,
    eps: f32,
) -> Vec<f32> {
    let tpg = n / 4; // N = TPG * 4 invariant.
    assert!(n.is_multiple_of(128), "n must be multiple of 128 (kernel invariant)");
    assert!(tpg <= 1024, "n / 4 must fit in 1024 (Apple TPG cap)");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("residual".into(), pack_bytes(residual, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; rows * n], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_rms_norm_residual::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("rms_norm_residual dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(rows * n);
    out
}

/// Naive reference: RMSNorm then residual add, all in f32.
fn naive(x: &[f32], residual: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let normed = naive_rms_norm_f32(x, w, n, eps);
    normed.iter().zip(residual).map(|(&v, &r)| v + r).collect()
}

#[test]
fn rms_norm_residual_matches_naive_f32_minimum_n() {
    // n=128 → TPG=32, exactly one simdgroup (the cross-simdgroup edge).
    let (n, rows, eps) = (128usize, 1usize, 1e-5_f32);
    let x = ramp(rows * n, 17, 8.0);
    let residual = ramp(rows * n, 13, 6.0);
    let w = ramp(n, 11, 5.0).iter().map(|v| 1.0 + 0.1 * v).collect::<Vec<_>>();
    let expected = naive(&x, &residual, &w, n, eps);
    let actual = run(&x, &residual, &w, Dt::F32, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "n={n}: max |diff| = {diff:.2e}");
}

#[test]
fn rms_norm_residual_matches_naive_f32_multi_row() {
    let (n, rows, eps) = (512usize, 4usize, 1e-5_f32);
    let x = ramp(rows * n, 23, 9.0);
    let residual = ramp(rows * n, 29, 7.0);
    let w = ramp(n, 13, 6.0).iter().map(|v| 1.0 + 0.05 * v).collect::<Vec<_>>();
    let expected = naive(&x, &residual, &w, n, eps);
    let actual = run(&x, &residual, &w, Dt::F32, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "n={n} rows={rows}: max |diff| = {diff:.2e}");
}

#[test]
fn rms_norm_residual_matches_naive_f32_max_n() {
    // n=4096 → TPG=1024, the Apple threads-per-threadgroup cap.
    let (n, rows, eps) = (4096usize, 1usize, 1e-5_f32);
    let x = ramp(rows * n, 31, 10.0);
    let residual = ramp(rows * n, 37, 8.0);
    let w = ramp(n, 19, 7.0).iter().map(|v| 1.0 + 0.02 * v).collect::<Vec<_>>();
    let expected = naive(&x, &residual, &w, n, eps);
    let actual = run(&x, &residual, &w, Dt::F32, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    // Looser at n=4096 — more partial sums, more fp32 reorder noise.
    assert!(diff < 5e-4, "n={n}: max |diff| = {diff:.2e}");
}

#[test]
fn rms_norm_residual_matches_naive_f16() {
    // f16 path: inputs are f16-rounded before the oracle so the
    // comparison isolates kernel arithmetic from storage rounding.
    let (n, rows, eps) = (512usize, 2usize, 1e-5_f32);
    let x: Vec<f32> = ramp(rows * n, 23, 9.0).iter().map(|&v| Dt::F16.round(v)).collect();
    let residual: Vec<f32> = ramp(rows * n, 29, 7.0).iter().map(|&v| Dt::F16.round(v)).collect();
    let w: Vec<f32> = ramp(n, 13, 6.0).iter().map(|v| Dt::F16.round(1.0 + 0.05 * v)).collect();
    let expected = naive(&x, &residual, &w, n, eps);
    let actual = run(&x, &residual, &w, Dt::F16, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-2, "f16 n={n}: max |diff| = {diff:.2e}");
}

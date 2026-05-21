//! GPU correctness for `ffai::gated_rmsnorm` — the fused GDN post-step
//! `out = w · rmsNorm(y) · silu(z)`.
//!
//! `y` is fp32 (the Gated-DeltaNet recurrence output); the gate `z`,
//! the weight `w`, and the output are in the activation dtype `T`. The
//! oracle pins three behaviours: the fp32-in / `T`-out dtype split, the
//! `silu(z)` gate applied *after* normalization, and the same
//! `N = TPG * 4` reduction invariant as `mt_rms_norm`.
//!
//! macOS-gated: needs a real Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_rmsnorm::ffai_gated_rmsnorm;

/// Naive reference: RMSNorm of the fp32 `y` row, scaled by `w`, gated
/// by `silu(z)`. `z` / `w` are pre-rounded to the kernel's dtype by the
/// caller so this stays a pure-arithmetic oracle.
fn naive_gated_rmsnorm(y: &[f32], z: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    assert_eq!(y.len() % n, 0);
    assert_eq!(w.len(), n);
    let rows = y.len() / n;
    let mut out = vec![0.0f32; y.len()];
    for r in 0..rows {
        let base = r * n;
        let ssq: f32 = y[base..base + n].iter().map(|v| v * v).sum();
        let rms = (ssq / (n as f32) + eps).sqrt().recip();
        for d in 0..n {
            let silu_z = z[base + d] / (1.0 + (-z[base + d]).exp());
            out[base + d] = y[base + d] * rms * w[d] * silu_z;
        }
    }
    out
}

/// Dispatch the kernel. `y` is always packed f32; `z` / `w` / `out` use `dt`.
fn run(y: &[f32], z: &[f32], w: &[f32], dt: Dt, n: usize, rows: usize, eps: f32) -> Vec<f32> {
    let tpg = n / 4; // N = TPG * 4 invariant.
    assert!(n.is_multiple_of(128), "n must be multiple of 128 (kernel invariant)");
    assert!(tpg <= 1024, "n / 4 must fit in 1024 (Apple TPG cap)");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    // `y` is fp32 regardless of `T`.
    buffers.insert("y".into(), pack_bytes(y, Dt::F32));
    buffers.insert("z".into(), pack_bytes(z, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; rows * n], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_gated_rmsnorm::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("gated_rmsnorm dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(rows * n);
    out
}

#[test]
fn gated_rmsnorm_matches_naive_f32_minimum_n() {
    let _g = gpu_lock();
    // n=128 → TPG=32, exactly one simdgroup (the cross-simdgroup edge).
    let (n, rows, eps) = (128usize, 1usize, 1e-5_f32);
    let y = ramp(rows * n, 17, 8.0);
    let z = ramp(rows * n, 13, 6.0);
    let w = ramp(n, 11, 5.0).iter().map(|v| 1.0 + 0.1 * v).collect::<Vec<_>>();
    let expected = naive_gated_rmsnorm(&y, &z, &w, n, eps);
    let actual = run(&y, &z, &w, Dt::F32, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "n={n}: max |diff| = {diff:.2e}");
}

#[test]
fn gated_rmsnorm_matches_naive_f32_multi_row() {
    let _g = gpu_lock();
    let (n, rows, eps) = (512usize, 4usize, 1e-5_f32);
    let y = ramp(rows * n, 23, 9.0);
    let z = ramp(rows * n, 29, 7.0);
    let w = ramp(n, 13, 6.0).iter().map(|v| 1.0 + 0.05 * v).collect::<Vec<_>>();
    let expected = naive_gated_rmsnorm(&y, &z, &w, n, eps);
    let actual = run(&y, &z, &w, Dt::F32, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "n={n} rows={rows}: max |diff| = {diff:.2e}");
}

#[test]
fn gated_rmsnorm_matches_naive_f32_max_n() {
    let _g = gpu_lock();
    // n=4096 → TPG=1024, the Apple threads-per-threadgroup cap.
    let (n, rows, eps) = (4096usize, 1usize, 1e-5_f32);
    let y = ramp(rows * n, 31, 10.0);
    let z = ramp(rows * n, 37, 8.0);
    let w = ramp(n, 19, 7.0).iter().map(|v| 1.0 + 0.02 * v).collect::<Vec<_>>();
    let expected = naive_gated_rmsnorm(&y, &z, &w, n, eps);
    let actual = run(&y, &z, &w, Dt::F32, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    // Looser at n=4096 — more partial sums, more fp32 reorder noise.
    assert!(diff < 5e-4, "n={n}: max |diff| = {diff:.2e}");
}

#[test]
fn gated_rmsnorm_matches_naive_f16_gate_and_weight() {
    let _g = gpu_lock();
    // The realistic GDN config: fp32 `y`, f16 gate / weight / output.
    let (n, rows, eps) = (512usize, 2usize, 1e-5_f32);
    let y = ramp(rows * n, 23, 9.0); // y stays full fp32 precision.
    // z / w are f16 in the model — round before the oracle.
    let z: Vec<f32> = ramp(rows * n, 29, 7.0).iter().map(|&v| Dt::F16.round(v)).collect();
    let w: Vec<f32> = ramp(n, 13, 6.0).iter().map(|v| Dt::F16.round(1.0 + 0.05 * v)).collect();
    let expected = naive_gated_rmsnorm(&y, &z, &w, n, eps);
    let actual = run(&y, &z, &w, Dt::F16, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-2, "f16 n={n}: max |diff| = {diff:.2e}");
}

#[test]
fn gated_rmsnorm_matches_naive_bf16_gate_and_weight() {
    let _g = gpu_lock();
    let (n, rows, eps) = (512usize, 2usize, 1e-5_f32);
    let y = ramp(rows * n, 23, 9.0);
    let z: Vec<f32> = ramp(rows * n, 29, 7.0).iter().map(|&v| Dt::Bf16.round(v)).collect();
    let w: Vec<f32> = ramp(n, 13, 6.0).iter().map(|v| Dt::Bf16.round(1.0 + 0.05 * v)).collect();
    let expected = naive_gated_rmsnorm(&y, &z, &w, n, eps);
    let actual = run(&y, &z, &w, Dt::Bf16, n, rows, eps);
    let diff = max_abs_diff(&expected, &actual);
    // bf16 has a 7-bit mantissa — wider tolerance.
    assert!(diff < 8e-2, "bf16 n={n}: max |diff| = {diff:.2e}");
}

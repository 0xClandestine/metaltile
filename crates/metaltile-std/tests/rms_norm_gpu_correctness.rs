//! End-to-end correctness test for `mlx::mt_rms_norm` on real Metal.
//!
//! Pins the kernel's `N = TPG * 4` invariant: each thread owns 4
//! consecutive elements of the row, so the threadgroup width must be
//! `n / 4`. The wrapper at `FFAI/Sources/FFAI/Ops.swift:Ops.rmsNorm`
//! enforces this; without it, hardcoding `tgWidth = 256` for an
//! `n=4096` row silently miscomputes (only 1024 of 4096 elements
//! contribute to the SSQ). That bug was the precursor to the
//! 2026-05-19 SDPA-decode GPU freeze — same class, different failure
//! mode. This test pins the correct dispatch shape so future codegen
//! changes can't silently regress it.
//!
//! Single-row test (n=128 — the minimum legal n, exercising the
//! `TPG = 32` simdgroup-edge path) and a 4-row test at n=512 to
//! exercise multi-row dispatch.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, naive_rms_norm_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::mt_rms_norm;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

fn run_rms_norm(x: &[f32], w: &[f32], n: usize, rows: usize, eps: f32) -> Vec<f32> {
    let tpg = n / 4; // N = TPG * 4 invariant.
    assert!(n.is_multiple_of(128), "n must be multiple of 128 (kernel invariant)");
    assert!(tpg <= 1024, "n / 4 must fit in 1024 (Apple TPG cap)");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), f32_slice_to_bytes(x));
    buffers.insert("w".into(), f32_slice_to_bytes(w));
    buffers.insert("out".into(), vec![0u8; x.len() * 4]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    // `kernel_ir_for` returns Elementwise mode by default; rms_norm
    // is reduction-mode (uses `reduce_sum`), same pattern as the
    // sdpa_decode test.
    let mut kernel = mt_rms_norm::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per row, `tpg` threads per group.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

#[test]
fn rms_norm_matches_naive_cpu_reference_f32_minimum_n() {
    // n=128 = 32 simdgroup lanes × 4 elements/lane. The smallest legal
    // n: TPG = 32, exactly one full simdgroup. Anything smaller (n=64
    // → TPG=16) would silently produce tg_ssq=0 because the cross-
    // simdgroup combine reads `n_simd = TPG/32 = 0` slots.
    let n = 128usize;
    let rows = 1usize;
    let eps = 1e-5_f32;

    let x = ramp(rows * n, 17, 8.0);
    let w = ramp(n, 11, 5.0).iter().map(|v| 1.0 + 0.1 * v).collect::<Vec<_>>();
    let expected = naive_rms_norm_f32(&x, &w, n, eps);
    let actual = run_rms_norm(&x, &w, n, rows, eps);

    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "rms_norm n={n} rows={rows} max |diff| = {diff:.2e} (expected < 1e-4)",);
}

#[test]
fn rms_norm_matches_naive_cpu_reference_f32_multi_row() {
    // 4 rows of width 512 — exercises the multi-row dispatch path
    // (`grid = (rows, 1, 1)`) and TPG = 128 (4 simdgroups), the most
    // common production size for small models.
    let n = 512usize;
    let rows = 4usize;
    let eps = 1e-5_f32;

    let x = ramp(rows * n, 23, 9.0);
    let w = ramp(n, 13, 6.0).iter().map(|v| 1.0 + 0.05 * v).collect::<Vec<_>>();
    let expected = naive_rms_norm_f32(&x, &w, n, eps);
    let actual = run_rms_norm(&x, &w, n, rows, eps);

    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "rms_norm n={n} rows={rows} max |diff| = {diff:.2e} (expected < 1e-4)",);
}

#[test]
fn rms_norm_matches_naive_cpu_reference_f32_max_n() {
    // n=4096 — the maximum legal n (TPG = 1024 = Apple TPG cap).
    // Llama-class hidden dim; this is what production touches.
    let n = 4096usize;
    let rows = 1usize;
    let eps = 1e-5_f32;

    let x = ramp(rows * n, 31, 10.0);
    let w = ramp(n, 19, 7.0).iter().map(|v| 1.0 + 0.02 * v).collect::<Vec<_>>();
    let expected = naive_rms_norm_f32(&x, &w, n, eps);
    let actual = run_rms_norm(&x, &w, n, rows, eps);

    let diff = max_abs_diff(&expected, &actual);
    // Slightly looser tolerance at n=4096 — more partial-sums means
    // more fp32 reordering noise from `reduce_sum`.
    assert!(diff < 5e-4, "rms_norm n={n} rows={rows} max |diff| = {diff:.2e} (expected < 5e-4)",);
}

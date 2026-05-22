//! End-to-end correctness for `mlx::mt_rms_norm_wide` — the wide-row
//! RMSNorm variant for rows past the 4096-element cap of `mt_rms_norm`.
//!
//! `mt_rms_norm` fixes `N = TPG * 4`, so a 1024-thread group tops out
//! at 4096. `mt_rms_norm_wide` has each thread stride over the row by
//! one full threadgroup, so any `n` is covered. Needed for
//! large-hidden models — Gemma 4 31B is hidden 5376.
//!
//! Dispatched at a fixed TPG = 1024 (the wrapper's choice); the kernel
//! derives its stride from `n_simd`, so the row width and the
//! threadgroup width are decoupled. macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, naive_rms_norm_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::mt_rms_norm_wide;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Run `mt_rms_norm_wide` over `rows` rows of width `n` at TPG = 1024.
fn run_rms_norm_wide(x: &[f32], w: &[f32], n: usize, rows: usize, eps: f32) -> Vec<f32> {
    const TPG: usize = 1024;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), f32_slice_to_bytes(x));
    buffers.insert("w".into(), f32_slice_to_bytes(w));
    buffers.insert("out".into(), vec![0u8; x.len() * 4]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_rms_norm_wide::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per row; the kernel strides over the row, so the
    // threadgroup width is independent of `n`.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [TPG, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

#[test]
fn rms_norm_wide_matches_cpu_reference_gemma4_31b_hidden() {
    // n = 5376 — Gemma 4 31B's hidden size, and the row width that
    // motivated this kernel. 5376 = 5·1024 + 256, so threads 0..255
    // run 6 strided iterations and 256..1023 run 5: the uneven tail
    // is exercised.
    let n = 5376usize;
    let rows = 1usize;
    let eps = 1e-6_f32;

    let x = ramp(rows * n, 37, 11.0);
    let w = ramp(n, 23, 8.0).iter().map(|v| 1.0 + 0.02 * v).collect::<Vec<_>>();
    let expected = naive_rms_norm_f32(&x, &w, n, eps);
    let actual = run_rms_norm_wide(&x, &w, n, rows, eps);

    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < 5e-4,
        "rms_norm_wide n={n} rows={rows} max |diff| = {diff:.2e} (expected < 5e-4)",
    );
}

#[test]
fn rms_norm_wide_matches_cpu_reference_multi_row() {
    // 3 rows of width 5376 — exercises the multi-row dispatch
    // (`grid = (rows, 1, 1)`) at the wide width.
    let n = 5376usize;
    let rows = 3usize;
    let eps = 1e-6_f32;

    let x = ramp(rows * n, 41, 9.0);
    let w = ramp(n, 29, 7.0).iter().map(|v| 1.0 + 0.015 * v).collect::<Vec<_>>();
    let expected = naive_rms_norm_f32(&x, &w, n, eps);
    let actual = run_rms_norm_wide(&x, &w, n, rows, eps);

    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < 5e-4,
        "rms_norm_wide n={n} rows={rows} max |diff| = {diff:.2e} (expected < 5e-4)",
    );
}

#[test]
fn rms_norm_wide_matches_cpu_reference_exact_tpg_multiple() {
    // n = 8192 = 8·1024 — every thread runs exactly 8 iterations, no
    // tail. A larger width than any current model hidden, confirming
    // the kernel has no upper bound.
    let n = 8192usize;
    let rows = 1usize;
    let eps = 1e-6_f32;

    let x = ramp(rows * n, 53, 12.0);
    let w = ramp(n, 31, 6.0).iter().map(|v| 1.0 + 0.01 * v).collect::<Vec<_>>();
    let expected = naive_rms_norm_f32(&x, &w, n, eps);
    let actual = run_rms_norm_wide(&x, &w, n, rows, eps);

    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < 5e-4,
        "rms_norm_wide n={n} rows={rows} max |diff| = {diff:.2e} (expected < 5e-4)",
    );
}

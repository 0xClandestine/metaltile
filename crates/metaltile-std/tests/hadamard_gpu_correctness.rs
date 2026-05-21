//! GPU correctness for `mlx::hadamard` — the order-N Walsh–Hadamard
//! transform.
//!
//! The oracle multiplies by the explicit Hadamard matrix
//! `H[i][j] = (-1)^popcount(i & j)` — an algorithm-independent
//! reference for the kernel's fast butterfly form.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::mlx::hadamard::{mt_hadamard_n64, mt_hadamard_n128, mt_hadamard_n256};

/// `y[i] = scale · Σ_j (-1)^popcount(i&j) · x[j]` per row.
fn naive_hadamard(x: &[f32], rows: usize, n: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * n];
    for r in 0..rows {
        for i in 0..n {
            let mut acc = 0.0_f32;
            for (j, &xj) in x[r * n..(r + 1) * n].iter().enumerate() {
                acc += if (i & j).count_ones() % 2 == 0 { xj } else { -xj };
            }
            out[r * n + i] = acc * scale;
        }
    }
    out
}

fn run(
    kernel_ir: fn(DType) -> Kernel,
    x: &[f32],
    dt: Dt,
    rows: usize,
    n: usize,
    scale: f32,
) -> Vec<f32> {
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("inp".into(), pack_bytes(x, dt));
    b.insert("out".into(), pack_bytes(&vec![0.0; rows * n], dt));
    b.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [rows, 1, 1], [n, 1, 1])
        .expect("hadamard dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(rows * n);
    out
}

fn ramp(rows: usize, n: usize) -> Vec<f32> {
    (0..rows * n).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect()
}

#[test]
fn hadamard_n64_matches_matrix_f32() {
    let _g = gpu_lock();
    let (rows, n) = (3, 64);
    let x = ramp(rows, n);
    let scale = 0.125; // 1/sqrt(64)
    let expected = naive_hadamard(&x, rows, n, scale);
    let actual = run(mt_hadamard_n64::kernel_ir_for, &x, Dt::F32, rows, n, scale);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "n64 f32 mismatch");
}

#[test]
fn hadamard_n128_matches_matrix_f32() {
    let _g = gpu_lock();
    let (rows, n) = (4, 128);
    let x = ramp(rows, n);
    let expected = naive_hadamard(&x, rows, n, 1.0);
    let actual = run(mt_hadamard_n128::kernel_ir_for, &x, Dt::F32, rows, n, 1.0);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "n128 f32 mismatch");
}

#[test]
fn hadamard_n256_matches_matrix_f32() {
    let _g = gpu_lock();
    let (rows, n) = (2, 256);
    let x = ramp(rows, n);
    let expected = naive_hadamard(&x, rows, n, 0.0625);
    let actual = run(mt_hadamard_n256::kernel_ir_for, &x, Dt::F32, rows, n, 0.0625);
    assert!(max_abs_diff(&actual, &expected) < 2e-4, "n256 f32 mismatch");
}

#[test]
fn hadamard_n64_f16() {
    let _g = gpu_lock();
    let (rows, n) = (2, 64);
    // Small inputs so the f16 accumulation stays in range.
    let x: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let expected = naive_hadamard(&x, rows, n, 0.125);
    let actual = run(mt_hadamard_n64::kernel_ir_for, &x, Dt::F16, rows, n, 0.125);
    assert!(max_abs_diff(&actual, &expected) < 1e-2, "n64 f16 mismatch");
}

#[test]
fn hadamard_n128_bf16() {
    let _g = gpu_lock();
    let (rows, n) = (2, 128);
    let x: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let expected = naive_hadamard(&x, rows, n, 1.0 / 128.0_f32.sqrt());
    let actual =
        run(mt_hadamard_n128::kernel_ir_for, &x, Dt::Bf16, rows, n, 1.0 / 128.0_f32.sqrt());
    assert!(max_abs_diff(&actual, &expected) < 5e-2, "n128 bf16 mismatch");
}

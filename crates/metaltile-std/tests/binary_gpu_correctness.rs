//! GPU correctness for `mlx::binary` — elementwise binary ops
//! (add, mul, sub, div, max, min, pow, atan2, remainder, logaddexp).
//!
//! Each test: synthetic ramp inputs → naive f32 CPU reference →
//! dispatch via Context::dispatch_with_grid → assert max_abs_diff.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::Kernel};
use metaltile_runtime::Context;
use metaltile_std::mlx::binary::{
    mt_atan2,
    mt_div,
    mt_logaddexp,
    mt_max_elem,
    mt_min_elem,
    mt_mul,
    mt_pow,
    mt_sub,
};

// Dispatch a two-input, one-output binary kernel (Grid3D — one thread per element).
fn run_binary(kernel_ir: fn(DType) -> Kernel, a: &[f32], b: &[f32], dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = kernel_ir(dt.to_dtype());
    // binary ops are Grid3D (one thread per output element).
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("binary dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

// `vector_add` (metaltile's flagship example kernel) names its output
// param `c`, not `out`, so it does not flow through the shared
// `run_binary` helper — a correct dedicated test is a tracked follow-up.

// --- mt_mul ---

#[test]
fn binary_mul_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.04 - 0.25).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    let actual = run_binary(mt_mul::kernel_ir_for, &a, &b, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-5, "mul f32 mismatch");
}

// --- mt_sub ---

#[test]
fn binary_sub_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.07 - 0.6).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.05 - 0.3).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let actual = run_binary(mt_sub::kernel_ir_for, &a, &b, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-5, "sub f32 mismatch");
}

// --- mt_div ---

#[test]
fn binary_div_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    // Avoid near-zero denominator: shift b away from 0.
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.06 - 0.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.08 + 0.2).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
    let actual = run_binary(mt_div::kernel_ir_for, &a, &b, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "div f32 mismatch");
}

// --- mt_max_elem ---

#[test]
fn binary_max_elem_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect();
    let actual = run_binary(mt_max_elem::kernel_ir_for, &a, &b, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-5, "max_elem f32 mismatch");
}

// --- mt_min_elem ---

#[test]
fn binary_min_elem_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect();
    let actual = run_binary(mt_min_elem::kernel_ir_for, &a, &b, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-5, "min_elem f32 mismatch");
}

// --- mt_pow ---

#[test]
fn binary_pow_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 256usize;
    // Keep base positive to avoid complex-valued results.
    let a: Vec<f32> = (0..n).map(|i| (i % 9) as f32 * 0.1 + 0.2).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.4 + 0.2).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x.powf(*y)).collect();
    let actual = run_binary(mt_pow::kernel_ir_for, &a, &b, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "pow f32 mismatch");
}

// --- mt_atan2 ---

#[test]
fn binary_atan2_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let y_vals: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let x_vals: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let expected: Vec<f32> = y_vals.iter().zip(x_vals.iter()).map(|(y, x)| y.atan2(*x)).collect();
    // Kernel arg order: y, x, out — matches the fn signature mt_atan2(y, x, out).
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("y".into(), pack_bytes(&y_vals, Dt::F32));
    buffers.insert("x".into(), pack_bytes(&x_vals, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
    let ctx = Context::new().expect("Context::new");
    let kernel = mt_atan2::kernel_ir_for(DType::F32);
    let tpg = 256;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("atan2 dispatch");
    let actual: Vec<f32> = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32)
        .into_iter()
        .take(n)
        .collect();
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "atan2 f32 mismatch");
}

// mt_remainder: floored-vs-truncated division semantics need
// reconciling against the MLX reference — a correct test is a tracked
// follow-up.

// --- mt_logaddexp ---

#[test]
fn binary_logaddexp_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.3 - 1.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.4 - 1.0).collect();
    let expected: Vec<f32> =
        a.iter().zip(b.iter()).map(|(x, y)| (x.exp() + y.exp()).ln()).collect();
    let actual = run_binary(mt_logaddexp::kernel_ir_for, &a, &b, Dt::F32, n);
    // logaddexp f32 compounds two exp + ln; tol widened to 1e-3.
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "logaddexp f32 mismatch");
}

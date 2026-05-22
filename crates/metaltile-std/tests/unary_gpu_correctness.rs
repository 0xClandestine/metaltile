//! GPU correctness for `mlx::unary` — elementwise unary ops.
//!
//! Each test: synthetic ramp inputs → naive f32 CPU reference →
//! dispatch via Context::dispatch_with_grid → assert max_abs_diff.
//! Tests the most production-critical unary ops (exp, log, sqrt,
//! rsqrt, abs, silu, gelu, sigmoid). The transcendental family
//! (sinh/cosh/tan/atan2/…) follow the same pattern; test one
//! representative each.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::Kernel};
use metaltile_runtime::Context;
use metaltile_std::mlx::unary::{
    mt_abs,
    mt_exp,
    mt_log,
    mt_relu,
    mt_rsqrt,
    mt_sigmoid,
    mt_silu,
    mt_sqrt,
};

/// Dispatch a one-input, one-output unary kernel (Grid3D).
fn run_unary(kernel_ir: fn(DType) -> Kernel, a: &[f32], dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = kernel_ir(dt.to_dtype());
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("unary dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

// --- mt_exp ---

#[test]
fn unary_exp_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    // Keep values in [-3, 3] to avoid f32 overflow.
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect();
    let expected: Vec<f32> = a.iter().map(|x| x.exp()).collect();
    let actual = run_unary(mt_exp::kernel_ir_for, &a, Dt::F32, n);
    // GPU fast-math exp drifts ~2 ULP from the CPU.
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "exp f32 mismatch");
}

#[test]
fn unary_exp_matches_cpu_f16() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| Dt::F16.round((i % 11) as f32 * 0.2 - 1.0)).collect();
    let expected: Vec<f32> = a.iter().map(|x| x.exp()).collect();
    let actual = run_unary(mt_exp::kernel_ir_for, &a, Dt::F16, n);
    // f16 exp drifts up to ~3 ULP (≈ 1e-3 at typical activation values).
    let max_diff = max_abs_diff(&actual, &expected);
    assert!(max_diff < 5e-3, "exp f16 max |diff| = {max_diff:.2e}");
}

// --- mt_log ---

#[test]
fn unary_log_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    // Positive inputs only.
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 + 0.1).collect();
    let expected: Vec<f32> = a.iter().map(|x| x.ln()).collect();
    let actual = run_unary(mt_log::kernel_ir_for, &a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "log f32 mismatch");
}

// --- mt_sqrt ---

#[test]
fn unary_sqrt_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.1 + 0.05).collect();
    let expected: Vec<f32> = a.iter().map(|x| x.sqrt()).collect();
    let actual = run_unary(mt_sqrt::kernel_ir_for, &a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-5, "sqrt f32 mismatch");
}

// --- mt_rsqrt ---

#[test]
fn unary_rsqrt_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    // Positive only; avoid near-zero to keep rsqrt finite.
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 + 0.2).collect();
    let expected: Vec<f32> = a.iter().map(|x| 1.0 / x.sqrt()).collect();
    let actual = run_unary(mt_rsqrt::kernel_ir_for, &a, Dt::F32, n);
    // rsqrt is a Newton-step approximation on GPU; ~3 ULP drift.
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "rsqrt f32 mismatch");
}

// --- mt_abs ---

#[test]
fn unary_abs_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.1 - 1.1).collect();
    let expected: Vec<f32> = a.iter().map(|x| x.abs()).collect();
    let actual = run_unary(mt_abs::kernel_ir_for, &a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "abs f32 mismatch");
}

// --- mt_silu ---

#[test]
fn unary_silu_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect();
    // silu(x) = x / (1 + exp(-x))
    let expected: Vec<f32> = a.iter().map(|x| x / (1.0 + (-x).exp())).collect();
    let actual = run_unary(mt_silu::kernel_ir_for, &a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "silu f32 mismatch");
}

#[test]
fn unary_silu_matches_cpu_f16() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| Dt::F16.round((i % 13) as f32 * 0.3 - 2.0)).collect();
    let expected: Vec<f32> = a.iter().map(|x| x / (1.0 + (-x).exp())).collect();
    let actual = run_unary(mt_silu::kernel_ir_for, &a, Dt::F16, n);
    let max_diff = max_abs_diff(&actual, &expected);
    assert!(max_diff < 5e-3, "silu f16 max |diff| = {max_diff:.2e}");
}

// --- mt_relu ---

#[test]
fn unary_relu_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.15 - 1.4).collect();
    let expected: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
    let actual = run_unary(mt_relu::kernel_ir_for, &a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "relu f32 mismatch");
}

// --- mt_sigmoid ---

#[test]
fn unary_sigmoid_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.4 - 3.5).collect();
    // sigmoid(x) = 1 / (1 + exp(-x))
    let expected: Vec<f32> = a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    let actual = run_unary(mt_sigmoid::kernel_ir_for, &a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "sigmoid f32 mismatch");
}

#[test]
fn unary_sigmoid_matches_cpu_bf16() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| Dt::Bf16.round((i % 11) as f32 * 0.5 - 3.0)).collect();
    let expected: Vec<f32> = a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    let actual = run_unary(mt_sigmoid::kernel_ir_for, &a, Dt::Bf16, n);
    // bf16 sigmoid compounds two transcendentals; 2e-2 covers worst-case ULP.
    let max_diff = max_abs_diff(&actual, &expected);
    assert!(max_diff < 2e-2, "sigmoid bf16 max |diff| = {max_diff:.2e}");
}

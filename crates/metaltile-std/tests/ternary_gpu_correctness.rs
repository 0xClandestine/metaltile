//! GPU correctness for `mlx::ternary` — elementwise select kernel.
//!
//! Verifies `mt_select<T>`: `out[i] = cond[i] != 0 ? on_true[i] : on_false[i]`.
//! Tests f32/f16/bf16 dtypes; covers all-true, all-false, and mixed-condition
//! patterns.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::Kernel;
use metaltile_runtime::Context;
use metaltile_std::mlx::ternary::mt_select;

fn run_select(on_true: &[f32], on_false: &[f32], cond: &[u8], dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("cond".into(), cond.to_vec());
    buffers.insert("on_true".into(), pack_bytes(on_true, dt));
    buffers.insert("on_false".into(), pack_bytes(on_false, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel: Kernel = mt_select::kernel_ir_for(dt.to_dtype());
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("select dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

fn cpu_select_reference(on_true: &[f32], on_false: &[f32], cond: &[u8]) -> Vec<f32> {
    cond.iter()
        .zip(on_true.iter().zip(on_false.iter()))
        .map(|(&c, (&t, &f))| if c != 0 { t } else { f })
        .collect()
}

#[test]
fn ternary_select_mixed_condition_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let on_true: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
    let on_false: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.03).collect();
    // Alternate true/false every element.
    let cond: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let expected = cpu_select_reference(&on_true, &on_false, &cond);
    let actual = run_select(&on_true, &on_false, &cond, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "select mixed f32 mismatch");
}

#[test]
fn ternary_select_all_true_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let on_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 1.0).collect();
    let on_false: Vec<f32> = vec![99.0f32; n];
    let cond: Vec<u8> = vec![1u8; n];
    let expected = on_true.clone();
    let actual = run_select(&on_true, &on_false, &cond, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "select all_true f32 mismatch");
}

#[test]
fn ternary_select_all_false_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    let on_true: Vec<f32> = vec![99.0f32; n];
    let on_false: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 2.0).collect();
    let cond: Vec<u8> = vec![0u8; n];
    let expected = on_false.clone();
    let actual = run_select(&on_true, &on_false, &cond, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "select all_false f32 mismatch");
}

#[test]
fn ternary_select_mixed_condition_f16() {
    let _g = gpu_lock();
    let n = 512usize;
    let on_true: Vec<f32> = (0..n).map(|i| Dt::F16.round((i % 13) as f32 * 0.1 - 0.5)).collect();
    let on_false: Vec<f32> = (0..n).map(|i| Dt::F16.round((i % 11) as f32 * 0.2 - 1.0)).collect();
    let cond: Vec<u8> = (0..n).map(|i| (i % 3 != 0) as u8).collect();
    let expected = cpu_select_reference(&on_true, &on_false, &cond);
    let actual = run_select(&on_true, &on_false, &cond, Dt::F16, n);
    // select is an exact copy — zero tolerance except f16 round-trip.
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "select mixed f16 mismatch");
}

#[test]
fn ternary_select_mixed_condition_bf16() {
    let _g = gpu_lock();
    let n = 256usize;
    let on_true: Vec<f32> = (0..n).map(|i| Dt::Bf16.round((i % 7) as f32 * 0.3 - 1.0)).collect();
    let on_false: Vec<f32> = (0..n).map(|i| Dt::Bf16.round((i % 9) as f32 * 0.2 - 0.8)).collect();
    let cond: Vec<u8> = (0..n).map(|i| (i % 4 < 2) as u8).collect();
    let expected = cpu_select_reference(&on_true, &on_false, &cond);
    let actual = run_select(&on_true, &on_false, &cond, Dt::Bf16, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-2, "select mixed bf16 mismatch");
}

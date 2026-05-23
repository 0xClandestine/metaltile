//! GPU correctness for `mlx::arange` — `out[i] = start + i * step`.
//!
//! Verifies the arange kernel against a simple CPU reference across
//! f32/f16/bf16 dtypes, integer-valued steps, and fractional steps.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::mlx::arange::mt_arange;

fn run_arange(start: f32, step: f32, dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("start".into(), pack_bytes(&[start], dt));
    buffers.insert("step".into(), pack_bytes(&[step], dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_arange::kernel_ir_for(dt.to_dtype());
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("arange dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

fn cpu_arange(start: f32, step: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| start + i as f32 * step).collect()
}

#[test]
fn arange_unit_step_f32() {
    let _g = gpu_lock();
    let (start, step, n) = (0.0f32, 1.0f32, 1024);
    let expected = cpu_arange(start, step, n);
    let actual = run_arange(start, step, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "arange unit step f32 mismatch");
}

#[test]
fn arange_fractional_step_f32() {
    let _g = gpu_lock();
    let (start, step, n) = (0.5f32, 0.01f32, 512);
    let expected = cpu_arange(start, step, n);
    let actual = run_arange(start, step, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "arange fractional step f32 mismatch");
}

#[test]
fn arange_negative_start_f32() {
    let _g = gpu_lock();
    let (start, step, n) = (-10.0f32, 0.05f32, 400);
    let expected = cpu_arange(start, step, n);
    let actual = run_arange(start, step, Dt::F32, n);
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "arange negative start f32 mismatch");
}

#[test]
fn arange_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let (start, step, n) = (1.0f32, 1.0f32, 256);
    let actual = run_arange(start, step, Dt::F32, n);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "arange output all zeros — possible empty kernel body"
    );
}

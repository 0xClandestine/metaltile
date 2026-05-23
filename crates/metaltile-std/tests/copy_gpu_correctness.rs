//! GPU correctness for `mlx::copy` — contiguous copy kernel.
//!
//! Verifies `mt_copy<T>`: output must exactly equal input for all
//! supported dtypes (f32/f16/bf16). The copy kernel is a pure
//! load+store with no arithmetic — tolerance is effectively zero.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::mlx::copy::mt_copy;

fn run_copy(a: &[f32], dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_copy::kernel_ir_for(dt.to_dtype());
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("copy dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

#[test]
fn copy_exact_match_f32() {
    let _g = gpu_lock();
    let n = 4096usize;
    let a = ramp(n, 23, 11.0);
    let actual = run_copy(&a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &a) < 1e-6, "copy f32: output differs from input");
}

#[test]
fn copy_exact_match_f16() {
    let _g = gpu_lock();
    let n = 2048usize;
    // Round through f16 first so the oracle matches what the GPU reads.
    let a: Vec<f32> = ramp(n, 17, 8.0).iter().map(|&v| Dt::F16.round(v)).collect();
    let actual = run_copy(&a, Dt::F16, n);
    assert!(max_abs_diff(&actual, &a) < 1e-3, "copy f16: output differs from input");
}

#[test]
fn copy_exact_match_bf16() {
    let _g = gpu_lock();
    let n = 2048usize;
    let a: Vec<f32> = ramp(n, 13, 6.0).iter().map(|&v| Dt::Bf16.round(v)).collect();
    let actual = run_copy(&a, Dt::Bf16, n);
    assert!(max_abs_diff(&actual, &a) < 1e-2, "copy bf16: output differs from input");
}

#[test]
fn copy_large_n_f32() {
    let _g = gpu_lock();
    // 1M elements — exercises the multi-threadgroup path.
    let n = 1 << 20;
    let a = ramp(n, 31, 15.0);
    let actual = run_copy(&a, Dt::F32, n);
    assert!(max_abs_diff(&actual, &a) < 1e-6, "copy large n f32 mismatch");
}

#[test]
fn copy_output_is_not_all_zeros_f32() {
    let _g = gpu_lock();
    // Smoke test: the kernel must produce non-zero output for non-zero input.
    let n = 256usize;
    let a: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
    let actual = run_copy(&a, Dt::F32, n);
    assert!(actual.iter().any(|&v| v != 0.0), "copy output is all zeros — empty kernel body?");
    assert!(max_abs_diff(&actual, &a) < 1e-6, "copy f32 non-zero mismatch");
}

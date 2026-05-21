//! GPU correctness for `mlx::binary_two` — fused two-output elementwise.
//!
//! Verifies `mt_binary_two<T>`: simultaneously computes `c = a + b` and
//! `d = a * b` in a single launch, saving one read of `a` and `b` versus
//! two separate kernels. The CPU oracle computes both outputs independently.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::mlx::binary_two::mt_binary_two;

fn run_binary_two(a: &[f32], b: &[f32], dt: Dt, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("c".into(), pack_bytes(&vec![0.0f32; n], dt));
    buffers.insert("d".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_binary_two::kernel_ir_for(dt.to_dtype());
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("binary_two dispatch");

    let c = {
        let mut v = unpack_bytes(result.outputs.get("c").expect("c"), dt);
        v.truncate(n);
        v
    };
    let d = {
        let mut v = unpack_bytes(result.outputs.get("d").expect("d"), dt);
        v.truncate(n);
        v
    };
    (c, d)
}

#[test]
fn binary_two_add_mul_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();

    let expected_c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let expected_d: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    let (actual_c, actual_d) = run_binary_two(&a, &b, Dt::F32, n);

    assert!(max_abs_diff(&actual_c, &expected_c) < 1e-5, "binary_two c (add) f32 mismatch");
    assert!(max_abs_diff(&actual_d, &expected_d) < 1e-5, "binary_two d (mul) f32 mismatch");
}

#[test]
fn binary_two_add_mul_f16() {
    let _g = gpu_lock();
    let n = 512usize;
    let a: Vec<f32> = (0..n).map(|i| Dt::F16.round((i % 13) as f32 * 0.1 - 0.5)).collect();
    let b: Vec<f32> = (0..n).map(|i| Dt::F16.round((i % 11) as f32 * 0.08 - 0.4)).collect();

    let expected_c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let expected_d: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    let (actual_c, actual_d) = run_binary_two(&a, &b, Dt::F16, n);

    assert!(max_abs_diff(&actual_c, &expected_c) < 1e-3, "binary_two c (add) f16 mismatch");
    assert!(max_abs_diff(&actual_d, &expected_d) < 1e-3, "binary_two d (mul) f16 mismatch");
}

#[test]
fn binary_two_add_mul_bf16() {
    let _g = gpu_lock();
    let n = 256usize;
    let a: Vec<f32> = (0..n).map(|i| Dt::Bf16.round((i % 11) as f32 * 0.12 - 0.6)).collect();
    let b: Vec<f32> = (0..n).map(|i| Dt::Bf16.round((i % 7) as f32 * 0.1 - 0.3)).collect();

    let expected_c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let expected_d: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    let (actual_c, actual_d) = run_binary_two(&a, &b, Dt::Bf16, n);

    // bf16 has 7-bit mantissa; wider tolerance.
    assert!(max_abs_diff(&actual_c, &expected_c) < 1e-2, "binary_two c (add) bf16 mismatch");
    assert!(max_abs_diff(&actual_d, &expected_d) < 1e-2, "binary_two d (mul) bf16 mismatch");
}

#[test]
fn binary_two_outputs_not_all_zeros_f32() {
    let _g = gpu_lock();
    let n = 64usize;
    let a: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=n as u32).map(|i| i as f32 * 2.0).collect();
    let (c, d) = run_binary_two(&a, &b, Dt::F32, n);
    assert!(c.iter().any(|&v| v != 0.0), "binary_two c all zeros");
    assert!(d.iter().any(|&v| v != 0.0), "binary_two d all zeros");
}

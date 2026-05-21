//! GPU correctness for `mlx::binary::vector_add<T>`.
//!
//! `vector_add` names its output parameter `c` (not `out`), unlike the other
//! binary kernels that use `out`. A previous test skeleton got this wrong.
//! This file pins the correct buffer-key contract and verifies arithmetic.
//!
//! Three cases:
//!   - f32: elementwise a + b, exact match within 1e-5
//!   - f16: ramp inputs rounded through f16 precision
//!   - Non-zero smoke: output must not be all zeros
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::binary::vector_add;

/// Dispatch `vector_add<T>` over `n` elements.
///
/// Buffer map: `a`, `b` (inputs), `c` (output — the kernel's third param name).
fn run_vector_add(a: &[f32], b: &[f32], dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    // Output key must be "c" — that is the kernel param name, not "out".
    buffers.insert("c".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = vector_add::kernel_ir_for(dt.to_dtype());

    // Grid3D: one thread per output element.
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("vector_add dispatch");

    let mut out = unpack_bytes(result.outputs.get("c").expect("c output buffer"), dt);
    out.truncate(n);
    out
}

#[test]
fn vector_add_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let a = ramp(n, 19, 9.0);
    let b = ramp(n, 13, 6.0);
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let actual = run_vector_add(&a, &b, Dt::F32, n);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "vector_add f32: max |diff| = {diff:.2e} > 1e-5");
}

#[test]
fn vector_add_matches_cpu_f16() {
    let _g = gpu_lock();
    let n = 512usize;
    // Round through f16 so the oracle uses the same precision the kernel sees.
    let a: Vec<f32> = ramp(n, 17, 8.0).iter().map(|&v| Dt::F16.round(v)).collect();
    let b: Vec<f32> = ramp(n, 11, 5.0).iter().map(|&v| Dt::F16.round(v)).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let actual = run_vector_add(&a, &b, Dt::F16, n);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-3, "vector_add f16: max |diff| = {diff:.2e} > 1e-3");
}

#[test]
fn vector_add_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    // Smoke: a non-zero input must produce a non-zero output. Guards against
    // the empty-kernel-body class of regression (all-zeros output is the tell).
    let n = 64usize;
    let a: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![1.0f32; n];
    let actual = run_vector_add(&a, &b, Dt::F32, n);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "vector_add output is all zeros — empty kernel body?",
    );
}

#[test]
fn vector_add_correct_dtype_f32() {
    let _g = gpu_lock();
    // Verify that kernel_ir_for picks up the DType::F32 path correctly and
    // not a zero default — the kernel is generic <T>, so the dtype must flow.
    let n = 256usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.2).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    // Dispatch the same kernel via the known f32 dtype.
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(&a, Dt::F32));
    buffers.insert("b".into(), pack_bytes(&b, Dt::F32));
    buffers.insert("c".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
    let ctx = Context::new().expect("Context::new");
    let kernel = vector_add::kernel_ir_for(DType::F32);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [n, 1, 1])
        .expect("dispatch");
    let actual = unpack_bytes(result.outputs.get("c").expect("c"), Dt::F32);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "vector_add f32 dtype: max |diff| = {diff:.2e}");
}

#[test]
#[ignore = "perf bench — run with --ignored --nocapture"]
fn vector_add_perf_bench_f32() {
    use std::time::Instant;
    let _g = gpu_lock();
    let n = 1 << 22; // 4 M elements
    let a = ramp(n, 31, 15.0);
    let b = ramp(n, 23, 11.0);
    let ctx = Context::new().expect("Context::new");
    let kernel = vector_add::kernel_ir_for(DType::F32);

    let tpg = 256;
    let groups = n.div_ceil(tpg);

    // Warmup
    for _ in 0..5 {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("a".into(), pack_bytes(&a, Dt::F32));
        buffers.insert("b".into(), pack_bytes(&b, Dt::F32));
        buffers.insert("c".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("warmup");
    }

    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("a".into(), pack_bytes(&a, Dt::F32));
        buffers.insert("b".into(), pack_bytes(&b, Dt::F32));
        buffers.insert("c".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("bench iter");
    }
    let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
    let bytes = n as f64 * 4.0 * 3.0; // 2 reads + 1 write
    let gb_s = bytes / elapsed_us / 1e3;
    println!("vector_add f32 N={n}: {elapsed_us:.1} µs  |  {gb_s:.1} GB/s");
}

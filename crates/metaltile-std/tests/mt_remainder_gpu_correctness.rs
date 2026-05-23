//! GPU correctness for `mlx::binary::mt_remainder<T>`.
//!
//! The DSL `remainder(a, b)` maps to MSL `fmod(a, b)`, which is the
//! **truncated-toward-zero** remainder — the same as C's `fmod` and Rust's
//! `%` operator for floats. This is distinct from the *floored* remainder
//! (Python's `%` / `math.fmod`): for negative `a` and positive `b`:
//!
//!   fmod(-7.0, 3.0) = -1.0   (truncated — sign follows dividend)
//!   floor(-7.0 % 3.0) = 2.0  (floored — always non-negative for positive b)
//!
//! This file verifies the truncated-division semantics and catches a
//! zero-tolerance-widening regression where the test was silenced to accept
//! floored results.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::binary::mt_remainder;

/// CPU oracle: truncated-division remainder — mirrors MSL `fmod(a, b)`.
/// Rust's `%` operator for f32 implements this semantics.
fn oracle_remainder(a: f32, b: f32) -> f32 { a % b }

/// Dispatch `mt_remainder<f32>` over `n` elements.
fn run_remainder(a: &[f32], b: &[f32], dt: Dt, n: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_remainder::kernel_ir_for(dt.to_dtype());
    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("remainder dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

#[test]
fn remainder_positive_inputs_matches_oracle_f32() {
    let _g = gpu_lock();
    let n = 512usize;
    // Both inputs positive — fmod and floored remainder agree here.
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.9 + 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.4 + 0.3).collect();
    let expected: Vec<f32> =
        a.iter().zip(b.iter()).map(|(x, y)| oracle_remainder(*x, *y)).collect();

    let actual = run_remainder(&a, &b, Dt::F32, n);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "remainder +/+ f32: max |diff| = {diff:.2e} > 1e-4");
}

#[test]
fn remainder_negative_dividend_truncated_semantics_f32() {
    let _g = gpu_lock();
    // Negative dividend exercises the truncated-vs-floored divergence.
    // fmod(-7, 3) = -1, not +2.  This is the critical sign-semantics test.
    let a: Vec<f32> = vec![-7.0, -3.5, -10.0, -1.0, -5.0, 5.0, 7.0, 3.5];
    let b: Vec<f32> = vec![3.0, 1.5, 3.0, 0.7, 2.1, 3.0, 3.0, 1.5];
    let n = a.len();
    let expected: Vec<f32> =
        a.iter().zip(b.iter()).map(|(x, y)| oracle_remainder(*x, *y)).collect();

    let actual = run_remainder(&a, &b, Dt::F32, n);
    for (i, (got, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "remainder sign semantics at i={i}: got {got}, expected {exp} (fmod({}, {}))",
            a[i],
            b[i],
        );
    }
}

#[test]
fn remainder_mixed_sign_inputs_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    // Mixed-sign ramp — covers all four sign combinations across the array.
    let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.3 - 3.0).collect();
    let b: Vec<f32> = (0..n)
        .map(|i| {
            // Avoid near-zero divisors (fmod(x, ε) is noisy).
            let raw = (i % 11) as f32 * 0.4 - 2.0;
            if raw.abs() < 0.3 { raw.signum() * 0.4 } else { raw }
        })
        .collect();
    let expected: Vec<f32> =
        a.iter().zip(b.iter()).map(|(x, y)| oracle_remainder(*x, *y)).collect();

    let actual = run_remainder(&a, &b, Dt::F32, n);
    let diff = max_abs_diff(&actual, &expected);
    // f32 fmod tolerance: 1e-4 covers the ULP drift on the GPU's
    // division-then-truncation path.
    assert!(diff < 1e-4, "remainder mixed-sign f32: max |diff| = {diff:.2e} > 1e-4");
}

#[test]
fn remainder_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let n = 64usize;
    let a: Vec<f32> = (1..=n as u32).map(|i| i as f32 * 1.7).collect();
    let b: Vec<f32> = vec![3.0f32; n];
    let actual = run_remainder(&a, &b, Dt::F32, n);
    assert!(actual.iter().any(|&v| v != 0.0), "remainder output is all zeros — empty kernel body?",);
}

#[test]
#[ignore = "perf bench — run with --ignored --nocapture"]
fn remainder_perf_bench_f32() {
    use std::time::Instant;
    let _g = gpu_lock();
    let n: usize = 1 << 22;
    let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.3 - 3.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.4 + 0.3).collect();
    let ctx = Context::new().expect("Context::new");
    let kernel = mt_remainder::kernel_ir_for(DType::F32);
    let tpg: usize = 256;
    let groups = n.div_ceil(tpg);
    for _ in 0..5 {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("a".into(), pack_bytes(&a, Dt::F32));
        buffers.insert("b".into(), pack_bytes(&b, Dt::F32));
        buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("warmup");
    }
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("a".into(), pack_bytes(&a, Dt::F32));
        buffers.insert("b".into(), pack_bytes(&b, Dt::F32));
        buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("bench");
    }
    let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
    let gb_s = n as f64 * 4.0 * 3.0 / elapsed_us / 1e3;
    println!("remainder f32 N={n}: {elapsed_us:.1} µs  |  {gb_s:.1} GB/s");
}

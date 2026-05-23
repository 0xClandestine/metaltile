//! GPU correctness for the `steel_gemm_splitk` two-kernel family —
//! `mlx::steel::gemm::steel_gemm_splitk`.
//!
//! Split-K GEMM is a two-pass dispatch:
//!   1. `mt_steel_gemm_splitk_*` — each K-split computes a partial
//!      `[M, N]` product over its K-slice into an `[n_splits, M, N]`
//!      fp32 partials buffer.
//!   2. `mt_steel_gemm_splitk_accum*` — reduces the partials into the
//!      final `[M, N]` output (plain sum, or the `axpby` fused-bias
//!      form `α·Σ + β·C_in`).
//!
//! This file dispatches both passes back-to-back and pins the result
//! against a plain triple-loop fp32 CPU matmul. Coverage:
//!   - 2-way and 3-way split ⇒ partials buffer + accum reduce
//!   - the `axpby` accum form ⇒ `α·Σ + β·C_in`
//!
//! macOS-gated: needs a real Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::gemm::steel_gemm_splitk::{
    mt_steel_gemm_splitk_64x64x16_2x2,
    mt_steel_gemm_splitk_accum,
    mt_steel_gemm_splitk_accum_axpby,
};

/// Naive fp32 matmul: `out[m, n] = sum_k a[m, k] * b[k, n]`.
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.3e} at {at} (expected {:.6}, got {:.6})",
        expected[at],
        actual[at],
    );
}

// Canonical shape: 64×64×16 / 2×2, tpg = 128. M / N = 2×block.
const BM: usize = 64;
const BN: usize = 64;
const TPG: usize = 128;

/// Pass 1: dispatch the split-K partial GEMM → fp32 `[n_splits, M, N]`.
#[allow(clippy::too_many_arguments)]
fn run_splitk_pass1(
    kernel_ir: fn(DType) -> Kernel,
    a: &[f32],
    b: &[f32],
    dt: Dt,
    m: usize,
    n: usize,
    k: usize,
    n_splits: usize,
    k_per_split: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    // Partials are always fp32.
    buffers.insert("partials".into(), pack_bytes(&vec![0.0f32; n_splits * m * n], Dt::F32));
    buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("k_per_split".into(), (k_per_split as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::SimdGroup2D;

    // 3-D grid: x = N-block, y = M-block, z = K-split.
    let grid = [n / BN, m / BM, n_splits];
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, [TPG, 1, 1])
        .expect("splitk pass1 dispatch");
    unpack_bytes(result.outputs.get("partials").expect("partials"), Dt::F32)
}

/// Pass 2 (plain sum): reduce `[n_splits, M, N]` partials → `[M, N]`.
fn run_splitk_accum(partials: &[f32], dt: Dt, m: usize, n: usize, n_splits: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("partials".into(), pack_bytes(partials, Dt::F32));
    buffers.insert("out".into(), vec![0u8; m * n * dt.bytes()]);
    buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("n_splits".into(), (n_splits as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_steel_gemm_splitk_accum::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Elementwise;
    // One thread per [M, N] output element.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m * n, 1, 1], [256, 1, 1])
        .expect("splitk accum dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

/// Pass 2 (axpby): `out = α·Σ partials + β·c_in`.
#[allow(clippy::too_many_arguments)]
fn run_splitk_accum_axpby(
    partials: &[f32],
    c_in: &[f32],
    dt: Dt,
    m: usize,
    n: usize,
    n_splits: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("partials".into(), pack_bytes(partials, Dt::F32));
    buffers.insert("c_in".into(), pack_bytes(c_in, dt));
    buffers.insert("out".into(), vec![0u8; m * n * dt.bytes()]);
    buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("n_splits".into(), (n_splits as u32).to_le_bytes().to_vec());
    buffers.insert("alpha".into(), alpha.to_le_bytes().to_vec());
    buffers.insert("beta".into(), beta.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_steel_gemm_splitk_accum_axpby::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Elementwise;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m * n, 1, 1], [256, 1, 1])
        .expect("splitk accum_axpby dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

#[test]
fn splitk_2way_matches_plain_matmul_f32() {
    let _g = gpu_lock();
    let (m, n, k) = (BM * 2, BN * 2, 64);
    let (n_splits, k_per_split) = (2, 32); // 2 × 32 = 64
    let a = ramp(m * k, 19, 7.0);
    let b = ramp(k * n, 23, 9.0);
    let expected = naive_matmul(&a, &b, m, k, n);

    let partials = run_splitk_pass1(
        mt_steel_gemm_splitk_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        Dt::F32,
        m,
        n,
        k,
        n_splits,
        k_per_split,
    );
    let actual = run_splitk_accum(&partials, Dt::F32, m, n, n_splits);
    assert_close(&actual, &expected, 3e-3, "splitk 2-way f32");
}

#[test]
fn splitk_3way_matches_plain_matmul_f32() {
    let _g = gpu_lock();
    let (m, n, k) = (BM * 2, BN * 2, 48);
    let (n_splits, k_per_split) = (3, 16); // 3 × 16 = 48
    let a = ramp(m * k, 31, 11.0);
    let b = ramp(k * n, 37, 13.0);
    let expected = naive_matmul(&a, &b, m, k, n);

    let partials = run_splitk_pass1(
        mt_steel_gemm_splitk_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        Dt::F32,
        m,
        n,
        k,
        n_splits,
        k_per_split,
    );
    let actual = run_splitk_accum(&partials, Dt::F32, m, n, n_splits);
    assert_close(&actual, &expected, 3e-3, "splitk 3-way f32");
}

#[test]
fn splitk_accum_axpby_matches_reference_f32() {
    let _g = gpu_lock();
    let (m, n, k) = (BM * 2, BN * 2, 64);
    let (n_splits, k_per_split) = (2, 32);
    let (alpha, beta) = (0.5f32, 2.0f32);
    let a = ramp(m * k, 19, 7.0);
    let b = ramp(k * n, 23, 9.0);
    let c_in = ramp(m * n, 41, 5.0);

    let prod = naive_matmul(&a, &b, m, k, n);
    let expected: Vec<f32> =
        prod.iter().zip(c_in.iter()).map(|(&p, &c)| alpha * p + beta * c).collect();

    let partials = run_splitk_pass1(
        mt_steel_gemm_splitk_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        Dt::F32,
        m,
        n,
        k,
        n_splits,
        k_per_split,
    );
    let actual = run_splitk_accum_axpby(&partials, &c_in, Dt::F32, m, n, n_splits, alpha, beta);
    assert_close(&actual, &expected, 3e-3, "splitk accum_axpby f32");
}

#[test]
fn splitk_2way_matches_plain_matmul_f16() {
    let _g = gpu_lock();
    let (m, n, k) = (BM * 2, BN * 2, 64);
    let (n_splits, k_per_split) = (2, 32);
    let a = ramp(m * k, 19, 7.0);
    let b = ramp(k * n, 23, 9.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_matmul(&round(&a), &round(&b), m, k, n);

    let partials = run_splitk_pass1(
        mt_steel_gemm_splitk_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        Dt::F16,
        m,
        n,
        k,
        n_splits,
        k_per_split,
    );
    // Partials are fp32 — only the final accum store quantises to f16.
    let actual = run_splitk_accum(&partials, Dt::F16, m, n, n_splits);
    assert_close(&actual, &expected, 8e-2, "splitk 2-way f16");
}

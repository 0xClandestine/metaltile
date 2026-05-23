//! GPU correctness for `mlx::gemv_masked` — masked matrix-vector multiply
//! (`mt_gemv_masked`).
//!
//! `mt_gemv_masked` computes `out[row] = Σ_j mat[row, j] * vec[j] * mask[j]`.
//! Each thread handles one column stripe via a strided reduce; one threadgroup
//! per output row.
//!
//! ## DISPATCH INVARIANTS (mt_gemv_masked)
//! - **Reduction mode** — uses `reduce_sum` across the threadgroup.
//! - `grid = [m, 1, 1]` (m = number of rows), `tg = [256, 1, 1]` (spec `tpg=256`).
//! - `k` passed as a constexpr (columns = reduction dimension).
//!
//! CPU oracle: element-wise masked matvec — identical to `mt_gemv` but with
//! the mask applied per column.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::gemv_masked::mt_gemv_masked;

/// Naive CPU masked matvec: `out[i] = Σ_j mat[i,j] * vec[j] * mask[j]`.
fn naive_masked_matvec(mat: &[f32], vec: &[f32], mask: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(mat.len(), m * k);
    assert_eq!(vec.len(), k);
    assert_eq!(mask.len(), k);
    let mut out = vec![0.0f32; m];
    for i in 0..m {
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += mat[i * k + j] * vec[j] * mask[j];
        }
        out[i] = acc;
    }
    out
}

fn run_gemv_masked(mat: &[f32], vec: &[f32], mask: &[f32], dt: Dt, m: usize, k: usize) -> Vec<f32> {
    let dt_bytes = dt.bytes();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("mat".into(), pack_bytes(mat, dt));
    buffers.insert("vec".into(), pack_bytes(vec, dt));
    buffers.insert("mask".into(), pack_bytes(mask, dt));
    buffers.insert("out".into(), vec![0u8; m * dt_bytes]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gemv_masked::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // One threadgroup per output row, 256 threads per group.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m, 1, 1], [256, 1, 1])
        .expect("gemv_masked dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(m);
    out
}

#[test]
fn gemv_masked_matches_cpu_oracle_f32_small() {
    let _g = gpu_lock();
    let (m, k) = (16usize, 256usize);
    let mat: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let vec: Vec<f32> = (0..k).map(|j| ((j % 7) as f32 - 3.0) * 0.02).collect();
    // Mask: alternating 0/1 — zeros out every other column.
    let mask: Vec<f32> = (0..k).map(|j| if j % 2 == 0 { 1.0 } else { 0.0 }).collect();

    let expected = naive_masked_matvec(&mat, &vec, &mask, m, k);
    let actual = run_gemv_masked(&mat, &vec, &mask, Dt::F32, m, k);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-3, "gemv_masked f32 small: max |diff| = {diff:.2e}");
}

#[test]
fn gemv_masked_all_ones_mask_equals_unmasked_f32() {
    let _g = gpu_lock();
    // When mask is all-ones, masked matvec == plain matvec.
    let (m, k) = (8usize, 512usize);
    let mat: Vec<f32> = (0..m * k).map(|i| (((i * 31 + 17) % 100) as f32 - 50.0) * 0.001).collect();
    let vec: Vec<f32> = (0..k).map(|j| (((j * 13 + 5) % 50) as f32 - 25.0) * 0.002).collect();
    let mask = vec![1.0f32; k];

    let expected = naive_masked_matvec(&mat, &vec, &mask, m, k);
    let actual = run_gemv_masked(&mat, &vec, &mask, Dt::F32, m, k);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-3, "gemv_masked all-ones mask: max |diff| = {diff:.2e}");
}

#[test]
fn gemv_masked_all_zeros_mask_gives_zero_output() {
    let _g = gpu_lock();
    // Zero mask → all outputs should be zero.
    let (m, k) = (8usize, 256usize);
    let mat: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
    let vec: Vec<f32> = (0..k).map(|j| j as f32 * 0.01).collect();
    let mask = vec![0.0f32; k];

    let actual = run_gemv_masked(&mat, &vec, &mask, Dt::F32, m, k);

    for (i, &v) in actual.iter().enumerate() {
        assert_eq!(v, 0.0, "gemv_masked zero mask [{i}] = {v} != 0");
    }
}

#[test]
fn gemv_masked_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let (m, k) = (4usize, 256usize);
    let mat: Vec<f32> = (1..=m * k).map(|i| i as f32 * 0.001).collect();
    let vec: Vec<f32> = (1..=k).map(|j| j as f32 * 0.001).collect();
    let mask = vec![1.0f32; k];

    let actual = run_gemv_masked(&mat, &vec, &mask, Dt::F32, m, k);
    assert!(actual.iter().any(|&v| v != 0.0), "gemv_masked output all zeros — empty kernel?");
}

#[test]
fn gemv_masked_production_size_f32() {
    let _g = gpu_lock();
    // 32 × 4096 — matches the bench spec (b=4096, n=4096 → m=n=32 here).
    // TPG=256 so each thread covers 4096/256 = 16 elements per row.
    let (m, k) = (32usize, 4096usize);
    let mat: Vec<f32> =
        (0..m * k).map(|i| (((i * 31 + 17) % 200) as f32 - 100.0) * 0.001).collect();
    let vec: Vec<f32> = (0..k).map(|j| (((j * 13 + 5) % 100) as f32 - 50.0) * 0.002).collect();
    // Mask: 3/4 of columns active.
    let mask: Vec<f32> = (0..k).map(|j| if j % 4 != 3 { 1.0 } else { 0.0 }).collect();

    let expected = naive_masked_matvec(&mat, &vec, &mask, m, k);
    let actual = run_gemv_masked(&mat, &vec, &mask, Dt::F32, m, k);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 5e-3, "gemv_masked f32 production: max |diff| = {diff:.2e}");
}

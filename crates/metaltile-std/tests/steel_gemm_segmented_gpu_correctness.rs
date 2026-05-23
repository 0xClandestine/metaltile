//! GPU correctness for the `steel_gemm_segmented` kernel family —
//! `mlx::steel::gemm::steel_gemm_segmented`.
//!
//! The kernel computes a ragged-K batched matmul: for each segment,
//!   C[seg] = A[:, k_start..k_end] · B[k_start..k_end, :]
//! where `A` is `[M, total_K]`, `B` is `[total_K, N]`, the output is
//! `[n_segments, M, N]`, and a `segments` descriptor gives each
//! segment's half-open `[k_start, k_end)` K-range.
//!
//! It pins the kernel against a triple-loop fp32 CPU reference.
//! Coverage:
//!   - a single full-K segment ⇒ plain `A · B`
//!   - multiple disjoint K-ranges ⇒ ragged segmentation
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
use metaltile_std::mlx::steel::gemm::steel_gemm_segmented::mt_steel_gemm_segmented_64x64x16_2x2;

/// Naive segmented fp32 reference.
///
/// For segment `seg` with K-range `[segments[2*seg], segments[2*seg+1])`:
///   out[seg, r, c] = sum_{k in range} a[r, k] * b[k, c]
/// `a` is `[m, total_k]`, `b` is `[total_k, n]`.
#[allow(clippy::too_many_arguments)]
fn naive_segmented_matmul(
    a: &[f32],
    b: &[f32],
    segments: &[u32],
    m: usize,
    n: usize,
    total_k: usize,
) -> Vec<f32> {
    let n_segments = segments.len() / 2;
    let mut out = vec![0.0f32; n_segments * m * n];
    for seg in 0..n_segments {
        let k_start = segments[2 * seg] as usize;
        let k_end = segments[2 * seg + 1] as usize;
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for k in k_start..k_end {
                    acc += a[r * total_k + k] * b[k * n + c];
                }
                out[seg * m * n + r * n + c] = acc;
            }
        }
    }
    out
}

/// Dispatch the segmented GEMM. Grid z = segment count.
#[allow(clippy::too_many_arguments)]
fn run_segmented_gemm(
    kernel_ir: fn(DType) -> Kernel,
    a: &[f32],
    b: &[f32],
    segments: &[u32],
    dt: Dt,
    m: usize,
    n: usize,
    total_k: usize,
    bm: usize,
    bn: usize,
    tpg: usize,
) -> Vec<f32> {
    let n_segments = segments.len() / 2;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("segments".into(), segments.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("out".into(), vec![0u8; n_segments * m * n * dt.bytes()]);
    buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("total_k".into(), (total_k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::SimdGroup2D;

    // 3-D grid: x = N-block, y = M-block, z = segment.
    let grid = [n / bn, m / bm, n_segments];
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, [tpg, 1, 1])
        .expect("dispatch_with_grid");
    unpack_bytes(result.outputs.get("out").expect("out buffer"), dt)
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

// Canonical shape: 64×64×16 / 2×2, tpg = 128. M / N = 2×block ⇒
// 2×2 output blocks per segment.
const BM: usize = 64;
const BN: usize = 64;
const TPG: usize = 128;

fn run_case(dt: Dt, total_k: usize, segments: &[u32], tol: f32, label: &str) {
    let _g = gpu_lock();
    let (m, n) = (BM * 2, BN * 2);
    let a = ramp(m * total_k, 19, 7.0);
    let b = ramp(total_k * n, 23, 9.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| dt.round(x)).collect() };
    let (a_r, b_r) = (round(&a), round(&b));
    let expected = naive_segmented_matmul(&a_r, &b_r, segments, m, n, total_k);
    let actual = run_segmented_gemm(
        mt_steel_gemm_segmented_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        segments,
        dt,
        m,
        n,
        total_k,
        BM,
        BN,
        TPG,
    );
    assert_close(&actual, &expected, tol, label);
}

#[test]
fn segmented_gemm_single_full_segment_f32() {
    // One segment spanning the full K — equivalent to a plain GEMM.
    let total_k = 48;
    let segments = vec![0u32, 48];
    run_case(Dt::F32, total_k, &segments, 2e-3, "segmented single-full f32");
}

#[test]
fn segmented_gemm_disjoint_ranges_f32() {
    // Three segments, each a distinct 16-wide K-block of a 48-wide K.
    let total_k = 48;
    let segments = vec![0u32, 16, 16, 32, 32, 48];
    run_case(Dt::F32, total_k, &segments, 2e-3, "segmented disjoint f32");
}

#[test]
fn segmented_gemm_uneven_ranges_f32() {
    // Two segments of different widths (16 and 32) over a 48-wide K.
    let total_k = 48;
    let segments = vec![0u32, 16, 16, 48];
    run_case(Dt::F32, total_k, &segments, 2e-3, "segmented uneven f32");
}

#[test]
fn segmented_gemm_disjoint_ranges_f16() {
    let total_k = 48;
    let segments = vec![0u32, 16, 16, 32, 32, 48];
    run_case(Dt::F16, total_k, &segments, 8e-2, "segmented disjoint f16");
}

#[test]
fn segmented_gemm_uneven_ranges_bf16() {
    let total_k = 48;
    let segments = vec![0u32, 16, 16, 48];
    run_case(Dt::Bf16, total_k, &segments, 5e-1, "segmented uneven bf16");
}

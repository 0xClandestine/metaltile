//! GPU correctness for the `steel_gemm_gather` kernel family —
//! `mlx::steel::gemm::steel_gemm_gather`.
//!
//! The kernel computes a row-major `C = A_gathered · B_gathered`:
//!   - output row `r` is computed from `A` row `lhs_indices[r]`,
//!   - output N-block `c` multiplies against `B` matrix
//!     `rhs_indices[c]` (one `[K, N]` matrix of several).
//!
//! This is the MLX `gather_mm` op — the dense matmul of a MoE FFN.
//!
//! It pins the kernel against a triple-loop fp32 CPU reference that
//! applies the same gather. Coverage:
//!   - identity `lhs_indices` + single B matrix ⇒ plain `A · B`
//!   - a permuted / repeated `lhs_indices` ⇒ row gather
//!   - two B matrices selected per N-block ⇒ rhs gather
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
use metaltile_std::mlx::steel::gemm::steel_gemm_gather::mt_steel_gemm_gather_64x64x16_2x2;

/// Naive gathered fp32 reference.
///
/// `out[r, c] = sum_k a[lhs_indices[r], k] * b_sel[k, c]` where
/// `b_sel` is B matrix `rhs_indices[c / bn]`, each `[k, n]` and stored
/// flat in `b` at offset `idx * k * n`.
#[allow(clippy::too_many_arguments)]
fn naive_gather_matmul(
    a: &[f32],
    b: &[f32],
    lhs_indices: &[u32],
    rhs_indices: &[u32],
    m: usize,
    k: usize,
    n: usize,
    bn: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        let a_row = lhs_indices[r] as usize;
        for c in 0..n {
            let b_base = rhs_indices[c / bn] as usize * k * n;
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[a_row * k + ki] * b[b_base + ki * n + c];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

/// Dispatch the gather GEMM.
#[allow(clippy::too_many_arguments)]
fn run_gather_gemm(
    kernel_ir: fn(DType) -> Kernel,
    a: &[f32],
    b: &[f32],
    lhs_indices: &[u32],
    rhs_indices: &[u32],
    dt: Dt,
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    tpg: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers
        .insert("lhs_indices".into(), lhs_indices.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers
        .insert("rhs_indices".into(), rhs_indices.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("out".into(), vec![0u8; m * n * dt.bytes()]);
    buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::SimdGroup2D;

    let grid = [n / bn, m / bm, 1];
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
// 2×2 output blocks; K = 48 ⇒ 3 K-blocks.
const BM: usize = 64;
const BN: usize = 64;
const TPG: usize = 128;

fn run_case(
    dt: Dt,
    n_a_rows: usize,
    n_b_mats: usize,
    lhs_indices: &[u32],
    rhs_indices: &[u32],
    tol: f32,
    label: &str,
) {
    let _g = gpu_lock();
    let (m, n, k) = (BM * 2, BN * 2, 48);
    // `a` has `n_a_rows` rows of K; `b` has `n_b_mats` flat [K, N] matrices.
    let a = ramp(n_a_rows * k, 19, 7.0);
    let b = ramp(n_b_mats * k * n, 23, 9.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| dt.round(x)).collect() };
    let (a_r, b_r) = (round(&a), round(&b));
    let expected = naive_gather_matmul(&a_r, &b_r, lhs_indices, rhs_indices, m, k, n, BN);
    let actual = run_gather_gemm(
        mt_steel_gemm_gather_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        lhs_indices,
        rhs_indices,
        dt,
        m,
        n,
        k,
        BM,
        BN,
        TPG,
    );
    assert_close(&actual, &expected, tol, label);
}

#[test]
fn gather_gemm_identity_indices_f32() {
    // lhs identity, single B matrix selected for every N-block.
    let m = BM * 2;
    let lhs: Vec<u32> = (0..m as u32).collect();
    let rhs = vec![0u32; 2]; // 2 N-blocks
    run_case(Dt::F32, m, 1, &lhs, &rhs, 2e-3, "gather identity f32");
}

#[test]
fn gather_gemm_permuted_lhs_f32() {
    // Permuted + repeated lhs indices — pulls from a larger A pool.
    let m = BM * 2;
    let n_a_rows = m + 16;
    let lhs: Vec<u32> = (0..m).map(|r| ((r * 7 + 3) % n_a_rows) as u32).collect();
    let rhs = vec![0u32; 2];
    run_case(Dt::F32, n_a_rows, 1, &lhs, &rhs, 2e-3, "gather permuted-lhs f32");
}

#[test]
fn gather_gemm_rhs_select_f32() {
    // Two B matrices; N-block 0 uses matrix 1, N-block 1 uses matrix 0.
    let m = BM * 2;
    let lhs: Vec<u32> = (0..m as u32).collect();
    let rhs = vec![1u32, 0u32];
    run_case(Dt::F32, m, 2, &lhs, &rhs, 2e-3, "gather rhs-select f32");
}

#[test]
fn gather_gemm_permuted_lhs_f16() {
    let m = BM * 2;
    let n_a_rows = m + 16;
    let lhs: Vec<u32> = (0..m).map(|r| ((r * 7 + 3) % n_a_rows) as u32).collect();
    let rhs = vec![0u32; 2];
    run_case(Dt::F16, n_a_rows, 1, &lhs, &rhs, 8e-2, "gather permuted-lhs f16");
}

#[test]
fn gather_gemm_rhs_select_bf16() {
    let m = BM * 2;
    let lhs: Vec<u32> = (0..m as u32).collect();
    let rhs = vec![1u32, 0u32];
    run_case(Dt::Bf16, m, 2, &lhs, &rhs, 5e-1, "gather rhs-select bf16");
}

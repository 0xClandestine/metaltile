//! GPU correctness for the `steel_gemm_masked` kernel family —
//! `mlx::steel::gemm::steel_gemm_masked`.
//!
//! The kernel computes a plain row-major `C = A · B` (`nn` layout) with
//! **block-level predication**:
//!   - an output-block mask zeroes whole `BM×BN` output blocks,
//!   - an operand-block mask scales (a `0` zeroes) each `BM×BK` /
//!     `BK×BN` K-block's contribution.
//!
//! It pins the kernel against a triple-loop fp32 CPU reference that
//! applies the same masks. Coverage:
//!   - all-ones masks ⇒ identical to a plain `steel_gemm_fused`
//!   - a checkerboard output-block mask ⇒ zeroed output blocks
//!   - a partial operand-block mask ⇒ dropped K-block contributions
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
use metaltile_std::mlx::steel::gemm::steel_gemm_masked::mt_steel_gemm_masked_64x64x16_2x2;

/// Naive masked fp32 reference.
///
/// `out[m, n] = sum over active K-blocks of out_mask[blk] * op_mask[blk, kb]
///              * sum_{k in block kb} a[m, k] * b[k, n]`.
/// `out_mask` is `[m/bm * n/bn]`; `op_mask` is `[m/bm * n/bn * k/16]`.
#[allow(clippy::too_many_arguments)]
fn naive_masked_matmul(
    a: &[f32],
    b: &[f32],
    out_mask: &[f32],
    op_mask: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bm: usize,
    bn: usize,
) -> Vec<f32> {
    let n_n_blocks = n / bn;
    let n_k_blocks = k / 16;
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let blk = (mi / bm) * n_n_blocks + (ni / bn);
            // Inactive output block ⇒ zero, no accumulation.
            if out_mask[blk] == 0.0 {
                out[mi * n + ni] = 0.0;
                continue;
            }
            let mut acc = 0.0f32;
            for kb in 0..n_k_blocks {
                let opm = op_mask[blk * n_k_blocks + kb];
                for ki in (kb * 16)..(kb * 16 + 16) {
                    acc += a[mi * k + ki] * opm * b[ki * n + ni];
                }
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Dispatch the masked GEMM. `a` is `[m,k]`, `b` is `[k,n]`, row-major.
#[allow(clippy::too_many_arguments)]
fn run_masked_gemm(
    kernel_ir: fn(DType) -> Kernel,
    a: &[f32],
    b: &[f32],
    out_mask: &[f32],
    op_mask: &[f32],
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
    buffers.insert("out_mask".into(), pack_bytes(out_mask, dt));
    buffers.insert("op_mask".into(), pack_bytes(op_mask, dt));
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

// Canonical shape: 64×64×16 / 2×2, tpg = 128. M / N = 2×block so the
// grid is 2×2 output blocks; K = 48 ⇒ 3 K-blocks.
const BM: usize = 64;
const BN: usize = 64;
const TPG: usize = 128;

fn run_case(dt: Dt, out_mask: &[f32], op_mask: &[f32], tol: f32, label: &str) {
    let _g = gpu_lock();
    let (m, n, k) = (BM * 2, BN * 2, 48);
    let a = ramp(m * k, 19, 7.0);
    let b = ramp(k * n, 23, 9.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| dt.round(x)).collect() };
    let (a_r, b_r) = (round(&a), round(&b));
    let expected = naive_masked_matmul(&a_r, &b_r, out_mask, op_mask, m, k, n, BM, BN);
    let actual = run_masked_gemm(
        mt_steel_gemm_masked_64x64x16_2x2::kernel_ir_for,
        &a,
        &b,
        out_mask,
        op_mask,
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
fn masked_gemm_all_ones_matches_plain_f32() {
    // 2×2 output blocks, 3 K-blocks ⇒ 4 out-mask entries, 12 op-mask.
    let out_mask = vec![1.0f32; 4];
    let op_mask = vec![1.0f32; 12];
    run_case(Dt::F32, &out_mask, &op_mask, 2e-3, "masked all-ones f32");
}

#[test]
fn masked_gemm_checkerboard_outmask_f32() {
    // Zero out the (0,1) and (1,0) output blocks.
    let out_mask = vec![1.0f32, 0.0, 0.0, 1.0];
    let op_mask = vec![1.0f32; 12];
    run_case(Dt::F32, &out_mask, &op_mask, 2e-3, "masked checkerboard f32");
}

#[test]
fn masked_gemm_partial_opmask_f32() {
    // Drop the middle K-block of every active output block.
    let out_mask = vec![1.0f32; 4];
    let op_mask: Vec<f32> = (0..12).map(|i| if i % 3 == 1 { 0.0 } else { 1.0 }).collect();
    run_case(Dt::F32, &out_mask, &op_mask, 2e-3, "masked partial-opmask f32");
}

#[test]
fn masked_gemm_checkerboard_outmask_f16() {
    let out_mask = vec![1.0f32, 0.0, 0.0, 1.0];
    let op_mask = vec![1.0f32; 12];
    run_case(Dt::F16, &out_mask, &op_mask, 8e-2, "masked checkerboard f16");
}

#[test]
fn masked_gemm_partial_opmask_bf16() {
    let out_mask = vec![1.0f32; 4];
    let op_mask: Vec<f32> = (0..12).map(|i| if i % 3 == 1 { 0.0 } else { 1.0 }).collect();
    run_case(Dt::Bf16, &out_mask, &op_mask, 5e-1, "masked partial-opmask bf16");
}

//! End-to-end correctness test for `mlx::mt_gemv` on real Metal.
//!
//! Pins the kernel's spec'd `tpg=256` and the adaptive `lsize` reduction
//! (kernel uses `strided_reduce_dot` + `reduce_sum`, both of which adapt
//! to the dispatched TPG). Wrapper dispatches `grid = (m * 256, 1, 1)`,
//! `tg = (256, 1, 1)` so Metal slices that into `m` threadgroups of 256.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::gemv::mt_gemv;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Naive matvec reference: `out[i] = Σ_j mat[i, j] * vec[j]`. f32.
fn naive_matvec_f32(mat: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(mat.len(), m * k);
    assert_eq!(vec.len(), k);
    let mut out = vec![0.0_f32; m];
    for i in 0..m {
        let mut acc = 0.0_f32;
        for j in 0..k {
            acc += mat[i * k + j] * vec[j];
        }
        out[i] = acc;
    }
    out
}

fn run_gemv(mat: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("mat".into(), f32_slice_to_bytes(mat));
    buffers.insert("vec".into(), f32_slice_to_bytes(vec));
    buffers.insert("out".into(), vec![0u8; m * 4]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_gemv::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per output row, 256 threads per group (kernel's
    // spec'd `tpg=256`). reduce_sum adapts to whatever TPG we dispatch
    // — 256 is the bench shape.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m, 1, 1], [256, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

#[test]
fn gemv_matches_naive_cpu_reference_f32_small() {
    // 16 × 256 — fits one TPG with no inner-loop iteration.
    let m = 16usize;
    let k = 256usize;
    let mat: Vec<f32> = (0..m * k).map(|i| ((i as f32 % 13.0) - 6.0) * 0.01).collect();
    let vec: Vec<f32> = (0..k).map(|j| ((j as f32 % 7.0) - 3.0) * 0.02).collect();
    let expected = naive_matvec_f32(&mat, &vec, m, k);
    let actual = run_gemv(&mat, &vec, m, k);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "gemv f32 small: max |diff| = {diff:.2e}");
}

#[test]
fn gemv_matches_naive_cpu_reference_f32_production_size() {
    // 32 × 4096 — Llama hidden_dim → q_proj per-head magnitude.
    // Each thread covers k / 256 = 16 elements via the strided reduce.
    let m = 32usize;
    let k = 4096usize;
    let mat: Vec<f32> =
        (0..m * k).map(|i| (((i * 31 + 17) % 200) as f32 - 100.0) * 0.001).collect();
    let vec: Vec<f32> = (0..k).map(|j| (((j * 13 + 5) % 100) as f32 - 50.0) * 0.002).collect();
    let expected = naive_matvec_f32(&mat, &vec, m, k);
    let actual = run_gemv(&mat, &vec, m, k);
    let diff = max_abs_diff(&expected, &actual);
    // Looser tol at k=4096 — more partials in the reduce → more
    // reordering noise vs the sequential CPU sum.
    assert!(diff < 5e-3, "gemv f32 production: max |diff| = {diff:.2e}");
}

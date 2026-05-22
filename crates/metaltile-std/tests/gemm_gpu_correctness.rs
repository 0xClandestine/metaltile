//! End-to-end GPU correctness for `ffai::gemm` — the multi-row tiled
//! GEMM `out[r, :] = weight · input[r, :]`.
//!
//! Validates proc-macro → IR → MSL → PSO → dispatch → readback against
//! a straight triple-loop CPU reference. Covers:
//!   - aligned shapes (n_rows / out_dim multiples of the 32×32 tile)
//!   - edge shapes (n_rows and out_dim not multiples of 32 — exercises
//!     the in-kernel load-clamp + store-skip)
//!   - a wide in_dim (the K tiling loop runs many iterations)
//!   - f32 / f16 / bf16
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemm::ffai_gemm;

/// CPU reference: out[r, o] = sum_k weight[o, k] * input[r, k].
fn naive_gemm(
    weight: &[f32],
    input: &[f32],
    n_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += weight[o * in_dim + k] * input[r * in_dim + k];
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}

fn run_gemm(
    weight: &[f32],
    input: &[f32],
    dt: Dt,
    n_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_rows * out_dim], dt));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_rows".into(), (n_rows as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_gemm::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // 2-D grid: tgid_x = output-column tile, tgid_y = row tile; TPG 1024.
    let n_tiles = out_dim.div_ceil(32);
    let m_tiles = n_rows.div_ceil(32);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tiles, m_tiles, 1], [
            1024, 1, 1,
        ])
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
        "{label}: max |diff| = {max_diff:.2e} at {at} (expected {:.6}, got {:.6})",
        expected[at],
        actual[at]
    );
}

#[test]
fn gemm_aligned_matches_cpu_f32() {
    let _g = gpu_lock();
    // 32-row block, both dims tile-aligned.
    let (n_rows, in_dim, out_dim) = (32usize, 64usize, 64usize);
    let weight = ramp(out_dim * in_dim, 31, 14.0);
    let input = ramp(n_rows * in_dim, 23, 9.0);
    let expected = naive_gemm(&weight, &input, n_rows, in_dim, out_dim);
    let actual = run_gemm(&weight, &input, Dt::F32, n_rows, in_dim, out_dim);
    assert_close(&actual, &expected, 1e-3, "gemm aligned f32");
}

#[test]
fn gemm_edge_shapes_match_cpu_f32() {
    let _g = gpu_lock();
    // n_rows and out_dim both NOT multiples of 32 — exercises the
    // in-kernel load-clamp + store-skip on the tile edges. in_dim stays
    // a multiple of 16 (the K-tile contract).
    let (n_rows, in_dim, out_dim) = (20usize, 48usize, 100usize);
    let weight = ramp(out_dim * in_dim, 29, 12.0);
    let input = ramp(n_rows * in_dim, 17, 7.0);
    let expected = naive_gemm(&weight, &input, n_rows, in_dim, out_dim);
    let actual = run_gemm(&weight, &input, Dt::F32, n_rows, in_dim, out_dim);
    assert_close(&actual, &expected, 1e-3, "gemm edge f32");
}

#[test]
fn gemm_wide_indim_matches_cpu_f32() {
    let _g = gpu_lock();
    // Nemotron-class projection in_dim (hidden 3072); keep out_dim small
    // so the CPU reference stays instant. Many K-tile iterations.
    let (n_rows, in_dim, out_dim) = (32usize, 3072usize, 96usize);
    let weight = ramp(out_dim * in_dim, 41, 20.0);
    let input = ramp(n_rows * in_dim, 37, 18.0);
    let expected = naive_gemm(&weight, &input, n_rows, in_dim, out_dim);
    let actual = run_gemm(&weight, &input, Dt::F32, n_rows, in_dim, out_dim);
    // Wider reduction → more fp32 accumulation noise; 3072 terms.
    assert_close(&actual, &expected, 2e-3, "gemm wide-indim f32");
}

#[test]
fn gemm_aligned_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_rows, in_dim, out_dim) = (32usize, 64usize, 64usize);
    let weight = ramp(out_dim * in_dim, 31, 14.0);
    let input = ramp(n_rows * in_dim, 23, 9.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_gemm(&round(&weight), &round(&input), n_rows, in_dim, out_dim);
    let actual = run_gemm(&weight, &input, Dt::F16, n_rows, in_dim, out_dim);
    assert_close(&actual, &expected, 5e-2, "gemm aligned f16");
}

#[test]
fn gemm_edge_shapes_match_cpu_bf16() {
    let _g = gpu_lock();
    let (n_rows, in_dim, out_dim) = (20usize, 48usize, 100usize);
    let weight = ramp(out_dim * in_dim, 29, 12.0);
    let input = ramp(n_rows * in_dim, 17, 7.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_gemm(&round(&weight), &round(&input), n_rows, in_dim, out_dim);
    let actual = run_gemm(&weight, &input, Dt::Bf16, n_rows, in_dim, out_dim);
    assert_close(&actual, &expected, 2e-1, "gemm edge bf16");
}

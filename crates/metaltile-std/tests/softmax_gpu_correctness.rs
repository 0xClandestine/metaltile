//! GPU correctness for `mlx::softmax` — row-wise numerically stable softmax.
//!
//! `mt_softmax<T>` computes `out[row, i] = exp(x[i] - max_row) / Σ exp(x[j] - max_row)`
//! using online max tracking and a two-pass cross-simdgroup reduce.
//!
//! ## DISPATCH INVARIANTS (mt_softmax)
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [256, 1, 1]` (4 elems/thread).
//! - `n` a multiple of 4. Scalar tail handles the ragged suffix.
//!
//! CPU oracle: naive f32 softmax (subtract max for stability, sum, divide).
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::softmax::mt_softmax;

fn cpu_softmax_f32(inp: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(inp.len() % n, 0);
    let rows = inp.len() / n;
    let mut out = vec![0.0f32; inp.len()];
    for r in 0..rows {
        let base = r * n;
        let row = &inp[base..base + n];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for (d, &e) in exps.iter().enumerate() {
            out[base + d] = e / s;
        }
    }
    out
}

fn run_softmax(inp: &[f32], dt: Dt, n: usize, rows: usize) -> Vec<f32> {
    let tpg = 256usize; // matches bench spec tpg=256
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; rows * n], dt));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_softmax::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("softmax dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(rows * n);
    out
}

#[test]
fn softmax_matches_cpu_f32_n1024() {
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 4usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
    let expected = cpu_softmax_f32(&inp, n);
    let actual = run_softmax(&inp, Dt::F32, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "softmax n=1024 f32 max |diff| = {diff:.2e}");
}

#[test]
fn softmax_output_sums_to_one_f32() {
    let _g = gpu_lock();
    let (n, rows) = (256usize, 3usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.2).collect();
    let actual = run_softmax(&inp, Dt::F32, n, rows);

    for r in 0..rows {
        let sum: f32 = actual[r * n..(r + 1) * n].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax row {r} sum = {sum:.6} (expected 1.0)");
    }
}

#[test]
fn softmax_all_equal_input_f32() {
    // softmax of uniform input = 1/n for all elements.
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 2usize);
    let inp = vec![0.0f32; rows * n];
    let actual = run_softmax(&inp, Dt::F32, n, rows);

    let expected_val = 1.0 / n as f32;
    for (i, &v) in actual.iter().enumerate() {
        assert!(
            (v - expected_val).abs() < 1e-5,
            "softmax uniform [{i}]: expected {expected_val:.6}, got {v:.6}"
        );
    }
}

#[test]
fn softmax_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let inp: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let actual = run_softmax(&inp, Dt::F32, n, 1);
    assert!(actual.iter().any(|&v| v != 0.0), "softmax output all zeros — empty kernel?");
}

#[test]
fn softmax_matches_cpu_f16_n1024() {
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 2usize);
    let inp_f32: Vec<f32> = (0..rows * n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
    let inp_rounded: Vec<f32> = inp_f32.iter().map(|&v| Dt::F16.round(v)).collect();

    let expected = cpu_softmax_f32(&inp_rounded, n);
    let actual = run_softmax(&inp_f32, Dt::F16, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    // f16 softmax accumulation drift is wider (exp is a transcendental).
    assert!(diff < 5e-3, "softmax n=1024 f16 max |diff| = {diff:.2e}");
}

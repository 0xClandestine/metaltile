//! GPU correctness for `mlx::logsumexp` — log(sum(exp(x))) per row.
//!
//! `mt_logsumexp<T>` computes the numerically stable logsumexp over each
//! row using online max tracking + a cross-simdgroup reduce.
//!
//! ## DISPATCH INVARIANTS (mt_logsumexp)
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [256, 1, 1]` (4 elems/thread).
//! - `n` must be a multiple of `TPG * 4 = 1024`. Partial-row handling
//!   covers the tail via the scalar loop.
//!
//! CPU oracle: naive f32 logsumexp (stable form: subtract the max first).
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::logsumexp::mt_logsumexp;

fn cpu_logsumexp_f32(inp: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(inp.len() % n, 0);
    let rows = inp.len() / n;
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &inp[r * n..(r + 1) * n];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = row.iter().map(|&v| (v - m).exp()).sum();
        out[r] = m + s.ln();
    }
    out
}

fn run_logsumexp(inp: &[f32], dt: Dt, n: usize, rows: usize) -> Vec<f32> {
    // TPG = 256; n must be divisible by TPG * 4 = 1024 for the vector loop,
    // or the scalar tail handles the remainder. For simplicity use n = 1024.
    let tpg = 256usize;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    // Output: one scalar per row.
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; rows], dt));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_logsumexp::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("logsumexp dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(rows);
    out
}

#[test]
fn logsumexp_matches_cpu_f32_n1024() {
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 4usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
    let expected = cpu_logsumexp_f32(&inp, n);
    let actual = run_logsumexp(&inp, Dt::F32, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "logsumexp n=1024 f32 max |diff| = {diff:.2e}");
}

#[test]
fn logsumexp_matches_cpu_f32_n256() {
    let _g = gpu_lock();
    // n=256 exercises the scalar-tail path (256 < TPG*4 = 1024 so the
    // vector loop has nf=0 and the scalar loop handles all elements).
    let (n, rows) = (256usize, 8usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.4).collect();
    let expected = cpu_logsumexp_f32(&inp, n);
    let actual = run_logsumexp(&inp, Dt::F32, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-3, "logsumexp n=256 f32 max |diff| = {diff:.2e}");
}

#[test]
fn logsumexp_uniform_input_f32() {
    // logsumexp of uniform input `c` over `n` elements = c + log(n).
    // A useful algebraic invariant: degenerate inputs stress the online
    // max tracker (all maximums are equal).
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 2usize);
    let c = 3.0f32;
    let inp = vec![c; rows * n];
    let expected_val = c + (n as f32).ln();
    let actual = run_logsumexp(&inp, Dt::F32, n, rows);

    for (r, &v) in actual.iter().enumerate() {
        assert!(
            (v - expected_val).abs() < 1e-3,
            "logsumexp uniform row {r}: expected {expected_val:.4}, got {v:.4}"
        );
    }
}

#[test]
fn logsumexp_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 2usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let actual = run_logsumexp(&inp, Dt::F32, n, rows);
    assert!(actual.iter().any(|&v| v != 0.0), "logsumexp output all zeros");
}

#[test]
fn logsumexp_matches_cpu_f16_n1024() {
    let _g = gpu_lock();
    let (n, rows) = (1024usize, 2usize);
    let inp_f32: Vec<f32> = (0..rows * n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
    let inp_rounded: Vec<f32> = inp_f32.iter().map(|&v| Dt::F16.round(v)).collect();

    let expected = cpu_logsumexp_f32(&inp_rounded, n);
    let actual = run_logsumexp(&inp_f32, Dt::F16, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    // f16 accumulation drift is larger for large-n sums.
    assert!(diff < 5e-2, "logsumexp n=1024 f16 max |diff| = {diff:.2e}");
}

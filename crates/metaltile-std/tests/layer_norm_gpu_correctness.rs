//! GPU correctness for `mlx::layer_norm` — LayerNorm with weight and bias.
//!
//! `mt_layer_norm<T>`: `out[row, i] = (x[row, i] - mean) * is * w[i] + b[i]`
//! where `is = rsqrt(var + eps)` and mean/var computed over the row.
//!
//! ## DISPATCH INVARIANTS (mt_layer_norm)
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [TPG, 1, 1]`.
//! - `TPG = n / 4`; `n` must be a power of two multiple of `lsize`.
//! - The kernel uses `lsize` (TPG) as the stride, so `n` divisible by
//!   `TPG * 4` is required.
//!
//! CPU oracle: naive f32 layer-norm.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::layer_norm::mt_layer_norm;

fn cpu_layer_norm_f32(x: &[f32], w: &[f32], b: &[f32], n: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % n, 0);
    assert_eq!(w.len(), n);
    assert_eq!(b.len(), n);
    let rows = x.len() / n;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let base = r * n;
        let sum: f32 = x[base..base + n].iter().sum();
        let mean = sum / n as f32;
        let sq_sum: f32 = x[base..base + n].iter().map(|v| (v - mean).powi(2)).sum();
        let var = sq_sum / n as f32;
        let is = 1.0 / (var + eps).sqrt();
        for d in 0..n {
            out[base + d] = (x[base + d] - mean) * is * w[d] + b[d];
        }
    }
    out
}

fn run_layer_norm(
    x: &[f32],
    w: &[f32],
    b_vec: &[f32],
    eps: f32,
    dt: Dt,
    n: usize,
    rows: usize,
) -> Vec<f32> {
    // TPG = n / 4 — kernel invariant: each thread handles 4 consecutive elements.
    let tpg = n / 4;
    assert!((32..=1024).contains(&tpg), "TPG must be in [32, 1024]");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("b".into(), pack_bytes(b_vec, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; rows * n], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_layer_norm::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("layer_norm dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(rows * n);
    out
}

#[test]
fn layer_norm_matches_cpu_f32_n128() {
    let _g = gpu_lock();
    let (n, rows, eps) = (128usize, 4usize, 1e-5f32);
    let x: Vec<f32> = (0..rows * n).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
    let b_vec: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.02 - 0.04).collect();

    let expected = cpu_layer_norm_f32(&x, &w, &b_vec, n, eps);
    let actual = run_layer_norm(&x, &w, &b_vec, eps, Dt::F32, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-4, "layer_norm n=128 f32 max |diff| = {diff:.2e}");
}

#[test]
fn layer_norm_matches_cpu_f32_n512() {
    let _g = gpu_lock();
    let (n, rows, eps) = (512usize, 3usize, 1e-5f32);
    let x: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.08).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 11) as f32 * 0.05).collect();
    let b_vec: Vec<f32> = (0..n).map(|i| (i % 9) as f32 * 0.03 - 0.12).collect();

    let expected = cpu_layer_norm_f32(&x, &w, &b_vec, n, eps);
    let actual = run_layer_norm(&x, &w, &b_vec, eps, Dt::F32, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 5e-4, "layer_norm n=512 f32 max |diff| = {diff:.2e}");
}

#[test]
fn layer_norm_output_has_near_zero_mean_f32() {
    // A correct layer_norm normalizes each row to mean≈0, std≈1 (ignoring
    // affine w/b). With w=1 and b=0 the mean of the output must be ~0.
    let _g = gpu_lock();
    let (n, rows, eps) = (256usize, 2usize, 1e-5f32);
    let x: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let w: Vec<f32> = vec![1.0f32; n]; // identity scale
    let b_vec: Vec<f32> = vec![0.0f32; n]; // zero bias

    let actual = run_layer_norm(&x, &w, &b_vec, eps, Dt::F32, n, rows);

    for r in 0..rows {
        let row = &actual[r * n..(r + 1) * n];
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        assert!(
            mean.abs() < 1e-4,
            "layer_norm row {r} mean {mean:.2e} != 0 (unnormalized output?)"
        );
    }
}

#[test]
fn layer_norm_matches_cpu_f16_n256() {
    let _g = gpu_lock();
    let (n, rows, eps) = (256usize, 2usize, 1e-5f32);
    let x_f32: Vec<f32> = (0..rows * n).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
    let w_f32: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32 * 0.05).collect();
    let b_f32: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.02 - 0.04).collect();

    // Round inputs through f16 so the CPU oracle matches what the GPU reads.
    let x: Vec<f32> = x_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let w: Vec<f32> = w_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let b: Vec<f32> = b_f32.iter().map(|&v| Dt::F16.round(v)).collect();

    let expected = cpu_layer_norm_f32(&x, &w, &b, n, eps);
    let actual = run_layer_norm(&x_f32, &w_f32, &b_f32, eps, Dt::F16, n, rows);

    let diff = max_abs_diff(&actual, &expected);
    // f16 accumulation drift; wider tolerance.
    assert!(diff < 5e-2, "layer_norm n=256 f16 max |diff| = {diff:.2e}");
}

//! Per-head RMSNorm coverage test for Qwen3-style q_norm / k_norm.
//!
//! Qwen3 (and Qwen3.5 / Qwen3.6) apply RMSNorm to the projected Q
//! and K tensors **before** RoPE, normalising over the `head_dim`
//! axis with a per-head_dim weight vector. See e.g. MLX-LM's
//! `Qwen3Attention.__call__` (`q_norm(q)` → `rope(q)`).
//!
//! `mt_rms_norm` is generic over `N = tpg * 4`: each thread owns four
//! consecutive elements, the partial sum-of-squares is reduced across
//! the threadgroup, then the per-element output is rescaled. The
//! bench wires it at `n=4096, tpg=1024` for the hidden-axis case, but
//! the same kernel covers the per-head case at `n=head_dim,
//! tpg=head_dim/4` once the caller dispatches one threadgroup per
//! `(batch * token * n_heads)` row.
//!
//! This file pins that dispatch shape — at Qwen3-class head_dim=128
//! and a realistic batch*token*n_heads row count — and asserts the
//! output matches a CPU naive reference. The intent is to document
//! the per-head path is a dispatch decision, not a separate kernel
//! that needs to be written. If a future tuning pass introduces a
//! head_dim-specialised kernel, this test pins what the calling
//! contract has to keep working.
//!
//! macOS-gated: needs a Metal device.

#![cfg(target_os = "macos")]

use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard, OnceLock},
};

use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::mt_rms_norm;

/// Serialise GPU dispatches across the tests in this file. Same race
/// pattern other gpu integration suites in this crate hit when cargo
/// runs them in parallel; the in-file mutex is lighter than requiring
/// `--test-threads=1` at the command line.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

fn cpu_rms_norm_reference(x: &[f32], w: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let base = r * n;
        let ssq: f32 = (0..n).map(|i| x[base + i] * x[base + i]).sum();
        let rms = (ssq / n as f32 + eps).sqrt().recip();
        for i in 0..n {
            out[base + i] = x[base + i] * rms * w[i];
        }
    }
    out
}

#[test]
fn mt_rms_norm_per_head_qwen3_shape_f32() {
    let _g = gpu_lock();

    // Qwen3-8B / Qwen3-14B Q-norm dispatch: head_dim=128, applied per
    // (batch*token*head). 4 batches × 8 tokens × 32 heads = 1024 rows
    // is small enough to keep the CPU reference cheap, dense enough to
    // exercise the dispatch grid past a single threadgroup.
    let head_dim = 128usize;
    let rows = 4 * 8 * 32; // B*T*n_heads = 1024
    let eps = 1e-6_f32;

    let x: Vec<f32> = (0..rows * head_dim)
        .map(|i| 0.5 + ((i % 17) as f32) * 0.03 - ((i % 11) as f32) * 0.02)
        .collect();
    let w: Vec<f32> = (0..head_dim).map(|i| 1.0 + ((i % 13) as f32) * 0.01).collect();
    let expected = cpu_rms_norm_reference(&x, &w, rows, head_dim, eps);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), x.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("out".into(), vec![0u8; rows * head_dim * 4]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (head_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_rms_norm::kernel_ir_for(DType::F32);
    // Reduction mode is what `tile bench` uses for mt_rms_norm; it's
    // what tells the codegen to emit `lsize`/`tid`/`tgid` aliases the
    // kernel body references.
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;

    // One threadgroup per row. tpg = head_dim / 4 — each thread owns
    // 4 consecutive head_dim elements (the kernel's tile invariant).
    let tpg = head_dim / 4;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = i;
        }
    }
    // 1e-4: f32 accumulator + `reduce_sum` reordering. Same tolerance
    // the existing `mt_rms_norm` bench oracle uses.
    assert!(
        max_diff < 1e-4,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_rms_norm_per_head_qwen3_shape_f16() {
    let _g = gpu_lock();

    let head_dim = 128usize;
    let rows = 4 * 8 * 32;
    let eps = 1e-6_f32;

    let x_f32: Vec<f32> = (0..rows * head_dim)
        .map(|i| 0.5 + ((i % 17) as f32) * 0.03 - ((i % 11) as f32) * 0.02)
        .collect();
    let w_f32: Vec<f32> = (0..head_dim).map(|i| 1.0 + ((i % 13) as f32) * 0.01).collect();

    // Round through f16 so the oracle reflects what the kernel sees
    // post-load-cast.
    let round = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let x: Vec<f32> = x_f32.iter().map(|&v| round(v)).collect();
    let w: Vec<f32> = w_f32.iter().map(|&v| round(v)).collect();
    let expected = cpu_rms_norm_reference(&x, &w, rows, head_dim, eps);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert(
        "x".into(),
        x.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
    );
    buffers.insert(
        "w".into(),
        w.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
    );
    buffers.insert("out".into(), vec![0u8; rows * head_dim * 2]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (head_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_rms_norm::kernel_ir_for(DType::F16);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;

    let tpg = head_dim / 4;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = i;
        }
    }
    // f16: ~3 digits of precision at output magnitude ≈ 1. 5e-3
    // covers half's ULP + the reduce_sum reordering noise.
    assert!(
        max_diff < 5e-3,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

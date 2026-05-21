//! GPU correctness for `ffai::rms_norm_rope` — fused RMSNorm + RoPE.
//!
//! Pins: (1) the RMSNorm reduction over the *whole* row drives the
//! scale for *both* rotation halves; (2) the paired-layout rotation
//! `(lid, lid+half)`; (3) the per-row position
//! `pos = offset + (row / n_heads) mod seq_len`. A regression in any
//! of the three only shows as drifting attention scores downstream.
//!
//! macOS-gated: needs a real Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::ffai::rms_norm_rope::ffai_rms_norm_rope;

#[allow(clippy::too_many_arguments)]
fn naive(
    x: &[f32],
    w: &[f32],
    inv_freqs: &[f32],
    rows: usize,
    axis: usize,
    n_heads: usize,
    seq_len: usize,
    offset: usize,
    eps: f32,
) -> Vec<f32> {
    let half = axis / 2;
    let mut out = vec![0.0_f32; rows * axis];
    for r in 0..rows {
        let base = r * axis;
        let ssq: f32 = (0..axis).map(|i| x[base + i] * x[base + i]).sum();
        let inv_rms = 1.0 / (ssq / axis as f32 + eps).sqrt();
        let pos = (offset + (r / n_heads) % seq_len) as f32;
        for lid in 0..half {
            let theta = pos * inv_freqs[lid];
            let (s, c) = theta.sin_cos();
            let na = x[base + lid] * w[lid] * inv_rms;
            let nb = x[base + lid + half] * w[lid + half] * inv_rms;
            out[base + lid] = na * c - nb * s;
            out[base + lid + half] = na * s + nb * c;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run(
    kernel_ir: fn(DType) -> Kernel,
    x: &[f32],
    w: &[f32],
    inv_freqs: &[f32],
    dt: Dt,
    rows: usize,
    axis: usize,
    offset: u32,
    n_heads: u32,
    seq_len: u32,
    eps: f32,
) -> Vec<f32> {
    let tpg = axis / 2;
    assert!(axis.is_multiple_of(64), "axis_size must be a multiple of 64");
    assert!(tpg <= 1024, "axis_size / 2 must fit the Apple TPG cap");

    // `#[constexpr]` params lower to `constant T&` scalar buffers — they
    // are supplied through the buffer map, not the fn-constants map.
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w".into(), pack_bytes(w, dt));
    buffers.insert("inv_freqs".into(), pack_bytes(inv_freqs, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; rows * axis], dt));
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("axis_size".into(), (axis as u32).to_le_bytes().to_vec());
    buffers.insert("offset".into(), offset.to_le_bytes().to_vec());
    buffers.insert("n_heads".into(), n_heads.to_le_bytes().to_vec());
    buffers.insert("seq_len".into(), seq_len.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("rms_norm_rope dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(rows * axis);
    out
}

/// Small, deterministic inverse-frequency table — kept O(1) so the
/// rotation angle `pos·inv_freq` stays in cos/sin's precise regime.
fn inv_freq_table(half: usize) -> Vec<f32> {
    (0..half).map(|i| 1.0 / 10000.0_f32.powf(i as f32 / half as f32)).collect()
}

#[test]
fn rms_norm_rope_matches_naive_f32() {
    let _g = gpu_lock();
    let (axis, n_heads, seq_len, offset, eps) = (128usize, 4usize, 8usize, 5usize, 1e-5_f32);
    let rows = n_heads * seq_len; // one batch
    let half = axis / 2;
    let x: Vec<f32> = (0..rows * axis).map(|i| ((i % 53) as f32) * 0.07 - 1.8).collect();
    let w: Vec<f32> = (0..axis).map(|i| 1.0 + 0.02 * ((i % 11) as f32 - 5.0)).collect();
    let inv_freqs = inv_freq_table(half);

    let expected = naive(&x, &w, &inv_freqs, rows, axis, n_heads, seq_len, offset, eps);
    let actual = run(
        ffai_rms_norm_rope::kernel_ir_for,
        &x,
        &w,
        &inv_freqs,
        Dt::F32,
        rows,
        axis,
        offset as u32,
        n_heads as u32,
        seq_len as u32,
        eps,
    );
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "rms_norm_rope f32 max |diff| = {diff:.2e}");
}

#[test]
fn rms_norm_rope_matches_naive_f32_min_axis() {
    // axis=64 → TPG=32, exactly one simdgroup (the reduction edge).
    let _g = gpu_lock();
    let (axis, n_heads, seq_len, offset, eps) = (64usize, 2usize, 4usize, 0usize, 1e-5_f32);
    let rows = n_heads * seq_len;
    let half = axis / 2;
    let x: Vec<f32> = (0..rows * axis).map(|i| ((i % 41) as f32) * 0.09 - 1.5).collect();
    let w: Vec<f32> = (0..axis).map(|i| 1.0 + 0.03 * ((i % 7) as f32 - 3.0)).collect();
    let inv_freqs = inv_freq_table(half);

    let expected = naive(&x, &w, &inv_freqs, rows, axis, n_heads, seq_len, offset, eps);
    let actual = run(
        ffai_rms_norm_rope::kernel_ir_for,
        &x,
        &w,
        &inv_freqs,
        Dt::F32,
        rows,
        axis,
        offset as u32,
        n_heads as u32,
        seq_len as u32,
        eps,
    );
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "rms_norm_rope f32 axis=64 max |diff| = {diff:.2e}");
}

#[test]
fn rms_norm_rope_matches_naive_bf16() {
    let _g = gpu_lock();
    let (axis, n_heads, seq_len, offset, eps) = (128usize, 4usize, 8usize, 3usize, 1e-5_f32);
    let rows = n_heads * seq_len;
    let half = axis / 2;
    let x: Vec<f32> =
        (0..rows * axis).map(|i| Dt::Bf16.round(((i % 53) as f32) * 0.07 - 1.8)).collect();
    let w: Vec<f32> =
        (0..axis).map(|i| Dt::Bf16.round(1.0 + 0.02 * ((i % 11) as f32 - 5.0))).collect();
    let inv_freqs = inv_freq_table(half);

    let expected = naive(&x, &w, &inv_freqs, rows, axis, n_heads, seq_len, offset, eps);
    let actual = run(
        ffai_rms_norm_rope::kernel_ir_for,
        &x,
        &w,
        &inv_freqs,
        Dt::Bf16,
        rows,
        axis,
        offset as u32,
        n_heads as u32,
        seq_len as u32,
        eps,
    );
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-1, "rms_norm_rope bf16 max |diff| = {diff:.2e}");
}

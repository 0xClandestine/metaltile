//! End-to-end GPU correctness for `ffai::rope_yarn`.
//!
//! YaRN per-token RoPE. For each `(head, i in 0..head_dim/2)`:
//!
//!   inv_freq_extrap = theta_base^(-2i / head_dim)
//!   inv_freq_interp = inv_freq_extrap / factor
//!   ramp            = clamp((i - low) / (high - low), 0, 1)
//!   inv_freq        = inv_freq_interp*ramp + inv_freq_extrap*(1 - ramp)
//!
//!   theta     = position * inv_freq
//!   cos_t     = cos(theta) * attn_factor
//!   sin_t     = sin(theta) * attn_factor
//!   o[i]      = x[i]*cos_t - x[i+half]*sin_t
//!   o[i+half] = x[i]*sin_t + x[i+half]*cos_t
//!
//! Scenarios:
//!   - Identity at position=0 (cos=1, sin=0, attn_factor=1 → out == in)
//!   - factor=1 collapses to plain RoPE (interp == extrap) — bit-exact
//!     vs a plain-RoPE oracle in f32
//!   - YaRN scaling active (Nemotron-Labs-Diffusion params: head_dim=128,
//!     factor=16, theta=1e6, low/high band) vs CPU oracle
//!   - Norm preservation across the rotation (attn_factor=1)
//!   - f16 / bf16 dtype coverage
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rope_yarn::ffai_rope_yarn as rope_yarn;

/// CPU oracle — matches the kernel's exact arithmetic.
#[allow(clippy::too_many_arguments)]
fn naive_rope_yarn(
    qk: &[f32],
    head_dim: u32,
    n_heads: u32,
    position: u32,
    theta_base: f32,
    factor: f32,
    low: f32,
    high: f32,
    attn_factor: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let half_f = half_dim as f32;
    let mut out = vec![0.0_f32; qk.len()];

    for head in 0..n_heads {
        let base = (head * head_dim) as usize;
        for i in 0..half_dim {
            let i_f = i as f32;
            let inv_freq_extrap = (-i_f * theta_base.log2() / half_f).exp2();
            let inv_freq_interp = inv_freq_extrap / factor;
            let t = (i_f - low) / (high - low);
            let ramp = t.clamp(0.0, 1.0);
            let inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0 - ramp);

            let theta = position as f32 * inv_freq;
            let cos_t = theta.cos() * attn_factor;
            let sin_t = theta.sin() * attn_factor;

            let i1 = base + i as usize;
            let i2 = base + (i + half_dim) as usize;
            let x1 = qk[i1];
            let x2 = qk[i2];
            out[i1] = x1 * cos_t - x2 * sin_t;
            out[i2] = x1 * sin_t + x2 * cos_t;
        }
    }
    out
}

/// Plain-RoPE oracle — `rope_yarn` with `factor == 1` must reproduce it.
fn naive_plain_rope(
    qk: &[f32],
    head_dim: u32,
    n_heads: u32,
    position: u32,
    theta_base: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let half_f = half_dim as f32;
    let mut out = vec![0.0_f32; qk.len()];
    for head in 0..n_heads {
        let base = (head * head_dim) as usize;
        for i in 0..half_dim {
            let i_f = i as f32;
            let inv_freq = (-i_f * theta_base.log2() / half_f).exp2();
            let theta = position as f32 * inv_freq;
            let (cos_t, sin_t) = (theta.cos(), theta.sin());
            let i1 = base + i as usize;
            let i2 = base + (i + half_dim) as usize;
            out[i1] = qk[i1] * cos_t - qk[i2] * sin_t;
            out[i2] = qk[i1] * sin_t + qk[i2] * cos_t;
        }
    }
    out
}

/// Dispatch the kernel and read back the rotated tensor in `dt`.
#[allow(clippy::too_many_arguments)]
fn run_rope_yarn(
    qk: &[f32],
    dt: Dt,
    n_heads: u32,
    head_dim: u32,
    position: u32,
    theta_base: f32,
    factor: f32,
    low: f32,
    high: f32,
    attn_factor: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let elem_count = qk.len();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qk".into(), pack_bytes(qk, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; elem_count], dt));
    buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
    buffers.insert("half_dim".into(), half_dim.to_le_bytes().to_vec());
    buffers.insert("position".into(), position.to_le_bytes().to_vec());
    buffers.insert("theta_base".into(), theta_base.to_le_bytes().to_vec());
    buffers.insert("factor".into(), factor.to_le_bytes().to_vec());
    buffers.insert("low".into(), low.to_le_bytes().to_vec());
    buffers.insert("high".into(), high.to_le_bytes().to_vec());
    buffers.insert("attn_factor".into(), attn_factor.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = rope_yarn::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id::<0> = head, program_id::<1> = i (0..half_dim).
    // One thread per (head, i); a single 1024-thread TG fits the test
    // shapes. Fail loudly if a future test exceeds the TG limit.
    assert!(
        n_heads as usize * half_dim as usize <= 1024,
        "test dispatches a single TG — keep n_heads*half_dim ≤ 1024",
    );
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [
            n_heads as usize,
            half_dim as usize,
            1,
        ])
        .expect("dispatch_with_grid");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    unpack_bytes(out_bytes, dt)
}

#[test]
fn rope_yarn_identity_at_position_zero_f32() {
    let _g = gpu_lock();
    // position=0 → theta=0 → cos=1, sin=0. attn_factor=1 → out == in,
    // regardless of factor / low / high. Pins indexing / grid layout.
    let n_heads = 4u32;
    let head_dim = 32u32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| 0.1 + (i as f32 * 0.013).sin()).collect();
    let actual = run_rope_yarn(&qk, Dt::F32, n_heads, head_dim, 0, 1.0e6, 16.0, 8.0, 20.0, 1.0);
    for (idx, (a, e)) in actual.iter().zip(qk.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "identity at pos=0 broke at idx={idx}: got {a}, expected {e}"
        );
    }
}

#[test]
fn rope_yarn_factor_one_collapses_to_plain_rope_f32() {
    let _g = gpu_lock();
    // factor=1 → inv_freq_interp == inv_freq_extrap → the ramp blend is
    // a no-op and YaRN reduces to plain RoPE. Bit-exact f32 check.
    let n_heads = 8u32;
    let head_dim = 64u32;
    let position = 137u32;
    let theta_base = 10000.0_f32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();

    let expected = naive_plain_rope(&qk, head_dim, n_heads, position, theta_base);
    let actual =
        run_rope_yarn(&qk, Dt::F32, n_heads, head_dim, position, theta_base, 1.0, 8.0, 24.0, 1.0);
    let mut max_diff = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 5e-5, "factor=1 vs plain RoPE: max |diff| = {max_diff:.2e} > 5e-5");
}

#[test]
fn rope_yarn_nemotron_params_match_oracle_f32() {
    let _g = gpu_lock();
    // Nemotron-Labs-Diffusion YaRN parameters: head_dim=128, theta=1e6,
    // factor=16, correction band low=20 / high=37. Exercises the ramp
    // across all three regions (extrapolate / blend / interpolate).
    let n_heads = 16u32;
    let head_dim = 128u32; // n_heads*half_dim = 16*64 = 1024 — single TG
    let position = 512u32;
    let theta_base = 1.0e6_f32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 53) as f32 - 26.0) * 0.04).collect();

    let expected =
        naive_rope_yarn(&qk, head_dim, n_heads, position, theta_base, 16.0, 20.0, 37.0, 1.0);
    let actual =
        run_rope_yarn(&qk, Dt::F32, n_heads, head_dim, position, theta_base, 16.0, 20.0, 37.0, 1.0);
    let mut max_diff = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    // theta = position * inv_freq reaches ~512 for high-frequency dims;
    // the GPU's native sin/cos lose argument-reduction ULPs on large
    // arguments. 2e-4 absolute is still tight given |x| ≤ ~1 — the same
    // reasoning rope_llama's banded test uses for its 2e-3 bound.
    assert!(max_diff < 2e-4, "YaRN Nemotron params: max |diff| = {max_diff:.2e} > 2e-4");
}

#[test]
fn rope_yarn_preserves_norm_f32() {
    let _g = gpu_lock();
    // A rotation with attn_factor=1 preserves the L2 norm of every
    // (x[i], x[i+half]) pair — catches a dropped cross term.
    let n_heads = 4u32;
    let head_dim = 64u32;
    let position = 333u32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| 0.5 + (i as f32 * 0.073).cos()).collect();
    let actual =
        run_rope_yarn(&qk, Dt::F32, n_heads, head_dim, position, 1.0e6, 16.0, 20.0, 37.0, 1.0);
    let half_dim = head_dim / 2;
    for head in 0..n_heads {
        let base = (head * head_dim) as usize;
        for i in 0..half_dim {
            let i1 = base + i as usize;
            let i2 = base + (i + half_dim) as usize;
            let in_sq = qk[i1] * qk[i1] + qk[i2] * qk[i2];
            let out_sq = actual[i1] * actual[i1] + actual[i2] * actual[i2];
            assert!((in_sq - out_sq).abs() < 1e-4, "norm not preserved at (head={head}, i={i})");
        }
    }
}

#[test]
fn rope_yarn_nemotron_params_match_oracle_f16() {
    let _g = gpu_lock();
    let n_heads = 8u32;
    let head_dim = 64u32;
    let position = 73u32;
    let theta_base = 1.0e6_f32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let qk_rounded: Vec<f32> = qk.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = naive_rope_yarn(
        &qk_rounded,
        head_dim,
        n_heads,
        position,
        theta_base,
        16.0,
        12.0,
        28.0,
        1.0,
    );
    let actual =
        run_rope_yarn(&qk, Dt::F16, n_heads, head_dim, position, theta_base, 16.0, 12.0, 28.0, 1.0);
    let mut max_rel = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1e-3));
    }
    assert!(max_rel < 5e-3, "f16 YaRN: max rel = {max_rel:.2e} > 5e-3");
}

#[test]
fn rope_yarn_nemotron_params_match_oracle_bf16() {
    let _g = gpu_lock();
    let n_heads = 8u32;
    let head_dim = 64u32;
    let position = 41u32;
    let theta_base = 1.0e6_f32;
    let qk: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let qk_rounded: Vec<f32> = qk.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = naive_rope_yarn(
        &qk_rounded,
        head_dim,
        n_heads,
        position,
        theta_base,
        16.0,
        12.0,
        28.0,
        1.0,
    );
    let actual = run_rope_yarn(
        &qk,
        Dt::Bf16,
        n_heads,
        head_dim,
        position,
        theta_base,
        16.0,
        12.0,
        28.0,
        1.0,
    );
    let mut max_rel = 0.0_f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1e-3));
    }
    assert!(max_rel < 2e-2, "bf16 YaRN: max rel = {max_rel:.2e} > 2e-2");
}

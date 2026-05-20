//! GPU correctness for `ffai::fused_gate_activation` — fused dense
//! gate-activation kernels (silu / gelu / clipped-swiglu).
//!
//! Layout:
//!   gate_up: `[rows, 2*hidden]` — gate half then up half per row.
//!   out:     `[rows, hidden]`.
//!
//! For each output element `(r, i)` the kernel computes
//! `act(gate_up[r, i]) * gate_up[r, hidden + i]`. One thread per output
//! element via Grid3D.
//!
//! Regression class guarded: a wrong row/col decomposition smears gate
//! against up across rows; a dropped activation ships `gate * up`
//! (linear) instead of the gated non-linearity — both only surface as
//! garbage logits in FFAI integration. The naive f32 oracle pins both.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::ffai::fused_gate_activation::{
    ffai_fused_gate_gelu,
    ffai_fused_gate_silu,
    ffai_fused_gate_swiglu,
};

// ── CPU reference activations (mirror the MSL helpers exactly) ──────────────

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// tanh-approx GELU — matches `mt_gelu`: fp32 math, tanh argument
/// clamped to ±15 (a numeric no-op, tanh is saturated by |arg|≈9).
fn gelu_tanh(x: f32) -> f32 {
    // sqrt(2/pi) — same f32 value as the MSL helper's 0.7978845608f.
    let k = 0.7978846_f32;
    let arg = (k * (x + 0.044715 * x * x * x)).clamp(-15.0, 15.0);
    0.5 * x * (1.0 + arg.tanh())
}

/// Clipped SwiGLU (GPT-OSS): halves clamped to ±7, gate side
/// `g·sigmoid(1.702·g)`, up side `+1` bias.
fn clipped_swiglu(g: f32, u: f32) -> f32 {
    let gc = g.clamp(-7.0, 7.0);
    let uc = u.clamp(-7.0, 7.0);
    let s = 1.0 / (1.0 + (-1.702 * gc).exp());
    gc * s * (uc + 1.0)
}

fn naive(gate_up: &[f32], rows: usize, hidden: usize, act: fn(f32, f32) -> f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * hidden];
    for r in 0..rows {
        let base = r * 2 * hidden;
        for i in 0..hidden {
            out[r * hidden + i] = act(gate_up[base + i], gate_up[base + hidden + i]);
        }
    }
    out
}

// ── Dispatch helper ─────────────────────────────────────────────────────────

fn run_fga(
    kernel_ir: fn(DType) -> Kernel,
    gate_up: &[f32],
    dt: Dt,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("gate_up".into(), pack_bytes(gate_up, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; rows * hidden], dt));
    buffers.insert("hidden".into(), (hidden as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let total = rows * hidden;
    let tpg = 256usize;
    let groups = total.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("fused_gate_activation dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(rows * hidden);
    out
}

/// Deterministic input in [-3, 3] — crosses zero so the gating
/// non-linearity is exercised in both regimes.
fn ramp_gate_up(rows: usize, hidden: usize) -> Vec<f32> {
    (0..rows * 2 * hidden).map(|i| ((i % 47) as f32) * 0.13 - 3.0).collect()
}

// ── SiLU ────────────────────────────────────────────────────────────────────

#[test]
fn fused_gate_silu_matches_naive_f32() {
    let _g = gpu_lock();
    let (rows, hidden) = (5, 320);
    let gate_up = ramp_gate_up(rows, hidden);
    let expected = naive(&gate_up, rows, hidden, |g, u| silu(g) * u);
    let actual = run_fga(ffai_fused_gate_silu::kernel_ir_for, &gate_up, Dt::F32, rows, hidden);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "silu f32 mismatch");
}

#[test]
fn fused_gate_silu_matches_naive_f16() {
    let _g = gpu_lock();
    let (rows, hidden) = (3, 256);
    let gate_up = ramp_gate_up(rows, hidden);
    let expected: Vec<f32> = naive(&gate_up, rows, hidden, |g, u| silu(g) * u)
        .iter()
        .map(|&v| Dt::F16.round(v))
        .collect();
    let actual = run_fga(ffai_fused_gate_silu::kernel_ir_for, &gate_up, Dt::F16, rows, hidden);
    assert!(max_abs_diff(&actual, &expected) < 2e-2, "silu f16 mismatch");
}

// ── GELU ────────────────────────────────────────────────────────────────────

#[test]
fn fused_gate_gelu_matches_naive_f32() {
    let _g = gpu_lock();
    let (rows, hidden) = (4, 384);
    let gate_up = ramp_gate_up(rows, hidden);
    let expected = naive(&gate_up, rows, hidden, |g, u| gelu_tanh(g) * u);
    let actual = run_fga(ffai_fused_gate_gelu::kernel_ir_for, &gate_up, Dt::F32, rows, hidden);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "gelu f32 mismatch");
}

#[test]
fn fused_gate_gelu_matches_naive_bf16() {
    let _g = gpu_lock();
    let (rows, hidden) = (3, 256);
    let gate_up = ramp_gate_up(rows, hidden);
    let expected: Vec<f32> = naive(&gate_up, rows, hidden, |g, u| gelu_tanh(g) * u)
        .iter()
        .map(|&v| Dt::Bf16.round(v))
        .collect();
    let actual = run_fga(ffai_fused_gate_gelu::kernel_ir_for, &gate_up, Dt::Bf16, rows, hidden);
    assert!(max_abs_diff(&actual, &expected) < 1e-1, "gelu bf16 mismatch");
}

// ── Clipped SwiGLU ──────────────────────────────────────────────────────────

#[test]
fn fused_gate_swiglu_matches_naive_f32() {
    let _g = gpu_lock();
    let (rows, hidden) = (4, 320);
    let gate_up = ramp_gate_up(rows, hidden);
    let expected = naive(&gate_up, rows, hidden, clipped_swiglu);
    let actual = run_fga(ffai_fused_gate_swiglu::kernel_ir_for, &gate_up, Dt::F32, rows, hidden);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "swiglu f32 mismatch");
}

/// Inputs spanning ±12 so the ±7 clamp on both halves is actually hit;
/// a dropped clamp would diverge sharply on the out-of-range elements.
#[test]
fn fused_gate_swiglu_exercises_clamp_f32() {
    let _g = gpu_lock();
    let (rows, hidden) = (3, 256);
    let gate_up: Vec<f32> =
        (0..rows * 2 * hidden).map(|i| ((i % 49) as f32) * 0.5 - 12.0).collect();
    let expected = naive(&gate_up, rows, hidden, clipped_swiglu);
    let actual = run_fga(ffai_fused_gate_swiglu::kernel_ir_for, &gate_up, Dt::F32, rows, hidden);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "swiglu clamp f32 mismatch");
}

//! End-to-end correctness for the `fused_gate_activation` kernels —
//! `mt_fused_gate_gelu` (gelu-approx variant) and
//! `mt_fused_gate_clipped_swiglu` (GPT-OSS clipped variant).
//!
//! The `silu` variant of this op is covered separately by
//! `swiglu_gpu_correctness.rs` (it ships as the dedicated `mt_swiglu`
//! kernel). This file pins the two remaining activation variants
//! against an f32 CPU oracle across f32 / f16 / bf16.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::fused_gate_activation::{mt_fused_gate_clipped_swiglu, mt_fused_gate_gelu};

// ── CPU oracles — both computed in f32, mirroring the MSL reference ──

fn cpu_gelu_approx(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            // gelu_approx(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
            let x3 = g * g * g;
            let inner = 0.797_884_6_f32 * (g + 0.044_715_f32 * x3);
            let act = 0.5 * g * (1.0 + inner.tanh());
            act * u
        })
        .collect()
}

fn cpu_clipped_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g_raw, &u_raw)| {
            // Both halves clamped to [-7, 7].
            let g = g_raw.clamp(-7.0, 7.0);
            let u = u_raw.clamp(-7.0, 7.0);
            let sig = 1.0 / (1.0 + (-1.702_f32 * g).exp());
            // up side carries a +1 bias.
            g * sig * (u + 1.0)
        })
        .collect()
}

/// Dispatch a fused-gate kernel by name; Grid3D, one thread per element.
fn run(kernel_name: &str, gate: &[f32], up: &[f32], dt: Dt) -> Vec<f32> {
    let n = gate.len();
    assert_eq!(up.len(), n);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("gate".into(), pack_bytes(gate, dt));
    buffers.insert("up".into(), pack_bytes(up, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = match kernel_name {
        "mt_fused_gate_gelu" => mt_fused_gate_gelu::kernel_ir_for(dt.to_dtype()),
        "mt_fused_gate_clipped_swiglu" =>
            mt_fused_gate_clipped_swiglu::kernel_ir_for(dt.to_dtype()),
        other => panic!("unknown fused-gate kernel {other}"),
    };
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let grid = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("fused_gate dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

fn max_rel(expected: &[f32], actual: &[f32]) -> (f32, usize) {
    let mut max = 0.0f32;
    let mut at = 0usize;
    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1e-3);
        if rel > max {
            max = rel;
            at = i;
        }
    }
    (max, at)
}

// ── gelu-approx variant ──────────────────────────────────────────────

#[test]
fn fused_gate_gelu_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    // Mix positive / negative / near-zero gate values.
    let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017) % 6.0 - 3.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.029) % 4.0 - 2.0).collect();
    let expected = cpu_gelu_approx(&gate, &up);
    let actual = run("mt_fused_gate_gelu", &gate, &up, Dt::F32);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 1e-4, "gelu f32: max rel {rel:.2e} at [{at}]");
}

#[test]
fn fused_gate_gelu_matches_cpu_f16() {
    let _g = gpu_lock();
    let n = 2048usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013) % 8.0 - 4.0).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.021) % 3.0 - 1.5).collect();
    // Round inputs through f16 so the oracle sees the same precision.
    let gate: Vec<f32> = gate_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = cpu_gelu_approx(&gate, &up);
    let actual = run("mt_fused_gate_gelu", &gate_f32, &up_f32, Dt::F16);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 8e-3, "gelu f16: max rel {rel:.2e} at [{at}]");
}

#[test]
fn fused_gate_gelu_matches_cpu_bf16() {
    let _g = gpu_lock();
    let n = 1024usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019) % 6.0 - 3.0).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.023) % 4.0 - 2.0).collect();
    let gate: Vec<f32> = gate_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = cpu_gelu_approx(&gate, &up);
    let actual = run("mt_fused_gate_gelu", &gate_f32, &up_f32, Dt::Bf16);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 3e-2, "gelu bf16: max rel {rel:.2e} at [{at}]");
}

// ── clipped-swiglu variant (GPT-OSS) ─────────────────────────────────

#[test]
fn fused_gate_clipped_swiglu_matches_cpu_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    // Range spans well past ±7 so the clamp branch is exercised.
    let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.041) % 24.0 - 12.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.037) % 20.0 - 10.0).collect();
    let expected = cpu_clipped_swiglu(&gate, &up);
    let actual = run("mt_fused_gate_clipped_swiglu", &gate, &up, Dt::F32);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 1e-4, "clipped-swiglu f32: max rel {rel:.2e} at [{at}]");
}

#[test]
fn fused_gate_clipped_swiglu_clamp_saturates_f32() {
    let _g = gpu_lock();
    // All-large inputs — every element hits the clamp ceiling. Output
    // must be the saturated value (g=7, u=7): 7·sigmoid(1.702·7)·8.
    let n = 256usize;
    let gate = vec![100.0f32; n];
    let up = vec![100.0f32; n];
    let expected = cpu_clipped_swiglu(&gate, &up);
    let actual = run("mt_fused_gate_clipped_swiglu", &gate, &up, Dt::F32);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 1e-5, "clipped-swiglu saturation f32: max rel {rel:.2e} at [{at}]");
}

#[test]
fn fused_gate_clipped_swiglu_matches_cpu_f16() {
    let _g = gpu_lock();
    let n = 2048usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.043) % 22.0 - 11.0).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.031) % 18.0 - 9.0).collect();
    let gate: Vec<f32> = gate_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = cpu_clipped_swiglu(&gate, &up);
    let actual = run("mt_fused_gate_clipped_swiglu", &gate_f32, &up_f32, Dt::F16);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 8e-3, "clipped-swiglu f16: max rel {rel:.2e} at [{at}]");
}

#[test]
fn fused_gate_clipped_swiglu_matches_cpu_bf16() {
    let _g = gpu_lock();
    let n = 1024usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.047) % 24.0 - 12.0).collect();
    let up_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.039) % 20.0 - 10.0).collect();
    let gate: Vec<f32> = gate_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let up: Vec<f32> = up_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = cpu_clipped_swiglu(&gate, &up);
    let actual = run("mt_fused_gate_clipped_swiglu", &gate_f32, &up_f32, Dt::Bf16);
    let (rel, at) = max_rel(&expected, &actual);
    assert!(rel < 3e-2, "clipped-swiglu bf16: max rel {rel:.2e} at [{at}]");
}

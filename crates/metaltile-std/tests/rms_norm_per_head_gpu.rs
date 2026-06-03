//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Per-head RMSNorm coverage test for Qwen3-style q_norm / k_norm.
//!
//! Qwen3 / Qwen3.5 / Qwen3.6 apply RMSNorm to projected Q and K
//! **before** RoPE, normalising over the `head_dim` axis with a
//! per-head_dim weight vector (e.g. `Qwen3Attention.__call__:
//! q_norm(q) → rope(q)` in MLX-LM).
//!
//! `mt_rms_norm` is generic over `N = tpg * 4`: each thread owns
//! four consecutive elements, the partial sum-of-squares reduces
//! across the threadgroup. The bench wires it at `n=4096, tpg=1024`
//! for the hidden-axis case, but the same kernel covers per-head at
//! `n = head_dim, tpg = head_dim/4` with the per-head_dim weight
//! broadcast across all `(batch*token*n_heads)` rows.
//!
//! This file pins that dispatch contract at every head_dim value
//! current production LLMs use:
//!
//! | head_dim | kernel               | tpg | Models                                  |
//! |----------|----------------------|-----|-----------------------------------------|
//! | 64       | `mt_rms_norm_small`  | 32  | older 7B-class architectures                  |
//! | 128      | `mt_rms_norm`        | 32  | Qwen3-8B/14B/32B, Qwen3-class |
//! | 256      | `mt_rms_norm`        | 64  | Gemma-2/3 (E2B, 9B), Phi-3-medium       |
//!
//! `mt_rms_norm` owns 4 consecutive elements per thread (better ILP
//! at large head_dim). `mt_rms_norm_small` owns 2 elements per thread
//! so head_dim=64 still hits the tpg=32 single-simdgroup minimum
//! that the 4-element variant misses (sub-simdgroup `simd_sum` reads
//! undefined inactive-lane partials → 600× output magnitude blowup
//! that this file catches if anyone regresses the dispatcher).
//!
//! Each row covered at f32, f16, and bf16 — the per-load
//! `.cast::<f32>()` into the f32 accumulator chain is dtype-agnostic
//! but the output store rounds through T, and bf16's 7-bit mantissa
//! drifts faster than f16's 10-bit at typical normalised-value
//! magnitudes.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::{mt_rms_norm, mt_rms_norm_small};

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

/// Run mt_rms_norm at (head_dim, rows, dtype) and compare against a
/// CPU naive reference. `tol_abs` is the absolute max-diff envelope;
/// the caller picks it per dtype (f32 ≈ 1e-4, f16 ≈ 5e-3, bf16 ≈ 5e-2).
fn run_and_check(head_dim: usize, rows: usize, dtype: Dt, tol_abs: f32) {
    let _g = gpu_lock();

    let eps = 1e-6_f32;
    // Magnitudes around 0.3-0.9 — small enough that ULP doesn't
    // blow up the relative comparison, large enough that
    // rsqrt(ssq/n + eps) doesn't saturate.
    let x_f32: Vec<f32> = (0..rows * head_dim)
        .map(|i| 0.5 + ((i % 17) as f32) * 0.03 - ((i % 11) as f32) * 0.02)
        .collect();
    let w_f32: Vec<f32> = (0..head_dim).map(|i| 1.0 + ((i % 13) as f32) * 0.01).collect();

    let x: Vec<f32> = x_f32.iter().map(|&v| dtype.round(v)).collect();
    let w: Vec<f32> = w_f32.iter().map(|&v| dtype.round(v)).collect();
    let expected = cpu_rms_norm_reference(&x, &w, rows, head_dim, eps);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, dtype));
    buffers.insert("w".into(), pack_bytes(&w, dtype));
    buffers.insert("out".into(), vec![0u8; rows * head_dim * dtype.bytes()]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (head_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    // Choose kernel by head_dim:
    //   head_dim ≥ 128 → mt_rms_norm  (4 elems/thread, tpg=head_dim/4)
    //   head_dim < 128 → mt_rms_norm_small  (2 elems/thread,
    //                                        tpg=head_dim/2 ≥ 32)
    // The 4-element variant has better ILP per lane but needs tpg≥32,
    // which only holds when head_dim ≥ 128. The 2-element variant
    // covers the head_dim=64 case (older 7B-class architectures) at
    // tpg=32 — the single-simdgroup minimum.
    let (mut kernel, tpg) = if head_dim >= 128 {
        (mt_rms_norm::kernel_ir_for(dtype.to_dtype()), head_dim / 4)
    } else {
        (mt_rms_norm_small::kernel_ir_for(dtype.to_dtype()), head_dim / 2)
    };
    kernel.mode = metaltile::core::ir::KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = unpack_bytes(out_bytes, dtype);
    assert_eq!(actual.len(), expected.len(), "output element count");

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = i;
        }
    }
    assert!(
        max_diff < tol_abs,
        "head_dim={head_dim} rows={rows} dtype={:?}: max |diff| = {max_diff:.2e} at index {max_at} \
         (expected {:.6}, got {:.6})",
        dtype.to_dtype(),
        expected[max_at],
        actual[max_at],
    );
}

// ── head_dim = 64 (older 7B-class architectures — uses mt_rms_norm_small) ─

#[test]
fn mt_rms_norm_per_head_small_head_dim_f32() {
    // older 7B-class architectures q_norm-like dispatch — head_dim=64
    // routes to `mt_rms_norm_small` (2 elems/thread, tpg=32 hits
    // the single-simdgroup minimum). 512 rows = enough to exercise
    // multi-TG grid without blowing up the CPU reference cost.
    run_and_check(64, 512, Dt::F32, 1e-4);
}

#[test]
fn mt_rms_norm_per_head_small_head_dim_f16() { run_and_check(64, 512, Dt::F16, 5e-3); }

#[test]
fn mt_rms_norm_per_head_small_head_dim_bf16() { run_and_check(64, 512, Dt::Bf16, 5e-2); }

// ── head_dim = 128 (Qwen3-class) ─────────────────────────────────────

#[test]
fn mt_rms_norm_per_head_qwen3_shape_f32() {
    // Qwen3-8B / Qwen3-14B Q-norm dispatch: 4 batches × 8 tokens
    // × 32 heads = 1024 rows. Same shape the bench-runner uses for
    // hidden RMSNorm at n=4096 — exercises multi-TG grid + full
    // simdgroup reduce.
    run_and_check(128, 1024, Dt::F32, 1e-4);
}

#[test]
fn mt_rms_norm_per_head_qwen3_shape_f16() { run_and_check(128, 1024, Dt::F16, 5e-3); }

#[test]
fn mt_rms_norm_per_head_qwen3_shape_bf16() {
    // bf16's 7-bit mantissa drifts faster than f16's 10-bit; envelope
    // 5e-2 ≈ 1 ULP at the normalised value magnitudes here.
    run_and_check(128, 1024, Dt::Bf16, 5e-2);
}

// ── head_dim = 256 (Gemma-2 / Gemma-3, Phi-3-medium) ─────────────────

#[test]
fn mt_rms_norm_per_head_gemma_head_dim_f32() {
    // Gemma-2 / Gemma-3 use head_dim=256. tpg=64 = 2 simdgroups,
    // so reduce_sum needs to cross-simdgroup-reduce (not just
    // simd_sum). This is the path that breaks if the codegen
    // regresses the multi-SG reduce.
    run_and_check(256, 256, Dt::F32, 1e-4);
}

#[test]
fn mt_rms_norm_per_head_gemma_head_dim_f16() { run_and_check(256, 256, Dt::F16, 5e-3); }

#[test]
fn mt_rms_norm_per_head_gemma_head_dim_bf16() { run_and_check(256, 256, Dt::Bf16, 5e-2); }

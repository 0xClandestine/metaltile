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
//! This file pins that dispatch contract at the head_dim values
//! production Qwen3 / Gemma models use:
//!
//! | head_dim | tpg | Models                                  |
//! |----------|-----|-----------------------------------------|
//! | 128      | 32  | Qwen3-8B / 14B / 32B, Qwen3-class |
//! | 256      | 64  | Gemma-2/3 (E2B, 9B), Phi-3-medium      |
//!
//! Each row covered at f32, f16, and bf16 — the per-load `.cast::<f32>()`
//! into the f32 accumulator chain is dtype-agnostic but the output
//! store rounds through T, and bf16's 7-bit mantissa drifts faster
//! than f16's 10-bit at typical normalised-value magnitudes.
//!
//! ## head_dim < 128 limitation
//!
//! `mt_rms_norm`'s tile invariant is 4 consecutive elements per thread,
//! so `tpg = head_dim / 4`. At head_dim < 128 → tpg < 32 (sub-simdgroup),
//! `reduce_sum` lowers to `simd_sum` which always sums across the
//! full 32-lane simdgroup — the inactive lanes (positions ≥ tpg)
//! contribute undefined `partial_ssq` values and the ssq blows up
//! by orders of magnitude. older 7B-class architectures and friends
//! (head_dim=64) need a separate kernel variant (or a tpg=32 layout
//! with 2 elements per thread); tracked as a follow-up.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::mt_rms_norm;

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

#[derive(Clone, Copy)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }
    fn to_dtype(self) -> DType {
        match self {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::Bf16 => DType::BF16,
        }
    }
    /// Round-trip a value through this dtype so the oracle reflects
    /// what the kernel reads post-load-cast.
    fn round(self, v: f32) -> f32 {
        match self {
            Dtype::F32 => v,
            Dtype::F16 => half::f16::from_f32(v).to_f32(),
            Dtype::Bf16 => half::bf16::from_f32(v).to_f32(),
        }
    }
    fn pack(self, vals: &[f32]) -> Vec<u8> {
        match self {
            Dtype::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::F16 =>
                vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
            Dtype::Bf16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect(),
        }
    }
    fn unpack(self, bytes: &[u8]) -> Vec<f32> {
        match self {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
        }
    }
}

/// Run mt_rms_norm at (head_dim, rows, dtype) and compare against a
/// CPU naive reference. `tol_abs` is the absolute max-diff envelope;
/// the caller picks it per dtype (f32 ≈ 1e-4, f16 ≈ 5e-3, bf16 ≈ 5e-2).
fn run_and_check(head_dim: usize, rows: usize, dtype: Dtype, tol_abs: f32) {
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
    buffers.insert("x".into(), dtype.pack(&x));
    buffers.insert("w".into(), dtype.pack(&w));
    buffers.insert("out".into(), vec![0u8; rows * head_dim * dtype.bytes()]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (head_dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_rms_norm::kernel_ir_for(dtype.to_dtype());
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;

    // tpg = head_dim/4 — each thread owns 4 consecutive head_dim
    // elements (the kernel's tile invariant). head_dim=128 →
    // tpg=32 (single simdgroup), head_dim=256 → tpg=64 (2
    // simdgroups). reduce_sum handles both single-SG and multi-SG.
    // head_dim < 128 hits the limitation documented in the file
    // header.
    let tpg = head_dim / 4;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = dtype.unpack(out_bytes);
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

// ── head_dim = 128 (Qwen3-class) ─────────────────────────────────────

#[test]
fn mt_rms_norm_per_head_qwen3_shape_f32() {
    // Qwen3-8B / Qwen3-14B Q-norm dispatch: 4 batches × 8 tokens
    // × 32 heads = 1024 rows. Same shape the bench-runner uses for
    // hidden RMSNorm at n=4096 — exercises multi-TG grid + full
    // simdgroup reduce.
    run_and_check(128, 1024, Dtype::F32, 1e-4);
}

#[test]
fn mt_rms_norm_per_head_qwen3_shape_f16() { run_and_check(128, 1024, Dtype::F16, 5e-3); }

#[test]
fn mt_rms_norm_per_head_qwen3_shape_bf16() {
    // bf16's 7-bit mantissa drifts faster than f16's 10-bit; envelope
    // 5e-2 ≈ 1 ULP at the normalised value magnitudes here.
    run_and_check(128, 1024, Dtype::Bf16, 5e-2);
}

// ── head_dim = 256 (Gemma-2 / Gemma-3, Phi-3-medium) ─────────────────

#[test]
fn mt_rms_norm_per_head_gemma_head_dim_f32() {
    // Gemma-2 / Gemma-3 use head_dim=256. tpg=64 = 2 simdgroups,
    // so reduce_sum needs to cross-simdgroup-reduce (not just
    // simd_sum). This is the path that breaks if the codegen
    // regresses the multi-SG reduce.
    run_and_check(256, 256, Dtype::F32, 1e-4);
}

#[test]
fn mt_rms_norm_per_head_gemma_head_dim_f16() { run_and_check(256, 256, Dtype::F16, 5e-3); }

#[test]
fn mt_rms_norm_per_head_gemma_head_dim_bf16() { run_and_check(256, 256, Dtype::Bf16, 5e-2); }

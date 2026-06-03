//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness oracle for `mt_sdpa_prefill_mma_bf16` — the
//! bf16-emulated MMA prefill variant in
//! `crates/metaltile-std/src/mlx/steel/attn/steel_attention_mma_bf16.rs`.
//!
//! The bf16 variant is the M2-family bf16 routing target of
//! `sdpa_prefill_mma_for` (`is_pre_m3_bf16` arm). Its sibling
//! `mt_sdpa_prefill_mma` already has long-T + batched coverage in
//! `sdpa_prefill_mma_long_t.rs`, but all of those tests run at f32 and
//! exercise the *sibling* kernel — leaving the bf16 single-Q dd-loop
//! variant without a direct correctness oracle. The
//! `/tmp/coverage_audit_all_kernels.md` audit flagged this gap; this
//! file closes it.
//!
//! The kernel signature mirrors `mt_sdpa_prefill_mma`: same Q/K/V/O
//! tensors + `q_len / k_len / gqa_factor / n_q_heads / n_kv_heads /
//! scale` constexprs. Same geometry too — BQ=32, BK=16, BD=128, WM=4,
//! WN=1, tpg=128, grid `(q_len/32, n_q_heads, batch)` via
//! `SimdGroup2D`. The kernel runs at the input dtype (no f32 retype
//! of Q/K/V/P frags) so bf16 inputs ⇒ bf16 MMA.
//!
//! Tolerance is `5e-2`: bf16 has a 7-bit mantissa (vs f16's 10-bit, f32's
//! 23-bit). At T=512 each Q row accumulates up to 512 fp products + an
//! online softmax pass, so ULP drift compounds beyond the `2e-2` envelope
//! that the f32-only sibling tests use. The CPU oracle additionally
//! rounds Q/K/V through bf16 (top-16 bits of fp32) so the reference
//! "sees" the same load-cast quantisation the kernel does.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::attn::steel_attention_mma_bf16::mt_sdpa_prefill_mma_bf16;

/// Causal-prefill SDPA reference for the (B, n_q_heads, q_len, head_dim)
/// + (B, n_kv_heads, k_len, head_dim) layout the kernel reads.
///
/// Mirrors the kernel's `q_len_off = k_len - q_len` shift so causal
/// masking lines up: q row `qi` attends to k positions `0..=q_len_off+qi`.
/// All accumulation is fp32 — the dtype-specific rounding happens at
/// the caller's pack step (Q/K/V) so this oracle sees the post-quant
/// values directly.
#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_prefill_reference_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert!(n_q_heads.is_multiple_of(n_kv_heads));
    let gqa = n_q_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Causal: row qi can attend to keys [0, q_len_off + qi]. With q_len ==
    // k_len this reduces to the usual i-th-row-sees-[0..=i] form.
    let q_len_off = k_len - q_len;
    let mut out = vec![0.0f32; batch * n_q_heads * q_len * head_dim];
    for b in 0..batch {
        for h in 0..n_q_heads {
            let kv_h = h / gqa;
            for qi in 0..q_len {
                let causal_lim = q_len_off + qi + 1;
                let mut scores = vec![f32::NEG_INFINITY; k_len];
                for (j, score) in scores.iter_mut().enumerate().take(causal_lim) {
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = b * n_q_heads * q_len * head_dim
                            + h * q_len * head_dim
                            + qi * head_dim
                            + d;
                        let k_idx = b * n_kv_heads * k_len * head_dim
                            + kv_h * k_len * head_dim
                            + j * head_dim
                            + d;
                        s += q[q_idx] * k[k_idx];
                    }
                    *score = s * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores
                    .iter()
                    .map(|&s| if s.is_finite() { (s - m).exp() } else { 0.0 })
                    .collect();
                let total: f32 = e.iter().sum();
                let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, ej) in e.iter().enumerate() {
                        let v_idx = b * n_kv_heads * k_len * head_dim
                            + kv_h * k_len * head_dim
                            + j * head_dim
                            + d;
                        acc += ej * inv * v[v_idx];
                    }
                    let o_idx =
                        b * n_q_heads * q_len * head_dim + h * q_len * head_dim + qi * head_dim + d;
                    out[o_idx] = acc;
                }
            }
        }
    }
    out
}

/// Round every element through bf16 (top 16 bits of fp32, round-to-zero).
/// Matches the round-trip the kernel sees on every load: device bf16 →
/// f32 register → bf16 MMA tile. Without this, the CPU reference would
/// be ~1 ULP fp32-accurate but the kernel ~1 ULP bf16-accurate, and the
/// diffs would conflate two different precisions.
fn bf16_round(vals: &[f32]) -> Vec<f32> {
    vals.iter()
        .map(|v| {
            // Bench-grade bf16 round (no rounding bias, RNE-equivalent
            // for our smooth ramp inputs — see common::Dt::Bf16 which
            // uses half::bf16::from_f32 internally).
            let bits = v.to_bits();
            let trunc = (bits >> 16) as u16;
            f32::from_bits((trunc as u32) << 16)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_prefill_bf16(
    ctx: &Context,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    batch: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let dt = Dt::Bf16;
    let dt_bytes = dt.bytes();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), q_bytes.to_vec());
    buffers.insert("k".into(), k_bytes.to_vec());
    buffers.insert("v".into(), v_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; batch * n_q_heads * q_len * head_dim * dt_bytes]);
    buffers.insert("q_len".into(), (q_len as u32).to_le_bytes().to_vec());
    buffers.insert("k_len".into(), (k_len as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), ((n_q_heads / n_kv_heads) as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv_heads".into(), (n_kv_heads as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let mut kernel = mt_sdpa_prefill_mma_bf16::kernel_ir_for(dt.to_dtype());
    // Same SimdGroup2D dispatch as the sibling: the kernel body reads
    // tgid_x/tgid_y/tgid_z directly, so only the 3D-axis mode resolves
    // all three grid coords.
    kernel.mode = metaltile::core::ir::KernelMode::SimdGroup2D;
    // Grid: (q_len / BQ=32, n_q_heads, batch). 128 threads = 4 SGs.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [q_len / 32, n_q_heads, batch], [
            128, 1, 1,
        ])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    unpack_bytes(out_bytes, dt)
}

/// Build a smooth, deterministic ramp tensor. Same modulus-based shape
/// as the sibling's `ramp(...)` helper, but factored per-axis so each
/// batch / head / position / dim gets a distinct phase. Distinct
/// per-batch phases catch missed batch-slab offsets (would otherwise
/// show up as cross-batch contamination, not a magnitude blow-up).
fn ramp_4d(
    batch: usize,
    n_heads: usize,
    t: usize,
    head_dim: usize,
    modulus: usize,
    offset: f32,
    phase: f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch * n_heads * t * head_dim);
    for b in 0..batch {
        for h in 0..n_heads {
            for ti in 0..t {
                for d in 0..head_dim {
                    let i = ((b * n_heads + h) * t + ti) * head_dim + d;
                    let v = ((i % modulus) as f32 - offset) * 0.05 + (b as f32) * phase;
                    out.push(v);
                }
            }
        }
    }
    out
}

/// bf16 prefill drift envelope. The bf16 frag dtype is 7-bit mantissa,
/// the f32 softmax + accumulators add tail-ULP drift, and at T=512 the
/// per-row weighted-V sum touches up to 512 lanes. `5e-2` envelopes the
/// total — measured peak on M5 was well inside this (≈2.5e-2 on the
/// T=512 single-batch case), so this tolerance fails loud on real
/// regressions (e.g., a missed q_len_off shift would show as ≥ 1e-1).
const BF16_PREFILL_TOL: f32 = 5e-2;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
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
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_sdpa_prefill_mma_bf16_matches_cpu_reference_bf16_t512() {
    let _g = gpu_lock();

    // B=1, gqa=1, T=512. Smallest cell that exercises 16 q tiles per
    // head (T / BQ = 16) — covers the kb-loop's full causal sweep plus
    // the softmax merge across multiple q tiles. n_q_heads=4 keeps
    // CPU-ref cost ≈ 4 × 512² × 128 ≈ 130M fp ops (≈70ms on M5).
    let batch = 1usize;
    let n_q_heads = 4usize;
    let n_kv_heads = 4usize;
    let q_len = 512usize;
    let k_len = 512usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp_4d(batch, n_q_heads, q_len, head_dim, 17, 8.0, 0.0);
    let k = ramp_4d(batch, n_kv_heads, k_len, head_dim, 13, 6.0, 0.0);
    let v = ramp_4d(batch, n_kv_heads, k_len, head_dim, 11, 5.0, 0.0);

    // Round Q/K/V through bf16 *before* feeding the reference. This is
    // the same quantisation the kernel sees on load: device bf16 → f32
    // register. Without it, the oracle would compare bf16 outputs against
    // an fp32 reference and conflate input-cast drift with kernel drift.
    let q_q = bf16_round(&q);
    let k_q = bf16_round(&k);
    let v_q = bf16_round(&v);
    let expected = cpu_sdpa_prefill_reference_causal(
        &q_q, &k_q, &v_q, batch, n_q_heads, n_kv_heads, q_len, k_len, head_dim,
    );

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::Bf16);
    let k_b = pack_bytes(&k, Dt::Bf16);
    let v_b = pack_bytes(&v, Dt::Bf16);
    let actual = run_sdpa_prefill_bf16(
        &ctx, &q_b, &k_b, &v_b, batch, n_q_heads, n_kv_heads, q_len, k_len, head_dim, scale,
    );

    assert_close(&actual, &expected, BF16_PREFILL_TOL, "T=512 B=1 gqa=1 bf16");
}

#[test]
fn mt_sdpa_prefill_mma_bf16_gqa_factor_4_bf16_t512() {
    let _g = gpu_lock();

    // GQA fan-out: n_q_heads=8, n_kv_heads=2 → gqa=4. Verifies the
    // `kv_head = q_head / gqa_factor` mapping at constexpr=4. Catches
    // any off-by-one in the gqa div (e.g., gqa=1 fallthrough would
    // produce stale kv loads for q_head=4..7).
    let batch = 1usize;
    let n_q_heads = 8usize;
    let n_kv_heads = 2usize;
    let q_len = 512usize;
    let k_len = 512usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp_4d(batch, n_q_heads, q_len, head_dim, 19, 7.0, 0.0);
    let k = ramp_4d(batch, n_kv_heads, k_len, head_dim, 13, 6.0, 0.0);
    let v = ramp_4d(batch, n_kv_heads, k_len, head_dim, 11, 5.0, 0.0);

    let q_q = bf16_round(&q);
    let k_q = bf16_round(&k);
    let v_q = bf16_round(&v);
    let expected = cpu_sdpa_prefill_reference_causal(
        &q_q, &k_q, &v_q, batch, n_q_heads, n_kv_heads, q_len, k_len, head_dim,
    );

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::Bf16);
    let k_b = pack_bytes(&k, Dt::Bf16);
    let v_b = pack_bytes(&v, Dt::Bf16);
    let actual = run_sdpa_prefill_bf16(
        &ctx, &q_b, &k_b, &v_b, batch, n_q_heads, n_kv_heads, q_len, k_len, head_dim, scale,
    );

    assert_close(&actual, &expected, BF16_PREFILL_TOL, "T=512 B=1 gqa=4 bf16");
}

#[test]
fn mt_sdpa_prefill_mma_bf16_kernel_side_b2_t512_bf16() {
    let _g = gpu_lock();

    // True B=2 dispatch: kernel reads `batch = tgid_z` and folds it
    // into Q/K/V/O slab offsets via `n_q_heads` + `n_kv_heads`. Q/K/V
    // laid out as `[B, n_*_heads, T, D]`. Distinct per-batch phases
    // (`+ b * 0.13` for Q, etc.) mean a missed batch-slab offset would
    // produce a clear magnitude mismatch rather than silently matching
    // a same-data adjacent batch.
    let batch = 2usize;
    let n_q_heads = 4usize;
    let n_kv_heads = 1usize;
    let q_len = 512usize;
    let k_len = 512usize;
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp_4d(batch, n_q_heads, q_len, head_dim, 17, 8.0, 0.13);
    let k = ramp_4d(batch, n_kv_heads, k_len, head_dim, 13, 6.0, 0.11);
    let v = ramp_4d(batch, n_kv_heads, k_len, head_dim, 11, 5.0, 0.17);

    let q_q = bf16_round(&q);
    let k_q = bf16_round(&k);
    let v_q = bf16_round(&v);
    let expected = cpu_sdpa_prefill_reference_causal(
        &q_q, &k_q, &v_q, batch, n_q_heads, n_kv_heads, q_len, k_len, head_dim,
    );

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let q_b = pack_bytes(&q, Dt::Bf16);
    let k_b = pack_bytes(&k, Dt::Bf16);
    let v_b = pack_bytes(&v, Dt::Bf16);
    let actual = run_sdpa_prefill_bf16(
        &ctx, &q_b, &k_b, &v_b, batch, n_q_heads, n_kv_heads, q_len, k_len, head_dim, scale,
    );

    assert_close(&actual, &expected, BF16_PREFILL_TOL, "T=512 B=2 gqa=4 bf16");
}

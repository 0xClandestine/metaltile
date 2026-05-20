//! End-to-end GPU correctness for M7's K=8/16 batched-Q SDPA decode
//! via the `mt_sdpa_prefill_mma` prefill-tile reuse path.
//!
//! Strategy: build padded Q (BQ=32 rows per head with the first K real)
//! and padded K/V (n_kv + BQ slots, real prefix in [0, n_kv)); dispatch
//! `mt_sdpa_prefill_mma` directly with `q_len=32, k_len=n_kv+32`; then
//! assert the first K rows of each head's output match the CPU
//! `naive_sdpa_causal_prefix_f32` reference at the same shape.
//!
//! Semantics: the prefill kernel's hardcoded causal mask gives
//! `attended_kv(i) = [0, n_kv + i + 1)` for the real Q rows i in 0..K
//! (q_len_off = k_len - q_len = n_kv). That's the standard speculative-
//! decode-verify pattern — different from the K=2/4 decode-form
//! kernels' flat `attended_kv = [0, n_kv)`.
//!
//! macOS-gated: needs an actual Metal device. Mirrors the layout of
//! `sdpa_decode_batched_gpu_correctness.rs`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, naive_sdpa_causal_prefix_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::attn::steel_attention_mma::mt_sdpa_prefill_mma;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

const BQ: usize = 32;
const TPG: usize = 128;

/// Pad `q_real` (shape `[n_q_heads, batch_q, head_dim]`) to the
/// prefill-tile-shaped `[n_q_heads, BQ, head_dim]`. First `batch_q`
/// rows per head are real, the rest are zeros.
fn pad_q_for_prefill(
    q_real: &[f32],
    n_q_heads: usize,
    batch_q: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q_real.len(), n_q_heads * batch_q * head_dim);
    let mut padded = vec![0.0f32; n_q_heads * BQ * head_dim];
    for h in 0..n_q_heads {
        let src = &q_real[h * batch_q * head_dim..(h + 1) * batch_q * head_dim];
        let dst_base = h * BQ * head_dim;
        padded[dst_base..dst_base + batch_q * head_dim].copy_from_slice(src);
    }
    padded
}

/// Pad K (or V) cache `[n_kv_heads, n_kv, head_dim]` to
/// `[n_kv_heads, n_kv + BQ, head_dim]`. First `n_kv` slots per head are
/// real, the trailing `BQ` slots are zeros (masked out by causal for
/// real Q rows but kept inside the kernel's KV walk range).
fn pad_kv_for_prefill(
    kv_real: &[f32],
    n_kv_heads: usize,
    n_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(kv_real.len(), n_kv_heads * n_kv * head_dim);
    let padded_n = n_kv + BQ;
    let mut padded = vec![0.0f32; n_kv_heads * padded_n * head_dim];
    for h in 0..n_kv_heads {
        let src = &kv_real[h * n_kv * head_dim..(h + 1) * n_kv * head_dim];
        let dst_base = h * padded_n * head_dim;
        padded[dst_base..dst_base + n_kv * head_dim].copy_from_slice(src);
    }
    padded
}

/// Take the BQ-padded output `[n_q_heads, BQ, head_dim]` and return
/// the first `batch_q` rows per head as `[n_q_heads, batch_q, head_dim]`.
fn unpad_out(padded: &[f32], n_q_heads: usize, batch_q: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(padded.len(), n_q_heads * BQ * head_dim);
    let mut out = Vec::with_capacity(n_q_heads * batch_q * head_dim);
    for h in 0..n_q_heads {
        let src_base = h * BQ * head_dim;
        out.extend_from_slice(&padded[src_base..src_base + batch_q * head_dim]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_tile_f32(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    q_padded: &[f32],
    k_padded: &[f32],
    v_padded: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k_len_padded: usize,
    scale: f32,
) -> Vec<f32> {
    let gqa_factor = n_q_heads / n_kv_heads;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(q_padded));
    buffers.insert("k".into(), f32_slice_to_bytes(k_padded));
    buffers.insert("v".into(), f32_slice_to_bytes(v_padded));
    buffers.insert("out".into(), vec![0u8; n_q_heads * BQ * head_dim * 4]);
    buffers.insert("q_len".into(), (BQ as u32).to_le_bytes().to_vec());
    buffers.insert("k_len".into(), (k_len_padded as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), (gqa_factor as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv_heads".into(), (n_kv_heads as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let result = ctx
        .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [1, n_q_heads, 1], [TPG, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
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

fn run_correctness_at(batch_q: usize, n_kv: usize, n_q_heads: usize, n_kv_heads: usize, tol: f32) {
    let _g = gpu_lock();
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let k_len_padded = n_kv + BQ;

    // Real candidate Q vectors, real prefix K and V — same deterministic
    // ramp pattern the existing K=2/4 tests use.
    let q_real = ramp(n_q_heads * batch_q * head_dim, 17, 8.0);
    let k_real = ramp(n_kv_heads * n_kv * head_dim, 13, 6.0);
    let v_real = ramp(n_kv_heads * n_kv * head_dim, 19, 9.0);

    let q_padded = pad_q_for_prefill(&q_real, n_q_heads, batch_q, head_dim);
    let k_padded = pad_kv_for_prefill(&k_real, n_kv_heads, n_kv, head_dim);
    let v_padded = pad_kv_for_prefill(&v_real, n_kv_heads, n_kv, head_dim);

    // CPU reference: causal-prefix SDPA at the padded shape, then take
    // first `batch_q` rows per head. q_len_off = k_len_padded - q_len =
    // n_kv. For Q row qi in 0..batch_q, attended = [0, n_kv + qi + 1).
    let expected_padded = naive_sdpa_causal_prefix_f32(
        &q_padded,
        &k_padded,
        &v_padded,
        n_q_heads,
        n_kv_heads,
        BQ,
        k_len_padded,
        head_dim,
        n_kv, // q_len_off
        scale,
    );
    let expected = unpad_out(&expected_padded, n_q_heads, batch_q, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_sdpa_prefill_mma::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::SimdGroup2D;
    kernel.bfloat_reinterpret_cast = true;

    let actual_padded = run_prefill_tile_f32(
        &ctx,
        &kernel,
        &q_padded,
        &k_padded,
        &v_padded,
        n_q_heads,
        n_kv_heads,
        head_dim,
        k_len_padded,
        scale,
    );
    let actual = unpad_out(&actual_padded, n_q_heads, batch_q, head_dim);

    assert_close(
        &actual,
        &expected,
        tol,
        &format!("K={batch_q}, n_kv={n_kv}: prefill-tile vs naive_sdpa_causal_prefix"),
    );
}

// ── Tolerance derivation ─────────────────────────────────────────────────
//
// The prefill-tile path runs `mt_sdpa_prefill_mma`, which accumulates
// in fp32 throughout but emits one narrowing cast per output store
// (controlled by `bfloat_reinterpret_cast = true`, mirroring how
// `run_sdpa_prefill` dispatches the kernel for inference). The tile
// uses simdgroup-matrix MMA with `simd_shuffle_xor`-based row
// reductions, which has different rounding behavior than the
// lane-quartile decode kernels — closer to the steel-attention
// reference path. The chosen tolerances mirror the prefill bench's
// existing `tol = 2e-2` at long context (matches
// `mt_sdpa_prefill_mma`'s own bench spec). At n_kv=64 we tighten to
// 5e-3 since the per-row visibility windows are short and the
// accumulator depth bounds the simd_shuffle_xor reorder noise more
// tightly.

// ── K=8 ──────────────────────────────────────────────────────────────────

#[test]
fn batched_q8_matches_causal_prefix_reference_small_f32() {
    // Small shape — n_kv=64 lets the causal mask exercise the first
    // few KV-tile boundaries (bk=16 → 4 tiles), and the padding rows
    // are still > the real K rows so any misindexing leaks across.
    // Args: batch_q, n_kv, n_q_heads, n_kv_heads, tol.
    run_correctness_at(8, 64, 2, 1, 5e-3);
}

#[test]
fn batched_q8_matches_causal_prefix_reference_large_n_kv_f32() {
    // Long-context shape — n_kv=1024, gqa_factor=4 to exercise both
    // the deeper softmax reduction and a non-trivial GQA mapping.
    // Args: batch_q, n_kv, n_q_heads, n_kv_heads, tol.
    run_correctness_at(8, 1024, 8, 2, 2e-2);
}

// ── K=16 ─────────────────────────────────────────────────────────────────

#[test]
fn batched_q16_matches_causal_prefix_reference_small_f32() {
    // Args: batch_q, n_kv, n_q_heads, n_kv_heads, tol.
    run_correctness_at(16, 64, 2, 1, 5e-3);
}

#[test]
fn batched_q16_matches_causal_prefix_reference_large_n_kv_f32() {
    // Args: batch_q, n_kv, n_q_heads, n_kv_heads, tol.
    run_correctness_at(16, 1024, 8, 2, 2e-2);
}

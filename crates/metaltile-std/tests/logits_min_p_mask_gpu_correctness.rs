//! GPU correctness coverage for `logits_min_p_mask`.
//!
//! Min-p sampling keeps every token whose probability is at least
//! `min_p` times the probability of the most-likely token and masks
//! the rest. Working in logit space, the keep test is
//! `exp(logit_i − logit_max) ≥ min_p` — no softmax needed. The kernel
//! is reduction-mode: one threadgroup per row, a `reduce_max` finds
//! the row max, a second pass masks every logit below the cutoff to
//! `-INFINITY` so the downstream categorical sampler sees zero
//! probability for it.
//!
//! This file pins that contract against a CPU oracle across:
//!
//! - **min_p → 0** — nothing masked (every token survives).
//! - **min_p → 1** — only the argmax survives.
//! - **mid-range min_p** — a genuine mix of kept and masked tokens.
//! - **production vocab** — Qwen3's 152 064-token row, looped at
//!   tpg=256 so the strided reduction is exercised at scale.
//! - **f32 / f16 / bf16** — the row max and the keep ratio are
//!   computed in f32 regardless of `T`; kept logits are stored
//!   bit-identical to the input, so the comparison is exact.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_runtime::Context;
use metaltile_std::ffai::logits_min_p::logits_min_p_mask;

/// Threadgroup size for every dispatch here. 256 is a multiple of 32
/// (single-simdgroup-or-more, no sub-simdgroup `reduce_max` hazard)
/// and the kernel's strided `range(rs + tid, re, lsize)` loops cover
/// any vocab length at this tpg.
const TPG: usize = 256;

/// CPU oracle: keep `v` iff `exp(v − row_max) ≥ min_p`, else `-inf`.
/// `logits` is already rounded through the dtype, so a kept logit is
/// returned verbatim — exactly what the kernel stores.
fn cpu_min_p_mask(logits: &[f32], n: usize, rows: usize, min_p: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let base = r * n;
        let row = &logits[base..base + n];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for (i, &v) in row.iter().enumerate() {
            out[base + i] = if (v - m).exp() >= min_p { v } else { f32::NEG_INFINITY };
        }
    }
    out
}

/// Dispatch `logits_min_p_mask` over `rows × n` logits at `dtype`.
/// `n` and `min_p` are `#[constexpr]` kernel args; like every other
/// constexpr in the suite they're handed in through the buffer map.
fn run_min_p_mask(logits: &[f32], n: usize, rows: usize, dtype: Dt, min_p: f32) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(logits, dtype));
    buffers.insert("out".into(), vec![0u8; rows * n * dtype.bytes()]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("min_p".into(), min_p.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = logits_min_p_mask::kernel_ir_for(dtype.to_dtype());
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [TPG, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    unpack_bytes(out_bytes, dtype)
}

/// Run the kernel and the oracle on the same logits and assert they
/// agree element-for-element. Masked positions must be `-inf` on both
/// sides; kept positions must match bit-exactly (a kept logit is the
/// input value, round-tripped through `T` identically by both paths).
fn check(logits: &[f32], n: usize, rows: usize, dtype: Dt, min_p: f32) {
    let _g = gpu_lock();

    let rounded: Vec<f32> = logits.iter().map(|&v| dtype.round(v)).collect();
    let expected = cpu_min_p_mask(&rounded, n, rows, min_p);
    let actual = run_min_p_mask(&rounded, n, rows, dtype, min_p);

    assert_eq!(actual.len(), expected.len(), "output element count");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if *e == f32::NEG_INFINITY {
            assert_eq!(
                *a,
                f32::NEG_INFINITY,
                "idx {i}: oracle masked this token but kernel kept {a} \
                 (n={n} rows={rows} dtype={:?} min_p={min_p})",
                dtype.to_dtype(),
            );
        } else {
            assert_eq!(
                a,
                e,
                "idx {i}: kept-token mismatch (n={n} rows={rows} \
                 dtype={:?} min_p={min_p})",
                dtype.to_dtype(),
            );
        }
    }
}

/// Wide-spread logit ramp: `(i % 53)` stepped by 0.2 and centred near
/// zero. Per-row spread ≈ 10.4, so `exp(v − max)` ranges across ~4.5
/// orders of magnitude — wide enough that a mid-range `min_p` keeps
/// some tokens and masks others rather than degenerating to all/none.
/// The 0.2 step keeps every token's probability clear of the `min_p`
/// values the tests use, so no token sits on the keep/mask boundary
/// where a GPU-vs-CPU `exp` ULP could flip the result.
fn ramp(n: usize, rows: usize) -> Vec<f32> {
    (0..n * rows).map(|i| (i % 53) as f32 * 0.2 - 5.0).collect()
}

// ── min_p → 0: nothing is masked ─────────────────────────────────────

#[test]
fn min_p_near_zero_keeps_every_token() {
    // min_p = 1e-6 ⇒ cutoff at v − max ≥ ln(1e-6) ≈ −13.8. The ramp's
    // per-row spread is ≈ 10.4 < 13.8, so every token survives. Assert
    // that directly (no `-inf` in the output) on top of the oracle
    // compare, so a kernel that masks too aggressively is caught.
    let (n, rows) = (320, 3);
    let logits = ramp(n, rows);
    check(&logits, n, rows, Dt::F32, 1e-6);

    let out = run_min_p_mask(&logits, n, rows, Dt::F32, 1e-6);
    assert!(out.iter().all(|v| v.is_finite()), "min_p→0 must keep every token");
}

// ── min_p → 1: only the argmax survives ──────────────────────────────

#[test]
fn min_p_near_one_keeps_only_argmax() {
    // Strictly increasing logits ⇒ a unique row max at the last index.
    // min_p = 0.999: the argmax has exp(0) = 1.0 ≥ 0.999 and every
    // other token (next-highest is exp(−0.1) ≈ 0.905) falls below the
    // cutoff. Exactly one finite value, at the final index.
    let n = 64;
    let logits: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    check(&logits, n, 1, Dt::F32, 0.999);

    let out = run_min_p_mask(&logits, n, 1, Dt::F32, 0.999);
    let kept: Vec<usize> =
        out.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
    assert_eq!(kept, vec![n - 1], "min_p→1 must keep only the argmax");
}

// ── mid-range min_p: a genuine mix of kept and masked tokens ──────────

#[test]
fn min_p_mid_range_matches_oracle_f32() {
    // min_p = 0.1 ⇒ cutoff v − max ≥ ln(0.1) ≈ −2.30. With the ramp's
    // 0.2 step that keeps the ~12 highest distinct values per row and
    // masks the rest — a real partition the oracle compare verifies.
    let (n, rows) = (320, 4);
    let logits = ramp(n, rows);
    check(&logits, n, rows, Dt::F32, 0.1);

    // Sanity-check the partition is non-trivial: both kept and masked
    // tokens must be present, else this test proves nothing.
    let out = run_min_p_mask(&logits, n, rows, Dt::F32, 0.1);
    assert!(out.iter().any(|v| v.is_finite()), "expected some kept tokens");
    assert!(out.iter().any(|v| !v.is_finite()), "expected some masked tokens");
}

// ── production vocab: Qwen3's 152 064-token row ──────────────────────

#[test]
fn min_p_qwen3_vocab_stress_f32() {
    // Qwen3 vocab = 152 064. One threadgroup per row at tpg=256 means
    // each lane strides ~594 logits — exercises the looped reduction
    // and the looped mask pass at production scale.
    let (n, rows) = (152_064, 2);
    let logits = ramp(n, rows);
    check(&logits, n, rows, Dt::F32, 0.1);
}

// ── dtype coverage: f16 / bf16 ───────────────────────────────────────

#[test]
fn min_p_mid_range_matches_oracle_f16() {
    let (n, rows) = (320, 4);
    let logits = ramp(n, rows);
    check(&logits, n, rows, Dt::F16, 0.1);
}

#[test]
fn min_p_mid_range_matches_oracle_bf16() {
    // bf16's 7-bit mantissa quantises the ramp coarsely, but the keep
    // test runs in f32 and a kept logit is stored verbatim, so the
    // comparison is still exact once the input is pre-rounded.
    let (n, rows) = (320, 4);
    let logits = ramp(n, rows);
    check(&logits, n, rows, Dt::Bf16, 0.1);
}

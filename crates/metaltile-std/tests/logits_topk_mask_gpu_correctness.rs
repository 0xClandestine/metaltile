//! GPU correctness for `ffai::logits_topk::logits_topk_mask`.
//!
//! The mask kernel takes a pre-computed threshold (typically the K-th
//! largest logit value, found via host-side argpartition) and replaces
//! every logit below it with `-inf`. Downstream softmax sees
//! `exp(-inf) = 0` so the filtered tokens contribute zero probability.
//!
//! Pinned invariants:
//!   - `threshold = -inf` keeps every value (no-op)
//!   - `threshold = +inf` masks every value to `-inf` (degenerate)
//!   - K=50, K=200, K=1: top-K count after the mask matches expected K
//!   - vocab=152K stress test (Qwen tokenizer scale)
//!   - dtype matrix (f32 / f16 / bf16)
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::logits_topk::logits_topk_mask;

fn run_topk_mask(logits: &[f32], dt: Dt, threshold: f32) -> Vec<f32> {
    let n = logits.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(logits, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; n], dt));
    buffers.insert("threshold".into(), threshold.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = logits_topk_mask::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("logits_topk_mask dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

/// Compute the K-th largest value (descending), matching how callers
/// pre-compute the threshold via argpartition.
fn kth_largest(logits: &[f32], k: usize) -> f32 {
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[k - 1]
}

#[test]
fn topk_mask_threshold_neg_inf_keeps_all_f32() {
    let _g = gpu_lock();
    // threshold = -inf → every value satisfies v >= threshold → no-op.
    let logits: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let actual = run_topk_mask(&logits, Dt::F32, f32::NEG_INFINITY);
    for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "idx={i}: expected {e}, got {a}");
    }
}

#[test]
fn topk_mask_threshold_pos_inf_masks_all_f32() {
    let _g = gpu_lock();
    // threshold = +inf → nothing is >= threshold → all become -inf.
    let logits: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let actual = run_topk_mask(&logits, Dt::F32, f32::INFINITY);
    for (i, a) in actual.iter().enumerate() {
        assert!(a.is_infinite() && a.is_sign_negative(), "idx={i}: expected -inf, got {a}",);
    }
}

#[test]
fn topk_mask_k_50_random_logits_f32() {
    let _g = gpu_lock();
    let n = 1024usize;
    let k = 50usize;
    let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin() * 5.0).collect();
    let threshold = kth_largest(&logits, k);

    let actual = run_topk_mask(&logits, Dt::F32, threshold);

    // Exactly K entries should be != -inf (no ties at threshold by
    // construction since the sin pattern produces distinct floats).
    let kept: Vec<usize> =
        actual.iter().enumerate().filter(|&(_, &v)| !v.is_infinite()).map(|(i, _)| i).collect();
    assert_eq!(kept.len(), k, "K=50 mask kept {} entries, expected {k}", kept.len());

    // Every kept entry must equal its source value.
    for &i in &kept {
        assert!((actual[i] - logits[i]).abs() < 1e-6, "idx={i} kept but value changed");
    }
}

#[test]
fn topk_mask_k_1_picks_only_argmax_f32() {
    let _g = gpu_lock();
    // K=1 + a clear unique max → only the argmax survives.
    let n = 1024usize;
    let mut logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
    logits[731] = 100.0;
    let threshold = kth_largest(&logits, 1);

    let actual = run_topk_mask(&logits, Dt::F32, threshold);

    for (i, a) in actual.iter().enumerate() {
        if i == 731 {
            assert!((a - 100.0).abs() < 1e-6, "argmax at 731 changed: {a}");
        } else {
            assert!(a.is_infinite() && a.is_sign_negative(), "idx={i} should be -inf, got {a}");
        }
    }
}

#[test]
fn topk_mask_vocab_152k_k_200_f32() {
    let _g = gpu_lock();
    // Qwen-scale vocab + realistic serving K. Verify the keep-count
    // matches expected (some ties possible at this scale; allow ±2).
    let n = 152_064usize;
    let k = 200usize;
    let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0091).sin() * 3.0).collect();
    let threshold = kth_largest(&logits, k);

    let actual = run_topk_mask(&logits, Dt::F32, threshold);

    let kept = actual.iter().filter(|&&v| !v.is_infinite()).count();
    assert!(kept.abs_diff(k) <= 2, "vocab=152K K=200: kept = {kept}, expected {k} (±2 for ties)",);

    // Every kept value must equal its source.
    for (i, (&a, &e)) in actual.iter().zip(logits.iter()).enumerate() {
        if !a.is_infinite() {
            assert!((a - e).abs() < 1e-6, "idx={i} kept but value changed");
        }
    }
}

#[test]
fn topk_mask_f16() {
    let _g = gpu_lock();
    let n = 1024usize;
    let k = 50usize;
    let logits_f32: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin() * 5.0).collect();
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let threshold = kth_largest(&logits, k);

    let actual = run_topk_mask(&logits, Dt::F16, threshold);

    // f16 reduces precision so ties may appear at the threshold;
    // allow ±5 entries of slack.
    let kept = actual.iter().filter(|&&v| !v.is_infinite()).count();
    assert!(kept.abs_diff(k) <= 5, "f16 K=50 kept {kept}, expected {k} (±5)");
}

#[test]
fn topk_mask_bf16() {
    let _g = gpu_lock();
    let n = 1024usize;
    let k = 50usize;
    let logits_f32: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin() * 5.0).collect();
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let threshold = kth_largest(&logits, k);

    let actual = run_topk_mask(&logits, Dt::Bf16, threshold);

    // The exact kept-count varies with bf16 tie density at the threshold.
    // Tolerance derived from first principles below; the underlying
    // *correctness* invariant — the dominance property — is checked
    // unconditionally below the count check.
    let kept = actual.iter().filter(|&&v| !v.is_infinite()).count();

    // bf16 step at magnitude v ≈ v / 128 (7-bit mantissa). For sin
    // scaled to ±5, the K=50 threshold lands near v ≈ 4.7 (top 5%
    // of a sin distribution), so step ≈ 0.037.
    // The sin distribution is non-uniform — density near extrema
    // scales as 1/sqrt(1 - (v/5)^2). At v=4.7 the density factor is
    // ~2.9× uniform. Combining: in a ±step window around threshold,
    // expected tied-value count =
    //   2 * step * (n / range) * density_factor
    //   = 2 * 0.037 * (1024 / 10) * 2.9 ≈ 22
    // Tolerance of ±30 leaves safety margin for the discrete-sampling
    // variance on top of the analytic estimate.
    assert!(
        kept.abs_diff(k) <= 30,
        "bf16 K=50 kept {kept}, expected {k} (±30 from bf16 tie density at threshold)",
    );

    // Stronger correctness invariant: every kept value must be >= every
    // dropped value's pre-mask source value (after bf16 round-trip).
    // This is the actual top-K-set property; it holds regardless of how
    // many values tie at the threshold.
    let mut max_dropped = f32::NEG_INFINITY;
    let mut min_kept = f32::INFINITY;
    for (i, &a) in actual.iter().enumerate() {
        let src = logits[i];
        if a.is_infinite() {
            max_dropped = max_dropped.max(src);
        } else {
            min_kept = min_kept.min(src);
            // Kept value must equal the source (no scaling).
            assert!((a - src).abs() < 1e-3, "kept idx={i}: a={a} src={src}");
        }
    }
    assert!(
        min_kept >= max_dropped,
        "bf16 top-K dominance violated: min_kept = {min_kept}, max_dropped = {max_dropped}",
    );
}

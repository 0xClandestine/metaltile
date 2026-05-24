//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end correctness test for `ffai::softmax_categorical_sample`
//! on real Metal.
//!
//! Pins the kernel's TPG=256 invariant (tg_max + tg_sum both 256-wide;
//! 8-stage halving over [128 → 1]) and the inverse-CDF determinism:
//! same logits + same temperature + same uniform draw → same index.
//!
//! Coverage rationale: `softmax_categorical_sample` had its body
//! silently emptied by PR #19's macro refactor (restored in this PR).
//! It has no `BenchDispatch` variant, so `tile bench` can't exercise
//! it — this is the only end-to-end check that the sampler picks the
//! right token.
//!
//! Three test shapes:
//!   - peaked logits → always picks the peak (uniform doesn't matter)
//!   - uniform logits → uniform CDF walk picks deterministically
//!   - matches CPU reference for a mixed distribution
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sampling::softmax_categorical_sample;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// CPU naive reference matching the kernel's three-pass shape:
/// max-shift → sum-exp → inverse-CDF walk with first-hit tie-break.
fn naive_sample(logits: &[f32], temperature: f32, uniform: f32) -> u32 {
    let inv_t = 1.0 / temperature;
    let max_val = logits.iter().map(|v| v * inv_t).fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().map(|v| (v * inv_t - max_val).exp()).sum();
    let target = uniform * total;
    let mut cum = 0.0_f32;
    for (i, &v) in logits.iter().enumerate() {
        cum += (v * inv_t - max_val).exp();
        if cum >= target {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

fn run_sample(logits: &[f32], temperature: f32, uniform: f32) -> u32 {
    let n = logits.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), f32_slice_to_bytes(logits));
    buffers.insert("out".into(), vec![0u8; 4]);
    buffers.insert("temperature_in".into(), temperature.to_le_bytes().to_vec());
    buffers.insert("uniform_in".into(), uniform.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = softmax_categorical_sample::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // Fixed TPG = 256 (kernel hard-codes this). 1 threadgroup total.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_u32_vec(out_bytes)[0]
}

#[test]
fn sample_peaked_distribution_always_picks_peak_f32() {
    let _g = gpu_lock();
    // One logit dominates; softmax mass on the peak is ~1, so every
    // uniform draw in [0,1) lands inside its CDF bucket. Pins the
    // kernel's exp/sum correctness — wrong reduction would diffuse
    // mass away from the peak.
    let mut logits = vec![0.0_f32; 512];
    logits[321] = 30.0; // exp(30) >> exp(0); CDF mass is ~all on idx 321
    for u in [0.001_f32, 0.5, 0.999] {
        let actual = run_sample(&logits, 1.0, u);
        assert_eq!(actual, 321, "peaked dist: u={u} → idx 321");
    }
}

#[test]
fn sample_uniform_distribution_tracks_cdf_f32() {
    let _g = gpu_lock();
    // All-zero logits → uniform softmax → CDF[i] = (i+1)/n.
    // u=0.001 → idx 0; u=0.999 → idx n-1; u=0.5 → idx n/2-1 or n/2.
    let n = 256usize;
    let logits = vec![0.0_f32; n];
    assert_eq!(run_sample(&logits, 1.0, 0.001), 0, "u≈0 → idx 0");
    let result_half = run_sample(&logits, 1.0, 0.5);
    assert!(
        result_half == (n as u32 / 2 - 1) || result_half == n as u32 / 2,
        "u=0.5 → idx {} or {} (got {result_half})",
        n / 2 - 1,
        n / 2,
    );
    assert_eq!(run_sample(&logits, 1.0, 0.999), (n - 1) as u32, "u≈1 → idx n-1");
}

#[test]
fn sample_matches_naive_reference_f32() {
    let _g = gpu_lock();
    // Mixed distribution; verify three different uniform draws all
    // match the CPU reference. Floating-point ordering differences
    // between the parallel reduce and the serial CPU walk could in
    // principle nudge a tie across a boundary — but `cum >= target`
    // is a hard inequality and the values are well-separated.
    let logits: Vec<f32> = (0..512)
        .map(|i| {
            let x = (i as f32 - 256.0) / 64.0;
            -0.5 * x * x // Gaussian-like, peaks near i=256
        })
        .collect();
    for u in [0.1_f32, 0.5, 0.9] {
        let expected = naive_sample(&logits, 0.8, u);
        let actual = run_sample(&logits, 0.8, u);
        assert_eq!(actual, expected, "u={u} mismatch — expected {expected}, got {actual}");
    }
}

#[test]
fn sample_qwen_vocab_152k_parallel_prefix_path_f32() {
    let _g = gpu_lock();
    // n=152K (Qwen tokenizer scale) exercises the parallel-prefix CDF
    // walk: chunk = ceil(152064/256) = 594 positions per lane. Every
    // lane runs phase A (chunk sum-exp); a single lane fires phase C
    // (the in-chunk serial walk). Uniform-logits distribution → CDF[i]
    // = (i+1)/n is linear in i, so the GPU vs CPU index drift from
    // FP-ordering ULPs on `total` is bounded by ±~50 tokens (vs
    // pathological ~100s in deep Gaussian tails where density is
    // exponentially small).
    let n = 152_064usize;
    let logits = vec![0.0_f32; n];
    for u in [0.01_f32, 0.2, 0.4, 0.6, 0.8, 0.95] {
        let expected = naive_sample(&logits, 1.0, u);
        let actual = run_sample(&logits, 1.0, u);
        // Uniform CDF: expected ≈ floor(u * n) ± 1. GPU's chunked sum
        // produces a `total` that differs from CPU's serial sum by a
        // few ULPs at n=152K; relative drift ~ 1e-7 × n ≈ 16 tokens
        // in the linear-CDF case. Tolerate ±64 to leave headroom.
        let diff = (expected as i64 - actual as i64).abs();
        assert!(diff <= 64, "u={u} expected {expected}, got {actual} (diff {diff} > 64)",);
    }
}

#[test]
fn sample_qwen_vocab_152k_peaked_picks_peak_f32() {
    let _g = gpu_lock();
    // Peaked distribution at vocab=152K: the dominant token sits at
    // index 99K (mid-vocab, well inside the parallel-prefix scan).
    // Tests that one lane's chunk owns ~all the CDF mass and every
    // uniform draw lands inside that lane's chunk.
    let n = 152_064usize;
    let peak_idx = 99_000usize;
    let mut logits = vec![0.0_f32; n];
    logits[peak_idx] = 30.0; // exp(30) >> exp(0)
    for u in [0.001_f32, 0.5, 0.999] {
        let actual = run_sample(&logits, 1.0, u);
        assert_eq!(actual, peak_idx as u32, "peaked vocab=152K: u={u} → idx {peak_idx}");
    }
}

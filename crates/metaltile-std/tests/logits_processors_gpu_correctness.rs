//! GPU correctness for the logits-processor kernels:
//!
//!   - `logits_temperature` — `logits /= temperature`, elementwise
//!   - `logits_repetition_penalty` — HuggingFace `transformers`
//!     semantics: `v > 0 → v /= p`, `v ≤ 0 → v *= p`
//!
//! These compose with `softmax_categorical_sample` to form the actual
//! decode-time sampling pipeline used in production serving — the
//! existing `softmax_categorical_sample` covers only the bare-softmax
//! draw at `temperature=1, no penalty`.
//!
//! Both kernels are Grid3D, generic-T, internally upcast to f32 for
//! the scale arithmetic. f16/bf16 inputs round-trip through the kernel
//! with the same precision contract as `cast::<T>()` round-trips
//! everywhere else in this repo.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::logits_processors::{logits_repetition_penalty, logits_temperature};

// ────────────────────────────────────────────────────────────────────
//  logits_temperature
// ────────────────────────────────────────────────────────────────────

fn run_temperature(logits: &[f32], dt: Dt, temperature: f32) -> Vec<f32> {
    let n = logits.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(logits, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; n], dt));
    buffers.insert("temperature".into(), temperature.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = logits_temperature::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("logits_temperature dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

#[test]
fn logits_temperature_identity_at_unity_f32() {
    let _g = gpu_lock();
    let logits: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() - 0.5).collect();
    let actual = run_temperature(&logits, Dt::F32, 1.0);
    for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "T=1 identity broke at idx={i}: expected {e}, got {a}",);
    }
}

#[test]
fn logits_temperature_scales_by_inv_t_f32() {
    let _g = gpu_lock();
    // T=2 → each logit halves. T=0.5 → each logit doubles.
    let logits: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
    for &t in &[0.5_f32, 2.0, 4.0, 0.25] {
        let actual = run_temperature(&logits, Dt::F32, t);
        let inv_t = 1.0 / t;
        for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
            let expected = e * inv_t;
            assert!((a - expected).abs() < 1e-4, "T={t} idx={i}: expected {expected}, got {a}",);
        }
    }
}

#[test]
fn logits_temperature_qwen_vocab_152k_f32() {
    let _g = gpu_lock();
    let n = 152_064usize;
    let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0091).sin() * 2.0).collect();
    let actual = run_temperature(&logits, Dt::F32, 0.7);
    let inv_t = 1.0 / 0.7;
    let mut max_diff = 0.0_f32;
    for (a, e) in actual.iter().zip(logits.iter()) {
        max_diff = max_diff.max((a - e * inv_t).abs());
    }
    assert!(max_diff < 1e-4, "vocab=152K T=0.7 max |diff| = {max_diff:.2e}");
}

#[test]
fn logits_temperature_f16() {
    let _g = gpu_lock();
    let logits_f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let actual = run_temperature(&logits, Dt::F16, 0.5);
    for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
        let expected = Dt::F16.round(e * 2.0);
        assert!((a - expected).abs() < 5e-2, "f16 T=0.5 idx={i}: expected {expected}, got {a}",);
    }
}

#[test]
fn logits_temperature_bf16() {
    let _g = gpu_lock();
    let logits_f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let actual = run_temperature(&logits, Dt::Bf16, 0.5);
    for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
        let expected = Dt::Bf16.round(e * 2.0);
        assert!((a - expected).abs() < 5e-1, "bf16 T=0.5 idx={i}: expected {expected}, got {a}",);
    }
}

// ────────────────────────────────────────────────────────────────────
//  logits_repetition_penalty
// ────────────────────────────────────────────────────────────────────

fn run_repetition_penalty(logits: &[f32], token_ids: &[u32], dt: Dt, penalty: f32) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("logits".into(), pack_bytes(logits, dt));
    buffers.insert("token_ids".into(), pack_u32_bytes(token_ids));
    buffers.insert("penalty".into(), penalty.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = logits_repetition_penalty::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let n_tokens = token_ids.len();
    let tpg = if n_tokens >= 256 { 256 } else { n_tokens.max(1) };
    let groups = n_tokens.div_ceil(tpg.max(1));
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("repetition_penalty dispatch");

    let mut out = unpack_bytes(result.outputs.get("logits").expect("logits"), dt);
    out.truncate(logits.len());
    out
}

fn cpu_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    for &tok in token_ids {
        let i = tok as usize;
        let v = logits[i];
        logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
    }
}

#[test]
fn repetition_penalty_no_op_at_unity_f32() {
    let _g = gpu_lock();
    let logits: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    let token_ids: Vec<u32> = vec![3, 7, 11, 137, 200];
    let actual = run_repetition_penalty(&logits, &token_ids, Dt::F32, 1.0);
    for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "penalty=1 should be no-op at idx={i}: expected {e}, got {a}",
        );
    }
}

#[test]
fn repetition_penalty_positive_logits_divide_f32() {
    let _g = gpu_lock();
    // All positive logits with penalty > 1 → each touched logit halves.
    let mut logits: Vec<f32> = (1..=256).map(|i| i as f32).collect();
    let token_ids: Vec<u32> = vec![0, 5, 100, 255];
    let penalty = 2.0_f32;

    let mut expected = logits.clone();
    cpu_repetition_penalty(&mut expected, &token_ids, penalty);

    let actual = run_repetition_penalty(&logits, &token_ids, Dt::F32, penalty);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-4, "+ logit penalty divide idx={i}: expected {e}, got {a}",);
    }
    // Sanity: untouched logit at idx=1 still equals 2.0.
    assert!((actual[1] - 2.0).abs() < 1e-6, "untouched idx=1 changed");
    let _ = &mut logits; // silence "unused mut"
}

#[test]
fn repetition_penalty_negative_logits_multiply_f32() {
    let _g = gpu_lock();
    // All-negative logits with penalty > 1 → each touched logit is more
    // negative (scaled further from zero). Direction matches the
    // HuggingFace convention: penalize toward 0 in magnitude direction,
    // which for negatives means moving farther negative.
    let logits: Vec<f32> = (1..=256).map(|i| -(i as f32)).collect();
    let token_ids: Vec<u32> = vec![0, 5, 100, 255];
    let penalty = 1.5_f32;

    let mut expected = logits.clone();
    cpu_repetition_penalty(&mut expected, &token_ids, penalty);

    let actual = run_repetition_penalty(&logits, &token_ids, Dt::F32, penalty);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-4, "- logit penalty multiply idx={i}: expected {e}, got {a}",);
    }
}

#[test]
fn repetition_penalty_mixed_signs_matches_cpu_f32() {
    let _g = gpu_lock();
    let logits: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.07).sin() * 5.0 - 0.5).collect();
    let token_ids: Vec<u32> = vec![7, 42, 137, 251, 513, 999];
    let penalty = 1.3_f32;

    let mut expected = logits.clone();
    cpu_repetition_penalty(&mut expected, &token_ids, penalty);

    let actual = run_repetition_penalty(&logits, &token_ids, Dt::F32, penalty);
    // Verify every touched index AND every untouched index unchanged.
    for i in 0..logits.len() {
        let touched = token_ids.contains(&(i as u32));
        let e = expected[i];
        let diff = (actual[i] - e).abs();
        let tol = if touched { 1e-4 } else { 1e-6 };
        assert!(
            diff < tol,
            "idx={i} touched={touched}: expected {e}, got {} (diff {diff:.2e})",
            actual[i],
        );
    }
}

#[test]
fn repetition_penalty_duplicate_token_ids_lands_on_one_outcome_f32() {
    let _g = gpu_lock();
    // Kernel doc says: "Callers MUST dedupe token_ids before dispatch."
    // If a caller skips dedup, multiple threads race-write the same
    // vocab slot — the result is one of: single-application (one
    // thread's write survives, others overwrite with the same value
    // since both read the same input) OR double-application (one
    // thread reads after another's write). Pin that the answer is
    // ALWAYS the single-application outcome — this happens to be
    // what Metal's threadgroup memory ordering guarantees for the
    // non-cooperating one-thread-per-token shape: all threads read
    // the original `logits[tok]` before any writes commit, so every
    // racing thread writes the same value. Test pins this so a
    // future change to the kernel shape (e.g. adding a barrier
    // between read and write) would surface the regression.
    let logits: Vec<f32> = (1..=256).map(|i| i as f32).collect();
    let token_ids: Vec<u32> = vec![5, 5, 5, 100, 100]; // dup at 5 ×3, at 100 ×2
    let penalty = 2.0_f32;

    let actual = run_repetition_penalty(&logits, &token_ids, Dt::F32, penalty);

    // Single-application = each duplicated slot halves exactly once.
    let single_applied_5 = 6.0 / penalty;
    let single_applied_100 = 101.0 / penalty;
    // Double-application would halve twice → much smaller. If we see
    // that, the kernel has changed shape; fail loudly with diagnostics.
    let diff_5 = (actual[5] - single_applied_5).abs();
    let diff_100 = (actual[100] - single_applied_100).abs();
    assert!(
        diff_5 < 1e-4 && diff_100 < 1e-4,
        "Duplicate token_ids should land on single-application: \
         actual[5] = {} (expected {}, diff {:.2e}), \
         actual[100] = {} (expected {}, diff {:.2e})",
        actual[5],
        single_applied_5,
        diff_5,
        actual[100],
        single_applied_100,
        diff_100,
    );
    // Untouched: every other slot.
    for (i, (a, e)) in actual.iter().zip(logits.iter()).enumerate() {
        if i == 5 || i == 100 {
            continue;
        }
        assert!((a - e).abs() < 1e-6, "untouched idx={i} changed");
    }
}

#[test]
fn repetition_penalty_f16_qwen_shape() {
    let _g = gpu_lock();
    // Qwen-class vocab penalty pass with a modest context window.
    let n = 32_768usize;
    let logits_f32: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0017).cos() * 4.0 - 0.2).collect();
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let token_ids: Vec<u32> = vec![0, 7, 137, 1024, 9000, 32_000];
    let penalty = 1.5_f32;

    let mut expected_f32 = logits.clone();
    cpu_repetition_penalty(&mut expected_f32, &token_ids, penalty);
    let expected: Vec<f32> = expected_f32.iter().map(|&v| Dt::F16.round(v)).collect();

    let actual = run_repetition_penalty(&logits, &token_ids, Dt::F16, penalty);
    for &tok in &token_ids {
        let i = tok as usize;
        // f16 has 10-bit mantissa; the round-trip + scale may differ
        // from the CPU oracle by a few ULPs of the value magnitude.
        let rel = (actual[i] - expected[i]).abs() / expected[i].abs().max(1e-3);
        assert!(rel < 5e-3, "f16 tok={tok}: rel = {rel:.2e}");
    }
}

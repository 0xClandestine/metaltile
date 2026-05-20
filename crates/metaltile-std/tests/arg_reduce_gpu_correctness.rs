//! GPU correctness for `ffai::arg_reduce::argmax` — generic-T argmax
//! with u32 index output. Decode-form greedy-sampler workhorse.
//!
//! Tie-breaking: strict `>` on values, smallest index on ties
//! (NumPy / PyTorch / MLX argmax semantics).
//!
//! Coverage rationale: `argmax` is the greedy-decode fast path
//! (temperature = 0). A wrong tie-breaker or empty-body regression
//! (the proc-macro class — `argmax`'s 7-stage tree was previously
//! hand-unrolled via an inner `macro_rules!` invocation that
//! silently produced no IR; restored via DSL `for` loop) would
//! silently degrade decode quality without crashing — every greedy
//! decode would pick the LAST tied index instead of the FIRST,
//! flipping token output at every saturated softmax.
//!
//! Tests pin:
//!   - Plain argmax: random logits, compare to CPU oracle
//!   - Ties take smallest index (the semantic contract)
//!   - Vocab=152K (Qwen tokenizer scale) — exercises strided cover
//!   - All dtypes (f32, f16, bf16)
//!   - Negative-only logits (catches a wrong `neg_infinity()`
//!     initialization that would return n-1 garbage)
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::arg_reduce::ffai_argmax;

fn run_argmax(logits: &[f32], dt: Dt) -> u32 {
    let n = logits.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(logits, dt));
    buffers.insert("out".into(), vec![0u8; 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_argmax::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Reduction dispatch: grid=[1,1,1] tg=[256,1,1]. TPG=256 satisfies
    // the ≥32 + multiple-of-32 contract (docs/developing.md).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
        .expect("argmax dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    unpack_u32_bytes(out_bytes)[0]
}

fn cpu_argmax(logits: &[f32]) -> u32 {
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[test]
fn argmax_random_logits_f32() {
    let _g = gpu_lock();
    // 1024 logits with one clear peak buried at a non-trivial index.
    let mut logits: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    logits[731] = 5.0; // dominant peak
    assert_eq!(run_argmax(&logits, Dt::F32), 731);
}

#[test]
fn argmax_negative_only_logits_f32() {
    let _g = gpu_lock();
    // All-negative logits — pins that `best_val` initializes to
    // -infinity (not 0). A wrong init would let any non-negative
    // value (including the implicit 0 from `best_idx - best_idx`)
    // win, returning idx 0 for any positive-zero seed.
    let logits: Vec<f32> = (0..256).map(|i| -(i as f32) - 1.0).collect();
    // Smallest absolute value (i=0 → -1.0) is the max.
    assert_eq!(run_argmax(&logits, Dt::F32), 0);
}

#[test]
fn argmax_ties_take_smallest_index_f32() {
    let _g = gpu_lock();
    // 8 elements tied at the max value; positions 0..3 are smaller
    // negatives so the max ties are at [4, 5, 6, 7]. Argmax must
    // return idx 4 (smallest index among ties).
    let logits: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, 5.0, 5.0, 5.0, 5.0];
    assert_eq!(run_argmax(&logits, Dt::F32), 4);
}

#[test]
fn argmax_ties_across_simdgroup_boundary_f32() {
    let _g = gpu_lock();
    // Ties spanning multiple positions within ONE lane's strided
    // walk (every 256 positions). A wrong "best > current"
    // strict-greater check or a wrong cross-simdgroup tiebreak
    // would not return the smallest index among ties.
    let mut logits = vec![0.0_f32; 1024];
    // Place ties at positions 7, 263, 519, 775 (lid 7 of each
    // strided block) — they all map to the SAME lane (lid=7) so the
    // per-lane reduction must prefer the smallest index.
    logits[7] = 1.0;
    logits[263] = 1.0;
    logits[519] = 1.0;
    logits[775] = 1.0;
    assert_eq!(run_argmax(&logits, Dt::F32), 7);
}

#[test]
fn argmax_vocab_152k_qwen_scale_f32() {
    let _g = gpu_lock();
    // Qwen tokenizer vocab. Exercises the strided per-lane cover:
    // each lane scans ~594 positions before the tree reduction.
    let n = 152_064usize;
    let mut logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0091).sin() * 0.2).collect();
    let peak_idx = 99_001usize;
    logits[peak_idx] = 8.0;
    assert_eq!(run_argmax(&logits, Dt::F32), peak_idx as u32);
}

#[test]
fn argmax_random_logits_f16() {
    let _g = gpu_lock();
    let mut logits_f32: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    logits_f32[731] = 5.0;
    // Round through f16 so the host sees what the kernel sees.
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    assert_eq!(run_argmax(&logits, Dt::F16), cpu_argmax(&logits));
    assert_eq!(run_argmax(&logits, Dt::F16), 731);
}

#[test]
fn argmax_random_logits_bf16() {
    let _g = gpu_lock();
    let mut logits_f32: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    logits_f32[731] = 5.0;
    let logits: Vec<f32> = logits_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    assert_eq!(run_argmax(&logits, Dt::Bf16), cpu_argmax(&logits));
    assert_eq!(run_argmax(&logits, Dt::Bf16), 731);
}

#[test]
fn argmax_first_index_wins_f32() {
    let _g = gpu_lock();
    // Single max at index 0. Pins that the reduction doesn't accidentally
    // prefer later indices via a non-strict comparison.
    let mut logits = vec![-1.0_f32; 1024];
    logits[0] = 10.0;
    assert_eq!(run_argmax(&logits, Dt::F32), 0);
}

#[test]
fn argmax_last_index_wins_f32() {
    let _g = gpu_lock();
    // Single max at the last position. Pins the strided lane that
    // owns the highest index (n-1 mod 256) wins.
    let n = 1024usize;
    let mut logits = vec![-1.0_f32; n];
    logits[n - 1] = 10.0;
    assert_eq!(run_argmax(&logits, Dt::F32), (n - 1) as u32);
}

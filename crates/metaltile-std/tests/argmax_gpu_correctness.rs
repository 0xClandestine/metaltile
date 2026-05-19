//! End-to-end correctness test for `ffai::ffai_argmax` on real Metal.
//!
//! Pins the kernel's TPG=256 invariant: tg_vals/tg_idxs are statically
//! allocated 256-wide and the 7-stage halving reduction assumes
//! TPG=256. Dispatching with any other TPG silently produces wrong
//! output (best case) or pinned GPU (if `simd_*` primitives were used,
//! which they aren't here — argmax uses a manual halving tree).
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, pack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::arg_reduce::ffai_argmax;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Naive argmax with tie-break-on-smallest-index — matches the kernel's
/// `(ov > tv) | ((ov == tv) & (oi < ti))` rule, mirrors NumPy / MLX /
/// PyTorch `argmax` semantics.
fn naive_argmax(values: &[f32]) -> u32 {
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn run_argmax(values: &[f32]) -> u32 {
    let n = values.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), f32_slice_to_bytes(values));
    buffers.insert("out".into(), vec![0u8; 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = ffai_argmax::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // Fixed TPG = 256 (the kernel hard-codes this). 1 threadgroup
    // total — the kernel itself loops to cover all `n` entries.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_u32_vec(out_bytes)[0]
}

#[test]
fn argmax_peak_at_known_index_f32() {
    // 1024 elements; peak deliberately past the first 256-stride to
    // exercise the per-thread `n_iters` loop (n / lsize = 4 iterations).
    let mut values = vec![0.0_f32; 1024];
    values[777] = 100.0;
    let expected = 777u32;
    let actual = run_argmax(&values);
    assert_eq!(actual, expected, "argmax should find peak at index 777");
}

#[test]
fn argmax_ties_break_to_smallest_index_f32() {
    // Three values equal at max; argmax must return the smallest.
    // Pins the `(ov == tv) & (oi < ti)` tie-break clause; without it,
    // a parallel reduction's index choice is non-deterministic.
    let mut values = vec![0.0_f32; 512];
    values[42] = 10.0;
    values[300] = 10.0;
    values[500] = 10.0;
    let actual = run_argmax(&values);
    assert_eq!(actual, 42, "argmax with ties should return smallest index");
}

#[test]
fn argmax_matches_naive_reference_f32_random_ramp() {
    // Pseudo-random ramp; verify against the CPU naive ref. The kernel
    // and the naive ref use the same tie-break rule, so they should
    // agree bit-exactly on any input.
    let n = 768usize;
    let values: Vec<f32> = (0..n)
        .map(|i| {
            // Mixed-sign deterministic pattern.
            ((i * 31 + 17) % 200) as f32 * if i % 3 == 0 { -1.0 } else { 1.0 }
        })
        .collect();
    let expected = naive_argmax(&values);
    let actual = run_argmax(&values);
    assert_eq!(actual, expected, "argmax should match naive reference");
}

#[test]
fn argmax_minimum_n_f32() {
    // n=256 = exactly one full TPG. Smallest legal n. Pins the
    // `if pos < n` guard in the per-thread loop — every thread of
    // the 256-thread group hits the guard exactly once.
    let mut values = vec![0.0_f32; 256];
    values[200] = 1.0;
    let actual = run_argmax(&values);
    assert_eq!(actual, 200u32);
}

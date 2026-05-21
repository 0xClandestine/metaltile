//! GPU correctness for `mlx::fp_quantized` — fp4 quantize-dequantize.
//!
//! `mt_fp4_quant_dequant` computes, per simdgroup of 32 threads:
//!   1. Compute simd-max of |x| over the group → `group_max`.
//!   2. `inv_scale = 6 / group_max` (0 if group_max == 0).
//!   3. For each element: round `|x| * inv_scale` to the nearest fp4 level
//!      (levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6), then restore sign and scale
//!      back: `result = sign * level * (group_max / 6)`.
//!
//! ## DISPATCH INVARIANTS (mt_fp4_quant_dequant)
//! - **Reduction mode** (uses `simd_max`) with **TPG = 32** (one simdgroup).
//! - `n` = total elements = `TPG * n_groups`. Grid = `[n_groups, 1, 1]`.
//! - Each simdgroup handles 32 consecutive elements independently.
//!
//! CPU oracle: replicates the exact 8-level fp4 rounding.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{gpu_lock, max_abs_diff};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::fp_quantized::mt_fp4_quant_dequant;

const FP4_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_BOUNDARIES: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

/// CPU fp4 quant-dequant oracle.
/// Groups of 32 elements share one scale computed from `simd_max(|x|)`.
fn cpu_fp4_quant_dequant(inp: &[f32]) -> Vec<f32> {
    assert_eq!(inp.len() % 32, 0, "input must be multiple of 32 (simdgroup width)");
    let n_groups = inp.len() / 32;
    let mut out = vec![0.0f32; inp.len()];

    for g in 0..n_groups {
        let base = g * 32;
        let group = &inp[base..base + 32];

        let group_max = group.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        let inv_scale = if group_max > 0.0 { 6.0 / group_max } else { 0.0 };

        for (i, &x) in group.iter().enumerate() {
            let ax = x.abs();
            let norm = ax * inv_scale;
            // Find the nearest fp4 level (same select cascade as the kernel).
            let level = if norm < FP4_BOUNDARIES[0] {
                FP4_LEVELS[0]
            } else if norm < FP4_BOUNDARIES[1] {
                FP4_LEVELS[1]
            } else if norm < FP4_BOUNDARIES[2] {
                FP4_LEVELS[2]
            } else if norm < FP4_BOUNDARIES[3] {
                FP4_LEVELS[3]
            } else if norm < FP4_BOUNDARIES[4] {
                FP4_LEVELS[4]
            } else if norm < FP4_BOUNDARIES[5] {
                FP4_LEVELS[5]
            } else if norm < FP4_BOUNDARIES[6] {
                FP4_LEVELS[6]
            } else {
                FP4_LEVELS[7]
            };
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            out[base + i] = sign * level * (group_max / 6.0);
        }
    }
    out
}

fn run_fp4_quant_dequant(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    assert_eq!(n % 32, 0, "n must be multiple of 32");
    let n_groups = n / 32;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), inp.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("out".into(), vec![0u8; n * 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    // The kernel uses simd_max (simdgroup reduction) — it's Reduction mode.
    let mut kernel = mt_fp4_quant_dequant::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // TPG = 32 (one simdgroup per threadgroup). Grid = [n_groups, 1, 1].
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("fp4_quant_dequant dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn fp4_quant_dequant_matches_cpu_oracle_basic() {
    let _g = gpu_lock();
    // 64 elements (2 groups of 32). Values spread across positive and
    // negative with magnitudes covering different fp4 levels.
    let inp: Vec<f32> = (0..64).map(|i| (i as f32 * 0.19) - 6.0).collect();
    let expected = cpu_fp4_quant_dequant(&inp);
    let actual = run_fp4_quant_dequant(&inp);

    let diff = max_abs_diff(&actual, &expected);
    // fp4 has only 8 levels — tolerance should be well within one level gap.
    assert!(diff < 1e-4, "fp4_quant_dequant basic max |diff| = {diff:.2e}");
}

#[test]
fn fp4_quant_dequant_preserves_sign() {
    let _g = gpu_lock();
    // 32 elements (1 group), mix of signs.
    let inp: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.5).collect();
    let actual = run_fp4_quant_dequant(&inp);

    for (i, (&a, &x)) in actual.iter().zip(inp.iter()).enumerate() {
        if x != 0.0 {
            assert_eq!(
                a.signum(),
                x.signum(),
                "sign mismatch at [{i}]: inp={x}, out={a}"
            );
        }
    }
}

#[test]
fn fp4_quant_dequant_output_not_all_zeros() {
    let _g = gpu_lock();
    let inp: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let actual = run_fp4_quant_dequant(&inp);
    assert!(actual.iter().any(|&v| v != 0.0), "fp4 output all zeros — empty kernel?");
}

#[test]
fn fp4_quant_dequant_uniform_group_uses_single_level() {
    let _g = gpu_lock();
    // All-positive group with the same magnitude → all map to level 6.0.
    let inp: Vec<f32> = vec![3.0f32; 32];
    let expected = cpu_fp4_quant_dequant(&inp);
    let actual = run_fp4_quant_dequant(&inp);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "fp4 uniform group diff = {diff:.2e}");
}

#[test]
fn fp4_quant_dequant_all_zero_group() {
    let _g = gpu_lock();
    // Zero-magnitude group → inv_scale=0 → output all zero.
    let inp = vec![0.0f32; 32];
    let actual = run_fp4_quant_dequant(&inp);
    for (i, &v) in actual.iter().enumerate() {
        assert_eq!(v, 0.0, "fp4 all-zero group [{i}] = {v} != 0");
    }
}

#[test]
fn fp4_quant_dequant_multi_group_independence() {
    let _g = gpu_lock();
    // Three groups with very different scales — verify that group[0]'s scale
    // doesn't bleed into group[1] or group[2].
    let mut inp = Vec::with_capacity(96);
    // Group 0: scale = 1.0 (max |x| = 1)
    inp.extend((0..32).map(|i| (i as f32 / 31.0) * 1.0 - 0.5));
    // Group 1: scale = 100.0
    inp.extend((0..32).map(|i| (i as f32 / 31.0) * 100.0 - 50.0));
    // Group 2: scale = 0.01
    inp.extend((0..32).map(|i| (i as f32 / 31.0) * 0.01 - 0.005));

    let expected = cpu_fp4_quant_dequant(&inp);
    let actual = run_fp4_quant_dequant(&inp);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-2, "fp4 multi-group independence diff = {diff:.2e}");
}

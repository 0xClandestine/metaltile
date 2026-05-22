//! GPU correctness for `mlx::fp_quantized::mt_fp4_quant_dequant` — the
//! NVFP4-style 4-bit quantize-dequantize round-trip.
//!
//! Each simdgroup of 32 consecutive elements shares one amax (`simd_max`).
//! The magnitude is scaled so the group max maps to 6, snapped to the fp4
//! codebook `{0, .5, 1, 1.5, 2, 3, 4, 6}` via a nested-`select` ladder,
//! then rescaled by `group_max / 6` and re-signed. The CPU oracle replays
//! the identical arithmetic — kernel and oracle implement the same grid,
//! so for any input not within a ULP of a codebook decision boundary they
//! agree bit-for-bit. The synthetic inputs here are deterministic (no RNG)
//! and verified to sit clear of every boundary.
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::fp_quantized::mt_fp4_quant_dequant;

/// The fp4 codebook — the eight non-negative magnitudes the format can
/// represent. The kernel re-signs after the snap, so these cover ±.
const FP4_CODEBOOK: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// fp4 codebook snap — maps a scaled magnitude to its nearest codebook
/// point. Mirrors the kernel's nested-`select` ladder exactly: each
/// threshold is the midpoint between adjacent codebook values.
fn fp4_snap(norm: f32) -> f32 {
    if norm < 0.25 {
        0.0
    } else if norm < 0.75 {
        0.5
    } else if norm < 1.25 {
        1.0
    } else if norm < 1.75 {
        1.5
    } else if norm < 2.5 {
        2.0
    } else if norm < 3.5 {
        3.0
    } else if norm < 5.0 {
        4.0
    } else {
        6.0
    }
}

/// CPU oracle: per-32-element-simdgroup max-scale → codebook snap →
/// rescale + sign. Mirrors the kernel's float arithmetic step for step.
fn oracle_fp4(inp: &[f32]) -> Vec<f32> {
    assert!(inp.len().is_multiple_of(32), "input length must be a multiple of 32");
    let mut out = vec![0.0f32; inp.len()];
    for (gi, group) in inp.chunks_exact(32).enumerate() {
        let group_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let inv_scale = if group_max > 0.0 { 6.0 / group_max } else { 0.0 };
        let rescale = group_max / 6.0;
        for (i, &x) in group.iter().enumerate() {
            let norm = x.abs() * inv_scale;
            let q = fp4_snap(norm);
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            out[gi * 32 + i] = sign * q * rescale;
        }
    }
    out
}

/// Dispatch the fp4 kernel over `inp` (length a multiple of 32). One
/// simdgroup (32 lanes) per group: `grid = [n/32,1,1]` threadgroups,
/// `tpg = [32,1,1]` — exactly `n` threads, so `simd_max` reduces the
/// per-group amax with no out-of-bounds lanes.
fn run_fp4(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    assert!(n.is_multiple_of(32), "fp4 dispatch needs a multiple of 32 elements");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_fp4_quant_dequant::kernel_ir_for();
    kernel.mode = KernelMode::Grid3D;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, 1, 1], [32, 1, 1])
        .expect("fp4 dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(n);
    out
}

/// A synthetic group of 32 spanning small, mid and large magnitudes plus
/// both signs and a zero. Values are deterministic and chosen to keep
/// scaled magnitudes clear of the codebook decision boundaries.
fn synthetic_group(seed: usize) -> Vec<f32> {
    (0..32)
        .map(|i| {
            // [-0.46, 0.5] in 0.03 steps — no value lands on a boundary
            // midpoint once scaled by the group max.
            let v = ((i * 7 + seed * 11) % 33) as f32 * 0.03 - 0.46;
            match i % 4 {
                0 => v * 10.0, // large
                1 => v * 0.05, // small
                2 => 0.0,      // exact zero
                _ => v,        // mid
            }
        })
        .collect()
}

#[test]
fn fp4_round_trip_matches_oracle() {
    let _g = gpu_lock();
    let inp: Vec<f32> = (0..4).flat_map(synthetic_group).collect();
    let expected = oracle_fp4(&inp);
    let actual = run_fp4(&inp);

    assert!(actual.iter().any(|&v| v != 0.0), "fp4: all-zero output (empty body?)");
    let diff = max_abs_diff(&actual, &expected);
    // GPU and CPU implement the identical codebook grid; the only
    // divergence is f32 ULP drift in the max-scale division (Metal
    // fast-math reciprocal). The synthetic inputs sit clear of every
    // codebook boundary, so a ULP cannot flip a cell — diff is ~0.
    assert!(diff < 1e-3, "fp4 round-trip: max |diff| = {diff:.2e}");
}

#[test]
fn fp4_codebook_values_round_trip_exactly() {
    let _g = gpu_lock();
    // Inputs that are exact codebook points scaled by a common factor:
    // group_max = 6*s → inv_scale = 1/s → norm == codebook value exactly.
    // Codebook points are the cell centres, maximally far from every
    // boundary, so the snap is the identity and the round-trip returns
    // the input unchanged.
    let scale = 4.0f32;
    let inp: Vec<f32> = (0..32)
        .map(|i| {
            let mag = FP4_CODEBOOK[i % 8] * scale;
            if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
    let actual = run_fp4(&inp);

    let diff = max_abs_diff(&actual, &inp);
    assert!(diff < 1e-4, "fp4 codebook-point round-trip not identity: max |diff| = {diff:.2e}");
}

#[test]
fn fp4_preserves_sign() {
    let _g = gpu_lock();
    // Alternating signs, all non-zero magnitudes — the round-trip must
    // never flip a sign (a regression where `sign` is dropped or the
    // re-sign multiply is lost would surface here).
    let inp: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.5 } else { -2.25 }).collect();
    let actual = run_fp4(&inp);
    for (i, (&x, &y)) in inp.iter().zip(actual.iter()).enumerate() {
        assert!((x >= 0.0) == (y >= 0.0), "fp4 flipped sign at [{i}]: in {x}, out {y}");
    }
}

#[test]
fn fp4_uniform_group_is_identity() {
    let _g = gpu_lock();
    // Every element equal → group_max == |value| → norm == 6 for all →
    // snap → 6 → result == sign * 6 * (group_max/6) == the input.
    let inp: Vec<f32> = vec![3.75f32; 32];
    let actual = run_fp4(&inp);
    for (i, &y) in actual.iter().enumerate() {
        let rel = (y - 3.75).abs() / 3.75;
        assert!(rel < 1e-3, "fp4 uniform-group drift at [{i}]: out {y}");
    }
}

#[test]
fn fp4_zero_group_stays_zero() {
    let _g = gpu_lock();
    // All zeros → group_max == 0 → inv_scale clamps to 0 → every output 0.
    // Pins the `group_max > 0` guard against a divide-by-zero NaN.
    let inp: Vec<f32> = vec![0.0f32; 32];
    let actual = run_fp4(&inp);
    for (i, &y) in actual.iter().enumerate() {
        assert!(y == 0.0, "fp4 zero group produced non-zero at [{i}]: {y}");
    }
}

#[test]
fn fp4_output_is_on_the_codebook_grid() {
    let _g = gpu_lock();
    // Every dequantized magnitude must be a codebook point times the
    // group's rescale factor — i.e. `|out| / rescale` lands on the grid.
    let inp: Vec<f32> = (0..2).flat_map(synthetic_group).collect();
    let actual = run_fp4(&inp);
    for (gi, group) in actual.chunks_exact(32).enumerate() {
        let in_group = &inp[gi * 32..gi * 32 + 32];
        let group_max = in_group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rescale = group_max / 6.0;
        for (i, &y) in group.iter().enumerate() {
            let codebook_val = if rescale > 0.0 { y.abs() / rescale } else { 0.0 };
            let on_grid = FP4_CODEBOOK.iter().any(|&c| (c - codebook_val).abs() < 1e-2);
            assert!(on_grid, "fp4 off-grid output at group {gi} [{i}]: {y} → {codebook_val}");
        }
    }
}

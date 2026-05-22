//! GPU correctness for `mlx::fp_quantized::mt_fp8_e4m3_quant_dequant` /
//! `mt_fp8_e5m2_quant_dequant` — the fp8 quantize-dequantize round-trip.
//!
//! fp8 quant-dequant is a pure arithmetic transform (per-group max-scale
//! → round each value's mantissa to the format's mantissa-bit count →
//! rescale). The CPU oracle replays the identical computation in f32 —
//! the GPU and oracle must agree to f32 arithmetic noise (the round-trip
//! is lossy vs the *original* input by construction, but kernel and
//! oracle implement the same fp8 grid, so they agree tightly).
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::fp_quantized::{mt_fp8_e4m3_quant_dequant, mt_fp8_e5m2_quant_dequant};

/// fp8 format parameters — must match the `fp8_kernel!` instantiations.
struct Fp8Fmt {
    mantissa_bits: f32,
    e_min: f32,
    e_max: f32,
    fp8_max: f32,
}

const E4M3: Fp8Fmt = Fp8Fmt { mantissa_bits: 3.0, e_min: -6.0, e_max: 8.0, fp8_max: 448.0 };
const E5M2: Fp8Fmt = Fp8Fmt { mantissa_bits: 2.0, e_min: -14.0, e_max: 15.0, fp8_max: 57344.0 };

/// CPU oracle: the fp8 quant-dequant round-trip, one group of 32 at a
/// time (each group is one simdgroup with its own amax). Mirrors the
/// kernel's float-arithmetic mantissa rounding exactly.
fn oracle_fp8_round_trip(inp: &[f32], fmt: &Fp8Fmt) -> Vec<f32> {
    assert!(inp.len().is_multiple_of(32), "input length must be a multiple of 32");
    let mut out = vec![0.0f32; inp.len()];
    for (gi, group) in inp.chunks_exact(32).enumerate() {
        let group_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let inv_scale = if group_max > 0.0 { fmt.fp8_max / group_max } else { 0.0 };
        let rescale = group_max / fmt.fp8_max;
        for (i, &x) in group.iter().enumerate() {
            let norm = x.abs() * inv_scale;
            let q = if norm > 0.0 {
                let raw_e = norm.log2().floor();
                let e = raw_e.clamp(fmt.e_min, fmt.e_max);
                let quantum = (e - fmt.mantissa_bits).exp2();
                (norm / quantum).round() * quantum
            } else {
                0.0
            };
            let q_clamped = q.min(fmt.fp8_max);
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            out[gi * 32 + i] = sign * q_clamped * rescale;
        }
    }
    out
}

/// Dispatch an fp8 quant-dequant kernel over `inp` (length a multiple of 32).
fn run_fp8(inp: &[f32], kernel: metaltile_core::ir::Kernel) -> Vec<f32> {
    let n = inp.len();
    assert!(n.is_multiple_of(32));

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel;
    kernel.mode = KernelMode::Grid3D;

    // One simdgroup per group of 32 elements: grid = [n,1,1], tpg = [32,1,1].
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n, 1, 1], [32, 1, 1])
        .expect("fp8 dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(n);
    out
}

/// A synthetic group of 32 spanning small, mid and large magnitudes plus
/// both signs and a zero — exercises the subnormal clamp, normal grid
/// and the saturating clamp.
fn synthetic_group(seed: usize) -> Vec<f32> {
    (0..32)
        .map(|i| {
            let v = ((i * 37 + seed * 13) % 100) as f32 * 0.01 - 0.5; // [-0.5, 0.49]
            // Spread across magnitudes: scale a few entries up/down.
            match i % 4 {
                0 => v * 100.0,
                1 => v * 0.001,
                2 => 0.0,
                _ => v,
            }
        })
        .collect()
}

#[test]
fn fp8_e4m3_round_trip_matches_oracle() {
    let _g = gpu_lock();
    let inp: Vec<f32> = (0..4).flat_map(synthetic_group).collect();
    let expected = oracle_fp8_round_trip(&inp, &E4M3);
    let actual = run_fp8(&inp, mt_fp8_e4m3_quant_dequant::kernel_ir_for());
    assert!(actual.iter().any(|&v| v != 0.0), "fp8 e4m3: all-zero output (empty body?)");
    let diff = max_abs_diff(&actual, &expected);
    // GPU vs CPU implement the identical fp8 grid — only f32 arithmetic
    // drift in log2/exp2 separates them.
    assert!(diff < 1e-2, "fp8 e4m3 round-trip: max |diff| = {diff:.2e}");
}

#[test]
fn fp8_e5m2_round_trip_matches_oracle() {
    let _g = gpu_lock();
    let inp: Vec<f32> = (0..4).flat_map(synthetic_group).collect();
    let expected = oracle_fp8_round_trip(&inp, &E5M2);
    let actual = run_fp8(&inp, mt_fp8_e5m2_quant_dequant::kernel_ir_for());
    assert!(actual.iter().any(|&v| v != 0.0), "fp8 e5m2: all-zero output (empty body?)");
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-1, "fp8 e5m2 round-trip: max |diff| = {diff:.2e}");
}

#[test]
fn fp8_e4m3_preserves_sign() {
    let _g = gpu_lock();
    // A group with a clear sign pattern — the round-trip must keep it.
    let inp: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.5 } else { -2.25 }).collect();
    let actual = run_fp8(&inp, mt_fp8_e4m3_quant_dequant::kernel_ir_for());
    for (i, (&x, &y)) in inp.iter().zip(actual.iter()).enumerate() {
        assert!((x >= 0.0) == (y >= 0.0), "fp8 e4m3 flipped sign at [{i}]: in {x}, out {y}",);
    }
}

#[test]
fn fp8_e4m3_round_trip_is_near_identity_for_exact_values() {
    let _g = gpu_lock();
    // Values that are exactly representable as e4m3 mantissa fractions
    // (k/8 of a power of two) survive the round-trip almost unchanged
    // once the group max-scale is the identity. Scale every value so the
    // group max is exactly fp8_max — then inv_scale * rescale == 1.
    let group_max = 448.0f32;
    let inp: Vec<f32> = (0..32)
        .map(|i| {
            // Mantissa fractions 1.0, 1.125, ..., representable exactly.
            let frac = 1.0 + (i % 8) as f32 / 8.0;
            let v = frac * 32.0; // a mid-range binade
            if i == 0 { group_max } else { v }
        })
        .collect();
    let actual = run_fp8(&inp, mt_fp8_e4m3_quant_dequant::kernel_ir_for());
    let expected = oracle_fp8_round_trip(&inp, &E4M3);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-2, "fp8 e4m3 exact-value round-trip: max |diff| = {diff:.2e}");
}

#[test]
fn fp8_e5m2_saturates_large_values() {
    let _g = gpu_lock();
    // All values equal → group_max == that value → every norm == fp8_max
    // → the round-trip is the identity (the saturating clamp is a no-op
    // at exactly fp8_max). Confirms the clamp doesn't corrupt the top.
    let inp: Vec<f32> = vec![12345.0f32; 32];
    let actual = run_fp8(&inp, mt_fp8_e5m2_quant_dequant::kernel_ir_for());
    for (i, &y) in actual.iter().enumerate() {
        let rel = (y - 12345.0).abs() / 12345.0;
        assert!(rel < 1e-2, "fp8 e5m2 uniform-group drift at [{i}]: out {y}");
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::moe_down_swiglu_accum::ffai_moe_down_swiglu_accum_int4_chain8`.
//!
//! Pins the three-stage fusion (swiglu + indexed dequant-gemv + scalar-FMA
//! chain8) of FFAI's MoE GPU-router decode path against a CPU oracle
//! that runs the same three stages independently:
//!
//!   for k in 0..8:
//!     inner[k]  = silu(gate[k]) * up[k]                    (mt_swiglu)
//!     down[k]   = W_down[expert[k]] @ inner[k]             (indexed dequant-gemv)
//!   out = Σ_k slot_weights[k] * down[k]                    (scalar_fma_chain8)
//!
//! Coverage points:
//! - 8-slot loop unrolling: every slot index 0..8 must contribute to
//!   the final accumulator (a fall-through bug at any slot drops 1/8
//!   of the magnitude).
//! - `expert_indices` is honoured per-slot: each slot k reads a
//!   different expert's weight slab. The CPU oracle exercises a
//!   non-trivial permutation (no two slots share an expert) so a
//!   constant-expert bug would show up as wrong magnitudes per row.
//! - `slot_weights` scales each slot before accumulation. Weights
//!   sum to 1 in production (softmax over chosen-k) but the test uses
//!   un-normalised values to expose any silent normalisation.
//! - Threadgroup-scratch `tg_inner` reuse: every slot writes-then-reads
//!   the same scratch buffer; missing the WAR barrier between slots
//!   would surface as bleed-through from later slots into earlier.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_down_swiglu_accum::ffai_moe_down_swiglu_accum_int4_chain8;

/// Affine per-group int4 quantize of one weight row, nibble-packed.
/// (Same shape as `batched_qkv_qgemv_gpu_correctness.rs`, the int4
/// pack-strided layout the indexed dequant-gemv consumes.)
fn quantize_int4_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let in_dim = row.len();
    let n_groups = in_dim / group_size;
    let mut packed = vec![0u32; in_dim / 8];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let gs = &row[g * group_size..(g + 1) * group_size];
        let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / 15.0 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in gs.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
            let d = g * group_size + i;
            packed[d / 8] |= q << ((d % 8) * 4);
        }
    }
    (packed, scales, biases)
}

/// Quantize a stacked-experts weight tensor of shape `[n_experts, out_dim, in_dim]`.
/// Returns flat row-major (packed, scales, biases) buffers matching the
/// kernel's `[n_experts, out_dim, in_dim/8]` and `[n_experts, out_dim, n_groups]`.
fn quantize_stacked(
    flat: &[f32],
    n_experts: usize,
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut w = Vec::new();
    let mut s = Vec::new();
    let mut b = Vec::new();
    for e in 0..n_experts {
        for row in 0..out_dim {
            let base = e * out_dim * in_dim + row * in_dim;
            let (pw, ps, pb) = quantize_int4_row(&flat[base..base + in_dim], group_size);
            w.extend(pw);
            s.extend(ps);
            b.extend(pb);
        }
    }
    (w, s, b)
}

/// CPU oracle for one slot's dequant-gemv against expert `e`.
///
/// Indexes into the stacked buffers exactly the way the kernel does
/// (`expert * out_dim * <per-row stride>` + per-row stride), to keep the
/// reference structurally aligned with the GPU code path.
fn dequant_gemv_one_expert(
    weights_stacked: &[u32],
    scales_stacked: &[f32],
    biases_stacked: &[f32],
    inner: &[f32],
    expert: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
) -> Vec<f32> {
    let n_packs_per_row = in_dim / 8;
    let n_groups = in_dim / group_size;
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_groups;
    (0..out_dim)
        .map(|row| {
            let rpo = weight_expert_off + row * n_packs_per_row;
            let rgo = scale_expert_off + row * n_groups;
            let mut acc = 0.0_f32;
            for pack_idx in 0..n_packs_per_row {
                let g = pack_idx / (group_size / 8);
                let scale = scales_stacked[rgo + g];
                let bias = biases_stacked[rgo + g];
                let packed = weights_stacked[rpo + pack_idx];
                let p_off = pack_idx * 8;
                for i in 0..8 {
                    let q = (packed >> (i * 4)) & 0xF;
                    acc += (q as f32 * scale + bias) * inner[p_off + i];
                }
            }
            acc
        })
        .collect()
}

/// CPU oracle for the full three-stage fusion.
///
/// Runs the unfused chain (swiglu, then 8 separate indexed dequant-gemv,
/// then scalar-FMA accumulation) at f32 precision. The kernel's
/// numerical equivalent is exactly this, modulo the order in which the
/// final per-thread `reduce_sum` folds the 8 slots' partials.
#[allow(clippy::too_many_arguments)]
fn naive_oracle(
    gates: &[Vec<f32>; 8],
    ups: &[Vec<f32>; 8],
    expert_indices: &[u32; 8],
    slot_weights: &[f32; 8],
    weights_stacked: &[u32],
    scales_stacked: &[f32],
    biases_stacked: &[f32],
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut acc = vec![0.0_f32; out_dim];
    for k in 0..8 {
        // swiglu: inner[d] = silu(gate[d]) * up[d]
        let inner: Vec<f32> = (0..in_dim)
            .map(|d| {
                let g = gates[k][d];
                let u = ups[k][d];
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();
        // indexed dequant-gemv: down[row] = W_down[expert[k]][row, :] · inner
        let down = dequant_gemv_one_expert(
            weights_stacked,
            scales_stacked,
            biases_stacked,
            &inner,
            expert_indices[k] as usize,
            in_dim,
            out_dim,
            group_size,
        );
        // scalar-FMA chain8 step: acc[i] += slot_weights[k] * down[i]
        for i in 0..out_dim {
            acc[i] += slot_weights[k] * down[i];
        }
    }
    acc
}

/// Deterministic pseudo-random source (xorshift), gentle range so the
/// silu nonlinearity is exercised across both branches (gate < 0 ramps
/// down, gate > 0 ramps up).
fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dt: Dt,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    n_experts: usize,
    tol: f32,
) {
    let _g = gpu_lock();
    assert_eq!(in_dim, 768, "kernel pins tg_inner alloc at 768 (Qwen3.6-A3B moeIntermediate)");
    assert!(n_experts >= 8, "need at least 8 distinct experts to exercise the slot permutation");

    // Build 8 gate / 8 up inputs (each [in_dim]), per-dtype round-trip
    // so the oracle sees the same load-cast quantisation the kernel does.
    let gates: [Vec<f32>; 8] = std::array::from_fn(|k| {
        source(in_dim, 0x1001 + k as u64 * 0x91, 3.0, 0.0)
            .iter()
            .map(|&v| dt.round(v))
            .collect()
    });
    let ups: [Vec<f32>; 8] = std::array::from_fn(|k| {
        source(in_dim, 0x2002 + k as u64 * 0x91, 2.0, 0.05)
            .iter()
            .map(|&v| dt.round(v))
            .collect()
    });

    // 8 distinct expert ids spread across the full n_experts range ,
    // catches a constant-expert / off-by-one expert offset bug.
    let expert_indices: [u32; 8] = [3, 17, 41, 58, 72, 89, 104, 121];
    assert!(expert_indices.iter().all(|&e| (e as usize) < n_experts));

    // Slot weights, un-normalised so silent renormalisation surfaces.
    let slot_weights_f32: [f32; 8] =
        [0.31, 0.19, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03].map(|v| dt.round(v));

    // Stacked W_down: [n_experts, out_dim, in_dim], f32 → int4-pack.
    let w_stacked_f32 = source(n_experts * out_dim * in_dim, 0x3003, 0.5, 0.0);
    let (w_packed, scales_f32, biases_f32) =
        quantize_stacked(&w_stacked_f32, n_experts, out_dim, in_dim, group_size);
    let scales_dt: Vec<f32> = scales_f32.iter().map(|&v| dt.round(v)).collect();
    let biases_dt: Vec<f32> = biases_f32.iter().map(|&v| dt.round(v)).collect();

    // ── CPU oracle ─────────────────────────────────────────────────────
    let expected = naive_oracle(
        &gates,
        &ups,
        &expert_indices,
        &slot_weights_f32,
        &w_packed,
        &scales_dt,
        &biases_dt,
        in_dim,
        out_dim,
        group_size,
    );

    // ── GPU dispatch ───────────────────────────────────────────────────
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for k in 0..8 {
        buffers.insert(format!("gate_{k}"), pack_bytes(&gates[k], dt));
        buffers.insert(format!("up_{k}"), pack_bytes(&ups[k], dt));
    }
    buffers.insert("expert_indices".into(), pack_u32_bytes(&expert_indices));
    buffers.insert("slot_weights".into(), pack_bytes(&slot_weights_f32, dt));
    buffers.insert("weights_stacked".into(), pack_u32_bytes(&w_packed));
    buffers.insert("scales_stacked".into(), pack_bytes(&scales_dt, dt));
    buffers.insert("biases_stacked".into(), pack_bytes(&biases_dt, dt));
    buffers.insert("output".into(), pack_bytes(&vec![0.0_f32; out_dim], dt));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_moe_down_swiglu_accum_int4_chain8::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // One TG per output row; 128 lanes per TG, balanced for
    // n_packs_per_row=96 at in_dim=768 (1-stride 6-iter inner loop, no
    // tail-lane idle waste).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim, 1, 1], [128, 1, 1])
        .expect("ffai_moe_down_swiglu_accum_int4_chain8 dispatch");
    let actual = unpack_bytes(result.outputs.get("output").expect("output"), dt);

    assert_eq!(actual.len(), out_dim);
    assert!(actual.iter().any(|&v| v != 0.0), "output is all zeros");

    // Per-element relative tolerance, matching `batched_qkv_qgemv_gpu_correctness`.
    let max_rel = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-3))
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel <= tol,
        "dt={:?}: max rel = {max_rel:.3e} > {tol:.3e}",
        dt as u32,
    );
}

// Production-shape coverage: hidden=2048, moeIntermediate=768, gs=64,
// n_experts=128: Qwen3.6-A3B exactly. Two dtypes per the acceptance
// criteria.

#[test]
fn moe_down_swiglu_accum_qwen36_f32() {
    run_case(Dt::F32, 768, 2048, 64, 128, 1e-3);
}

#[test]
fn moe_down_swiglu_accum_qwen36_bf16() {
    run_case(Dt::Bf16, 768, 2048, 64, 128, 5e-2);
}

// f16 too (confirms the load-side `cast::<f32>()` keeps the chain8
// accumulation stable across the third float dtype the kernel ships.
#[test]
fn moe_down_swiglu_accum_qwen36_f16() {
    run_case(Dt::F16, 768, 2048, 64, 128, 5e-2);
}

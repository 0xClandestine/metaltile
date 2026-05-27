//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end correctness for `ffai::dequant_gather_int{2,4}` on real Metal.
//!
//! Pins the dequantizing-gather arithmetic: per output element
//! `(token, d)`, look up the row `indices[token]` selects, unpack the
//! N-bit value at bit offset `d*N`, and dequantize via `q*scale + bias`
//! with the per-group `(scale, bias)`.
//!
//! Coverage rationale: the `dequant_gather_int{2,3,4,5,6,8}` kernels had
//! their bodies silently emptied by PR #19's macro refactor — they
//! shipped as empty MSL producing all-zeros output (restored in this
//! PR). They carry no `BenchDispatch` variant, so `tile bench` never
//! exercises them, and `xcrun metal` happily compiles an empty body.
//! This GPU correctness test is the only thing that would catch a
//! re-break: it dispatches on the real device and compares against a
//! naive CPU reference. int4 is the representative pack-strided
//! (nibble-aligned) bit width; int2 (added with the 2-bit quant work)
//! pins the smallest power-of-two layout (16 codes per u32).
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::dequant_gather::{dequant_gather_int2, dequant_gather_int4};

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Per-group affine quantize one vocab row to `bits`-wide values packed as
/// a u32 bit-stream. Power-of-two `bits` (2, 4, 8) means values never span
/// a word boundary, matching the kernel's `spill == 0` fast path.
fn quantize_row(row: &[f32], group_size: usize, bits: u32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let hidden = row.len();
    assert_eq!(hidden % group_size, 0, "hidden must be a multiple of group_size");
    assert_eq!(
        (hidden * bits as usize) % 32,
        0,
        "hidden * bits must be a multiple of 32 (u32 row boundary)",
    );
    let vals_per_pack = 32 / bits as usize;
    let max_q = (1u32 << bits) - 1;
    let n_groups = hidden / group_size;
    let n_u32 = hidden * bits as usize / 32;
    let mut packed = vec![0u32; n_u32];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];

    for g in 0..n_groups {
        let g_slice = &row[g * group_size..(g + 1) * group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / max_q as f32 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, max_q as f32) as u32;
            let d = g * group_size + i;
            packed[d / vals_per_pack] |= q << ((d % vals_per_pack) * bits as usize);
        }
    }
    (packed, scales, biases)
}

/// CPU equivalent of `dequant_gather`: for each `(token, d)`, gather row
/// `indices[token]`, unpack the `bits`-wide value at bit offset `d*bits`
/// from the row's u32 bit-stream, dequantize.
fn naive_dequant_gather(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    indices: &[u32],
    hidden: usize,
    group_size: usize,
    bits: u32,
) -> Vec<f32> {
    let n_tokens = indices.len();
    let groups_per_row = hidden / group_size;
    let vals_per_pack = 32 / bits as usize;
    let u32_per_row = hidden / vals_per_pack;
    let mask = (1u32 << bits) - 1;
    let mut out = vec![0.0_f32; n_tokens * hidden];
    for token in 0..n_tokens {
        let token_id = indices[token] as usize;
        for d in 0..hidden {
            let word = weight[token_id * u32_per_row + d / vals_per_pack];
            let q = ((word >> ((d % vals_per_pack) * bits as usize)) & mask) as f32;
            let g = d / group_size;
            let scale = scales[token_id * groups_per_row + g];
            let bias = biases[token_id * groups_per_row + g];
            out[token * hidden + d] = q * scale + bias;
        }
    }
    out
}

/// Dispatch the dequant_gather kernel for the given `bits` and assert it
/// matches the naive CPU oracle. Empty-body regressions surface as
/// all-zeros output; the explicit `any(|v| v != 0.0)` assert below makes
/// that intent unmissable.
fn run_one_test(bits: u32, hidden: usize, group_size: usize, tol: f32) {
    let _g = gpu_lock();

    let vocab = 8usize;
    let vals_per_pack = 32 / bits as usize;
    let n_groups = hidden / group_size;

    // Quantize a synthetic vocab table — each row a deterministic ramp
    // kept small so per-group min/max maps cleanly into the bit grid.
    let mut weight: Vec<u32> = Vec::with_capacity(vocab * hidden / vals_per_pack);
    let mut scales: Vec<f32> = Vec::with_capacity(vocab * n_groups);
    let mut biases: Vec<f32> = Vec::with_capacity(vocab * n_groups);
    for r in 0..vocab {
        let row: Vec<f32> = (0..hidden).map(|d| (((r + d) % 17) as f32 - 8.0) * 0.05).collect();
        let (pk, sc, bs) = quantize_row(&row, group_size, bits);
        weight.extend(pk);
        scales.extend(sc);
        biases.extend(bs);
    }

    // Gather order deliberately non-monotonic + repeats a row (idx 4
    // appears twice) so a token→row indexing bug can't hide.
    let indices: Vec<u32> = vec![3, 0, 7, 1, 4, 4];
    let n_tokens = indices.len();

    let expected =
        naive_dequant_gather(&weight, &scales, &biases, &indices, hidden, group_size, bits);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), pack_u32_bytes(&weight));
    buffers.insert("scales".into(), f32_slice_to_bytes(&scales));
    buffers.insert("biases".into(), f32_slice_to_bytes(&biases));
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), vec![0u8; n_tokens * hidden * 4]);
    buffers.insert("hidden".into(), (hidden as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = match bits {
        2 => dequant_gather_int2::kernel_ir_for(DType::F32),
        4 => dequant_gather_int4::kernel_ir_for(DType::F32),
        _ => unreachable!("only int2 / int4 wired here"),
    };
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per output element. `n_tokens` groups of
    // `hidden` threads — `program_id::<0>()` walks 0..n_tokens*hidden.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tokens, 1, 1], [hidden, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = bytes_to_f32_vec(out_bytes);

    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < tol,
        "dequant_gather int{bits} f32: max |diff| = {diff:.2e} (expected < {tol:.2e})",
    );

    // Guard against the regression this test exists for: an empty
    // kernel body would leave `out` all-zeros.
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "dequant_gather int{bits} emitted all-zeros output — kernel body is empty",
    );
}

#[test]
fn dequant_gather_int4_matches_naive_cpu_reference_f32() { run_one_test(4, 256, 64, 1e-4); }

#[test]
fn dequant_gather_int2_matches_naive_cpu_reference_f32() {
    // int2: 16 codes per u32, quant step = range / 3. Group size must be a
    // multiple of `vals_per_pack = 16`; 64 satisfies that and gives 4
    // groups per row across hidden=256.
    run_one_test(2, 256, 64, 1e-4);
}

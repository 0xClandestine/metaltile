//! End-to-end correctness test for `ffai::dequant_gather_int4` on real Metal.
//!
//! Pins the dequantizing-gather arithmetic: per output element
//! `(token, d)`, look up the row `indices[token]` selects, unpack the
//! 4-bit value at bit offset `d*4`, and dequantize via `q*scale + bias`
//! with the per-group `(scale, bias)`.
//!
//! Coverage rationale: the `dequant_gather_int{3,4,5,6,8}` kernels had
//! their bodies silently emptied by PR #19's macro refactor — they
//! shipped as empty MSL producing all-zeros output (restored in this
//! PR). They carry no `BenchDispatch` variant, so `tile bench` never
//! exercises them, and `xcrun metal` happily compiles an empty body.
//! This GPU correctness test is the only thing that would catch a
//! re-break: it dispatches on the real device and compares against a
//! naive CPU reference. int4 is the representative bit width — its
//! 8-nibbles-per-u32 layout is the same bit-stream format the other
//! widths walk, just nibble-aligned so no word-spill.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{gpu_lock, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes, Dt};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::dequant_gather::dequant_gather_int4;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Per-group int4 quantize of one vocab row → (packed u32 words, scales,
/// biases). 8 nibbles per u32; `bits=4` is nibble-aligned so no value
/// ever spans a word boundary. Mirrors the format the kernel decodes.
fn quantize_row_int4(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let hidden = row.len();
    assert_eq!(hidden % group_size, 0, "hidden must be a multiple of group_size");
    assert_eq!(hidden % 8, 0, "int4 needs hidden divisible by 8 (8 nibbles per u32)");
    let n_groups = hidden / group_size;
    let mut packed = vec![0u32; hidden / 8];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];

    for g in 0..n_groups {
        let g_slice = &row[g * group_size..(g + 1) * group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / 15.0 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
            let d = g * group_size + i;
            packed[d / 8] |= q << ((d % 8) * 4);
        }
    }
    (packed, scales, biases)
}

/// CPU equivalent of `dequant_gather_int4`: for each `(token, d)`,
/// gather row `indices[token]`, unpack nibble `d`, dequantize.
fn naive_dequant_gather(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    indices: &[u32],
    hidden: usize,
    group_size: usize,
) -> Vec<f32> {
    let n_tokens = indices.len();
    let groups_per_row = hidden / group_size;
    let u32_per_row = hidden / 8; // hidden * 4 bits / 32
    let mut out = vec![0.0_f32; n_tokens * hidden];
    for token in 0..n_tokens {
        let token_id = indices[token] as usize;
        for d in 0..hidden {
            let word = weight[token_id * u32_per_row + d / 8];
            let q = ((word >> ((d % 8) * 4)) & 0xf) as f32;
            let g = d / group_size;
            let scale = scales[token_id * groups_per_row + g];
            let bias = biases[token_id * groups_per_row + g];
            out[token * hidden + d] = q * scale + bias;
        }
    }
    out
}

#[test]
fn dequant_gather_int4_matches_naive_cpu_reference_f32() {
    let _g = gpu_lock();

    let vocab = 8usize;
    let hidden = 256usize;
    let group_size = 64usize;
    let n_groups = hidden / group_size;

    // Quantize a synthetic vocab table — each row a deterministic ramp
    // kept small so per-group min/max maps cleanly into 4-bit levels.
    let mut weight: Vec<u32> = Vec::with_capacity(vocab * hidden / 8);
    let mut scales: Vec<f32> = Vec::with_capacity(vocab * n_groups);
    let mut biases: Vec<f32> = Vec::with_capacity(vocab * n_groups);
    for r in 0..vocab {
        let row: Vec<f32> = (0..hidden).map(|d| (((r + d) % 17) as f32 - 8.0) * 0.05).collect();
        let (pk, sc, bs) = quantize_row_int4(&row, group_size);
        weight.extend(pk);
        scales.extend(sc);
        biases.extend(bs);
    }

    // Gather order deliberately non-monotonic + repeats a row (idx 4
    // appears twice) so a token→row indexing bug can't hide.
    let indices: Vec<u32> = vec![3, 0, 7, 1, 4, 4];
    let n_tokens = indices.len();

    let expected = naive_dequant_gather(&weight, &scales, &biases, &indices, hidden, group_size);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), pack_u32_bytes(&weight));
    buffers.insert("scales".into(), f32_slice_to_bytes(&scales));
    buffers.insert("biases".into(), f32_slice_to_bytes(&biases));
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), vec![0u8; n_tokens * hidden * 4]);
    buffers.insert("hidden".into(), (hidden as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = dequant_gather_int4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per output element. `n_tokens` groups of
    // `hidden` threads — `program_id::<0>()` walks 0..n_tokens*hidden.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tokens, 1, 1], [hidden, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = bytes_to_f32_vec(out_bytes);

    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "dequant_gather int4 f32: max |diff| = {diff:.2e} (expected < 1e-4)");

    // Guard against the regression this test exists for: an empty
    // kernel body would leave `out` all-zeros. The reference has
    // non-zero values, so a zero `actual` would fail `max_abs_diff`
    // above — but assert it explicitly so the intent is unmissable.
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "dequant_gather emitted all-zeros output — kernel body is empty (PR #19 regression)",
    );
}

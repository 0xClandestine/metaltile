//! End-to-end correctness test for `ffai::aura_value_int4` on real Metal.
//!
//! Grid3D kernel — `(dim, q_heads, 1)` threads, one thread per
//! `(q_head, dim)` output element. Each thread runs a sequential loop
//! over tokens and accumulates its dim slot's compressed-domain value:
//!
//!   `output[head, d] = Σ_t weight[head, t] · norm[kvh, t] · codebook[unpack(packed[kvh, t, d])]`
//!
//! where `kvh = head / repeat_count` (GQA fan-out) and tokens whose
//! `weight < sparse_threshold` are skipped entirely.
//!
//! Covers: f32, an MHA case (`repeat_count = 1`), a GQA case
//! (`repeat_count = 2`), and a `sparse_threshold` that actually skips
//! some tokens.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::aura_value::aura_value_int4;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Bit-pack a flat `[kv_heads, tokens, dim]` int4 index array into
/// `[kv_heads, tokens, packed_width]` u32 words. Mirrors what
/// `aura_encode` produces and what the kernel's bit-unpack reads back.
fn pack_int4_indices(indices: &[u32], kv_heads: usize, tokens: usize, dim: usize) -> Vec<u32> {
    let bits = 4;
    let packed_width = (dim * bits).div_ceil(32);
    let mut packed = vec![0u32; kv_heads * tokens * packed_width];
    for kvh in 0..kv_heads {
        for t in 0..tokens {
            for d in 0..dim {
                let idx = indices[(kvh * tokens + t) * dim + d];
                let bit_offset = d * bits;
                let word_idx = bit_offset / 32;
                let shift = bit_offset & 31;
                packed[(kvh * tokens + t) * packed_width + word_idx] |= (idx & 0xf) << shift;
            }
        }
    }
    packed
}

/// Naive CPU reference for `aura_value`. Mirrors the kernel exactly:
/// per `(q_head, d)` element, sum the codebook-decoded contribution of
/// every token whose weight clears `sparse_threshold`.
#[allow(clippy::too_many_arguments)]
fn naive_aura_value(
    weights: &[f32],
    indices: &[u32],
    norms: &[f32],
    codebook: &[f32],
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    dim: usize,
    sparse_threshold: f32,
) -> Vec<f32> {
    let repeat = q_heads / kv_heads;
    let mut out = vec![0.0_f32; q_heads * dim];
    for qh in 0..q_heads {
        let kvh = qh / repeat;
        for d in 0..dim {
            let mut acc = 0.0_f32;
            for t in 0..tokens {
                let w = weights[qh * tokens + t];
                if w >= sparse_threshold {
                    let norm_val = norms[kvh * tokens + t];
                    let q = indices[(kvh * tokens + t) * dim + d];
                    let centroid = codebook[q as usize];
                    acc += w * norm_val * centroid;
                }
            }
            out[qh * dim + d] = acc;
        }
    }
    out
}

/// Run the kernel for one configuration and assert it matches the
/// naive CPU reference within fp32 tolerance.
fn run_aura_value_case(
    q_heads: usize,
    kv_heads: usize,
    tokens: usize,
    sparse_threshold: f32,
    label: &str,
) {
    let bits = 4usize;
    let dim = 128usize;
    let packed_width = (dim * bits).div_ceil(32); // 16 u32 words.
    let repeat = q_heads / kv_heads;
    assert_eq!(q_heads % kv_heads, 0, "q_heads must be a multiple of kv_heads");

    // 16-level symmetric codebook in [-1, 1].
    let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();

    // Pseudo-random codebook indices in [0, 16).
    let indices: Vec<u32> =
        (0..kv_heads * tokens * dim).map(|i| ((i * 13 + 5) % 16) as u32).collect();
    let packed = pack_int4_indices(&indices, kv_heads, tokens, dim);

    // Per-(kv_head, token) norm-correction factors.
    let norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.4 + 0.07 * i as f32).collect();

    // Per-(q_head, token) attention weights. Deliberately span both
    // sides of any non-zero `sparse_threshold` so the skip branch is
    // genuinely exercised: roughly half the weights land below 0.1.
    let weights: Vec<f32> = (0..q_heads * tokens)
        .map(|i| {
            let phase = (i * 7 + 3) % 10;
            phase as f32 * 0.04 // 0.0, 0.04, …, 0.36
        })
        .collect();

    let expected = naive_aura_value(
        &weights,
        &indices,
        &norms,
        &codebook,
        q_heads,
        kv_heads,
        tokens,
        dim,
        sparse_threshold,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weights".into(), f32_slice_to_bytes(&weights));
    buffers.insert("packed".into(), pack_u32_bytes(&packed));
    buffers.insert("norms".into(), f32_slice_to_bytes(&norms));
    buffers.insert("codebook".into(), f32_slice_to_bytes(&codebook));
    buffers.insert("output".into(), vec![0u8; q_heads * dim * 4]);
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    buffers.insert("packed_width".into(), (packed_width as u32).to_le_bytes().to_vec());
    buffers.insert("tokens".into(), (tokens as u32).to_le_bytes().to_vec());
    buffers.insert("repeat_count".into(), (repeat as u32).to_le_bytes().to_vec());
    buffers.insert("sparse_threshold".into(), sparse_threshold.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = aura_value_int4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: gid.x = d, gid.y = q_head. One thread per output element;
    // tg.x carries the full `dim` extent, q_heads via grid_groups on y.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, q_heads, 1], [dim, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("output").expect("`output` buffer in dispatch result");
    let actual = bytes_to_f32_vec(out_bytes);

    let diff = max_abs_diff(&expected, &actual);
    // Sequential per-thread token accumulation in both kernel and
    // reference — only fp32 rounding drift, no simd-reorder noise.
    assert!(diff < 1e-5, "aura_value int4 [{label}]: max |diff| = {diff:.2e}");
}

#[test]
fn aura_value_int4_mha_matches_naive_reference_f32() {
    // MHA: repeat_count = 1, no GQA fan-out. sparse_threshold = 0 so
    // every token contributes (covers the dense aggregation path).
    run_aura_value_case(4, 4, 8, 0.0, "mha dense");
}

#[test]
fn aura_value_int4_gqa_matches_naive_reference_f32() {
    // GQA: 8 q_heads over 2 kv_heads → repeat_count = 4. Exercises the
    // `kv_head = q_head / repeat_count` index mapping.
    run_aura_value_case(8, 2, 8, 0.0, "gqa dense");
}

#[test]
fn aura_value_int4_sparse_threshold_skips_tokens_f32() {
    // sparse_threshold = 0.1 — with the weight pattern above, every
    // token whose phase is 0, 1, or 2 (weight 0.0 / 0.04 / 0.08) is
    // below threshold and must be skipped. GQA repeat_count = 2 so the
    // skip branch and the GQA mapping are exercised together.
    run_aura_value_case(4, 2, 10, 0.1, "gqa sparse");
}

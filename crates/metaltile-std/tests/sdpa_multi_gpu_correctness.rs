//! End-to-end GPU correctness for `ffai::sdpa_multi` — the multi-query
//! SDPA kernel that attends a whole block of query rows against a
//! shared K/V cache in one dispatch.
//!
//! Validates proc-macro → IR → MSL → PSO → dispatch → readback against
//! a straight-translation CPU reference. Covers:
//!   - full mode (`causal == 0`): every query attends the full
//!     `[0, base_kv + n_query)` range
//!   - causal mode (`causal == 1`): query `r` attends
//!     `[0, base_kv + r + 1)`
//!   - a non-zero `base_kv` prefix (cached context before the block)
//!   - GQA fan-out (`n_q_heads > n_kv_heads`)
//!   - f32 / f16 / bf16
//!
//! Shapes stay small (head_dim hardcoded to 128 — the kernel's
//! requirement) so the CPU reference is instant and eyeball-able.
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_multi::ffai_sdpa_multi;

/// CPU reference: per (query, q_head), softmax(Q·Kᵀ·scale)·V over the
/// query's attended `[0, n_kv)` range. fp32 throughout.
#[allow(clippy::too_many_arguments)]
fn naive_sdpa_multi(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let gqa = n_q_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
    for r in 0..n_query {
        let n_kv = if causal { base_kv + r + 1 } else { base_kv + n_query };
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = (r * n_q_heads + qh) * head_dim;
            let kv_slab = kvh * kv_stride * head_dim;
            let mut scores = vec![0.0f32; n_kv];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_off = kv_slab + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - m).exp();
                sum += *score;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, score) in scores.iter().enumerate() {
                    acc += *score * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_multi(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dt: Dt,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let heads_per_group = n_q_heads / n_kv_heads;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_query * n_q_heads * head_dim], dt));
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("base_kv".into(), (base_kv as u32).to_le_bytes().to_vec());
    buffers.insert("n_query".into(), (n_query as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("causal".into(), (causal as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_sdpa_multi::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per (query, q_head); TPG = 1024 (the kernel's hard
    // invariant — a smaller TPG would make n_simd=0 and freeze the GPU).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads * n_query, 1, 1], [
            1024, 1, 1,
        ])
        .expect("dispatch_with_grid");
    unpack_bytes(result.outputs.get("out").expect("out buffer"), dt)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at {at} (expected {:.6}, got {:.6})",
        expected[at],
        actual[at]
    );
}

#[test]
fn sdpa_multi_full_mode_matches_cpu_f32() {
    let _g = gpu_lock();
    // No prefix, 8-query block, full (bidirectional) attention.
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 128usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_multi(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, false, scale,
    );
    let actual = run_sdpa_multi(
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        false,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_multi full f32");
}

#[test]
fn sdpa_multi_causal_mode_matches_cpu_f32() {
    let _g = gpu_lock();
    // Causal within the block — query r attends [0, r+1).
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 4usize, 128usize);
    let (base_kv, n_query) = (0usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_multi(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, true, scale,
    );
    let actual = run_sdpa_multi(
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        true,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_multi causal f32");
}

#[test]
fn sdpa_multi_with_prefix_and_gqa_matches_cpu_f32() {
    let _g = gpu_lock();
    // Non-zero cached prefix + GQA fan-out (32 q-heads over 8 kv-heads,
    // the Nemotron-Labs-Diffusion shape). Causal mode.
    let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 128usize);
    let (base_kv, n_query) = (20usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 29, 12.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);

    let expected = naive_sdpa_multi(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, true, scale,
    );
    let actual = run_sdpa_multi(
        &q,
        &k,
        &v,
        Dt::F32,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        true,
        scale,
    );
    assert_close(&actual, &expected, 1e-4, "sdpa_multi prefix+GQA causal f32");
}

#[test]
fn sdpa_multi_full_mode_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 128usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };

    let expected = naive_sdpa_multi(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        false,
        scale,
    );
    let actual = run_sdpa_multi(
        &q,
        &k,
        &v,
        Dt::F16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        false,
        scale,
    );
    assert_close(&actual, &expected, 5e-3, "sdpa_multi full f16");
}

#[test]
fn sdpa_multi_causal_mode_matches_cpu_bf16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, head_dim) = (4usize, 2usize, 128usize);
    let (base_kv, n_query) = (12usize, 8usize);
    let kv_stride = base_kv + n_query;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = ramp(n_query * n_q_heads * head_dim, 23, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };

    let expected = naive_sdpa_multi(
        &round(&q),
        &round(&k),
        &round(&v),
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        true,
        scale,
    );
    let actual = run_sdpa_multi(
        &q,
        &k,
        &v,
        Dt::Bf16,
        n_q_heads,
        n_kv_heads,
        head_dim,
        base_kv,
        n_query,
        kv_stride,
        true,
        scale,
    );
    assert_close(&actual, &expected, 2e-2, "sdpa_multi causal bf16");
}

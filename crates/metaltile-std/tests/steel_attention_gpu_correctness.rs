//! GPU correctness for `mlx::steel::attn::steel_attention` — the scalar
//! flash-prefill kernel `mt_sdpa_prefill`.
//!
//! The scalar flash variant uses BQ=32 queries per threadgroup, BK=16 keys
//! per outer block, and 128 threads (4 simdgroups × 32 lanes). It is the
//! non-MMA path (`mt_sdpa_prefill` vs `mt_sdpa_prefill_mma`) and serves
//! as the fallback for older GPUs or smaller shapes.
//!
//! ## DISPATCH INVARIANTS (mt_sdpa_prefill)
//! - `SimdGroup2D` mode: the kernel reads `tgid_x` / `tgid_y` / `tgid_z`.
//! - Grid: `[q_len / 32, n_q_heads, batch]`, TPG = 128 (4 simdgroups).
//! - `q_len` must be a multiple of 32 (`BQ`).
//! - `head_dim` is hardcoded 128 in the kernel body.
//!
//! CPU oracle: causal SDPA reference — matches `sdpa_prefill_mma_long_t.rs`.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::attn::steel_attention::mt_sdpa_prefill;

/// Naive causal SDPA for a single batch.
/// Q/K/V layout: `[n_heads, q_len, head_dim]` (contiguous).
#[allow(clippy::too_many_arguments)]
fn naive_sdpa_prefill_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(n_q_heads.is_multiple_of(n_kv_heads));
    let gqa = n_q_heads / n_kv_heads;
    let q_len_off = k_len - q_len;
    let mut out = vec![0.0f32; n_q_heads * q_len * head_dim];
    for qh in 0..n_q_heads {
        let kvh = qh / gqa;
        let q_off = qh * q_len * head_dim;
        let kv_off = kvh * k_len * head_dim;
        for qi in 0..q_len {
            let causal_lim = q_len_off + qi + 1;
            let mut scores = vec![f32::NEG_INFINITY; k_len];
            for (j, s) in scores.iter_mut().enumerate().take(causal_lim) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + qi * head_dim + d] * k[kv_off + j * head_dim + d];
                }
                *s = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> =
                scores.iter().map(|&s| if s.is_finite() { (s - m).exp() } else { 0.0 }).collect();
            let total: f32 = e.iter().sum();
            let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &ej) in e.iter().enumerate() {
                    acc += ej * inv * v[kv_off + j * head_dim + d];
                }
                out[q_off + qi * head_dim + d] = acc;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dt: Dt,
    batch: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let dt_bytes = dt.bytes();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("out".into(), vec![0u8; batch * n_q_heads * q_len * head_dim * dt_bytes]);
    buffers.insert("q_len".into(), (q_len as u32).to_le_bytes().to_vec());
    buffers.insert("k_len".into(), (k_len as u32).to_le_bytes().to_vec());
    buffers.insert("gqa_factor".into(), ((n_q_heads / n_kv_heads) as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv_heads".into(), (n_kv_heads as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_sdpa_prefill::kernel_ir_for(dt.to_dtype());
    // The kernel reads tgid_x/y/z — requires SimdGroup2D dispatch mode.
    kernel.mode = KernelMode::SimdGroup2D;
    // Grid: (q_len / BQ=32, n_q_heads, batch), TPG = 128.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [q_len / 32, n_q_heads, batch], [
            128, 1, 1,
        ])
        .expect("sdpa_prefill dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(batch * n_q_heads * q_len * head_dim);
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (e - a).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at [{max_at}] (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn steel_attention_prefill_matches_cpu_t128_f32() {
    let _g = gpu_lock();
    // Minimal shape: q_len=128=32*4 tiles, head_dim=128, gqa=1.
    let (n_q_heads, n_kv_heads, q_len, k_len, head_dim) = (4, 4, 128, 128, 128);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * q_len * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * k_len * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * k_len * head_dim, 11, 5.0);

    let expected =
        naive_sdpa_prefill_causal(&q, &k, &v, n_q_heads, n_kv_heads, q_len, k_len, head_dim, scale);
    let actual = run_sdpa_prefill(
        &q,
        &k,
        &v,
        Dt::F32,
        1,
        n_q_heads,
        n_kv_heads,
        q_len,
        k_len,
        head_dim,
        scale,
    );

    assert_close(&actual, &expected, 2e-2, "scalar prefill T=128 f32");
}

#[test]
fn steel_attention_prefill_gqa_factor2_f32() {
    let _g = gpu_lock();
    // GQA: 4 Q heads, 2 KV heads → factor=2.
    let (n_q_heads, n_kv_heads, q_len, k_len, head_dim) = (4, 2, 128, 128, 128);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * q_len * head_dim, 19, 9.0);
    let k = ramp(n_kv_heads * k_len * head_dim, 7, 3.0);
    let v = ramp(n_kv_heads * k_len * head_dim, 11, 5.0);

    let expected =
        naive_sdpa_prefill_causal(&q, &k, &v, n_q_heads, n_kv_heads, q_len, k_len, head_dim, scale);
    let actual = run_sdpa_prefill(
        &q,
        &k,
        &v,
        Dt::F32,
        1,
        n_q_heads,
        n_kv_heads,
        q_len,
        k_len,
        head_dim,
        scale,
    );

    assert_close(&actual, &expected, 2e-2, "scalar prefill GQA=2 f32");
}

#[test]
fn steel_attention_prefill_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, q_len, k_len, head_dim) = (2, 2, 64, 64, 128);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q: Vec<f32> = (1..=n_q_heads * q_len * head_dim).map(|i| i as f32 * 0.001).collect();
    let k: Vec<f32> = (1..=n_kv_heads * k_len * head_dim).map(|i| i as f32 * 0.001).collect();
    let v: Vec<f32> = (1..=n_kv_heads * k_len * head_dim).map(|i| i as f32 * 0.001).collect();
    let actual = run_sdpa_prefill(
        &q,
        &k,
        &v,
        Dt::F32,
        1,
        n_q_heads,
        n_kv_heads,
        q_len,
        k_len,
        head_dim,
        scale,
    );
    assert!(actual.iter().any(|&x| x != 0.0), "sdpa_prefill output all zeros — empty kernel?");
}

#[test]
fn steel_attention_prefill_matches_cpu_f16() {
    let _g = gpu_lock();
    let (n_q_heads, n_kv_heads, q_len, k_len, head_dim) = (4, 4, 64, 64, 128);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_f32 = ramp(n_q_heads * q_len * head_dim, 17, 8.0);
    let k_f32 = ramp(n_kv_heads * k_len * head_dim, 13, 6.0);
    let v_f32 = ramp(n_kv_heads * k_len * head_dim, 11, 5.0);
    // Round through f16 for the CPU reference.
    let q_r: Vec<f32> = q_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let k_r: Vec<f32> = k_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let v_r: Vec<f32> = v_f32.iter().map(|&v| Dt::F16.round(v)).collect();

    let expected = naive_sdpa_prefill_causal(
        &q_r, &k_r, &v_r, n_q_heads, n_kv_heads, q_len, k_len, head_dim, scale,
    );
    let actual = run_sdpa_prefill(
        &q_f32,
        &k_f32,
        &v_f32,
        Dt::F16,
        1,
        n_q_heads,
        n_kv_heads,
        q_len,
        k_len,
        head_dim,
        scale,
    );

    // f16 scalar flash has wider drift than f32.
    assert_close(&actual, &expected, 5e-2, "scalar prefill T=64 f16");
}

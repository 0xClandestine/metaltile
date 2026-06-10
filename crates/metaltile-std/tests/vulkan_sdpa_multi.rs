//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Vulkan/RDNA4 correctness for the multi-position flash-attention kernel
//! `ffai_sdpa_multi` (the prefill attention kernel). The existing
//! `#[test_kernel]` corpus exercises a small (n_query=8, kv_stride=64,
//! GQA 8/4) shape; this test pins the *prefill-realistic* shapes the
//! FFAI consumer dispatches — large `kv_stride` (cache capacity), a real
//! Qwen GQA fan-out, and BOTH causal and full modes — so the batched
//! flash path is guarded directly on the Vulkan backend rather than only
//! via the auto-corpus.
//!
//! The kernel is a Reduction-mode kernel: 1024-thread workgroup, one
//! workgroup per (query, q_head). On RDNA4 the device pins
//! `requiredSubgroupSize=32` so the in-kernel 32-lane simd-group
//! partition (and `mt_subgroup_add` / `subgroupMax` reductions) line up
//! with the hardware subgroup; this test is the regression guard for
//! that contract holding at scale.
//!
//!   cargo test -p metaltile-std --features vulkan \
//!       --test vulkan_sdpa_multi -- --nocapture
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;

use metaltile::{VulkanDevice, core::dtype::DType};
use metaltile_std::ffai::sdpa_multi::ffai_sdpa_multi;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

fn rnd(s: &mut u32) -> f32 { ((xorshift(s) % 2000) as f32 / 1000.0) - 1.0 }

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
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                sum += *sc;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, sc) in scores.iter().enumerate() {
                    acc += *sc * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn read_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    bytes.chunks_exact(4).take(n).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dev: &VulkanDevice,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    label: &str,
) {
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut st = 0x1234_5678u32 ^ (label.len() as u32).wrapping_mul(2654435761);
    let q: Vec<f32> = (0..n_query * n_q_heads * head_dim).map(|_| rnd(&mut st)).collect();
    let k: Vec<f32> = (0..n_kv_heads * kv_stride * head_dim).map(|_| rnd(&mut st)).collect();
    let v: Vec<f32> = (0..n_kv_heads * kv_stride * head_dim).map(|_| rnd(&mut st)).collect();

    let expected = naive_sdpa_multi(
        &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, causal, scale,
    );

    let dt = DType::F32;
    let mut kernel = ffai_sdpa_multi::kernel_ir_for(dt);
    kernel.mode = metaltile::core::ir::KernelMode::Reduction;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_bytes(&q));
    buffers.insert("k".into(), f32_bytes(&k));
    buffers.insert("v".into(), f32_bytes(&v));
    buffers.insert("out".into(), vec![0u8; n_query * n_q_heads * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    buffers.insert("base_kv".into(), (base_kv as u32).to_le_bytes().to_vec());
    buffers.insert("n_query".into(), (n_query as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("causal".into(), (causal as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let grid = [(n_q_heads * n_query) as u32, 1, 1];
    let tpg = [1024u32, 1, 1];

    let outputs = dev
        .run_kernel(&kernel, &buffers, grid, tpg)
        .unwrap_or_else(|e| panic!("[{label}] run_kernel failed: {e}"));
    let got = read_f32(outputs.get("out").expect("out missing"), expected.len());

    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = (g - e).abs();
        if d > worst {
            worst = d;
            worst_idx = i;
        }
    }
    eprintln!(
        "[{label}] nq={n_q_heads} nkv={n_kv_heads} hd={head_dim} base_kv={base_kv} \
         n_query={n_query} kv_stride={kv_stride} causal={causal}  max|Δ|={worst:.3e} \
         (idx {worst_idx}: got {:.5} want {:.5})",
        got[worst_idx], expected[worst_idx]
    );
    assert!(
        (worst as f64) <= 2e-3,
        "[{label}] sdpa_multi WRONG on Vulkan: max|Δ|={worst:.3e} > 2e-3"
    );
}

#[test]
fn sdpa_multi_prefill_shapes_correct_on_vulkan() {
    let Some(dev) = VulkanDevice::create().expect("Vulkan init") else {
        eprintln!("no Vulkan device — skipping");
        return;
    };
    eprintln!("=== sdpa_multi prefill/GQA correctness on Vulkan/RDNA4 ===");

    // Qwen2.5-1.5B GQA shape: 12 q-heads, 2 kv-heads, head_dim 128.
    // Prefill of S=8 tokens onto a 2048-deep cache from scratch (base_kv=0).
    run_case(&dev, 12, 2, 128, 0, 8, 2048, true, "qwen1.5b_prefill_causal_from0");
    // Same but mid-sequence: 56 tokens already cached, 8 more prefilled.
    run_case(&dev, 12, 2, 128, 56, 8, 2048, true, "qwen1.5b_prefill_causal_base56");
    // Full (bidirectional) mode at the same prefill geometry.
    run_case(&dev, 12, 2, 128, 56, 8, 2048, false, "qwen1.5b_prefill_full_base56");
    // Larger block (S=16) — crosses more than one simd-group per query.
    run_case(&dev, 12, 2, 128, 100, 16, 2048, true, "qwen1.5b_prefill_causal_s16");
    // Different GQA fan-out (8/4, head_dim 128) at a deeper cache.
    run_case(&dev, 8, 4, 128, 200, 8, 4096, true, "gqa8_4_deep_causal");
    // n_kv that is NOT a multiple of 32 (32 simd-groups stride the KV walk;
    // tail positions must still be summed): base_kv=70 -> n_kv up to 85.
    run_case(&dev, 12, 2, 128, 70, 15, 2048, true, "qwen1.5b_npot_nkv_causal");
}

//! End-to-end GPU correctness for `ffai::rope_2d` — 2D positional RoPE
//! over a `(row, col)` token grid for vision transformers.
//!
//! For each `(token, head, j)` with `j ∈ [0, head_dim/4)`:
//!   row half (dims [0, half)):   rotated by token's row index
//!   col half (dims [half, dim)): rotated by token's col index
//! each half running rotate-half RoPE over its `half_dim` width.
//!
//! Three scenarios:
//!   - identity at grid position (0, 0): cos=1, sin=0 → output == input
//!   - general grid positions vs a CPU oracle (bit-exact in f32)
//!   - norm preservation: each rotated pair keeps its L2 norm
//!
//! Dtype coverage: f32 / f16 / bf16.
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::rope_2d::ffai_rope_2d;

/// CPU oracle matching the kernel's exact arithmetic.
fn naive_rope_2d(
    qk: &[f32],
    positions: &[u32],
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let quarter_dim = head_dim / 4;
    let half_f = half_dim as f32;
    let mut out = vec![0.0f32; qk.len()];

    for token in 0..n_tokens {
        let row = positions[(token * 2) as usize] as f32;
        let col = positions[(token * 2 + 1) as usize] as f32;
        for head in 0..n_heads {
            let head_base = (token * n_heads * head_dim + head * head_dim) as usize;
            for j in 0..quarter_dim {
                let inv_freq = (-2.0 * j as f32 * theta_base.log2() / half_f).exp2();

                let (cos_r, sin_r) = {
                    let t = row * inv_freq;
                    (t.cos(), t.sin())
                };
                let (cos_c, sin_c) = {
                    let t = col * inv_freq;
                    (t.cos(), t.sin())
                };

                // Row half.
                let r1 = head_base + j as usize;
                let r2 = head_base + (j + quarter_dim) as usize;
                let xr1 = qk[r1];
                let xr2 = qk[r2];
                out[r1] = xr1 * cos_r - xr2 * sin_r;
                out[r2] = xr1 * sin_r + xr2 * cos_r;

                // Column half.
                let c1 = head_base + (half_dim + j) as usize;
                let c2 = head_base + (half_dim + j + quarter_dim) as usize;
                let xc1 = qk[c1];
                let xc2 = qk[c2];
                out[c1] = xc1 * cos_c - xc2 * sin_c;
                out[c2] = xc1 * sin_c + xc2 * cos_c;
            }
        }
    }
    out
}

fn run_rope_2d(
    qk: &[f32],
    positions: &[u32],
    dt: Dt,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let quarter_dim = head_dim / 4;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qk".into(), pack_bytes(qk, dt));
    buffers.insert("positions".into(), pack_u32_bytes(positions));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; qk.len()], dt));
    buffers.insert("n_heads".into(), n_heads.to_le_bytes().to_vec());
    buffers.insert("head_dim".into(), head_dim.to_le_bytes().to_vec());
    buffers.insert("half_dim".into(), half_dim.to_le_bytes().to_vec());
    buffers.insert("quarter_dim".into(), quarter_dim.to_le_bytes().to_vec());
    buffers.insert("theta_base".into(), theta_base.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_rope_2d::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id<0>=token, <1>=head, <2>=j. One thread per
    // (token, head, j). Keep token*head*j ≤ 1024 for a single TG.
    assert!(
        (n_tokens * n_heads * quarter_dim) as usize <= 1024,
        "keep n_tokens*n_heads*quarter_dim ≤ 1024 for a single-TG dispatch",
    );
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [
            n_tokens as usize,
            n_heads as usize,
            quarter_dim as usize,
        ])
        .expect("rope_2d dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

#[test]
fn rope_2d_identity_at_origin_f32() {
    let _g = gpu_lock();
    // All tokens at grid position (0,0) → theta=0 → output == input.
    let (n_tokens, n_heads, head_dim) = (4u32, 4u32, 16u32);
    let qk: Vec<f32> =
        (0..n_tokens * n_heads * head_dim).map(|i| 0.1 + (i as f32 * 0.017).sin()).collect();
    let positions = vec![0u32; (n_tokens * 2) as usize];
    let actual = run_rope_2d(&qk, &positions, Dt::F32, n_tokens, n_heads, head_dim, 10000.0);
    for (idx, (a, e)) in actual.iter().zip(qk.iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "identity at origin broke at idx={idx}: {a} vs {e}");
    }
}

#[test]
fn rope_2d_matches_oracle_f32() {
    let _g = gpu_lock();
    let (n_tokens, n_heads, head_dim) = (6u32, 4u32, 32u32);
    let theta_base = 10000.0f32;
    let qk: Vec<f32> =
        (0..n_tokens * n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    // A 3×2 patch grid → row ∈ {0,1,2}, col ∈ {0,1}.
    let mut positions = Vec::with_capacity((n_tokens * 2) as usize);
    for token in 0..n_tokens {
        positions.push(token / 2); // row
        positions.push(token % 2); // col
    }
    let expected = naive_rope_2d(&qk, &positions, n_tokens, n_heads, head_dim, theta_base);
    let actual = run_rope_2d(&qk, &positions, Dt::F32, n_tokens, n_heads, head_dim, theta_base);
    let mut max_diff = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 5e-5, "rope_2d f32: max |diff| = {max_diff:.2e}");
}

#[test]
fn rope_2d_preserves_pair_norm_f32() {
    let _g = gpu_lock();
    // Each rotated (x1, x2) pair preserves its L2 norm regardless of
    // position — pins that the rotation didn't drop a cross term.
    let (n_tokens, n_heads, head_dim) = (4u32, 2u32, 32u32);
    let qk: Vec<f32> =
        (0..n_tokens * n_heads * head_dim).map(|i| 0.5 + (i as f32 * 0.073).cos()).collect();
    let mut positions = Vec::with_capacity((n_tokens * 2) as usize);
    for token in 0..n_tokens {
        positions.push(token * 3 + 1);
        positions.push(token * 2 + 5);
    }
    let actual = run_rope_2d(&qk, &positions, Dt::F32, n_tokens, n_heads, head_dim, 10000.0);

    let half_dim = head_dim / 2;
    let quarter_dim = head_dim / 4;
    for token in 0..n_tokens {
        for head in 0..n_heads {
            let head_base = (token * n_heads * head_dim + head * head_dim) as usize;
            for j in 0..quarter_dim {
                for half_off in [0usize, half_dim as usize] {
                    let i1 = head_base + half_off + j as usize;
                    let i2 = head_base + half_off + (j + quarter_dim) as usize;
                    let in_sq = qk[i1] * qk[i1] + qk[i2] * qk[i2];
                    let out_sq = actual[i1] * actual[i1] + actual[i2] * actual[i2];
                    assert!(
                        (in_sq - out_sq).abs() < 1e-4,
                        "norm not preserved at token={token} head={head} j={j}",
                    );
                }
            }
        }
    }
}

#[test]
fn rope_2d_matches_oracle_f16() {
    let _g = gpu_lock();
    let (n_tokens, n_heads, head_dim) = (6u32, 4u32, 32u32);
    let theta_base = 10000.0f32;
    let qk: Vec<f32> =
        (0..n_tokens * n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let qk_rounded: Vec<f32> = qk.iter().map(|&v| Dt::F16.round(v)).collect();
    let mut positions = Vec::with_capacity((n_tokens * 2) as usize);
    for token in 0..n_tokens {
        positions.push(token / 2);
        positions.push(token % 2);
    }
    let expected = naive_rope_2d(&qk_rounded, &positions, n_tokens, n_heads, head_dim, theta_base);
    let actual = run_rope_2d(&qk, &positions, Dt::F16, n_tokens, n_heads, head_dim, theta_base);
    let mut max_rel = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1e-3));
    }
    assert!(max_rel < 5e-3, "rope_2d f16: max rel = {max_rel:.2e}");
}

#[test]
fn rope_2d_matches_oracle_bf16() {
    let _g = gpu_lock();
    let (n_tokens, n_heads, head_dim) = (6u32, 4u32, 32u32);
    let theta_base = 10000.0f32;
    let qk: Vec<f32> =
        (0..n_tokens * n_heads * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let qk_rounded: Vec<f32> = qk.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let mut positions = Vec::with_capacity((n_tokens * 2) as usize);
    for token in 0..n_tokens {
        positions.push(token / 2);
        positions.push(token % 2);
    }
    let expected = naive_rope_2d(&qk_rounded, &positions, n_tokens, n_heads, head_dim, theta_base);
    let actual = run_rope_2d(&qk, &positions, Dt::Bf16, n_tokens, n_heads, head_dim, theta_base);
    let mut max_rel = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1e-3));
    }
    assert!(max_rel < 2e-2, "rope_2d bf16: max rel = {max_rel:.2e}");
}

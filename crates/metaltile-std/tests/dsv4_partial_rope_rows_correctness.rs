//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Batched partial RoPE (ffai_dsv4_partial_rope_rows) must equal the per-token
//! reference (token t roped at position t). NO model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::dsv4_partial_rope_rows::ffai_dsv4_partial_rope_rows;

#[test]
fn partial_rope_rows_matches_per_token_oracle() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip");
        return;
    }
    drop(probe);

    let n_heads = 8usize;
    let head_dim = 512usize;
    let n_nope = 448usize;
    let half_rot = 32usize;
    let n_tokens = 40usize;
    let theta_base = 10_000.0f32;

    let qk: Vec<f32> = (0..n_tokens * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.011 - 0.4).sin() * 1.2)
        .collect();

    // Oracle: token t roped at position t (adjacent GPT-J pairs, forward).
    let mut want = qk.clone();
    for t in 0..n_tokens {
        for h in 0..n_heads {
            for p in 0..half_rot {
                let inv_freq =
                    (-(p as f32) * 2.0 * theta_base.ln() / (2.0 * half_rot as f32)).exp();
                let theta = t as f32 * inv_freq;
                let c = theta.cos();
                let s = theta.sin();
                let base = t * n_heads * head_dim + h * head_dim + n_nope + 2 * p;
                let x_lo = qk[base];
                let x_hi = qk[base + 1];
                want[base] = x_lo * c - x_hi * s;
                want[base + 1] = x_lo * s + x_hi * c;
            }
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qk".into(), pack_bytes(&qk, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&qk, Dt::F32)); // pass-through nope dims
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_nope".into(), (n_nope as u32).to_le_bytes().to_vec());
    buffers.insert("half_rot".into(), (half_rot as u32).to_le_bytes().to_vec());
    buffers.insert("n_heads".into(), (n_heads as u32).to_le_bytes().to_vec());
    buffers.insert("base_position".into(), 0u32.to_le_bytes().to_vec());
    buffers.insert("theta_base".into(), theta_base.to_le_bytes().to_vec());
    buffers.insert("inverse_flag".into(), 0u32.to_le_bytes().to_vec());
    buffers.insert("freq_scale".into(), 1.0f32.to_le_bytes().to_vec());
    buffers.insert("ext_factor".into(), 0.0f32.to_le_bytes().to_vec());
    buffers.insert("corr_low".into(), 0.0f32.to_le_bytes().to_vec());
    buffers.insert("corr_high".into(), 0.0f32.to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_dsv4_partial_rope_rows::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Grid3D;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_heads, half_rot, n_tokens], [
            1, 1, 1,
        ])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut maxd = 0.0f32;
    let mut nan = 0;
    for (a, b) in want.iter().zip(&got) {
        if !b.is_finite() {
            nan += 1;
            continue;
        }
        maxd = maxd.max((a - b).abs());
    }
    eprintln!("nan={nan} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0);
    assert!(maxd < 1e-3, "maxAbsDiff {maxd} too large — batched rope diverges from per-token");
}

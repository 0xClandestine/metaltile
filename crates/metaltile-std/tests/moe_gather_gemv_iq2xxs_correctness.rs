//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::moe_gather_gemv_iq2xxs` — the fused
//! 6-expert IQ2_XXS inline-dequant gather GEMV used by the DSv4 decode
//! FFN. Validates GPU output against a CPU reference that runs the
//! identical (production-proven) IQ2_XXS dequant formula, so a wrong
//! index/stride or a wrongly folded gather surfaces here instead of as
//! garbage decode in FFAI.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_gather_gemv_iq2xxs::ffai_moe_gather_gemv_iq2xxs;

const N_SLOTS: usize = 6;

/// Deterministic pseudo-random u32 stream (xorshift) — avoids a dev-dep
/// on `rand` and keeps the test reproducible.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn frand(state: &mut u32) -> f32 {
    // Uniform-ish in [-1, 1).
    (xorshift(state) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// CPU reference: dequant the IQ2_XXS split buffers and dot each
/// expert/row against `x`. Mirrors
/// `ffai_gguf_dequant_iq2_xxs` exactly.
fn reference(
    x: &[f32],
    qs_all: &[u32],
    d_all: &[f32],
    grid: &[u8],
    signs: &[u8],
    k_in: usize,
    m_out: usize,
) -> Vec<f32> {
    let blocks_per_row = k_in / 256;
    let nblk_per_expert = m_out * blocks_per_row;
    let mut out = vec![0.0f32; N_SLOTS * m_out];
    for slot in 0..N_SLOTS {
        for m in 0..m_out {
            let qs_row_base = (slot * nblk_per_expert + m * blocks_per_row) * 16;
            let d_row_base = slot * nblk_per_expert + m * blocks_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                for group in 0..8usize {
                    let aux_idx = qs_all[qs_row_base + b * 16 + group * 2];
                    let aux_sgn = qs_all[qs_row_base + b * 16 + group * 2 + 1];
                    let scale_4bit = aux_sgn >> 28;
                    let db = d_all[d_row_base + b] * ((scale_4bit as f32 + 0.5) * 0.25);
                    let x_grp = b * 256 + group * 32;
                    for j in 0..4usize {
                        let grid_key = ((aux_idx >> (j * 8)) & 0xff) as usize;
                        let sign_idx = ((aux_sgn >> (j * 7)) & 0x7f) as usize;
                        let sign_mask = signs[sign_idx] as u32;
                        for l in 0..8usize {
                            let octet = grid[grid_key * 8 + l] as f32;
                            let sign = if (sign_mask & (1 << l)) != 0 { -1.0 } else { 1.0 };
                            let w = db * sign * octet;
                            acc += w * x[x_grp + j * 8 + l];
                        }
                    }
                }
            }
            out[slot * m_out + m] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    x: &[f32],
    qs_all: &[u32],
    d_all: &[f32],
    grid: &[u8],
    signs: &[u8],
    dt: Dt,
    k_in: usize,
    m_out: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("qs_all".into(), pack_u32_bytes(qs_all));
    buffers.insert("d_all".into(), pack_bytes(d_all, Dt::F32));
    // Identity expert ids → slot s reads expert s (reproduces the
    // contiguous-slot reference).
    let eids: Vec<u32> = (0..N_SLOTS as u32).collect();
    buffers.insert("expert_ids".into(), pack_u32_bytes(&eids));
    buffers.insert("grid".into(), grid.to_vec());
    buffers.insert("signs".into(), signs.to_vec());
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; N_SLOTS * m_out], dt));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_moe_gather_gemv_iq2xxs::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // grid (threadgroups) = [m_out, n_slots, 1], one 32-lane simdgroup each.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, N_SLOTS, 1], [32, 1, 1])
        .expect("gather_gemv_iq2xxs dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

fn run_case(dt: Dt, k_in: usize, m_out: usize, tol: f32) {
    let _g = gpu_lock();
    let blocks_per_row = k_in / 256;
    let nblk_per_expert = m_out * blocks_per_row;

    let mut st = 0x1234_5678u32;
    let x: Vec<f32> = (0..k_in).map(|_| frand(&mut st)).collect();
    let qs_all: Vec<u32> = (0..N_SLOTS * nblk_per_expert * 16).map(|_| xorshift(&mut st)).collect();
    // Positive-ish scales so output magnitudes stay reasonable.
    let d_all: Vec<f32> =
        (0..N_SLOTS * nblk_per_expert).map(|_| frand(&mut st).abs() + 0.1).collect();
    // iq2xxs_grid octets are small magnitudes; emulate with bytes 0..47.
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();

    let want = reference(&x, &qs_all, &d_all, &grid, &signs, k_in, m_out);
    let got = run_gpu(&x, &qs_all, &d_all, &grid, &signs, dt, k_in, m_out);

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let rel = (g - w).abs() / denom;
        assert!(
            rel < tol,
            "dt={dt:?} idx={i} (slot={}, m={}): got={g} want={w} rel={rel}",
            i / m_out,
            i % m_out
        );
    }
}

#[test]
fn gather_gemv_iq2xxs_f32() {
    // k_in=512 → 2 blocks/row; k_in=4096 → 16 blocks/row (production gate/up).
    run_case(Dt::F32, 512, 4, 1e-3);
    run_case(Dt::F32, 4096, 8, 1e-3);
}

#[test]
fn gather_gemv_iq2xxs_f16() {
    run_case(Dt::F16, 512, 4, 2e-2);
    run_case(Dt::F16, 4096, 8, 3e-2);
}

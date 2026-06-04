//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::moe_gather_down_q2k` — fused 6-expert
//! Q2_K inline-dequant down-projection + router-weighted sum. Validates
//! against a CPU reference running the identical (production-proven)
//! Q2_K dequant formula.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_gather_down_q2k::ffai_moe_gather_down_q2k;
// The Q2_K output-index → (qs byte, 2-bit shift) map is the single shared
// definition in `quant::gguf`: the kernel, the quantizer, and this oracle all
// read it, so the layout can't drift apart (getting it wrong was PR #264).
use metaltile_std::quant::gguf::q2_k_qpos;

const N_SLOTS: usize = 6;

fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn frand(state: &mut u32) -> f32 { (xorshift(state) as f32 / u32::MAX as f32) * 2.0 - 1.0 }

#[allow(clippy::too_many_arguments)]
fn reference(
    inners_all: &[f32],
    qs_all: &[u32],
    scales_all: &[u8],
    d_all: &[f32],
    dmin_all: &[f32],
    weights: &[f32],
    k_in: usize,
    m_out: usize,
) -> Vec<f32> {
    let blocks_per_row = k_in / 256;
    let nblk_per_expert = m_out * blocks_per_row;
    let mut out = vec![0.0f32; m_out];
    #[allow(clippy::needless_range_loop)]
    for m in 0..m_out {
        let mut acc = 0.0f32;
        for slot in 0..N_SLOTS {
            let w_slot = weights[slot];
            let qs_row_base = (slot * nblk_per_expert + m * blocks_per_row) * 16;
            let blk_row_base = slot * nblk_per_expert + m * blocks_per_row;
            let inner_base = slot * k_in;
            for k in 0..k_in {
                let b = k / 256;
                let in_block = k % 256;
                // Canonical Q2_K layout — identical mapping to
                // `gguf_dequant_q2_k` and the kernel under test. Scale index
                // is `in_block / 16`; the qs byte + 2-bit shift come from
                // `q2_k_qpos`, NOT the naive 4-consecutive-per-byte order.
                let sub = in_block / 16;
                let (q_byte, shift) = q2_k_qpos(in_block);
                let word_idx = q_byte / 4;
                let byte_in_word = q_byte % 4;
                let word = qs_all[qs_row_base + b * 16 + word_idx];
                let qs_byte = (word >> (byte_in_word * 8)) & 0xff;
                let q_2bit = (qs_byte >> shift) & 0x3;
                let scale_byte = scales_all[qs_row_base + b * 16 + sub] as u32;
                let scale_4bit = scale_byte & 0xf;
                let min_4bit = (scale_byte >> 4) & 0xf;
                let d = d_all[blk_row_base + b];
                let dmin = dmin_all[blk_row_base + b];
                let wq = d * scale_4bit as f32 * q_2bit as f32 - dmin * min_4bit as f32;
                acc += w_slot * wq * inners_all[inner_base + k];
            }
        }
        out[m] = acc;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    inners_all: &[f32],
    qs_all: &[u32],
    scales_all: &[u8],
    d_all: &[f32],
    dmin_all: &[f32],
    weights: &[f32],
    dt: Dt,
    k_in: usize,
    m_out: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inners_all".into(), pack_bytes(inners_all, dt));
    buffers.insert("qs_all".into(), pack_u32_bytes(qs_all));
    buffers.insert("scales_all".into(), scales_all.to_vec());
    buffers.insert("d_all".into(), pack_bytes(d_all, Dt::F32));
    buffers.insert("dmin_all".into(), pack_bytes(dmin_all, Dt::F32));
    let eids: Vec<u32> = (0..N_SLOTS as u32).collect();
    buffers.insert("expert_ids".into(), pack_u32_bytes(&eids));
    buffers.insert("weights".into(), pack_bytes(weights, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; m_out], dt));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_slots".into(), (N_SLOTS as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = ffai_moe_gather_down_q2k::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, 1, 1], [32, 1, 1])
        .expect("gather_down_q2k dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

fn run_case(dt: Dt, k_in: usize, m_out: usize, tol: f32) {
    let _g = gpu_lock();
    let blocks_per_row = k_in / 256;
    let nblk_per_expert = m_out * blocks_per_row;

    let mut st = 0x9e37_79b9u32;
    let inners_all: Vec<f32> = (0..N_SLOTS * k_in).map(|_| frand(&mut st)).collect();
    let qs_all: Vec<u32> = (0..N_SLOTS * nblk_per_expert * 16).map(|_| xorshift(&mut st)).collect();
    let scales_all: Vec<u8> =
        (0..N_SLOTS * nblk_per_expert * 16).map(|_| (xorshift(&mut st) & 0xff) as u8).collect();
    let d_all: Vec<f32> =
        (0..N_SLOTS * nblk_per_expert).map(|_| frand(&mut st).abs() * 0.05 + 0.01).collect();
    let dmin_all: Vec<f32> =
        (0..N_SLOTS * nblk_per_expert).map(|_| frand(&mut st).abs() * 0.05 + 0.01).collect();
    let weights: Vec<f32> = (0..N_SLOTS).map(|_| frand(&mut st).abs() + 0.1).collect();

    let want =
        reference(&inners_all, &qs_all, &scales_all, &d_all, &dmin_all, &weights, k_in, m_out);
    let got =
        run_gpu(&inners_all, &qs_all, &scales_all, &d_all, &dmin_all, &weights, dt, k_in, m_out);

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let rel = (g - w).abs() / denom;
        assert!(rel < tol, "dt={dt:?} m={i}: got={g} want={w} rel={rel}");
    }
}

#[test]
fn gather_down_q2k_f32() {
    run_case(Dt::F32, 512, 4, 1e-3);
    run_case(Dt::F32, 2048, 8, 1e-3);
}

#[test]
fn gather_down_q2k_f16() {
    run_case(Dt::F16, 512, 4, 2e-2);
    run_case(Dt::F16, 2048, 8, 3e-2);
}

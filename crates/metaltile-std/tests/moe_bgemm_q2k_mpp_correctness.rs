//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_gather_bgemm_q2k_mpp` — prefill Q2_K
//! grouped BGEMM. Oracle: per-row Q2_K dequant gemv. Cosine ≥ 0.99.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_bgemm_q2k_mpp::ffai_moe_gather_bgemm_q2k_mpp;
// Shared Q2_K output-index → (qs byte, 2-bit shift) map (see PR #264/#265): the
// kernel, quantizer, and this oracle all read the one definition in quant::gguf.
use metaltile_std::quant::gguf::q2_k_qpos;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn bgemm_q2k_mpp_matches_gemv_oracle() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip bgemm_q2k_mpp: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 256usize;
    let n_out = 64usize;
    let t_rows = 64usize;
    let nblk = n_out * k_in / 256;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let mut st = 0x2C0F_FEE2u32;
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let scales: Vec<u8> =
        (0..n_experts * nblk * 16).map(|_| (xorshift(&mut st) & 0xff) as u8).collect();
    let d: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001).collect();
    let dmin: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let deq = |e: usize, o: usize, k: usize| -> f32 {
        let vidx = o * k_in + k;
        let block = vidx / 256;
        let in_block = vidx % 256;
        // Canonical Q2_K layout — identical to `gguf_dequant_q2_k` and the
        // kernel under test. Scale index is `in_block / 16`; the qs byte +
        // 2-bit shift come from `q2_k_qpos`, NOT the naive 4-per-byte order.
        let sub = in_block / 16;
        let (q_byte, shift) = q2_k_qpos(in_block);
        let word_idx = q_byte / 4;
        let byte_in_word = q_byte % 4;
        let word = qs[e * nblk * 16 + block * 16 + word_idx];
        let qs_byte = (word >> (byte_in_word * 8)) & 0xff;
        let q2 = (qs_byte >> shift) & 0x3;
        let scale_byte = scales[e * nblk * 16 + block * 16 + sub] as u32;
        let s4 = scale_byte & 0xf;
        let m4 = (scale_byte >> 4) & 0xf;
        d[e * nblk + block] * s4 as f32 * q2 as f32 - dmin[e * nblk + block] * m4 as f32
    };
    let mut want = vec![0.0f32; t_rows * n_out];
    for r in 0..t_rows {
        let e = indices[r] as usize;
        for o in 0..n_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(e, o, k) * x[r * k_in + k];
            }
            want[r * n_out + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("qs".into(), pack_u32_bytes(&qs));
    buffers.insert("scales".into(), scales.clone());
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_gather_bgemm_q2k_mpp::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut nan = 0;
    for (a, b) in want.iter().zip(&got) {
        if !a.is_finite() || !b.is_finite() {
            nan += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("nan={nan} cos={cos:.6}");
    assert_eq!(nan, 0);
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
}

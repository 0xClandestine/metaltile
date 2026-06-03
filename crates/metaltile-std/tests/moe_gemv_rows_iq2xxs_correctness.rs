//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_gemv_rows_iq2xxs` — the fast
//! gemv-over-rows prefill MoE kernel (replaces the slow coop-tile bgemm).
//! Oracle: per-row IQ2_XXS dequant gemv with PER-ROW x and per-row expert.
//! Same dequant as the proven gather_gemv; only x is row-indexed. Cos ≥ 0.99.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_gemv_rows_iq2xxs::ffai_moe_gemv_rows_iq2xxs;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn gemv_rows_iq2xxs_matches_oracle() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 8usize;
    let k_in = 512usize; // 2 blocks/row
    let m_out = 64usize;
    let m_total = 40usize;
    let nblk = m_out * (k_in / 256);

    // Each row picks an expert (sparse, like real routing).
    let mut st = 0x71AC_0011u32;
    let expert_ids: Vec<u32> = (0..m_total).map(|_| xorshift(&mut st) % n_experts as u32).collect();
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..m_total * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let deq = |e: usize, o: usize, k: usize| -> f32 {
        let vidx = o * k_in + k;
        let block = vidx / 256;
        let in_block = vidx % 256;
        let group = in_block / 32;
        let in_group = in_block % 32;
        let owi = in_group / 8;
        let lio = in_group % 8;
        let qb = e * nblk * 16 + block * 16 + group * 2;
        let aux_idx = qs[qb];
        let aux_sgn = qs[qb + 1];
        let s4 = aux_sgn >> 28;
        let db = d[e * nblk + block] * ((s4 as f32 + 0.5) * 0.25);
        let key = ((aux_idx >> (owi * 8)) & 0xff) as usize;
        let octet = grid[key * 8 + lio] as f32;
        let sign_mask = signs[((aux_sgn >> (owi * 7)) & 0x7f) as usize] as u32;
        let sign = if (sign_mask & (1 << lio)) != 0 { -1.0 } else { 1.0 };
        db * sign * octet
    };
    let mut want = vec![0.0f32; m_total * m_out];
    for r in 0..m_total {
        let e = expert_ids[r] as usize;
        for o in 0..m_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(e, o, k) * x[r * k_in + k];
            }
            want[r * m_out + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("qs_all".into(), pack_u32_bytes(&qs));
    buffers.insert("d_all".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("expert_ids".into(), pack_u32_bytes(&expert_ids));
    buffers.insert("grid".into(), grid.clone());
    buffers.insert("signs".into(), signs.clone());
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; m_total * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_gemv_rows_iq2xxs::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [m_out, m_total, 1], [32, 1, 1])
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
    eprintln!("want[0..6]={:?}", &want[..6]);
    eprintln!("got[0..6]={:?}", &got[..6]);
    eprintln!("nan={nan} cos={cos:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
}

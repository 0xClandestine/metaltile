//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_gather_bgemm_iq2xxs_mpp` — the
//! prefill IQ2_XXS grouped BGEMM. Oracle: per-row IQ2_XXS dequant gemv
//! (same formula as ffai_gguf_dequant_iq2_xxs). Cosine ≥ 0.99 (MMA
//! accumulation order differs from the scalar oracle).
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_bgemm_iq2xxs_mpp::ffai_moe_gather_bgemm_iq2xxs_mpp;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn bgemm_iq2xxs_mpp_matches_gemv_oracle() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    let family = probe.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip bgemm_iq2xxs_mpp: needs Apple10+ GPU (chip_family={family:?})");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 256usize; // 1 IQ2 block per row
    let n_out = 64usize; // BN=32 → 2 tiles
    let t_rows = 64usize; // BM=16 → 4 tiles
    let nblk_per_expert = n_out * k_in / 256;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let mut st = 0x1F2D_3C4Bu32;
    let qs: Vec<u32> = (0..n_experts * nblk_per_expert * 16).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> = (0..n_experts * nblk_per_expert)
        .map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001)
        .collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    // Oracle: per-row IQ2_XXS dequant gemv.
    let deq = |e: usize, o: usize, k: usize| -> f32 {
        let vidx = o * k_in + k;
        let block = vidx / 256;
        let in_block = vidx % 256;
        let group = in_block / 32;
        let in_group = in_block % 32;
        let owi = in_group / 8;
        let lio = in_group % 8;
        let qb = e * nblk_per_expert * 16 + block * 16 + group * 2;
        let aux_idx = qs[qb];
        let aux_sgn = qs[qb + 1];
        let s4 = aux_sgn >> 28;
        let db = d[e * nblk_per_expert + block] * ((s4 as f32 + 0.5) * 0.25);
        let key = ((aux_idx >> (owi * 8)) & 0xff) as usize;
        let octet = grid[key * 8 + lio] as f32;
        let sign_mask = signs[((aux_sgn >> (owi * 7)) & 0x7f) as usize] as u32;
        let sign = if (sign_mask & (1 << lio)) != 0 { -1.0 } else { 1.0 };
        db * sign * octet
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
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("grid".into(), grid.clone());
    buffers.insert("signs".into(), signs.clone());
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_gather_bgemm_iq2xxs_mpp::kernel_ir_for(Dt::F32.to_dtype());
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
    eprintln!("want[0..6]={:?}", &want[..6]);
    eprintln!("got[0..6]={:?}", &got[..6]);
    eprintln!("nan={nan} cos={cos:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
}

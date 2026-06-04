//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_gemv_ws_q2k` — the WEIGHT-STATIONARY
//! prefill MoE Q2_K gemv (down projection). It dequants each expert's weight
//! row ONCE into threadgroup memory and reuses it across the tile; the math is
//! identical to the proven `ffai_moe_gemv_rows_q2k` (same canonical Q2_K
//! dequant, same split pool, same per-row dot). Oracle = the gemv-rows kernel
//! on the SAME inputs. Exact-ish f32 agreement (cosine ≥ 0.999), NO model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_gemv_rows_q2k::ffai_moe_gemv_rows_q2k,
    moe_gemv_ws_q2k::ffai_moe_gemv_ws_q2k,
};

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn gemv_ws_q2k_matches_gemv_rows() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 256usize; // 1 block/row
    let m_out = 64usize;
    let m_total = 40usize;
    let rows_per_tile = 8usize;
    let nblk = m_out * (k_in / 256);

    // Rows pre-permuted by expert (contiguous) — the WS layout.
    let expert_ids: Vec<u32> = (0..m_total).map(|r| (r * n_experts / m_total) as u32).collect();
    let mut st = 0x9E37_79B9u32;
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let scales: Vec<u8> =
        (0..n_experts * nblk * 16).map(|_| (xorshift(&mut st) & 0xff) as u8).collect();
    let d: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001).collect();
    let dmin: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001).collect();
    let x: Vec<f32> =
        (0..m_total * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let ctx = Context::new().unwrap();

    // Reference: proven gemv-rows kernel.
    let mut bref: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    bref.insert("x".into(), pack_bytes(&x, Dt::F32));
    bref.insert("qs".into(), pack_u32_bytes(&qs));
    bref.insert("scales".into(), scales.clone());
    bref.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    bref.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    bref.insert("expert_ids".into(), pack_u32_bytes(&expert_ids));
    bref.insert("out".into(), pack_bytes(&vec![0.0f32; m_total * m_out], Dt::F32));
    bref.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    bref.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    bref.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    let mut kr = ffai_moe_gemv_rows_q2k::kernel_ir_for(Dt::F32.to_dtype());
    kr.mode = KernelMode::Reduction;
    let rr = ctx
        .dispatch_with_grid(&kr, &bref, &BTreeMap::new(), [m_out, m_total, 1], [32, 1, 1])
        .unwrap();
    let want = unpack_bytes(rr.outputs.get("out").unwrap(), Dt::F32);

    // Weight-stationary kernel.
    let mut bws: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    bws.insert("x".into(), pack_bytes(&x, Dt::F32));
    bws.insert("qs".into(), pack_u32_bytes(&qs));
    bws.insert("scales".into(), scales.clone());
    bws.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    bws.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    bws.insert("expert_ids".into(), pack_u32_bytes(&expert_ids));
    bws.insert("out".into(), pack_bytes(&vec![0.0f32; m_total * m_out], Dt::F32));
    bws.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    bws.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    bws.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    bws.insert("rows_per_tile".into(), (rows_per_tile as u32).to_le_bytes().to_vec());
    let mut kw = ffai_moe_gemv_ws_q2k::kernel_ir_for(Dt::F32.to_dtype());
    kw.mode = KernelMode::Reduction;
    let gy = m_total.div_ceil(rows_per_tile);
    let rw =
        ctx.dispatch_with_grid(&kw, &bws, &BTreeMap::new(), [m_out, gy, 1], [32, 1, 1]).unwrap();
    let got = unpack_bytes(rw.outputs.get("out").unwrap(), Dt::F32);

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut nan = 0;
    let mut maxd = 0.0f32;
    for (a, b) in want.iter().zip(&got) {
        if !a.is_finite() || !b.is_finite() {
            nan += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
        maxd = maxd.max((a - b).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("rows[0..6]={:?} ws[0..6]={:?}", &want[..6], &got[..6]);
    eprintln!("nan={nan} cos={cos:.6} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

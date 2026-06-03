//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Direct comparison: ffai_moe_bgemm_iq2xxs_bm64 vs ffai_moe_gemv_rows_iq2xxs
//! on IDENTICAL pool/x/indices. Both claim to compute gateP[row,m] =
//! W[expert(row),m,:]·x[row,:]; the prefill shows them disagreeing. This
//! reproduces it in isolation. cosine should be ~1.0.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_iq2xxs_bm64::ffai_moe_bgemm_iq2xxs_bm64,
    moe_gemv_rows_iq2xxs::ffai_moe_gemv_rows_iq2xxs,
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
fn bm64_matches_gemvrows() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip");
        return;
    }
    drop(probe);

    let n_experts = 64usize;
    let k_in = 4096usize;
    let n_out = 2048usize;
    let t_rows = 30usize;
    let nblk = n_out * k_in / 256;
    // Prefill N=5 case: M=30 rows, each a DISTINCT expert (sorted), in one
    // partial 64-tile — many size-1 sub-runs + sentinel padding to 64.
    let indices: Vec<u32> = (0..t_rows as u32).collect();
    let mut st = 0xA11CE5u32;
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let ctx = Context::new().unwrap();

    // bm64
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("x".into(), pack_bytes(&x, Dt::F32));
    b.insert("qs".into(), pack_u32_bytes(&qs));
    b.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    b.insert("grid".into(), grid.clone());
    b.insert("signs".into(), signs.clone());
    b.insert("indices".into(), pack_u32_bytes(&indices));
    b.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    b.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    b.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    let mut kb = ffai_moe_bgemm_iq2xxs_bm64::kernel_ir_for(Dt::F32.to_dtype());
    kb.mode = KernelMode::Reduction;
    let rb = ctx
        .dispatch_with_grid(&kb, &b, &BTreeMap::new(), [n_out / 64, t_rows.div_ceil(64), 1], [
            128, 1, 1,
        ])
        .unwrap();
    let got_bm = unpack_bytes(rb.outputs.get("out").unwrap(), Dt::F32);

    // gemv-rows (qs_all/d_all naming, m_out/m_total)
    let mut g: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    g.insert("x".into(), pack_bytes(&x, Dt::F32));
    g.insert("qs_all".into(), pack_u32_bytes(&qs));
    g.insert("d_all".into(), pack_bytes(&d, Dt::F32));
    g.insert("expert_ids".into(), pack_u32_bytes(&indices));
    g.insert("grid".into(), grid.clone());
    g.insert("signs".into(), signs.clone());
    g.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    g.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    g.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
    g.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    let mut kg = ffai_moe_gemv_rows_iq2xxs::kernel_ir_for(Dt::F32.to_dtype());
    kg.mode = KernelMode::Reduction;
    let rg =
        ctx.dispatch_with_grid(&kg, &g, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1]).unwrap();
    let got_gv = unpack_bytes(rg.outputs.get("out").unwrap(), Dt::F32);

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut maxd = 0.0f32;
    for (a, bb) in got_gv.iter().zip(&got_bm) {
        dot += (*a as f64) * (*bb as f64);
        na += (*a as f64).powi(2);
        nb += (*bb as f64).powi(2);
        maxd = maxd.max((a - bb).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("gemvrows[0..6]={:?}", &got_gv[..6]);
    eprintln!("bm64[0..6]   ={:?}", &got_bm[..6]);
    eprintln!("cos={cos:.6} maxAbsDiff={maxd:.4}");
    assert!(cos >= 0.999, "bm64 vs gemv-rows DISAGREE: cosine {cos:.6}");
}

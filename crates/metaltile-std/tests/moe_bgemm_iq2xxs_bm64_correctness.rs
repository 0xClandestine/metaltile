//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for the high-throughput bm64 IQ2_XXS BGEMM — must match
//! the proven 16×32 pool kernel (ffai_moe_gather_bgemm_iq2xxs_mpp) on
//! identical weights/x/indices (same dequant, only the tile geometry differs).
//! cosine ≥ 0.999. NO 86GB model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_iq2xxs_bm64::ffai_moe_bgemm_iq2xxs_bm64,
    moe_bgemm_iq2xxs_mpp::ffai_moe_gather_bgemm_iq2xxs_mpp,
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
fn bgemm_iq2xxs_bm64_matches_pool_kernel() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 64usize; // MANY small expert runs (2 rows each) — like
    let k_in = 256usize; // real prefill routing (~topK rows/expert),
    let n_out = 128usize; // many sub-runs per 64-row m-tile.
    let t_rows = 128usize;
    let nblk = n_out * k_in / 256;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let mut st = 0x13572468u32;
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let ctx = Context::new().unwrap();

    // Reference: proven 16×32 pool kernel.
    let mut pool: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    pool.insert("x".into(), pack_bytes(&x, Dt::F32));
    pool.insert("qs".into(), pack_u32_bytes(&qs));
    pool.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    pool.insert("grid".into(), grid.clone());
    pool.insert("signs".into(), signs.clone());
    pool.insert("indices".into(), pack_u32_bytes(&indices));
    pool.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    pool.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    pool.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    pool.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    let mut kp = ffai_moe_gather_bgemm_iq2xxs_mpp::kernel_ir_for(Dt::F32.to_dtype());
    kp.mode = KernelMode::Reduction;
    let rp = ctx
        .dispatch_with_grid(&kp, &pool, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let want = unpack_bytes(rp.outputs.get("out").unwrap(), Dt::F32);

    // bm64 kernel (same buffers; 128 threads, n_out/64 × M/64 grid).
    let mut b: BTreeMap<String, Vec<u8>> = pool.clone();
    b.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    let mut kb = ffai_moe_bgemm_iq2xxs_bm64::kernel_ir_for(Dt::F32.to_dtype());
    kb.mode = KernelMode::Reduction;
    let rb = ctx
        .dispatch_with_grid(&kb, &b, &BTreeMap::new(), [n_out / 64, t_rows.div_ceil(64), 1], [
            128, 1, 1,
        ])
        .unwrap();
    let got = unpack_bytes(rb.outputs.get("out").unwrap(), Dt::F32);

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut nan = 0;
    let mut maxd = 0.0f32;
    for (a, bb) in want.iter().zip(&got) {
        if !a.is_finite() || !bb.is_finite() {
            nan += 1;
            continue;
        }
        dot += (*a as f64) * (*bb as f64);
        na += (*a as f64).powi(2);
        nb += (*bb as f64).powi(2);
        maxd = maxd.max((a - bb).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("pool[0..6]={:?}", &want[..6]);
    eprintln!("bm64[0..6]={:?}", &got[..6]);
    eprintln!("nan={nan} cos={cos:.6} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

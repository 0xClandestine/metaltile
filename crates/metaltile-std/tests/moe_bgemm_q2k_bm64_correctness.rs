//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! bm64 Q2_K BGEMM must match the proven 16×32 pool kernel
//! (ffai_moe_gather_bgemm_q2k_mpp). cosine ≥ 0.999. NO 86GB model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_q2k_bm64::ffai_moe_bgemm_q2k_bm64,
    moe_bgemm_q2k_mpp::ffai_moe_gather_bgemm_q2k_mpp,
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
fn bgemm_q2k_bm64_matches_pool_kernel() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 2048usize; // production down intermediate dim
    let n_out = 4096usize; // production down hidden dim
    let t_rows = 128usize;
    let nblk = n_out * k_in / 256;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let mut st = 0x2BAD_F00Du32;
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let scales: Vec<u8> =
        (0..n_experts * nblk * 16).map(|_| (xorshift(&mut st) & 0xff) as u8).collect();
    let d: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001).collect();
    let dmin: Vec<f32> =
        (0..n_experts * nblk).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let ctx = Context::new().unwrap();
    let mut pool: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    pool.insert("x".into(), pack_bytes(&x, Dt::F32));
    pool.insert("qs".into(), pack_u32_bytes(&qs));
    pool.insert("scales".into(), scales.clone());
    pool.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    pool.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    pool.insert("indices".into(), pack_u32_bytes(&indices));
    pool.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    pool.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    pool.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    pool.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    let mut kp = ffai_moe_gather_bgemm_q2k_mpp::kernel_ir_for(Dt::F32.to_dtype());
    kp.mode = KernelMode::Reduction;
    let rp = ctx
        .dispatch_with_grid(&kp, &pool, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let want = unpack_bytes(rp.outputs.get("out").unwrap(), Dt::F32);

    let mut b = pool.clone();
    b.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    let mut kb = ffai_moe_bgemm_q2k_bm64::kernel_ir_for(Dt::F32.to_dtype());
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
    eprintln!("nan={nan} cos={cos:.6} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

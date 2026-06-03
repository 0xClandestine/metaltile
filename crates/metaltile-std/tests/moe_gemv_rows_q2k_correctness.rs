//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_gemv_rows_q2k` — proves it matches the
//! PROVEN pool bgemm `ffai_moe_gather_bgemm_q2k_mpp` on identical weights +
//! per-row x (same canonical Q2_K dequant; only the GEMM structure differs).
//! cosine ≥ 0.999. NO 86GB model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_q2k_mpp::ffai_moe_gather_bgemm_q2k_mpp,
    moe_gemv_rows_q2k::ffai_moe_gemv_rows_q2k,
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
fn gemv_rows_q2k_matches_pool_kernel() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 256usize;
    let m_out = 64usize;
    let t_rows = 64usize; // == m_total
    let nblk = m_out * (k_in / 256);

    // Identity routing (row r -> expert r/(t_rows/n_experts)) so the pool
    // kernel (indices=expert) and gemv-rows (expert_ids=expert) read the
    // same expert per row; per-row x is shared between both.
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let mut st = 0x9E37_79B9u32;
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

    // Pool kernel reference.
    let mut pool: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    pool.insert("x".into(), pack_bytes(&x, Dt::F32));
    pool.insert("qs".into(), pack_u32_bytes(&qs));
    pool.insert("scales".into(), scales.clone());
    pool.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    pool.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    pool.insert("indices".into(), pack_u32_bytes(&indices));
    pool.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * m_out], Dt::F32));
    pool.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    pool.insert("n_out".into(), (m_out as u32).to_le_bytes().to_vec());
    pool.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    let mut kp = ffai_moe_gather_bgemm_q2k_mpp::kernel_ir_for(Dt::F32.to_dtype());
    kp.mode = KernelMode::Reduction;
    let rp = ctx
        .dispatch_with_grid(&kp, &pool, &BTreeMap::new(), [m_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let want = unpack_bytes(rp.outputs.get("out").unwrap(), Dt::F32);

    // gemv-rows kernel.
    let mut gv: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    gv.insert("x".into(), pack_bytes(&x, Dt::F32));
    gv.insert("qs".into(), pack_u32_bytes(&qs));
    gv.insert("scales".into(), scales.clone());
    gv.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    gv.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    gv.insert("expert_ids".into(), pack_u32_bytes(&indices));
    gv.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * m_out], Dt::F32));
    gv.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    gv.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    gv.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    let mut kv = ffai_moe_gemv_rows_q2k::kernel_ir_for(Dt::F32.to_dtype());
    kv.mode = KernelMode::Reduction;
    let rv =
        ctx.dispatch_with_grid(&kv, &gv, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1]).unwrap();
    let got = unpack_bytes(rv.outputs.get("out").unwrap(), Dt::F32);

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
    eprintln!("pool[0..6]={:?}", &want[..6]);
    eprintln!("gemv[0..6]={:?}", &got[..6]);
    eprintln!("nan={nan} cos={cos:.6} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_bgemm_q2k_view` — the ZERO-COPY Q2_K
//! grouped BGEMM that reads raw 84-byte Q2_K blocks straight from a no-copy
//! mmap VIEW buffer. Rather than re-derive a scalar oracle for the canonical
//! Q2_K layout, this proves the view kernel produces the SAME output as the
//! PROVEN pool kernel (`ffai_moe_gather_bgemm_q2k_mpp`, the validated "Tokyo"
//! path) on identical logical weights — the view just reads the raw bytes the
//! pool would have been repacked from. cosine ≥ 0.999. NO 86 GB model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use half::f16;
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_q2k_mpp::ffai_moe_gather_bgemm_q2k_mpp,
    moe_bgemm_q2k_view::ffai_moe_bgemm_q2k_view,
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
fn bgemm_q2k_view_matches_pool_kernel() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip bgemm_q2k_view: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 256usize;
    let n_out = 64usize;
    let t_rows = 64usize;
    let nblk = n_out * k_in / 256;
    let block_bytes = 84usize; // scales[16] + qs[64] + d_f16[2] + dmin_f16[2]
    let expert_byte_stride = nblk * block_bytes;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let mut st = 0x2C0F_FEE2u32;
    let qs: Vec<u32> = (0..n_experts * nblk * 16).map(|_| xorshift(&mut st)).collect();
    let scales: Vec<u8> =
        (0..n_experts * nblk * 16).map(|_| (xorshift(&mut st) & 0xff) as u8).collect();
    // d/dmin through fp16 (the view stores raw 2-byte fp16; the pool's d_f32
    // must be the same fp16-rounded value for an exact match).
    let d_f16: Vec<f16> = (0..n_experts * nblk)
        .map(|_| f16::from_f32((xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001))
        .collect();
    let dmin_f16: Vec<f16> = (0..n_experts * nblk)
        .map(|_| f16::from_f32((xorshift(&mut st) % 1000) as f32 * 0.0003 + 0.001))
        .collect();
    let d: Vec<f32> = d_f16.iter().map(|h| h.to_f32()).collect();
    let dmin: Vec<f32> = dmin_f16.iter().map(|h| h.to_f32()).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    // ── Run the PROVEN pool kernel (split qs/scales/d/dmin arrays). ──
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
    let ctx = Context::new().unwrap();
    let mut kp = ffai_moe_gather_bgemm_q2k_mpp::kernel_ir_for(Dt::F32.to_dtype());
    kp.mode = KernelMode::Reduction;
    let rp = ctx
        .dispatch_with_grid(&kp, &pool, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let want = unpack_bytes(rp.outputs.get("out").unwrap(), Dt::F32);

    // ── Pack the SAME logical weights as raw 84-byte Q2_K blocks. ──
    // scales[0..16], qs[16..80] (LE bytes of the 16 u32), d_f16[80..82],
    // dmin_f16[82..84].
    let mut view = vec![0u8; n_experts * expert_byte_stride];
    for e in 0..n_experts {
        for b in 0..nblk {
            let base = e * expert_byte_stride + b * block_bytes;
            let sc0 = e * nblk * 16 + b * 16;
            view[base..base + 16].copy_from_slice(&scales[sc0..sc0 + 16]);
            let qb0 = e * nblk * 16 + b * 16;
            for w in 0..16 {
                let o = base + 16 + w * 4;
                view[o..o + 4].copy_from_slice(&qs[qb0 + w].to_le_bytes());
            }
            let db = d_f16[e * nblk + b].to_bits();
            view[base + 80] = (db & 0xff) as u8;
            view[base + 81] = (db >> 8) as u8;
            let mb = dmin_f16[e * nblk + b].to_bits();
            view[base + 82] = (mb & 0xff) as u8;
            view[base + 83] = (mb >> 8) as u8;
        }
    }

    let mut vb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    vb.insert("x".into(), pack_bytes(&x, Dt::F32));
    vb.insert("view_u8".into(), view);
    vb.insert("indices".into(), pack_u32_bytes(&indices));
    vb.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    vb.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    vb.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    vb.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    vb.insert("tensor_byte_off".into(), 0u32.to_le_bytes().to_vec());
    vb.insert("expert_byte_stride".into(), (expert_byte_stride as u32).to_le_bytes().to_vec());
    let mut kv = ffai_moe_bgemm_q2k_view::kernel_ir_for(Dt::F32.to_dtype());
    kv.mode = KernelMode::Reduction;
    let rv = ctx
        .dispatch_with_grid(&kv, &vb, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
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
    eprintln!("view[0..6]={:?}", &got[..6]);
    eprintln!("nan={nan} cos={cos:.6} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (view kernel diverges from pool)");
}

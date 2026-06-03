//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Batched grouped Q8 gemv (ffai_grouped_gemv_q8_rows) must equal the
//! per-token single kernel (ffai_grouped_gemv_q8) row by row. NO model load.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemv_q8::{ffai_grouped_gemv_q8, ffai_grouped_gemv_q8_rows};

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn grouped_gemv_q8_rows_matches_single() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip");
        return;
    }
    drop(probe);

    let m_out = 256usize; // n_groups*rows_per_group
    let rows_per_group = 64usize; // n_groups = 4
    let n_groups = m_out / rows_per_group;
    let k_in = 128usize;
    let bpr = k_in / 32;
    let n_tokens = 6usize;

    let mut st = 0x0DEFACE0u32;
    let qs: Vec<u32> = (0..m_out * bpr * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> = (0..m_out * bpr)
        .map(|_| ((xorshift(&mut st) % 1000) as f32 / 1000.0 - 0.5) * 0.05)
        .collect();
    let x: Vec<f32> = (0..n_tokens * n_groups * k_in)
        .map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0)
        .collect();

    let ctx = Context::new().unwrap();
    let consts = |kk: usize, mm: usize, rpg: usize| -> BTreeMap<String, Vec<u8>> {
        let mut c = BTreeMap::new();
        c.insert("k_in".into(), (kk as u32).to_le_bytes().to_vec());
        c.insert("m_out".into(), (mm as u32).to_le_bytes().to_vec());
        c.insert("rows_per_group".into(), (rpg as u32).to_le_bytes().to_vec());
        c
    };

    // Reference: single kernel per token.
    let mut want = vec![0.0f32; n_tokens * m_out];
    let mut ks = ffai_grouped_gemv_q8::kernel_ir_for(Dt::F32.to_dtype());
    ks.mode = KernelMode::Reduction;
    for t in 0..n_tokens {
        let xt = &x[t * n_groups * k_in..(t + 1) * n_groups * k_in];
        let mut b: BTreeMap<String, Vec<u8>> = consts(k_in, m_out, rows_per_group);
        b.insert("qs".into(), pack_u32_bytes(&qs));
        b.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
        b.insert("x".into(), pack_bytes(xt, Dt::F32));
        b.insert("out".into(), pack_bytes(&vec![0.0f32; m_out], Dt::F32));
        let r =
            ctx.dispatch_with_grid(&ks, &b, &BTreeMap::new(), [m_out, 1, 1], [32, 1, 1]).unwrap();
        let o = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);
        want[t * m_out..(t + 1) * m_out].copy_from_slice(&o);
    }

    // Batched kernel.
    let mut bb: BTreeMap<String, Vec<u8>> = consts(k_in, m_out, rows_per_group);
    bb.insert("qs".into(), pack_u32_bytes(&qs));
    bb.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    bb.insert("x".into(), pack_bytes(&x, Dt::F32));
    bb.insert("out".into(), pack_bytes(&vec![0.0f32; n_tokens * m_out], Dt::F32));
    let mut kr = ffai_grouped_gemv_q8_rows::kernel_ir_for(Dt::F32.to_dtype());
    kr.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&kr, &bb, &BTreeMap::new(), [m_out, n_tokens, 1], [32, 1, 1])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut maxd = 0.0f32;
    let mut nan = 0;
    for (a, b) in want.iter().zip(&got) {
        if !b.is_finite() {
            nan += 1;
            continue;
        }
        maxd = maxd.max((a - b).abs());
    }
    eprintln!("want[0..4]={:?} got[0..4]={:?}", &want[..4], &got[..4]);
    eprintln!("nan={nan} maxAbsDiff={maxd:.6}");
    assert_eq!(nan, 0);
    assert!(maxd < 1e-3, "maxAbsDiff {maxd} — batched grouped gemv diverges from single");
}

//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_grouped_gemv_q8_rows_tiled` — the
//! token-TILED grouped Q8 gemv (8-fold weight-DRAM amortization). It must
//! equal the proven `ffai_grouped_gemv_q8_rows` row-by-row: same grouped Q8
//! dequant and dot product, only the per-token weight reuse differs. NO model
//! load. (Mirrors `grouped_gemv_q8_rows_correctness.rs`, which validates
//! `_rows` against the per-token single kernel.)
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemv_q8::{ffai_grouped_gemv_q8_rows, ffai_grouped_gemv_q8_rows_tiled};

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn grouped_gemv_q8_rows_tiled_matches_rows() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let m_out = 256usize; // n_groups*rows_per_group
    let rows_per_group = 64usize; // n_groups = 4
    let n_groups = m_out / rows_per_group;
    let k_in = 128usize;
    let bpr = k_in / 32;
    // n_tokens deliberately NOT a multiple of tokens_per_tile (8) so the
    // final tile is partial — exercises the `tok < n_tokens` guard.
    let n_tokens = 19usize;

    let mut st = 0x71ED_0011u32;
    let qs: Vec<u32> = (0..m_out * bpr * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> = (0..m_out * bpr)
        .map(|_| ((xorshift(&mut st) % 1000) as f32 / 1000.0 - 0.5) * 0.05)
        .collect();
    let x: Vec<f32> = (0..n_tokens * n_groups * k_in)
        .map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0)
        .collect();

    let ctx = Context::new().unwrap();

    // Reference: the proven batched `_rows` kernel.
    let mut bref: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    bref.insert("qs".into(), pack_u32_bytes(&qs));
    bref.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    bref.insert("x".into(), pack_bytes(&x, Dt::F32));
    bref.insert("out".into(), pack_bytes(&vec![0.0f32; n_tokens * m_out], Dt::F32));
    bref.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    bref.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    bref.insert("rows_per_group".into(), (rows_per_group as u32).to_le_bytes().to_vec());
    let mut kr = ffai_grouped_gemv_q8_rows::kernel_ir_for(Dt::F32.to_dtype());
    kr.mode = KernelMode::Reduction;
    let rr = ctx
        .dispatch_with_grid(&kr, &bref, &BTreeMap::new(), [m_out, n_tokens, 1], [32, 1, 1])
        .unwrap();
    let want = unpack_bytes(rr.outputs.get("out").unwrap(), Dt::F32);

    // Tiled kernel: grid y = ceil(n_tokens / 8).
    let mut bt: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    bt.insert("qs".into(), pack_u32_bytes(&qs));
    bt.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    bt.insert("x".into(), pack_bytes(&x, Dt::F32));
    bt.insert("out".into(), pack_bytes(&vec![0.0f32; n_tokens * m_out], Dt::F32));
    bt.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    bt.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    bt.insert("rows_per_group".into(), (rows_per_group as u32).to_le_bytes().to_vec());
    bt.insert("n_tokens".into(), (n_tokens as u32).to_le_bytes().to_vec());
    let mut kt = ffai_grouped_gemv_q8_rows_tiled::kernel_ir_for(Dt::F32.to_dtype());
    kt.mode = KernelMode::Reduction;
    let gy = n_tokens.div_ceil(8);
    let rt =
        ctx.dispatch_with_grid(&kt, &bt, &BTreeMap::new(), [m_out, gy, 1], [32, 1, 1]).unwrap();
    let got = unpack_bytes(rt.outputs.get("out").unwrap(), Dt::F32);

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
    assert_eq!(nan, 0, "non-finite output");
    assert!(maxd < 1e-3, "maxAbsDiff {maxd} — tiled grouped gemv diverges from _rows");
}

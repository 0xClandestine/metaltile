//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Q2_K view-bm64 (raw 84-byte blocks) must match the validated pool q2k_bm64
//! on the SAME source. Multi-expert grouped indices + absolute zeroRows check
//! (the view-vs-pool diff alone masks bugs shared by both). macOS, no model.
#![cfg(target_os = "macos")]
mod common;
use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_q2k_bm64::ffai_moe_bgemm_q2k_bm64,
    moe_bgemm_q2k_view_u16_bm64::ffai_moe_bgemm_q2k_view_u16_bm64,
};

fn xs(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn q2k_view_u16_bm64_matches_pool() {
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    if ctx.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+");
        return;
    }
    let n_experts = 4usize;
    let pool_cap = 8usize; // pool bigger than experts
    let k_in = 2048usize;
    let n_out = 4096usize;
    let m_total = 72usize;
    let nblk = n_out * k_in / 256;

    // RAW Q2_K blocks: 84 bytes/block = scales[16] + qs[64] + d f16 + dmin f16.
    let mut st = 0x2BADF00Du32;
    let mut raw = vec![0u8; pool_cap * nblk * 84];
    for e in 0..n_experts {
        for b in 0..nblk {
            let base = (e * nblk + b) * 84;
            for j in 0..80 {
                raw[base + j] = (xs(&mut st) & 0xff) as u8;
            } // scales+qs
            let d = half::f16::from_f32(((xs(&mut st) % 1000) as f32) * 0.0003 + 0.001).to_bits();
            let dm = half::f16::from_f32(((xs(&mut st) % 1000) as f32) * 0.0003 + 0.001).to_bits();
            raw[base + 80..base + 82].copy_from_slice(&d.to_le_bytes());
            raw[base + 82..base + 84].copy_from_slice(&dm.to_le_bytes());
        }
    }
    // Deinterleave → pool: qs u32[*16], scales u8[*16], d f32, dmin f32.
    let mut qs = vec![0u32; pool_cap * nblk * 16];
    let mut scales = vec![0u8; pool_cap * nblk * 16];
    let mut d = vec![0f32; pool_cap * nblk];
    let mut dmin = vec![0f32; pool_cap * nblk];
    for e in 0..n_experts {
        for b in 0..nblk {
            let base = (e * nblk + b) * 84;
            let blk = e * nblk + b;
            for j in 0..16 {
                scales[blk * 16 + j] = raw[base + j];
            }
            for w in 0..16 {
                let o = base + 16 + w * 4;
                qs[blk * 16 + w] = u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
            }
            d[blk] = half::f16::from_le_bytes([raw[base + 80], raw[base + 81]]).to_f32();
            dmin[blk] = half::f16::from_le_bytes([raw[base + 82], raw[base + 83]]).to_f32();
        }
    }
    let x: Vec<f32> =
        (0..m_total * k_in).map(|_| ((xs(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();
    let indices: Vec<u32> = (0..m_total).map(|r| (r * n_experts / m_total) as u32).collect();
    let dt = Dt::F32;

    let run = |buffers: BTreeMap<String, Vec<u8>>, k: metaltile_core::ir::Kernel| -> Vec<f32> {
        let mut kk = k;
        kk.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(
                &kk,
                &buffers,
                &BTreeMap::new(),
                [n_out / 64, m_total.div_ceil(64), 1],
                [128, 1, 1],
            )
            .expect("dispatch");
        unpack_bytes(r.outputs.get("out").unwrap(), dt)
    };
    let mut pb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    pb.insert("x".into(), pack_bytes(&x, dt));
    pb.insert("qs".into(), pack_u32_bytes(&qs));
    pb.insert("scales".into(), scales);
    pb.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    pb.insert("dmin_f32".into(), pack_bytes(&dmin, Dt::F32));
    pb.insert("indices".into(), pack_u32_bytes(&indices));
    pb.insert("out".into(), vec![0u8; m_total * n_out * 4]);
    pb.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    pb.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    pb.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    let pool_out = run(pb, ffai_moe_bgemm_q2k_bm64::kernel_ir_for(DType::F32));

    let raw_u16: Vec<u8> = raw.clone(); // bytes; view_u16/view_f16 reinterpret
    let mut vb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    vb.insert("x".into(), pack_bytes(&x, dt));
    vb.insert("view_u16".into(), raw_u16.clone());
    vb.insert("view_f16".into(), raw_u16);
    vb.insert("indices".into(), pack_u32_bytes(&indices));
    vb.insert("out".into(), vec![0u8; m_total * n_out * 4]);
    vb.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    vb.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    vb.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    vb.insert("tensor_byte_off".into(), 0u32.to_le_bytes().to_vec());
    vb.insert("expert_byte_stride".into(), ((nblk * 84) as u32).to_le_bytes().to_vec());
    let view_out = run(vb, ffai_moe_bgemm_q2k_view_u16_bm64::kernel_ir_for(DType::F32));

    let mut worst = 0.0f32;
    let mut wi = 0;
    for i in 0..pool_out.len() {
        let dd = (pool_out[i] - view_out[i]).abs();
        if dd > worst {
            worst = dd;
            wi = i;
        }
    }
    let mag: f32 = pool_out.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
    let vz =
        (0..m_total).filter(|&r| (0..n_out).all(|c| view_out[r * n_out + c].abs() < 1e-6)).count();
    eprintln!(
        "[q2k-view] worst={worst:.4e} @i={wi} pool={:.4} view={:.4} mag={mag:.2} viewZeroRows={vz}/{m_total}",
        pool_out[wi], view_out[wi]
    );
    assert!(worst < mag * 2e-3, "q2k view-bm64 != pool: worst={worst:.4e} (mag {mag:.2})");
    assert_eq!(vz, 0, "q2k view left {vz}/{m_total} rows zero");
}

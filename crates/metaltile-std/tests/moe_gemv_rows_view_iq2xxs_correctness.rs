//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_gemv_rows_view_iq2xxs` (u8-recombine
//! reads) and `ffai_moe_gemv_rows_view_u16_iq2xxs` (aligned u16 reads) — the
//! zero-copy gemv-over-rows MoE kernels that read raw 66-byte IQ2_XXS blocks
//! straight from a no-copy view buffer. Oracle: per-row IQ2_XXS dequant gemv
//! (same dequant as the proven `moe_gemv_rows_iq2xxs`), with the super-scale
//! `d` round-tripped through fp16 (the view stores raw 2-byte fp16 `d`).
//! Cosine ≥ 0.99.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use half::f16;
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_gemv_rows_view_iq2xxs::{
    ffai_moe_gemv_rows_view_iq2xxs,
    ffai_moe_gemv_rows_view_u16_iq2xxs,
};

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

struct Case {
    k_in: usize,
    m_out: usize,
    m_total: usize,
    expert_byte_stride: usize,
    expert_ids: Vec<u32>,
    grid: Vec<u8>,
    signs: Vec<u8>,
    x: Vec<f32>,
    view: Vec<u8>,
    want: Vec<f32>,
}

fn build_case() -> Case {
    let n_experts = 8usize;
    let k_in = 512usize; // 2 blocks/row
    let m_out = 64usize;
    let m_total = 40usize;
    let nblk_per_expert = m_out * (k_in / 256);
    let block_bytes = 66usize;
    let expert_byte_stride = nblk_per_expert * block_bytes;

    let mut st = 0x71AC_0011u32;
    let expert_ids: Vec<u32> = (0..m_total).map(|_| xorshift(&mut st) % n_experts as u32).collect();
    let qs: Vec<u32> = (0..n_experts * nblk_per_expert * 16).map(|_| xorshift(&mut st)).collect();
    // d rounded THROUGH fp16 so the oracle matches the kernel's inline decode.
    let d_f16: Vec<f16> = (0..n_experts * nblk_per_expert)
        .map(|_| f16::from_f32((xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001))
        .collect();
    let d: Vec<f32> = d_f16.iter().map(|h| h.to_f32()).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..m_total * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    // Raw IQ2_XXS view: per expert, nblk blocks of 66 bytes:
    // [d_lo, d_hi, then 8 groups of (aux_idx u32 LE, aux_sgn u32 LE)].
    let mut view = vec![0u8; n_experts * expert_byte_stride];
    for e in 0..n_experts {
        for b in 0..nblk_per_expert {
            let base = e * expert_byte_stride + b * block_bytes;
            let db = d_f16[e * nblk_per_expert + b].to_bits();
            view[base] = (db & 0xff) as u8;
            view[base + 1] = (db >> 8) as u8;
            for grp in 0..8 {
                let qb = e * nblk_per_expert * 16 + b * 16 + grp * 2;
                let g0 = base + 2 + grp * 8;
                view[g0..g0 + 4].copy_from_slice(&qs[qb].to_le_bytes());
                view[g0 + 4..g0 + 8].copy_from_slice(&qs[qb + 1].to_le_bytes());
            }
        }
    }

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
    let mut want = vec![0.0f32; m_total * m_out];
    for r in 0..m_total {
        let e = expert_ids[r] as usize;
        for o in 0..m_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(e, o, k) * x[r * k_in + k];
            }
            want[r * m_out + o] = acc;
        }
    }

    Case { k_in, m_out, m_total, expert_byte_stride, expert_ids, grid, signs, x, view, want }
}

fn cosine(want: &[f32], got: &[f32]) -> (f64, usize) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut nan = 0;
    for (a, b) in want.iter().zip(got) {
        if !a.is_finite() || !b.is_finite() {
            nan += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12), nan)
}

fn skip_no_gpu() -> bool {
    let probe = Context::new().expect("Context::new");
    let skip = probe.chip_family().is_none_or(|lvl| lvl < 10);
    if skip {
        eprintln!("skip: needs Apple10+ GPU");
    }
    skip
}

#[test]
fn gemv_rows_view_iq2xxs_u8_matches_oracle() {
    let _g = gpu_lock();
    if skip_no_gpu() {
        return;
    }
    let c = build_case();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&c.x, Dt::F32));
    buffers.insert("view_u8".into(), c.view.clone());
    buffers.insert("grid".into(), c.grid.clone());
    buffers.insert("signs".into(), c.signs.clone());
    buffers.insert("expert_ids".into(), pack_u32_bytes(&c.expert_ids));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; c.m_total * c.m_out], Dt::F32));
    buffers.insert("k_in".into(), (c.k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (c.m_out as u32).to_le_bytes().to_vec());
    buffers.insert("m_total".into(), (c.m_total as u32).to_le_bytes().to_vec());
    buffers.insert("tensor_byte_off".into(), 0u32.to_le_bytes().to_vec());
    buffers
        .insert("expert_byte_stride".into(), (c.expert_byte_stride as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_gemv_rows_view_iq2xxs::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [c.m_out, c.m_total, 1], [32, 1, 1])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let (cos, nan) = cosine(&c.want, &got);
    eprintln!(
        "[u8] want[0..6]={:?} got[0..6]={:?} nan={nan} cos={cos:.6}",
        &c.want[..6],
        &got[..6]
    );
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
}

#[test]
fn gemv_rows_view_u16_iq2xxs_matches_oracle() {
    let _g = gpu_lock();
    if skip_no_gpu() {
        return;
    }
    let c = build_case();
    // The u16 variant reads the SAME raw bytes via aligned u16 loads.
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&c.x, Dt::F32));
    buffers.insert("view_u16".into(), c.view.clone());
    buffers.insert("grid".into(), c.grid.clone());
    buffers.insert("signs".into(), c.signs.clone());
    buffers.insert("expert_ids".into(), pack_u32_bytes(&c.expert_ids));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; c.m_total * c.m_out], Dt::F32));
    buffers.insert("k_in".into(), (c.k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (c.m_out as u32).to_le_bytes().to_vec());
    buffers.insert("m_total".into(), (c.m_total as u32).to_le_bytes().to_vec());
    buffers.insert("tensor_byte_off".into(), 0u32.to_le_bytes().to_vec());
    buffers
        .insert("expert_byte_stride".into(), (c.expert_byte_stride as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_gemv_rows_view_u16_iq2xxs::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [c.m_out, c.m_total, 1], [32, 1, 1])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let (cos, nan) = cosine(&c.want, &got);
    eprintln!(
        "[u16] want[0..6]={:?} got[0..6]={:?} nan={nan} cos={cos:.6}",
        &c.want[..6],
        &got[..6]
    );
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
}

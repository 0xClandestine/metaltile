//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_moe_bgemm_iq2xxs_view` — the ZERO-COPY
//! prefill IQ2_XXS grouped BGEMM that reads raw 66-byte IQ2_XXS blocks
//! straight from a no-copy mmap VIEW buffer (vs the repacked qs/d_f32 pool).
//! Same oracle as the pool kernel (per-row IQ2_XXS dequant gemv), but the
//! super-scale d round-trips through fp16 (the view stores the raw 2-byte
//! fp16 d, which the kernel decodes inline). Cosine ≥ 0.99.
//!
//! This is the memory-safe validation of the view byte math — NO 86 GB model
//! load. If this passes, the zero-copy resident-expert path is byte-correct
//! and the only remaining work is leak-safe mmap residency.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use half::f16;
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe_bgemm_iq2xxs_view::ffai_moe_bgemm_iq2xxs_view;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[test]
fn bgemm_iq2xxs_view_matches_gemv_oracle() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    let family = probe.chip_family();
    if family.is_none_or(|lvl| lvl < 10) {
        eprintln!("skip bgemm_iq2xxs_view: needs Apple10+ GPU (chip_family={family:?})");
        return;
    }
    drop(probe);

    let n_experts = 4usize;
    let k_in = 256usize; // 1 IQ2 block per row
    let n_out = 64usize; // BN=32 → 2 tiles
    let t_rows = 64usize; // BM=16 → 4 tiles
    let nblk_per_expert = n_out * k_in / 256;
    let block_bytes = 66usize; // 2-byte fp16 d + 64-byte qs (8 groups × 8 bytes)
    let expert_byte_stride = nblk_per_expert * block_bytes;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let mut st = 0x1F2D_3C4Bu32;
    let qs: Vec<u32> = (0..n_experts * nblk_per_expert * 16).map(|_| xorshift(&mut st)).collect();
    // d values rounded THROUGH fp16 so the oracle matches the kernel's inline
    // fp16 decode exactly (the view stores raw 2-byte fp16).
    let d_f16: Vec<f16> = (0..n_experts * nblk_per_expert)
        .map(|_| f16::from_f32((xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001))
        .collect();
    let d: Vec<f32> = d_f16.iter().map(|h| h.to_f32()).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    // Pack the raw IQ2_XXS view: per expert, nblk blocks of 66 bytes.
    // block: [d_lo, d_hi, then 8 groups of (aux_idx u32 LE, aux_sgn u32 LE)].
    let mut view = vec![0u8; n_experts * expert_byte_stride];
    for e in 0..n_experts {
        for b in 0..nblk_per_expert {
            let base = e * expert_byte_stride + b * block_bytes;
            let db = d_f16[e * nblk_per_expert + b].to_bits();
            view[base] = (db & 0xff) as u8;
            view[base + 1] = (db >> 8) as u8;
            for grp in 0..8 {
                let qb = e * nblk_per_expert * 16 + b * 16 + grp * 2;
                let aux_idx = qs[qb];
                let aux_sgn = qs[qb + 1];
                let g0 = base + 2 + grp * 8;
                view[g0..g0 + 4].copy_from_slice(&aux_idx.to_le_bytes());
                view[g0 + 4..g0 + 8].copy_from_slice(&aux_sgn.to_le_bytes());
            }
        }
    }

    // Oracle: per-row IQ2_XXS dequant gemv (d already fp16-rounded).
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
    let mut want = vec![0.0f32; t_rows * n_out];
    for r in 0..t_rows {
        let e = indices[r] as usize;
        for o in 0..n_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(e, o, k) * x[r * k_in + k];
            }
            want[r * n_out + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("view_u8".into(), view);
    buffers.insert("grid".into(), grid.clone());
    buffers.insert("signs".into(), signs.clone());
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("tensor_byte_off".into(), 0u32.to_le_bytes().to_vec());
    buffers.insert("expert_byte_stride".into(), (expert_byte_stride as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_bgemm_iq2xxs_view::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut nan = 0;
    for (a, b) in want.iter().zip(&got) {
        if !a.is_finite() || !b.is_finite() {
            nan += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("want[0..6]={:?}", &want[..6]);
    eprintln!("got[0..6]={:?}", &got[..6]);
    eprintln!("nan={nan} cos={cos:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99");
}

/// Production-shape variant: k_in=4096 (16 blocks/row, exercises block
/// crossing), n_out=2048, and a NONZERO tensor_byte_off (simulates a tensor
/// that does not start at byte 0 of its no-copy view window). Catches
/// block-indexing / base-offset bugs the k_in=256, offset=0 case can't.
#[test]
fn bgemm_iq2xxs_view_prod_dims_nonzero_offset() {
    let _g = gpu_lock();
    let probe = Context::new().expect("Context::new");
    if probe.chip_family().is_none_or(|lvl| lvl < 10) {
        eprintln!("skip: needs Apple10+ GPU");
        return;
    }
    drop(probe);

    let n_experts = 3usize;
    let k_in = 4096usize; // 16 IQ2 blocks per row
    let n_out = 64usize; // keep t_rows*n_out small
    let t_rows = 48usize;
    let nblk_per_expert = n_out * k_in / 256;
    let block_bytes = 66usize;
    let expert_byte_stride = nblk_per_expert * block_bytes;
    // Nonzero base: pretend the tensor starts 12,345,678 bytes into the view.
    let tensor_byte_off = 12_345_678usize;

    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
    let mut st = 0x55AA_1234u32;
    let qs: Vec<u32> = (0..n_experts * nblk_per_expert * 16).map(|_| xorshift(&mut st)).collect();
    let d_f16: Vec<f16> = (0..n_experts * nblk_per_expert)
        .map(|_| f16::from_f32((xorshift(&mut st) % 1000) as f32 * 0.0005 + 0.001))
        .collect();
    let d: Vec<f32> = d_f16.iter().map(|h| h.to_f32()).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 48) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    // view buffer = tensor_byte_off padding + n_experts blocks.
    let mut view = vec![0u8; tensor_byte_off + n_experts * expert_byte_stride];
    for e in 0..n_experts {
        for b in 0..nblk_per_expert {
            let base = tensor_byte_off + e * expert_byte_stride + b * block_bytes;
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
    let mut want = vec![0.0f32; t_rows * n_out];
    for r in 0..t_rows {
        let e = indices[r] as usize;
        for o in 0..n_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(e, o, k) * x[r * k_in + k];
            }
            want[r * n_out + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("view_u8".into(), view);
    buffers.insert("grid".into(), grid.clone());
    buffers.insert("signs".into(), signs.clone());
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * n_out], Dt::F32));
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("tensor_byte_off".into(), (tensor_byte_off as u32).to_le_bytes().to_vec());
    buffers.insert("expert_byte_stride".into(), (expert_byte_stride as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut k = ffai_moe_bgemm_iq2xxs_view::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(16), 1], [
            32, 1, 1,
        ])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut nan = 0;
    for (a, b) in want.iter().zip(&got) {
        if !a.is_finite() || !b.is_finite() {
            nan += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("prod want[0..6]={:?}", &want[..6]);
    eprintln!("prod got[0..6]={:?}", &got[..6]);
    eprintln!("prod nan={nan} cos={cos:.6}");
    assert_eq!(nan, 0, "non-finite output");
    assert!(cos >= 0.99, "cosine {cos:.6} < 0.99 (prod-dims/nonzero-offset)");
}

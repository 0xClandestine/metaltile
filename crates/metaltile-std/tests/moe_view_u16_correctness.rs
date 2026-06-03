//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Localize the u16 view-bm64 dequant bug: run the POOL bm64 (validated) and
//! the view-u16 bm64 on the SAME IQ2_XXS source (raw 66-byte blocks →
//! deinterleaved split pool) with single-expert indices. They must match.
//! In-model the view path gives argmax 79872 (wrong) vs the pool's Tokyo;
//! this isolates whether it's the raw u16 read / d-decode / indexing.
//! macOS-gated.
#![cfg(target_os = "macos")]

mod common;
use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_iq2xxs_bm64::ffai_moe_bgemm_iq2xxs_bm64,
    moe_bgemm_iq2xxs_view_u16_bm64::ffai_moe_bgemm_iq2xxs_view_u16_bm64,
};

struct Lcg(u64);
impl Lcg {
    fn u16v(&mut self) -> u16 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as u16
    }
}

#[test]
fn view_u16_bm64_matches_pool_bm64() {
    let _g = gpu_lock();
    // In-model condition (VIEW_RAGGED=1): M ragged (72 = 64+8, NOT a mult of 64),
    // indices grouped across multiple experts (permuted-by-expert layout), and a
    // pool sized LARGER than the experts actually filled (poolCap > n_experts) so
    // some slots are uninitialized. The original case (M=256, single expert,
    // n_experts==poolCap) passes — this exercises the ragged/multi-expert path.
    let ragged = std::env::var("VIEW_RAGGED").is_ok();
    // VIEW_TINYRUNS: mimic in-model — M=72 across MANY experts (~1-2 rows each
    // = lots of tiny runs in a 64-row tile), oversized pool. This is what the
    // in-model prefill actually feeds (47 experts / 72 rows).
    let tiny = std::env::var("VIEW_TINYRUNS").is_ok();
    let k_in = 4096usize;
    let n_out = 2048usize;
    let m_total = if ragged || tiny { 72usize } else { 256usize };
    let n_experts = if tiny { 47usize } else { 4usize };
    let pool_cap = if ragged || tiny { 64usize } else { n_experts };
    let nblk = n_out * k_in / 256;
    // RAW blocks: per expert, nblk blocks of 33 u16 (1 d-f16-bits + 32 qs).
    let mut rng = Lcg(0xBEEF);
    let constant = std::env::var("VIEW_CONST").is_ok();
    // Buffers sized to pool_cap experts (tail slots beyond n_experts left zero —
    // mimics the in-model oversized resident pool). Real data only in 0..n_experts.
    let raw_u16: Vec<u16> = (0..pool_cap * nblk * 33)
        .map(|i| {
            let e = i / (nblk * 33);
            if e >= n_experts {
                return 0u16;
            }
            if i % 33 == 0 {
                half::f16::from_f32(if constant {
                    1.0
                } else {
                    ((rng.u16v() % 200) as f32 - 100.0) * 0.001
                })
                .to_bits()
            } else if constant {
                0u16
            } else {
                rng.u16v()
            }
        })
        .collect();
    // Deinterleave → split pool: qs u32[pool_cap*nblk*16], d f32[pool_cap*nblk].
    let mut qs = vec![0u32; pool_cap * nblk * 16];
    let mut d = vec![0f32; pool_cap * nblk];
    for e in 0..n_experts {
        for b in 0..nblk {
            let base = (e * nblk + b) * 33;
            d[e * nblk + b] = half::f16::from_bits(raw_u16[base]).to_f32();
            for i in 0..16 {
                // 32 qs u16 → 16 u32
                qs[(e * nblk + b) * 16 + i] = raw_u16[base + 1 + 2 * i] as u32
                    | ((raw_u16[base + 1 + 2 * i + 1] as u32) << 16);
            }
        }
    }
    let x: Vec<f32> = (0..m_total * k_in).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 5) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|i| (i * 13 % 256) as u8).collect();
    let dt = Dt::F32;
    // indices = GLOBAL expert id per row (== slot, pool is compacted). Ragged:
    // grouped runs across n_experts (permuted-by-expert), so the last m-tile
    // (rows 64..71) is a partial run — the exact in-model shape.
    // group(r) = run id (0..n_experts), rows grouped by run (permuted-by-expert).
    // VIEW_SCRAMBLE (default for tiny/ragged): map each run to a NON-MONOTONIC
    // slot via a multiplicative permutation — mimics in-model rawG.slotOf, which
    // assigns slots in routing order (not sorted). All slots stay in [0,n_experts)
    // so they're all filled. This is the unreplicated in-model condition.
    let scramble = std::env::var("VIEW_NOSCRAMBLE").is_err() && (tiny || ragged);
    let indices: Vec<u32> = if std::env::var("VIEW_INMODEL").is_ok() {
        // EXACT in-model L0 gate slotGIdx (72 rows, 47 experts, runs up to len 4).
        [
            16, 9, 9, 40, 40, 4, 21, 15, 15, 5, 0, 0, 42, 11, 1, 1, 29, 41, 41, 18, 14, 14, 23, 7,
            7, 17, 17, 39, 39, 39, 39, 43, 27, 31, 12, 10, 10, 24, 24, 13, 13, 19, 19, 19, 8, 35,
            28, 44, 25, 25, 26, 6, 20, 33, 33, 3, 3, 3, 36, 30, 30, 38, 45, 45, 32, 2, 2, 22, 37,
            37, 34, 46,
        ]
        .iter()
        .map(|&x| x as u32)
        .collect()
    } else if let Ok(s) = std::env::var("VIEW_SINGLE_E") {
        let e: u32 = s.parse().unwrap();
        vec![e; m_total] // all rows → ONE expert (isolate per-expert offset)
    } else {
        (0..m_total)
            .map(|r| {
                let g = r * n_experts / m_total;
                let slot = if scramble { (g * 7) % n_experts } else { g };
                slot as u32
            })
            .collect()
    };

    // POOL bm64
    let ctx = Context::new().unwrap();
    let run =
        |buffers: BTreeMap<String, Vec<u8>>, kernel_ir: metaltile_core::ir::Kernel| -> Vec<f32> {
            let mut k = kernel_ir;
            k.mode = KernelMode::Reduction;
            let r = ctx
                .dispatch_with_grid(
                    &k,
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
    pb.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    pb.insert("grid".into(), grid.clone());
    pb.insert("signs".into(), signs.clone());
    pb.insert("indices".into(), pack_u32_bytes(&indices));
    pb.insert("out".into(), vec![0u8; m_total * n_out * 4]);
    pb.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    pb.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    pb.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    let pool_out = run(pb, ffai_moe_bgemm_iq2xxs_bm64::kernel_ir_for(DType::F32));

    // VIEW-u16 bm64
    let raw_bytes: Vec<u8> = raw_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut vb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    vb.insert("x".into(), pack_bytes(&x, dt));
    vb.insert("view_u16".into(), raw_bytes.clone());
    vb.insert("view_f16".into(), raw_bytes);
    vb.insert("grid".into(), grid);
    vb.insert("signs".into(), signs);
    vb.insert("indices".into(), pack_u32_bytes(&indices));
    vb.insert("out".into(), vec![0u8; m_total * n_out * 4]);
    vb.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    vb.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    vb.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    vb.insert("tensor_byte_off".into(), 0u32.to_le_bytes().to_vec());
    vb.insert("expert_byte_stride".into(), ((nblk * 66) as u32).to_le_bytes().to_vec());
    let view_out = run(vb, ffai_moe_bgemm_iq2xxs_view_u16_bm64::kernel_ir_for(DType::F32));

    let mut worst = 0.0f32;
    let mut wi = 0;
    for i in 0..pool_out.len() {
        let dd = (pool_out[i] - view_out[i]).abs();
        if dd > worst {
            worst = dd;
            wi = i;
        }
    }
    eprintln!(
        "[view-u16] worst |pool-view|={worst:.4e} @i={wi} pool={:.4} view={:.4}; pool[0..4]={:?} view[0..4]={:?}",
        pool_out[wi],
        view_out[wi],
        &pool_out[0..4],
        &view_out[0..4]
    );
    // ABSOLUTE row-coverage check: the view-vs-pool diff masks bugs SHARED by
    // both kernels (e.g. a run-detection that drops rows). Count nonzero rows
    // in EACH independently — both must cover all M rows.
    let rowzero = |out: &[f32]| -> usize {
        (0..m_total).filter(|&r| (0..n_out).all(|c| out[r * n_out + c].abs() < 1e-6)).count()
    };
    let pz = rowzero(&pool_out);
    let vz = rowzero(&view_out);
    eprintln!("[view-u16] zeroRows pool={pz}/{m_total} view={vz}/{m_total}");
    let mag: f32 = pool_out.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
    assert!(
        worst < mag * 1e-3,
        "view-u16 bm64 diverges from pool bm64: worst={worst:.4e} (mag {mag:.2})"
    );
    assert_eq!(
        vz, 0,
        "view-u16 bm64 left {vz}/{m_total} rows ZERO (run-detection drops rows — masked by view==pool)"
    );
    assert_eq!(pz, 0, "pool bm64 left {pz}/{m_total} rows ZERO");
}

//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Reproduction: bm64 amortized MoE GEMM vs gemv-rows (known-correct) on
//! RAGGED expert segments that straddle 64-row tile boundaries.
//!
//! The bm64 bench (`moe_bgemm_iq2xxs_bm64`) feeds `indices = zeros` → ALL
//! rows share expert 0 → the segment-straddle path (sub_offset/sub_end
//! probing inside a 64-row tile) is NEVER exercised. In-model the routed
//! rows are ragged (variable run-length per expert, boundaries not
//! 64-aligned) and bm64 diverges from gemv-rows on provably-identical
//! inputs. This test reproduces that WITHOUT the 86GB model: same pool,
//! same ragged indices, per-row diff to localize the bug.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    moe_bgemm_iq2xxs_bm64::ffai_moe_bgemm_iq2xxs_bm64,
    moe_gemv_rows_iq2xxs::ffai_moe_gemv_rows_iq2xxs,
};

fn read_u32(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Replay the EXACT in-model dumped bytes (/tmp/bm64dump from FFAI
/// FFAI_DBG_DUMP=1). If bm64 diverges from gemv-rows here → data-triggered
/// kernel bug. If it matches → the in-model divergence is OUTSIDE the kernel
/// (buffer aliasing / cmd-buffer hazard / stale pool).
#[test]
#[ignore] // needs /tmp/bm64dump; run explicitly with --ignored
fn bm64_replay_real_dump() {
    let _g = gpu_lock();
    let dir = "/tmp/bm64dump";
    let meta = std::fs::read_to_string(format!("{dir}/meta.txt"))
        .expect("dump meta — run FFAI with FFAI_DBG_DUMP=1");
    eprintln!("[replay] {meta}");
    // M=30 hidden=4096 intermediate=2048 cap=64 nblk=32768
    let g = |k: &str| {
        meta.split_whitespace()
            .find_map(|t| t.strip_prefix(k).map(|v| v.parse::<usize>().unwrap()))
            .unwrap()
    };
    let m_total = g("M=");
    let k_in = g("hidden=");
    let n_out = g("intermediate=");

    let qs = read_u32(&format!("{dir}/qs.u32"));
    let d = read_f32(&format!("{dir}/d.f32"));
    let grid = std::fs::read(format!("{dir}/grid.u8")).unwrap();
    let signs = std::fs::read(format!("{dir}/signs.u8")).unwrap();
    let idx = read_u32(&format!("{dir}/idx.u32"));
    let x = read_f32(&format!("{dir}/x.f32"));
    eprintln!("[replay] qs={} d={} idx={:?}", qs.len(), d.len(), idx);

    for dt in [Dt::F32, Dt::F16] {
        let bm = run_bm64(&x, &qs, &d, &grid, &signs, &idx, m_total, n_out, k_in, dt);
        let gv = run_gemv(&x, &qs, &d, &grid, &signs, &idx, m_total, n_out, k_in, dt);
        let mut worst = 0.0f32;
        let mut wr = 0;
        let mut bad = 0;
        for r in 0..m_total {
            let mut rm = 0.0f32;
            for c in 0..n_out {
                let dd = (bm[r * n_out + c] - gv[r * n_out + c]).abs();
                if dd > rm {
                    rm = dd;
                }
            }
            if rm > 1e-2 {
                bad += 1;
            }
            if rm > worst {
                worst = rm;
                wr = r;
            }
        }
        let sab: f32 = bm.iter().map(|v| v.abs()).sum();
        let sag: f32 = gv.iter().map(|v| v.abs()).sum();
        eprintln!(
            "[replay] dt={dt:?} bad={bad}/{m_total} worst={worst:.4e} @row={wr} slot={}  sumAbs bm={sab:.1} gv={sag:.1}",
            idx[wr]
        );
        // Compare metaltile-runtime bm64 output vs the FFAI-runtime bm64 output
        // on the SAME bytes. If replay-bm == in-model-gemv but != in-model-bm,
        // the FFAI runtime wrongly executes the identical kernel.
        if let (Ok(imb), Ok(img)) = (
            std::fs::read(format!("{dir}/gpBm_inmodel.f32")),
            std::fs::read(format!("{dir}/gpGv_inmodel.f32")),
        ) {
            let imb: Vec<f32> =
                imb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let img: Vec<f32> =
                img.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let sab_im: f32 = imb.iter().map(|v| v.abs()).sum();
            let sag_im: f32 = img.iter().map(|v| v.abs()).sum();
            let dbm: f32 = bm.iter().zip(&imb).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            let dgm: f32 = bm.iter().zip(&img).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            eprintln!(
                "[replay]   in-model: bm={sab_im:.1} gv={sag_im:.1} | replayBM vs inModelBM worst={dbm:.3} | replayBM vs inModelGV worst={dgm:.3}"
            );
        }
    }
}

/// Deterministic LCG so the test is reproducible without rand.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        // small symmetric values so dequant*x stays in a sane range
        ((self.next_u32() % 2000) as f32 - 1000.0) * 0.001
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bm64(
    x: &[f32],
    qs: &[u32],
    d: &[f32],
    grid: &[u8],
    signs: &[u8],
    idx: &[u32],
    m_total: usize,
    n_out: usize,
    k_in: usize,
    dt: Dt,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("qs".into(), pack_u32_bytes(qs));
    buffers.insert("d_f32".into(), pack_bytes(d, Dt::F32));
    buffers.insert("grid".into(), grid.to_vec());
    buffers.insert("signs".into(), signs.to_vec());
    buffers.insert("indices".into(), pack_u32_bytes(idx));
    buffers.insert("out".into(), vec![0u8; m_total * n_out * dt.bytes()]);
    // #[constexpr] params bind as constant BUFFERS (LE u32), not fn_consts.
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = ffai_moe_bgemm_iq2xxs_bm64::kernel_ir_for(dt.to_dtype());
    k.mode = KernelMode::Reduction;
    let gx = n_out / 64;
    let gy = m_total.div_ceil(64);
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [gx, gy, 1], [128, 1, 1])
        .expect("bm64 dispatch");
    unpack_bytes(r.outputs.get("out").unwrap(), dt)
}

#[allow(clippy::too_many_arguments)]
fn run_gemv(
    x: &[f32],
    qs: &[u32],
    d: &[f32],
    grid: &[u8],
    signs: &[u8],
    idx: &[u32],
    m_total: usize,
    n_out: usize,
    k_in: usize,
    dt: Dt,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("qs_all".into(), pack_u32_bytes(qs));
    buffers.insert("d_all".into(), pack_bytes(d, Dt::F32));
    buffers.insert("expert_ids".into(), pack_u32_bytes(idx));
    buffers.insert("grid".into(), grid.to_vec());
    buffers.insert("signs".into(), signs.to_vec());
    buffers.insert("out".into(), vec![0u8; m_total * n_out * dt.bytes()]);
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = ffai_moe_gemv_rows_iq2xxs::kernel_ir_for(dt.to_dtype());
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, m_total, 1], [32, 1, 1])
        .expect("gemv dispatch");
    unpack_bytes(r.outputs.get("out").unwrap(), dt)
}

/// Build a ragged expert assignment: runs of variable length crossing the
/// 64-row tile boundary. Returns indices[m_total].
fn ragged_indices(m_total: usize, n_experts: u32, seed: u64) -> Vec<u32> {
    let mut rng = Lcg(seed);
    let mut idx = Vec::with_capacity(m_total);
    let mut e = 0u32;
    while idx.len() < m_total {
        let run = 1 + (rng.next_u32() % 7) as usize; // runs 1..7 → straddle 64
        for _ in 0..run {
            if idx.len() >= m_total {
                break;
            }
            idx.push(e);
        }
        e = (e + 1) % n_experts;
    }
    idx
}

#[test]
fn bm64_matches_gemv_rows_on_ragged_segments_f32() { run_case(Dt::F32); }

#[test]
fn bm64_matches_gemv_rows_on_ragged_segments_f16() { run_case(Dt::F16); }

fn run_case(dtype: Dt) {
    let _g = gpu_lock();
    let n_experts = 6u32;
    let k_in = 4096usize; // 16 blocks/row — PRODUCTION gate/up dim
    let n_out = 2048usize; // 32 n-tiles of 64 — PRODUCTION intermediate
    let m_total = 200usize; // 4 m-tiles, ragged straddles
    let nblk = n_out * k_in / 256;

    let mut rng = Lcg(0xDEADBEEF);
    let qs: Vec<u32> = (0..n_experts as usize * nblk * 16).map(|_| rng.next_u32()).collect();
    let d: Vec<f32> = (0..n_experts as usize * nblk)
        .map(|_| (rng.next_u32() % 100) as f32 * 0.01 + 0.01)
        .collect();
    let grid: Vec<u8> = (0..2048).map(|i| ((i * 7) % 5) as u8).collect();
    let signs: Vec<u8> = (0..128).map(|_| (rng.next_u32() & 0xff) as u8).collect();
    let x: Vec<f32> = (0..m_total * k_in).map(|_| rng.next_f32()).collect();
    let idx = ragged_indices(m_total, n_experts, 0x1234);

    // Log the segment structure so we can correlate divergence with runs.
    let mut runs: Vec<(u32, usize, usize)> = vec![]; // (expert, start, len)
    let mut s = 0;
    while s < m_total {
        let e = idx[s];
        let mut t = s;
        while t < m_total && idx[t] == e {
            t += 1;
        }
        runs.push((e, s, t - s));
        s = t;
    }
    eprintln!("[ragged] m_total={m_total} runs={runs:?}");
    let tile_straddles: Vec<_> =
        runs.iter().filter(|(_, st, ln)| st / 64 != (st + ln - 1) / 64).collect();
    eprintln!("[ragged] runs straddling a 64-row tile boundary: {tile_straddles:?}");

    let bm = run_bm64(&x, &qs, &d, &grid, &signs, &idx, m_total, n_out, k_in, dtype);
    let gv = run_gemv(&x, &qs, &d, &grid, &signs, &idx, m_total, n_out, k_in, dtype);
    // RELATIVE tolerance: bm64 (f16-staged MMA, f32 accum) vs gemv-rows
    // (f32 dot) differ only by rounding; the values reach ±300 so an
    // absolute tol flags pure rounding. Scale by the max magnitude.
    let mag: f32 = gv.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
    let tol = mag
        * match dtype {
            Dt::F32 => 1e-4,
            _ => 4e-3,
        };
    let sab: f32 = bm.iter().map(|v| v.abs()).sum();
    let sag: f32 = gv.iter().map(|v| v.abs()).sum();
    eprintln!("[case] dtype={dtype:?} mag={mag:.1} tol={tol:.3e} sumAbs bm={sab:.2} gv={sag:.2}");
    assert!(sag > 1e-3, "gemv output is all-zero — test is vacuous (dispatch not computing)");

    // Per-row diagnostics.
    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    let mut bad_rows = 0usize;
    for row in 0..m_total {
        let mut row_max = 0.0f32;
        for c in 0..n_out {
            let dlt = (bm[row * n_out + c] - gv[row * n_out + c]).abs();
            if dlt > row_max {
                row_max = dlt;
            }
        }
        let in_tile = row / 64;
        let pos_in_tile = row % 64;
        if row_max > tol {
            bad_rows += 1;
            eprintln!(
                "[diff] row={row:>3} expert={} tile={in_tile} pos={pos_in_tile:>2} row_max={row_max:.4e}  bm[0]={:.4} gv[0]={:.4}",
                idx[row],
                bm[row * n_out],
                gv[row * n_out]
            );
        }
        if row_max > worst {
            worst = row_max;
            worst_row = row;
        }
    }
    eprintln!(
        "[summary] bad_rows={bad_rows}/{m_total} worst={worst:.4e} @row={worst_row} (expert={}, tile={}, pos={})",
        idx[worst_row],
        worst_row / 64,
        worst_row % 64
    );

    assert!(
        worst < tol,
        "bm64 diverges from gemv-rows ({dtype:?}) on ragged segments: worst |diff|={worst:.4e} @row={worst_row}"
    );
}

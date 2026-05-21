//! GPU correctness for `ffai::moe_mpp_bm64::mt_moe_gather_qmm_mma_int4_bm64_mpp`.
//!
//! BM=BN=64 MPP MoE kernel — same output semantics as the BM=16 sibling but
//! scaled up to a 64×64 output tile with 4 SGs (WM=WN=2) per TG. Validated
//! against the scalar `mt_moe_gather_qmm_int4` oracle on a clean-tile shape
//! (n_experts=4, T=64, N=64, K=64, group_size=32) — one TG covers the
//! whole 64×64 output exactly.
//!
//! Requires macOS 26+ / Metal 4 for the MPP header to be available. On
//! older toolchains the kernel falls through to a zero-write stub and this
//! test fails loudly — that's the intended signal.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::mt_moe_gather_qmm_int4;
use metaltile_std::ffai::moe_mpp_bm64;

/// Pack a row of int4 weights into uint32s (8 per uint, LSB-first per
/// nibble). Identical to the helper used by the bm16_mpp test —
/// duplicated to keep this test file self-contained.
fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    assert!(weights.len() % 8 == 0);
    weights
        .chunks_exact(8)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xf) << (i * 4);
            }
            packed
        })
        .collect()
}

/// Clean-tile correctness: BM=64 MPP MoE kernel matches the scalar m1
/// reference at n_experts=4, T=64, N=64, K=64, group_size=32. Cosine
/// ≥ 0.999.
///
/// "Clean tile" = all dims are multiples of BM=64 × BN=64 × BK=32 — no
/// per-row mask edge cases, no K-remainder. Exactly one TG covers the
/// 64×64 output. Same shape the bm16_mpp variant validates on (with one
/// fewer TG since it only covers 16×32 per TG).
#[test]
fn moe_gather_qmm_mma_int4_bm64_mpp_matches_m1_clean_tile() {
    let _g = gpu_lock();
    let n_experts = 4usize;
    let k_in = 64usize; // multiple of 32 (= BK) and 8 (pack size)
    let n_out = 64usize; // BN=64 → 1 n-tile
    let group_size = 32usize;
    let t_rows = 64usize; // BM=64 → 1 m-tile

    // Per-row expert indices, sorted: rows 0..16 → e0, 16..32 → e1, etc.
    // This is the post-permute layout the MoE pipeline produces. Forces
    // multi-sub-run dispatch within the single 64-row TG (4 sub-runs of
    // 16 rows each).
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    // m1 reference uses expert_offsets (first-row-of-each-expert), not the
    // per-row indices — build it from the sorted indices.
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    // ── Reference: scalar m1 ─────────────────────────────────────────────
    let y_m1 = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // ── Under test: BM=64 MPP MoE kernel ─────────────────────────────────
    let y_mpp = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = moe_mpp_bm64::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        // Grid: [ceil(N/64), ceil(T/64), 1]. TG: 128 lanes = 4 SGs (WM=WN=2).
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out.div_ceil(64), t_rows.div_ceil(64), 1],
                [128, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    // MPP cooperative-tensor accumulator vs scalar reduction — fp
    // accumulation order differs, so cosine is the right metric (same
    // criterion the bm16_mpp variant uses).
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut nan_count = 0usize;
    for (a, b) in y_m1.iter().zip(&y_mpp) {
        if !a.is_finite() || !b.is_finite() {
            nan_count += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("y_m1[0..8]  = {:?}", &y_m1[..8]);
    eprintln!("y_mpp[0..8] = {:?}", &y_mpp[..8]);
    eprintln!("nan_count   = {nan_count} / {}", t_rows * n_out);
    assert_eq!(nan_count, 0, "MPP BM=64 kernel produced non-finite values");
    assert!(cos >= 0.999, "MPP MoE BM=64 vs m1 cosine = {cos:.6} (want ≥ 0.999)");
}

/// Larger shape with multiple TGs and uneven sub-run distribution, closer
/// to a production tile chunk. n_experts=8, T=128 (= 2 m-tiles), N=128
/// (= 2 n-tiles), K=128, group_size=64 (production Qwen3.6 default).
#[test]
fn moe_gather_qmm_mma_int4_bm64_mpp_matches_m1_multi_tile() {
    let _g = gpu_lock();
    let n_experts = 8usize;
    let k_in = 128usize;
    let n_out = 128usize;
    let group_size = 64usize;
    let t_rows = 128usize;

    // Sorted-per-expert layout — 16 rows per expert. With BM=64, each TG
    // covers 4 experts → 4 sub-runs per TG.
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 11 + 5) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.07).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.11).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.017).sin()).collect();

    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
        *off = indices
            .iter()
            .position(|&e| e as usize >= e_idx)
            .map(|p| p as u32)
            .unwrap_or(t_rows as u32);
    }
    expert_offsets[n_experts] = t_rows as u32;

    let y_m1 = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    let y_mpp = {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
        buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
        buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
        buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
        buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = moe_mpp_bm64::kernel_ir_for(Dt::F32.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(
                &k,
                &buffers,
                &BTreeMap::new(),
                [n_out.div_ceil(64), t_rows.div_ceil(64), 1],
                [128, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    };

    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut nan_count = 0usize;
    for (a, b) in y_m1.iter().zip(&y_mpp) {
        if !a.is_finite() || !b.is_finite() {
            nan_count += 1;
            continue;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    eprintln!("multi-tile y_m1[0..8]  = {:?}", &y_m1[..8]);
    eprintln!("multi-tile y_mpp[0..8] = {:?}", &y_mpp[..8]);
    eprintln!("multi-tile nan_count   = {nan_count} / {}", t_rows * n_out);
    assert_eq!(nan_count, 0, "MPP BM=64 kernel produced non-finite values (multi-tile)");
    assert!(cos >= 0.999, "MPP MoE BM=64 vs m1 cosine = {cos:.6} (want ≥ 0.999) (multi-tile)");
}

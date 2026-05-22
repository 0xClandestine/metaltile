#![allow(clippy::manual_is_multiple_of)]

//! GPU correctness for the bit-width-generalized MMA MoE BGEMMs
//! `ffai::moe::mt_moe_gather_qmm_mma_b{3,5,6,8}`.
//!
//! Same tiled-MMA algorithm as `mt_moe_gather_qmm_mma_int4`, but the
//! weight coop-dequant pulls codes from a contiguous LSB-first bit-stream
//! so any bit-width works. Each variant is validated against a naive CPU
//! gather-matmul oracle over the bit-stream-packed weights — cosine ≥ 0.999.
//!
//! Shape: n_experts=4, T=64, N=64, K=64, group_size=32 — every dim a
//! multiple of the BM=BN=BK=32 tile, and `K*bits % 32 == 0` for all four
//! bit-widths so the bit-stream rows are word-aligned.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::{
    mt_moe_gather_qmm_mma_b3,
    mt_moe_gather_qmm_mma_b5,
    mt_moe_gather_qmm_mma_b6,
    mt_moe_gather_qmm_mma_b8,
};

/// Pack one weight row's `k_in` codes into a contiguous LSB-first
/// `bits`-wide bit-stream of `k_in*bits/32` uint32 words.
fn pack_bitstream_row(codes: &[u32], bits: u32) -> Vec<u32> {
    let nwords = codes.len() * bits as usize / 32;
    let mut w = vec![0u32; nwords];
    for (c, &code) in codes.iter().enumerate() {
        let bo = c * bits as usize;
        for bi in 0..bits as usize {
            if (code >> bi) & 1 == 1 {
                let abs = bo + bi;
                w[abs / 32] |= 1u32 << (abs % 32);
            }
        }
    }
    w
}

/// `out[t,n] = Σ_k (scale[g]·code + bias[g]) · x[t,k]`, expert = indices[t],
/// g = k / group_size.
#[allow(clippy::too_many_arguments)]
fn cpu_oracle(
    codes: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    indices: &[u32],
    n_experts: usize,
    t_rows: usize,
    n_out: usize,
    k_in: usize,
    group_size: usize,
) -> Vec<f32> {
    let groups = k_in / group_size;
    let mut out = vec![0.0f32; t_rows * n_out];
    for t in 0..t_rows {
        let e = indices[t] as usize;
        for n in 0..n_out {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                let code = codes[(e * n_out + n) * k_in + k] as f32;
                let g = (e * n_out + n) * groups + k / group_size;
                acc += (scales[g] * code + biases[g]) * x[t * k_in + k];
            }
            out[t * n_out + n] = acc;
        }
    }
    let _ = n_experts;
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn run_bitwidth(
    bits: u32,
    kernel_ir: fn(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel,
) {
    let _g = gpu_lock();
    let n_experts = 4usize;
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;
    let t_rows = 64usize;
    let max_code = (1u32 << bits) - 1;

    // Sorted-by-expert per-row indices (the post-permute MoE layout).
    let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();

    let total = n_experts * n_out * k_in;
    let codes: Vec<u32> = (0..total)
        .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_shr(13) & max_code)
        .collect();
    let packed: Vec<u32> =
        codes.chunks_exact(k_in).flat_map(|row| pack_bitstream_row(row, bits)).collect();

    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let expected = cpu_oracle(
        &codes, &scales, &biases, &x, &indices, n_experts, t_rows, n_out, k_in, group_size,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("w".into(), packed.iter().flat_map(|w| w.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
    buffers.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new");
    let mut k = kernel_ir(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    // BM=BN=32 → grid [N/32, ceil(T/32), 1], TG = 4 simdgroups (128 lanes).
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_out / 32, t_rows.div_ceil(32), 1], [
            128, 1, 1,
        ])
        .expect("dispatch");
    let actual = unpack_bytes(r.outputs.get("out").expect("out"), Dt::F32);

    let cos = cosine(&expected, &actual);
    eprintln!(
        "[gather_qmm_mma b{bits}] cos={cos:.6}  exp[0..4]={:?} got[0..4]={:?}",
        &expected[..4],
        &actual[..4]
    );
    assert!(cos >= 0.999, "gather_qmm_mma b{bits} vs CPU oracle cosine = {cos:.6} (want ≥ 0.999)");
}

#[test]
fn moe_gather_qmm_mma_b3_matches_cpu_oracle() {
    run_bitwidth(3, mt_moe_gather_qmm_mma_b3::kernel_ir_for);
}

#[test]
fn moe_gather_qmm_mma_b5_matches_cpu_oracle() {
    run_bitwidth(5, mt_moe_gather_qmm_mma_b5::kernel_ir_for);
}

#[test]
fn moe_gather_qmm_mma_b6_matches_cpu_oracle() {
    run_bitwidth(6, mt_moe_gather_qmm_mma_b6::kernel_ir_for);
}

#[test]
fn moe_gather_qmm_mma_b8_matches_cpu_oracle() {
    run_bitwidth(8, mt_moe_gather_qmm_mma_b8::kernel_ir_for);
}

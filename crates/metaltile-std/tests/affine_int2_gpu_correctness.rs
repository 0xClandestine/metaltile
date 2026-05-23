//! GPU correctness for the int2 affine quantize / dequantize kernels
//! (`mlx::quantized::mt_affine_quantize_int2` / `mt_affine_dequantize_int2`).
//!
//! int2 packs 16 two-bit codes cleanly into one uint32, so both kernels
//! follow the power-of-2 int4 / int8 template. The dequant kernel is a
//! Grid3D one-thread-per-pack expansion; the quantize kernel is a
//! Reduction-mode one-threadgroup-per-group min/max + parallel pack.
//!
//! Two checks:
//!   1. `dequantize_int2` vs an explicit CPU bit-unpack reference.
//!   2. quantize → dequantize round-trip — every input value should
//!      land within one quantization step (`scale`) of its original.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]
#![allow(clippy::needless_range_loop)]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::{mt_affine_dequantize_int2, mt_affine_quantize_int2};

const GROUP_SIZE: usize = 64;
const PACK_FACTOR: usize = 16; // 16 two-bit codes per uint32

// ── 1. dequantize vs explicit CPU bit-unpack ──────────────────────────────

#[test]
fn dequantize_int2_matches_cpu_unpack_f32() {
    let _g = gpu_lock();
    let n_groups = 5usize;
    let n_elem = n_groups * GROUP_SIZE;
    let n_packs = n_elem / PACK_FACTOR;

    // Deterministic packed weights — arbitrary bit patterns.
    let w: Vec<u32> =
        (0..n_packs).map(|i| (i as u32).wrapping_mul(0x9e37_79b9) ^ 0x5bd1_e995).collect();
    let scales: Vec<f32> = (0..n_groups).map(|g| 0.05 + g as f32 * 0.01).collect();
    let biases: Vec<f32> = (0..n_groups).map(|g| -0.2 + g as f32 * 0.03).collect();

    // CPU reference: q = (val >> (k*2)) & 0x3, then scale*q + bias.
    let mut expected = vec![0.0f32; n_elem];
    for (pack_idx, &val) in w.iter().enumerate() {
        let oindex = pack_idx * PACK_FACTOR;
        let g = oindex / GROUP_SIZE;
        for k in 0..PACK_FACTOR {
            let q = ((val >> (k * 2)) & 0x3) as f32;
            expected[oindex + k] = scales[g] * q + biases[g];
        }
    }

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("w".into(), pack_u32_bytes(&w));
    b.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    b.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0; n_elem], Dt::F32));
    b.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_affine_dequantize_int2::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [n_packs.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("dequantize_int2 dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(n_elem);

    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-6, "elem {i}: got {a}, want {e}");
    }
}

// ── 2. quantize → dequantize round-trip ───────────────────────────────────

#[test]
fn quantize_then_dequantize_int2_round_trips_f32() {
    let _g = gpu_lock();
    let n_groups = 4usize;
    let n_elem = n_groups * GROUP_SIZE;
    let n_packs = n_elem / PACK_FACTOR;

    // Input values spanning a moderate range so quantization is exercised.
    let w_in: Vec<f32> = (0..n_elem).map(|i| ((i % 29) as f32 - 14.0) * 0.07).collect();

    // ── quantize ──
    let mut qb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    qb.insert("w".into(), pack_bytes(&w_in, Dt::F32));
    qb.insert("out".into(), vec![0u8; n_packs * 4]);
    qb.insert("scales".into(), pack_bytes(&vec![0.0; n_groups], Dt::F32));
    qb.insert("biases".into(), pack_bytes(&vec![0.0; n_groups], Dt::F32));
    qb.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut qkernel = mt_affine_quantize_int2::kernel_ir_for(Dt::F32.to_dtype());
    qkernel.mode = KernelMode::Reduction;
    // Reduction mode: one threadgroup per group, 32 threads (one simdgroup).
    let qres = ctx
        .dispatch_with_grid(&qkernel, &qb, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int2 dispatch");

    let packed = unpack_u32_bytes(qres.outputs.get("out").expect("out"));
    let scales = unpack_bytes(qres.outputs.get("scales").expect("scales"), Dt::F32);
    let biases = unpack_bytes(qres.outputs.get("biases").expect("biases"), Dt::F32);
    assert_eq!(packed.len(), n_packs);

    // ── dequantize the packed weights back ──
    let mut db: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    db.insert("w".into(), pack_u32_bytes(&packed));
    db.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    db.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    db.insert("out".into(), pack_bytes(&vec![0.0; n_elem], Dt::F32));
    db.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let mut dkernel = mt_affine_dequantize_int2::kernel_ir_for(Dt::F32.to_dtype());
    dkernel.mode = KernelMode::Grid3D;
    let dres = ctx
        .dispatch_with_grid(&dkernel, &db, &BTreeMap::new(), [n_packs.div_ceil(64), 1, 1], [
            64, 1, 1,
        ])
        .expect("dequantize_int2 dispatch");
    let mut recon = unpack_bytes(dres.outputs.get("out").expect("out"), Dt::F32);
    recon.truncate(n_elem);

    // Round-trip error is bounded by half a quantization step per group.
    // The kernel rounds (`+ 0.5` then floor), so the worst case is `scale`.
    for g in 0..n_groups {
        let step = scales[g];
        for k in 0..GROUP_SIZE {
            let i = g * GROUP_SIZE + k;
            let err = (recon[i] - w_in[i]).abs();
            assert!(
                err <= step + 1e-5,
                "group {g} elem {k}: |{} - {}| = {err} > step {step}",
                recon[i],
                w_in[i],
            );
        }
    }
}

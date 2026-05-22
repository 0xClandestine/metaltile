//! GPU correctness for the int3/int5/int6 affine quantize kernels
//! (`mlx::quantized::mt_affine_quantize_int{3,5,6}`).
//!
//! These kernels implement the quantize (float → bit-stream) direction
//! for non-power-of-2 bit widths. The bit-stream output format matches
//! the existing dequantize kernels, so the primary check is a
//! quantize → dequantize round-trip: every value should land within one
//! quantization step of its original.
//!
//! ## DISPATCH INVARIANTS (mt_affine_quantize_int{3,5,6})
//! - Reduction mode (simd_min / simd_max). TPG = 32 (one simdgroup).
//! - Grid: [n_groups, 1, 1].
//! - group_size = 32.
//! - Output buffer size (uint32 words per group): int3=3, int5=5, int6=6.
//!
//! ## DISPATCH INVARIANTS (mt_affine_dequantize_int{3,5,6} — paired)
//! - Grid3D mode. TPG = 16.
//! - int3/int5: 8 values per pack → n_packs = n_elem / 8.
//! - int6: 4 values per pack → n_packs = n_elem / 4.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]
#![allow(clippy::needless_range_loop)]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::{
    mt_affine_dequantize_int3,
    mt_affine_dequantize_int5,
    mt_affine_dequantize_int6,
    mt_affine_quantize_int3,
    mt_affine_quantize_int5,
    mt_affine_quantize_int6,
};

const GROUP_SIZE: usize = 32;

// ── int3 round-trip ──────────────────────────────────────────────────────

#[test]
fn quantize_then_dequantize_int3_round_trips_f32() {
    let _g = gpu_lock();
    let n_groups = 4usize;
    let n_elem = n_groups * GROUP_SIZE;
    // int3 with group_size=32: 3 uint32 words per group (12 bytes).
    let n_out_words = n_groups * 3;

    let inp: Vec<f32> = (0..n_elem).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    // ── quantize ──
    let mut qb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    qb.insert("w".into(), pack_bytes(&inp, Dt::F32));
    qb.insert("out".into(), vec![0u8; n_out_words * 4]);
    qb.insert("scales".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    qb.insert("biases".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    qb.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut qk = mt_affine_quantize_int3::kernel_ir_for(Dt::F32.to_dtype());
    qk.mode = KernelMode::Reduction;
    let qres = ctx
        .dispatch_with_grid(&qk, &qb, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int3 dispatch");

    let packed = unpack_u32_bytes(qres.outputs.get("out").expect("out"));
    let scales = unpack_bytes(qres.outputs.get("scales").expect("scales"), Dt::F32);
    let biases = unpack_bytes(qres.outputs.get("biases").expect("biases"), Dt::F32);

    // ── dequantize back ──
    // int3: 8 values per pack → n_packs = n_elem / 8.
    let n_packs = n_elem / 8;
    let mut db: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    db.insert("w".into(), pack_u32_bytes(&packed));
    db.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    db.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    db.insert("out".into(), pack_bytes(&vec![0.0f32; n_elem], Dt::F32));
    db.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let mut dk = mt_affine_dequantize_int3::kernel_ir_for(Dt::F32.to_dtype());
    dk.mode = KernelMode::Grid3D;
    let dres = ctx
        .dispatch_with_grid(&dk, &db, &BTreeMap::new(), [n_packs.div_ceil(16), 1, 1], [16, 1, 1])
        .expect("dequantize_int3 dispatch");

    let mut recon = unpack_bytes(dres.outputs.get("out").expect("out"), Dt::F32);
    recon.truncate(n_elem);

    // Round-trip error bounded by one quantization step per group.
    for g in 0..n_groups {
        let step = scales[g];
        for k in 0..GROUP_SIZE {
            let i = g * GROUP_SIZE + k;
            let err = (recon[i] - inp[i]).abs();
            assert!(
                err <= step + 1e-5,
                "int3 group {g} elem {k}: |{:.6} - {:.6}| = {err:.6} > step {step:.6}",
                recon[i],
                inp[i],
            );
        }
    }
}

#[test]
fn quantize_int3_output_not_all_zeros() {
    let _g = gpu_lock();
    let n_groups = 2usize;
    let n_elem = n_groups * GROUP_SIZE;
    let n_out_words = n_groups * 3;
    let inp: Vec<f32> = (1..=n_elem).map(|i| i as f32 * 0.01).collect();

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("w".into(), pack_bytes(&inp, Dt::F32));
    b.insert("out".into(), vec![0u8; n_out_words * 4]);
    b.insert("scales".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    b.insert("biases".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    b.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut k = mt_affine_quantize_int3::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let res = ctx
        .dispatch_with_grid(&k, &b, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int3 dispatch");

    let packed = unpack_u32_bytes(res.outputs.get("out").expect("out"));
    assert!(packed.iter().any(|&v| v != 0), "int3 quantize output all zeros — empty kernel?");
}

// ── int5 round-trip ──────────────────────────────────────────────────────

#[test]
fn quantize_then_dequantize_int5_round_trips_f32() {
    let _g = gpu_lock();
    let n_groups = 4usize;
    let n_elem = n_groups * GROUP_SIZE;
    // int5 with group_size=32: 5 uint32 words per group (20 bytes).
    let n_out_words = n_groups * 5;

    let inp: Vec<f32> = (0..n_elem).map(|i| ((i % 23) as f32 - 11.0) * 0.04).collect();

    // ── quantize ──
    let mut qb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    qb.insert("w".into(), pack_bytes(&inp, Dt::F32));
    qb.insert("out".into(), vec![0u8; n_out_words * 4]);
    qb.insert("scales".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    qb.insert("biases".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    qb.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut qk = mt_affine_quantize_int5::kernel_ir_for(Dt::F32.to_dtype());
    qk.mode = KernelMode::Reduction;
    let qres = ctx
        .dispatch_with_grid(&qk, &qb, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int5 dispatch");

    let packed = unpack_u32_bytes(qres.outputs.get("out").expect("out"));
    let scales = unpack_bytes(qres.outputs.get("scales").expect("scales"), Dt::F32);
    let biases = unpack_bytes(qres.outputs.get("biases").expect("biases"), Dt::F32);

    // ── dequantize back ──
    // int5: 8 values per pack → n_packs = n_elem / 8.
    let n_packs = n_elem / 8;
    let mut db: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    db.insert("w".into(), pack_u32_bytes(&packed));
    db.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    db.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    db.insert("out".into(), pack_bytes(&vec![0.0f32; n_elem], Dt::F32));
    db.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let mut dk = mt_affine_dequantize_int5::kernel_ir_for(Dt::F32.to_dtype());
    dk.mode = KernelMode::Grid3D;
    let dres = ctx
        .dispatch_with_grid(&dk, &db, &BTreeMap::new(), [n_packs.div_ceil(16), 1, 1], [16, 1, 1])
        .expect("dequantize_int5 dispatch");

    let mut recon = unpack_bytes(dres.outputs.get("out").expect("out"), Dt::F32);
    recon.truncate(n_elem);

    for g in 0..n_groups {
        let step = scales[g];
        for k in 0..GROUP_SIZE {
            let i = g * GROUP_SIZE + k;
            let err = (recon[i] - inp[i]).abs();
            assert!(
                err <= step + 1e-5,
                "int5 group {g} elem {k}: |{:.6} - {:.6}| = {err:.6} > step {step:.6}",
                recon[i],
                inp[i],
            );
        }
    }
}

#[test]
fn quantize_int5_output_not_all_zeros() {
    let _g = gpu_lock();
    let n_groups = 2usize;
    let n_elem = n_groups * GROUP_SIZE;
    let n_out_words = n_groups * 5;
    let inp: Vec<f32> = (1..=n_elem).map(|i| i as f32 * 0.01).collect();

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("w".into(), pack_bytes(&inp, Dt::F32));
    b.insert("out".into(), vec![0u8; n_out_words * 4]);
    b.insert("scales".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    b.insert("biases".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    b.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut k = mt_affine_quantize_int5::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let res = ctx
        .dispatch_with_grid(&k, &b, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int5 dispatch");

    let packed = unpack_u32_bytes(res.outputs.get("out").expect("out"));
    assert!(packed.iter().any(|&v| v != 0), "int5 quantize output all zeros — empty kernel?");
}

// ── int6 round-trip ──────────────────────────────────────────────────────

#[test]
fn quantize_then_dequantize_int6_round_trips_f32() {
    let _g = gpu_lock();
    let n_groups = 4usize;
    let n_elem = n_groups * GROUP_SIZE;
    // int6 with group_size=32: 6 uint32 words per group (24 bytes).
    let n_out_words = n_groups * 6;

    let inp: Vec<f32> = (0..n_elem).map(|i| ((i % 29) as f32 - 14.0) * 0.03).collect();

    // ── quantize ──
    let mut qb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    qb.insert("w".into(), pack_bytes(&inp, Dt::F32));
    qb.insert("out".into(), vec![0u8; n_out_words * 4]);
    qb.insert("scales".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    qb.insert("biases".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    qb.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut qk = mt_affine_quantize_int6::kernel_ir_for(Dt::F32.to_dtype());
    qk.mode = KernelMode::Reduction;
    let qres = ctx
        .dispatch_with_grid(&qk, &qb, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int6 dispatch");

    let packed = unpack_u32_bytes(qres.outputs.get("out").expect("out"));
    let scales = unpack_bytes(qres.outputs.get("scales").expect("scales"), Dt::F32);
    let biases = unpack_bytes(qres.outputs.get("biases").expect("biases"), Dt::F32);

    // ── dequantize back ──
    // int6: 4 values per pack → n_packs = n_elem / 4.
    let n_packs = n_elem / 4;
    let mut db: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    db.insert("w".into(), pack_u32_bytes(&packed));
    db.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    db.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    db.insert("out".into(), pack_bytes(&vec![0.0f32; n_elem], Dt::F32));
    db.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let mut dk = mt_affine_dequantize_int6::kernel_ir_for(Dt::F32.to_dtype());
    dk.mode = KernelMode::Grid3D;
    let dres = ctx
        .dispatch_with_grid(&dk, &db, &BTreeMap::new(), [n_packs.div_ceil(16), 1, 1], [16, 1, 1])
        .expect("dequantize_int6 dispatch");

    let mut recon = unpack_bytes(dres.outputs.get("out").expect("out"), Dt::F32);
    recon.truncate(n_elem);

    for g in 0..n_groups {
        let step = scales[g];
        for k in 0..GROUP_SIZE {
            let i = g * GROUP_SIZE + k;
            let err = (recon[i] - inp[i]).abs();
            assert!(
                err <= step + 1e-5,
                "int6 group {g} elem {k}: |{:.6} - {:.6}| = {err:.6} > step {step:.6}",
                recon[i],
                inp[i],
            );
        }
    }
}

#[test]
fn quantize_int6_output_not_all_zeros() {
    let _g = gpu_lock();
    let n_groups = 2usize;
    let n_elem = n_groups * GROUP_SIZE;
    let n_out_words = n_groups * 6;
    let inp: Vec<f32> = (1..=n_elem).map(|i| i as f32 * 0.01).collect();

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("w".into(), pack_bytes(&inp, Dt::F32));
    b.insert("out".into(), vec![0u8; n_out_words * 4]);
    b.insert("scales".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    b.insert("biases".into(), pack_bytes(&vec![0.0f32; n_groups], Dt::F32));
    b.insert("group_size".into(), (GROUP_SIZE as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut k = mt_affine_quantize_int6::kernel_ir_for(Dt::F32.to_dtype());
    k.mode = KernelMode::Reduction;
    let res = ctx
        .dispatch_with_grid(&k, &b, &BTreeMap::new(), [n_groups, 1, 1], [32, 1, 1])
        .expect("quantize_int6 dispatch");

    let packed = unpack_u32_bytes(res.outputs.get("out").expect("out"));
    assert!(packed.iter().any(|&v| v != 0), "int6 quantize output all zeros — empty kernel?");
}

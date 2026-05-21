//! GPU correctness for `quantize_kv_int4/int8` + `bulk_dequant_kv_int4/int8`
//! via raw → quantize → dequant round-trip.
//!
//! These four kernels ship in `ffai::kv_cache` but had no end-to-end
//! coverage before this file. They're how `AffineQuantizedKVCache`
//! shrinks per-token K/V slots 4× (int4) or 2× (int8) at decode time —
//! a wrong index formula in either direction would silently corrupt
//! the cache without crashing.
//!
//! Coverage rationale (mirrors `kv_cache_update_gpu_correctness.rs`):
//! `quantize_kv_*` and `bulk_dequant_kv_*` are emitted from `macro_rules!`
//! shells (the proc-macro doesn't expand the inner declarative macro,
//! so embedding kernel bodies in nested macros would silently produce
//! empty kernels). The round-trip pins both the quantize geometry
//! (group-min/max scan, scale derivation, pack) and the dequant
//! geometry (group lookup, unpack, dequantize).
//!
//! Matrix:
//!   - f32 / f16 / bf16 source dtype
//!   - int4 (4-bit, vals_per_pack=8) and int8 (8-bit, vals_per_pack=4)
//!   - Qwen-realistic shape: n_kv_heads=8, head_dim=128, group_size=32
//!
//! Each test:
//!   1. Build random [n_kv_heads, head_dim] source slot (centred so
//!      group ranges are well-distributed)
//!   2. Dispatch `quantize_kv_*` → cache buffers at `position`
//!   3. Dispatch `bulk_dequant_kv_*` reading the whole [0..n_positions)
//!      slice → reconstructed values
//!   4. For the slot we just wrote, compare reconstructed vs source
//!      with a per-bits relative tolerance (int4: ±range/15 = one
//!      quantization step + per-dtype roundoff; int8: ±range/255).
//!
//! macOS-gated. Serial GPU lock (shared common::gpu_lock).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::kv_cache::{
    bulk_dequant_kv_int4,
    bulk_dequant_kv_int8,
    quantize_kv_int4,
    quantize_kv_int8,
};

/// Shape parameters covering Qwen3-class K/V slots.
struct Shape {
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    group_size: usize,
    position: usize,
    n_positions: usize, // window the bulk_dequant covers
}

impl Shape {
    fn qwen_decode() -> Self {
        Self {
            n_kv_heads: 8,
            head_dim: 128,
            max_seq: 64,
            group_size: 32,
            position: 7,
            n_positions: 16,
        }
    }
}

fn build_source(shape: &Shape, dt: Dt, seed: u64) -> Vec<f32> {
    // Deterministic, lightly noisy values with non-trivial per-group
    // range so the affine quant has signal to compress.
    let mut s = seed;
    let n = shape.n_kv_heads * shape.head_dim;
    (0..n)
        .map(|i| {
            // xorshift-ish noise → [-1, 1]
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let raw = ((s as i64 % 20_000) as f32) / 10_000.0;
            // Per-group mean varies (sinusoidal in position) so the
            // affine quant can't collapse groups to identical scales.
            let group_offset = ((i / shape.group_size) as f32 * 0.7).sin();
            dt.round(raw + group_offset)
        })
        .collect()
}

fn quantize_dispatch_grid(shape: &Shape, _bits: u32) -> ([usize; 3], [usize; 3]) {
    // One thread per group; we dispatch a single threadgroup with
    // `total_groups` threads (small enough that 1×TG covers it for
    // Qwen-decode shapes: 8 heads × 4 groups/head = 32 threads).
    let total_groups = shape.n_kv_heads * (shape.head_dim / shape.group_size);
    ([1, 1, 1], [total_groups, 1, 1])
}

fn dequant_dispatch_grid(shape: &Shape) -> ([usize; 3], [usize; 3]) {
    // One thread per output element. Fits in a single TG (8×16×128=16384,
    // larger than 1024) so split across multiple TGs of 256.
    let total = shape.n_kv_heads * shape.n_positions * shape.head_dim;
    let tpg = 256usize;
    let groups = total.div_ceil(tpg);
    ([groups, 1, 1], [tpg, 1, 1])
}

/// Run the int4 round-trip and return reconstructed values aligned to
/// [n_kv_heads, n_positions, head_dim].
fn roundtrip_int4(shape: &Shape, dt: Dt, source: &[f32]) -> Vec<f32> {
    let dtype = dt.to_dtype();
    let bits = 4u32;
    let vals_per_pack = 32u32 / bits;
    let groups_per_head = shape.head_dim / shape.group_size;

    let n_packed_per_slot = shape.head_dim / vals_per_pack as usize;
    let n_groups_per_slot = groups_per_head;

    // Cache buffers sized for the WHOLE [n_kv_heads, max_seq, ...] not
    // just one slot — kernel writes at `position`, dequant reads
    // [0..n_positions).
    let w_total = shape.n_kv_heads * shape.max_seq * n_packed_per_slot;
    let s_total = shape.n_kv_heads * shape.max_seq * n_groups_per_slot;

    let ctx = Context::new().expect("Context::new on macOS");

    // ── Quantize ────────────────────────────────────────────────────
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(source, dt));
    buffers.insert("out_w".into(), pack_u32_bytes(&vec![0u32; w_total]));
    buffers.insert("out_s".into(), pack_bytes(&vec![0.0f32; s_total], dt));
    buffers.insert("out_b".into(), pack_bytes(&vec![0.0f32; s_total], dt));
    buffers.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (shape.position as u32).to_le_bytes().to_vec());

    let mut qkernel = quantize_kv_int4::kernel_ir_for(dtype);
    qkernel.mode = KernelMode::Grid3D;
    let (grid, tpg) = quantize_dispatch_grid(shape, bits);
    let q_out = ctx
        .dispatch_with_grid(&qkernel, &buffers, &BTreeMap::new(), grid, tpg)
        .expect("quantize_kv_int4 dispatch");

    let w_bytes = q_out.outputs.get("out_w").expect("out_w buffer").clone();
    let s_bytes = q_out.outputs.get("out_s").expect("out_s buffer").clone();
    let b_bytes = q_out.outputs.get("out_b").expect("out_b buffer").clone();

    // ── Dequantize ──────────────────────────────────────────────────
    let recon_total = shape.n_kv_heads * shape.max_seq * shape.head_dim;
    let mut dbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    dbuf.insert("in_w".into(), w_bytes);
    dbuf.insert("in_s".into(), s_bytes);
    dbuf.insert("in_b".into(), b_bytes);
    dbuf.insert("out".into(), pack_bytes(&vec![0.0f32; recon_total], dt));
    dbuf.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    dbuf.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    dbuf.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    dbuf.insert("n_positions".into(), (shape.n_positions as u32).to_le_bytes().to_vec());

    let mut dkernel = bulk_dequant_kv_int4::kernel_ir_for(dtype);
    dkernel.mode = KernelMode::Grid3D;
    let (dgrid, dtpg) = dequant_dispatch_grid(shape);
    let d_out = ctx
        .dispatch_with_grid(&dkernel, &dbuf, &BTreeMap::new(), dgrid, dtpg)
        .expect("bulk_dequant_kv_int4 dispatch");

    let out_bytes = d_out.outputs.get("out").expect("out buffer");
    unpack_bytes(out_bytes, dt)
}

/// Same but int8.
fn roundtrip_int8(shape: &Shape, dt: Dt, source: &[f32]) -> Vec<f32> {
    let dtype = dt.to_dtype();
    let bits = 8u32;
    let vals_per_pack = 32u32 / bits;
    let groups_per_head = shape.head_dim / shape.group_size;

    let n_packed_per_slot = shape.head_dim / vals_per_pack as usize;
    let n_groups_per_slot = groups_per_head;

    let w_total = shape.n_kv_heads * shape.max_seq * n_packed_per_slot;
    let s_total = shape.n_kv_heads * shape.max_seq * n_groups_per_slot;

    let ctx = Context::new().expect("Context::new on macOS");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(source, dt));
    buffers.insert("out_w".into(), pack_u32_bytes(&vec![0u32; w_total]));
    buffers.insert("out_s".into(), pack_bytes(&vec![0.0f32; s_total], dt));
    buffers.insert("out_b".into(), pack_bytes(&vec![0.0f32; s_total], dt));
    buffers.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (shape.position as u32).to_le_bytes().to_vec());

    let mut qkernel = quantize_kv_int8::kernel_ir_for(dtype);
    qkernel.mode = KernelMode::Grid3D;
    let (grid, tpg) = quantize_dispatch_grid(shape, bits);
    let q_out = ctx
        .dispatch_with_grid(&qkernel, &buffers, &BTreeMap::new(), grid, tpg)
        .expect("quantize_kv_int8 dispatch");

    let w_bytes = q_out.outputs.get("out_w").expect("out_w buffer").clone();
    let s_bytes = q_out.outputs.get("out_s").expect("out_s buffer").clone();
    let b_bytes = q_out.outputs.get("out_b").expect("out_b buffer").clone();

    let recon_total = shape.n_kv_heads * shape.max_seq * shape.head_dim;
    let mut dbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    dbuf.insert("in_w".into(), w_bytes);
    dbuf.insert("in_s".into(), s_bytes);
    dbuf.insert("in_b".into(), b_bytes);
    dbuf.insert("out".into(), pack_bytes(&vec![0.0f32; recon_total], dt));
    dbuf.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    dbuf.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    dbuf.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    dbuf.insert("n_positions".into(), (shape.n_positions as u32).to_le_bytes().to_vec());

    let mut dkernel = bulk_dequant_kv_int8::kernel_ir_for(dtype);
    dkernel.mode = KernelMode::Grid3D;
    let (dgrid, dtpg) = dequant_dispatch_grid(shape);
    let d_out = ctx
        .dispatch_with_grid(&dkernel, &dbuf, &BTreeMap::new(), dgrid, dtpg)
        .expect("bulk_dequant_kv_int8 dispatch");

    let out_bytes = d_out.outputs.get("out").expect("out buffer");
    unpack_bytes(out_bytes, dt)
}

/// Compare reconstructed slice against source at the slot we wrote.
/// `levels` = number of quantization levels (15 for int4, 255 for int8).
fn assert_roundtrip(
    shape: &Shape,
    dt: Dt,
    source: &[f32],
    recon: &[f32],
    levels: f32,
    label: &str,
) {
    // recon layout in `out` buffer is [n_kv_heads, max_seq, head_dim]
    // (matches raw KVCache; `n_positions` only governs how much of the
    // window the dequant kernel walks).
    let mut max_abs_err = 0.0_f32;
    let mut worst_idx = (0usize, 0usize);
    for h in 0..shape.n_kv_heads {
        for d in 0..shape.head_dim {
            let src_idx = h * shape.head_dim + d;
            let cache_idx =
                h * shape.max_seq * shape.head_dim + shape.position * shape.head_dim + d;
            let s = source[src_idx];
            let r = recon[cache_idx];
            let err = (s - r).abs();
            if err > max_abs_err {
                max_abs_err = err;
                worst_idx = (h, d);
            }
        }
    }
    // Per-group range upper bound (source values live in roughly [-2, 2]).
    // Step size = range / levels; one-step quantization error + ~1 ULP
    // of dtype roundtrip = combined tolerance.
    let group_range_ub = 4.0_f32; // worst-case source range per group
    let step = group_range_ub / levels;
    let dtype_slack = match dt {
        Dt::F32 => 0.0,
        Dt::F16 => 1e-3,
        Dt::Bf16 => 1e-2,
    };
    let tol = step * 1.5 + dtype_slack;
    assert!(
        max_abs_err <= tol,
        "{label}: max abs err = {max_abs_err:.4} > tol {tol:.4} at (h={}, d={})",
        worst_idx.0,
        worst_idx.1,
    );
}

// ── int4 tests ───────────────────────────────────────────────────────

#[test]
fn kv_cache_int4_roundtrip_f32() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F32, 0x9E37_79B9);
    let recon = roundtrip_int4(&shape, Dt::F32, &source);
    assert_roundtrip(&shape, Dt::F32, &source, &recon, 15.0, "int4 f32");
}

#[test]
fn kv_cache_int4_roundtrip_f16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F16, 0xDEAD_BEEF);
    let recon = roundtrip_int4(&shape, Dt::F16, &source);
    assert_roundtrip(&shape, Dt::F16, &source, &recon, 15.0, "int4 f16");
}

#[test]
fn kv_cache_int4_roundtrip_bf16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::Bf16, 0xCAFE_BABE);
    let recon = roundtrip_int4(&shape, Dt::Bf16, &source);
    assert_roundtrip(&shape, Dt::Bf16, &source, &recon, 15.0, "int4 bf16");
}

// ── int8 tests ───────────────────────────────────────────────────────

#[test]
fn kv_cache_int8_roundtrip_f32() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F32, 0x9E37_79B9);
    let recon = roundtrip_int8(&shape, Dt::F32, &source);
    assert_roundtrip(&shape, Dt::F32, &source, &recon, 255.0, "int8 f32");
}

#[test]
fn kv_cache_int8_roundtrip_f16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::F16, 0xDEAD_BEEF);
    let recon = roundtrip_int8(&shape, Dt::F16, &source);
    assert_roundtrip(&shape, Dt::F16, &source, &recon, 255.0, "int8 f16");
}

#[test]
fn kv_cache_int8_roundtrip_bf16() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let source = build_source(&shape, Dt::Bf16, 0xCAFE_BABE);
    let recon = roundtrip_int8(&shape, Dt::Bf16, &source);
    assert_roundtrip(&shape, Dt::Bf16, &source, &recon, 255.0, "int8 bf16");
}

// ── Cross-slot isolation ─────────────────────────────────────────────
//
// `quantize_kv_*` writes only to its `position` slot — verify by
// pre-filling neighboring slots with a sentinel and checking they
// survive a quantize+dequant cycle. Catches index formula regressions
// (e.g. accidentally striding by head_dim instead of max_seq).
#[test]
fn kv_cache_int8_does_not_touch_other_slots_f32() {
    let _g = gpu_lock();
    let shape = Shape::qwen_decode();
    let dt = Dt::F32;
    let dtype = dt.to_dtype();
    let bits = 8u32;
    let vals_per_pack = 32u32 / bits;
    let groups_per_head = shape.head_dim / shape.group_size;
    let n_packed_per_slot = shape.head_dim / vals_per_pack as usize;
    let n_groups_per_slot = groups_per_head;

    let w_total = shape.n_kv_heads * shape.max_seq * n_packed_per_slot;
    let s_total = shape.n_kv_heads * shape.max_seq * n_groups_per_slot;

    // Pre-fill with a known sentinel pattern so an out-of-slot write
    // shows up as a divergence from the sentinel after dequant.
    let sentinel_w: Vec<u32> = (0..w_total).map(|i| 0xDEAD0000 | (i as u32 & 0xFFFF)).collect();
    let sentinel_s = vec![1.5_f32; s_total];
    let sentinel_b = vec![-0.25_f32; s_total];

    let source = build_source(&shape, dt, 0x1234_5678);

    let ctx = Context::new().expect("Context::new on macOS");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(&source, dt));
    buffers.insert("out_w".into(), pack_u32_bytes(&sentinel_w));
    buffers.insert("out_s".into(), pack_bytes(&sentinel_s, dt));
    buffers.insert("out_b".into(), pack_bytes(&sentinel_b, dt));
    buffers.insert("head_dim".into(), (shape.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (shape.max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (shape.group_size as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (shape.position as u32).to_le_bytes().to_vec());

    let mut qkernel = quantize_kv_int8::kernel_ir_for(dtype);
    qkernel.mode = KernelMode::Grid3D;
    let (grid, tpg) = quantize_dispatch_grid(&shape, bits);
    let q_out = ctx
        .dispatch_with_grid(&qkernel, &buffers, &BTreeMap::new(), grid, tpg)
        .expect("quantize_kv_int8 dispatch");

    let w_after = unpack_u32_bytes(q_out.outputs.get("out_w").expect("out_w"));
    let s_after = unpack_bytes(q_out.outputs.get("out_s").expect("out_s"), dt);
    let b_after = unpack_bytes(q_out.outputs.get("out_b").expect("out_b"), dt);

    // Slots OTHER than `position` must retain their sentinel exactly.
    for h in 0..shape.n_kv_heads {
        for p in 0..shape.max_seq {
            if p == shape.position {
                continue;
            }
            // Weight stripe: head*max_seq*n_packed_per_slot + p*n_packed_per_slot
            for w in 0..n_packed_per_slot {
                let idx = (h * shape.max_seq + p) * n_packed_per_slot + w;
                assert_eq!(
                    w_after[idx], sentinel_w[idx],
                    "weight cross-slot bleed at (h={h}, p={p}, w={w})",
                );
            }
            // Scale/bias stripe.
            for g in 0..n_groups_per_slot {
                let idx = (h * shape.max_seq + p) * n_groups_per_slot + g;
                assert!(
                    (s_after[idx] - sentinel_s[idx]).abs() < 1e-6,
                    "scale cross-slot bleed at (h={h}, p={p}, g={g})",
                );
                assert!(
                    (b_after[idx] - sentinel_b[idx]).abs() < 1e-6,
                    "bias cross-slot bleed at (h={h}, p={p}, g={g})",
                );
            }
        }
    }
}

//! End-to-end correctness test for `ffai::aura_encode` on real Metal.
//!
//! Pins the kernel's three invariants:
//!   - **TPG = dim** (one thread per rotated coordinate).
//!   - **dim % 32 == 0** (`simd_sum` requires full simdgroups).
//!   - **dim ≤ 1024** (`threadgroup_alloc("shared_unit", 1024)`).
//!
//! The wrapper at `FFAI/Sources/FFAI/Ops.swift:Ops.auraEncode` enforces
//! these. Without them, a caller passing `dim = 33` (or any non-multiple
//! of 32) produces undefined `simd_sum` results and silently
//! miscompresses the K/V cache. See FFAI post-mortem 2026-05-19.
//!
//! Test approach: identity rotation + simple symmetric int4 codebook
//! aligned to a uniform-grid Lloyd-Max boundary set. Compares
//! `packed_out` (bit-exact, since quantisation is deterministic) and
//! `norms_out` (within fp32 tolerance) against the CPU reference.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{
    Dt,
    max_abs_diff,
    naive_aura_encode_f32,
    pack_bytes,
    pack_u32_bytes,
    ramp,
    unpack_bytes,
    unpack_u32_bytes,
};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::aura_encode::aura_encode_int4;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Identity rotation `[dim, dim]` — row d is the unit vector e_d.
fn identity_rotation(dim: usize) -> Vec<f32> {
    let mut r = vec![0.0_f32; dim * dim];
    for d in 0..dim {
        r[d * dim + d] = 1.0;
    }
    r
}

/// Symmetric int4 codebook: 16 evenly-spaced centroids in [-1, 1].
/// 15 boundaries at the midpoints. Mirrors a simplified Lloyd-Max
/// table; enough to exercise the kernel's quantisation arithmetic
/// without needing the production llama.cpp k_quants table.
fn int4_uniform_codebook() -> (Vec<f32>, Vec<f32>) {
    let levels = 16;
    let codebook: Vec<f32> =
        (0..levels).map(|i| -1.0 + 2.0 * (i as f32) / (levels as f32 - 1.0)).collect();
    let boundaries: Vec<f32> =
        (0..levels - 1).map(|i| 0.5 * (codebook[i] + codebook[i + 1])).collect();
    (codebook, boundaries)
}

#[test]
fn aura_encode_int4_matches_naive_cpu_reference_f32() {
    // dim = 128 — production AURA dim and the smallest dim that
    // exercises the cross-simdgroup combine path (128 / 32 = 4
    // simdgroups; n_simd > 1 means `shared_norm` actually accumulates).
    let dim = 128usize;
    let bits = 4usize;
    let rows = 2usize;
    let packed_width = (dim * bits).div_ceil(32); // 16 u32 words for dim=128, bits=4.

    let (codebook, boundaries) = int4_uniform_codebook();
    let rotation = identity_rotation(dim);
    // ramp values in roughly [-0.4, 0.4] after the divide-by-norm —
    // covers most of the codebook range without saturating either end.
    let input = ramp(rows * dim, 23, 9.0);

    let (expected_packed, expected_norms) =
        naive_aura_encode_f32(&input, &rotation, &boundaries, &codebook, rows, dim, bits);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), f32_slice_to_bytes(&input));
    buffers.insert("rotation".into(), f32_slice_to_bytes(&rotation));
    buffers.insert("boundaries".into(), f32_slice_to_bytes(&boundaries));
    buffers.insert("codebook".into(), f32_slice_to_bytes(&codebook));
    buffers.insert("packed_out".into(), pack_u32_bytes(&vec![0u32; rows * packed_width]));
    buffers.insert("norms_out".into(), f32_slice_to_bytes(&vec![0.0_f32; rows]));
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    buffers.insert("packed_width".into(), (packed_width as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = aura_encode_int4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per row, `dim` threads per group (the kernel's
    // design TPG). For dim=128 that's 128 threads = 4 simdgroups.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [dim, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let packed_bytes =
        result.outputs.get("packed_out").expect("`packed_out` buffer in dispatch result");
    let norms_bytes =
        result.outputs.get("norms_out").expect("`norms_out` buffer in dispatch result");
    let actual_packed = unpack_u32_bytes(packed_bytes);
    let actual_norms = bytes_to_f32_vec(norms_bytes);

    // packed_out: bit-exact match. The quantisation is deterministic
    // boundary counting and the identity rotation removes any
    // matmul reordering noise, so there's no slack to give.
    assert_eq!(actual_packed, expected_packed, "packed_out mismatch — quantisation indices differ",);

    // norms_out: fp32 tolerance. `simd_sum` reorders the partials
    // relative to the CPU's left-fold, so a few ulp of drift is
    // expected at these magnitudes.
    let diff = max_abs_diff(&expected_norms, &actual_norms);
    assert!(diff < 1e-4, "norms_out diverges from CPU reference: max |diff| = {diff:.2e}",);
}

#[test]
fn aura_encode_int4_minimum_dim_f32() {
    // dim = 32 = exactly one Apple simdgroup. Smallest legal dim;
    // pins the n_simd = 1 path where `shared_norm` only holds one
    // partial. Anything smaller (dim = 16) would silently reduce wrong.
    let dim = 32usize;
    let bits = 4usize;
    let rows = 1usize;
    let packed_width = (dim * bits).div_ceil(32); // 4 u32 words.

    let (codebook, boundaries) = int4_uniform_codebook();
    let rotation = identity_rotation(dim);
    let input = ramp(rows * dim, 13, 6.0);

    let (expected_packed, expected_norms) =
        naive_aura_encode_f32(&input, &rotation, &boundaries, &codebook, rows, dim, bits);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), f32_slice_to_bytes(&input));
    buffers.insert("rotation".into(), f32_slice_to_bytes(&rotation));
    buffers.insert("boundaries".into(), f32_slice_to_bytes(&boundaries));
    buffers.insert("codebook".into(), f32_slice_to_bytes(&codebook));
    buffers.insert("packed_out".into(), pack_u32_bytes(&vec![0u32; rows * packed_width]));
    buffers.insert("norms_out".into(), f32_slice_to_bytes(&vec![0.0_f32; rows]));
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    buffers.insert("packed_width".into(), (packed_width as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = aura_encode_int4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [dim, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let packed_bytes =
        result.outputs.get("packed_out").expect("`packed_out` buffer in dispatch result");
    let norms_bytes =
        result.outputs.get("norms_out").expect("`norms_out` buffer in dispatch result");
    let actual_packed = unpack_u32_bytes(packed_bytes);
    let actual_norms = bytes_to_f32_vec(norms_bytes);

    assert_eq!(actual_packed, expected_packed, "packed_out mismatch at dim=32");
    let diff = max_abs_diff(&expected_norms, &actual_norms);
    assert!(diff < 1e-4, "norms_out diverges: max |diff| = {diff:.2e}");
}

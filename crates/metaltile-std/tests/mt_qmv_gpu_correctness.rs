//! End-to-end correctness test for `mlx::mt_qmv` on real Metal.
//!
//! Pins the kernel's strict `tpg=64` (2 simdgroups × 32 lanes), 8-row
//! tile-per-TG dispatch invariant. Wrapper must use
//! `grid = (m/8 * 64, 1, 1)`, `tg = (64, 1, 1)` — anything else
//! silently miscomputes (different from the GPU-pin risk in
//! sdpa_decode, since `mt_qmv` doesn't have a `n_simd / 32` divide,
//! but the row-tile arithmetic uses `tg * 8 + sg * 4` which is wrong
//! for non-2-simdgroup layouts).
//!
//! Quantization layout (matches MLX qmv_fast):
//!   weight  [m, k/8]            u32 — 8 nibbles per u32
//!   scales  [m, k/group_size]   T
//!   biases  [m, k/group_size]   T
//!   x       [k]                 T
//!   out     [m]                 T
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::mt_qmv;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

fn quantize_int4_per_group(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let k = row.len();
    let n_groups = k / group_size;
    let mut packed = vec![0u32; k / 8];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let g_off = g * group_size;
        let g_slice = &row[g_off..g_off + group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / 15.0 };
        let bias = mn;
        scales[g] = scale;
        biases[g] = bias;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - bias) / scale).round().clamp(0.0, 15.0) as u32;
            let idx = g_off + i;
            let word = idx / 8;
            let shift = (idx % 8) * 4;
            packed[word] |= q << shift;
        }
    }
    (packed, scales, biases)
}

fn naive_qmv_f32(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    let n_groups = k / group_size;
    let packs_per_row = k / 8;
    let mut out = vec![0.0_f32; m];
    for i in 0..m {
        let mut acc = 0.0_f32;
        for j in 0..k {
            let g = j / group_size;
            let scale = scales[i * n_groups + g];
            let bias = biases[i * n_groups + g];
            let word = packed[i * packs_per_row + j / 8];
            let shift = (j % 8) * 4;
            let q = ((word >> shift) & 0xf) as f32;
            acc += (q * scale + bias) * x[j];
        }
        out[i] = acc;
    }
    out
}

#[test]
fn mt_qmv_matches_naive_cpu_reference_f32() {
    // m must be a multiple of 8 (kernel processes 8 rows per TG).
    // k = 4096 — production Llama hidden dim.
    let m = 32usize;
    let k = 4096usize;
    let group_size = 64usize;
    let n_groups_per_row = k / group_size;

    let mut packed_rows: Vec<Vec<u32>> = Vec::with_capacity(m);
    let mut scales: Vec<f32> = Vec::with_capacity(m * n_groups_per_row);
    let mut biases: Vec<f32> = Vec::with_capacity(m * n_groups_per_row);
    for i in 0..m {
        let row: Vec<f32> = (0..k).map(|j| (((i * 7 + j) % 23) as f32 - 11.0) * 0.02).collect();
        let (pk, sc, bs) = quantize_int4_per_group(&row, group_size);
        packed_rows.push(pk);
        scales.extend(sc);
        biases.extend(bs);
    }
    let packed: Vec<u32> = packed_rows.into_iter().flatten().collect();
    let x: Vec<f32> = (0..k).map(|j| (((j * 13) % 11) as f32 - 5.0) * 0.05).collect();

    let expected = naive_qmv_f32(&packed, &scales, &biases, &x, m, k, group_size);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(&packed));
    buffers.insert("scales".into(), f32_slice_to_bytes(&scales));
    buffers.insert("biases".into(), f32_slice_to_bytes(&biases));
    buffers.insert("x".into(), f32_slice_to_bytes(&x));
    buffers.insert("out".into(), vec![0u8; m * 4]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (n_groups_per_row as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = mt_qmv::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // m/8 threadgroups (8-row tile per TG), 64 threads per TG
    // (2 simdgroups × 32 lanes — kernel's `tpg=64`).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m / 8, 1, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = bytes_to_f32_vec(out_bytes);

    let diff = max_abs_diff(&expected, &actual);
    // Looser tol — qmv runs an 8-row × 512-block × 16-lane reduction
    // tree, more reordering than plain mt_gemv.
    assert!(diff < 5e-3, "mt_qmv f32: max |diff| = {diff:.2e} (expected < 5e-3)",);
}

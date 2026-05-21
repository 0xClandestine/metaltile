//! GPU correctness for `mlx::reduce` — the column-reduce and
//! segmented-reduce families (`mt_col_reduce*` / `mt_seg_reduce*`).
//!
//! Both are Grid3D one-thread-per-output kernels: column reduce folds
//! a strided column of a `[rows, cols]` matrix; segmented reduce folds
//! a contiguous fixed-length run of a flat buffer. The naive CPU
//! oracle reduces in f32 — the kernel must agree within a tolerance
//! that absorbs the half-precision accumulation drift.
//!
//! Coverage rationale: the Grid3D path is the trap here — a
//! `reduce_*` finishing step would lower to `simd_sum` and silently
//! sum 32 independent columns / segments together. The against-oracle
//! tests on a non-trivial `cols` (not a multiple of 32) catch that.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::Kernel};
use metaltile_runtime::Context;
use metaltile_std::mlx::reduce::{
    mt_col_reduce,
    mt_col_reduce_max,
    mt_col_reduce_min,
    mt_col_reduce_prod,
    mt_seg_reduce,
    mt_seg_reduce_max,
    mt_seg_reduce_min,
    mt_seg_reduce_prod,
};

/// Dispatch a Grid3D reduce kernel: `grid = [ceil(n_out/256), 1, 1]`,
/// `tg = [256, 1, 1]`, one thread per output element.
fn run(
    kernel_ir: fn(DType) -> Kernel,
    inp: &[f32],
    n_out: usize,
    dim_a: u32,
    dim_b: u32,
    dt: Dt,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    // Pad the output to ≥4 bytes for the 2-byte half dtypes.
    buffers.insert("out".into(), pack_bytes(&vec![0.0; n_out.max(2)], dt));
    // The two constexprs: (rows, cols) for col-reduce or
    // (n_segments, seg_len) for seg-reduce — same buffer layout.
    buffers.insert("rows".into(), dim_a.to_le_bytes().to_vec());
    buffers.insert("cols".into(), dim_b.to_le_bytes().to_vec());
    buffers.insert("n_segments".into(), dim_a.to_le_bytes().to_vec());
    buffers.insert("seg_len".into(), dim_b.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    // Grid3D kernel keeps its default Elementwise/Grid3D mode.
    let kernel = kernel_ir(dt.to_dtype());
    let grid_x = n_out.div_ceil(256);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_x, 1, 1], [256, 1, 1])
        .expect("reduce dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

/// Reduce f32 over `idx` indices selected by `pick`.
fn fold(init: f32, picks: impl Iterator<Item = f32>, op: fn(f32, f32) -> f32) -> f32 {
    picks.fold(init, op)
}

// ── Column reduce ────────────────────────────────────────────────────────

fn cpu_col(inp: &[f32], rows: usize, cols: usize, init: f32, op: fn(f32, f32) -> f32) -> Vec<f32> {
    (0..cols).map(|c| fold(init, (0..rows).map(|r| inp[r * cols + c]), op)).collect()
}

#[test]
fn col_reduce_sum_f32() {
    let _g = gpu_lock();
    let (rows, cols) = (37, 100); // cols not a multiple of 32 — catches simd_sum bleed
    let inp: Vec<f32> = (0..rows * cols).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
    let expected = cpu_col(&inp, rows, cols, 0.0, |a, b| a + b);
    let actual = run(mt_col_reduce::kernel_ir_for, &inp, cols, rows as u32, cols as u32, Dt::F32);
    assert!(actual.iter().any(|&v| v != 0.0), "all zeros");
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "col sum mismatch");
}

#[test]
fn col_reduce_max_min_f32() {
    let _g = gpu_lock();
    let (rows, cols) = (50, 70);
    let inp: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 1000) as f32 * 0.01 - 5.0).collect();
    let exp_max = cpu_col(&inp, rows, cols, f32::NEG_INFINITY, f32::max);
    let exp_min = cpu_col(&inp, rows, cols, f32::INFINITY, f32::min);
    let act_max =
        run(mt_col_reduce_max::kernel_ir_for, &inp, cols, rows as u32, cols as u32, Dt::F32);
    let act_min =
        run(mt_col_reduce_min::kernel_ir_for, &inp, cols, rows as u32, cols as u32, Dt::F32);
    assert!(max_abs_diff(&act_max, &exp_max) < 1e-4, "col max mismatch");
    assert!(max_abs_diff(&act_min, &exp_min) < 1e-4, "col min mismatch");
}

#[test]
fn col_reduce_prod_f32() {
    let _g = gpu_lock();
    let (rows, cols) = (8, 40);
    // Values near 1.0 so the product stays in range.
    let inp: Vec<f32> = (0..rows * cols).map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.02).collect();
    let expected = cpu_col(&inp, rows, cols, 1.0, |a, b| a * b);
    let actual =
        run(mt_col_reduce_prod::kernel_ir_for, &inp, cols, rows as u32, cols as u32, Dt::F32);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "col prod mismatch");
}

#[test]
fn col_reduce_sum_f16() {
    let _g = gpu_lock();
    let (rows, cols) = (20, 64);
    let inp_f32: Vec<f32> = (0..rows * cols).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let inp: Vec<f32> = inp_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = cpu_col(&inp, rows, cols, 0.0, |a, b| a + b);
    let actual = run(mt_col_reduce::kernel_ir_for, &inp, cols, rows as u32, cols as u32, Dt::F16);
    assert!(max_abs_diff(&actual, &expected) < 5e-2, "col sum f16 mismatch");
}

#[test]
fn col_reduce_sum_bf16() {
    let _g = gpu_lock();
    let (rows, cols) = (20, 64);
    let inp_f32: Vec<f32> = (0..rows * cols).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let inp: Vec<f32> = inp_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = cpu_col(&inp, rows, cols, 0.0, |a, b| a + b);
    let actual = run(mt_col_reduce::kernel_ir_for, &inp, cols, rows as u32, cols as u32, Dt::Bf16);
    assert!(max_abs_diff(&actual, &expected) < 2e-1, "col sum bf16 mismatch");
}

// ── Segmented reduce ─────────────────────────────────────────────────────

fn cpu_seg(
    inp: &[f32],
    n_seg: usize,
    seg_len: usize,
    init: f32,
    op: fn(f32, f32) -> f32,
) -> Vec<f32> {
    (0..n_seg)
        .map(|s| fold(init, inp[s * seg_len..(s + 1) * seg_len].iter().copied(), op))
        .collect()
}

#[test]
fn seg_reduce_sum_f32() {
    let _g = gpu_lock();
    let (n_seg, seg_len) = (300, 17); // many short segments — the seg-reduce use case
    let inp: Vec<f32> = (0..n_seg * seg_len).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
    let expected = cpu_seg(&inp, n_seg, seg_len, 0.0, |a, b| a + b);
    let actual =
        run(mt_seg_reduce::kernel_ir_for, &inp, n_seg, n_seg as u32, seg_len as u32, Dt::F32);
    assert!(actual.iter().any(|&v| v != 0.0), "all zeros");
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "seg sum mismatch");
}

#[test]
fn seg_reduce_max_min_f32() {
    let _g = gpu_lock();
    let (n_seg, seg_len) = (128, 33);
    let inp: Vec<f32> =
        (0..n_seg * seg_len).map(|i| ((i * 6151) % 2000) as f32 * 0.005 - 5.0).collect();
    let exp_max = cpu_seg(&inp, n_seg, seg_len, f32::NEG_INFINITY, f32::max);
    let exp_min = cpu_seg(&inp, n_seg, seg_len, f32::INFINITY, f32::min);
    let act_max =
        run(mt_seg_reduce_max::kernel_ir_for, &inp, n_seg, n_seg as u32, seg_len as u32, Dt::F32);
    let act_min =
        run(mt_seg_reduce_min::kernel_ir_for, &inp, n_seg, n_seg as u32, seg_len as u32, Dt::F32);
    assert!(max_abs_diff(&act_max, &exp_max) < 1e-4, "seg max mismatch");
    assert!(max_abs_diff(&act_min, &exp_min) < 1e-4, "seg min mismatch");
}

#[test]
fn seg_reduce_prod_f32() {
    let _g = gpu_lock();
    let (n_seg, seg_len) = (64, 12);
    let inp: Vec<f32> = (0..n_seg * seg_len).map(|i| 1.0 + ((i % 5) as f32 - 2.0) * 0.03).collect();
    let expected = cpu_seg(&inp, n_seg, seg_len, 1.0, |a, b| a * b);
    let actual =
        run(mt_seg_reduce_prod::kernel_ir_for, &inp, n_seg, n_seg as u32, seg_len as u32, Dt::F32);
    assert!(max_abs_diff(&actual, &expected) < 1e-4, "seg prod mismatch");
}

#[test]
fn seg_reduce_sum_bf16() {
    let _g = gpu_lock();
    let (n_seg, seg_len) = (100, 24);
    let inp_f32: Vec<f32> = (0..n_seg * seg_len).map(|i| ((i % 11) as f32 - 5.0) * 0.04).collect();
    let inp: Vec<f32> = inp_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = cpu_seg(&inp, n_seg, seg_len, 0.0, |a, b| a + b);
    let actual =
        run(mt_seg_reduce::kernel_ir_for, &inp, n_seg, n_seg as u32, seg_len as u32, Dt::Bf16);
    assert!(max_abs_diff(&actual, &expected) < 1e-1, "seg sum bf16 mismatch");
}

//! GPU correctness for `mlx::strided::mt_strided_copy<T>`.
//!
//! `mt_strided_copy` copies a 2-D slice from a strided (padded) source buffer
//! into a contiguous output buffer. The kernel signature is:
//!
//! ```
//! fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>,
//!                        #[constexpr] cols: u32)
//! ```
//!
//! The `#[strided]` attribute causes the runtime to expect **three** named
//! buffers for `src`:
//! - `src`          — the raw data bytes
//! - `src_shape`    — `[rows, cols]` as 4 LE u32 bytes each
//! - `src_strides`  — `[row_stride, col_stride]` (row_stride = src_cols + pad;
//!   col_stride = 1)
//!
//! If `src_shape` and `src_strides` are omitted from the buffer map, the
//! runtime derives them from the kernel's declared param shape. Since the
//! kernel declares its shape as `[Unknown, Unknown]` (strided, runtime-size),
//! they cannot be derived — the caller MUST supply them.
//!
//! Dispatch: Grid3D, `grid = [rows, dest_cols, 1]`, `tpg = [1, 1, 1]`.
//! Constexpr `cols = dest_cols` — the contiguous column count of the output.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::strided::{mt_strided_copy, mt_strided_copy_nd};

/// Pack a slice of u32 values as LE bytes (used for shape/stride buffers).
fn pack_u32_slice(vals: &[u32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }

/// Dispatch `mt_strided_copy<T>` to copy a `rows × dest_cols` tile from a
/// `rows × src_cols` padded source (where `src_cols >= dest_cols`).
///
/// # Arguments
/// * `src_data`   — the full padded source matrix (row-major, `rows * src_cols` elements)
/// * `src_cols`   — physical columns of the source (= dest_cols + padding)
/// * `dest_cols`  — logical columns to copy (the output width)
/// * `rows`       — row count
fn run_strided_copy(
    src_data: &[f32],
    dt: Dt,
    rows: usize,
    src_cols: usize,
    dest_cols: usize,
) -> Vec<f32> {
    let n_out = rows * dest_cols;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    // src — raw padded data
    buffers.insert("src".into(), pack_bytes(src_data, dt));
    // src_shape = [rows, dest_cols] (the logical 2-D view the kernel indexes)
    buffers.insert("src_shape".into(), pack_u32_slice(&[rows as u32, dest_cols as u32]));
    // src_strides = [src_cols, 1] (physical row stride, unit col stride)
    buffers.insert("src_strides".into(), pack_u32_slice(&[src_cols as u32, 1u32]));
    // out — contiguous output
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    // constexpr cols — the contiguous column count of the output
    buffers.insert("cols".into(), (dest_cols as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_strided_copy::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per (row, col) pair in the output.
    // grid = [rows, dest_cols, 1], tpg = [1, 1, 1].
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, dest_cols, 1], [1, 1, 1])
        .expect("strided_copy dispatch");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(n_out);
    out
}

/// CPU oracle: extract the `rows × dest_cols` submatrix from the padded source.
fn oracle_strided_copy(src: &[f32], rows: usize, src_cols: usize, dest_cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * dest_cols);
    for r in 0..rows {
        out.extend_from_slice(&src[r * src_cols..r * src_cols + dest_cols]);
    }
    out
}

#[test]
fn strided_copy_simple_submatrix_f32() {
    let _g = gpu_lock();
    // 4 rows × 8-column source, copy the first 4 columns (pad = 4).
    let rows = 4;
    let src_cols = 8;
    let dest_cols = 4;
    // Source: value = row * dest_cols + col (logical coords), padding = -999.
    let src: Vec<f32> =
        (0..rows)
            .flat_map(|r| {
                (0..src_cols).map(move |c| {
                    if c < dest_cols { (r * dest_cols + c) as f32 + 1.0 } else { -999.0 }
                })
            })
            .collect();
    let expected = oracle_strided_copy(&src, rows, src_cols, dest_cols);
    let actual = run_strided_copy(&src, Dt::F32, rows, src_cols, dest_cols);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy simple f32: max |diff| = {diff:.2e}");
    // Also verify that the padding values did not leak into the output.
    assert!(
        actual.iter().all(|&v| v > 0.0),
        "strided_copy: padding value (-999) leaked into output",
    );
}

#[test]
fn strided_copy_matches_bench_shape_f32() {
    let _g = gpu_lock();
    // Match the BenchSpec shape: m=8, n=16, pad=4 (from the correctness sub-problem
    // in `run_spec.rs`: cm=8, cn=16, cp=4, src_stride=20).
    let rows = 8;
    let dest_cols = 16;
    let pad = 4;
    let src_cols = dest_cols + pad;
    let src: Vec<f32> =
        (0..rows)
            .flat_map(|r| {
                (0..src_cols).map(move |c| {
                    if c < dest_cols { (r * dest_cols + c) as f32 + 1.0 } else { -999.0 }
                })
            })
            .collect();
    let expected: Vec<f32> = (0..rows * dest_cols).map(|i| i as f32 + 1.0).collect();
    let actual = run_strided_copy(&src, Dt::F32, rows, src_cols, dest_cols);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy bench shape f32: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_no_padding_is_identity_f32() {
    let _g = gpu_lock();
    // When src_cols == dest_cols, strided copy is an identity copy.
    let rows = 4;
    let cols = 6;
    let src: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
    let expected = src.clone();
    let actual = run_strided_copy(&src, Dt::F32, rows, cols, cols);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy identity f32: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_matches_oracle_f16() {
    let _g = gpu_lock();
    let rows = 4;
    let src_cols = 8;
    let dest_cols = 4;
    let src: Vec<f32> = (0..rows * src_cols)
        .map(|i| if i % src_cols < dest_cols { Dt::F16.round((i as f32 - 8.0) * 0.2) } else { 0.0 })
        .collect();
    let expected = oracle_strided_copy(&src, rows, src_cols, dest_cols);
    let actual = run_strided_copy(&src, Dt::F16, rows, src_cols, dest_cols);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-3, "strided_copy f16: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let rows = 2;
    let src_cols = 4;
    let dest_cols = 2;
    let src: Vec<f32> = (1..=rows * src_cols).map(|i| i as f32).collect();
    let actual = run_strided_copy(&src, Dt::F32, rows, src_cols, dest_cols);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "strided_copy: all-zero output for non-zero input (empty kernel body?)",
    );
}

// ─── mt_strided_copy_nd — general N-D strided copy ───────────────────────

/// Dispatch `mt_strided_copy_nd<T>`: copy a logical `shape` tensor out of a
/// physically-strided `src` buffer into a contiguous output.
///
/// `shape` / `strides` are per-dimension u32 arrays (major-to-minor). The
/// output is contiguous row-major with `n_out = product(shape)` elements.
fn run_strided_copy_nd(src_data: &[f32], dt: Dt, shape: &[u32], strides: &[u32]) -> Vec<f32> {
    let n_out: usize = shape.iter().map(|&s| s as usize).product();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(src_data, dt));
    buffers.insert("shape".into(), pack_u32_slice(shape));
    buffers.insert("strides".into(), pack_u32_slice(strides));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    buffers.insert("rank".into(), (shape.len() as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_strided_copy_nd::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per output element.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_out, 1, 1], [1, 1, 1])
        .expect("strided_copy_nd dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out buffer"), dt);
    out.truncate(n_out);
    out
}

/// CPU oracle for N-D strided copy: for each contiguous output index `p`,
/// unravel against `shape` and gather `src[Σ coord_d · strides[d]]`.
fn oracle_strided_copy_nd(src: &[f32], shape: &[u32], strides: &[u32]) -> Vec<f32> {
    let n_out: usize = shape.iter().map(|&s| s as usize).product();
    let rank = shape.len();
    let mut out = Vec::with_capacity(n_out);
    for p in 0..n_out {
        let mut rem = p;
        let mut src_off = 0usize;
        for d in (0..rank).rev() {
            let extent = shape[d] as usize;
            let coord = rem % extent;
            rem /= extent;
            src_off += coord * strides[d] as usize;
        }
        out.push(src[src_off]);
    }
    out
}

#[test]
fn strided_copy_nd_1d_contiguous_f32() {
    let _g = gpu_lock();
    // Rank-1 unit-stride copy is a plain contiguous copy.
    let shape = [6u32];
    let strides = [1u32];
    let src: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    let expected = oracle_strided_copy_nd(&src, &shape, &strides);
    let actual = run_strided_copy_nd(&src, Dt::F32, &shape, &strides);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy_nd 1d f32: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_nd_2d_padded_matches_2d_kernel_f32() {
    let _g = gpu_lock();
    // 2-D padded view: 4 rows × 4 logical cols, physical row stride 8.
    // Equivalent to the original `mt_strided_copy` padded-submatrix case.
    let rows = 4u32;
    let cols = 4u32;
    let row_stride = 8u32;
    let shape = [rows, cols];
    let strides = [row_stride, 1u32];
    let src: Vec<f32> = (0..rows * row_stride)
        .map(|i| if i % row_stride < cols { (i as f32) + 1.0 } else { -999.0 })
        .collect();
    let expected = oracle_strided_copy_nd(&src, &shape, &strides);
    let actual = run_strided_copy_nd(&src, Dt::F32, &shape, &strides);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy_nd 2d padded f32: max |diff| = {diff:.2e}");
    assert!(actual.iter().all(|&v| v > 0.0), "padding leaked into output");
}

#[test]
fn strided_copy_nd_3d_matches_oracle_f32() {
    let _g = gpu_lock();
    // 3-D logical [2, 3, 4] out of a physically larger [2, 3, 6] buffer
    // (innermost dim padded by 2).
    let shape = [2u32, 3u32, 4u32];
    let phys = [2usize, 3, 6];
    let strides = [(phys[1] * phys[2]) as u32, phys[2] as u32, 1u32];
    let total: usize = phys.iter().product();
    let src: Vec<f32> = (0..total).map(|i| i as f32 * 0.25 - 3.0).collect();
    let expected = oracle_strided_copy_nd(&src, &shape, &strides);
    let actual = run_strided_copy_nd(&src, Dt::F32, &shape, &strides);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy_nd 3d f32: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_nd_3d_transpose_f32() {
    let _g = gpu_lock();
    // Transpose of a contiguous [3, 4, 5] tensor: logical shape [5, 4, 3],
    // strides are the contiguous strides permuted to (2, 1, 0) order.
    // Source layout is contiguous; only the strides express the transpose.
    let src_dims = [3usize, 4, 5];
    let cont = [(src_dims[1] * src_dims[2]) as u32, src_dims[2] as u32, 1u32];
    // Permutation (0,1,2) -> (2,1,0): logical dim 0 ← src dim 2, etc.
    let shape = [src_dims[2] as u32, src_dims[1] as u32, src_dims[0] as u32];
    let strides = [cont[2], cont[1], cont[0]];
    let total: usize = src_dims.iter().product();
    let src: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let expected = oracle_strided_copy_nd(&src, &shape, &strides);
    let actual = run_strided_copy_nd(&src, Dt::F32, &shape, &strides);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy_nd 3d transpose f32: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_nd_4d_broadcast_axis_f32() {
    let _g = gpu_lock();
    // 4-D logical [2, 3, 2, 4] where dim 2 is a broadcast axis: its
    // source stride is 0, so every coord on that axis reads the same row.
    let shape = [2u32, 3u32, 2u32, 4u32];
    // Source physically holds [2, 3, 1, 4]; the size-2 logical axis 2
    // broadcasts from a single physical slot (stride 0).
    let strides = [(3u32 * 4u32), 4u32, 0u32, 1u32];
    let phys_total = (2 * 3) * 4;
    let src: Vec<f32> = (0..phys_total).map(|i| i as f32 * 1.5 + 0.5).collect();
    let expected = oracle_strided_copy_nd(&src, &shape, &strides);
    let actual = run_strided_copy_nd(&src, Dt::F32, &shape, &strides);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "strided_copy_nd 4d broadcast f32: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_nd_3d_matches_oracle_f16() {
    let _g = gpu_lock();
    let shape = [2u32, 2u32, 3u32];
    let phys = [2usize, 2, 5];
    let strides = [(phys[1] * phys[2]) as u32, phys[2] as u32, 1u32];
    let total: usize = phys.iter().product();
    let src: Vec<f32> = (0..total).map(|i| Dt::F16.round(i as f32 * 0.1)).collect();
    let expected = oracle_strided_copy_nd(&src, &shape, &strides);
    let actual = run_strided_copy_nd(&src, Dt::F16, &shape, &strides);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-3, "strided_copy_nd 3d f16: max |diff| = {diff:.2e}");
}

#[test]
fn strided_copy_nd_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let shape = [2u32, 3u32];
    let strides = [4u32, 1u32];
    let src: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let actual = run_strided_copy_nd(&src, Dt::F32, &shape, &strides);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "strided_copy_nd: all-zero output for non-zero input (empty kernel body?)",
    );
}

#[test]
#[ignore = "perf bench — run with --ignored --nocapture"]
fn strided_copy_perf_bench_f32() {
    use std::time::Instant;
    let _g = gpu_lock();
    let rows = 1024;
    let dest_cols = 4096;
    let pad = 128;
    let src_cols = dest_cols + pad;
    let src: Vec<f32> = (0..rows * src_cols).map(|i| (i % 256) as f32 * 0.01).collect();
    let ctx = Context::new().expect("Context::new");
    let mut kernel = mt_strided_copy::kernel_ir_for(metaltile_core::dtype::DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Warmup
    for _ in 0..5 {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("src".into(), pack_bytes(&src, Dt::F32));
        buffers.insert("src_shape".into(), pack_u32_slice(&[rows as u32, dest_cols as u32]));
        buffers.insert("src_strides".into(), pack_u32_slice(&[src_cols as u32, 1u32]));
        buffers.insert("out".into(), vec![0u8; rows * dest_cols * 4]);
        buffers.insert("cols".into(), (dest_cols as u32).to_le_bytes().to_vec());
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, dest_cols, 1], [
            1, 1, 1,
        ])
        .expect("warmup");
    }
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("src".into(), pack_bytes(&src, Dt::F32));
        buffers.insert("src_shape".into(), pack_u32_slice(&[rows as u32, dest_cols as u32]));
        buffers.insert("src_strides".into(), pack_u32_slice(&[src_cols as u32, 1u32]));
        buffers.insert("out".into(), vec![0u8; rows * dest_cols * 4]);
        buffers.insert("cols".into(), (dest_cols as u32).to_le_bytes().to_vec());
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, dest_cols, 1], [
            1, 1, 1,
        ])
        .expect("bench");
    }
    let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
    let bytes = rows as f64 * dest_cols as f64 * 4.0 * 2.0;
    let gb_s = bytes / elapsed_us / 1e3;
    println!("strided_copy f32 {rows}×{dest_cols}+{pad}: {elapsed_us:.1} µs  |  {gb_s:.1} GB/s");
}

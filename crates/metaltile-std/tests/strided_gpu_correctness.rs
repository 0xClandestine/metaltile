//! GPU correctness for `mlx::strided` — strided copy (`mt_strided_copy`).
//!
//! `mt_strided_copy<T>`: reads a 2D strided tensor `src[(row, col)]` and
//! writes contiguously to `out[row * cols + col]`. The `#[strided]` attribute
//! tells the codegen to use strided indexing for `src` while `out` is flat.
//!
//! ## DISPATCH (mt_strided_copy)
//! Grid3D: `grid = [rows, cols, 1]`, `tg = [1, 1, 1]` (one thread per element).
//! `program_id::<0>()` = row, `program_id::<1>()` = col.
//!
//! CPU oracle: direct copy of the input — for a contiguous source with a
//! stride that matches the column count the two are identical.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::strided::mt_strided_copy;

/// Run mt_strided_copy over a 2D [rows × cols] tensor.
///
/// `src_stride` is the distance between consecutive rows in the source
/// (elements, not bytes). For a contiguous source, `src_stride == cols`.
fn run_strided_copy(src: &[f32], rows: usize, cols: usize, src_stride: usize, dt: Dt) -> Vec<f32> {
    // The kernel reads `src[(row, col)]` via strided indexing and writes
    // contiguously to `out`. We pass the strides as constexprs.
    // The `#[strided]` tensor parameter expects (shape, strides) metadata
    // in the buffer map. For a 2D row-major layout with stride `src_stride`:
    //   strides = [src_stride, 1] (row-stride, col-stride in elements)
    //
    // Encoding: the runtime expects the buffer keyed by the tensor name;
    // shapes/strides are embedded via constexprs (each becomes a u32 entry).
    let n_out = rows * cols;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(src, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));

    // `cols` is a constexpr for the flat-output indexing formula.
    let mut constexprs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    constexprs.insert("cols".into(), (cols as u32).to_le_bytes().to_vec());
    // Strided tensor shape/strides: passed as additional constexprs.
    // The DSL `#[strided]` annotation generates shape[0], shape[1],
    // stride[0], stride[1] constexprs in the kernel IR.
    constexprs.insert("src_shape_0".into(), (rows as u32).to_le_bytes().to_vec());
    constexprs.insert("src_shape_1".into(), (cols as u32).to_le_bytes().to_vec());
    constexprs.insert("src_stride_0".into(), (src_stride as u32).to_le_bytes().to_vec());
    constexprs.insert("src_stride_1".into(), 1u32.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_strided_copy::kernel_ir_for(dt.to_dtype());

    // Grid3D: one thread per output element — rows × cols threads total.
    let result = ctx
        .dispatch_with_grid(
            &kernel,
            &buffers,
            &constexprs,
            [rows, cols, 1],
            [1, 1, 1],
        )
        .expect("strided_copy dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

/// CPU oracle: for a 2D contiguous input, strided copy is identity.
fn cpu_strided_copy(src: &[f32], rows: usize, cols: usize, src_stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(src[r * src_stride + c]);
        }
    }
    out
}

#[test]
fn strided_copy_contiguous_f32() {
    let _g = gpu_lock();
    let (rows, cols) = (8usize, 16usize);
    let src: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
    let expected = cpu_strided_copy(&src, rows, cols, cols);
    let actual = run_strided_copy(&src, rows, cols, cols, Dt::F32);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "strided_copy contiguous f32 mismatch");
}

#[test]
fn strided_copy_with_row_padding_f32() {
    // Padded source: each row is `cols` elements followed by `pad` extra elements.
    // Strided copy should skip the padding.
    let _g = gpu_lock();
    let (rows, cols, pad) = (4usize, 8usize, 4usize);
    let src_stride = cols + pad;
    let src: Vec<f32> = (0..rows * src_stride).map(|i| i as f32).collect();
    let expected = cpu_strided_copy(&src, rows, cols, src_stride);
    let actual = run_strided_copy(&src, rows, cols, src_stride, Dt::F32);
    assert!(max_abs_diff(&actual, &expected) < 1e-6, "strided_copy padded f32 mismatch");
}

#[test]
fn strided_copy_contiguous_f16() {
    let _g = gpu_lock();
    let (rows, cols) = (4usize, 32usize);
    let src: Vec<f32> = (0..rows * cols).map(|i| Dt::F16.round(i as f32 * 0.1 - 1.0)).collect();
    let expected = cpu_strided_copy(&src, rows, cols, cols);
    let actual = run_strided_copy(&src, rows, cols, cols, Dt::F16);
    assert!(max_abs_diff(&actual, &expected) < 1e-3, "strided_copy contiguous f16 mismatch");
}

#[test]
fn strided_copy_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let (rows, cols) = (2usize, 8usize);
    let src: Vec<f32> = (1..=rows * cols as usize).map(|i| i as f32).collect();
    let actual = run_strided_copy(&src, rows, cols, cols, Dt::F32);
    assert!(actual.iter().any(|&v| v != 0.0), "strided_copy output all zeros");
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Strided copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel]
pub fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>, #[constexpr] cols: u32) {
    let row = program_id::<0>();
    let col = program_id::<1>();
    let flat_out = row * cols + col;
    let val = load(src[(row, col)]);
    store(out[flat_out], val);
}

// ─── mt_strided_copy_nd ──────────────────────────────────────────────────
//
// General N-D strided copy — the MLX `copy_g` / `copy_g_nd{1,2,3}`
// counterpart. The 2-D `mt_strided_copy` above only handles a
// row-major-padded `[rows, cols]` source; this kernel copies an
// arbitrary-rank logical tensor out of a source buffer whose physical
// layout is described by per-dimension `shape` + `strides` arrays.
//
// The destination is always contiguous row-major: output element `p`
// (a flat index in `[0, n_out)`) maps to the multi-index obtained by
// unravelling `p` against `out_shape` (== logical `shape`), then the
// source byte offset is `Σ_d coord_d · strides[d]`. This is exactly
// MLX's `elem_to_loc` (`mlx/backend/metal/kernels/utils.h`).
//
// Because the source strides are *arbitrary* (not necessarily a
// padded row-major view), this generalises:
//   - padded copies         (the 2-D `mt_strided_copy` case),
//   - transposes            (strides permuted vs shape),
//   - broadcasts            (a stride of 0 on a broadcast axis),
//   - any slice / dilation  (non-unit innermost stride).
//
// Inputs:
//   src     — source data buffer (raw, physically strided)
//   shape   — [rank]   u32  logical extent of each dimension
//   strides — [rank]   u32  element stride of each source dimension
//   out     — [n_out]  contiguous row-major output
//
// Constexpr:
//   rank    — number of dimensions (logical). Compile-time constant so
//             the unravel loop is fully unrolled — no dynamic trip count.
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D** — one thread per output element, no cross-thread
//   cooperation. `program_id::<0>()` is the flat output index.
// - **Grid: `[n_out, 1, 1]`, TPG: `[1, 1, 1]`** (or any
//   `grid·tpg == n_out` split). `n_out == Π shape[d]`.
// - **`rank >= 1`.** `shape` and `strides` must each hold exactly
//   `rank` u32 entries; a short buffer reads out of bounds.
// - The unravel walks dimensions **last → first**: the running
//   remainder is divided by `shape[d]` from `d = rank-1` down to `0`,
//   so `strides` is interpreted in the same major-to-minor order as
//   `shape` (row-major logical indexing).
#[kernel]
pub fn mt_strided_copy_nd<T>(
    src: Tensor<T>,
    shape: Tensor<u32>,
    strides: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] rank: u32,
) {
    let p = program_id::<0>();
    // Unravel the flat output index `p` against `shape`, walking
    // dimensions from the innermost (last) to the outermost (first).
    // `rem` carries the not-yet-consumed portion of `p`; at each step
    // `coord = rem % shape[d]` peels off dimension `d`'s index and
    // `rem /= shape[d]` advances to the next-coarser dimension. The
    // source offset accumulates `coord · strides[d]`.
    let mut rem = p;
    let mut src_off = 0u32;
    for _i in range(0u32, rank, 1u32) {
        // d counts down: rank-1, rank-2, ..., 0.
        let d = rank - 1u32 - _i;
        let extent = load(shape[d]);
        let coord = rem - (rem / extent) * extent; // rem % extent
        rem = rem / extent;
        src_off = src_off + coord * load(strides[d]);
    }
    store(out[p], load(src[src_off]));
}

// ── bottom of source file ─────────────────────────────────────────────────
mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestSetup, TestBuffer}};

    fn pack_f32(vals: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice::<f32, u8>(vals).to_vec()
    }

    fn pack_u32_slice(vals: &[u32]) -> Vec<u8> {
        bytemuck::cast_slice::<u32, u8>(vals).to_vec()
    }

    /// CPU oracle: extract the `rows × dest_cols` submatrix from a padded source.
    fn oracle_strided_copy(
        src: &[f32],
        rows: usize,
        src_cols: usize,
        dest_cols: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * dest_cols);
        for r in 0..rows {
            out.extend_from_slice(&src[r * src_cols..r * src_cols + dest_cols]);
        }
        out
    }

    /// CPU oracle for N-D strided copy.
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

    #[test_kernel(name = "mlx/strided_copy_simple_f32", dtypes = [f32], tol = 1e-6)]
    fn test_strided_copy_simple_f32(dt: DType) -> TestSetup {
        let rows = 4usize;
        let src_cols = 8usize;
        let dest_cols = 4usize;
        let src: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..src_cols).map(move |c| {
                    if c < dest_cols { (r * dest_cols + c) as f32 + 1.0 } else { -999.0 }
                })
            })
            .collect();
        let expected = oracle_strided_copy(&src, rows, src_cols, dest_cols);
        let n_out = rows * dest_cols;
        let mut k = mt_strided_copy::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("src", pack_f32(&src), dt))
            .input(TestBuffer::from_vec(
                "src_shape",
                pack_u32_slice(&[rows as u32, dest_cols as u32]),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "src_strides",
                pack_u32_slice(&[src_cols as u32, 1u32]),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; n_out]), dt))
            .constexpr("cols", dest_cols as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(rows as u32, dest_cols as u32, 1, [1, 1, 1])
    }

    #[test_kernel(name = "mlx/strided_copy_identity_f32", dtypes = [f32], tol = 1e-6)]
    fn test_strided_copy_identity_f32(dt: DType) -> TestSetup {
        let rows = 4usize;
        let cols = 6usize;
        let src: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let expected = src.clone();
        let mut k = mt_strided_copy::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("src", pack_f32(&src), dt))
            .input(TestBuffer::from_vec(
                "src_shape",
                pack_u32_slice(&[rows as u32, cols as u32]),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "src_strides",
                pack_u32_slice(&[cols as u32, 1u32]),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; rows * cols]), dt))
            .constexpr("cols", cols as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(rows as u32, cols as u32, 1, [1, 1, 1])
    }

    #[test_kernel(name = "mlx/strided_copy_nd_2d_padded_f32", dtypes = [f32], tol = 1e-6)]
    fn test_strided_copy_nd_2d_padded_f32(dt: DType) -> TestSetup {
        let rows = 4u32;
        let cols = 4u32;
        let row_stride = 8u32;
        let shape = [rows, cols];
        let strides = [row_stride, 1u32];
        let src: Vec<f32> = (0..rows * row_stride)
            .map(|i| if i % row_stride < cols { (i as f32) + 1.0 } else { -999.0 })
            .collect();
        let expected = oracle_strided_copy_nd(&src, &shape, &strides);
        let n_out = expected.len();
        let mut k = mt_strided_copy_nd::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("src", pack_f32(&src), dt))
            .input(TestBuffer::from_vec("shape", pack_u32_slice(&shape), DType::U32))
            .input(TestBuffer::from_vec("strides", pack_u32_slice(&strides), DType::U32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; n_out]), dt))
            .constexpr("rank", shape.len() as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(n_out as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(name = "mlx/strided_copy_nd_3d_f32", dtypes = [f32], tol = 1e-6)]
    fn test_strided_copy_nd_3d_f32(dt: DType) -> TestSetup {
        let shape = [2u32, 3u32, 4u32];
        let phys = [2usize, 3, 6];
        let strides = [(phys[1] * phys[2]) as u32, phys[2] as u32, 1u32];
        let total: usize = phys.iter().product();
        let src: Vec<f32> = (0..total).map(|i| i as f32 * 0.25 - 3.0).collect();
        let expected = oracle_strided_copy_nd(&src, &shape, &strides);
        let n_out = expected.len();
        let mut k = mt_strided_copy_nd::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("src", pack_f32(&src), dt))
            .input(TestBuffer::from_vec("shape", pack_u32_slice(&shape), DType::U32))
            .input(TestBuffer::from_vec("strides", pack_u32_slice(&strides), DType::U32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; n_out]), dt))
            .constexpr("rank", shape.len() as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(n_out as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(name = "mlx/strided_copy_nd_3d_transpose_f32", dtypes = [f32], tol = 1e-6)]
    fn test_strided_copy_nd_3d_transpose_f32(dt: DType) -> TestSetup {
        let src_dims = [3usize, 4, 5];
        let cont = [(src_dims[1] * src_dims[2]) as u32, src_dims[2] as u32, 1u32];
        let shape = [src_dims[2] as u32, src_dims[1] as u32, src_dims[0] as u32];
        let strides = [cont[2], cont[1], cont[0]];
        let total: usize = src_dims.iter().product();
        let src: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let expected = oracle_strided_copy_nd(&src, &shape, &strides);
        let n_out = expected.len();
        let mut k = mt_strided_copy_nd::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("src", pack_f32(&src), dt))
            .input(TestBuffer::from_vec("shape", pack_u32_slice(&shape), DType::U32))
            .input(TestBuffer::from_vec("strides", pack_u32_slice(&strides), DType::U32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; n_out]), dt))
            .constexpr("rank", shape.len() as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(n_out as u32, 1, 1, [1, 1, 1])
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Reduce benchmarks — #[kernel] DSL vs MLX metal/reduce.metal
//!
//! Covers four reduction shapes:
//!   - **all-reduce** — `mt_all_reduce*`: one threadgroup folds the
//!     whole input to a scalar (Reduction mode).
//!   - **row-reduce** — `mt_row_reduce*`: one threadgroup per row of a
//!     `[rows, n]` input (Reduction mode).
//!   - **column-reduce** — `mt_col_reduce*`: one thread per column of a
//!     `[rows, cols]` input; each thread walks its column with a
//!     `cols`-strided `strided_reduce` (Grid3D, no threadgroup
//!     cooperation). Mirrors MLX's `col_reduce_*` family.
//!   - **segmented-reduce** — `mt_seg_reduce*`: one thread per segment
//!     of a flat input split into `n_segments` fixed-length contiguous
//!     runs; each thread contiguously folds its `seg_len`-element run
//!     (Grid3D). Suits many short segments where the row-reduce
//!     threadgroup-per-row layout would under-occupy the GPU.

use metaltile::kernel;

#[kernel]
pub fn mt_all_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, sum);
    let result = reduce_sum(acc);
    store(out[0], result);
}

#[kernel]
pub fn mt_all_reduce_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let off = 0;
    let acc = strided_reduce(inp, off, n, product);
    let result = reduce_product(acc);
    store(out[0], result);
}

#[kernel]
pub fn mt_all_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, max);
    let result = reduce_max(acc);
    store(out[0], result);
}

#[kernel]
pub fn mt_all_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let zero = 0;
    let acc = strided_reduce(inp, zero, n, min);
    let result = reduce_min(acc);
    store(out[0], result);
}

#[kernel]
pub fn mt_row_reduce<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, sum);
    let result = reduce_sum(acc);
    store(out[row], result);
}

#[kernel]
pub fn mt_row_reduce_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, product);
    let result = reduce_product(acc);
    store(out[row], result);
}

#[kernel]
pub fn mt_row_reduce_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, max);
    let result = reduce_max(acc);
    store(out[row], result);
}

#[kernel]
pub fn mt_row_reduce_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc = strided_reduce(inp, rs, re, min);
    let result = reduce_min(acc);
    store(out[row], result);
}

// ── Column reduce ────────────────────────────────────────────────────────
//
// `inp` is a row-major `[rows, cols]` matrix; `out` is `[cols]` with
// `out[c] = reduce over r of inp[r * cols + c]`. One thread per output
// column (Grid3D). Each thread folds its column with a `cols`-strided
// `strided_reduce`: offset = c, stride = cols, end = rows * cols.
//
// Grid3D mode emits the `for (_i = off; _i < end; _i += stride)` form
// (see codegen `emit_block.rs` — the `stride` field is honoured only
// outside Reduction mode), so the strided walk is correct here.
//
// Unlike the Reduction-mode `mt_row_reduce`, NO `reduce_*(acc)`
// finishing step is applied: in Grid3D the `strided_reduce` loop is
// run by a single thread and already folds the whole column. A
// `reduce_sum` here would lower to `simd_sum` and wrongly sum 32
// independent columns together.
//
// The four ops share one body; the outer `macro_rules!` wraps the
// whole `#[kernel]` declaration so the proc-macro sees concrete tokens
// (an inner macro inside the body would silently emit no IR — see
// docs/developing.md kernel-authoring hazards).

#[rustfmt::skip]
macro_rules! col_reduce_kernel {
    ($name:ident, $reduce_op:ident, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            inp: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] rows: u32,
            #[constexpr] cols: u32,
        ) {
            let col = program_id::<0>();
            if col < cols {
                let end = rows * cols;
                let acc = strided_reduce(inp, col, cols, end, $reduce_op);
                store(out[col], acc.cast::<T>());
            }
        }
    };
}

col_reduce_kernel!(mt_col_reduce, sum, "sum");
col_reduce_kernel!(mt_col_reduce_prod, product, "prod");
col_reduce_kernel!(mt_col_reduce_max, max, "max");
col_reduce_kernel!(mt_col_reduce_min, min, "min");

// ── Segmented reduce ─────────────────────────────────────────────────────
//
// `inp` is a flat buffer split into `n_segments` contiguous runs of
// `seg_len` elements; `out` is `[n_segments]` with
// `out[s] = reduce(inp[s * seg_len .. (s + 1) * seg_len])`. One thread
// per segment (Grid3D), each folding its run contiguously
// (stride = 1).
//
// This is the one-thread-per-segment counterpart to `mt_row_reduce`'s
// one-threadgroup-per-row layout: for many short segments the
// threadgroup-per-row form under-occupies the GPU (most lanes idle),
// whereas one thread per segment keeps every lane busy.

#[rustfmt::skip]
macro_rules! seg_reduce_kernel {
    ($name:ident, $reduce_op:ident, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            inp: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] n_segments: u32,
            #[constexpr] seg_len: u32,
        ) {
            let seg = program_id::<0>();
            if seg < n_segments {
                let start = seg * seg_len;
                let end = start + seg_len;
                // Grid3D: one thread folds the whole segment — no
                // `reduce_*` finishing step (see col-reduce note above).
                let acc = strided_reduce(inp, start, 1u32, end, $reduce_op);
                store(out[seg], acc.cast::<T>());
            }
        }
    };
}

seg_reduce_kernel!(mt_seg_reduce, sum, "sum");
seg_reduce_kernel!(mt_seg_reduce_prod, product, "prod");
seg_reduce_kernel!(mt_seg_reduce_max, max, "max");
seg_reduce_kernel!(mt_seg_reduce_min, min, "min");

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn dt_round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn cpu_col_reduce(
        inp: &[f32],
        rows: usize,
        cols: usize,
        init: f32,
        op: fn(f32, f32) -> f32,
    ) -> Vec<f32> {
        (0..cols).map(|c| (0..rows).map(|r| inp[r * cols + c]).fold(init, op)).collect()
    }

    fn cpu_seg_reduce(
        inp: &[f32],
        n_seg: usize,
        seg_len: usize,
        init: f32,
        op: fn(f32, f32) -> f32,
    ) -> Vec<f32> {
        (0..n_seg)
            .map(|s| inp[s * seg_len..(s + 1) * seg_len].iter().copied().fold(init, op))
            .collect()
    }

    fn make_col_setup(
        rows: usize,
        cols: usize,
        dt: DType,
        kernel_ir: fn(DType) -> metaltile::core::ir::Kernel,
        expected: Vec<f32>,
    ) -> TestSetup {
        let n_out = cols;
        let inp: Vec<f32> =
            (0..rows * cols).map(|i| dt_round(((i % 19) as f32 - 9.0) * 0.1, dt)).collect();
        let grid_x = n_out.div_ceil(256) as u32;
        TestSetup::new(kernel_ir(dt))
            .input(TestBuffer::from_vec("inp", pack(&inp, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("rows", rows as u32)
            .constexpr("cols", cols as u32)
            .grid_3d(grid_x, 1, 1, [256, 1, 1])
    }

    fn make_seg_setup(
        n_seg: usize,
        seg_len: usize,
        dt: DType,
        kernel_ir: fn(DType) -> metaltile::core::ir::Kernel,
        expected: Vec<f32>,
    ) -> TestSetup {
        let inp: Vec<f32> =
            (0..n_seg * seg_len).map(|i| dt_round(((i % 23) as f32 - 11.0) * 0.07, dt)).collect();
        let grid_x = n_seg.div_ceil(256) as u32;
        TestSetup::new(kernel_ir(dt))
            .input(TestBuffer::from_vec("inp", pack(&inp, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n_segments", n_seg as u32)
            .constexpr("seg_len", seg_len as u32)
            .grid_3d(grid_x, 1, 1, [256, 1, 1])
    }

    // ── col_reduce sum f32 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/col_reduce/sum_f32", dtypes = [f32], tol = 1e-3)]
    fn test_col_reduce_sum_f32(dt: DType) -> TestSetup {
        let (rows, cols) = (37, 100);
        let inp: Vec<f32> = (0..rows * cols).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
        let expected = cpu_col_reduce(&inp, rows, cols, 0.0, |a, b| a + b);
        make_col_setup(rows, cols, dt, mt_col_reduce::kernel_ir_for, expected)
    }

    // ── col_reduce max f32 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/col_reduce/max_f32", dtypes = [f32], tol = 1e-4)]
    fn test_col_reduce_max_f32(dt: DType) -> TestSetup {
        let (rows, cols) = (50, 70);
        let inp: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 7919) % 1000) as f32 * 0.01 - 5.0).collect();
        let expected = cpu_col_reduce(&inp, rows, cols, f32::NEG_INFINITY, f32::max);
        make_col_setup(rows, cols, dt, mt_col_reduce_max::kernel_ir_for, expected)
    }

    // ── col_reduce min f32 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/col_reduce/min_f32", dtypes = [f32], tol = 1e-4)]
    fn test_col_reduce_min_f32(dt: DType) -> TestSetup {
        let (rows, cols) = (50, 70);
        let inp: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 7919) % 1000) as f32 * 0.01 - 5.0).collect();
        let expected = cpu_col_reduce(&inp, rows, cols, f32::INFINITY, f32::min);
        make_col_setup(rows, cols, dt, mt_col_reduce_min::kernel_ir_for, expected)
    }

    // ── col_reduce prod f32 ───────────────────────────────────────────
    #[test_kernel(name = "mlx/col_reduce/prod_f32", dtypes = [f32], tol = 1e-4)]
    fn test_col_reduce_prod_f32(dt: DType) -> TestSetup {
        let (rows, cols) = (8, 40);
        let inp: Vec<f32> = (0..rows * cols).map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.02).collect();
        let expected = cpu_col_reduce(&inp, rows, cols, 1.0, |a, b| a * b);
        make_col_setup(rows, cols, dt, mt_col_reduce_prod::kernel_ir_for, expected)
    }

    // ── col_reduce sum f16 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/col_reduce/sum_f16", dtypes = [f16], tol = 5e-2)]
    fn test_col_reduce_sum_f16(dt: DType) -> TestSetup {
        let (rows, cols) = (20, 64);
        let inp: Vec<f32> =
            (0..rows * cols).map(|i| dt_round(((i % 13) as f32 - 6.0) * 0.05, dt)).collect();
        let expected = cpu_col_reduce(&inp, rows, cols, 0.0, |a, b| a + b);
        make_col_setup(rows, cols, dt, mt_col_reduce::kernel_ir_for, expected)
    }

    // ── col_reduce sum bf16 ───────────────────────────────────────────
    #[test_kernel(name = "mlx/col_reduce/sum_bf16", dtypes = [bf16], tol = 2e-1)]
    fn test_col_reduce_sum_bf16(dt: DType) -> TestSetup {
        let (rows, cols) = (20, 64);
        let inp: Vec<f32> =
            (0..rows * cols).map(|i| dt_round(((i % 13) as f32 - 6.0) * 0.05, dt)).collect();
        let expected = cpu_col_reduce(&inp, rows, cols, 0.0, |a, b| a + b);
        make_col_setup(rows, cols, dt, mt_col_reduce::kernel_ir_for, expected)
    }

    // ── seg_reduce sum f32 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/seg_reduce/sum_f32", dtypes = [f32], tol = 1e-3)]
    fn test_seg_reduce_sum_f32(dt: DType) -> TestSetup {
        let (n_seg, seg_len) = (300, 17);
        let inp: Vec<f32> = (0..n_seg * seg_len).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
        let expected = cpu_seg_reduce(&inp, n_seg, seg_len, 0.0, |a, b| a + b);
        make_seg_setup(n_seg, seg_len, dt, mt_seg_reduce::kernel_ir_for, expected)
    }

    // ── seg_reduce max f32 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/seg_reduce/max_f32", dtypes = [f32], tol = 1e-4)]
    fn test_seg_reduce_max_f32(dt: DType) -> TestSetup {
        let (n_seg, seg_len) = (128, 33);
        let inp: Vec<f32> =
            (0..n_seg * seg_len).map(|i| ((i * 6151) % 2000) as f32 * 0.005 - 5.0).collect();
        let expected = cpu_seg_reduce(&inp, n_seg, seg_len, f32::NEG_INFINITY, f32::max);
        make_seg_setup(n_seg, seg_len, dt, mt_seg_reduce_max::kernel_ir_for, expected)
    }

    // ── seg_reduce min f32 ────────────────────────────────────────────
    #[test_kernel(name = "mlx/seg_reduce/min_f32", dtypes = [f32], tol = 1e-4)]
    fn test_seg_reduce_min_f32(dt: DType) -> TestSetup {
        let (n_seg, seg_len) = (128, 33);
        let inp: Vec<f32> =
            (0..n_seg * seg_len).map(|i| ((i * 6151) % 2000) as f32 * 0.005 - 5.0).collect();
        let expected = cpu_seg_reduce(&inp, n_seg, seg_len, f32::INFINITY, f32::min);
        make_seg_setup(n_seg, seg_len, dt, mt_seg_reduce_min::kernel_ir_for, expected)
    }

    // ── seg_reduce prod f32 ───────────────────────────────────────────
    #[test_kernel(name = "mlx/seg_reduce/prod_f32", dtypes = [f32], tol = 1e-4)]
    fn test_seg_reduce_prod_f32(dt: DType) -> TestSetup {
        let (n_seg, seg_len) = (64, 12);
        let inp: Vec<f32> =
            (0..n_seg * seg_len).map(|i| 1.0 + ((i % 5) as f32 - 2.0) * 0.03).collect();
        let expected = cpu_seg_reduce(&inp, n_seg, seg_len, 1.0, |a, b| a * b);
        make_seg_setup(n_seg, seg_len, dt, mt_seg_reduce_prod::kernel_ir_for, expected)
    }

    // ── seg_reduce sum bf16 ───────────────────────────────────────────
    #[test_kernel(name = "mlx/seg_reduce/sum_bf16", dtypes = [bf16], tol = 1e-1)]
    fn test_seg_reduce_sum_bf16(dt: DType) -> TestSetup {
        let (n_seg, seg_len) = (100, 24);
        let inp: Vec<f32> =
            (0..n_seg * seg_len).map(|i| dt_round(((i % 11) as f32 - 5.0) * 0.04, dt)).collect();
        let expected = cpu_seg_reduce(&inp, n_seg, seg_len, 0.0, |a, b| a + b);
        make_seg_setup(n_seg, seg_len, dt, mt_seg_reduce::kernel_ir_for, expected)
    }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile::core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "all_reduce/sum", dtypes = [f32, f16, bf16])]
    fn bench_mt_all_reduce(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_all_reduce(
            mt_all_reduce::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ALL_REDUCE_N,
            crate::mlx::benches::ALL_REDUCE_TPG,
        )
    }

    #[bench(name = "all_reduce/prod", dtypes = [f32, f16, bf16])]
    fn bench_mt_all_reduce_prod(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_all_reduce(
            mt_all_reduce_prod::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ALL_REDUCE_N,
            crate::mlx::benches::ALL_REDUCE_TPG,
        )
    }

    #[bench(name = "all_reduce/max", dtypes = [f32, f16, bf16])]
    fn bench_mt_all_reduce_max(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_all_reduce(
            mt_all_reduce_max::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ALL_REDUCE_N,
            crate::mlx::benches::ALL_REDUCE_TPG,
        )
    }

    #[bench(name = "all_reduce/min", dtypes = [f32, f16, bf16])]
    fn bench_mt_all_reduce_min(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_all_reduce(
            mt_all_reduce_min::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ALL_REDUCE_N,
            crate::mlx::benches::ALL_REDUCE_TPG,
        )
    }

    #[bench(name = "row_reduce/sum", dtypes = [f32, f16, bf16])]
    fn bench_mt_row_reduce(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_row_reduce(
            mt_row_reduce::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ROW_REDUCE_B,
            crate::mlx::benches::ROW_REDUCE_N,
            crate::mlx::benches::ROW_REDUCE_TPG,
        )
    }

    #[bench(name = "row_reduce/prod", dtypes = [f32, f16, bf16])]
    fn bench_mt_row_reduce_prod(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_row_reduce(
            mt_row_reduce_prod::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ROW_REDUCE_B,
            crate::mlx::benches::ROW_REDUCE_N,
            crate::mlx::benches::ROW_REDUCE_TPG,
        )
    }

    #[bench(name = "row_reduce/max", dtypes = [f32, f16, bf16])]
    fn bench_mt_row_reduce_max(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_row_reduce(
            mt_row_reduce_max::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ROW_REDUCE_B,
            crate::mlx::benches::ROW_REDUCE_N,
            crate::mlx::benches::ROW_REDUCE_TPG,
        )
    }

    #[bench(name = "row_reduce/min", dtypes = [f32, f16, bf16])]
    fn bench_mt_row_reduce_min(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_row_reduce(
            mt_row_reduce_min::kernel_ir_for(dt),
            dt,
            crate::mlx::benches::ROW_REDUCE_B,
            crate::mlx::benches::ROW_REDUCE_N,
            crate::mlx::benches::ROW_REDUCE_TPG,
        )
    }
}

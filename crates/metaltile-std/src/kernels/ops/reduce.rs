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

#[kernel]
pub fn mt_col_reduce<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] rows: u32,
    #[constexpr] cols: u32,
) {
    let col = program_id::<0>();
    if col < cols {
        let end = rows * cols;
        let acc = strided_reduce(inp, col, cols, end, sum);
        store(out[col], acc.cast::<T>());
    }
}

#[kernel]
pub fn mt_col_reduce_prod<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] rows: u32,
    #[constexpr] cols: u32,
) {
    let col = program_id::<0>();
    if col < cols {
        let end = rows * cols;
        let acc = strided_reduce(inp, col, cols, end, product);
        store(out[col], acc.cast::<T>());
    }
}

#[kernel]
pub fn mt_col_reduce_max<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] rows: u32,
    #[constexpr] cols: u32,
) {
    let col = program_id::<0>();
    if col < cols {
        let end = rows * cols;
        let acc = strided_reduce(inp, col, cols, end, max);
        store(out[col], acc.cast::<T>());
    }
}

#[kernel]
pub fn mt_col_reduce_min<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] rows: u32,
    #[constexpr] cols: u32,
) {
    let col = program_id::<0>();
    if col < cols {
        let end = rows * cols;
        let acc = strided_reduce(inp, col, cols, end, min);
        store(out[col], acc.cast::<T>());
    }
}

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

#[kernel]
pub fn mt_seg_reduce<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_segments: u32,
    #[constexpr] seg_len: u32,
) {
    let seg = program_id::<0>();
    if seg < n_segments {
        let start = seg * seg_len;
        let end = start + seg_len;
        let acc = strided_reduce(inp, start, 1u32, end, sum);
        store(out[seg], acc.cast::<T>());
    }
}

#[kernel]
pub fn mt_seg_reduce_prod<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_segments: u32,
    #[constexpr] seg_len: u32,
) {
    let seg = program_id::<0>();
    if seg < n_segments {
        let start = seg * seg_len;
        let end = start + seg_len;
        let acc = strided_reduce(inp, start, 1u32, end, product);
        store(out[seg], acc.cast::<T>());
    }
}

#[kernel]
pub fn mt_seg_reduce_max<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_segments: u32,
    #[constexpr] seg_len: u32,
) {
    let seg = program_id::<0>();
    if seg < n_segments {
        let start = seg * seg_len;
        let end = start + seg_len;
        let acc = strided_reduce(inp, start, 1u32, end, max);
        store(out[seg], acc.cast::<T>());
    }
}

#[kernel]
pub fn mt_seg_reduce_min<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_segments: u32,
    #[constexpr] seg_len: u32,
) {
    let seg = program_id::<0>();
    if seg < n_segments {
        let start = seg * seg_len;
        let end = start + seg_len;
        let acc = strided_reduce(inp, start, 1u32, end, min);
        store(out[seg], acc.cast::<T>());
    }
}

/// New-syntax correctness for the reduce family.
///
/// all/row reduce are Reduction-mode (`.mode(Reduction)`, one threadgroup per
/// row); col/seg reduce are Grid3D (one thread per output). Oracles fold the
/// dtype-rounded inputs in f32. max/min are exact; sum/prod widen per dtype to
/// cover f16/bf16 accumulation order vs the f32 oracle. Inputs are kept small
/// (prod stays near 1) so the accumulation drift is bounded.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn fold(init: f32, xs: impl Iterator<Item = f32>, op: fn(f32, f32) -> f32) -> f32 {
        xs.fold(init, op)
    }

    // ── all-reduce: one threadgroup folds `n` elements → out[0] ───────────
    fn all_setup_for(
        kernel: Kernel,
        n: usize,
        vals: &[f32],
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let v = unpack_f32(&pack_f32(vals, dt), dt);
        let expected = fold(init, v.into_iter(), op);
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(vals, dt), dt))
            .input(TestBuffer::zeros("out", 1, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&[expected], dt), dt))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }

    fn sum_vals(n: usize) -> Vec<f32> { (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect() }
    fn prod_vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.001).collect()
    }
    fn ext_vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7919 % 1000) as f32) * 0.01 - 5.0).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 2.0, 16.0])]
    fn test_all_reduce_sum(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce::kernel_ir_for(dt),
            2048,
            &sum_vals(2048),
            0.0,
            |a, b| a + b,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 1.0])]
    fn test_all_reduce_prod(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce_prod::kernel_ir_for(dt),
            512,
            &prod_vals(512),
            1.0,
            |a, b| a * b,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_all_reduce_max(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce_max::kernel_ir_for(dt),
            2048,
            &ext_vals(2048),
            f32::NEG_INFINITY,
            f32::max,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_all_reduce_min(dt: DType) -> TestSetup {
        all_setup_for(
            mt_all_reduce_min::kernel_ir_for(dt),
            2048,
            &ext_vals(2048),
            f32::INFINITY,
            f32::min,
            dt,
        )
    }

    // ── row-reduce: one threadgroup per row of [rows, n] → out[row] ────────
    fn row_setup_for(
        kernel: Kernel,
        rows: usize,
        n: usize,
        per_row: &dyn Fn(usize) -> Vec<f32>,
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let mut inp = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = per_row(r);
            let rd = unpack_f32(&pack_f32(&row, dt), dt);
            expected.push(fold(init, rd.into_iter(), op));
            inp.extend_from_slice(&row);
        }
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", rows, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1.0, 8.0])]
    fn test_row_reduce_sum(dt: DType) -> TestSetup {
        row_setup_for(
            mt_row_reduce::kernel_ir_for(dt),
            4,
            1024,
            &|r| (0..1024).map(|i| ((i % 17) as f32 - 8.0) * 0.01 + r as f32 * 0.001).collect(),
            0.0,
            |a, b| a + b,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_row_reduce_max(dt: DType) -> TestSetup {
        row_setup_for(
            mt_row_reduce_max::kernel_ir_for(dt),
            4,
            1024,
            &|r| (0..1024).map(|i| ((i * 7919 % 1000) as f32) * 0.01 - 5.0 + r as f32).collect(),
            f32::NEG_INFINITY,
            f32::max,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_row_reduce_min(dt: DType) -> TestSetup {
        row_setup_for(
            mt_row_reduce_min::kernel_ir_for(dt),
            4,
            1024,
            &|r| (0..1024).map(|i| ((i * 7919 % 1000) as f32) * 0.01 - 5.0 + r as f32).collect(),
            f32::INFINITY,
            f32::min,
            dt,
        )
    }

    // ── col-reduce: Grid3D, one thread per column of [rows, cols] ─────────
    fn col_setup_for(
        kernel: Kernel,
        rows: usize,
        cols: usize,
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let inp: Vec<f32> = (0..rows * cols).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
        let id = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected: Vec<f32> =
            (0..cols).map(|c| fold(init, (0..rows).map(|r| id[r * cols + c]), op)).collect();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", cols, dt))
            .constexpr("rows", rows as u32)
            .constexpr("cols", cols as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(cols, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 5e-1])]
    fn test_col_reduce_sum(dt: DType) -> TestSetup {
        col_setup_for(mt_col_reduce::kernel_ir_for(dt), 37, 100, 0.0, |a, b| a + b, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_col_reduce_max(dt: DType) -> TestSetup {
        col_setup_for(mt_col_reduce_max::kernel_ir_for(dt), 50, 70, f32::NEG_INFINITY, f32::max, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_col_reduce_min(dt: DType) -> TestSetup {
        col_setup_for(mt_col_reduce_min::kernel_ir_for(dt), 50, 70, f32::INFINITY, f32::min, dt)
    }

    // ── seg-reduce: Grid3D, one thread per contiguous segment ─────────────
    fn seg_setup_for(
        kernel: Kernel,
        n_segments: usize,
        seg_len: usize,
        init: f32,
        op: fn(f32, f32) -> f32,
        dt: DType,
    ) -> TestSetup {
        let inp: Vec<f32> =
            (0..n_segments * seg_len).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let id = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected: Vec<f32> = (0..n_segments)
            .map(|s| fold(init, (0..seg_len).map(|j| id[s * seg_len + j]), op))
            .collect();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n_segments, dt))
            .constexpr("n_segments", n_segments as u32)
            .constexpr("seg_len", seg_len as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_segments, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 5e-1])]
    fn test_seg_reduce_sum(dt: DType) -> TestSetup {
        seg_setup_for(mt_seg_reduce::kernel_ir_for(dt), 64, 48, 0.0, |a, b| a + b, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_seg_reduce_max(dt: DType) -> TestSetup {
        seg_setup_for(mt_seg_reduce_max::kernel_ir_for(dt), 64, 48, f32::NEG_INFINITY, f32::max, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_seg_reduce_min(dt: DType) -> TestSetup {
        seg_setup_for(mt_seg_reduce_min::kernel_ir_for(dt), 64, 48, f32::INFINITY, f32::min, dt)
    }

    // ── prod: separate setups with inputs near 1.0 so the running product
    // stays O(1) (a `((i%19)-9)*0.1`-style input underflows to 0 over a long
    // reduction). Modest reduction lengths keep f16/bf16 well-conditioned.
    fn prod_inputs(n: usize) -> Vec<f32> {
        (0..n).map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.05).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-2, 1e-1])]
    fn test_col_reduce_prod(dt: DType) -> TestSetup {
        let (rows, cols) = (8usize, 40usize);
        let inp = prod_inputs(rows * cols);
        let id = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected: Vec<f32> =
            (0..cols).map(|c| (0..rows).map(|r| id[r * cols + c]).product()).collect();
        TestSetup::new(mt_col_reduce_prod::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", cols, dt))
            .constexpr("rows", rows as u32)
            .constexpr("cols", cols as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(cols, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-2, 1e-1])]
    fn test_seg_reduce_prod(dt: DType) -> TestSetup {
        let (n_segments, seg_len) = (64usize, 12usize);
        let inp = prod_inputs(n_segments * seg_len);
        let id = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected: Vec<f32> =
            (0..n_segments).map(|s| (0..seg_len).map(|j| id[s * seg_len + j]).product()).collect();
        TestSetup::new(mt_seg_reduce_prod::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n_segments, dt))
            .constexpr("n_segments", n_segments as u32)
            .constexpr("seg_len", seg_len as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_segments, 256)
    }
}

/// New-syntax benchmarks for the reduce family (vs MLX `metal/reduce.metal`).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::utils::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    const ALL_REDUCE_N: usize = 16 * 1024 * 1024;
    const ROW_REDUCE_ROWS: usize = 4096;
    const ROW_REDUCE_N: usize = 4096;

    // all-reduce: one threadgroup folds N elements to a scalar. Attaches the MLX
    // `metal/reduce.metal` `all_reduce_<sub><tn>` reference for an A/B perf +
    // correctness comparison. MLX `all_reduce(in, out, in_size, row_size)` folds
    // a single block when `in_size == row_size == N` (matching MT's single
    // threadgroup); both are `size_t` (8-byte) scalars.
    //
    // `inp` is shared by name with the reference (the runner injects the MT
    // bytes), so both kernels reduce identical `Positive` data. `tol_floor` is
    // the legacy reduction floor — large for sum/prod because MT accumulates in
    // f32 while MLX accumulates in the (lossy) reduce dtype over 16M elements.
    fn all_ref(
        kernel: Kernel,
        dt: DType,
        mlx_sub: &str,
        tol_floor: f32,
        f32_only_ref: bool,
    ) -> BenchSetup {
        let n = ALL_REDUCE_N;
        let tn = mlx_tname(dt);
        let base = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(input_buffer("inp", n, dt, InputDomain::Tiny))
            .buffer(BenchBuffer::zeros("out", 1, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved((n * dt.size_bytes()) as u64);
        // `sum` folds 16M elements: MT accumulates in f32 but MLX accumulates in
        // the (lossy) reduce dtype, so for f16/bf16 the two legitimately diverge
        // by thousands — no meaningful tolerance. Compare only f32, where both
        // are faithful; f16/bf16 stay perf-only rows.
        if f32_only_ref && dt != DType::F32 {
            return base;
        }
        base.with_reference(
                RefKernel::new(
                    format!("all_reduce_{mlx_sub}{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/reduce.metal")),
                )
                // in[0] shared by name with the MT `inp` above (placeholder).
                .buffer(BenchBuffer::zeros("inp", n, dt))
                .buffer(BenchBuffer::zeros("out", 1, dt).output())
                // in_size + row_size are both `size_t` (8 bytes) = N → single block.
                .buffer(BenchBuffer::from_vec("in_size", (n as u64).to_le_bytes().to_vec(), DType::U32))
                .buffer(BenchBuffer::from_vec("row_size", (n as u64).to_le_bytes().to_vec(), DType::U32))
                .grid(Grid::new_3d(1, 1, 1, [256, 1, 1]))
                .tol(dtype_tol(dt).max(tol_floor)),
            )
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_all_sum(dt: DType) -> BenchSetup {
        all_ref(mt_all_reduce::kernel_ir_for(dt), dt, "sum", 256.0, true)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_all_prod(dt: DType) -> BenchSetup {
        all_ref(mt_all_reduce_prod::kernel_ir_for(dt), dt, "prod", 1024.0, false)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_all_max(dt: DType) -> BenchSetup {
        all_ref(mt_all_reduce_max::kernel_ir_for(dt), dt, "max", 0.0, false)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_all_min(dt: DType) -> BenchSetup {
        all_ref(mt_all_reduce_min::kernel_ir_for(dt), dt, "min", 0.0, false)
    }

    // row-reduce: one threadgroup per row. Attaches the MLX
    // `metal/reduce.metal` `row_reduce_simple_<sub><tn>` reference. That kernel
    // indexes the row via `gid.y`, so its dispatch grid puts the rows on the Y
    // axis (`[1, rows, 1]`) — unlike the MT kernel which uses `program_id::<0>()`
    // (X axis). `reduction_size` is `size_t` (8 bytes) = N; `out_size` is
    // `int64_t` (8 bytes) = rows.
    fn row_ref(kernel: Kernel, dt: DType, mlx_sub: &str, tol_floor: f32) -> BenchSetup {
        let (rows, n) = (ROW_REDUCE_ROWS, ROW_REDUCE_N);
        let tn = mlx_tname(dt);
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(input_buffer("inp", rows * n, dt, InputDomain::Tiny))
            .buffer(BenchBuffer::zeros("out", rows, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((rows * n * dt.size_bytes()) as u64)
            .with_reference(
                RefKernel::new(
                    format!("row_reduce_simple_{mlx_sub}{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/reduce.metal")),
                )
                // in[0] shared by name with the MT `inp` above (placeholder).
                .buffer(BenchBuffer::zeros("inp", rows * n, dt))
                .buffer(BenchBuffer::zeros("out", rows, dt).output())
                // reduction_size (size_t, 8 bytes) = N; out_size (int64_t, 8 bytes) = rows.
                .buffer(BenchBuffer::from_vec("reduction_size", (n as u64).to_le_bytes().to_vec(), DType::U32))
                .buffer(BenchBuffer::from_vec("out_size", (rows as u64).to_le_bytes().to_vec(), DType::U32))
                // MLX `row_reduce_simple` reads the row from `gid.y` → rows on Y.
                .grid(Grid::new_3d(1, rows as u32, 1, [256, 1, 1]))
                .tol(dtype_tol(dt).max(tol_floor)),
            )
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_row_sum(dt: DType) -> BenchSetup {
        row_ref(mt_row_reduce::kernel_ir_for(dt), dt, "sum", 128.0)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_row_prod(dt: DType) -> BenchSetup {
        row_ref(mt_row_reduce_prod::kernel_ir_for(dt), dt, "prod", 32.0)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_row_max(dt: DType) -> BenchSetup {
        row_ref(mt_row_reduce_max::kernel_ir_for(dt), dt, "max", 0.0)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_row_min(dt: DType) -> BenchSetup {
        row_ref(mt_row_reduce_min::kernel_ir_for(dt), dt, "min", 0.0)
    }

    // col-reduce: Grid3D, one thread per output column of a [rows, cols] matrix.
    fn col_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let (rows, cols) = (4096usize, 4096usize);
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", rows * cols, dt))
            .buffer(BenchBuffer::zeros("out", cols, dt).output())
            .constexpr("rows", rows as u32)
            .constexpr("cols", cols as u32)
            .grid_1d(cols, 256)
            .bytes_moved((rows * cols * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_col_sum(dt: DType) -> BenchSetup { col_b(mt_col_reduce::kernel_ir_for(dt), dt) }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_col_prod(dt: DType) -> BenchSetup { col_b(mt_col_reduce_prod::kernel_ir_for(dt), dt) }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_col_max(dt: DType) -> BenchSetup { col_b(mt_col_reduce_max::kernel_ir_for(dt), dt) }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_col_min(dt: DType) -> BenchSetup { col_b(mt_col_reduce_min::kernel_ir_for(dt), dt) }

    // seg-reduce: Grid3D, one thread per contiguous segment.
    fn seg_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let (n_segments, seg_len) = (65536usize, 256usize);
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("inp", n_segments * seg_len, dt))
            .buffer(BenchBuffer::zeros("out", n_segments, dt).output())
            .constexpr("n_segments", n_segments as u32)
            .constexpr("seg_len", seg_len as u32)
            .grid_1d(n_segments, 256)
            .bytes_moved((n_segments * seg_len * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_seg_sum(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce::kernel_ir_for(dt), dt) }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_seg_prod(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce_prod::kernel_ir_for(dt), dt) }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_seg_max(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce_max::kernel_ir_for(dt), dt) }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_seg_min(dt: DType) -> BenchSetup { seg_b(mt_seg_reduce_min::kernel_ir_for(dt), dt) }
}

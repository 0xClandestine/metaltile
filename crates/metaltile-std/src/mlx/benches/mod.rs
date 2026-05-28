//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Shared bench-setup helpers for `metaltile-std` kernel bench functions.
//!
//! Each helper constructs a [`BenchSetup`] for one canonical dispatch class.
//! Kernel bench functions (emitted as `pub mod kernel_benches` blocks inside
//! each kernel file) call these helpers to avoid repeating boilerplate.
//!
//! # Standard dispatch parameters
//!
//! | Class          | Default N      | Default TPG |
//! |----------------|---------------|-------------|
//! | Unary/Binary   | 64 MiB elems  | 256         |
//! | AllReduce      | 64 MiB elems  | 256         |
//! | RowReduce      | 1024 × 4096   | 256         |
//! | RowNorm        | 1024 × 4096   | 1024/256    |
//! | MatVec         | 4096 × 4096   | 512         |
//! | Sort/Scan      | 1024 rows × N | 256         |

use metaltile_core::{
    DType,
    bench::{BenchBuffer, BenchSetup},
    ir::Kernel,
};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const ELEMENTWISE_N: usize = 64 * 1024 * 1024;
pub const ELEMENTWISE_TPG: u32 = 256;
pub const ALL_REDUCE_N: usize = 64 * 1024 * 1024;
pub const ALL_REDUCE_TPG: u32 = 256;
pub const ROW_REDUCE_B: usize = 1024;
pub const ROW_REDUCE_N: usize = 4096;
pub const ROW_REDUCE_TPG: u32 = 256;
pub const ROW_NORM_B: usize = 1024;
pub const ROW_NORM_N: usize = 4096;
pub const ROW_NORM_TPG: u32 = 256;
pub const MAT_VEC_B: usize = 4096;
pub const MAT_VEC_N: usize = 4096;
pub const MAT_VEC_TPG: u32 = 512;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Elementwise unary: `(a, out)` — 1D grid.
///
/// Buffer names: `a` (random input), `out` (zero-initialised output).
pub fn bench_unary(kernel: Kernel, dt: DType, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("a", n, dt))
        .buffer(BenchBuffer::zeros("out", n, dt).output())
        .grid_1d(n, tpg)
}

/// Elementwise binary: `(a, b, out)` — 1D grid.
///
/// `a_name`, `b_name`, `out_name` let callers match exact kernel parameter names
/// (e.g. `"c"` for `vector_add`, `"gate"/"up"` for `mt_swiglu`).
pub fn bench_binary(
    kernel: Kernel,
    dt: DType,
    n: usize,
    tpg: u32,
    a_name: &'static str,
    b_name: &'static str,
    out_name: &'static str,
) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random(a_name, n, dt))
        .buffer(BenchBuffer::random(b_name, n, dt))
        .buffer(BenchBuffer::zeros(out_name, n, dt).output())
        .grid_1d(n, tpg)
}

/// All-reduce: single threadgroup folds N elements to a scalar.
///
/// Buffer names: `inp` (random), `out` (zero, 1 element).
/// Constexpr: `n`.
pub fn bench_all_reduce(kernel: Kernel, dt: DType, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("inp", n, dt))
        .buffer(BenchBuffer::zeros("out", 1, dt).output())
        .constexpr("n", n as u32)
        .grid_3d(1, 1, 1, [tpg, 1, 1])
}

/// Row-reduce: one threadgroup per row of a `rows × n` input.
///
/// Buffer names: `inp` (random, `rows × n` elems), `out` (zero, `rows` elems).
/// Constexpr: `n`.
pub fn bench_row_reduce(kernel: Kernel, dt: DType, rows: usize, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("inp", rows * n, dt))
        .buffer(BenchBuffer::zeros("out", rows, dt).output())
        .constexpr("n", n as u32)
        .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
}

/// Ternary select: `(cond: u8, on_true, on_false, out)` — 1D grid.
pub fn bench_select(kernel: Kernel, dt: DType, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("cond", n, DType::U8))
        .buffer(BenchBuffer::random("on_true", n, dt))
        .buffer(BenchBuffer::random("on_false", n, dt))
        .buffer(BenchBuffer::zeros("out", n, dt).output())
        .grid_1d(n, tpg)
}

/// Binary-two: `(a, b, mut c, mut d)` — two inputs, two outputs, 1D grid.
pub fn bench_binary_two(kernel: Kernel, dt: DType, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("a", n, dt))
        .buffer(BenchBuffer::random("b", n, dt))
        .buffer(BenchBuffer::zeros("c", n, dt).output())
        .buffer(BenchBuffer::zeros("d", n, dt).output())
        .grid_1d(n, tpg)
}

/// ArgReduce: finds index of max/min in N elements.
///
/// Buffer names: `inp` (random), `out` (output index u32, 1 element).
pub fn bench_arg_reduce(kernel: Kernel, dt: DType, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("inp", n, dt))
        .buffer(BenchBuffer::zeros("out", 1, DType::U32).output())
        .constexpr("n", n as u32)
        .grid_3d(1, 1, 1, [tpg, 1, 1])
}

/// Softmax / logsumexp-style row-norm: `(inp, out, n)`.
///
/// Input is `rows × n`, output is `rows × n` (softmax) or `rows` (logsumexp).
pub fn bench_row_norm(
    kernel: Kernel,
    dt: DType,
    rows: usize,
    n: usize,
    tpg: u32,
    out_per_row: usize,
) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("inp", rows * n, dt))
        .buffer(BenchBuffer::zeros("out", rows * out_per_row, dt).output())
        .constexpr("n", n as u32)
        .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
}

/// RMS-norm: `(x, w, out, eps_buf, n)`.
///
/// `x` is `rows × n`, `w` is `n`, `out` is `rows × n`, `eps_buf` is a
/// single f32 scalar.
pub fn bench_rms_norm(kernel: Kernel, dt: DType, rows: usize, n: usize, tpg: u32) -> BenchSetup {
    let eps_bytes: Vec<u8> = 1e-5f32.to_le_bytes().to_vec();
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("x", rows * n, dt))
        .buffer(BenchBuffer::random("w", n, dt))
        .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
        .buffer(BenchBuffer::from_vec("eps_buf", eps_bytes, DType::F32))
        .constexpr("n", n as u32)
        .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
}

/// Layer-norm: `(x, w, b, out, eps_buf, n)`.
///
/// `x` is `rows × n`, `w` / `b` are `n`, `out` is `rows × n`.
pub fn bench_layer_norm(
    kernel: Kernel,
    dt: DType,
    rows: usize,
    n: usize,
    tpg: u32,
) -> BenchSetup {
    let eps_bytes: Vec<u8> = 1e-5f32.to_le_bytes().to_vec();
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("x", rows * n, dt))
        .buffer(BenchBuffer::random("w", n, dt))
        .buffer(BenchBuffer::zeros("b", n, dt))
        .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
        .buffer(BenchBuffer::from_vec("eps_buf", eps_bytes, DType::F32))
        .constexpr("n", n as u32)
        .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
}

/// Matrix-vector multiply: `(mat, vec, out, k)`.
///
/// `mat` is `rows × k`, `vec` is `k`, `out` is `rows`.
pub fn bench_mat_vec(kernel: Kernel, dt: DType, rows: usize, k: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("mat", rows * k, dt))
        .buffer(BenchBuffer::random("vec", k, dt))
        .buffer(BenchBuffer::zeros("out", rows, dt).output())
        .constexpr("k", k as u32)
        .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
}

/// Masked matrix-vector multiply: `(mat, vec, mask, out, k)`.
pub fn bench_mat_vec_masked(
    kernel: Kernel,
    dt: DType,
    rows: usize,
    k: usize,
    tpg: u32,
) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("mat", rows * k, dt))
        .buffer(BenchBuffer::random("vec", k, dt))
        .buffer(BenchBuffer::random("mask", k, dt))
        .buffer(BenchBuffer::zeros("out", rows, dt).output())
        .constexpr("k", k as u32)
        .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
}

/// Sort / scan per-row: `(inp, out, n)` with y-axis batch dimension.
///
/// Grid is `[1, rows, 1] × [tpg, 1, 1]` matching the scan/sort dispatch
/// convention used in `metaltile-std`.
pub fn bench_row_op_y(kernel: Kernel, dt: DType, rows: usize, n: usize, tpg: u32) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("inp", rows * n, dt))
        .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
        .constexpr("n", n as u32)
        .grid_3d(1, rows as u32, 1, [tpg, 1, 1])
}

/// Strided copy: `(src, out, cols)` — 2D grid over `m × n` with padding.
///
/// The `#[strided]` attribute on `src` makes the runner add shape/stride
/// buffers automatically. Here we provide the data buffer + output only.
pub fn bench_strided_copy(
    kernel: Kernel,
    dt: DType,
    m: usize,
    n: usize,
    tpg: u32,
) -> BenchSetup {
    BenchSetup::new(kernel)
        .buffer(BenchBuffer::random("src", m * n, dt))
        .buffer(BenchBuffer::zeros("out", m * n, dt).output())
        .constexpr("cols", n as u32)
        .grid_2d(m as u32, 1, [tpg, 1])
}

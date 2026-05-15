//! `BenchSpec` — declarative kernel benchmark descriptors.
//!
//! Each `#[bench_kernel(...)]` annotation on a `#[kernel]` fn generates one
//! `BenchSpec` and registers it via `inventory::submit!`. The bench runner
//! iterates `inventory::iter::<BenchSpec>`, sorts by `(op, subop)`, then calls
//! `spec.run(runner, dt)` per dtype.
//!
//! For the 10 "simple" class types (Unary, Binary, AllReduce, RowReduce,
//! Arange, BinaryTwo, Select, RowNorm, MatVec, MatVecMasked), the macro
//! generates a `ShapeSpec` and sets `dispatch = BenchDispatch::Generic`.
//! The generic runner in `spec_runner.rs` handles all of these uniformly.
//!
//! The 9 "complex" class types (Sort, Scan, ArgReduce, Random, FpQuantized,
//! QuantizedMatVec, Rope, Attention, StridedCopy) keep specialized runners.

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};

// ── Default sizes ─────────────────────────────────────────────────────────────

pub const ELEMENTWISE_N_BENCH: usize = 64 * 1024 * 1024;
pub const ELEMENTWISE_N_CHECK: usize = 2_048;
pub const ELEMENTWISE_TPG: usize = 256;

pub const BINARY_TPG: usize = 1_024;
pub const BINARY_N_PER_THREAD: usize = 2;

pub const ALL_REDUCE_N: usize = 64 * 1024 * 1024;
pub const ALL_REDUCE_N_CHECK: usize = 16_384;
pub const ALL_REDUCE_TPG: usize = 256;

pub const ROW_REDUCE_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
pub const ROW_REDUCE_CHECK_B: usize = 8;
pub const ROW_REDUCE_CHECK_N: usize = 512;
pub const ROW_REDUCE_TPG: usize = 256;

pub const ARANGE_N: usize = 64 * 1024 * 1024;
pub const ARANGE_N_CHECK: usize = 4_096;
pub const ARANGE_TPG: usize = 1_024;

pub const BINARY_TWO_TPG: usize = 1_024;

pub const SELECT_TPG: usize = 256;

// ── Single-dtype shorthands ───────────────────────────────────────────────────

pub const F32_ONLY: &[DType] = &[DType::F32];
pub const F16_ONLY: &[DType] = &[DType::F16];

// ── Dim: runtime size expression ─────────────────────────────────────────────

/// A size dimension, resolved at runtime given (n, b).
/// `b` is the row/batch count; `n` is the column/width count.
#[derive(Clone, Copy, Debug)]
pub enum Dim {
    N,    // n
    B,    // b
    BxN,  // b * n
    One,  // 1
}

impl Dim {
    pub fn resolve(self, n: usize, b: usize) -> usize {
        match self {
            Dim::N => n,
            Dim::B => b,
            Dim::BxN => b * n,
            Dim::One => 1,
        }
    }
}

// ── BufInit: how to initialise a buffer ──────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum BufInit {
    Zeros,
    Half,         // 0.5 repeating
    Signed,       // [-3, -1.5, -0.5, 0, 0.25, 0.75, 1.5, 3] cycling
    Positive,     // 0.25, 0.5, … 4.0 cycling
    Unit,         // all 1.0
    Fill(f32),    // constant value (e.g. Fill(1e-5) for eps)
    AltZeroOne,   // alternating 0.0 / 1.0 (Select condition)
}

impl BufInit {
    pub fn generate(self, n: usize) -> Vec<f32> {
        match self {
            BufInit::Zeros => vec![0.0; n],
            BufInit::Half => vec![0.5; n],
            BufInit::Signed => (0..n)
                .map(|i| match i % 8 {
                    0 => -3.0,
                    1 => -1.5,
                    2 => -0.5,
                    3 => 0.0,
                    4 => 0.25,
                    5 => 0.75,
                    6 => 1.5,
                    _ => 3.0,
                })
                .collect(),
            BufInit::Positive => (0..n).map(|i| 0.25 + (i % 16) as f32 * 0.25).collect(),
            BufInit::Unit => vec![1.0; n],
            BufInit::Fill(v) => vec![v; n],
            BufInit::AltZeroOne => (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
        }
    }
}

// ── TensorBufSpec ─────────────────────────────────────────────────────────────

/// Specification for one tensor kernel parameter.
/// Order matches `kernel.params` (tensor params only, no constexprs).
#[derive(Clone, Copy, Debug)]
pub struct TensorBufSpec {
    /// Number of elements (resolved via n/b at runtime).
    pub count: Dim,
    /// How to fill the buffer with data.
    pub init: BufInit,
    /// Override element dtype (None → use the benchmark DType).
    /// Use Some(DType::F32) for params like `eps_buf: Tensor<f32>`.
    pub dtype_override: Option<DType>,
}

// ── ScalarBufSpec: GPU-only constexpr scalar buffers ─────────────────────────

/// One constexpr GPU buffer (`constant uint& n [[buffer(k)]]`).
/// These are appended after tensor param buffers in dispatch order.
#[derive(Clone, Copy, Debug)]
pub enum ScalarBufSpec {
    U32N,  // runner.buffer_u32(n as u32)
    U32B,  // runner.buffer_u32(b as u32)
    U64N,  // runner.buffer_u64(n as u64)
    U64B,  // runner.buffer_u64(b as u64)
    I64B,  // runner.buffer_i64(b as i64)
}

// ── DispatchGrid: how to compute [gx, gy, gz] from n/b/tpg ──────────────────

#[derive(Clone, Copy, Debug)]
pub enum DispatchGrid {
    DivCeilN,   // [n.div_ceil(tpg), 1, 1]
    DivCeilN2,  // [n.div_ceil(tpg*2), 1, 1]  — Binary MLX N_PER_THREAD=2
    RowsB,      // [b, 1, 1]
    RowsBY,     // [1, b, 1]  — RowReduce MLX
    Single,     // [1, 1, 1]
}

impl DispatchGrid {
    pub fn eval(self, n: usize, b: usize, tpg: usize) -> [usize; 3] {
        match self {
            DispatchGrid::DivCeilN => [n.div_ceil(tpg), 1, 1],
            DispatchGrid::DivCeilN2 => [n.div_ceil(tpg * 2), 1, 1],
            DispatchGrid::RowsB => [b, 1, 1],
            DispatchGrid::RowsBY => [1, b, 1],
            DispatchGrid::Single => [1, 1, 1],
        }
    }
}

// ── MlxArg: one argument in the MLX reference kernel call ────────────────────

#[derive(Clone, Copy, Debug)]
pub enum MlxArg {
    /// Reuse tensor buf at position i (input bufs only).
    TensorBuf(usize),
    /// Create a fresh zeroed output buffer of the same size as tensor buf i.
    FreshOut(usize),
    /// Inline U32 = n.
    U32N,
    /// Inline U64 = n.
    U64N,
    /// Inline U64 = b.
    U64B,
    /// Inline I64 = b.
    I64B,
    /// An 8-byte zeroed dummy buffer.
    Zeros8,
    /// A 1-byte-per-element alternating 0/1 buffer of N booleans.
    BoolAltN,
    /// A constant uint32 buffer with the given fixed value (e.g. w_stride=1).
    U32V(u32),
}

// ── ShapeSpec: full spec for one benchmark shape ──────────────────────────────

pub struct ShapeSpec {
    /// Human-readable label (e.g. "N=64M" or "B=1024 N=4096").
    pub label: &'static str,

    // ── Bench dimensions ────────────────────────────────────────────────────
    pub n: usize,
    pub b: usize,  // rows (or 1 for elementwise)

    // ── Check dimensions (smaller for speed) ────────────────────────────────
    pub check_n: usize,
    pub check_b: usize,

    // ── Kernel dispatch ─────────────────────────────────────────────────────
    /// Kernel execution mode (sets how MSL is generated).
    pub mode: KernelMode,
    pub tpg: usize,
    pub grid: DispatchGrid,

    // ── Buffer layout ────────────────────────────────────────────────────────
    /// One entry per `kernel.params` tensor parameter, in order.
    pub tensor_bufs: &'static [TensorBufSpec],
    /// Constexpr scalar GPU buffers appended after tensor params.
    pub scalar_bufs: &'static [ScalarBufSpec],
    /// Constexpr values fed to the interpreter (name, dim).
    pub cexprs: &'static [(&'static str, Dim)],

    // ── Output size ──────────────────────────────────────────────────────────
    /// Number of output elements per shape (default: Dim::BxN for row ops).
    /// Used for bytes formula and interpreter output map lookup.
    pub out_elems: Dim,

    // ── Throughput formula ───────────────────────────────────────────────────
    /// How many times the input data is read (for bytes formula).
    pub reads: usize,
    /// Bytes formula: (n, b, reads, out_count, elem_bytes) → total_bytes.
    pub bytes_fn: fn(usize, usize, usize, usize, usize) -> usize,

    // ── MLX reference ────────────────────────────────────────────────────────
    /// Arguments for the MLX ref kernel (None → no ref benchmark).
    pub mlx_args: Option<&'static [MlxArg]>,
    /// MLX grid override (None → same as MT grid).
    pub mlx_grid: Option<DispatchGrid>,
    /// MLX tpg override (0 → same as tpg).
    pub mlx_tpg: usize,
}

// ── BenchDispatch ─────────────────────────────────────────────────────────────

/// Dispatch strategy for running a benchmark.
pub enum BenchDispatch {
    /// Generic data-driven runner using `BenchSpec.shapes`.
    Generic,
    // Complex ops with specialized runners:
    Sort { b: usize, n: usize, tpg: usize },
    Scan { shapes: &'static [(usize, usize)], tpg: usize },
    ArgReduce { n: usize, check_n: usize, tpg: usize },
    Random { n: usize, tpg: usize },
    FpQuantized { n: usize, tpg: usize },
    QuantizedMatVec { shapes: &'static [(usize, usize)], group_size: usize, tpg: usize },
    Rope { b: usize, h: usize, l: usize, d: usize, n_per_group: usize },
    Attention { shapes: &'static [(usize, usize, usize)], tpg: usize },
    StridedCopy { m: usize, n: usize, pad: usize },
}

// ── BenchSpec ─────────────────────────────────────────────────────────────────

pub struct BenchSpec {
    pub op: &'static str,
    pub subop: &'static str,
    pub kernel_name: &'static str,
    pub kernel_ir: fn(DType) -> Kernel,
    pub dtypes: &'static [DType],
    pub tol: f32,
    pub metal_file: Option<&'static str>,
    /// MLX Metal source (used by both Generic and complex dispatches).
    pub mlx_src: Option<&'static str>,
    /// MLX kernel name pattern; `{tn}` → MLX type name.
    pub mlx_pattern: Option<&'static str>,
    /// Shape specs for Generic dispatch (empty slice for complex dispatch).
    pub shapes: &'static [ShapeSpec],
    pub dispatch: BenchDispatch,
}

inventory::collect!(BenchSpec);

// ── Standard bytes formulas ───────────────────────────────────────────────────

/// bytes = n * eb * (reads + 1)  — elementwise with 1 output write
pub fn bytes_elementwise(n: usize, _b: usize, reads: usize, _out: usize, eb: usize) -> usize {
    n * eb * (reads + 1)
}

/// bytes = b * n * eb * reads + out * eb  — row-wise reduction
pub fn bytes_row_op(n: usize, b: usize, reads: usize, out: usize, eb: usize) -> usize {
    b * n * eb * reads + out * eb
}

/// bytes = b * n * eb + n * eb + out * eb  — mat-vec (mat + vec + result)
pub fn bytes_mat_vec(n: usize, b: usize, _reads: usize, out: usize, eb: usize) -> usize {
    (b * n + n + out) * eb
}

/// bytes = b * n * eb + 2 * n * eb + out * eb  — masked mat-vec
pub fn bytes_mat_vec_masked(n: usize, b: usize, _reads: usize, out: usize, eb: usize) -> usize {
    (b * n + 2 * n + out) * eb
}

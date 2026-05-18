//! `BenchSpec` — declarative kernel benchmark descriptors.
//!
//! Each `#[bench_kernel(...)]` annotation on a `#[kernel]` fn generates one
//! `BenchSpec` and registers it via `inventory::submit!`. The bench runner
//! iterates `inventory::iter::<BenchSpec>`, sorts by `(op, subop)`, then calls
//! `run_spec(spec, runner, dt)` per dtype.
//!
//! All ops flow through the single `run_generic` runner. Complex ops (rope,
//! sort, attention, …) express their custom logic through fn-pointer fields
//! in `ShapeSpec` / `BenchSpec`. The `#[bench_kernel]` macro wires up
//! class-level fn helpers based on the `class` attribute.

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};

use crate::{
    bench_types::EquivResult,
    runner::{CompiledKernel, GpuBuffer, GpuRunner},
};

// ── Default sizes ───────────────────────────────────────────────────────

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

// ── Single-dtype shorthands ──────────────────────────────────────────────

pub const F32_ONLY: &[DType] = &[DType::F32];
pub const F16_ONLY: &[DType] = &[DType::F16];

// ── Dim: runtime size expression ─────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum Dim {
    N,   // n
    B,   // b
    BxN, // b * n
    One, // 1
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

// ── BufInit ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum BufInit {
    Zeros,
    Half,
    Signed,
    Positive,
    Unit,
    Fill(f32),
    AltZeroOne,
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
            BufInit::Unit =>
                (0..n).map(|i| [-0.9f32, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9][i % 7]).collect(),
            BufInit::Fill(v) => vec![v; n],
            BufInit::AltZeroOne => (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
        }
    }
}

// ── TensorBufSpec ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct TensorBufSpec {
    pub count: Dim,
    pub init: BufInit,
    pub dtype_override: Option<DType>,
}

// ── ScalarBufSpec ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum ScalarBufSpec {
    U32N,
    U32B,
    U64N,
    U64B,
    I64B,
}

// ── DispatchGrid ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum DispatchGrid {
    DivCeilN,
    DivCeilN2,
    RowsB,
    RowsBY,
    Single,
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

// ── MlxArg ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum MlxArg {
    TensorBuf(usize),
    FreshOut(usize),
    U32N,
    U64N,
    U64B,
    I64B,
    Zeros8,
    BoolAltN,
    U32V(u32),
}

// ── ShapeSpec ────────────────────────────────────────────────────────────

pub struct ShapeSpec {
    pub label: &'static str,
    pub n: usize,
    pub b: usize,
    pub check_n: usize,
    pub check_b: usize,
    pub mode: KernelMode,
    pub tpg: usize,
    pub grid: DispatchGrid,
    pub tensor_bufs: &'static [TensorBufSpec],
    pub scalar_bufs: &'static [ScalarBufSpec],
    pub cexprs: &'static [(&'static str, Dim)],
    pub out_elems: Dim,
    pub reads: usize,
    pub bytes_fn: fn(usize, usize, usize, usize, usize) -> usize,
    pub mlx_args: Option<&'static [MlxArg]>,
    pub mlx_grid: Option<DispatchGrid>,
    pub mlx_tpg: usize,

    // ── fn-pointer hooks (None = use static data path) ───────────────────

    /// Extra class-specific shape dims (e.g. rope: `[l, n_per_group, …]`).
    /// Access by per-class stable index convention.
    pub extra: [usize; 8],

    /// Build MT buffers.  When `Some`, replaces `tensor_bufs`+`scalar_bufs`.
    /// `is_bench=false` → check sizes; `is_bench=true` → perf sizes.
    pub mt_bufs_fn: Option<fn(&GpuRunner, &ShapeSpec, bool, DType) -> Vec<GpuBuffer>>,

    /// Index of the output buffer in the `mt_bufs_fn` result.
    /// When `None`, falls back to the first `is_output` param.
    pub mt_out_idx: Option<usize>,

    /// Custom MT dispatch grid.  `fn(shape, is_bench, tpg) → [x, y, z]`
    pub mt_grid_fn: Option<fn(&ShapeSpec, bool, usize) -> [usize; 3]>,

    /// Build MLX buffers.  When `Some`, replaces `mlx_args`.
    pub mlx_bufs_fn: Option<fn(&GpuRunner, &ShapeSpec, bool, DType) -> Vec<GpuBuffer>>,

    /// Index of the output buffer in the `mlx_bufs_fn` result.
    pub mlx_out_idx: usize,

    /// Custom MLX dispatch grid.  `fn(shape, is_bench, tpg) → [x, y, z]`
    pub mlx_grid_fn: Option<fn(&ShapeSpec, bool, usize) -> [usize; 3]>,

    /// Number of output elements to read back.
    /// When `Some`, overrides the `out_elems.resolve()` calculation.
    pub out_n_fn: Option<fn(&ShapeSpec, bool) -> usize>,
}

// ── BenchDispatch ────────────────────────────────────────────────────────
// All ops use the Generic runner. Complex ops provide fn-pointers in
// ShapeSpec/BenchSpec set by the #[bench_kernel] macro.
//
// Scan, Attention, QuantizedMatVec still carry shapes data because
// they iterate over compile-time shapes arrays. They will migrate
// to fn-pointers in a follow-up.

pub enum BenchDispatch {
    Generic,
    Scan { shapes: &'static [(usize, usize)], tpg: usize },
    Attention { shapes: &'static [(usize, usize, usize)], tpg: usize },
    QuantizedMatVec { shapes: &'static [(usize, usize)], group_size: usize, tpg: usize },
    /// Two-pass SDPA decode: pass1+pass2 chained dispatch.
    SdpaVector2Pass {
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        batch: usize,
        blocks: usize,
        pass2_kernel_name: &'static str,
        pass2_kernel_ir: fn(DType) -> Kernel,
    },
}

impl BenchDispatch {
    pub fn default_mode(&self, shapes: &[ShapeSpec]) -> KernelMode {
        match self {
            BenchDispatch::Generic =>
                shapes.first().map(|s| s.mode).unwrap_or(KernelMode::Elementwise),
            BenchDispatch::Scan { .. }
            | BenchDispatch::Attention { .. }
            | BenchDispatch::QuantizedMatVec { .. }
            | BenchDispatch::SdpaVector2Pass { .. } => KernelMode::Reduction,
        }
    }
}

// ── BenchSpec ───────────────────────────────────────────────────────────

pub struct BenchSpec {
    pub op: &'static str,
    pub subop: &'static str,
    pub kernel_name: &'static str,
    pub kernel_ir: fn(DType) -> Kernel,
    pub dtypes: &'static [DType],
    pub tol: f32,
    pub mlx_src: Option<&'static str>,
    pub mlx_pattern: Option<&'static str>,
    pub shapes: &'static [ShapeSpec],
    pub dispatch: BenchDispatch,
    /// Optional explicit kernel mode override.
    pub kernel_mode: Option<KernelMode>,

    /// Custom MLX compile fn.  `fn(runner, src, dt) → Option<CompiledKernel>`
    /// When `Some`, replaces the standard `compile(src, mlx_pattern)` call.
    pub mlx_compile_fn: Option<fn(&GpuRunner, &str, DType) -> Option<CompiledKernel>>,

    /// Custom correctness check.  `fn(ref_vals, mt_vals, tol, chunk_n) → EquivResult`
    /// `chunk_n` = shape.check_n — useful for sort (chunk size) and ignored otherwise.
    /// When `None`, uses `check_equiv` (element-wise abs error + cosine sim).
    pub check_fn: Option<fn(&[f32], &[f32], f32, usize) -> EquivResult>,
}

inventory::collect!(BenchSpec);

// ── Standard bytes formulas ──────────────────────────────────────────────

pub fn bytes_elementwise(n: usize, _b: usize, reads: usize, _out: usize, eb: usize) -> usize {
    n * eb * (reads + 1)
}

pub fn bytes_row_op(n: usize, b: usize, reads: usize, out: usize, eb: usize) -> usize {
    b * n * eb * reads + out * eb
}

pub fn bytes_mat_vec(n: usize, b: usize, _reads: usize, out: usize, eb: usize) -> usize {
    (b * n + n + out) * eb
}

pub fn bytes_mat_vec_masked(n: usize, b: usize, _reads: usize, out: usize, eb: usize) -> usize {
    (b * n + 2 * n + out) * eb
}

/// Select: cond is always 1-byte bool (matching MLX v_Select{T} interface).
pub fn bytes_select(n: usize, _b: usize, _reads: usize, _out: usize, eb: usize) -> usize {
    n + 3 * n * eb // cond(1 byte) + on_true(eb) + on_false(eb) + out(eb)
}

pub fn effective_mode(spec: &BenchSpec) -> metaltile_core::ir::KernelMode {
    spec.kernel_mode.unwrap_or_else(|| spec.dispatch.default_mode(spec.shapes))
}

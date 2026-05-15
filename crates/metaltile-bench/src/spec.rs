//! `BenchSpec` — declarative kernel benchmark descriptors.
//!
//! Each `#[bench_kernel(...)]` annotation on a `#[kernel]` fn generates one
//! `BenchSpec` and registers it via `inventory::submit!`.  The bench runner
//! iterates `inventory::iter::<BenchSpec>`, sorts by `(op, subop)` for
//! subop-primary display order, then calls `spec.run(runner, dt)` per dtype.
//!
//! Canonical sizes — override inside `BenchClass` variants when needed.

use metaltile_core::{dtype::DType, ir::Kernel};

// ── Default sizes ─────────────────────────────────────────────────────────────

pub const ELEMENTWISE_N_BENCH: usize = 64 * 1024 * 1024;
pub const ELEMENTWISE_N_CHECK: usize = 2_048;
pub const ELEMENTWISE_TPG: usize = 256;

pub const BINARY_TPG: usize = 1_024;
pub const BINARY_N_PER_THREAD: usize = 2; // MLX processes 2 elements per thread

pub const ALL_REDUCE_N: usize = 64 * 1024 * 1024;
pub const ALL_REDUCE_N_CHECK: usize = 16_384;
pub const ALL_REDUCE_TPG: usize = 256;

pub const ROW_REDUCE_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
pub const ROW_REDUCE_CHECK_B: usize = 8;
pub const ROW_REDUCE_CHECK_N: usize = 512;
pub const ROW_REDUCE_TPG: usize = 256;

// ── InputGen ──────────────────────────────────────────────────────────────────

/// How to generate f32 inputs for correctness checks and warm-up.
#[derive(Clone, Copy)]
pub enum InputGen {
    /// Cycling: -3, -1.5, -0.5, 0, 0.25, 0.75, 1.5, 3 — covers negative values.
    Signed,
    /// 0.25 + (i % 16) × 0.25 — strictly positive (for log, sqrt, etc.).
    Positive,
    /// Constant 0.5 — safe for all ops.
    Half,
    /// Constant 1.0.
    Unit,
}

impl InputGen {
    pub fn generate(self, n: usize) -> Vec<f32> {
        match self {
            InputGen::Signed => (0..n)
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
            InputGen::Positive => (0..n).map(|i| 0.25 + (i % 16) as f32 * 0.25).collect(),
            InputGen::Half => vec![0.5f32; n],
            InputGen::Unit => vec![1.0f32; n],
        }
    }
}

// ── BenchClass ────────────────────────────────────────────────────────────────

/// Execution class — drives dispatch geometry, buffer layout, and bytes/s calc.
pub enum BenchClass {
    /// Single-input elementwise: `out[i] = f(a[i])`.
    Unary {
        cpu: fn(f32) -> f32,
        inputs: InputGen,
        mlx_src: Option<&'static str>,
        /// Kernel name pattern; `{tn}` is replaced with the MLX type name
        /// (e.g. `"v_Exp{tn}{tn}"` → `"v_Expfloat32float32"`).
        mlx_pattern: Option<&'static str>,
    },
    /// Two-input elementwise: `out[i] = f(a[i], b[i])`.
    Binary {
        cpu: fn(f32, f32) -> f32,
        inputs_a: InputGen,
        inputs_b: InputGen,
        /// MLX dispatches `N_PER_THREAD=2` elements per thread; used for ref grid.
        ref_n_per_thread: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    /// Reduce entire flat array to a single scalar.
    AllReduce {
        cpu: fn(&[f32]) -> f32,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    /// Reduce each row of a 2-D tensor independently.
    RowReduce {
        shapes: &'static [(usize, usize)], // (B, N) pairs
        cpu: fn(&[f32]) -> f32,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
}

// ── BenchSpec ─────────────────────────────────────────────────────────────────

/// Fully describes how to benchmark one MetalTile kernel.
///
/// Register via `#[bench_kernel(...)]` (generates `inventory::submit!`) or
/// manually with `inventory::submit! { BenchSpec { ... } }`.
pub struct BenchSpec {
    /// Op group name — used for blank-line grouping in the table (e.g. `"unary"`).
    pub op: &'static str,
    /// Sub-operation label — displayed as `"op (subop)"` (e.g. `"exp"`).
    pub subop: &'static str,
    /// Metal kernel name as it appears in the compiled library (e.g. `"mt_exp"`).
    pub kernel_name: &'static str,
    /// Returns the kernel IR for a given dtype. Points to the generated
    /// `mt_exp::kernel_ir_for` function from `#[kernel]`.
    pub kernel_ir: fn(DType) -> Kernel,
    /// Dtypes to benchmark (usually `FLOAT_DTYPES`).
    pub dtypes: &'static [DType],
    /// Maximum absolute error for correctness check.
    pub tol: f32,
    /// Execution class — drives how the kernel is dispatched and measured.
    pub class: BenchClass,
}

inventory::collect!(BenchSpec);

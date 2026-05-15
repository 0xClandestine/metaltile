//! `BenchSpec` — declarative kernel benchmark descriptors.
//!
//! Each `#[bench_kernel(...)]` annotation on a `#[kernel]` fn generates one
//! `BenchSpec` and registers it via `inventory::submit!`.  The bench runner
//! iterates `inventory::iter::<BenchSpec>`, sorts by `(op, subop)` for
//! subop-primary display order, then calls `spec.run(runner, dt)` per dtype.
//!
//! Correctness is provided by `metaltile_interp::Interpreter` — the kernel IR
//! is executed on CPU and compared against GPU output. No hand-written cpu
//! reference functions needed.

use metaltile_core::{dtype::DType, ir::Kernel};

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

// ── InputGen ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum InputGen {
    Signed,
    Positive,
    Half,
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

// ── ExtraInput ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum ExtraInput {
    WeightPerCol { val: f32 },
    BiasPerCol { val: f32 },
    ScalarF32 { val: f32 },
}

// ── BenchClass ────────────────────────────────────────────────────────────────

/// Execution class — drives dispatch geometry, buffer layout, and bytes/s calc.
/// No cpu reference functions; correctness comes from the interpreter.
pub enum BenchClass {
    Unary {
        inputs: InputGen,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    Binary {
        inputs_a: InputGen,
        inputs_b: InputGen,
        ref_n_per_thread: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    AllReduce {
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    RowReduce {
        shapes: &'static [(usize, usize)],
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    Arange {
        start: f32,
        step: f32,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    BinaryTwo {
        inputs_a: InputGen,
        inputs_b: InputGen,
    },
    Select {
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    RowNorm {
        shapes: &'static [(usize, usize)],
        tpg: usize,
        reads: usize,
        out_elements: usize,
        extra: &'static [ExtraInput],
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
        mlx_extra_slots: usize,
    },
    // ── Phase 4-6 classes ──────────────────────────────────────────────────
    Sort {
        b: usize,
        n: usize,
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    Scan {
        shapes: &'static [(usize, usize)],
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    ArgReduce {
        n: usize,
        check_n: usize,
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    Random {
        n: usize,
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    FpQuantized {
        n: usize,
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    MatVec {
        shapes: &'static [(usize, usize)],
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    MatVecMasked {
        shapes: &'static [(usize, usize)],
        tpg: usize,
    },
    QuantizedMatVec {
        shapes: &'static [(usize, usize)],
        group_size: usize,
        tpg: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
    /// RoPE rotation — f16 only, Grid3D dispatch.
    /// MLX ref name is hardcoded in spec_runner (needs bool function constants).
    Rope {
        b: usize,
        h: usize,
        l: usize,
        d: usize,
        n_per_group: usize,
        mlx_src: Option<&'static str>,
    },
    /// Scaled dot-product attention — f32/f16, Reduction dispatch.
    /// MLX ref names are hardcoded in spec_runner (needs bool function constants).
    Attention {
        shapes: &'static [(usize, usize, usize)],
        tpg: usize,
        mlx_src: Option<&'static str>,
    },
    StridedCopy {
        m: usize,
        n: usize,
        pad: usize,
        mlx_src: Option<&'static str>,
        mlx_pattern: Option<&'static str>,
    },
}

// ── BenchSpec ─────────────────────────────────────────────────────────────────

pub struct BenchSpec {
    pub op: &'static str,
    pub subop: &'static str,
    pub kernel_name: &'static str,
    pub kernel_ir: fn(DType) -> Kernel,
    pub dtypes: &'static [DType],
    pub tol: f32,
    pub class: BenchClass,
    pub metal_file: Option<&'static str>,
}

inventory::collect!(BenchSpec);

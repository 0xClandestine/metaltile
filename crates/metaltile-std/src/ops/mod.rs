//! Op-kernel benches/registrations.
//!
//! Organised by whether the kernel has an MLX side-by-side comparison
//! today (see `ops/mlx/` and `ops/ffai/` for the split criteria).

pub mod ffai;
pub mod mlx;

pub use crate::bench_types::{
    CorrectnessStatus, DEFAULT_MIN_COSINE_SIM, DType, DtypeCtx, EquivResult, EquivTolerance,
    FLOAT_DTYPE_STRS, FLOAT_DTYPES, INTEGER_DTYPES, OpBench, OpResult, SuitePrinter, dtype_label,
    dtype_tol, dtype_tol_reduce, elem_bytes, generate_elementwise_msl, generate_reduction_msl,
    mlx_tname, print_suite, quantize_roundtrip, set_result_reporter, validate_results,
};

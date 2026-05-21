//! Scatter along an axis — contiguous form of MLX's `scatter_axis`.
//!
//! `out[o, indices[o, a, i], i] = updates[o, a, i]` — each update
//! element is written to a row-`indices`-selected slot of `out`. One
//! thread per update element. `out` is pre-initialized by the caller
//! (typically a copy of the source) and the kernel overwrites the
//! scattered slots.
//!
//! Layout (row-contiguous):
//!   updates: [outer, axis_upd,  inner]  T
//!   indices: [outer, axis_upd,  inner]  u32
//!   out:     [outer, axis_size, inner]  T  (pre-initialized)
//!
//! Assignment (no-reduce) form: distinct `indices` are required for a
//! deterministic result — colliding indices race, matching MLX
//! `scatter_axis` with `reduce = None`. The general strided + reducing
//! kernel is a follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, one thread per update element over `outer*axis_upd*inner`.
//!
//! Codegen-only; correctness pinned by
//! `tests/scatter_axis_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[kernel]
pub fn mt_scatter_axis<T>(
    updates: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] axis_upd: u32,
    #[constexpr] axis_size: u32,
    #[constexpr] inner: u32,
) {
    let idx = program_id::<0>();
    let i = idx - (idx / inner) * inner;
    let o = idx / (axis_upd * inner);
    let scattered = load(indices[idx]);
    let out_off = (o * axis_size + scattered) * inner + i;
    store(out[out_off], load(updates[idx]));
}

inventory::submit! {
    BenchSpec {
        op: "indexing",
        subop: "scatter_axis",
        kernel_name: "mt_scatter_axis",
        kernel_ir: mt_scatter_axis::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Grid3D),
    }
}

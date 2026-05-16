//! fence benchmarks — metal/fence.metal  (MLX, Apache-2.0)
//!
//! Synchronisation primitives for multi-kernel pipelines:
//!   input_coherent  — ensure input buffer visibility
//!   fence_update    — write a fence counter
//!   fence_wait      — spin-wait on a fence counter
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   These kernels use `volatile coherent(system) device` memory
//!   qualifiers and `metal::atomic_thread_fence` with system-scope
//!   memory ordering. The DSL has no primitives for atomics, device
//!   memory fences, or volatile/system-coherent memory annotations.
//!   These are infrastructure kernels, not computational ops.

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/metal/fence.metal"));

pub fn bench_fence(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }

use crate::ops::{KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "fence",
        mt_kernel: "—".into(),
        metal_file: "fence.metal",
        ref_spec: RefSpec::None(
            "GPU memory barrier — handled by Metal command encoder; no MT kernel needed",
        ),
        dtypes: &[],
    }]
}

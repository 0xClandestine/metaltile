//! steel_conv_general benchmarks — metal/steel/conv/steel_conv_general.metal  (MLX, Apache-2.0)
//!
//! General N-D convolution fallback using SIMD matrix multiply instructions.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!("../../../metal/steel/conv/steel_conv_general.metal");

pub fn bench_conv_general(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }

use crate::ops::{KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "steel/conv/steel_conv_general",
        mt_kernel: "—".into(),
        metal_file: "steel/conv/steel_conv_general.metal",
        ref_spec: RefSpec::None("general convolution not yet in MT bench"),
        dtypes: &[],
    }]
}

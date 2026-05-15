//! steel_conv_3d benchmarks — metal/steel/conv/steel_conv_3d.metal  (MLX, Apache-2.0)
//!
//! Tiled 3D convolution using SIMD matrix multiply instructions.
//!
//! TODO: implement benchmarks

use crate::{ops::OpResult, runner::GpuRunner};

static _SRC: &str = include_str!("../../../metal/steel/conv/steel_conv_3d.metal");

pub fn bench_conv3d(_runner: &GpuRunner) -> Vec<OpResult> { vec![] }

use crate::ops::{KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "steel/conv/steel_conv_3d",
        mt_kernel: "—".into(),
        metal_file: "steel/conv/steel_conv_3d.metal",
        ref_spec: RefSpec::None("3D implicit GEMM convolution not yet in MT bench"),
        dtypes: &[],
    }]
}

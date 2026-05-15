//! Op-level benchmark modules.
//!
//! The folder structure mirrors src/metal/ exactly:
//!
//!   ops/arange.rs            ↔  metal/arange.metal
//!   ops/arg_reduce.rs        ↔  metal/arg_reduce.metal
//!   ops/binary.rs            ↔  metal/binary.metal
//!   ops/binary_two.rs        ↔  metal/binary_two.metal
//!   ops/conv.rs              ↔  metal/conv.metal
//!   ops/copy.rs              ↔  metal/copy.metal
//!   ops/fence.rs             ↔  metal/fence.metal
//!   ops/fft.rs               ↔  metal/fft.metal
//!   ops/fp_quantized.rs      ↔  metal/fp_quantized.metal
//!   ops/fp_quantized_nax.rs  ↔  metal/fp_quantized_nax.metal
//!   ops/gemv.rs              ↔  metal/gemv.metal
//!   ops/gemv_masked.rs       ↔  metal/gemv_masked.metal
//!   ops/layer_norm.rs        ↔  metal/layer_norm.metal
//!   ops/logsumexp.rs         ↔  metal/logsumexp.metal
//!   ops/quantized.rs         ↔  metal/quantized.metal
//!   ops/quantized_nax.rs     ↔  metal/quantized_nax.metal
//!   ops/random.rs            ↔  metal/random.metal
//!   ops/reduce.rs            ↔  metal/reduce.metal
//!   ops/rms_norm.rs          ↔  metal/rms_norm.metal
//!   ops/rope.rs              ↔  metal/rope.metal
//!   ops/scaled_dot_product_attention.rs ↔ metal/scaled_dot_product_attention.metal
//!   ops/scan.rs              ↔  metal/scan.metal
//!   ops/softmax.rs           ↔  metal/softmax.metal
//!   ops/sort.rs              ↔  metal/sort.metal
//!   ops/ternary.rs           ↔  metal/ternary.metal
//!   ops/unary.rs             ↔  metal/unary.metal
//!   ops/steel/attn/          ↔  metal/steel/attn/
//!   ops/steel/conv/          ↔  metal/steel/conv/
//!   ops/steel/gemm/          ↔  metal/steel/gemm/

pub mod arange;
pub mod arg_reduce;
pub mod binary;
pub mod binary_two;
pub mod conv;
pub mod copy;
pub mod fence;
pub mod fft;
pub mod fp_quantized;
#[cfg(feature = "nax")]
pub mod fp_quantized_nax;
pub mod gemv;
pub mod gemv_masked;
pub mod layer_norm;
pub mod logsumexp;
pub mod quantized;
#[cfg(feature = "nax")]
pub mod quantized_nax;
pub mod random;
pub mod reduce;
pub mod rms_norm;
pub mod rope;
pub mod scaled_dot_product_attention;
pub mod scan;
mod shared;
pub mod softmax;
pub mod sort;
pub mod steel;
pub mod strided;
pub mod ternary;
pub mod unary;

pub use arange::bench_arange;
pub use arg_reduce::bench_arg_reduce;
pub use binary::{bench_binary_ops, bench_elementwise};
pub use binary_two::bench_binary_two;
pub use copy::bench_copy;
pub use fp_quantized::bench_fp_quantized;
pub use gemv::bench_gemv;
pub use gemv_masked::bench_gemv_masked;
pub use layer_norm::bench_layer_norm;
pub use logsumexp::bench_logsumexp;
pub use quantized::bench_quantized;
pub use random::bench_random;
pub use reduce::bench_reduce;
pub use rms_norm::bench_rms_norm;
pub use rope::bench_rope;
pub use scaled_dot_product_attention::{bench_sdpa_vector, bench_sdpa_vector_f16};
pub use scan::bench_scan;
pub use shared::{
    CorrectnessStatus,
    DEFAULT_MIN_COSINE_SIM,
    DType,
    DtypeCtx,
    EquivResult,
    EquivTolerance,
    FLOAT_DTYPE_STRS,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    KernelSpec,
    OpBench,
    OpResult,
    RefSpec,
    SuitePrinter,
    bench_all_dtypes,
    bench_gbps,
    buffer_typed,
    check_equiv,
    check_equiv_with,
    dtype_label,
    dtype_tol,
    dtype_tol_reduce,
    elem_bytes,
    generate_elementwise_msl,
    generate_reduction_msl,
    mlx_tname,
    print_suite,
    quantize_roundtrip,
    read_typed,
    set_result_reporter,
    validate_results,
    zeros_typed,
};
pub(crate) use shared::{run_f16_once_as_f32, run_f32_once, run_typed_once, to_gflops};
pub use softmax::bench_softmax;
pub use sort::bench_sort;
pub use steel::gemm::{
    bench_matmul_fp16,
    bench_matmul_gather,
    bench_matmul_masked,
    bench_matmul_segmented,
};
pub use strided::bench_strided;
pub use ternary::bench_select;
pub use unary::bench_all_unary;

/// Collect coverage specs from every op module.
///
/// Used by the `kernel_table` binary to cross-reference MetalTile kernels
/// against their MLX Metal reference counterparts without requiring a GPU.
pub fn all_kernel_specs() -> Vec<KernelSpec> {
    let mut specs = Vec::new();
    specs.extend(unary::kernel_specs());
    specs.extend(binary::kernel_specs());
    specs.extend(binary_two::kernel_specs());
    specs.extend(copy::kernel_specs());
    specs.extend(arange::kernel_specs());
    specs.extend(ternary::kernel_specs());
    specs.extend(softmax::kernel_specs());
    specs.extend(rms_norm::kernel_specs());
    specs.extend(layer_norm::kernel_specs());
    specs.extend(logsumexp::kernel_specs());
    specs.extend(reduce::kernel_specs());
    specs.extend(gemv::kernel_specs());
    specs.extend(gemv_masked::kernel_specs());
    specs.extend(rope::kernel_specs());
    specs.extend(scaled_dot_product_attention::kernel_specs());
    specs.extend(scan::kernel_specs());
    specs.extend(arg_reduce::kernel_specs());
    specs.extend(sort::kernel_specs());
    specs.extend(random::kernel_specs());
    specs.extend(fp_quantized::kernel_specs());
    specs.extend(quantized::kernel_specs());
    specs.extend(strided::kernel_specs());
    specs.extend(steel::gemm::steel_gemm_fused::kernel_specs());
    specs.extend(steel::gemm::steel_gemm_gather::kernel_specs());
    specs.extend(steel::gemm::steel_gemm_masked::kernel_specs());
    specs.extend(steel::gemm::steel_gemm_segmented::kernel_specs());
    specs.extend(steel::gemm::steel_gemm_splitk::kernel_specs());
    specs.extend(steel::attn::steel_attention::kernel_specs());
    specs.extend(steel::conv::steel_conv::kernel_specs());
    specs.extend(steel::conv::steel_conv_3d::kernel_specs());
    specs.extend(steel::conv::steel_conv_general::kernel_specs());
    specs.extend(conv::kernel_specs());
    specs.extend(fft::kernel_specs());
    specs.extend(fence::kernel_specs());
    specs
}

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
pub(crate) use shared::{run_f16_once_as_f32, run_typed_once, to_gflops};
pub use steel::gemm::{
    bench_matmul_fp16,
    bench_matmul_gather,
    bench_matmul_masked,
    bench_matmul_segmented,
};
/// Collect coverage specs from every op module.
///
/// Used by the `kernel_table` binary to cross-reference MetalTile kernels
/// against their MLX Metal reference counterparts without requiring a GPU.
pub fn all_kernel_specs() -> Vec<KernelSpec> {
    let mut specs = Vec::new();

    // Inventory-registered ops (ported to #[bench_kernel]) — derive coverage
    // from the macro annotations so op files need no kernel_specs() function.
    for spec in ::inventory::iter::<crate::spec::BenchSpec> {
        let Some(metal_file) = spec.metal_file else { continue };
        let ref_spec = match &spec.class {
            crate::spec::BenchClass::Unary { mlx_pattern, .. }
            | crate::spec::BenchClass::Binary { mlx_pattern, .. }
            | crate::spec::BenchClass::AllReduce { mlx_pattern, .. }
            | crate::spec::BenchClass::RowReduce { mlx_pattern, .. }
            | crate::spec::BenchClass::Arange { mlx_pattern, .. }
            | crate::spec::BenchClass::Select { mlx_pattern, .. }
            | crate::spec::BenchClass::Sort { mlx_pattern, .. }
            | crate::spec::BenchClass::Scan { mlx_pattern, .. }
            | crate::spec::BenchClass::ArgReduce { mlx_pattern, .. }
            | crate::spec::BenchClass::Random { mlx_pattern, .. }
            | crate::spec::BenchClass::FpQuantized { mlx_pattern, .. }
            | crate::spec::BenchClass::MatVec { mlx_pattern, .. }
            | crate::spec::BenchClass::QuantizedMatVec { mlx_pattern, .. }
            | crate::spec::BenchClass::StridedCopy { mlx_pattern, .. }
            | crate::spec::BenchClass::RowNorm { mlx_pattern, .. } => match mlx_pattern {
                Some(p) => RefSpec::Format(p),
                None => RefSpec::None("no MLX reference"),
            },
            crate::spec::BenchClass::BinaryTwo { .. }
            | crate::spec::BenchClass::MatVecMasked { .. } => RefSpec::None("no MLX equivalent"),
            crate::spec::BenchClass::Rope { .. } => RefSpec::Literal("rope_float16"),
            crate::spec::BenchClass::Attention { .. } =>
                RefSpec::Literal("sdpa_vector_float_128_128"),
        };
        specs.push(KernelSpec {
            op: spec.op,
            mt_kernel: spec.kernel_name.into(),
            metal_file,
            ref_spec,
            dtypes: FLOAT_DTYPE_STRS,
        });
    }
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

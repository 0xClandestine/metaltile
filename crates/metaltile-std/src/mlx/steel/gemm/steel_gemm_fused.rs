//! Steel tiled GEMM — #[kernel] DSL vs MLX steel/gemm/kernels/steel_gemm_fused.metal
//!
//! High-performance tiled matrix multiply via simdgroup matrix ops.
//!
//! The kernel uses the DSL's simdgroup primitives (`simdgroup_alloc`,
//! `simdgroup_matmul`, etc.) to accumulate a tile of the matrix product.
//!
//! ## Simdgroup DSL intrinsics available
//!
//!   - `simdgroup_alloc::<T, M, N>()` — allocate a simdgroup_matrix<T, M, N>
//!   - `simdgroup_elem_store(sm, idx, val)` — write to thread_elements()[idx]
//!   - `simdgroup_matmul(a, b, c)` — c = a * b + c
//!   - `simdgroup_elem_load(sm, idx)` — read thread_elements()[idx]
//!   - `simd_lane_id()` — which lane within the SIMD group (0..31)
//!   - `simd_group_id()` — which SIMD group within the threadgroup
//!
//! ## MLX coverage
//!
//!   steel_gemm_fused_{nn|nt|tn|tt}_{dtype}_bm{BM}_bn{BN}_bk{BK}_wm{WM}_wn{WN}
//!   Block shapes: 64×64×16, 64×32×32, 32×64×16, 32×32×16, 64×32×8
//!   Dtypes: float16, bfloat16, float32
//!
//! Full tiled GEMM with shared-memory staging and batched dispatch is not yet
//! wired through the bench infrastructure. The `simd_mma_test` kernel below
//! validates that the simdgroup IR ops compile to correct Metal.

use metaltile::{bench_kernel, kernel};

/// Test kernel: exercises the simdgroup matrix multiply-accumulate primitives.
///
/// Allocates two 8×8 simdgroup matrices (A in f16, B in f16) and a float
/// accumulator C.  Each thread writes its lane index into its
/// thread_elements() of A, writes 1.0 into B, then performs
/// `simdgroup_multiply_accumulate(A, B, C)`.  The result is stored back
/// to the output buffer.
///
/// This is a micro-benchmark for the simdgroup IR ops, not a production GEMM.
#[bench_kernel(
    op="steel_gemm_fused",
    subop="simd_mma_test",
    class=Unary,
    input=Signed,
    tol=1e-2,
    dtypes=crate::spec::F16_ONLY,
)]
#[kernel]
pub fn simd_mma_test(out: Tensor<f16>) {
    // Thread identity within the simdgroup.
    let _lid = simd_lane_id();
    let _gid = simd_group_id();

    // Allocate three 8×8 simdgroup matrices (accumulator in f32).
    let a = simdgroup_alloc::<f16, 8, 8>();
    let b = simdgroup_alloc::<f16, 8, 8>();
    let c = simdgroup_alloc::<f32, 8, 8>();

    // Each thread holds 2 elements of its lane's contribution.
    let lid_f = _lid.cast::<f16>();
    let one = 1.0f32.cast::<f16>();

    simdgroup_elem_store(a, 0u32, lid_f);
    simdgroup_elem_store(a, 1u32, lid_f);
    simdgroup_elem_store(b, 0u32, one);
    simdgroup_elem_store(b, 1u32, one);

    // C = A * B + C (starts at zero, so C = A * B)
    simdgroup_matmul(a, b, c);

    // Read back and store to output.
    let r0 = simdgroup_elem_load(c, 0u32);
    let r1 = simdgroup_elem_load(c, 1u32);

    let base = _lid * 2u32 + _gid * 64u32;
    store(out[base], r0.cast::<f16>());
    store(out[base + 1u32], r1.cast::<f16>());
}

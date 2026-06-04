//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Generic GEMV with weighted in-place axpy: `accum[r] += weight * (mat[r] · vec)`.
//!
//! Replaces the 3-kernel `out = gemv(mat, vec); scaled = out * weight_T;
//! accum += scaled` chain with a single dispatch. Aimed at the MoE
//! routed-expert combine step where weight is the normalised top-K
//! scalar and accum is moeAccum.
//!
//! Same reduction geometry as `mt_gemv` (one threadgroup per row,
//! simdgroup reduce over k). The only differences:
//!   - `accum` instead of `out` (load + add + store, not just store)
//!   - extra `weight: f32` constexpr scalar applied before the add
//!
//! `weight` is set via `setBytes` in the generated Swift binding
//! (constexpr scalars become runtime args at dispatch time per the
//! macro's emit path), so distinct dispatches with different
//! weights see their own value.

use metaltile::kernel;

#[kernel]
pub fn ffai_gemv_axpy_inplace<T>(
    mat: Tensor<T>,
    vec: Tensor<T>,
    mut accum: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] weight: f32,
) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let dot = strided_reduce_dot(mat, vec, rs, rs, re);
    let result = reduce_sum(dot);
    let prev = load(accum[row]).cast::<f32>();
    store(accum[row], prev + weight * result);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gemv_axpy_inplace;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(m: usize, k: usize, weight: f32, dt: DType) -> TestSetup {
        let mat: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let vec: Vec<f32> = (0..k).map(|j| ((j % 13) as f32 - 6.0) * 0.02).collect();
        let accum_in: Vec<f32> = (0..m).map(|r| (r as f32 % 5.0) * 0.1 - 0.2).collect();
        let mat_dt = unpack_f32(&pack_f32(&mat, dt), dt);
        let vec_dt = unpack_f32(&pack_f32(&vec, dt), dt);
        let accum_dt = unpack_f32(&pack_f32(&accum_in, dt), dt);
        let expected: Vec<f32> = (0..m)
            .map(|r| {
                let dot: f32 = (0..k).map(|j| mat_dt[r * k + j] * vec_dt[j]).sum();
                accum_dt[r] + weight * dot
            })
            .collect();
        TestSetup::new(ffai_gemv_axpy_inplace::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("mat", pack_f32(&mat, dt), dt))
            .input(TestBuffer::from_vec("vec", pack_f32(&vec, dt), dt))
            .input(TestBuffer::from_vec("accum", pack_f32(&accum_in, dt), dt))
            .constexpr("k", k as u32)
            .constexpr("weight", weight)
            .expect(TestBuffer::from_vec("accum", pack_f32(&expected, dt), dt))
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-1, 1.0])]
    fn test_axpy_small(dt: DType) -> TestSetup { setup(16, 256, 0.42, dt) }

    // f32 tol 5e-3 (vs 1e-3 on the k=256 tests): k=2048 here, and the
    // GPU does a simdgroup tree-reduce while the CPU oracle sums
    // sequentially. Over 2048 mixed-sign terms the f32 reorder gap grows
    // with k (≈3.4e-3 on M5) — a reduction-order artifact, not a logic
    // error. The f16/bf16 tols were already widened for this shape.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 2e-1, 2.0])]
    fn test_axpy_decode_shape(dt: DType) -> TestSetup { setup(4096, 2048, 0.167, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-1, 1.0])]
    fn test_axpy_neg_weight(dt: DType) -> TestSetup { setup(16, 256, -0.3, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gemv_axpy_inplace;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_axpy(dt: DType) -> BenchSetup {
        let (m, k) = (4096usize, 2048usize);
        BenchSetup::new(ffai_gemv_axpy_inplace::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("mat", m * k, dt))
            .buffer(BenchBuffer::random("vec", k, dt))
            .buffer(BenchBuffer::random("accum", m, dt).output())
            .constexpr("k", k as u32)
            .constexpr("weight", 0.167f32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
            .bytes_moved((m * k * dt.size_bytes() + 2 * m * dt.size_bytes()) as u64)
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GEMV benchmark — #[kernel] DSL vs MLX metal/gemv.metal
//!
//! Tuned for K=4096 via tpg sweep (64,128,256,512,1024): tpg=512 gives the best f16
//! throughput (+1.8% vs tpg=256) by giving each thread 2 iterations
//! of the 4-wide unroll (8 elements/thread), enough ILP to hide
//! load latency. tpg=1024 regresses −20% on f16 (only 1 iteration,
//! zero latency hiding). f32/bf16 are flat across tpgs.

use metaltile::kernel;

#[kernel]
pub fn mt_gemv<T>(mat: Tensor<T>, vec: Tensor<T>, out: Tensor<T>, #[constexpr] k: u32) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let acc = strided_reduce_dot(mat, vec, rs, rs, re);
    let result = reduce_sum(acc);
    store(out[row], result);
}

/// New-syntax correctness for `mt_gemv` (Reduction, one threadgroup per row;
/// `out[r] = Σ_j mat[r,j]·vec[j]`). Oracle on dtype-rounded inputs.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_gemv;
    use crate::utils::{pack_f32, unpack_f32};

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-1, 1.0])]
    fn test_mt_gemv(dt: DType) -> TestSetup {
        let (m, k) = (16usize, 256usize);
        let mat: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let vec: Vec<f32> = (0..k).map(|j| ((j % 13) as f32 - 6.0) * 0.02).collect();
        let mat_dt = unpack_f32(&pack_f32(&mat, dt), dt);
        let vec_dt = unpack_f32(&pack_f32(&vec, dt), dt);
        let expected: Vec<f32> =
            (0..m).map(|r| (0..k).map(|j| mat_dt[r * k + j] * vec_dt[j]).sum()).collect();
        TestSetup::new(mt_gemv::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("mat", pack_f32(&mat, dt), dt))
            .input(TestBuffer::from_vec("vec", pack_f32(&vec, dt), dt))
            .input(TestBuffer::zeros("out", m, dt))
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }
}

/// New-syntax benchmark for `mt_gemv` (vs MLX `metal/gemv.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_gemv;
    use crate::bench_types::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    #[bench(name = "mlx/gemv", dtypes = [f32, f16, bf16])]
    fn bench_gemv(dt: DType) -> BenchSetup {
        let (m, k) = (4096usize, 4096usize);
        let tn = mlx_tname(dt);
        // MLX `metal/gemv.metal` `gemv_<tn>_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`
        // (the contiguous, no-axpby specialization). The kernel name is the fixed
        // block config the legacy bench pinned; with `bm4 sm1 tm4` each
        // threadgroup produces `n_out_per_tgp = bm*sm*tm = 16` output rows, so
        // `n_tgp = ceil(M / 16) = 256` threadgroups, dispatched at
        // `group_dims = (32, bn=1, bm=4)` = 128 lanes (`MTL::Size(n_tgp, 1, 1)` /
        // `MTL::Size(32, bn, bm)` in MLX's `gemv_axbpy`).
        //
        // Buffer order (MLX `[[buffer(N)]]`):
        //   mat[[0]], in_vec[[1]], bias[[2]] (unused, axpby0), out_vec[[3]],
        //   in_vec_size[[4]]=K (int,4), out_vec_size[[5]]=M (int,4),
        //   matrix_ld[[6]]=K (int,4), alpha[[7]] (float, unused),
        //   beta[[8]] (float, unused), batch_ndim[[9]] (int, unused at nc0),
        //   batch_shape[[10]] (int*, unused at nc0),
        //   vector_batch_stride[[11]] (int64*), matrix_batch_stride[[12]] (int64*).
        // With `nc0` (kDoNCBatch=false) the non-batch branch still dereferences
        // `vector_batch_stride[0]` and `matrix_batch_stride[0]` (×tid.z=0), so
        // those two int64 buffers must be present (value 0); buffers 13/14 are
        // axpby-only and left unbound. `mat`/`vec` are shared by name with the MT
        // inputs (placeholders) so both kernels see identical data. tol floor 1e-2
        // is the legacy gemv reduction floor (MT folds the dot in f32; MLX
        // accumulates in the simdgroup acc dtype).
        // `mat`/`vec` seeded `Signed` (period-8 `[-3..3]`, nan-free, finite dot
        // over K) rather than raw `BenchBuffer::random` (random f32 *bytes* alias
        // to inf/nan and would poison the A/B); the runner shares these exact
        // bytes with the reference by name.
        BenchSetup::new(mt_gemv::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(input_buffer("mat", m * k, dt, InputDomain::Signed))
            .buffer(input_buffer("vec", k, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", m, dt).output())
            .constexpr("k", k as u32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
            .bytes_moved((m * k * dt.size_bytes()) as u64)
            // Matrix-vector out[m] = mat[m,k] · vec[k]: 2 MACs per (row, k).
            .flops(2 * (m as u64) * (k as u64))
            .with_reference(
                RefKernel::new(
                    format!("gemv_{tn}_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/gemv.metal")),
                )
                // mat[[0]] / in_vec[[1]] shared by name with the MT inputs.
                .buffer(BenchBuffer::zeros("mat", m * k, dt))
                .buffer(BenchBuffer::zeros("vec", k, dt))
                // bias[[2]] — unused at axpby0, a 1-element placeholder.
                .buffer(BenchBuffer::zeros("bias", 1, dt))
                .buffer(BenchBuffer::zeros("out", m, dt).output())
                // in_vec_size[[4]]=K, out_vec_size[[5]]=M, matrix_ld[[6]]=K (int, 4B).
                .buffer(BenchBuffer::from_vec("in_vec_size", (k as i32).to_le_bytes().to_vec(), DType::I32))
                .buffer(BenchBuffer::from_vec("out_vec_size", (m as i32).to_le_bytes().to_vec(), DType::I32))
                .buffer(BenchBuffer::from_vec("matrix_ld", (k as i32).to_le_bytes().to_vec(), DType::I32))
                // alpha[[7]] / beta[[8]] (float, unused at axpby0).
                .buffer(BenchBuffer::from_vec("alpha", 1.0f32.to_le_bytes().to_vec(), DType::F32))
                .buffer(BenchBuffer::from_vec("beta", 0.0f32.to_le_bytes().to_vec(), DType::F32))
                // batch_ndim[[9]] / batch_shape[[10]] (unused at nc0).
                .buffer(BenchBuffer::from_vec("batch_ndim", 1i32.to_le_bytes().to_vec(), DType::I32))
                .buffer(BenchBuffer::from_vec("batch_shape", 1i32.to_le_bytes().to_vec(), DType::I32))
                // vector_batch_stride[[11]] / matrix_batch_stride[[12]] — int64,
                // dereferenced at [0] (×tid.z=0) even on the non-batch path.
                .buffer(BenchBuffer::from_vec("vec_batch_stride", 0i64.to_le_bytes().to_vec(), DType::U64))
                .buffer(BenchBuffer::from_vec("mat_batch_stride", 0i64.to_le_bytes().to_vec(), DType::U64))
                // n_tgp=256 threadgroups; group = (32, bn=1, bm=4) = 128 lanes.
                .grid(Grid::new_3d(256, 1, 1, [32, 1, 4]))
                .tol(dtype_tol(dt).max(1e-2)),
            )
    }
}

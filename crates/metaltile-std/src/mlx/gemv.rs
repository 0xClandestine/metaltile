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

// ── Tests ─────────────────────────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };

    use super::*;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }

    fn naive_matvec(mat: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
        assert_eq!(mat.len(), m * k);
        assert_eq!(vec.len(), k);
        let mut out = vec![0.0_f32; m];
        for i in 0..m {
            let mut acc = 0.0_f32;
            for j in 0..k {
                acc += mat[i * k + j] * vec[j];
            }
            out[i] = acc;
        }
        out
    }

    #[test_kernel(name = "mlx/gemv/small", dtypes = [f32], tol = 1e-3)]
    fn test_gemv_small(dt: DType) -> TestSetup {
        let m = 16usize;
        let k = 256usize;
        let mat: Vec<f32> = (0..m * k).map(|i| ((i as f32 % 13.0) - 6.0) * 0.01).collect();
        let vec: Vec<f32> = (0..k).map(|j| ((j as f32 % 7.0) - 3.0) * 0.02).collect();
        let expected = naive_matvec(&mat, &vec, m, k);
        let mut kernel = mt_gemv::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("mat", pack_f32(&mat), DType::F32))
            .input(TestBuffer::from_vec("vec", pack_f32(&vec), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("k", k as u32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(name = "mlx/gemv/production", dtypes = [f32], tol = 5e-3)]
    fn test_gemv_production(dt: DType) -> TestSetup {
        let m = 32usize;
        let k = 4096usize;
        let mat: Vec<f32> =
            (0..m * k).map(|i| (((i * 31 + 17) % 200) as f32 - 100.0) * 0.001).collect();
        let vec: Vec<f32> = (0..k).map(|j| (((j * 13 + 5) % 100) as f32 - 50.0) * 0.002).collect();
        let expected = naive_matvec(&mat, &vec, m, k);
        let mut kernel = mt_gemv::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("mat", pack_f32(&mat), DType::F32))
            .input(TestBuffer::from_vec("vec", pack_f32(&vec), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .constexpr("k", k as u32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile_core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "gemv/gemv", dtypes = [f32, f16, bf16])]
    fn bench_mt_gemv(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_mat_vec(mt_gemv::kernel_ir_for(dt), dt, 4096, 4096, 512)
    }
}

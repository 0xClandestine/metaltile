//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Masked GEMV benchmark — #[kernel] DSL (no MLX reference)

use metaltile::kernel;

#[kernel]
pub fn mt_gemv_masked<T>(
    mat: Tensor<T>,
    vec: Tensor<T>,
    mask: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let mut acc = 0.0f32;
    for _i in range(rs + tid, re, lsize) {
        let col = _i - rs;
        let m_val = load(mask[col]).cast::<f32>();
        acc = acc + load(mat[_i]).cast::<f32>() * load(vec[col]).cast::<f32>() * m_val;
    }
    let result = reduce_sum(acc);
    store(out[row], result.cast::<T>());
}

// ── bottom of source file ────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        use DType::*;
        match dt {
            F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    /// Naive CPU masked matvec: `out[i] = Σ_j mat[i,j] * vec[j] * mask[j]`.
    fn naive_masked_matvec(mat: &[f32], vec: &[f32], mask: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m];
        for i in 0..m {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += mat[i * k + j] * vec[j] * mask[j];
            }
            out[i] = acc;
        }
        out
    }

    #[test_kernel(name = "mlx/gemv_masked/small_f32", dtypes = [f32], tol = 1e-3)]
    fn test_gemv_masked_small_f32(dt: DType) -> TestSetup {
        let (m, k) = (16usize, 256usize);
        let mat: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let vec: Vec<f32> = (0..k).map(|j| ((j % 7) as f32 - 3.0) * 0.02).collect();
        let mask: Vec<f32> = (0..k).map(|j| if j % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let expected = naive_masked_matvec(&mat, &vec, &mask, m, k);
        let mut kernel = mt_gemv_masked::kernel_ir_for(dt);
        kernel.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("mat", pack(&mat, dt), dt))
            .input(TestBuffer::from_vec("vec", pack(&vec, dt), dt))
            .input(TestBuffer::from_vec("mask", pack(&mask, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("k", k as u32)
            .grid_2d(m as u32, 1, [256, 1])
    }

    #[test_kernel(name = "mlx/gemv_masked/all_ones_mask_f32", dtypes = [f32], tol = 1e-3)]
    fn test_gemv_masked_all_ones_mask_f32(dt: DType) -> TestSetup {
        let (m, k) = (8usize, 512usize);
        let mat: Vec<f32> =
            (0..m * k).map(|i| (((i * 31 + 17) % 100) as f32 - 50.0) * 0.001).collect();
        let vec: Vec<f32> = (0..k).map(|j| (((j * 13 + 5) % 50) as f32 - 25.0) * 0.002).collect();
        let mask = vec![1.0f32; k];
        let expected = naive_masked_matvec(&mat, &vec, &mask, m, k);
        let mut kernel = mt_gemv_masked::kernel_ir_for(dt);
        kernel.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("mat", pack(&mat, dt), dt))
            .input(TestBuffer::from_vec("vec", pack(&vec, dt), dt))
            .input(TestBuffer::from_vec("mask", pack(&mask, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("k", k as u32)
            .grid_2d(m as u32, 1, [256, 1])
    }

    #[test_kernel(name = "mlx/gemv_masked/production_f32", dtypes = [f32], tol = 5e-3)]
    fn test_gemv_masked_production_f32(dt: DType) -> TestSetup {
        let (m, k) = (32usize, 4096usize);
        let mat: Vec<f32> =
            (0..m * k).map(|i| (((i * 31 + 17) % 200) as f32 - 100.0) * 0.001).collect();
        let vec: Vec<f32> = (0..k).map(|j| (((j * 13 + 5) % 100) as f32 - 50.0) * 0.002).collect();
        let mask: Vec<f32> = (0..k).map(|j| if j % 4 != 3 { 1.0 } else { 0.0 }).collect();
        let expected = naive_masked_matvec(&mat, &vec, &mask, m, k);
        let mut kernel = mt_gemv_masked::kernel_ir_for(dt);
        kernel.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("mat", pack(&mat, dt), dt))
            .input(TestBuffer::from_vec("vec", pack(&vec, dt), dt))
            .input(TestBuffer::from_vec("mask", pack(&mask, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("k", k as u32)
            .grid_2d(m as u32, 1, [256, 1])
    }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile_core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "gemv_masked/gemv_masked", dtypes = [f32, f16, bf16])]
    fn bench_mt_gemv_masked(dt: DType) -> BenchSetup {
        crate::benches::bench_mat_vec_masked(mt_gemv_masked::kernel_ir_for(dt), dt, 4096, 4096, 256)
    }
}

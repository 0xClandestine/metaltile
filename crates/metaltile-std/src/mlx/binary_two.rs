//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! binary_two benchmark — #[kernel] DSL fused two-output elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_binary_two<T>(a: Tensor<T>, b: Tensor<T>, mut c: Tensor<T>, mut d: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    let y = load(b[idx]);
    store(c[idx], x + y);
    store(d[idx], x * y);
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

    fn round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    // binary_two produces two outputs (c = a+b, d = a*b).
    // Register one test per output per dtype.

    #[test_kernel(name = "mlx/binary_two/add_f32", dtypes = [f32], tol = 1e-5)]
    fn test_binary_two_add_f32(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();
        let expected_c: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
        TestSetup::new(mt_binary_two::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("c", pack(&expected_c, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/binary_two/mul_f32", dtypes = [f32], tol = 1e-5)]
    fn test_binary_two_mul_f32(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();
        let expected_d: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
        TestSetup::new(mt_binary_two::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("d", pack(&expected_d, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/binary_two/add_f16", dtypes = [f16], tol = 1e-3)]
    fn test_binary_two_add_f16(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| round((i % 13) as f32 * 0.1 - 0.5, dt)).collect();
        let b: Vec<f32> = (0..n).map(|i| round((i % 11) as f32 * 0.08 - 0.4, dt)).collect();
        let expected_c: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
        TestSetup::new(mt_binary_two::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("c", pack(&expected_c, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/binary_two/add_bf16", dtypes = [bf16], tol = 1e-2)]
    fn test_binary_two_add_bf16(dt: DType) -> TestSetup {
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|i| round((i % 11) as f32 * 0.12 - 0.6, dt)).collect();
        let b: Vec<f32> = (0..n).map(|i| round((i % 7) as f32 * 0.1 - 0.3, dt)).collect();
        let expected_c: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
        TestSetup::new(mt_binary_two::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("c", pack(&expected_c, dt), dt))
            .grid_1d(n, 256)
    }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile_core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "binary_two/add_mul", dtypes = [f32, f16, bf16])]
    fn bench_mt_binary_two(dt: DType) -> BenchSetup {
        crate::benches::bench_binary_two(mt_binary_two::kernel_ir_for(dt), dt, crate::benches::ELEMENTWISE_N, crate::benches::ELEMENTWISE_TPG)
    }
}

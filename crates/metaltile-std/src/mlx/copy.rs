//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    #[test_kernel(name = "mlx/copy/f32", dtypes = [f32], tol = 1e-6)]
    fn test_copy_f32(dt: DType) -> TestSetup {
        let n = 4096usize;
        let a = ramp(n, 23, 11.0);
        TestSetup::new(mt_copy::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&a, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/copy/f16", dtypes = [f16], tol = 1e-3)]
    fn test_copy_f16(dt: DType) -> TestSetup {
        let n = 2048usize;
        let a: Vec<f32> = ramp(n, 17, 8.0).iter().map(|&v| round_dt(v, dt)).collect();
        TestSetup::new(mt_copy::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&a, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/copy/bf16", dtypes = [bf16], tol = 1e-2)]
    fn test_copy_bf16(dt: DType) -> TestSetup {
        let n = 2048usize;
        let a: Vec<f32> = ramp(n, 13, 6.0).iter().map(|&v| round_dt(v, dt)).collect();
        TestSetup::new(mt_copy::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&a, dt), dt))
            .grid_1d(n, 256)
    }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile_core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "copy/copy", dtypes = [f32, f16, bf16])]
    fn bench_mt_copy(dt: DType) -> BenchSetup {
        crate::benches::bench_unary(mt_copy::kernel_ir_for(dt), dt, crate::benches::ELEMENTWISE_N, crate::benches::ELEMENTWISE_TPG)
    }
}

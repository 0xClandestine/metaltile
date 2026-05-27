//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Elementwise binary ops — #[kernel] DSL vs MLX metal/binary.metal

use metaltile::kernel;

#[kernel]
pub fn vector_add<T>(a: Tensor<T>, b: Tensor<T>, c: Tensor<T>) {
    let idx = program_id(0);
    store(c[idx], load(a[idx]) + load(b[idx]));
}

#[kernel]
pub fn mt_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) * load(b[idx]));
}

#[kernel]
pub fn mt_sub<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) - load(b[idx]));
}

#[kernel]
pub fn mt_div<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) / load(b[idx]));
}

#[kernel]
pub fn mt_max_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], max(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_min_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], min(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_pow<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], pow(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_atan2<T>(y: Tensor<T>, x: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atan2(load(y[idx]), load(x[idx])));
}

#[kernel]
pub fn mt_remainder<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], remainder(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_logaddexp<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(exp(load(a[idx])) + exp(load(b[idx]))));
}

mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestSetup, TestBuffer}};

    fn pack_f32(vals: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice::<f32, u8>(vals).to_vec()
    }

    fn oracle_remainder(a: f32, b: f32) -> f32 { a % b }

    fn make_remainder_setup(a: Vec<f32>, b: Vec<f32>, dt: DType) -> TestSetup {
        let n = a.len();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| oracle_remainder(x, y)).collect();
        let kernel = mt_remainder::kernel_ir_for(dt);
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("a",   pack_f32(&a), dt))
            .input(TestBuffer::from_vec("b",   pack_f32(&b), dt))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; n]), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/remainder_positive_f32", dtypes = [f32], tol = 1e-4)]
    fn test_remainder_positive_f32(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.9 + 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.4 + 0.3).collect();
        make_remainder_setup(a, b, dt)
    }

    #[test_kernel(name = "mlx/remainder_negative_dividend_f32", dtypes = [f32], tol = 1e-5)]
    fn test_remainder_negative_dividend_f32(dt: DType) -> TestSetup {
        let a = vec![-7.0f32, -3.5, -10.0, -1.0, -5.0, 5.0, 7.0, 3.5];
        let b = vec![ 3.0f32,  1.5,   3.0,  0.7,  2.1, 3.0, 3.0, 1.5];
        make_remainder_setup(a, b, dt)
    }

    #[test_kernel(name = "mlx/remainder_mixed_sign_f32", dtypes = [f32], tol = 1e-4)]
    fn test_remainder_mixed_sign_f32(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.3 - 3.0).collect();
        let b: Vec<f32> = (0..n).map(|i| {
            let raw = (i % 11) as f32 * 0.4 - 2.0;
            if raw.abs() < 0.3 { raw.signum() * 0.4 } else { raw }
        }).collect();
        make_remainder_setup(a, b, dt)
    }
}

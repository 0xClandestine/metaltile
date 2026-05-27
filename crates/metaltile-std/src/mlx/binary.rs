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

// ── bottom of source file ────────────────────────────────────────────────

mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestSetup, TestBuffer}};

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        use DType::*;
        match dt {
            F32  => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            F16  => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _    => panic!("unsupported dtype {dt:?}"),
        }
    }

    // ── vector_add ───────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/vector_add", dtypes = [f32, f16], tol = 1e-3)]
    fn test_vector_add(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
        TestSetup::new(vector_add::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("c", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_mul ───────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/mul", dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_mul(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.04 - 0.25).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
        TestSetup::new(mt_mul::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_sub ───────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/sub", dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_sub(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.07 - 0.6).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.05 - 0.3).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
        TestSetup::new(mt_sub::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_div ───────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/div", dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_div(dt: DType) -> TestSetup {
        let n = 512usize;
        // Avoid near-zero denominator: shift b away from 0.
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.06 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.08 + 0.2).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x / y).collect();
        TestSetup::new(mt_div::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_max_elem ──────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/max_elem", dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_max_elem(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
        TestSetup::new(mt_max_elem::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_min_elem ──────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/min_elem", dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_min_elem(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.06 - 0.35).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.min(y)).collect();
        TestSetup::new(mt_min_elem::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_pow ───────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/pow", dtypes = [f32], tol = 1e-3)]
    fn test_pow(dt: DType) -> TestSetup {
        let n = 256usize;
        // Keep base positive to avoid complex-valued results.
        let a: Vec<f32> = (0..n).map(|i| (i % 9) as f32 * 0.1 + 0.2).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.4 + 0.2).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.powf(y)).collect();
        TestSetup::new(mt_pow::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_atan2 ─────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/atan2", dtypes = [f32], tol = 1e-4)]
    fn test_atan2(dt: DType) -> TestSetup {
        let n = 512usize;
        let y_vals: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
        let x_vals: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let expected: Vec<f32> =
            y_vals.iter().zip(&x_vals).map(|(&y, &x)| y.atan2(x)).collect();
        // Kernel arg order: y, x, out — matches mt_atan2(y, x, out).
        TestSetup::new(mt_atan2::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("y", pack(&y_vals, dt), dt))
            .input(TestBuffer::from_vec("x", pack(&x_vals, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── mt_logaddexp ─────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/logaddexp", dtypes = [f32], tol = 1e-3)]
    fn test_logaddexp(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.3 - 1.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.4 - 1.0).collect();
        let expected: Vec<f32> =
            a.iter().zip(&b).map(|(&x, &y)| (x.exp() + y.exp()).ln()).collect();
        TestSetup::new(mt_logaddexp::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }
}

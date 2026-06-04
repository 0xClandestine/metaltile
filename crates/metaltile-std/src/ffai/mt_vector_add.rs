//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Generic elementwise vector add — `out[i] = a[i] + b[i]`.
//!
//! Trivial kernel that didn't already exist in the binary-op surface
//! (which is shape-spec-tied to MLX's `Binary` class registration).
//! Lands in `ffai/` so the bare-`#[kernel]` form emits cleanly into
//! `MetalTileKernels.swift` for FFAI callers.

use metaltile::kernel;

#[kernel]
pub fn ffai_vector_add<T>(a: Tensor<T>, b: Tensor<T>, mut out: Tensor<T>) {
    let i = tid;
    let av = load(a[i]).cast::<f32>();
    let bv = load(b[i]).cast::<f32>();
    store(out[i], av + bv);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_vector_add;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, dt: DType) -> TestSetup {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.2).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.1).cos() * 0.8).collect();
        let a_dt = unpack_f32(&pack_f32(&a, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&b, dt), dt);
        let expected: Vec<f32> = a_dt.iter().zip(&b_dt).map(|(x, y)| x + y).collect();
        TestSetup::new(ffai_vector_add::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_vector_add_decode(dt: DType) -> TestSetup { setup(4096, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_vector_add_long(dt: DType) -> TestSetup { setup(16384, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_vector_add;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_add(dt: DType) -> BenchSetup {
        let n = 4096usize;
        BenchSetup::new(ffai_vector_add::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a", n, dt))
            .buffer(BenchBuffer::random("b", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }
}

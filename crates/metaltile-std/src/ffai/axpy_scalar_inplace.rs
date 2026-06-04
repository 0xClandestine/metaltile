//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! In-place scaled-add — `accum[i] += scalar * src[i]`.
//!
//! Fuses the MoE per-expert epilogue (broadcast-mul + element-add into
//! the running accumulator) into a single dispatch. Saves 1 buffer
//! allocation (the broadcast scalar tensor) + 1 kernel dispatch (the
//! mul) per expert per layer. At top-K=6 × 43 layers = 258 saves per
//! token on DSv4-Flash.

use metaltile::kernel;

#[kernel]
pub fn ffai_axpy_scalar_inplace<T>(src: Tensor<T>, mut accum: Tensor<T>, #[constexpr] scalar: f32) {
    let i = tid;
    let s = load(src[i]).cast::<f32>();
    let a = load(accum[i]).cast::<f32>();
    store(accum[i], a + scalar * s);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_axpy_scalar_inplace;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, dt: DType) -> TestSetup {
        let scalar = 0.42f32;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.2).collect();
        let accum_in: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.1).cos() * 0.8).collect();
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let accum_dt = unpack_f32(&pack_f32(&accum_in, dt), dt);
        let expected: Vec<f32> =
            accum_dt.iter().zip(&src_dt).map(|(a, s)| a + scalar * s).collect();
        TestSetup::new(ffai_axpy_scalar_inplace::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("accum", pack_f32(&accum_in, dt), dt))
            .constexpr("scalar", scalar)
            .expect(TestBuffer::from_vec("accum", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_axpy_scalar_decode(dt: DType) -> TestSetup { setup(4096, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_axpy_scalar_inplace;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_axpy_scalar(dt: DType) -> BenchSetup {
        let n = 4096usize;
        BenchSetup::new(ffai_axpy_scalar_inplace::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("src", n, dt))
            .buffer(BenchBuffer::random("accum", n, dt).output())
            .constexpr("scalar", 0.5f32)
            .grid_1d(n, 256)
            .bytes_moved((n * dt.size_bytes() * 2) as u64)
    }
}

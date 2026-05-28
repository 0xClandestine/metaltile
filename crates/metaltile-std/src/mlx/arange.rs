//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile arange — `out[i] = start + i * step`, one thread per element.

use metaltile::kernel;
pub use metaltile::test::*;
pub use crate::utils::{pack_f32, scalar_bytes};

#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    let s = load(start[0]);
    let st = load(step[0]);
    store(out[idx], s + idx.cast::<T>() * st);
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]
    use super::*;

    fn setup(start: f32, step: f32, n: usize, dt: DType) -> TestSetup {
        let expected: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        TestSetup::new(mt_arange::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("out",   vec![0u8; n * dt.size_bytes()], dt))
            .input(TestBuffer::from_vec("start", scalar_bytes(start, dt), dt))
            .input(TestBuffer::from_vec("step",  scalar_bytes(step,  dt), dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // Power-of-2 step (0.5) — exactly representable in every dtype; all
    // results should be bit-exact or within 1 ULP of f32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_arange_ascending(dt: DType) -> TestSetup { setup(0.0, 0.5, 512, dt) }

    // Negative step — verifies the GPU handles descending sequences.
    // Values are small integers, exact in every dtype.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-5)]
    fn test_mt_arange_descending(dt: DType) -> TestSetup { setup(16.0, -1.0, 16, dt) }

    // Non-power-of-2 step (0.1) — exercises dtype rounding on the step
    // value; tolerances widened per dtype accordingly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1e-3, 1e-2])]
    fn test_mt_arange_fractional_step(dt: DType) -> TestSetup { setup(0.0, 0.1, 64, dt) }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]
    use super::*;
    use metaltile::bench; // explicit: `bench` conflicts with std's built-in #[bench]

    // 64 M elements matches the MLX default bench size for elementwise ops.
    // bytes_moved = output only; the two 4-byte start/step reads are negligible.
    #[bench(name = "mlx/arange", dtypes = [f32, f16, bf16])]
    fn bench_arange(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_arange::kernel_ir_for(dt))
            .buffer(BenchBuffer::zeros("out",   n, dt).output())
            .buffer(BenchBuffer::from_vec("start", scalar_bytes(0.0, dt), dt))
            .buffer(BenchBuffer::from_vec("step",  scalar_bytes(1.0, dt), dt))
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
            .bytes_moved((n * dt.size_bytes()) as u64)
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}

/// New-syntax correctness for `mt_copy` (elementwise, bit-exact).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_copy;
    use crate::utils::pack_f32;

    // Copy is bit-exact within the dtype, so the expected output is just the
    // input packed to `dt` — the GPU reproduces it byte for byte.
    fn setup(n: usize, dt: DType) -> TestSetup {
        let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.1 - 1.0).collect();
        TestSetup::new(mt_copy::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&a, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_copy(dt: DType) -> TestSetup { setup(1024, dt) }
}

/// New-syntax benchmark for `mt_copy` (vs MLX `metal/copy.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_copy;
    use crate::bench_types::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    // 64M elements (MLX default elementwise size); reads `a`, writes `out`.
    //
    // Same shape as unary's `ub_ref`: MLX `metal/copy.metal` `v_copy<tn><tn>`
    // (`copy_v`, 1 element/thread) takes `src [[buffer(0)]]`, `dst [[buffer(1)]]`,
    // `size` — so the reference binds `a` (shared by name with the MT input),
    // `out`, then the U32 element count. Legacy spec: input=Signed, tol=1e-6.
    #[bench(name = "mlx/copy", dtypes = [f32, f16, bf16])]
    fn bench_copy(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        let tn = mlx_tname(dt);
        BenchSetup::new(mt_copy::kernel_ir_for(dt))
            .buffer(input_buffer("a", n, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
            .with_reference(
                RefKernel::new(
                    format!("v_copy{tn}{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/copy.metal")),
                )
                // "a" is shared by name with the MT input above (same data); the
                // runner overrides this placeholder with the MT bytes.
                .buffer(BenchBuffer::zeros("a", n, dt))
                .buffer(BenchBuffer::zeros("out", n, dt).output())
                .buffer(BenchBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
                .grid_1d(n, 256)
                .tol(dtype_tol(dt)),
            )
    }
}

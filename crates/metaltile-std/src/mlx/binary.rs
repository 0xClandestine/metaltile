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

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile_core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "binary/add", dtypes = [f32, f16, bf16])]
    fn bench_vector_add(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(vector_add::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "c")
    }

    #[bench(name = "binary/mul", dtypes = [f32, f16, bf16])]
    fn bench_mt_mul(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_mul::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/sub", dtypes = [f32, f16, bf16])]
    fn bench_mt_sub(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_sub::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/div", dtypes = [f32, f16, bf16])]
    fn bench_mt_div(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_div::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/maximum", dtypes = [f32, f16, bf16])]
    fn bench_mt_max_elem(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_max_elem::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/minimum", dtypes = [f32, f16, bf16])]
    fn bench_mt_min_elem(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_min_elem::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/pow", dtypes = [f32, f16, bf16])]
    fn bench_mt_pow(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_pow::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/atan2", dtypes = [f32, f16, bf16])]
    fn bench_mt_atan2(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_atan2::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "y", "x", "out")
    }

    #[bench(name = "binary/remainder", dtypes = [f32, f16, bf16])]
    fn bench_mt_remainder(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_remainder::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }

    #[bench(name = "binary/logaddexp", dtypes = [f32, f16, bf16])]
    fn bench_mt_logaddexp(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_binary(mt_logaddexp::kernel_ir_for(dt), dt, crate::mlx::benches::ELEMENTWISE_N, crate::mlx::benches::ELEMENTWISE_TPG, "a", "b", "out")
    }
}

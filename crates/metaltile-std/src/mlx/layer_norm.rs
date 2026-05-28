//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Layer normalization benchmark — #[kernel] DSL vs MLX metal/layer_norm.metal

use metaltile::kernel;

#[kernel]
pub fn mt_layer_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(x[base]).cast::<f32>();
        let v1 = load(x[base + 1u32]).cast::<f32>();
        let v2 = load(x[base + 2u32]).cast::<f32>();
        let v3 = load(x[base + 3u32]).cast::<f32>();
        s = s + v0 + v1 + v2 + v3;
        sq = sq + v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        s = s + xi;
        sq = sq + xi * xi;
    }
    let st = reduce_sum(s);
    let sqt = reduce_sum(sq);
    let mean = st / n;
    let var = sqt / n - mean * mean;
    let eps = load(eps_buf[0]);
    let is = rsqrt(var + eps);
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = (load(x[base]).cast::<f32>() - mean) * is * load(w[col]).cast::<f32>()
            + load(b[col]).cast::<f32>();
        let n1 =
            (load(x[base + 1u32]).cast::<f32>() - mean) * is * load(w[col + 1u32]).cast::<f32>()
                + load(b[col + 1u32]).cast::<f32>();
        let n2 =
            (load(x[base + 2u32]).cast::<f32>() - mean) * is * load(w[col + 2u32]).cast::<f32>()
                + load(b[col + 2u32]).cast::<f32>();
        let n3 =
            (load(x[base + 3u32]).cast::<f32>() - mean) * is * load(w[col + 3u32]).cast::<f32>()
                + load(b[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        let ci = _i - rs;
        let norm = (xi - mean) * is * load(w[ci]).cast::<f32>() + load(b[ci]).cast::<f32>();
        store(out[_i], norm.cast::<T>());
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
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

    fn dt_round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn cpu_layer_norm(x: &[f32], w: &[f32], b: &[f32], n: usize, eps: f32) -> Vec<f32> {
        let rows = x.len() / n;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let base = r * n;
            let sum: f32 = x[base..base + n].iter().sum();
            let mean = sum / n as f32;
            let sq_sum: f32 = x[base..base + n].iter().map(|v| (v - mean).powi(2)).sum();
            let var = sq_sum / n as f32;
            let is = 1.0 / (var + eps).sqrt();
            for d in 0..n {
                out[base + d] = (x[base + d] - mean) * is * w[d] + b[d];
            }
        }
        out
    }

    fn make_setup(n: usize, rows: usize, eps: f32, dt: DType) -> TestSetup {
        let tpg = n / 4;
        let x: Vec<f32> =
            (0..rows * n).map(|i| dt_round(((i % 23) as f32 - 11.0) * 0.1, dt)).collect();
        let w: Vec<f32> = (0..n).map(|i| dt_round(1.0 + (i % 7) as f32 * 0.1, dt)).collect();
        let b: Vec<f32> = (0..n).map(|i| dt_round((i % 5) as f32 * 0.02 - 0.04, dt)).collect();
        let expected = cpu_layer_norm(&x, &w, &b, n, eps);
        let mut kernel = mt_layer_norm::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])
    }

    #[test_kernel(name = "mlx/layer_norm/n128_rows4_f32", dtypes = [f32], tol = 1e-4)]
    fn test_layer_norm_n128_rows4_f32(dt: DType) -> TestSetup { make_setup(128, 4, 1e-5, dt) }

    #[test_kernel(name = "mlx/layer_norm/n512_rows3_f32", dtypes = [f32], tol = 5e-4)]
    fn test_layer_norm_n512_rows3_f32(dt: DType) -> TestSetup { make_setup(512, 3, 1e-5, dt) }

    #[test_kernel(name = "mlx/layer_norm/n256_rows2_f16", dtypes = [f16], tol = 5e-2)]
    fn test_layer_norm_n256_rows2_f16(dt: DType) -> TestSetup { make_setup(256, 2, 1e-5, dt) }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile::core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "layer_norm/layer_norm", dtypes = [f32, f16, bf16])]
    fn bench_mt_layer_norm(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_layer_norm(
            mt_layer_norm::kernel_ir_for(dt),
            dt,
            1024,
            4096,
            1024,
        )
    }
}

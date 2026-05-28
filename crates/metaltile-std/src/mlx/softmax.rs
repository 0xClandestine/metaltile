//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Softmax benchmark — #[kernel] DSL vs MLX metal/softmax.metal

use metaltile::kernel;

#[kernel]
pub fn mt_softmax<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut lm = neg_infinity();
    let mut ls = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(inp[base]).cast::<f32>();
        let v1 = load(inp[base + 1u32]).cast::<f32>();
        let v2 = load(inp[base + 2u32]).cast::<f32>();
        let v3 = load(inp[base + 3u32]).cast::<f32>();
        let cm = max(max(v0, v1), max(v2, v3));
        let nm = max(lm, cm);
        let sc = exp(lm - nm);
        let e0 = exp(v0 - nm);
        let e1 = exp(v1 - nm);
        let e2 = exp(v2 - nm);
        let e3 = exp(v3 - nm);
        ls = ls * sc + e0 + e1 + e2 + e3;
        lm = nm;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(inp[_i]).cast::<f32>();
        let nm = max(lm, xi);
        ls = ls * exp(lm - nm) + exp(xi - nm);
        lm = nm;
    }
    let rm = reduce_max(lm);
    let rsl = ls * exp(lm - rm);
    let rs_sum = reduce_sum(rsl);
    let is = recip(rs_sum);
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let f0 = exp(load(inp[base]).cast::<f32>() - rm) * is;
        let f1 = exp(load(inp[base + 1u32]).cast::<f32>() - rm) * is;
        let f2 = exp(load(inp[base + 2u32]).cast::<f32>() - rm) * is;
        let f3 = exp(load(inp[base + 3u32]).cast::<f32>() - rm) * is;
        store(out[base], f0.cast::<T>());
        store(out[base + 1u32], f1.cast::<T>());
        store(out[base + 2u32], f2.cast::<T>());
        store(out[base + 3u32], f3.cast::<T>());
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let fi = exp(load(inp[_i]).cast::<f32>() - rm) * is;
        store(out[_i], fi.cast::<T>());
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
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

    fn cpu_softmax(inp: &[f32], n: usize) -> Vec<f32> {
        let rows = inp.len() / n;
        let mut out = vec![0.0f32; inp.len()];
        for r in 0..rows {
            let base = r * n;
            let row = &inp[base..base + n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            for (d, &e) in exps.iter().enumerate() {
                out[base + d] = e / s;
            }
        }
        out
    }

    fn make_setup(n: usize, rows: usize, dt: DType) -> TestSetup {
        const TPG: u32 = 256;
        let inp: Vec<f32> =
            (0..rows * n).map(|i| dt_round(((i % 23) as f32 - 11.0) * 0.3, dt)).collect();
        let expected = cpu_softmax(&inp, n);
        let mut kernel = mt_softmax::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack(&inp, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [TPG, 1, 1])
    }

    #[test_kernel(name = "mlx/softmax/n1024_rows4_f32", dtypes = [f32], tol = 1e-4)]
    fn test_softmax_n1024_rows4_f32(dt: DType) -> TestSetup { make_setup(1024, 4, dt) }

    #[test_kernel(name = "mlx/softmax/n256_rows3_f32", dtypes = [f32], tol = 1e-4)]
    fn test_softmax_n256_rows3_f32(dt: DType) -> TestSetup { make_setup(256, 3, dt) }

    #[test_kernel(name = "mlx/softmax/n1024_rows2_f16", dtypes = [f16], tol = 5e-3)]
    fn test_softmax_n1024_rows2_f16(dt: DType) -> TestSetup { make_setup(1024, 2, dt) }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Ternary select benchmark — #[kernel] DSL vs MLX metal/ternary.metal
//!
//! MLX kernel: v_Selectfloat32 / v_Selectfloat16 / v_Selectbfloat16 (ternary.metal)
//!   Params: (cond: device T*, a: device T*, b: device T*, dst: device T*,
//!            size: constant uint&) — slots [0, 1, 2, 3, 4]
//!   Grid: [ceil(N/TPG), 1, 1] × [TPG, 1, 1]
//!   Algorithm: dst[i] = cond[i] != 0 ? a[i] : b[i]  (one thread per element)
//!
//! MetalTile: mt_select — same algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_select<T>(cond: Tensor<u8>, on_true: Tensor<T>, on_false: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let c = load(cond[idx]);
    let t = load(on_true[idx]);
    let f = load(on_false[idx]);
    store(out[idx], select(c, t, f));
}

// ── bottom of source file ────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        use DType::*;
        match dt {
            F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn cpu_select(on_true: &[f32], on_false: &[f32], cond: &[u8]) -> Vec<f32> {
        cond.iter()
            .zip(on_true.iter().zip(on_false.iter()))
            .map(|(&c, (&t, &f))| if c != 0 { t } else { f })
            .collect()
    }

    #[test_kernel(name = "mlx/select/mixed_f32", dtypes = [f32], tol = 1e-6)]
    fn test_select_mixed_f32(dt: DType) -> TestSetup {
        let n = 1024usize;
        let on_true: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
        let on_false: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.03).collect();
        let cond: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let expected = cpu_select(&on_true, &on_false, &cond);
        TestSetup::new(mt_select::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("cond", cond, DType::U8))
            .input(TestBuffer::from_vec("on_true", pack(&on_true, dt), dt))
            .input(TestBuffer::from_vec("on_false", pack(&on_false, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/select/all_true_f32", dtypes = [f32], tol = 1e-6)]
    fn test_select_all_true_f32(dt: DType) -> TestSetup {
        let n = 512usize;
        let on_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 1.0).collect();
        let on_false: Vec<f32> = vec![99.0f32; n];
        let cond: Vec<u8> = vec![1u8; n];
        let expected = on_true.clone();
        TestSetup::new(mt_select::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("cond", cond, DType::U8))
            .input(TestBuffer::from_vec("on_true", pack(&on_true, dt), dt))
            .input(TestBuffer::from_vec("on_false", pack(&on_false, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/select/all_false_f32", dtypes = [f32], tol = 1e-6)]
    fn test_select_all_false_f32(dt: DType) -> TestSetup {
        let n = 512usize;
        let on_true: Vec<f32> = vec![99.0f32; n];
        let on_false: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 2.0).collect();
        let cond: Vec<u8> = vec![0u8; n];
        let expected = on_false.clone();
        TestSetup::new(mt_select::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("cond", cond, DType::U8))
            .input(TestBuffer::from_vec("on_true", pack(&on_true, dt), dt))
            .input(TestBuffer::from_vec("on_false", pack(&on_false, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/select/mixed_f16", dtypes = [f16], tol = 1e-3)]
    fn test_select_mixed_f16(dt: DType) -> TestSetup {
        let n = 512usize;
        let on_true: Vec<f32> = (0..n).map(|i| round((i % 13) as f32 * 0.1 - 0.5, dt)).collect();
        let on_false: Vec<f32> = (0..n).map(|i| round((i % 11) as f32 * 0.2 - 1.0, dt)).collect();
        let cond: Vec<u8> = (0..n).map(|i| (i % 3 != 0) as u8).collect();
        let expected = cpu_select(&on_true, &on_false, &cond);
        TestSetup::new(mt_select::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("cond", cond, DType::U8))
            .input(TestBuffer::from_vec("on_true", pack(&on_true, dt), dt))
            .input(TestBuffer::from_vec("on_false", pack(&on_false, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/select/mixed_bf16", dtypes = [bf16], tol = 1e-2)]
    fn test_select_mixed_bf16(dt: DType) -> TestSetup {
        let n = 256usize;
        let on_true: Vec<f32> = (0..n).map(|i| round((i % 7) as f32 * 0.3 - 1.0, dt)).collect();
        let on_false: Vec<f32> = (0..n).map(|i| round((i % 9) as f32 * 0.2 - 0.8, dt)).collect();
        let cond: Vec<u8> = (0..n).map(|i| (i % 4 < 2) as u8).collect();
        let expected = cpu_select(&on_true, &on_false, &cond);
        TestSetup::new(mt_select::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("cond", cond, DType::U8))
            .input(TestBuffer::from_vec("on_true", pack(&on_true, dt), dt))
            .input(TestBuffer::from_vec("on_false", pack(&on_false, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }
}

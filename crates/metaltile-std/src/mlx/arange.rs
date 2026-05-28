//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Arange benchmark — #[kernel] DSL vs MLX metal/arange.metal
//!
//! MLX kernel: arangefloat32 / arangefloat16 / arangebfloat16 (arange.metal)
//!   Params: (start: constant T&, step: constant T&, out: device T*) — slots [0, 1, 2]
//!   Grid: [ceil(N/1024), 1, 1] × [1024, 1, 1]  (TPG=1024)
//!   Algorithm: out[index] = start + index * step  (one thread per element)
//!
//! MetalTile: mt_arange — same one-thread-per-element algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    let s = load(start[0]);
    let st = load(step[0]);
    store(out[idx], s + idx.cast::<T>() * st);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
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

    fn cpu_arange(start: f32, step: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| start + i as f32 * step).collect()
    }

    #[test_kernel(name = "mlx/arange/unit_step", dtypes = [f32], tol = 1e-4)]
    fn test_arange_unit_step(dt: DType) -> TestSetup {
        let (start, step, n) = (0.0f32, 1.0f32, 1024usize);
        let expected = cpu_arange(start, step, n);
        TestSetup::new(mt_arange::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("start", pack(&[start], dt), dt))
            .input(TestBuffer::from_vec("step", pack(&[step], dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/arange/fractional_step", dtypes = [f32], tol = 1e-4)]
    fn test_arange_fractional_step(dt: DType) -> TestSetup {
        let (start, step, n) = (0.5f32, 0.01f32, 512usize);
        let expected = cpu_arange(start, step, n);
        TestSetup::new(mt_arange::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("start", pack(&[start], dt), dt))
            .input(TestBuffer::from_vec("step", pack(&[step], dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/arange/negative_start", dtypes = [f32], tol = 1e-3)]
    fn test_arange_negative_start(dt: DType) -> TestSetup {
        let (start, step, n) = (-10.0f32, 0.05f32, 400usize);
        let expected = cpu_arange(start, step, n);
        TestSetup::new(mt_arange::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("start", pack(&[start], dt), dt))
            .input(TestBuffer::from_vec("step", pack(&[step], dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
    }
}

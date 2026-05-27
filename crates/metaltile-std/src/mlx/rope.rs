//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! RoPE benchmark — #[kernel] DSL vs MLX metal/rope.metal

use metaltile::kernel;

#[kernel]
pub fn mt_rope<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] h_stride: u32,
    #[constexpr] seq_stride: u32,
    #[constexpr] grid_x: u32,
    #[constexpr] base: f32,
) {
    let px = program_id::<0>();
    let py = program_id::<1>();
    let pz = program_id::<2>();
    let px_f = px.cast::<f32>();
    let gx_f = grid_x.cast::<f32>();
    let d_norm = px_f / gx_f;
    let inv_freq = exp2(-(d_norm * base));
    let theta = py.cast::<f32>() * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    let head_base = pz * 4;
    for i in range(0, 4, 1) {
        let head = head_base + i;
        let idx1 = py * seq_stride + head * h_stride + px;
        let idx2 = idx1 + grid_x;
        let x1 = load(inp[idx1]).cast::<f32>();
        let x2 = load(inp[idx2]).cast::<f32>();
        let rx1 = x1 * cos_t - x2 * sin_t;
        let rx2 = x1 * sin_t + x2 * cos_t;
        store(out[idx1], rx1.cast::<T>());
        store(out[idx2], rx2.cast::<T>());
    }
}

mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestSetup, TestBuffer}};

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32  => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16  => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _           => panic!("unsupported dtype {dt:?}"),
        }
    }

    /// CPU oracle for rotate-half RoPE — matches the kernel's exp2 arithmetic exactly.
    fn naive_rope(inp: &[f32], n_heads: u32, seq_len: u32, head_dim: u32, theta_base: f32) -> Vec<f32> {
        let grid_x = head_dim / 2;
        let h_stride = seq_len * head_dim;
        let seq_stride = head_dim;
        let base = theta_base.log2();
        let mut out = vec![0.0f32; inp.len()];
        for pz in 0..n_heads / 4 {
            for py in 0..seq_len {
                for px in 0..grid_x {
                    let d_norm = px as f32 / grid_x as f32;
                    let inv_freq = (-(d_norm * base)).exp2();
                    let theta = py as f32 * inv_freq;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    for i in 0..4 {
                        let head = pz * 4 + i;
                        let idx1 = (py * seq_stride + head * h_stride + px) as usize;
                        let idx2 = idx1 + grid_x as usize;
                        let x1 = inp[idx1];
                        let x2 = inp[idx2];
                        out[idx1] = x1 * cos_t - x2 * sin_t;
                        out[idx2] = x1 * sin_t + x2 * cos_t;
                    }
                }
            }
        }
        out
    }

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16  => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _           => v,
        }
    }

    fn make_rope_setup(
        n_heads: u32, seq_len: u32, head_dim: u32, theta_base: f32, dt: DType,
    ) -> TestSetup {
        assert!(n_heads % 4 == 0 && head_dim % 2 == 0);
        let grid_x  = head_dim / 2;
        let h_stride   = seq_len * head_dim;
        let seq_stride = head_dim;
        let base = theta_base.log2();
        let n = (n_heads * seq_len * head_dim) as usize;
        let inp_f32: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
        let inp_rounded: Vec<f32> = inp_f32.iter().map(|&v| round_dt(v, dt)).collect();
        let expected = naive_rope(&inp_rounded, n_heads, seq_len, head_dim, theta_base);

        let mut kernel = mt_rope::kernel_ir_for(dt);
        kernel.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp",  pack(&inp_f32, dt), dt))
            .input(TestBuffer::from_vec("out",  pack(&vec![0.0f32; n], dt), dt))
            .input(TestBuffer::from_vec("h_stride",   h_stride.to_le_bytes().to_vec(),   DType::U32))
            .input(TestBuffer::from_vec("seq_stride",  seq_stride.to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("grid_x",     grid_x.to_le_bytes().to_vec(),     DType::U32))
            .input(TestBuffer::from_vec("base",       base.to_le_bytes().to_vec(),        DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(grid_x, seq_len, n_heads / 4, [grid_x, seq_len, n_heads / 4])
    }

    #[test_kernel(name = "mlx/rope_f32", dtypes = [f32], tol = 5e-5)]
    fn test_rope_f32(dt: DType) -> TestSetup {
        make_rope_setup(8, 6, 16, 10000.0, dt)
    }

    #[test_kernel(name = "mlx/rope_f16", dtypes = [f16], tol = 0.005)]
    fn test_rope_f16(dt: DType) -> TestSetup {
        make_rope_setup(8, 6, 16, 10000.0, dt)
    }

    #[test_kernel(name = "mlx/rope_bf16", dtypes = [bf16], tol = 0.02)]
    fn test_rope_bf16(dt: DType) -> TestSetup {
        make_rope_setup(8, 6, 16, 10000.0, dt)
    }
}

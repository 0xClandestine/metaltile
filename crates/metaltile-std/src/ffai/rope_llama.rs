//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Llama-style RoPE with optional Llama-3 frequency-band scaling.
//! Per-token decode form (single position constexpr), generic over T.
//!
//! Different from `mt_rope` (in `mlx/rope.rs`):
//!   - decode-only (no batch / seq grid)
//!   - generic dtype (mt_rope is f16-only)
//!   - supports Llama-3 wavelength banding (low / high / smoothed)
//!
//! For each (head, i in 0..head_dim/2):
//!
//!   base inv_freq = 1 / theta_base^(2i / head_dim)
//!   wavelen       = 2*pi / inv_freq
//!   if wavelen > low_freq_wavelen:        inv_freq /= scale_factor      (low-freq band)
//!   else if wavelen < high_freq_wavelen:  inv_freq                       (high-freq band)
//!   else (medium band):                   smoothed interpolation
//!
//! To turn scaling OFF, pass scale_factor=1, low_freq_factor=1,
//! high_freq_factor=1, original_max_position=very_large (e.g. 1e9).
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::kernel;

#[kernel]
pub fn ffai_rope_llama<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] scale_factor: f32,
    #[constexpr] low_freq_factor: f32,
    #[constexpr] high_freq_factor: f32,
    #[constexpr] original_max_position: f32,
) {
    let head = program_id::<0>();
    let i = program_id::<1>();
    let i_f = i.cast::<f32>();
    let half_f = half_dim.cast::<f32>();
    let inv_freq_base = exp2(-i_f * log2(theta_base) / half_f);
    let two_pi = 6.283185307179586f32;
    let wavelen = two_pi / inv_freq_base;
    let low_freq_wavelen = original_max_position / low_freq_factor;
    let high_freq_wavelen = original_max_position / high_freq_factor;
    let scaled = inv_freq_base / scale_factor;
    let smooth_num = original_max_position / wavelen - low_freq_factor;
    let smooth_den = high_freq_factor - low_freq_factor;
    let s = smooth_num / smooth_den;
    let smoothed = (1.0f32 - s) * scaled + s * inv_freq_base;
    let is_low_freq = wavelen > low_freq_wavelen;
    let is_high_freq = wavelen < high_freq_wavelen;
    let inv_freq = select(is_low_freq, scaled, select(is_high_freq, inv_freq_base, smoothed));
    let pos_f = position.cast::<f32>();
    let theta = pos_f * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    let base = head * head_dim;
    let i1 = base + i;
    let i2 = base + i + half_dim;
    let x1 = load(qk[i1]).cast::<f32>();
    let x2 = load(qk[i2]).cast::<f32>();
    let o1 = x1 * cos_t - x2 * sin_t;
    let o2 = x1 * sin_t + x2 * cos_t;
    store(out[i1], o1.cast::<T>());
    store(out[i2], o2.cast::<T>());
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

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16  => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _           => v,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_rope_llama(
        qk: &[f32], head_dim: u32, n_heads: u32, position: u32,
        theta_base: f32, scale_factor: f32,
        low_freq_factor: f32, high_freq_factor: f32, original_max_position: f32,
    ) -> Vec<f32> {
        let half_dim = head_dim / 2;
        let half_f = half_dim as f32;
        let two_pi = std::f32::consts::TAU;
        let mut out = vec![0.0_f32; qk.len()];
        for head in 0..n_heads {
            let base = (head * head_dim) as usize;
            for i in 0..half_dim {
                let i_f = i as f32;
                let inv_freq_base = (-i_f * theta_base.log2() / half_f).exp2();
                let wavelen = two_pi / inv_freq_base;
                let low_wavelen = original_max_position / low_freq_factor;
                let high_wavelen = original_max_position / high_freq_factor;
                let scaled = inv_freq_base / scale_factor;
                let smooth_num = original_max_position / wavelen - low_freq_factor;
                let smooth_den = high_freq_factor - low_freq_factor;
                let s = smooth_num / smooth_den;
                let smoothed = (1.0 - s) * scaled + s * inv_freq_base;
                let inv_freq = if wavelen > low_wavelen {
                    scaled
                } else if wavelen < high_wavelen {
                    inv_freq_base
                } else {
                    smoothed
                };
                let theta = position as f32 * inv_freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let i1 = base + i as usize;
                let i2 = base + (i + half_dim) as usize;
                let x1 = qk[i1]; let x2 = qk[i2];
                out[i1] = x1 * cos_t - x2 * sin_t;
                out[i2] = x1 * sin_t + x2 * cos_t;
            }
        }
        out
    }

    fn make_setup(
        n_heads: u32, head_dim: u32, position: u32,
        theta_base: f32, scale_factor: f32,
        low_freq_factor: f32, high_freq_factor: f32, original_max_position: f32,
        dt: DType,
    ) -> TestSetup {
        let half_dim = head_dim / 2;
        let n = (n_heads * head_dim) as usize;
        let qk_f32: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
        let qk_rounded: Vec<f32> = qk_f32.iter().map(|&v| round_dt(v, dt)).collect();
        let expected = naive_rope_llama(
            &qk_rounded, head_dim, n_heads, position,
            theta_base, scale_factor, low_freq_factor, high_freq_factor, original_max_position,
        );
        let mut kernel = ffai_rope_llama::kernel_ir_for(dt);
        kernel.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("qk",  pack(&qk_f32, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0_f32; n], dt), dt))
            .input(TestBuffer::from_vec("head_dim",              head_dim.to_le_bytes().to_vec(),             DType::U32))
            .input(TestBuffer::from_vec("half_dim",              half_dim.to_le_bytes().to_vec(),             DType::U32))
            .input(TestBuffer::from_vec("position",              position.to_le_bytes().to_vec(),             DType::U32))
            .input(TestBuffer::from_vec("theta_base",            theta_base.to_le_bytes().to_vec(),           DType::F32))
            .input(TestBuffer::from_vec("scale_factor",          scale_factor.to_le_bytes().to_vec(),         DType::F32))
            .input(TestBuffer::from_vec("low_freq_factor",       low_freq_factor.to_le_bytes().to_vec(),      DType::F32))
            .input(TestBuffer::from_vec("high_freq_factor",      high_freq_factor.to_le_bytes().to_vec(),     DType::F32))
            .input(TestBuffer::from_vec("original_max_position", original_max_position.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(n_heads, half_dim, 1, [n_heads, half_dim, 1])
    }

    // No-scaling params (disable banding).
    fn no_scaling() -> (f32, f32, f32, f32) { (1.0, 1.0, 1.0, 1.0e10) }
    // Llama-3.1 8B official params.
    fn llama3_scaling() -> (f32, f32, f32, f32) { (8.0, 1.0, 4.0, 8192.0) }

    #[test_kernel(name = "ffai/rope_llama_f32", dtypes = [f32], tol = 5e-5)]
    fn test_rope_llama_f32(dt: DType) -> TestSetup {
        let (sf, lf, hf, mp) = no_scaling();
        make_setup(8, 64, 137, 10000.0, sf, lf, hf, mp, dt)
    }

    #[test_kernel(name = "ffai/rope_llama_llama3_f32", dtypes = [f32], tol = 2e-3)]
    fn test_rope_llama_llama3_f32(dt: DType) -> TestSetup {
        let (sf, lf, hf, mp) = llama3_scaling();
        make_setup(8, 64, 16000, 500000.0, sf, lf, hf, mp, dt)
    }

    #[test_kernel(name = "ffai/rope_llama_f16", dtypes = [f16], tol = 0.005)]
    fn test_rope_llama_f16(dt: DType) -> TestSetup {
        let (sf, lf, hf, mp) = no_scaling();
        make_setup(8, 64, 73, 10000.0, sf, lf, hf, mp, dt)
    }

    #[test_kernel(name = "ffai/rope_llama_bf16", dtypes = [bf16], tol = 0.02)]
    fn test_rope_llama_bf16(dt: DType) -> TestSetup {
        let (sf, lf, hf, mp) = no_scaling();
        make_setup(8, 64, 41, 10000.0, sf, lf, hf, mp, dt)
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! YaRN RoPE — per-token decode form, generic over T.
//!
//! YaRN ("Yet another RoPE extensioN") rescales the rotary frequencies
//! to extend a model's usable context. Per dimension it blends between
//! **extrapolation** (the original frequency — kept for high-frequency
//! dimensions) and **interpolation** (the frequency divided by
//! `factor` — applied to low-frequency dimensions), with a linear ramp
//! across a `[low, high]` correction band:
//!
//!   inv_freq_extrap = theta_base^(-2i/head_dim)
//!   inv_freq_interp = inv_freq_extrap / factor
//!   ramp            = clamp((i - low) / (high - low), 0, 1)
//!   inv_freq        = inv_freq_interp*ramp + inv_freq_extrap*(1 - ramp)
//!
//! `low` / `high` are the YaRN correction-range bounds. They derive
//! from `beta_fast` / `beta_slow` via a `floor`/`ceil`/`ln` computation
//! that is constant across the whole dispatch, so the caller computes
//! them once and passes them as constexpr (see `Ops.ropeYaRN`).
//! `attn_factor` is YaRN's mscale attention scaling — `1.0` when the
//! checkpoint's `mscale == mscale_all_dim` (the common case, including
//! Nemotron-Labs-Diffusion).
//!
//! Same Grid3D dispatch shape as `ffai_rope_llama`: one thread per
//! (head, i in 0..head_dim/2), each thread rotating the pair
//! (i, i + half_dim). No reduction, no threadgroup memory.
//!
//! Codegen-only. Validated by `rope_yarn_gpu_correctness` + FFAI
//! integration tests.

use metaltile::kernel;

#[kernel]
pub fn ffai_rope_yarn<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] factor: f32,
    #[constexpr] low: f32,
    #[constexpr] high: f32,
    #[constexpr] attn_factor: f32,
) {
    let head = program_id::<0>();
    let i = program_id::<1>();
    let i_f = i.cast::<f32>();
    let half_f = half_dim.cast::<f32>();
    // Base (extrapolation) frequency — identical to plain RoPE.
    let inv_freq_extrap = exp2(-i_f * log2(theta_base) / half_f);
    // Interpolation frequency — extended context by `factor`.
    let inv_freq_interp = inv_freq_extrap / factor;
    // Linear ramp over the [low, high] correction band, clamped to
    // [0, 1]. ramp=0 → pure extrapolation; ramp=1 → pure interpolation.
    // The caller guarantees high > low, so the divide is safe.
    let t = (i_f - low) / (high - low);
    let ramp = select(t < 0.0f32, 0.0f32, select(t > 1.0f32, 1.0f32, t));
    let inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0f32 - ramp);
    let pos_f = position.cast::<f32>();
    let theta = pos_f * inv_freq;
    let cos_t = cos(theta) * attn_factor;
    let sin_t = sin(theta) * attn_factor;
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

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_rope_yarn(
        qk: &[f32],
        head_dim: u32,
        n_heads: u32,
        position: u32,
        theta_base: f32,
        factor: f32,
        low: f32,
        high: f32,
        attn_factor: f32,
    ) -> Vec<f32> {
        let half_dim = head_dim / 2;
        let half_f = half_dim as f32;
        let mut out = vec![0.0_f32; qk.len()];
        for head in 0..n_heads {
            let base = (head * head_dim) as usize;
            for i in 0..half_dim {
                let i_f = i as f32;
                let inv_freq_extrap = (-i_f * theta_base.log2() / half_f).exp2();
                let inv_freq_interp = inv_freq_extrap / factor;
                let t = (i_f - low) / (high - low);
                let ramp = t.clamp(0.0, 1.0);
                let inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0 - ramp);
                let theta = position as f32 * inv_freq;
                let cos_t = theta.cos() * attn_factor;
                let sin_t = theta.sin() * attn_factor;
                let i1 = base + i as usize;
                let i2 = base + (i + half_dim) as usize;
                let x1 = qk[i1];
                let x2 = qk[i2];
                out[i1] = x1 * cos_t - x2 * sin_t;
                out[i2] = x1 * sin_t + x2 * cos_t;
            }
        }
        out
    }

    fn make_setup(
        n_heads: u32,
        head_dim: u32,
        position: u32,
        theta_base: f32,
        factor: f32,
        low: f32,
        high: f32,
        attn_factor: f32,
        dt: DType,
    ) -> TestSetup {
        let half_dim = head_dim / 2;
        let n = (n_heads * head_dim) as usize;
        let qk_f32: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
        let qk_rounded: Vec<f32> = qk_f32.iter().map(|&v| round_dt(v, dt)).collect();
        let expected = naive_rope_yarn(
            &qk_rounded,
            head_dim,
            n_heads,
            position,
            theta_base,
            factor,
            low,
            high,
            attn_factor,
        );
        let mut kernel = ffai_rope_yarn::kernel_ir_for(dt);
        kernel.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("qk", pack(&qk_f32, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0_f32; n], dt), dt))
            .input(TestBuffer::from_vec("head_dim", head_dim.to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("half_dim", half_dim.to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("position", position.to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec(
                "theta_base",
                theta_base.to_le_bytes().to_vec(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec("factor", factor.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec("low", low.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec("high", high.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec(
                "attn_factor",
                attn_factor.to_le_bytes().to_vec(),
                DType::F32,
            ))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(n_heads, half_dim, 1, [n_heads, half_dim, 1])
    }

    #[test_kernel(name = "ffai/rope_yarn_f32", dtypes = [f32], tol = 5e-5)]
    fn test_rope_yarn_f32(dt: DType) -> TestSetup {
        // Nemotron-Labs-Diffusion params, n_heads*half_dim = 16*64 = 1024.
        make_setup(16, 128, 512, 1.0e6, 16.0, 20.0, 37.0, 1.0, dt)
    }

    #[test_kernel(name = "ffai/rope_yarn_f16", dtypes = [f16], tol = 0.005)]
    fn test_rope_yarn_f16(dt: DType) -> TestSetup {
        make_setup(8, 64, 73, 1.0e6, 16.0, 12.0, 28.0, 1.0, dt)
    }

    #[test_kernel(name = "ffai/rope_yarn_bf16", dtypes = [bf16], tol = 0.02)]
    fn test_rope_yarn_bf16(dt: DType) -> TestSetup {
        make_setup(8, 64, 41, 1.0e6, 16.0, 12.0, 28.0, 1.0, dt)
    }
}

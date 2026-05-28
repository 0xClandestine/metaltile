//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused RMSNorm + residual add — `out = residual + w * x * inv_rms`.
//!
//! Combines RMS normalization with the residual (skip-connection) add
//! in one dispatch. Saves a kernel launch at every post-attention and
//! post-FFN norm+residual site (≈3 calls/layer).
//!
//! Uses `mt_rms_inv_scalar` (from `mlx/rms_norm.rs`) via cross-kernel
//! call for the shared reduction phase: each thread computes its
//! `partial_ssq`, then calls `mt_rms_inv_scalar(partial_ssq, eps_buf, n)`
//! which inlines the `reduce_sum + rsqrt` body. The second phase applies
//! the residual add and stores the normalized+residual output.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel; the threadgroup geometry is part of its API.
//! Violating it silently miscomputes (best case) or freezes the GPU.
//!
//! - **`N = TPG * 4`.** Each thread owns 4 consecutive elements of the
//!   row; the wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full simdgroup) and
//!   **`TPG ≤ 1024`**. Combined: `n` multiple of 128, `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row** — `program_id::<0>()` = row index.
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_residual_gpu_correctness.rs`.

use metaltile::kernel;

/// `out[r, i] = residual[r, i] + w[i] * x[r, i] * rsqrt(mean(x[r]²) + eps)`.
#[kernel]
pub fn ffai_rms_norm_residual<T>(
    x: Tensor<T>,
    residual: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // Each thread owns 4 consecutive elements (N = TPG * 4). OOB lanes
    // re-read row[0..3] (benign — their SSQ contribution is masked to 0)
    // and skip their stores, mirroring `mt_rms_norm`'s freeze-safe guard.
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col;
    let x0 = load(x[safe_base]).cast::<f32>();
    let x1 = load(x[safe_base + 1u32]).cast::<f32>();
    let x2 = load(x[safe_base + 2u32]).cast::<f32>();
    let x3 = load(x[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    // Cross-kernel call: KernelInlinePass splices mt_rms_inv_scalar's body
    // here. partial_ssq is a Value arg (pre-computed f32 scalar, no load);
    // eps_buf and n are Tensor args (renamed in callee's loads transparently).
    let rms = mt_rms_inv_scalar(partial_ssq, eps_buf, n);
    if in_bounds {
        let o0 = load(residual[base]).cast::<f32>() + x0 * rms * load(w[col]).cast::<f32>();
        let o1 = load(residual[base + 1u32]).cast::<f32>()
            + x1 * rms * load(w[col + 1u32]).cast::<f32>();
        let o2 = load(residual[base + 2u32]).cast::<f32>()
            + x2 * rms * load(w[col + 2u32]).cast::<f32>();
        let o3 = load(residual[base + 3u32]).cast::<f32>()
            + x3 * rms * load(w[col + 3u32]).cast::<f32>();
        store(out[base], o0.cast::<T>());
        store(out[base + 1u32], o1.cast::<T>());
        store(out[base + 2u32], o2.cast::<T>());
        store(out[base + 3u32], o3.cast::<T>());
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

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    fn naive_rms_norm(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
        let rows = x.len() / n;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let base = r * n;
            let ssq: f32 = x[base..base + n].iter().map(|v| v * v).sum();
            let rms = (ssq / n as f32 + eps).sqrt().recip();
            for d in 0..n {
                out[base + d] = x[base + d] * rms * w[d];
            }
        }
        out
    }

    fn naive_rms_norm_residual(
        x: &[f32],
        residual: &[f32],
        w: &[f32],
        n: usize,
        eps: f32,
    ) -> Vec<f32> {
        let normed = naive_rms_norm(x, w, n, eps);
        normed.iter().zip(residual).map(|(&v, &r)| v + r).collect()
    }

    fn make_setup(n: usize, rows: usize, eps: f32, dt: DType) -> TestSetup {
        let tpg = n / 4;
        let x: Vec<f32> = ramp(rows * n, 23, 9.0).into_iter().map(|v| dt_round(v, dt)).collect();
        let residual: Vec<f32> =
            ramp(rows * n, 29, 7.0).into_iter().map(|v| dt_round(v, dt)).collect();
        let w: Vec<f32> =
            ramp(n, 13, 6.0).into_iter().map(|v| dt_round(1.0 + 0.05 * v, dt)).collect();
        let expected = naive_rms_norm_residual(&x, &residual, &w, n, eps);
        let mut kernel = ffai_rms_norm_residual::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("residual", pack(&residual, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])
    }

    #[test_kernel(name = "ffai/rms_norm_residual/n128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_residual_n128_f32(dt: DType) -> TestSetup { make_setup(128, 1, 1e-5, dt) }

    #[test_kernel(name = "ffai/rms_norm_residual/n512_rows4_f32", dtypes = [f32], tol = 1e-4)]
    fn test_residual_n512_rows4_f32(dt: DType) -> TestSetup { make_setup(512, 4, 1e-5, dt) }

    #[test_kernel(name = "ffai/rms_norm_residual/n4096_f32", dtypes = [f32], tol = 5e-4)]
    fn test_residual_n4096_f32(dt: DType) -> TestSetup { make_setup(4096, 1, 1e-5, dt) }

    #[test_kernel(name = "ffai/rms_norm_residual/n512_f16", dtypes = [f16], tol = 2e-2)]
    fn test_residual_n512_f16(dt: DType) -> TestSetup { make_setup(512, 2, 1e-5, dt) }
}

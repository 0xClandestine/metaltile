//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused gated RMSNorm — `out = rmsNorm(y) · silu(z)`.
//!
//! The post-step of a Gated-DeltaNet (GDN) layer. After the GDN
//! recurrence (`mt_gated_delta_step` / `_chunk`) produces the linear-
//! attention output `y`, Qwen3.5 / Qwen3.6 apply a *gated* RMSNorm:
//!
//! ```text
//!   out[r, i] = w[i] · y[r, i] · rsqrt(mean(y[r]²) + eps) · silu(z[r, i])
//! ```
//!
//! The distinguishing feature versus the plain `mt_rms_norm` is the
//! **dtype split**: `y` arrives as **fp32** — the GDN recurrence
//! accumulates its state in fp32 and emits `y` in fp32 (a bf16 `y`
//! drifts after a few dozen decode steps, the same reason
//! `gated_delta` / `ssm_step` keep an fp32 accumulator). The gate `z`,
//! the weight `w`, and the output are in the model's activation dtype
//! `T`. No existing GPU norm consumes an fp32 row and writes a `T`
//! row, so without this kernel the GDN post-step runs host-side — one
//! CPU↔GPU sync per GDN layer (≈75 % of Qwen3.5/3.6 layers).
//!
//! `silu(x) = x · sigmoid(x)` is computed in fp32 from the `z` gate
//! (cast up from `T`); the normalized-and-gated result is rounded to
//! `T` at the store.
//!
//! Algorithm-identical reduction to `mlx/rms_norm.rs`'s `mt_rms_norm`
//! — f32 sum-of-squares accumulator, threadgroup-wide `reduce_sum`,
//! `rsqrt(ssq/n + eps)` scaling — with the fp32 `y` input and the
//! extra `silu(z)` gate multiply.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel; the threadgroup geometry is part of its API.
//! Violating it silently miscomputes (best case) or freezes the GPU
//! (worst case — see `docs/developing.md`).
//!
//! - **`N = TPG * 4`.** Each thread owns 4 consecutive elements of the
//!   row; the wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full Apple simdgroup) and
//!   **`TPG ≤ 1024`**. Combined: `n` a multiple of 128, `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row** — `program_id::<0>()` = row index.
//!   Multi-row dispatch uses `grid = (nRows * TPG, 1, 1)`,
//!   `tg = (TPG, 1, 1)`.
//!
//! Codegen-only; correctness pinned by
//! `tests/gated_rmsnorm_gpu_correctness.rs`.

use metaltile::kernel;

/// `out[r, i] = w[i] · y[r, i] · rsqrt(mean(y[r]²) + eps) · silu(z[r, i])`.
///
/// `y` is fp32 (the GDN recurrence output); `z`, `w`, `out` are `T`.
#[kernel]
pub fn ffai_gated_rmsnorm<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
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
    // `y` is already fp32. The explicit `.cast::<f32>()` is a no-op
    // numerically but forces codegen to bind a *named* scalar for each
    // element — without it the float4-load vectorizer collapses the
    // element names and the post-reduction store references an
    // undeclared identifier (the names must survive across the
    // threadgroup `reduce_sum`).
    let y0 = load(y[safe_base]).cast::<f32>();
    let y1 = load(y[safe_base + 1u32]).cast::<f32>();
    let y2 = load(y[safe_base + 2u32]).cast::<f32>();
    let y3 = load(y[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        // silu(x) = x / (1 + exp(-x)) — inlined in fp32 (same form as
        // mt_swiglu) to keep the gate precise before the round to T.
        let z0 = load(z[base]).cast::<f32>();
        let z1 = load(z[base + 1u32]).cast::<f32>();
        let z2 = load(z[base + 2u32]).cast::<f32>();
        let z3 = load(z[base + 3u32]).cast::<f32>();
        let g0 = z0 / (1.0f32 + exp(0.0f32 - z0));
        let g1 = z1 / (1.0f32 + exp(0.0f32 - z1));
        let g2 = z2 / (1.0f32 + exp(0.0f32 - z2));
        let g3 = z3 / (1.0f32 + exp(0.0f32 - z3));
        let o0 = y0 * rms * load(w[col]).cast::<f32>() * g0;
        let o1 = y1 * rms * load(w[col + 1u32]).cast::<f32>() * g1;
        let o2 = y2 * rms * load(w[col + 2u32]).cast::<f32>() * g2;
        let o3 = y3 * rms * load(w[col + 3u32]).cast::<f32>() * g3;
        store(out[base], o0.cast::<T>());
        store(out[base + 1u32], o1.cast::<T>());
        store(out[base + 2u32], o2.cast::<T>());
        store(out[base + 3u32], o3.cast::<T>());
    }
}

mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };
    use metaltile::test_kernel;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => {
                vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
            },
            DType::BF16 => {
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect()
            },
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

    /// Naive reference: RMSNorm of the fp32 `y` row, scaled by `w`, gated by `silu(z)`.
    fn naive_gated_rmsnorm(y: &[f32], z: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
        let rows = y.len() / n;
        let mut out = vec![0.0f32; y.len()];
        for r in 0..rows {
            let base = r * n;
            let ssq: f32 = y[base..base + n].iter().map(|v| v * v).sum();
            let rms = (ssq / n as f32 + eps).sqrt().recip();
            for d in 0..n {
                let silu_z = z[base + d] / (1.0 + (-z[base + d]).exp());
                out[base + d] = y[base + d] * rms * w[d] * silu_z;
            }
        }
        out
    }

    fn make_setup(n: usize, rows: usize, eps: f32, dt: DType) -> TestSetup {
        let tpg = n / 4;
        let y: Vec<f32> = ramp(rows * n, 23, 9.0);
        let z: Vec<f32> = ramp(rows * n, 29, 7.0)
            .into_iter()
            .map(|v| dt_round(v, dt))
            .collect();
        let w: Vec<f32> = ramp(n, 13, 6.0)
            .into_iter()
            .map(|v| dt_round(1.0 + 0.05 * v, dt))
            .collect();
        let expected = naive_gated_rmsnorm(&y, &z, &w, n, eps);
        let mut kernel = ffai_gated_rmsnorm::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("y", pack(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack(&z, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])
    }

    #[test_kernel(name = "ffai/gated_rmsnorm/n128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_gated_rmsnorm_n128_f32(dt: DType) -> TestSetup {
        make_setup(128, 1, 1e-5, dt)
    }

    #[test_kernel(name = "ffai/gated_rmsnorm/n512_rows4_f32", dtypes = [f32], tol = 1e-4)]
    fn test_gated_rmsnorm_n512_rows4_f32(dt: DType) -> TestSetup {
        make_setup(512, 4, 1e-5, dt)
    }

    #[test_kernel(name = "ffai/gated_rmsnorm/n4096_f32", dtypes = [f32], tol = 5e-4)]
    fn test_gated_rmsnorm_n4096_f32(dt: DType) -> TestSetup {
        make_setup(4096, 1, 1e-5, dt)
    }

    #[test_kernel(name = "ffai/gated_rmsnorm/n512_f16", dtypes = [f16], tol = 2e-2)]
    fn test_gated_rmsnorm_n512_f16(dt: DType) -> TestSetup {
        make_setup(512, 2, 1e-5, dt)
    }

    #[test_kernel(name = "ffai/gated_rmsnorm/n512_bf16", dtypes = [bf16], tol = 8e-2)]
    fn test_gated_rmsnorm_n512_bf16(dt: DType) -> TestSetup {
        make_setup(512, 2, 1e-5, dt)
    }
}

//! Fused RMSNorm + residual add — `out = residual + w * x * inv_rms`.
//!
//! Combines RMS normalization with the residual (skip-connection) add
//! in one dispatch. Saves a kernel launch at every post-attention and
//! post-FFN norm+residual site (≈3 calls/layer).
//!
//! Algorithm-identical to `mlx/rms_norm.rs`'s `mt_rms_norm` — f32
//! sum-of-squares accumulator, threadgroup-wide `reduce_sum`,
//! `rsqrt(ssq/n + eps)` scaling — with one extra `residual` input
//! tensor added to the normalized result before the store.
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
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

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
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
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

inventory::submit! {
    BenchSpec {
        op: "rms_norm_residual",
        subop: "rms_norm_residual",
        kernel_name: "ffai_rms_norm_residual",
        kernel_ir: ffai_rms_norm_residual::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-4,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

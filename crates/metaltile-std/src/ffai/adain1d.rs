//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Adaptive Instance Normalization (AdaIN-1d) — the style-conditioning
//! op the StyleTTS2 / Kokoro acoustic stack applies all through its
//! decoder and prosody/duration predictors.
//!
//! Each row (one `(instance, channel)` of a `[B, C, T]` feature map,
//! flattened to `[rows, T]` with `rows = B·C`) is **instance-normalized**
//! over the time axis, then affine-transformed by a per-row style
//! `(gamma, beta)` the style encoder produced:
//!
//!   `mean_r = mean_t x[r,t]`,  `var_r = mean_t x[r,t]² − mean_r²`
//!   `out[r,t] = gamma[r] · (x[r,t] − mean_r) / sqrt(var_r + eps) + beta[r]`
//!
//! Distinct from RMSNorm/LayerNorm (which normalize across the feature
//! dim with a per-element weight): AdaIN normalizes across **time** per
//! channel, and the scale/shift are per-row **scalars** from the style
//! vector. Replaces a CPU two-pass mean/var + affine loop.
//!
//! Layouts:
//!   input  `[rows, seq_len]`   T
//!   gamma  `[rows]`            f32
//!   beta   `[rows]`            f32
//!   out    `[rows, seq_len]`   T
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction, one threadgroup per row — dispatch with
//! `grid_3d(rows, 1, 1, [256, 1, 1])` (256 threads = 8 simdgroups × 32).
//! Two passes over `seq_len` inside the kernel: reduce (sum, sum²) then
//! write. `gamma`/`beta` length == `rows`.

use metaltile::kernel;

#[kernel(
    bench(
        op = "norm",
        subop = "adain1d",
        class = GenericEmpty,
        tol = 1e-3,
        kernel_mode = Reduction,
    )
)]
pub fn ffai_adain1d<T>(
    input: Tensor<T>,
    gamma: Tensor<f32>,
    beta: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] seq_len: u32,
    #[constexpr] eps: f32,
) {
    let row = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    let tid = sg * 32u32 + lane;
    let nthreads = ns * 32u32;
    let row_base = row * seq_len;
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_sumsq", 32);

    // ── Pass 1: reduce sum + sum-of-squares over the time axis. ──
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    for t in range(tid, seq_len, nthreads) {
        let x = load(input[row_base + t]).cast::<f32>();
        s = s + x;
        sq = sq + x * x;
    }
    let s_sg = simd_sum(s);
    let sq_sg = simd_sum(sq);
    if lane == 0u32 {
        threadgroup_store("tg_sum", sg, s_sg);
        threadgroup_store("tg_sumsq", sg, sq_sg);
    }
    threadgroup_barrier();
    // Simdgroup 0 combines the per-simdgroup partials.
    if sg == 0u32 {
        let ps = select(lane < ns, threadgroup_load("tg_sum", lane), 0.0f32);
        let total_s = simd_sum(ps);
        let pq = select(lane < ns, threadgroup_load("tg_sumsq", lane), 0.0f32);
        let total_sq = simd_sum(pq);
        if lane == 0u32 {
            threadgroup_store("tg_sum", 0, total_s);
            threadgroup_store("tg_sumsq", 0, total_sq);
        }
    }
    threadgroup_barrier();
    let total_s = threadgroup_load("tg_sum", 0);
    let total_sq = threadgroup_load("tg_sumsq", 0);
    let n_f = seq_len.cast::<f32>();
    let mean = total_s / n_f;
    let var = total_sq / n_f - mean * mean;
    let inv_std = rsqrt(var + eps);
    let g = load(gamma[row]);
    let b = load(beta[row]);

    // ── Pass 2: normalize + per-row affine. ──
    for t in range(tid, seq_len, nthreads) {
        let x = load(input[row_base + t]).cast::<f32>();
        let y = g * (x - mean) * inv_std + b;
        store(out[row_base + t], y.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_adain1d;
    use crate::utils::{pack_f32, unpack_f32};

    fn naive(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rows: usize,
        t: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * t];
        for r in 0..rows {
            let base = r * t;
            let mut s = 0.0f32;
            let mut sq = 0.0f32;
            for i in 0..t {
                let x = input[base + i];
                s += x;
                sq += x * x;
            }
            let mean = s / t as f32;
            let var = sq / t as f32 - mean * mean;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..t {
                out[base + i] = gamma[r] * (input[base + i] - mean) * inv + beta[r];
            }
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    fn setup(dt: DType, rows: usize, t: usize) -> TestSetup {
        let eps = 1e-5f32;
        let input_f = ramp(rows * t, 0.013, -0.4);
        let gamma: Vec<f32> = (0..rows).map(|i| 0.5 + i as f32 * 0.1).collect();
        let beta: Vec<f32> = (0..rows).map(|i| -0.2 + i as f32 * 0.05).collect();
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = naive(&input, &gamma, &beta, rows, t, eps);
        TestSetup::new(ffai_adain1d::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("gamma", pack_f32(&gamma, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("beta", pack_f32(&beta, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", rows * t, dt))
            .constexpr("seq_len", t as u32)
            .constexpr("eps", eps)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    // Short sequence (single simdgroup's worth of strided work).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 2e-2])]
    fn test_adain1d_short(dt: DType) -> TestSetup { setup(dt, 6, 40) }

    // Long sequence (every simdgroup contributes; exercises the reduction).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 2e-2])]
    fn test_adain1d_long(dt: DType) -> TestSetup { setup(dt, 4, 600) }
}

/// New-syntax bench: Kokoro decoder feature map (512 channels, 300 frames).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_adain1d;

    #[bench(name = "ffai/norm/adain1d", dtypes = [f32, f16, bf16])]
    fn bench_adain1d(dt: DType) -> BenchSetup {
        let (rows, t) = (512usize, 300usize);
        BenchSetup::new(ffai_adain1d::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", rows * t, dt))
            .buffer(BenchBuffer::random("gamma", rows, DType::F32))
            .buffer(BenchBuffer::random("beta", rows, DType::F32))
            .buffer(BenchBuffer::zeros("out", rows * t, dt).output())
            .constexpr("seq_len", t as u32)
            .constexpr("eps", 1e-5f32)
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * rows * t * dt.size_bytes()) as u64)
    }
}

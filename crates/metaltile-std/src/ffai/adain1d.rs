//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Adaptive Instance Normalization (AdaIN-1d) — instance-normalize each row
//! over the time axis, then apply a per-row affine.
//!
//! `out[r,t] = gamma[r]·(x[r,t] − μ_r) / sqrt(σ²_r + eps) + beta[r]`, where a
//! "row" is one `(batch, channel)` line of a `[B, C, T]` feature map flattened
//! to `[rows, T]` (`rows = B·C`). Distinct from RMS/LayerNorm: it normalizes
//! across **time** per channel, and the scale/shift are per-row **scalars**.
//! The style-conditioning op style-vector TTS decoders apply (e.g. the
//! StyleTTS2 family). The caller folds any `(1 + γ)` convention into `gamma`.
//!
//! One threadgroup per `(batch, channel)` row, strided over the time axis so
//! any `length` works.
//!
//! Layouts:
//!   x      `[rows, length]`   T
//!   gamma  `[rows]`           T
//!   beta   `[rows]`           T
//!   out    `[rows, length]`   T
//!   eps    `[1]`              f32

use metaltile::kernel;

#[kernel]
pub fn adain1d<T>(
    x: Tensor<T>,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    mut out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] length: u32,
) {
    // One threadgroup per (batch, channel) row; row index doubles as the
    // gamma/beta index since both are [batch, channels] row-major.
    let row = program_id::<0>();
    let rs = row * length;
    let tpg = n_simd * 32u32;
    // Pass 1: strided sum + sum-of-squares over the time axis. A thread
    // whose stride walks past `length` contributes 0 but still reaches
    // the reductions (Apple simdgroup reductions need all lanes active).
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    for i in range(tid, length, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        s = s + xi;
        sq = sq + xi * xi;
    }
    let tot = reduce_sum(s);
    let tot_sq = reduce_sum(sq);
    let mean = tot / length;
    // E[x²] − E[x]² is exact in real arithmetic but suffers catastrophic
    // cancellation in f32 when the variance is tiny relative to the mean²
    // (e.g. a near-constant channel over a long time axis): the result can go
    // slightly negative, and `rsqrt(negative)` is NaN. The true variance is
    // ≥ 0, so clamp before the reciprocal-sqrt.
    let var_raw = tot_sq / length - mean * mean;
    let var = select(var_raw > 0.0f32, var_raw, 0.0f32);
    let eps = load(eps_buf[0]);
    let inv = rsqrt(var + eps);
    let g = load(gamma[row]).cast::<f32>();
    let bta = load(beta[row]).cast::<f32>();
    // Pass 2: strided affine-normalised store.
    for i in range(tid, length, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        store(out[rs + i], ((xi - mean) * inv * g + bta).cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::adain1d;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp + start).collect()
    }

    // Per-(b,c) instance norm over time, then scale/shift.
    fn naive_adain(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rows: usize,
        length: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * length];
        for r in 0..rows {
            let row = &x[r * length..(r + 1) * length];
            let mean: f32 = row.iter().sum::<f32>() / length as f32;
            let var: f32 = row.iter().map(|&v| v * v).sum::<f32>() / length as f32 - mean * mean;
            let inv = 1.0 / (var + eps).sqrt();
            for (t, &xi) in row.iter().enumerate() {
                out[r * length + t] = (xi - mean) * inv * gamma[r] + beta[r];
            }
        }
        out
    }

    fn adain_setup(rows: usize, length: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        let x_f = ramp(rows * length, 19, 3.0, 0.0);
        let gamma_f = ramp(rows, 7, 1.0, 1.0);
        let beta_f = ramp(rows, 5, 0.5, 0.0);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let gamma = unpack_f32(&pack_f32(&gamma_f, dt), dt);
        let beta = unpack_f32(&pack_f32(&beta_f, dt), dt);
        let expected = naive_adain(&x, &gamma, &beta, rows, length, eps);
        TestSetup::new(adain1d::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("gamma", pack_f32(&gamma_f, dt), dt))
            .input(TestBuffer::from_vec("beta", pack_f32(&beta_f, dt), dt))
            .input(TestBuffer::zeros("out", rows * length, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("length", length as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [1024, 1, 1])
    }

    // (batch*channels) rows; non-128-aligned length to exercise the strided path.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_adain1d(dt: DType) -> TestSetup { adain_setup(8, 300, dt) }
}

/// New-syntax bench: an AdaIN decoder block (512 channels, time 1024, batch 4).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::adain1d;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_adain1d(dt: DType) -> BenchSetup {
        let (batch, channels, length) = (4usize, 512usize, 1024usize);
        let rows = batch * channels;
        BenchSetup::new(adain1d::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * length, dt))
            .buffer(BenchBuffer::random("gamma", rows, dt))
            .buffer(BenchBuffer::random("beta", rows, dt))
            .buffer(BenchBuffer::zeros("out", rows * length, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("length", length as u32)
            .grid_3d(rows as u32, 1, 1, [1024, 1, 1])
            .bytes_moved((2 * rows * length * dt.size_bytes()) as u64)
            // Per row: mean (1·len) + variance (2·len) + normalize·γ+β (4·len).
            .flops((rows as u64) * (length as u64) * 7)
    }
}

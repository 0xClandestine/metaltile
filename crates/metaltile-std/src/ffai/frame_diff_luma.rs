//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Luma down-scale + frame-difference (SAD) — the motion-gating primitive
//! for the real-time video loop (multi-modal-support.md §7c / K7).
//!
//! Given two consecutive RGB frames, each output cell box-averages a
//! `ds × ds` block of each frame to a single BT.601 luma value and writes
//! the absolute difference `|luma₀ − luma₁|`. Summing the output gives a
//! cheap global motion score; the caller gates the (expensive) vision tower
//! on it — skip near-static frames, process only when motion exceeds a
//! threshold — so a 30 fps capture only pays tower cost a few times/sec.
//!
//! Down-scaling first makes the diff robust to sensor noise and sub-pixel
//! jitter (a per-pixel SAD is dominated by noise). Luma is linear in RGB, so
//! averaging RGB over the block then converting equals averaging luma.
//!
//! Layouts:
//!   frame0 / frame1  `[in_h, in_w, 3]` interleaved RGB   T
//!   out              `[out_h, out_w]` abs luma diff       T
//!   out_h = in_h / ds,  out_w = in_w / ds  (caller sizes so they divide)
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output cell `(oy, ox)` — dispatch with
//! `grid_1d(out_h * out_w, 256)`. Each thread reads a `ds × ds × 3` block
//! from BOTH frames; blocks tile the input exactly (`out_* · ds ≤ in_*`).

use metaltile::kernel;

#[kernel]
pub fn frame_diff_luma<T>(
    frame0: Tensor<T>,
    frame1: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] ds: u32,
) {
    let idx = program_id::<0>();
    let ox = idx % out_w;
    let oy = idx / out_w;
    let y0 = oy * ds;
    let x0 = ox * ds;
    // BT.601 luma weights.
    let wr = 0.299f32;
    let wg = 0.587f32;
    let wb = 0.114f32;
    let inv = 1.0f32 / (ds * ds).cast::<f32>();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for dy in range(0u32, ds, 1u32) {
        let row = (y0 + dy) * in_w;
        for dx in range(0u32, ds, 1u32) {
            let px = (row + x0 + dx) * 3u32;
            let r0 = load(frame0[px]).cast::<f32>();
            let g0 = load(frame0[px + 1u32]).cast::<f32>();
            let b0 = load(frame0[px + 2u32]).cast::<f32>();
            sum0 = sum0 + wr * r0 + wg * g0 + wb * b0;
            let r1 = load(frame1[px]).cast::<f32>();
            let g1 = load(frame1[px + 1u32]).cast::<f32>();
            let b1 = load(frame1[px + 2u32]).cast::<f32>();
            sum1 = sum1 + wr * r1 + wg * g1 + wb * b1;
        }
    }
    let luma0 = sum0 * inv;
    let luma1 = sum1 * inv;
    let diff = luma0 - luma1;
    let abs_diff = select(diff < 0.0f32, 0.0f32 - diff, diff);
    store(out[idx], abs_diff.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::frame_diff_luma;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| (start + i as f32 * step).rem_euclid(1.0)).collect()
    }

    /// Box-downscale BT.601 luma + abs diff oracle.
    fn naive(
        f0: &[f32],
        f1: &[f32],
        in_w: usize,
        out_h: usize,
        out_w: usize,
        ds: usize,
    ) -> Vec<f32> {
        let (wr, wg, wb) = (0.299f32, 0.587f32, 0.114f32);
        let inv = 1.0 / (ds * ds) as f32;
        let mut out = vec![0.0f32; out_h * out_w];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let (mut s0, mut s1) = (0.0f32, 0.0f32);
                for dy in 0..ds {
                    for dx in 0..ds {
                        let px = ((oy * ds + dy) * in_w + ox * ds + dx) * 3;
                        s0 += wr * f0[px] + wg * f0[px + 1] + wb * f0[px + 2];
                        s1 += wr * f1[px] + wg * f1[px + 1] + wb * f1[px + 2];
                    }
                }
                out[oy * out_w + ox] = (s0 * inv - s1 * inv).abs();
            }
        }
        out
    }

    fn setup(dt: DType, in_h: usize, in_w: usize, ds: usize) -> TestSetup {
        let out_h = in_h / ds;
        let out_w = in_w / ds;
        let f0_f = ramp(in_h * in_w * 3, 0.013, 0.1);
        let f1_f = ramp(in_h * in_w * 3, 0.011, 0.4);
        let f0 = unpack_f32(&pack_f32(&f0_f, dt), dt);
        let f1 = unpack_f32(&pack_f32(&f1_f, dt), dt);
        let expected = naive(&f0, &f1, in_w, out_h, out_w, ds);
        TestSetup::new(frame_diff_luma::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("frame0", pack_f32(&f0_f, dt), dt))
            .input(TestBuffer::from_vec("frame1", pack_f32(&f1_f, dt), dt))
            .input(TestBuffer::zeros("out", out_h * out_w, dt))
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("ds", ds as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(out_h * out_w, 256)
    }

    // 64×48 RGB → 8× downscale → 8×6 motion map.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 3e-2])]
    fn test_frame_diff_luma_8x(dt: DType) -> TestSetup { setup(dt, 48, 64, 8) }

    // 30×20 RGB → 5× downscale → 6×4 (non-power-of-two block).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 3e-2])]
    fn test_frame_diff_luma_5x(dt: DType) -> TestSetup { setup(dt, 20, 30, 5) }
}

/// New-syntax bench: 720p frame pair → 16× downscale motion map.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::frame_diff_luma;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_frame_diff_luma(dt: DType) -> BenchSetup {
        let (in_h, in_w, ds) = (720usize, 1280usize, 16usize);
        let out_h = in_h / ds;
        let out_w = in_w / ds;
        BenchSetup::new(frame_diff_luma::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("frame0", in_h * in_w * 3, dt))
            .buffer(BenchBuffer::random("frame1", in_h * in_w * 3, dt))
            .buffer(BenchBuffer::zeros("out", out_h * out_w, dt).output())
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("ds", ds as u32)
            .grid_1d(out_h * out_w, 256)
            .bytes_moved((2 * in_h * in_w * 3 * dt.size_bytes()) as u64)
    }
}

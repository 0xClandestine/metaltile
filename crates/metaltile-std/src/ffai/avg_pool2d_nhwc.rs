//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! NHWC average pooling — `k_h × k_w` box average with stride, channel-last.
//!
//! The pooling stage the Gemma 4 vision tower (and the tile/pan-and-scan VL
//! models) apply to downsample the patch grid before the multi-modal
//! projector — replaces a scalar CPU pool loop. Channel-last `[B, H, W, C]`
//! to match the towers' NHWC feature maps (sibling of
//! [`super::depthwise_conv2d_nhwc`]).
//!
//! Each output element averages the (valid) taps of its `k_h × k_w` window;
//! padding taps that fall outside the input are excluded from BOTH the sum
//! and the divisor (`count_include_pad = false`, the torch/Gemma default),
//! so edge windows divide by their real tap count.
//!
//! Layouts:
//!   input  `[batch, in_h,  in_w,  ch]`   T
//!   out    `[batch, out_h, out_w, ch]`   T
//!   out_h = (in_h + 2*pad - k_h) / stride + 1   (k_w → out_w)
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element `(n, oh, ow, c)` — dispatch with
//! `grid_1d(n_out, 256)` where `n_out = batch * out_h * out_w * ch`. `out_h`
//! / `out_w` must match the formula above.

use metaltile::kernel;

#[kernel]
pub fn avg_pool2d_nhwc<T>(
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] k_h: u32,
    #[constexpr] k_w: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
) {
    let idx = program_id::<0>();
    let c = idx % ch;
    let t1 = idx / ch;
    let ow = t1 % out_w;
    let t2 = t1 / out_w;
    let oh = t2 % out_h;
    let n = t2 / out_h;
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let n_base = n * in_h * in_w;
    let mut acc = 0.0f32;
    let mut count = 0.0f32;
    for ky in range(0u32, k_h, 1u32) {
        let ph = ph0 + ky;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k_w, 1u32) {
            let pw = pw0 + kx;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            let in_idx = (n_base + ih * in_w + iw) * ch + c;
            let x = load(input[in_idx]).cast::<f32>();
            acc = acc + select(valid, x, 0.0f32);
            count = count + select(valid, 1.0f32, 0.0f32);
        }
    }
    let mean = select(count > 0.0f32, acc / count, 0.0f32);
    store(out[idx], mean.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::avg_pool2d_nhwc;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    fn out_dim(in_d: usize, k: usize, stride: usize, pad: usize) -> usize {
        (in_d + 2 * pad - k) / stride + 1
    }

    /// NHWC average-pool oracle, `count_include_pad = false`.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        input: &[f32],
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let out_h = out_dim(in_h, k_h, stride, pad);
        let out_w = out_dim(in_w, k_w, stride, pad);
        let mut out = vec![0.0f32; batch * out_h * out_w * ch];
        for n in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for c in 0..ch {
                        let mut acc = 0.0f32;
                        let mut count = 0usize;
                        for ky in 0..k_h {
                            let ph = oh * stride + ky;
                            if ph < pad || ph >= pad + in_h {
                                continue;
                            }
                            let ih = ph - pad;
                            for kx in 0..k_w {
                                let pw = ow * stride + kx;
                                if pw < pad || pw >= pad + in_w {
                                    continue;
                                }
                                let iw = pw - pad;
                                acc += input[((n * in_h + ih) * in_w + iw) * ch + c];
                                count += 1;
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * ch + c] =
                            if count > 0 { acc / count as f32 } else { 0.0 };
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        kernel: Kernel,
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = out_dim(in_h, k_h, stride, pad);
        let out_w = out_dim(in_w, k_w, stride, pad);
        let n_out = batch * out_h * out_w * ch;
        let input_f = ramp(batch * in_h * in_w * ch, 13, 6.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = naive(&input, batch, ch, in_h, in_w, k_h, k_w, stride, pad);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("k_h", k_h as u32)
            .constexpr("k_w", k_w as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Gemma 4-style 3×3 stride-3 non-overlapping pool (the pooling kernel).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 3e-2])]
    fn test_avg_pool2d_nhwc_3x3_s3(dt: DType) -> TestSetup {
        setup(avg_pool2d_nhwc::kernel_ir_for(dt), 1, 16, 24, 24, 3, 3, 3, 0, dt)
    }

    // 2×2 stride-2 with padding-1 — exercises the partial-window divisor.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 3e-2])]
    fn test_avg_pool2d_nhwc_2x2_pad(dt: DType) -> TestSetup {
        setup(avg_pool2d_nhwc::kernel_ir_for(dt), 2, 8, 15, 15, 2, 2, 2, 1, dt)
    }
}

/// New-syntax bench: Gemma 4 vision pool (28×28 grid, 1152 ch, 3×3 stride-3).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::avg_pool2d_nhwc;

    #[bench(name = "ffai/pool/avg_pool2d_nhwc", dtypes = [f32, f16, bf16])]
    fn bench_avg_pool2d_nhwc(dt: DType) -> BenchSetup {
        let (batch, ch, in_h, in_w, k, stride, pad) =
            (1usize, 1152usize, 24usize, 24usize, 3usize, 3usize, 0usize);
        let out_h = (in_h + 2 * pad - k) / stride + 1;
        let out_w = (in_w + 2 * pad - k) / stride + 1;
        let n_out = batch * out_h * out_w * ch;
        BenchSetup::new(avg_pool2d_nhwc::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_h * in_w * ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("ch", ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("k_h", k as u32)
            .constexpr("k_w", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // k² window adds + 1 reciprocal-scale per output element.
            .flops((n_out as u64) * (k as u64 * k as u64 + 1))
    }
}

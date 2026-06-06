//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! NHWC depthwise 2D convolution — `groups == channels`, **channel-last**.
//!
//! Identical math to [`super::depthwise_conv2d`] (each output channel
//! convolves only its own input channel with a `k_h × k_w` filter), but
//! laid out channel-last to match the **FastVLM / FastViTHD (MobileCLIP)**
//! vision stem, which keeps its whole feature pyramid in NHWC. Wrapping the
//! NCHW kernel for that tower would force an NHWC↔NCHW transpose around
//! every depthwise conv; this variant indexes NHWC directly so the stem
//! stays channel-last end-to-end on the GPU (replacing the CPU
//! `concurrentPerform` depthwise loop in `FastVLMVision.swift`).
//!
//! Supports an independent `k_h` / `k_w` (FastViT mixes square and, in the
//! RepMixer/token-mixer paths, non-square depthwise kernels).
//!
//! Layouts (channel-last; the depthwise weight squeezed to `[ch, k_h, k_w]`
//! since `in_ch_per_group == 1`):
//!
//!   input    [batch, in_h,  in_w,  ch]    T
//!   weight   [ch, k_h, k_w]               T
//!   bias     [ch]                          T
//!   out      [batch, out_h, out_w, ch]    T
//!
//!   out_h = (in_h + 2*pad - dilation*(k_h-1) - 1) / stride + 1   (k_w → out_w)
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element `(n, oh, ow, c)` — dispatch with
//! `grid_1d(n_out, 256)` where `n_out = batch * out_h * out_w * ch` (NOT
//! `grid_3d`, which would launch `n_out × tpg` threads and stride past the
//! output). `out_h` / `out_w` must match the formula above; `bias` must
//! have `ch` elements. Padding/dilation taps outside the real input
//! contribute zero (index clamped to 0 and masked).

use metaltile::kernel;

#[kernel]
pub fn depthwise_conv2d_nhwc<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
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
    #[constexpr] dilation: u32,
) {
    // Flat NHWC output index → (n, oh, ow, c) with channel fastest, so the
    // flat index equals `((n*out_h + oh)*out_w + ow)*ch + c` and the output
    // store can write `out[idx]` directly.
    let idx = program_id::<0>();
    let c = idx % ch;
    let t1 = idx / ch;
    let ow = t1 % out_w;
    let t2 = t1 / out_w;
    let oh = t2 % out_h;
    let n = t2 / out_h;
    // Receptive-field anchor in the *padded* input frame (same as the NCHW
    // variant); only the buffer indexing differs (channel-last).
    let ph0 = oh * stride;
    let pw0 = ow * stride;
    let n_base = n * in_h * in_w;
    let w_c_base = c * k_h * k_w;
    let mut acc = load(bias[c]).cast::<f32>();
    for ky in range(0u32, k_h, 1u32) {
        let ph = ph0 + ky * dilation;
        let valid_h = (ph >= pad) & (ph < pad + in_h);
        let ih = select(valid_h, ph - pad, 0u32);
        for kx in range(0u32, k_w, 1u32) {
            let pw = pw0 + kx * dilation;
            let valid_w = (pw >= pad) & (pw < pad + in_w);
            let iw = select(valid_w, pw - pad, 0u32);
            let valid = valid_h & valid_w;
            // NHWC: input[((n*in_h + ih)*in_w + iw)*ch + c].
            let in_idx = (n_base + ih * in_w + iw) * ch + c;
            let x = load(input[in_idx]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let wt = load(weight[w_c_base + ky * k_w + kx]).cast::<f32>();
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::depthwise_conv2d_nhwc;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    fn out_dim(in_d: usize, k: usize, stride: usize, pad: usize, dilation: usize) -> usize {
        (in_d + 2 * pad - dilation * (k - 1) - 1) / stride + 1
    }

    /// Direct depthwise 2D conv oracle, **NHWC** input / output,
    /// `[ch, k_h, k_w]` weight. Padding/dilation taps zero. f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_depthwise_conv2d_nhwc(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        ch: usize,
        in_h: usize,
        in_w: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let out_h = out_dim(in_h, k_h, stride, pad, dilation);
        let out_w = out_dim(in_w, k_w, stride, pad, dilation);
        let mut out = vec![0.0f32; batch * out_h * out_w * ch];
        for n in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for c in 0..ch {
                        let mut acc = bias[c];
                        for ky in 0..k_h {
                            let ph = oh * stride + ky * dilation;
                            if ph < pad || ph >= pad + in_h {
                                continue;
                            }
                            let ih = ph - pad;
                            for kx in 0..k_w {
                                let pw = ow * stride + kx * dilation;
                                if pw < pad || pw >= pad + in_w {
                                    continue;
                                }
                                let iw = pw - pad;
                                let in_idx = ((n * in_h + ih) * in_w + iw) * ch + c;
                                let w_idx = (c * k_h + ky) * k_w + kx;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * ch + c] = acc;
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
        dilation: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = out_dim(in_h, k_h, stride, pad, dilation);
        let out_w = out_dim(in_w, k_w, stride, pad, dilation);
        let n_out = batch * out_h * out_w * ch;
        let input_f = ramp(batch * in_h * in_w * ch, 13, 6.0);
        let weight_f = ramp(ch * k_h * k_w, 11, 4.0);
        let bias_f = ramp(ch, 5, 2.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive_depthwise_conv2d_nhwc(
            &input, &weight, &bias, batch, ch, in_h, in_w, k_h, k_w, stride, pad, dilation,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
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
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // FastViT/MobileCLIP-style depthwise 3×3 stride-1 pad-1 (same-size).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_nhwc_3x3_s1(dt: DType) -> TestSetup {
        setup(depthwise_conv2d_nhwc::kernel_ir_for(dt), 1, 8, 16, 16, 3, 3, 1, 1, 1, dt)
    }

    // Strided downsample: depthwise 3×3 stride-2 pad-1 (halves H/W).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_nhwc_3x3_s2(dt: DType) -> TestSetup {
        setup(depthwise_conv2d_nhwc::kernel_ir_for(dt), 2, 6, 24, 24, 3, 3, 2, 1, 1, dt)
    }

    // Non-square depthwise (k_h ≠ k_w) — FastViT token-mixer style.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2])]
    fn test_depthwise_conv2d_nhwc_7x7(dt: DType) -> TestSetup {
        setup(depthwise_conv2d_nhwc::kernel_ir_for(dt), 1, 5, 20, 20, 7, 7, 1, 3, 1, dt)
    }
}

/// New-syntax bench for `depthwise_conv2d_nhwc` (FastViT stem shape: 64
/// channels, 112×112 feature map, depthwise 3×3 stride-2). Grid3D,
/// `grid_1d(n_out, 256)`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::depthwise_conv2d_nhwc;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_depthwise_conv2d_nhwc(dt: DType) -> BenchSetup {
        let (batch, ch, in_h, in_w, k, stride, pad, dilation) =
            (1usize, 64usize, 112usize, 112usize, 3usize, 2usize, 1usize, 1usize);
        let out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * out_h * out_w * ch;
        BenchSetup::new(depthwise_conv2d_nhwc::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_h * in_w * ch, dt))
            .buffer(BenchBuffer::random("weight", ch * k * k, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
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
            .constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // Depthwise (groups = ch ⇒ 1 in-channel/group): 2·N·ch·Ho·Wo·kh·kw.
            .flops(
                2 * (batch as u64)
                    * (ch as u64)
                    * (out_h as u64)
                    * (out_w as u64)
                    * (k as u64)
                    * (k as u64),
            )
    }
}

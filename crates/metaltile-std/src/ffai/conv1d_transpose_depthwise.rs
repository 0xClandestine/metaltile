//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Depthwise (grouped, `groups == channels`) transposed Conv1d.
//!
//! The StyleTTS2 / Kokoro `AdainResBlk1d` up-sample `pool` —
//! `ConvTranspose1d(C, C, kernel_size=3, stride=2, padding=1, groups=C)` —
//! where every channel has its own length-`k` kernel and never mixes with the
//! others. The dense [`super::conv1d_transpose`] sums over all input channels;
//! this variant restricts each output channel to its own input channel, so it
//! is the grouped op the decoder needs without an `O(C²)` block-diagonal weight.
//!
//! Gather (adjoint) form, exactly as the dense transpose: output position `op`
//! collects taps where `op + pad == ip*stride + kx*dilation`, i.e.
//! `ip = (op + pad − kx*dilation) / stride`, valid iff non-negative, divisible
//! by `stride`, and `ip < in_len`. One thread per output element — no scatter.
//!
//! Layouts:
//!   input  `[channels, in_len]`    T
//!   weight `[channels, k]`         T   (per-channel kernel; depthwise `[C,1,k]`)
//!   bias   `[channels]`            T
//!   out    `[channels, out_len]`   T
//!   out_len = (in_len − 1)·stride − 2·pad + dilation·(k − 1) + 1
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D, one thread per output element; grid width == `channels · out_len`.
//!   - `out_len` matches the formula above (caller-computed).
//!   - `weight` is `channels · k`; `bias` is `channels`.

use metaltile::kernel;

#[kernel]
pub fn ffai_conv1d_transpose_depthwise<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] channels: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let c = idx / out_len;
    let opp = op + pad;
    let in_base = c * in_len;
    let w_base = c * k;
    let mut acc = load(bias[c]).cast::<f32>();
    for kx in range(0u32, k, 1u32) {
        let tap = kx * dilation;
        let has = opp >= tap;
        let num = select(has, opp - tap, 0u32);
        let on_grid = (num % stride) == 0u32;
        let ip = num / stride;
        let valid = has & on_grid & (ip < in_len);
        let ix = select(valid, ip, 0u32);
        let x = load(input[in_base + ix]).cast::<f32>();
        let x_m = select(valid, x, 0.0f32);
        let wt = load(weight[w_base + kx]).cast::<f32>();
        acc = acc + x_m * wt;
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_conv1d_transpose_depthwise;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Depthwise ConvTranspose1d oracle (per-channel, no cross-channel sum).
    #[allow(clippy::too_many_arguments)]
    fn naive(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        channels: usize,
        in_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + 1 - 2 * pad;
        let mut out = vec![0.0f32; channels * out_len];
        for c in 0..channels {
            for op in 0..out_len {
                let mut acc = bias[c];
                let opp = op + pad;
                for kx in 0..k {
                    let tap = kx * dilation;
                    if opp < tap {
                        continue;
                    }
                    let num = opp - tap;
                    if !num.is_multiple_of(stride) {
                        continue;
                    }
                    let ip = num / stride;
                    if ip < in_len {
                        acc += input[c * in_len + ip] * weight[c * k + kx];
                    }
                }
                out[c * out_len + op] = acc;
            }
        }
        out
    }

    // The StyleTTS2 pool: k3, stride 2, pad 1 (out_len = 2·in_len − 1).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_conv1d_transpose_depthwise(dt: DType) -> TestSetup {
        let (channels, in_len, k, stride, pad, dilation) =
            (6usize, 13usize, 3usize, 2usize, 1usize, 1usize);
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + 1 - 2 * pad;
        let input_f = ramp(channels * in_len, 17, 4.0);
        let weight_f = ramp(channels * k, 7, 2.0);
        let bias_f = ramp(channels, 5, 0.5);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let exp = naive(&input, &weight, &bias, channels, in_len, k, stride, pad, dilation);
        TestSetup::new(ffai_conv1d_transpose_depthwise::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", channels * out_len, dt))
            .constexpr("channels", channels as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d((channels * out_len) as u32, 1, 1, [1, 1, 1])
    }
}

/// New-syntax bench: a decoder up-sample pool (1090 ch × 65 → 129).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_conv1d_transpose_depthwise;

    #[bench(name = "ffai/conv/conv1d_transpose_depthwise", dtypes = [f32, f16, bf16])]
    fn bench_conv1d_transpose_depthwise(dt: DType) -> BenchSetup {
        let (channels, in_len, k, stride, pad) = (1090usize, 65usize, 3usize, 2usize, 1usize);
        let out_len = (in_len - 1) * stride + (k - 1) + 1 - 2 * pad;
        let n_out = channels * out_len;
        BenchSetup::new(ffai_conv1d_transpose_depthwise::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", channels * in_len, dt))
            .buffer(BenchBuffer::random("weight", channels * k, dt))
            .buffer(BenchBuffer::random("bias", channels, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("channels", channels as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", 1u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // k taps (mul-add) per output element.
            .flops((n_out as u64) * (k as u64) * 2)
    }
}

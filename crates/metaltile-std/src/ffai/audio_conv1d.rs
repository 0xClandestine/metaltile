//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Wide-stride multi-channel 1D convolution — the STT audio patch
//! embedding.
//!
//! After the log-Mel front-end (`mel_spectrogram`), a speech encoder
//! (Whisper, Qwen-Omni audio, Parakeet) downsamples the Mel sequence
//! with one or two strided 1D convolutions before the transformer
//! stack. Whisper's stem is `Conv1d(n_mels→d_model, k=3, s=1)` then
//! `Conv1d(d_model→d_model, k=3, s=2)`; the strided second conv halves
//! the time axis. This is a *dense, multi-channel, strided* conv —
//! distinct from the depthwise single-channel `conv1d_causal_step` in
//! `ssm.rs`, which streams one SSM-state column with `groups == channels`.
//!
//! Layouts (NCL — the PyTorch `nn.Conv1d` convention):
//!
//!   input    [batch, in_ch,  in_len]    T
//!   weight   [out_ch, in_ch, k]         T
//!   bias     [out_ch]                   T
//!   out      [batch, out_ch, out_len]   T
//!
//!   out_len = (in_len + 2*pad - k) / stride + 1
//!
//! One thread per output element `(n, oc, op)`. The thread walks the
//! `in_ch × k` receptive field, accumulating in fp32. Padding taps
//! (position outside the real input) contribute zero — the load is
//! clamped to index 0 and masked. Indices stay in the *padded* frame so
//! every value is a non-negative u32 (no i32 arithmetic). Generic over T.
//!
//! Codegen-only. Correctness validated by `audio_conv1d_gpu_correctness`.

use metaltile::kernel;

#[kernel]
pub fn audio_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
) {
    // Flat output index → (n, oc, op). One thread per output element.
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    // Receptive-field anchor in the *padded* input frame: tap `kx` of
    // output position `op` lands at padded index `op*stride + kx`, which
    // maps to real input index `p - pad`, valid iff `pad <= p < pad+in_len`.
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let w_oc_stride = in_ch * k;
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let w_ic_base = oc * w_oc_stride + ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = p0 + kx;
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let wt = load(weight[w_ic_base + kx]).cast::<f32>();
            acc = acc + x_m * wt;
        }
    }
    store(out[idx], acc.cast::<T>());
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

    fn ramp(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| ((i + seed) as f32 % scale) / scale * 2.0 - 1.0).collect()
    }

    fn naive_conv1d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        out_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            let in_idx = (n * in_ch + ic) * in_len + ix;
                            let w_idx = (oc * in_ch + ic) * k + kx;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    fn make_setup(
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dt: DType,
        input_seed: usize,
        weight_seed: usize,
        bias_seed: usize,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let input_f32 = ramp(batch * in_ch * in_len, input_seed, 18.0);
        let weight_f32 = ramp(out_ch * in_ch * k, weight_seed, 20.0);
        let bias_f32 = ramp(out_ch, bias_seed, 3.0);
        let expected = naive_conv1d(
            &input_f32,
            &weight_f32,
            &bias_f32,
            batch,
            in_ch,
            in_len,
            out_ch,
            out_len,
            k,
            stride,
            pad,
        );
        let mut kernel = audio_conv1d::kernel_ir_for(dt);
        kernel.mode = KernelMode::Grid3D;
        let tpg = 256u32;
        let grid_x = (n_out as u32).div_ceil(tpg);
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("input", pack(&input_f32, dt), dt))
            .input(TestBuffer::from_vec("weight", pack(&weight_f32, dt), dt))
            .input(TestBuffer::from_vec("bias", pack(&bias_f32, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; n_out], dt), dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(grid_x, 1, 1, [tpg, 1, 1])
    }

    #[test_kernel(name = "ffai/conv/audio_conv1d_stride1", dtypes = [f32], tol = 1e-3)]
    fn test_audio_conv1d_stride1_f32(dt: DType) -> TestSetup {
        make_setup(1, 8, 50, 16, 3, 1, 1, dt, 37, 41, 7)
    }

    #[test_kernel(name = "ffai/conv/audio_conv1d_stride2", dtypes = [f32], tol = 1e-3)]
    fn test_audio_conv1d_stride2_f32(dt: DType) -> TestSetup {
        make_setup(2, 12, 64, 12, 3, 2, 1, dt, 29, 31, 5)
    }

    #[test_kernel(name = "ffai/conv/audio_conv1d_wide_stride", dtypes = [f32], tol = 1e-3)]
    fn test_audio_conv1d_wide_stride_f32(dt: DType) -> TestSetup {
        make_setup(1, 4, 100, 8, 10, 5, 0, dt, 23, 17, 3)
    }

    #[test_kernel(name = "ffai/conv/audio_conv1d_f16", dtypes = [f16], tol = 5e-2)]
    fn test_audio_conv1d_f16(dt: DType) -> TestSetup {
        make_setup(1, 8, 40, 8, 3, 2, 1, dt, 37, 41, 7)
    }

    #[test_kernel(name = "ffai/conv/audio_conv1d_bf16", dtypes = [bf16], tol = 1e-1)]
    fn test_audio_conv1d_bf16(dt: DType) -> TestSetup {
        make_setup(1, 6, 32, 6, 3, 1, 1, dt, 23, 17, 3)
    }
}

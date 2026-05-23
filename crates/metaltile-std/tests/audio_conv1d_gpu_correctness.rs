//! End-to-end GPU correctness for `ffai::audio_conv1d` — the wide-stride
//! multi-channel 1D convolution that downsamples the STT audio sequence.
//!
//! Validates against a four-loop CPU reference. Covers:
//!   - Whisper-style stride-1 k=3 conv (the first stem conv)
//!   - Whisper-style stride-2 k=3 conv (the time-halving stem conv)
//!   - a padded conv exercising the in-kernel padding clamp
//!   - f32 / f16 / bf16
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::audio_conv1d::audio_conv1d;

#[derive(Clone, Copy)]
struct Conv1dShape {
    batch: usize,
    in_ch: usize,
    in_len: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl Conv1dShape {
    fn out_len(&self) -> usize { (self.in_len + 2 * self.pad - self.k) / self.stride + 1 }
}

/// CPU reference: NCL input, OIK weight, NCL output. Padding taps
/// contribute zero. All f32.
fn naive_conv1d(input: &[f32], weight: &[f32], bias: &[f32], s: &Conv1dShape) -> Vec<f32> {
    let out_len = s.out_len();
    let mut out = vec![0.0f32; s.batch * s.out_ch * out_len];
    for n in 0..s.batch {
        for oc in 0..s.out_ch {
            for op in 0..out_len {
                let mut acc = bias[oc];
                for ic in 0..s.in_ch {
                    for kx in 0..s.k {
                        let p = op * s.stride + kx;
                        if p < s.pad || p >= s.pad + s.in_len {
                            continue;
                        }
                        let ix = p - s.pad;
                        let in_idx = (n * s.in_ch + ic) * s.in_len + ix;
                        let w_idx = (oc * s.in_ch + ic) * s.k + kx;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
                out[(n * s.out_ch + oc) * out_len + op] = acc;
            }
        }
    }
    out
}

fn run_conv1d(input: &[f32], weight: &[f32], bias: &[f32], dt: Dt, s: &Conv1dShape) -> Vec<f32> {
    let out_len = s.out_len();
    let n_out = s.batch * s.out_ch * out_len;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("bias".into(), pack_bytes(bias, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("batch".into(), u(s.batch));
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_len".into(), u(s.in_len));
    buffers.insert("out_ch".into(), u(s.out_ch));
    buffers.insert("out_len".into(), u(out_len));
    buffers.insert("k".into(), u(s.k));
    buffers.insert("stride".into(), u(s.stride));
    buffers.insert("pad".into(), u(s.pad));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = audio_conv1d::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("audio_conv1d dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

#[test]
fn audio_conv1d_stride1_matches_naive_f32() {
    let _g = gpu_lock();
    // Whisper stem conv #1: n_mels→d_model, k=3, stride 1, pad 1.
    let s = Conv1dShape { batch: 1, in_ch: 8, in_len: 50, out_ch: 16, k: 3, stride: 1, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_len, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.k, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let expected = naive_conv1d(&input, &weight, &bias, &s);
    let actual = run_conv1d(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "audio_conv1d stride1 f32: max |diff| = {diff:.2e}");
}

#[test]
fn audio_conv1d_stride2_matches_naive_f32() {
    let _g = gpu_lock();
    // Whisper stem conv #2: d_model→d_model, k=3, stride 2 (halves time).
    let s = Conv1dShape { batch: 2, in_ch: 12, in_len: 64, out_ch: 12, k: 3, stride: 2, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_len, 29, 14.0);
    let weight = ramp(s.out_ch * s.in_ch * s.k, 31, 15.0);
    let bias = ramp(s.out_ch, 5, 2.0);
    let expected = naive_conv1d(&input, &weight, &bias, &s);
    let actual = run_conv1d(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "audio_conv1d stride2 f32: max |diff| = {diff:.2e}");
}

#[test]
fn audio_conv1d_wide_stride_no_pad_matches_naive_f32() {
    let _g = gpu_lock();
    // Wide-stride patch-embed-style conv: k=10, stride 5, no padding.
    let s = Conv1dShape { batch: 1, in_ch: 4, in_len: 100, out_ch: 8, k: 10, stride: 5, pad: 0 };
    let input = ramp(s.batch * s.in_ch * s.in_len, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.k, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv1d(&input, &weight, &bias, &s);
    let actual = run_conv1d(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "audio_conv1d wide-stride f32: max |diff| = {diff:.2e}");
}

#[test]
fn audio_conv1d_matches_naive_f16() {
    let _g = gpu_lock();
    let s = Conv1dShape { batch: 1, in_ch: 8, in_len: 40, out_ch: 8, k: 3, stride: 2, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_len, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.k, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_conv1d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv1d(&input, &weight, &bias, Dt::F16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 5e-2, "audio_conv1d f16: max |diff| = {diff:.2e}");
}

#[test]
fn audio_conv1d_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = Conv1dShape { batch: 1, in_ch: 6, in_len: 32, out_ch: 6, k: 3, stride: 1, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_len, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.k, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_conv1d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv1d(&input, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-1, "audio_conv1d bf16: max |diff| = {diff:.2e}");
}

//! End-to-end GPU correctness for `ffai::vocoder` — the inverse-STFT
//! overlap-add waveform synthesis for TTS vocoders.
//!
//! The decisive test is a round-trip: take a known waveform, forward-STFT
//! it on the CPU, feed the spectrogram to the kernel, and check the
//! kernel reconstructs the original signal. With a COLA-satisfying
//! window (periodic Hann, hop = n_fft/4) the iSTFT is an exact inverse
//! away from the signal edges, so this pins both the inverse DFT and the
//! overlap-add normalisation. A second test compares against a direct
//! CPU iSTFT reference that mirrors the kernel's arithmetic.
//!
//! Coverage: f32 / f16 / bf16.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::vocoder::vocoder_istft;

#[derive(Clone, Copy)]
struct StftShape {
    n_frames: usize,
    n_fft: usize,
    hop_length: usize,
}

impl StftShape {
    fn n_freq(&self) -> usize { self.n_fft / 2 + 1 }
    fn out_len(&self) -> usize { (self.n_frames - 1) * self.hop_length + self.n_fft }
}

/// Periodic Hann window.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

/// Forward STFT of a real signal: for each frame, window then DFT, keep
/// the n_freq non-redundant bins. Returns (re, im) planes.
fn forward_stft(signal: &[f32], window: &[f32], s: &StftShape) -> (Vec<f32>, Vec<f32>) {
    let n_freq = s.n_freq();
    let mut re = vec![0.0f32; s.n_frames * n_freq];
    let mut im = vec![0.0f32; s.n_frames * n_freq];
    let neg_two_pi_over_n = -2.0 * std::f32::consts::PI / s.n_fft as f32;
    for f in 0..s.n_frames {
        let start = f * s.hop_length;
        for k in 0..n_freq {
            let angle_step = neg_two_pi_over_n * k as f32;
            let mut r = 0.0f32;
            let mut i = 0.0f32;
            for t in 0..s.n_fft {
                let xw = signal[start + t] * window[t];
                let angle = angle_step * t as f32;
                r += xw * angle.cos();
                i += xw * angle.sin();
            }
            re[f * n_freq + k] = r;
            im[f * n_freq + k] = i;
        }
    }
    (re, im)
}

/// Direct CPU iSTFT mirroring the kernel: per output sample, gather
/// covering frames, inverse-DFT with Hermitian symmetry, COLA-normalise.
fn naive_istft(spec_re: &[f32], spec_im: &[f32], window: &[f32], s: &StftShape) -> Vec<f32> {
    let n_freq = s.n_freq();
    let out_len = s.out_len();
    let nyquist = s.n_fft / 2;
    let inv_n = 1.0 / s.n_fft as f32;
    let two_pi_over_n = 2.0 * std::f32::consts::PI / s.n_fft as f32;
    let mut out = vec![0.0f32; out_len];

    for (t, o) in out.iter_mut().enumerate() {
        let f_hi = (t / s.hop_length).min(s.n_frames - 1);
        let f_lo = if t + 1 > s.n_fft { (t + 1 - s.n_fft).div_ceil(s.hop_length) } else { 0 };
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for f in f_lo..=f_hi {
            let tau = t - f * s.hop_length;
            let angle_step = two_pi_over_n * tau as f32;
            let row = f * n_freq;
            let mut sample = 0.0f32;
            for k in 0..n_freq {
                let re = spec_re[row + k];
                let im = spec_im[row + k];
                let angle = angle_step * k as f32;
                let contrib = re * angle.cos() - im * angle.sin();
                let w = if k == 0 || k == nyquist { 1.0 } else { 2.0 };
                sample += w * contrib;
            }
            sample *= inv_n;
            let win = window[tau];
            num += sample * win;
            den += win * win;
        }
        *o = if den > 1e-8 { num / den } else { 0.0 };
    }
    out
}

fn run_istft(spec_re: &[f32], spec_im: &[f32], window: &[f32], dt: Dt, s: &StftShape) -> Vec<f32> {
    let out_len = s.out_len();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("spec_re".into(), pack_bytes(spec_re, dt));
    buffers.insert("spec_im".into(), pack_bytes(spec_im, dt));
    buffers.insert("window".into(), pack_bytes(window, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; out_len], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("n_frames".into(), u(s.n_frames));
    buffers.insert("n_fft".into(), u(s.n_fft));
    buffers.insert("n_freq".into(), u(s.n_freq()));
    buffers.insert("hop_length".into(), u(s.hop_length));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = vocoder_istft::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 128usize;
    let grid = out_len.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("vocoder_istft dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(out_len);
    out
}

#[test]
fn vocoder_istft_round_trip_reconstructs_signal_f32() {
    let _g = gpu_lock();
    // STFT then iSTFT must reconstruct the original signal (away from the
    // edges where the overlap-add window energy tapers). hop = n_fft/4
    // periodic-Hann satisfies COLA.
    let s = StftShape { n_frames: 8, n_fft: 16, hop_length: 4 };
    let window = hann(s.n_fft);
    let signal: Vec<f32> = (0..s.out_len())
        .map(|i| (i as f32 * 0.3).sin() * 0.5 + (i as f32 * 0.11).cos() * 0.3)
        .collect();
    let (re, im) = forward_stft(&signal, &window, &s);
    let actual = run_istft(&re, &im, &window, Dt::F32, &s);

    // Compare only the COLA-valid interior — the first/last n_fft samples
    // are only partially covered by frames so the window energy is low.
    let edge = s.n_fft;
    let mut max_diff = 0.0f32;
    for t in edge..(s.out_len() - edge) {
        max_diff = max_diff.max((actual[t] - signal[t]).abs());
    }
    assert!(max_diff < 2e-3, "iSTFT round-trip f32: max |diff| = {max_diff:.2e}");
}

#[test]
fn vocoder_istft_matches_naive_f32() {
    let _g = gpu_lock();
    // Direct comparison against the CPU iSTFT reference over the whole
    // output, including the partially-covered edges.
    let s = StftShape { n_frames: 6, n_fft: 16, hop_length: 4 };
    let window = hann(s.n_fft);
    let signal: Vec<f32> =
        (0..s.out_len()).map(|i| (i as f32 * 0.21).sin() + (i as f32 * 0.07).cos() * 0.4).collect();
    let (re, im) = forward_stft(&signal, &window, &s);
    let expected = naive_istft(&re, &im, &window, &s);
    let actual = run_istft(&re, &im, &window, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "iSTFT vs naive f32: max |diff| = {diff:.2e}");
}

#[test]
fn vocoder_istft_kokoro_shape_matches_naive_f32() {
    let _g = gpu_lock();
    // Kokoro-style iSTFTNet tail: small n_fft=20, hop 5.
    let s = StftShape { n_frames: 10, n_fft: 20, hop_length: 5 };
    let window = hann(s.n_fft);
    let signal: Vec<f32> = (0..s.out_len()).map(|i| (i as f32 * 0.17).sin() * 0.6).collect();
    let (re, im) = forward_stft(&signal, &window, &s);
    let expected = naive_istft(&re, &im, &window, &s);
    let actual = run_istft(&re, &im, &window, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "iSTFT Kokoro-shape f32: max |diff| = {diff:.2e}");
}

#[test]
fn vocoder_istft_matches_naive_f16() {
    let _g = gpu_lock();
    let s = StftShape { n_frames: 6, n_fft: 16, hop_length: 4 };
    let window = hann(s.n_fft);
    let signal: Vec<f32> = (0..s.out_len()).map(|i| (i as f32 * 0.21).sin() * 0.5).collect();
    let (re, im) = forward_stft(&signal, &window, &s);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_istft(&round(&re), &round(&im), &round(&window), &s);
    let actual = run_istft(&re, &im, &window, Dt::F16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 3e-2, "iSTFT f16: max |diff| = {diff:.2e}");
}

#[test]
fn vocoder_istft_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = StftShape { n_frames: 6, n_fft: 16, hop_length: 4 };
    let window = hann(s.n_fft);
    let signal: Vec<f32> = (0..s.out_len()).map(|i| (i as f32 * 0.21).sin() * 0.5).collect();
    let (re, im) = forward_stft(&signal, &window, &s);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_istft(&round(&re), &round(&im), &round(&window), &s);
    let actual = run_istft(&re, &im, &window, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1.5e-1, "iSTFT bf16: max |diff| = {diff:.2e}");
}

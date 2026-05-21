//! End-to-end GPU correctness for `ffai::mel_spectrogram` — the fused
//! STFT + Mel-filterbank + log audio front-end.
//!
//! The CPU reference does the same direct DFT the kernel does (not an
//! FFT), so f32 comparison is tight. Covers:
//!   - a small STFT with a Hann window and an identity-band filterbank
//!     (each Mel bin = one FFT bin) — pins the DFT + log path
//!   - a multi-tap triangular filterbank — pins the filterbank matmul
//!   - f32 / f16
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::mel_spectrogram::mel_spectrogram;

#[derive(Clone, Copy)]
struct MelShape {
    n_samples: usize,
    n_fft: usize,
    n_mels: usize,
    hop_length: usize,
    log_eps: f32,
}

impl MelShape {
    fn n_freq(&self) -> usize { self.n_fft / 2 + 1 }
    fn n_frames(&self) -> usize {
        // Frames that fully fit in n_samples.
        (self.n_samples - self.n_fft) / self.hop_length + 1
    }
}

/// CPU reference: per (frame, mel_bin), direct DFT → power → filterbank →
/// log, all f32 — mirrors the kernel exactly.
fn naive_mel(audio: &[f32], window: &[f32], mel_weight: &[f32], s: &MelShape) -> Vec<f32> {
    let n_freq = s.n_freq();
    let n_frames = s.n_frames();
    let mut out = vec![0.0f32; n_frames * s.n_mels];
    let neg_two_pi_over_n = -2.0 * std::f32::consts::PI / s.n_fft as f32;

    for frame in 0..n_frames {
        let frame_start = frame * s.hop_length;
        // Power spectrum for this frame.
        let mut power = vec![0.0f32; n_freq];
        for (k, p) in power.iter_mut().enumerate() {
            let angle_step = neg_two_pi_over_n * k as f32;
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for t in 0..s.n_fft {
                let xw = audio[frame_start + t] * window[t];
                let angle = angle_step * t as f32;
                re += xw * angle.cos();
                im += xw * angle.sin();
            }
            *p = re * re + im * im;
        }
        // Filterbank + log.
        for mel_bin in 0..s.n_mels {
            let mut acc = 0.0f32;
            for (k, &p) in power.iter().enumerate() {
                acc += mel_weight[mel_bin * n_freq + k] * p;
            }
            out[frame * s.n_mels + mel_bin] = (acc + s.log_eps).ln();
        }
    }
    out
}

/// Periodic Hann window of length n.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn run_mel(audio: &[f32], window: &[f32], mel_weight: &[f32], dt: Dt, s: &MelShape) -> Vec<f32> {
    let n_out = s.n_frames() * s.n_mels;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("audio".into(), pack_bytes(audio, dt));
    buffers.insert("window".into(), pack_bytes(window, dt));
    buffers.insert("mel_weight".into(), pack_bytes(mel_weight, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("n_fft".into(), u(s.n_fft));
    buffers.insert("n_freq".into(), u(s.n_freq()));
    buffers.insert("n_mels".into(), u(s.n_mels));
    buffers.insert("hop_length".into(), u(s.hop_length));
    buffers.insert("log_eps".into(), s.log_eps.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mel_spectrogram::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 128usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("mel_spectrogram dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

/// Identity filterbank: Mel bin m = FFT bin m (n_mels == n_freq).
fn identity_filterbank(n_freq: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n_freq * n_freq];
    for k in 0..n_freq {
        w[k * n_freq + k] = 1.0;
    }
    w
}

/// Simple overlapping triangular filterbank — each Mel bin sums a small
/// span of FFT bins with triangular weights.
fn triangular_filterbank(n_mels: usize, n_freq: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n_mels * n_freq];
    for m in 0..n_mels {
        // Center each filter proportionally across the frequency axis.
        let center = (m + 1) * n_freq / (n_mels + 1);
        let span = 2usize.max(n_freq / n_mels);
        for k in 0..n_freq {
            let dist = (k as isize - center as isize).unsigned_abs();
            if dist < span {
                w[m * n_freq + k] = 1.0 - dist as f32 / span as f32;
            }
        }
    }
    w
}

#[test]
fn mel_spectrogram_identity_filterbank_matches_naive_f32() {
    let _g = gpu_lock();
    // Identity filterbank → output is log of the raw power spectrum.
    let s = MelShape { n_samples: 64, n_fft: 16, n_mels: 9, hop_length: 8, log_eps: 1e-6 };
    assert_eq!(s.n_mels, s.n_freq(), "identity filterbank needs n_mels == n_freq");
    let audio: Vec<f32> =
        (0..s.n_samples).map(|i| (i as f32 * 0.3).sin() * 0.5 + (i as f32 * 0.11).cos()).collect();
    let window = hann(s.n_fft);
    let mel_weight = identity_filterbank(s.n_freq());
    let expected = naive_mel(&audio, &window, &mel_weight, &s);
    let actual = run_mel(&audio, &window, &mel_weight, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "mel identity f32: max |diff| = {diff:.2e}");
}

#[test]
fn mel_spectrogram_triangular_filterbank_matches_naive_f32() {
    let _g = gpu_lock();
    let s = MelShape { n_samples: 160, n_fft: 32, n_mels: 12, hop_length: 16, log_eps: 1e-5 };
    let audio: Vec<f32> =
        (0..s.n_samples).map(|i| (i as f32 * 0.21).sin() + (i as f32 * 0.07).cos() * 0.3).collect();
    let window = hann(s.n_fft);
    let mel_weight = triangular_filterbank(s.n_mels, s.n_freq());
    let expected = naive_mel(&audio, &window, &mel_weight, &s);
    let actual = run_mel(&audio, &window, &mel_weight, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 3e-3, "mel triangular f32: max |diff| = {diff:.2e}");
}

#[test]
fn mel_spectrogram_matches_naive_f16() {
    let _g = gpu_lock();
    let s = MelShape { n_samples: 96, n_fft: 16, n_mels: 8, hop_length: 8, log_eps: 1e-4 };
    let audio: Vec<f32> = (0..s.n_samples).map(|i| (i as f32 * 0.27).sin() * 0.4).collect();
    let window = hann(s.n_fft);
    let mel_weight = triangular_filterbank(s.n_mels, s.n_freq());
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_mel(&round(&audio), &round(&window), &round(&mel_weight), &s);
    let actual = run_mel(&audio, &window, &mel_weight, Dt::F16, &s);
    // log of a small power value amplifies f16 noise — relative check.
    let mut max_rel = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1.0));
    }
    assert!(max_rel < 5e-2, "mel f16: max rel = {max_rel:.2e}");
}

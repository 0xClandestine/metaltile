//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! **Magnitude** log-Mel spectrogram — the `|STFT|` (amplitude) front-end,
//! as opposed to the power (`|STFT|²`) front-end of
//! [`super::mel_spectrogram`].
//!
//! The Gemma 4 audio encoder (and several streaming-ASR front-ends) project
//! the **magnitude** spectrum `sqrt(re² + im²)` — not the power spectrum —
//! through the Mel filterbank before the log. Feeding power into a
//! magnitude-trained filterbank shifts the log-Mel energies and degrades the
//! encoder; this kernel is the amplitude-correct front-end that replaces the
//! Gemma 4 CPU DFT (`Gemma4AudioFrontend`).
//!
//! Identical to `mel_spectrogram` except the per-frequency contribution is
//! the magnitude `sqrt(re²+im²)` rather than the power `re²+im²`. Direct DFT
//! per `(frame, mel_bin)` thread; generic over T; accumulation in fp32.
//!
//! Layouts:
//!   audio       `[n_samples]`             T
//!   window      `[n_fft]`                 T   (e.g. Hann)
//!   mel_weight  `[n_mels, n_freq]`        T   (Mel filterbank)
//!   out         `[n_frames, n_mels]`      T   (log-Mel)
//!   n_freq = n_fft/2 + 1
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output `(frame, mel_bin)` — dispatch with
//! `grid_1d(n_frames * n_mels, 256)`. `n_samples >= (n_frames-1)*hop +
//! n_fft` is a caller precondition.

use metaltile::kernel;

#[kernel(
    bench(
        op="mel_spectrogram",
        subop="mel_spectrogram_magnitude",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Grid3D,
    )
)]
pub fn mel_spectrogram_magnitude<T>(
    audio: Tensor<T>,
    window: Tensor<T>,
    mel_weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] n_mels: u32,
    #[constexpr] hop_length: u32,
    #[constexpr] log_eps: f32,
) {
    let idx = program_id::<0>();
    let mel_bin = idx % n_mels;
    let frame = idx / n_mels;
    let frame_start = frame * hop_length;
    let n_fft_f = n_fft.cast::<f32>();
    let neg_two_pi_over_n = -6.283185307179586f32 / n_fft_f;
    let mel_row = mel_bin * n_freq;
    let mut mel_acc = 0.0f32;
    for k in range(0u32, n_freq, 1u32) {
        let k_f = k.cast::<f32>();
        let angle_step = neg_two_pi_over_n * k_f;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for t in range(0u32, n_fft, 1u32) {
            let sample = load(audio[frame_start + t]).cast::<f32>();
            let win = load(window[t]).cast::<f32>();
            let xw = sample * win;
            let angle = angle_step * t.cast::<f32>();
            re = re + xw * cos(angle);
            im = im + xw * sin(angle);
        }
        // Magnitude spectrum: |STFT| = sqrt(re² + im²) (vs. power re²+im²).
        let mag = sqrt(re * re + im * im);
        let w = load(mel_weight[mel_row + k]).cast::<f32>();
        mel_acc = mel_acc + w * mag;
    }
    let log_mel = log(mel_acc + log_eps);
    store(out[idx], log_mel.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mel_spectrogram_magnitude;
    use crate::utils::{pack_f32, unpack_f32};

    const PI: f32 = std::f32::consts::PI;

    fn hann(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos()).collect()
    }

    fn triangular_filterbank(n_mels: usize, n_freq: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n_mels * n_freq];
        for m in 0..n_mels {
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

    /// Direct-DFT **magnitude** log-Mel oracle (mirrors the kernel in f32).
    #[allow(clippy::too_many_arguments)]
    fn naive_mel_mag(
        audio: &[f32],
        window: &[f32],
        mel_weight: &[f32],
        n_fft: usize,
        n_freq: usize,
        n_mels: usize,
        hop_length: usize,
        n_frames: usize,
        log_eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_frames * n_mels];
        let neg_two_pi_over_n = -2.0 * PI / n_fft as f32;
        for frame in 0..n_frames {
            let frame_start = frame * hop_length;
            let mut mag = vec![0.0f32; n_freq];
            for (k, p) in mag.iter_mut().enumerate() {
                let angle_step = neg_two_pi_over_n * k as f32;
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for t in 0..n_fft {
                    let xw = audio[frame_start + t] * window[t];
                    let angle = angle_step * t as f32;
                    re += xw * angle.cos();
                    im += xw * angle.sin();
                }
                *p = (re * re + im * im).sqrt();
            }
            for mel_bin in 0..n_mels {
                let mut acc = 0.0f32;
                for (k, &p) in mag.iter().enumerate() {
                    acc += mel_weight[mel_bin * n_freq + k] * p;
                }
                out[frame * n_mels + mel_bin] = (acc + log_eps).ln();
            }
        }
        out
    }

    // Looser f32 tol than the power-mel sibling: magnitude folds a `sqrt`
    // before the filterbank + log, which amplifies the (benign) GPU↔CPU
    // DFT accumulation-order difference. Still tight on O(±10) log-Mels.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1.5e-2, 1e-1, 4e-1])]
    fn test_mel_spectrogram_magnitude(dt: DType) -> TestSetup {
        let (n_samples, n_fft, n_mels, hop_length, log_eps) =
            (160usize, 32usize, 12usize, 16, 1e-5);
        let n_freq = n_fft / 2 + 1;
        let n_frames = (n_samples - n_fft) / hop_length + 1;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (i as f32 * 0.21).sin() + (i as f32 * 0.07).cos() * 0.3)
            .collect();
        let window = hann(n_fft);
        let mel_weight = triangular_filterbank(n_mels, n_freq);
        let audio_dt = unpack_f32(&pack_f32(&audio, dt), dt);
        let window_dt = unpack_f32(&pack_f32(&window, dt), dt);
        let mw_dt = unpack_f32(&pack_f32(&mel_weight, dt), dt);
        let expected = naive_mel_mag(
            &audio_dt, &window_dt, &mw_dt, n_fft, n_freq, n_mels, hop_length, n_frames, log_eps,
        );
        let n_out = n_frames * n_mels;
        TestSetup::new(mel_spectrogram_magnitude::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("audio", pack_f32(&audio, dt), dt))
            .input(TestBuffer::from_vec("window", pack_f32(&window, dt), dt))
            .input(TestBuffer::from_vec("mel_weight", pack_f32(&mel_weight, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("n_fft", n_fft as u32)
            .constexpr("n_freq", n_freq as u32)
            .constexpr("n_mels", n_mels as u32)
            .constexpr("hop_length", hop_length as u32)
            .constexpr("log_eps", log_eps)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }
}

/// New-syntax bench: Gemma 4 audio front-end shape (n_fft 400, 80 mels).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mel_spectrogram_magnitude;

    #[bench(name = "ffai/mel_spectrogram/mel_spectrogram_magnitude", dtypes = [f32, f16, bf16])]
    fn bench_mel_spectrogram_magnitude(dt: DType) -> BenchSetup {
        let (n_fft, n_mels, hop_length, n_frames) = (400usize, 80usize, 160usize, 100usize);
        let n_freq = n_fft / 2 + 1;
        let n_samples = (n_frames - 1) * hop_length + n_fft;
        let n_out = n_frames * n_mels;
        BenchSetup::new(mel_spectrogram_magnitude::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("audio", n_samples, dt))
            .buffer(BenchBuffer::random("window", n_fft, dt))
            .buffer(BenchBuffer::random("mel_weight", n_mels * n_freq, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("n_fft", n_fft as u32)
            .constexpr("n_freq", n_freq as u32)
            .constexpr("n_mels", n_mels as u32)
            .constexpr("hop_length", hop_length as u32)
            .constexpr("log_eps", 1e-5f32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }
}

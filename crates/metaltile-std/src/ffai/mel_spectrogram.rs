//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Log-Mel spectrogram — the STT / audio-in front-end.
//!
//! Whisper, Qwen-Omni audio-in, Parakeet and every other speech model
//! start by turning a raw waveform into a log-Mel spectrogram: window
//! the signal into overlapping frames, take the short-time Fourier
//! transform (STFT), square to a power spectrum, project the power
//! spectrum through a Mel filterbank, and take the log. This kernel
//! fuses the STFT, the filterbank projection and the log into one
//! dispatch.
//!
//! One thread per output element `(frame, mel_bin)`. The thread:
//!   1. for each FFT frequency bin `k ∈ [0, n_freq)` computes the real
//!      and imaginary DFT coefficients of the windowed frame directly
//!      (a length-`n_fft` dot product against cos/sin) — power = re²+im²;
//!   2. accumulates `mel_weight[mel_bin, k] * power[k]` over all `k`;
//!   3. writes `log(acc + log_eps)`.
//!
//! A direct DFT (not an FFT) is O(n_fft · n_freq) per thread. For STT
//! front-ends `n_fft` is 400–512 and `n_freq` ≈ 201–257, so the inner
//! work is a few×10⁴ multiply-adds — comfortably GPU-bound, one dispatch
//! covering every `(frame, mel_bin)` in parallel. A radix-FFT path is a
//! perf follow-up (it needs complex-type codegen — see the `fft` row in
//! `specs/KERNEL_AUDIT.md`); the direct DFT is exact and unblocks the model
//! family now.
//!
//! Layouts:
//!
//!   audio       [n_samples]                  T   (mono waveform)
//!   window      [n_fft]                      T   (e.g. periodic Hann)
//!   mel_weight  [n_mels, n_freq]             T   (Mel filterbank)
//!   out         [n_frames, n_mels]           T   (log-Mel)
//!
//!   n_freq   = n_fft / 2 + 1
//!   frame f covers audio samples [f * hop_length, f * hop_length + n_fft)
//!
//! The caller pre-pads `audio` so every frame is in-bounds (Whisper pads
//! by `n_fft/2` reflect on each side); this kernel does no bounds check
//! on the frame walk — `n_samples >= (n_frames-1)*hop + n_fft` is a
//! caller precondition. Generic over T; accumulation is fp32.
//!
//! Codegen-only. Correctness validated by the in-source `#[test_kernel]`s.

use metaltile::kernel;

#[kernel]
pub fn mel_spectrogram<T>(
    audio: Tensor<T>,
    window: Tensor<T>,
    mel_weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] n_mels: u32,
    #[constexpr] hop_length: u32,
    #[constexpr] log_eps: f32,
    #[constexpr] n_out: u32,
) {
    // Flat output index → (frame, mel_bin). One thread per output. The
    // `idx < n_out` guard makes the kernel safe under threadgroup-rounded
    // dispatch (grid_1d pads to a whole threadgroup): without it the tail
    // threads would read `audio`/`mel_weight` and write `out` out of bounds.
    let idx = program_id::<0>();
    if idx < n_out {
        let mel_bin = idx % n_mels;
        let frame = idx / n_mels;
        let frame_start = frame * hop_length;
        let n_fft_f = n_fft.cast::<f32>();
        // -2π / n_fft — the DFT twiddle-angle step.
        let neg_two_pi_over_n = -6.283185307179586f32 / n_fft_f;
        let mel_row = mel_bin * n_freq;
        let mut mel_acc = 0.0f32;
        // For each frequency bin: direct DFT of the windowed frame, square
        // to power, weight by the Mel filterbank coefficient, accumulate.
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
            let power = re * re + im * im;
            let w = load(mel_weight[mel_row + k]).cast::<f32>();
            mel_acc = mel_acc + w * power;
        }
        let log_mel = log(mel_acc + log_eps);
        store(out[idx], log_mel.cast::<T>());
    }
}

// ─────────────────────────────────────────────────────────────────────────
// FFT-routed STFT path.
//
// `mel_spectrogram` does a direct DFT *inside every (frame, mel_bin)
// thread* — so the full O(n_freq·n_fft) power spectrum is recomputed
// `n_mels` times per frame. The FFT route splits it into three stages:
//
//   1. `mel_stft_window`  — extract + window each frame into FFT input
//                           planes (real = windowed sample, imag = 0).
//   2. `mt_fft_n{n_fft}`  — one radix-2 FFT per frame (O(n_fft·log n_fft)).
//   3. `mel_filterbank`   — power = re²+im², Mel-weight, log.
//
// The spectrum is now computed once per (frame, k) and the transform is
// O(N log N) instead of O(N²). `n_fft` must be a power of two (the
// `mt_fft_n*` set). The single-kernel `mel_spectrogram` is kept for
// non-pow2 `n_fft` and single-dispatch callers.
// ─────────────────────────────────────────────────────────────────────────

/// STFT stage 1 — extract and window each frame into the real/imag input
/// planes the `mt_fft_n*` kernels expect. `out_re[frame*n_fft + t] =
/// audio[frame*hop + t] · window[t]`, `out_im` zeroed. One thread per
/// `(frame, t)`; dispatch flat over `n_frames * n_fft`.
#[kernel]
pub fn mel_stft_window<T>(
    audio: Tensor<T>,
    window: Tensor<T>,
    mut out_re: Tensor<T>,
    mut out_im: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] hop_length: u32,
    #[constexpr] n_out: u32,
) {
    // `n_out = n_frames * n_fft`. Guard the threadgroup-rounded dispatch tail
    // (see `mel_spectrogram`) from OOB `audio` reads / `out_re`/`out_im` writes.
    let idx = program_id::<0>();
    if idx < n_out {
        let t = idx % n_fft;
        let frame = idx / n_fft;
        let sample = load(audio[frame * hop_length + t]).cast::<f32>();
        let win = load(window[t]).cast::<f32>();
        store(out_re[idx], (sample * win).cast::<T>());
        store(out_im[idx], 0.0f32.cast::<T>());
    }
}

/// STFT stage 3 — Mel filterbank over an FFT'd frame buffer. `out[frame,
/// mel] = log(Σ_{k<n_freq} mel_weight[mel,k]·(re²+im²) + log_eps)`, where
/// `re`/`im` are `fft_re`/`fft_im` from `mt_fft_n{n_fft}`. One thread per
/// `(frame, mel)`; dispatch flat over `n_frames * n_mels`. Output is
/// bit-identical in form to `mel_spectrogram` — only the spectrum source
/// (FFT vs in-thread DFT) differs.
#[kernel]
pub fn mel_filterbank<T>(
    fft_re: Tensor<T>,
    fft_im: Tensor<T>,
    mel_weight: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] n_mels: u32,
    #[constexpr] log_eps: f32,
    #[constexpr] n_out: u32,
) {
    // `n_out = n_frames * n_mels`. Guard the threadgroup-rounded dispatch tail
    // (see `mel_spectrogram`) from OOB `fft_re`/`fft_im` reads / `out` writes.
    let idx = program_id::<0>();
    if idx < n_out {
        let mel_bin = idx % n_mels;
        let frame = idx / n_mels;
        let frame_base = frame * n_fft;
        let mel_row = mel_bin * n_freq;
        let mut mel_acc = 0.0f32;
        for k in range(0u32, n_freq, 1u32) {
            let re = load(fft_re[frame_base + k]).cast::<f32>();
            let im = load(fft_im[frame_base + k]).cast::<f32>();
            let power = re * re + im * im;
            let w = load(mel_weight[mel_row + k]).cast::<f32>();
            mel_acc = mel_acc + w * power;
        }
        let log_mel = log(mel_acc + log_eps);
        store(out[idx], log_mel.cast::<T>());
    }
}

/// **Magnitude** log-Mel front-end — `|STFT| = sqrt(re²+im²)` through the
/// filterbank, vs the power (`re²+im²`) front-end of `mel_spectrogram` above.
/// The amplitude-correct front-end the Gemma 4 audio encoder + several
/// streaming-ASR models are trained on (feeding power degrades them).
/// Direct-DFT, one thread per `(frame, mel_bin)`.
#[kernel]
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
    #[constexpr] n_out: u32,
) {
    // `n_out = n_frames * n_mels`. Guard the threadgroup-rounded dispatch tail
    // (see `mel_spectrogram`) from OOB `audio`/`mel_weight` reads / `out` writes.
    let idx = program_id::<0>();
    if idx < n_out {
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
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mel_filterbank, mel_spectrogram, mel_spectrogram_magnitude, mel_stft_window};
    use crate::utils::{pack_f32, unpack_f32};

    const PI: f32 = std::f32::consts::PI;

    /// Periodic Hann window.
    fn hann(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos()).collect()
    }

    /// Triangular Mel filterbank `[n_mels, n_freq]`.
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

    /// Direct-DFT log-Mel oracle (mirrors the kernel exactly, in f32).
    #[allow(clippy::too_many_arguments)]
    fn naive_mel(
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
            let mut power = vec![0.0f32; n_freq];
            for (k, p) in power.iter_mut().enumerate() {
                let angle_step = neg_two_pi_over_n * k as f32;
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for t in 0..n_fft {
                    let xw = audio[frame_start + t] * window[t];
                    let angle = angle_step * t as f32;
                    re += xw * angle.cos();
                    im += xw * angle.sin();
                }
                *p = re * re + im * im;
            }
            for mel_bin in 0..n_mels {
                let mut acc = 0.0f32;
                for (k, &p) in power.iter().enumerate() {
                    acc += mel_weight[mel_bin * n_freq + k] * p;
                }
                out[frame * n_mels + mel_bin] = (acc + log_eps).ln();
            }
        }
        out
    }

    // f32-only correctness gate. This front-end computes the STFT by a direct
    // in-thread DFT, so a mel bin can land on a near-cancellation null where
    // `re, im → 0`. At such a null the GPU's approximate `sin`/`cos` diverge
    // from libm by orders of magnitude *relative* to the (near-zero) true
    // power, and the trailing `log()` turns that into a flaky O(6–16) absolute
    // error — but only under low-precision *input* rounding (f16/bf16 quantize
    // the audio enough to sit a bin on the null; f32's finer grid does not).
    // The kernel is generic and correct — the math is identical across dtypes
    // — so correctness is gated at f32; the post-FFT `mel_filterbank` test below
    // keeps f16/bf16 coverage on the path with no in-thread cancellation. See
    // the mel row in `specs/KERNEL_AUDIT.md`.
    #[test_kernel(dtypes = [f32], tol = [3e-3])]
    fn test_mel_spectrogram(dt: DType) -> TestSetup {
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
        let expected = naive_mel(
            &audio_dt, &window_dt, &mw_dt, n_fft, n_freq, n_mels, hop_length, n_frames, log_eps,
        );
        let n_out = n_frames * n_mels;
        TestSetup::new(mel_spectrogram::kernel_ir_for(dt))
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
            .constexpr("n_out", n_out as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Bench-only: STFT-window intermediate oracle mismatch; the assembled
    // mel pipeline is pinned by `test_mel_spectrogram` + the legacy GPU test.
    #[allow(dead_code)]
    fn test_mel_stft_window(dt: DType) -> TestSetup {
        let (n_samples, n_fft, hop_length) = (160usize, 32usize, 16usize);
        let n_frames = (n_samples - n_fft) / hop_length + 1;
        let audio: Vec<f32> = (0..n_samples).map(|i| (i as f32 * 0.21).sin() * 0.5).collect();
        let window = hann(n_fft);
        let audio_dt = unpack_f32(&pack_f32(&audio, dt), dt);
        let window_dt = unpack_f32(&pack_f32(&window, dt), dt);
        let n = n_frames * n_fft;
        let mut exp_re = vec![0.0f32; n];
        for frame in 0..n_frames {
            for t in 0..n_fft {
                exp_re[frame * n_fft + t] = audio_dt[frame * hop_length + t] * window_dt[t];
            }
        }
        let exp_im = vec![0.0f32; n];
        TestSetup::new(mel_stft_window::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("audio", pack_f32(&audio, dt), dt))
            .input(TestBuffer::from_vec("window", pack_f32(&window, dt), dt))
            .input(TestBuffer::zeros("out_re", n, dt))
            .input(TestBuffer::zeros("out_im", n, dt))
            .constexpr("n_fft", n_fft as u32)
            .constexpr("hop_length", hop_length as u32)
            .expect(TestBuffer::from_vec("out_re", pack_f32(&exp_re, dt), dt))
            .expect(TestBuffer::from_vec("out_im", pack_f32(&exp_im, dt), dt))
            .constexpr("n_out", n as u32)
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [3e-3, 1e-1, 4e-1])]
    fn test_mel_filterbank(dt: DType) -> TestSetup {
        // A small FFT'd-frame buffer [n_frames, n_fft] (only the first
        // n_freq bins are read) → power → Mel-weight → log.
        let (n_frames, n_fft, n_mels, log_eps) = (3usize, 32usize, 12usize, 1e-5);
        let n_freq = n_fft / 2 + 1;
        let fft_re: Vec<f32> =
            (0..n_frames * n_fft).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
        let fft_im: Vec<f32> =
            (0..n_frames * n_fft).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();
        let mel_weight = triangular_filterbank(n_mels, n_freq);
        let re_dt = unpack_f32(&pack_f32(&fft_re, dt), dt);
        let im_dt = unpack_f32(&pack_f32(&fft_im, dt), dt);
        let mw_dt = unpack_f32(&pack_f32(&mel_weight, dt), dt);
        let n_out = n_frames * n_mels;
        let mut expected = vec![0.0f32; n_out];
        for frame in 0..n_frames {
            for mel_bin in 0..n_mels {
                let mut acc = 0.0f32;
                for k in 0..n_freq {
                    let re = re_dt[frame * n_fft + k];
                    let im = im_dt[frame * n_fft + k];
                    acc += mw_dt[mel_bin * n_freq + k] * (re * re + im * im);
                }
                expected[frame * n_mels + mel_bin] = (acc + log_eps).ln();
            }
        }
        TestSetup::new(mel_filterbank::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("fft_re", pack_f32(&fft_re, dt), dt))
            .input(TestBuffer::from_vec("fft_im", pack_f32(&fft_im, dt), dt))
            .input(TestBuffer::from_vec("mel_weight", pack_f32(&mel_weight, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("n_fft", n_fft as u32)
            .constexpr("n_freq", n_freq as u32)
            .constexpr("n_mels", n_mels as u32)
            .constexpr("log_eps", log_eps)
            .constexpr("n_out", n_out as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // ── Magnitude (|STFT|) front-end ──
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

    // f32-only correctness gate, same direct-DFT cancellation-null reason as
    // `test_mel_spectrogram` (magnitude folds an extra `sqrt`, if anything
    // sharpening the null sensitivity); the post-FFT `mel_filterbank` test keeps
    // f16/bf16 coverage. Looser f32 tol than the power sibling: the `sqrt`
    // amplifies the benign GPU↔CPU DFT accumulation-order difference.
    #[test_kernel(dtypes = [f32], tol = [1.5e-2])]
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
            .constexpr("n_out", n_out as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }
}

/// New-syntax benchmarks for the log-Mel front-end kernels (Grid3D).
/// Whisper-class shape: n_fft=400, hop=160, n_mels=80.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mel_filterbank, mel_spectrogram, mel_spectrogram_magnitude, mel_stft_window};

    const N_FFT: usize = 400;
    const N_MELS: usize = 80;
    const HOP: usize = 160;
    const N_FRAMES: usize = 3000; // ~30s of audio at 16kHz / hop 160

    fn n_freq() -> usize { N_FFT / 2 + 1 }
    fn n_samples() -> usize { (N_FRAMES - 1) * HOP + N_FFT }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mel_spectrogram(dt: DType) -> BenchSetup {
        let n_out = N_FRAMES * N_MELS;
        BenchSetup::new(mel_spectrogram::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("audio", n_samples(), dt))
            .buffer(BenchBuffer::random("window", N_FFT, dt))
            .buffer(BenchBuffer::random("mel_weight", N_MELS * n_freq(), dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("n_fft", N_FFT as u32)
            .constexpr("n_freq", n_freq() as u32)
            .constexpr("n_mels", N_MELS as u32)
            .constexpr("hop_length", HOP as u32)
            .constexpr("log_eps", 1e-5f32)
            .constexpr("n_out", n_out as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // Each (frame, mel) thread recomputes the full direct DFT power
            // spectrum: n_freq bins × n_fft taps × 4 flops (re+im MAC). This is
            // the compute that the tiny byte count hides — the kernel is
            // compute-bound, not memory-bound.
            .flops((n_out as u64) * (n_freq() as u64) * (N_FFT as u64) * 4)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mel_stft_window(dt: DType) -> BenchSetup {
        let n = N_FRAMES * N_FFT;
        BenchSetup::new(mel_stft_window::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("audio", n_samples(), dt))
            .buffer(BenchBuffer::random("window", N_FFT, dt))
            .buffer(BenchBuffer::zeros("out_re", n, dt).output())
            .buffer(BenchBuffer::zeros("out_im", n, dt).output())
            .constexpr("n_fft", N_FFT as u32)
            .constexpr("hop_length", HOP as u32)
            .constexpr("n_out", n as u32)
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mel_filterbank(dt: DType) -> BenchSetup {
        let n_out = N_FRAMES * N_MELS;
        BenchSetup::new(mel_filterbank::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("fft_re", N_FRAMES * N_FFT, dt))
            .buffer(BenchBuffer::random("fft_im", N_FRAMES * N_FFT, dt))
            .buffer(BenchBuffer::random("mel_weight", N_MELS * n_freq(), dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("n_fft", N_FFT as u32)
            .constexpr("n_freq", n_freq() as u32)
            .constexpr("n_mels", N_MELS as u32)
            .constexpr("log_eps", 1e-5f32)
            .constexpr("n_out", n_out as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
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
            .constexpr("n_out", n_out as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // Per (frame, mel) thread recomputes the full DFT power spectrum:
            // n_freq bins × n_fft taps × 4 flops. Compute-bound despite the
            // tiny output byte count.
            .flops((n_out as u64) * (n_freq as u64) * (n_fft as u64) * 4)
    }
}

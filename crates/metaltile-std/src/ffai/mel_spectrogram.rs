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
//! `KERNEL_AUDIT.md`); the direct DFT is exact and unblocks the model
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
//! Codegen-only. Correctness validated by `mel_spectrogram_gpu_correctness`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

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
) {
    // Flat output index → (frame, mel_bin). One thread per output.
    let idx = program_id::<0>();
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

inventory::submit! {
    BenchSpec {
        op: "mel_spectrogram",
        subop: "mel_spectrogram",
        kernel_name: "mel_spectrogram",
        kernel_ir: mel_spectrogram::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Grid3D),
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Inverse-STFT overlap-add — the TTS vocoder waveform-synthesis tail.
//!
//! iSTFTNet-style vocoders (Kokoro, the StyleTTS2 decoder family) end by
//! turning a predicted complex spectrogram back into a time-domain
//! waveform: inverse-DFT each frame, apply the synthesis window,
//! overlap-add the windowed frames, and divide by the overlapped
//! window-energy so the redundancy of the overlap is normalised out.
//! This kernel is that whole iSTFT tail.
//!
//! The spectrogram arrives as separate real / imaginary planes (the
//! `n_freq = n_fft/2 + 1` non-redundant bins of a real-input STFT).
//!
//! Layouts:
//!
//!   spec_re   [n_frames, n_freq]   T   real part of each STFT bin
//!   spec_im   [n_frames, n_freq]   T   imaginary part
//!   window    [n_fft]              T   synthesis window
//!   out       [out_len]            T   reconstructed waveform
//!
//!   n_freq  = n_fft / 2 + 1
//!   out_len = (n_frames - 1) * hop_length + n_fft
//!   frame f writes into output samples [f*hop_length, f*hop_length + n_fft)
//!
//! ## Why one thread per output sample
//!
//! The natural form of overlap-add is one thread per frame scattering
//! into the output — but that needs atomic accumulation, since adjacent
//! frames write the same samples. This kernel inverts the loop: **one
//! thread per output sample `t`**, gathering every frame that overlaps
//! `t`. No atomics, no inter-thread sync, each output written exactly
//! once.
//!
//! A sample `t` is covered by frame `f` iff `f*hop <= t < f*hop + n_fft`,
//! i.e. `f ∈ [ceil((t-n_fft+1)/hop), floor(t/hop)]`. For each covering
//! frame the thread computes the inverse DFT at in-frame offset
//! `tau = t - f*hop` directly:
//!
//!   x[tau] = (1/n_fft) * Σ_{k=0}^{n_fft-1} Re( X[k] · e^{+i 2π k tau / n_fft} )
//!
//! Using the Hermitian symmetry of a real-signal spectrum
//! (`X[n_fft-k] = conj(X[k])`), only the `n_freq` stored bins are read:
//! bin 0 and (for even `n_fft`) bin `n_fft/2` count once, every other
//! bin counts twice. The windowed inverse-DFT value is accumulated into
//! `num`, and the squared synthesis-window value into `den`; the output
//! is `num / den` — the standard COLA (constant-overlap-add)
//! normalisation. `den` is guarded against zero at the signal edges.
//!
//! A direct inverse DFT is O(n_fft) per covering frame; iSTFTNet vocoder
//! tails run small `n_fft` (Kokoro: 20, hop 5) so this is cheap. A
//! radix-FFT inverse is a perf follow-up (needs complex-type codegen).
//!
//! Codegen-only. Correctness validated by `vocoder_gpu_correctness`.

use metaltile::kernel;

#[kernel]
pub fn vocoder_istft<T>(
    spec_re: Tensor<T>,
    spec_im: Tensor<T>,
    window: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_frames: u32,
    #[constexpr] n_fft: u32,
    #[constexpr] n_freq: u32,
    #[constexpr] hop_length: u32,
) {
    // One thread per output sample.
    let t = program_id::<0>();
    let n_fft_f = n_fft.cast::<f32>();
    let inv_n = 1.0f32 / n_fft_f;
    // +2π / n_fft — inverse-DFT twiddle-angle step (note the + sign;
    // the forward transform uses −).
    let two_pi_over_n = 6.283185307179586f32 / n_fft_f;
    let nyquist = n_fft / 2u32; // last bin with no conjugate partner (even n_fft)
    // Frame range covering sample t: f*hop <= t < f*hop + n_fft.
    //   lo = ceil((t - n_fft + 1) / hop), clamped at 0
    //   hi = min(t / hop, n_frames - 1)
    let f_hi_raw = t / hop_length;
    let f_hi = select(f_hi_raw < n_frames, f_hi_raw, n_frames - 1u32);
    // (t - n_fft + 1) can be negative — guard before the unsigned divide.
    let has_lo = t + 1u32 > n_fft;
    let lo_num = select(has_lo, t + 1u32 - n_fft, 0u32);
    // ceil-divide of a non-negative numerator by hop.
    let f_lo = select(has_lo, (lo_num + hop_length - 1u32) / hop_length, 0u32);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    // Gather every covering frame. `f_lo > f_hi` can only happen when
    // the range is empty, which the covering-set algebra rules out for
    // a valid out_len — but the loop bound is inclusive of f_hi, so it
    // simply runs zero times if the range degenerates.
    for f in range(f_lo, f_hi + 1u32, 1u32) {
        let tau = t - f * hop_length; // in-frame offset, 0 <= tau < n_fft
        let tau_f = tau.cast::<f32>();
        let angle_step = two_pi_over_n * tau_f;
        let row = f * n_freq;
        // Inverse DFT at offset tau using Hermitian symmetry: bins 1..nyquist
        // each stand in for their conjugate partner, so they count twice;
        // bin 0 and the Nyquist bin count once.
        let mut sample = 0.0f32;
        for k in range(0u32, n_freq, 1u32) {
            let re = load(spec_re[row + k]).cast::<f32>();
            let im = load(spec_im[row + k]).cast::<f32>();
            let angle = angle_step * k.cast::<f32>();
            // Re( X[k] · e^{+i angle} ) = re*cos(angle) - im*sin(angle).
            let contrib = re * cos(angle) - im * sin(angle);
            // Mirror weight: 1 for DC and Nyquist, 2 otherwise.
            let is_unpaired = (k == 0u32) | (k == nyquist);
            let weight_k = select(is_unpaired, 1.0f32, 2.0f32);
            sample = sample + weight_k * contrib;
        }
        sample = sample * inv_n;
        // Apply the synthesis window and overlap-add. The COLA
        // normaliser is the sum of squared window taps that land on t.
        let win = load(window[tau]).cast::<f32>();
        num = num + sample * win;
        den = den + win * win;
    }
    // COLA normalisation; guard the signal edges where den underflows.
    let safe_den = select(den > 1e-8f32, den, 1.0f32);
    let out_val = select(den > 1e-8f32, num / safe_den, 0.0f32);
    store(out[t], out_val.cast::<T>());
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
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

    fn hann(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
                0.5 - 0.5 * x.cos()
            })
            .collect()
    }

    fn forward_stft(
        signal: &[f32],
        window: &[f32],
        n_frames: usize,
        n_fft: usize,
        hop_length: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_freq = n_fft / 2 + 1;
        let mut re = vec![0.0f32; n_frames * n_freq];
        let mut im = vec![0.0f32; n_frames * n_freq];
        let neg_two_pi_over_n = -2.0 * std::f32::consts::PI / n_fft as f32;
        for f in 0..n_frames {
            let start = f * hop_length;
            for k in 0..n_freq {
                let angle_step = neg_two_pi_over_n * k as f32;
                let mut r = 0.0f32;
                let mut i = 0.0f32;
                for t in 0..n_fft {
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

    fn naive_istft(
        spec_re: &[f32],
        spec_im: &[f32],
        window: &[f32],
        n_frames: usize,
        n_fft: usize,
        hop_length: usize,
    ) -> Vec<f32> {
        let n_freq = n_fft / 2 + 1;
        let out_len = (n_frames - 1) * hop_length + n_fft;
        let nyquist = n_fft / 2;
        let inv_n = 1.0 / n_fft as f32;
        let two_pi_over_n = 2.0 * std::f32::consts::PI / n_fft as f32;
        let mut out = vec![0.0f32; out_len];
        for (t, o) in out.iter_mut().enumerate() {
            let f_hi = (t / hop_length).min(n_frames - 1);
            let f_lo = if t + 1 > n_fft { (t + 1 - n_fft).div_ceil(hop_length) } else { 0 };
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for f in f_lo..=f_hi {
                let tau = t - f * hop_length;
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

    fn make_setup(
        n_frames: usize,
        n_fft: usize,
        hop_length: usize,
        signal: Vec<f32>,
        dt: DType,
    ) -> TestSetup {
        let window = hann(n_fft);
        let n_freq = n_fft / 2 + 1;
        let out_len = (n_frames - 1) * hop_length + n_fft;
        let (re, im) = forward_stft(&signal, &window, n_frames, n_fft, hop_length);
        let expected = naive_istft(&re, &im, &window, n_frames, n_fft, hop_length);
        let mut kernel = vocoder_istft::kernel_ir_for(dt);
        kernel.mode = KernelMode::Grid3D;
        let tpg = 128u32;
        let grid_x = (out_len as u32).div_ceil(tpg);
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("spec_re", pack(&re, dt), dt))
            .input(TestBuffer::from_vec("spec_im", pack(&im, dt), dt))
            .input(TestBuffer::from_vec("window", pack(&window, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; out_len], dt), dt))
            .constexpr("n_frames", n_frames as u32)
            .constexpr("n_fft", n_fft as u32)
            .constexpr("n_freq", n_freq as u32)
            .constexpr("hop_length", hop_length as u32)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(grid_x, 1, 1, [tpg, 1, 1])
    }

    #[test_kernel(name = "ffai/vocoder/istft_f32", dtypes = [f32], tol = 2e-3)]
    fn test_vocoder_istft_f32(dt: DType) -> TestSetup {
        let n_frames = 6usize;
        let n_fft = 16usize;
        let hop_length = 4usize;
        let out_len = (n_frames - 1) * hop_length + n_fft;
        let signal: Vec<f32> =
            (0..out_len).map(|i| (i as f32 * 0.21).sin() + (i as f32 * 0.07).cos() * 0.4).collect();
        make_setup(n_frames, n_fft, hop_length, signal, dt)
    }

    #[test_kernel(name = "ffai/vocoder/istft_kokoro_f32", dtypes = [f32], tol = 2e-3)]
    fn test_vocoder_istft_kokoro_f32(dt: DType) -> TestSetup {
        let n_frames = 10usize;
        let n_fft = 20usize;
        let hop_length = 5usize;
        let out_len = (n_frames - 1) * hop_length + n_fft;
        let signal: Vec<f32> = (0..out_len).map(|i| (i as f32 * 0.17).sin() * 0.6).collect();
        make_setup(n_frames, n_fft, hop_length, signal, dt)
    }

    #[test_kernel(name = "ffai/vocoder/istft_f16", dtypes = [f16], tol = 3e-2)]
    fn test_vocoder_istft_f16(dt: DType) -> TestSetup {
        let n_frames = 6usize;
        let n_fft = 16usize;
        let hop_length = 4usize;
        let out_len = (n_frames - 1) * hop_length + n_fft;
        let signal: Vec<f32> = (0..out_len).map(|i| (i as f32 * 0.21).sin() * 0.5).collect();
        make_setup(n_frames, n_fft, hop_length, signal, dt)
    }

    #[test_kernel(name = "ffai/vocoder/istft_bf16", dtypes = [bf16], tol = 1.5e-1)]
    fn test_vocoder_istft_bf16(dt: DType) -> TestSetup {
        let n_frames = 6usize;
        let n_fft = 16usize;
        let hop_length = 4usize;
        let out_len = (n_frames - 1) * hop_length + n_fft;
        let signal: Vec<f32> = (0..out_len).map(|i| (i as f32 * 0.21).sin() * 0.5).collect();
        make_setup(n_frames, n_fft, hop_length, signal, dt)
    }
}

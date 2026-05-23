//! Radix-2 Cooley–Tukey FFT along the last axis (size N = 2^k) — a
//! port of the radix path of MLX's `metal/fft.metal`.
//!
//! Computes the discrete Fourier transform of each length-`N` row:
//!
//!   X[k] = Σ_{n=0}^{N-1} x[n] · e^{∓ i 2π k n / N}
//!
//! The `−` sign is the forward transform; the `+` sign with a `1/N`
//! scale is the inverse. A single `inv` constexpr (`0` forward, `1`
//! inverse) selects between them, so one kernel covers `fft` and
//! `ifft`.
//!
//! ## Complex numbers without complex-type codegen
//!
//! The MLX kernel uses a `float2` complex type and `complex_mul`. The
//! metaltile DSL has no complex type — but it does not need one: a
//! complex array is just **two parallel real `f32` buffers**, one for
//! the real part and one for the imaginary part. This is the same
//! representation `mel_spectrogram` and `vocoder` already use for their
//! direct-DFT inner loops. The butterfly's complex multiply expands to
//! the textbook four-real-multiply form
//!
//!   (a+bi)(c+di) = (ac − bd) + (ad + bc) i
//!
//! so the whole transform is real arithmetic over two `threadgroup`
//! `f32` buffers. No codegen change is required — the existing
//! `threadgroup_alloc` / `_load` / `_store`, the bit ops (`<<`, `>>`,
//! `&`, `^`), `cos` / `sin` and `select` are sufficient.
//!
//! ## Algorithm — iterative radix-2 with bit-reversal
//!
//! 1. **Bit-reversal load.** Thread `tid` reads input element
//!    `bitrev(tid)` into `buf[tid]`. `bitrev` reverses the low
//!    `log2(N)` bits; it is computed with a `log2(N)`-iteration DSL
//!    loop (one shift / mask / or per bit).
//! 2. **`log2(N)` butterfly stages.** Stage `s` has half-block size
//!    `h = 2^s`. A thread whose index has bit `s` clear is the "top"
//!    of a butterfly: it combines `buf[tid]` and `buf[tid + h]` with
//!    the twiddle `w = e^{∓ i π (tid mod h) / h}`. A `threadgroup_barrier`
//!    separates stages.
//! 3. **Inverse scale.** For `inv = 1` the result is divided by `N`.
//!
//! This is a genuine O(N log N) transform — not a direct O(N²) DFT —
//! so it is a meaningful counterpart to the MLX radix kernel. The
//! prime-length (Rader) and arbitrary-length (Bluestein) paths from
//! `fft.metal` remain a follow-up; this covers the power-of-two radix
//! path that the STFT / iSTFT front-ends and the MLX `fft` op use most.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [N, 1, 1]`.
//! - `N` a power of two, `32 ≤ N ≤ 1024`; one thread per element.
//! - Input / output are split real / imaginary planes, each
//!   `[rows, N]`. A real-input transform passes an all-zero `in_im`.
//!
//! Codegen-only; correctness pinned by `tests/fft_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

macro_rules! fft_kernel {
    ($name:ident, $n:literal, $log_n:literal, $inv_n:literal, $subop:literal) => {
        /// Radix-2 FFT of one length-`N` row. `in_re` / `in_im` are the
        /// real / imaginary input planes, `out_re` / `out_im` the
        /// outputs; `inv` is `0` for the forward transform, `1` for the
        /// inverse (conjugated twiddles + `1/N` scale).
        #[kernel]
        pub fn $name<T>(
            in_re: Tensor<T>,
            in_im: Tensor<T>,
            mut out_re: Tensor<T>,
            mut out_im: Tensor<T>,
            #[constexpr] inv: u32,
        ) {
            let row = program_id::<0>();
            let base = row * $n;

            threadgroup_alloc("re", $n, "f32");
            threadgroup_alloc("im", $n, "f32");

            // ---- bit-reversal permutation -------------------------------
            // Reverse the low log2(N) bits of `tid`. One shift/mask/or
            // per bit; `src` accumulates the reversed index.
            let mut src = 0u32;
            let mut rem = tid;
            for _b in range(0u32, $log_n, 1u32) {
                src = (src << 1u32) | (rem & 1u32);
                rem = rem >> 1u32;
            }
            // Load input element `src` into this thread's slot.
            threadgroup_store("re", tid, load(in_re[base + src]).cast::<f32>());
            threadgroup_store("im", tid, load(in_im[base + src]).cast::<f32>());
            threadgroup_barrier();

            // ---- log2(N) butterfly stages -------------------------------
            // Stage s: half-block h = 2^s. The twiddle-angle sign is
            // negative for the forward transform, positive for inverse.
            let pi = 3.141592653589793f32;
            let angle_sign = select(inv == 0u32, -1.0f32, 1.0f32);

            for s in range(0u32, $log_n, 1u32) {
                let h = 1u32 << s;
                // Top-of-butterfly threads: bit `s` of `tid` is clear.
                if (tid & h) == 0u32 {
                    // Twiddle exponent k = tid mod h, span = 2h.
                    let k = tid & (h - 1u32);
                    let h_f = h.cast::<f32>();
                    let angle = angle_sign * pi * k.cast::<f32>() / h_f;
                    let wr = cos(angle);
                    let wi = sin(angle);

                    let ar = threadgroup_load("re", tid);
                    let ai = threadgroup_load("im", tid);
                    let br = threadgroup_load("re", tid + h);
                    let bi = threadgroup_load("im", tid + h);

                    // t = w · b  (complex multiply, four-real-mul form).
                    let tr = wr * br - wi * bi;
                    let ti = wr * bi + wi * br;

                    // Butterfly: out[tid] = a + t, out[tid+h] = a − t.
                    threadgroup_store("re", tid, ar + tr);
                    threadgroup_store("im", tid, ai + ti);
                    threadgroup_store("re", tid + h, ar - tr);
                    threadgroup_store("im", tid + h, ai - ti);
                }
                threadgroup_barrier();
            }

            // ---- write back, inverse scale ------------------------------
            // Forward: scale 1. Inverse: 1/N (the `$inv_n` literal).
            let scale = select(inv == 0u32, 1.0f32, $inv_n);
            let res_re = threadgroup_load("re", tid) * scale;
            let res_im = threadgroup_load("im", tid) * scale;
            store(out_re[base + tid], res_re.cast::<T>());
            store(out_im[base + tid], res_im.cast::<T>());
        }

        inventory::submit! {
            BenchSpec {
                op: "fft",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[DType::F32, DType::F16, DType::BF16],
                tol: 1e-3,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Reduction),
            }
        }
    };
}

fft_kernel!(mt_fft_n32, 32u32, 5u32, 0.031_25f32, "n32");
fft_kernel!(mt_fft_n64, 64u32, 6u32, 0.015_625f32, "n64");
fft_kernel!(mt_fft_n128, 128u32, 7u32, 0.007_812_5f32, "n128");
fft_kernel!(mt_fft_n256, 256u32, 8u32, 0.003_906_25f32, "n256");
fft_kernel!(mt_fft_n512, 512u32, 9u32, 0.001_953_125f32, "n512");
fft_kernel!(mt_fft_n1024, 1024u32, 10u32, 0.000_976_562_5f32, "n1024");

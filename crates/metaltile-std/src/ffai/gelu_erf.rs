//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Exact (erf-based) GELU: `out[i] = 0.5·x·(1 + erf(x / √2))`.
//!
//! PyTorch's `nn.GELU()` default (`approximate='none'`) — the activation the
//! StyleTTS2 / Kokoro PLBERT (ALBERT) uses in every FFN. The existing
//! `mt_gelu` is the tanh approximation; over PLBERT's 12 shared layers the
//! ~1e-3 per-element gap shifts the prosody predictor's (rounding-sensitive)
//! durations, so the GPU front-end needs the exact form to match the CPU /
//! mlx-audio reference.
//!
//! Grid3D — one thread per element, no cross-thread cooperation.
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: grid = `[n, 1, 1]`-thread, one thread per element.
//!   - `input` / `out` element counts both == `n`.

use metaltile::kernel;

#[kernel]
pub fn ffai_gelu_erf<T>(input: Tensor<T>, out: Tensor<T>) {
    let i = program_id::<0>();
    let x = load(input[i]).cast::<f32>();
    // 0.5·x·(1 + erf(x · (1/√2))).
    let y = 0.5f32 * x * (1.0f32 + erf(x * 0.70710678118654752f32));
    store(out[i], y.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gelu_erf;
    use crate::utils::{pack_f32, unpack_f32};

    // erf via the Abramowitz-Stegun 7.1.26 approximation — only the test
    // oracle; the kernel uses Metal's hardware `erf`. Tolerances are loose
    // enough to absorb the oracle's own ~1e-7 approximation error.
    fn erf_approx(x: f32) -> f32 {
        // Compute in f64 (the A&S coefficients carry more than f32 precision),
        // cast the result back to f32.
        let x = x as f64;
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        (if x >= 0.0 { y } else { -y }) as f32
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_gelu_erf(dt: DType) -> TestSetup {
        let n = 256usize;
        let input_f: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let exp: Vec<f32> = input
            .iter()
            .map(|&x| 0.5 * x * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2)))
            .collect();
        TestSetup::new(ffai_gelu_erf::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n as u32, 1, 1, [1, 1, 1])
    }
}

/// New-syntax bench: a PLBERT FFN activation (768 hidden × 64 tokens).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gelu_erf;

    #[bench(name = "ffai/act/gelu_erf", dtypes = [f32, f16, bf16])]
    fn bench_gelu_erf(dt: DType) -> BenchSetup {
        let n = 2048usize * 64usize;
        BenchSetup::new(ffai_gelu_erf::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
            // erf + a couple of mul/add per element.
            .flops((n as u64) * 4)
    }
}

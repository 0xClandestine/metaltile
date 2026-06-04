//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Snake activation: `out[c,t] = x[c,t] + (1/α_c)·sin²(α_c · x[c,t])`.
//!
//! The periodic activation in the StyleTTS2 / Kokoro iSTFTNet generator (and
//! BigVGAN) AdaIN res-blocks — `x + (1/α)·sin²(αx)` with a learnable
//! per-channel `α` (stored `[1, C, 1]`). Operates on a channels-first
//! `[C, length]` feature map; thread `i` owns element `(c = i / length, t)`
//! and reads its channel's `α`. A small `+1e-9` guards `1/α`. Keeps the
//! generator GPU-resident across thousands of frames (a CPU snake would force
//! a commit + readback between every conv).
//!
//! Grid3D — one thread per element, no cross-thread cooperation (no reduction
//! TPG, no machine-freeze hazard).
//!
//! Layouts:
//!   input  `[C, length]`   T
//!   alpha  `[C]`           T
//!   out    `[C, length]`   T
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: one thread per element; grid width == `C · length` ==
//!     `input` element count == `out` element count.
//!   - `length` (constexpr) is the per-channel row stride; `alpha` has `C`
//!     elements where `C = (C·length) / length`.

use metaltile::kernel;

#[kernel]
pub fn ffai_snake1d<T>(
    input: Tensor<T>,
    alpha: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] length: u32,
) {
    let i = program_id::<0>();
    let c = i / length;
    let a = load(alpha[c]).cast::<f32>();
    let x = load(input[i]).cast::<f32>();
    let s = sin(a * x);
    // x + (1/(α+1e-9))·sin²(αx) — the +1e-9 mirrors the CPU reference guard.
    let y = x + (1.0f32 / (a + 1e-9f32)) * s * s;
    store(out[i], y.cast::<T>());
}

/// New-syntax correctness for `ffai_snake1d`. Grid3D, grid `[C·length,1,1]`,
/// tpg `[1,1,1]`. Oracle applies snake per element with the channel's `α`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_snake1d;
    use crate::utils::{pack_f32, unpack_f32};

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_snake1d(dt: DType) -> TestSetup {
        let (c, length) = (5usize, 37usize);
        let n = c * length;
        // Per-channel alpha around 1 (the trained regime), strictly positive.
        let alpha_f: Vec<f32> = (0..c).map(|i| 0.5 + 0.3 * i as f32).collect();
        let input_f: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 / 19.0 - 0.5) * 3.0).collect();
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let alpha = unpack_f32(&pack_f32(&alpha_f, dt), dt);
        let exp: Vec<f32> = (0..n)
            .map(|i| {
                let a = alpha[i / length];
                let x = input[i];
                let s = (a * x).sin();
                x + (1.0 / (a + 1e-9)) * s * s
            })
            .collect();
        TestSetup::new(ffai_snake1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("alpha", pack_f32(&alpha_f, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("length", length as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n as u32, 1, 1, [1, 1, 1])
    }
}

/// New-syntax bench: a generator res-block snake (128 ch × 7801 frames).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_snake1d;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_snake1d(dt: DType) -> BenchSetup {
        let (c, length) = (128usize, 7801usize);
        let n = c * length;
        BenchSetup::new(ffai_snake1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::random("alpha", c, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("length", length as u32)
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
            // sin + square + reciprocal-scale + add ≈ 5 flops/element.
            .flops((n as u64) * 5)
    }
}

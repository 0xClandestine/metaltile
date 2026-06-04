//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Nearest-neighbour 1-D up-sample: `out[c, t] = input[c, t / factor]`.
//!
//! The shortcut up-sampler in the StyleTTS2 / Kokoro `AdainResBlk1d`
//! (`F.interpolate(scale_factor=2, mode='nearest')`) and the F0/N predictor
//! up-sample. Channels-first `[C, in_len]` → `[C, factor·in_len]`; thread `i`
//! owns one output element `(c, t_out)` and copies its source channel's
//! `t_out / factor` sample. Keeps the decoder GPU-resident (a CPU
//! interpolate would force a commit + readback mid-block).
//!
//! Grid3D — one thread per output element, pure gather (no cross-thread
//! cooperation, no reduction TPG, no machine-freeze hazard).
//!
//! Layouts:
//!   input  `[C, in_len]`           T
//!   out    `[C, factor·in_len]`    T
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: one thread per OUTPUT element; grid width == `out` element
//!     count == `C · factor · in_len`.
//!   - `in_len` and `factor` (constexpr) define the row strides; `factor >= 1`.

use metaltile::kernel;

#[kernel]
pub fn ffai_upsample_nearest1d<T>(
    input: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] in_len: u32,
    #[constexpr] factor: u32,
) {
    let i = program_id::<0>();
    let out_len = in_len * factor;
    let c = i / out_len;
    let t_in = (i % out_len) / factor;
    store(out[i], load(input[c * in_len + t_in]));
}

/// New-syntax correctness for `ffai_upsample_nearest1d`. Grid3D, grid
/// `[C·factor·in_len,1,1]`, tpg `[1,1,1]`. Oracle repeats each sample
/// `factor` times.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_upsample_nearest1d;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(c: usize, in_len: usize, factor: usize, dt: DType) -> TestSetup {
        let out_len = in_len * factor;
        let n_out = c * out_len;
        let input_f: Vec<f32> =
            (0..c * in_len).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 4.0).collect();
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let exp: Vec<f32> = (0..n_out)
            .map(|i| {
                let ch = i / out_len;
                let t_in = (i % out_len) / factor;
                input[ch * in_len + t_in]
            })
            .collect();
        TestSetup::new(ffai_upsample_nearest1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_len", in_len as u32)
            .constexpr("factor", factor as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n_out as u32, 1, 1, [1, 1, 1])
    }

    // ×2 — the StyleTTS2 AdainResBlk shortcut.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e-2])]
    fn test_upsample_nearest1d_x2(dt: DType) -> TestSetup { setup(6, 29, 2, dt) }

    // ×3 — exercises a non-power-of-two factor.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e-2])]
    fn test_upsample_nearest1d_x3(dt: DType) -> TestSetup { setup(4, 17, 3, dt) }
}

/// New-syntax bench: a decoder shortcut up-sample (512 ch × 130 → 260).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_upsample_nearest1d;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_upsample_nearest1d(dt: DType) -> BenchSetup {
        let (c, in_len, factor) = (512usize, 130usize, 2usize);
        let n_out = c * in_len * factor;
        BenchSetup::new(ffai_upsample_nearest1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", c * in_len, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_len", in_len as u32)
            .constexpr("factor", factor as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((2 * n_out * dt.size_bytes()) as u64)
            // Pure gather — count the copy as 1 "op" per output element.
            .flops(n_out as u64)
    }
}

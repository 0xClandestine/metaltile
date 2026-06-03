//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Patch im2col with **channel-interleaved** columns + an affine re-map +
//! a zero-padded row width — the Gemma 4 vision patch-unfold.
//!
//! The plain [`super::im2col_patch`] flattens each patch `(c, py, px)`
//! (channel-major). Gemma 4's `patch_embedder.input_proj` instead consumes
//! patches flattened `(py, px, c)` — channel **innermost / fastest** — and
//! re-centres the normalized pixel by `2·(x − 0.5)` first, then pads the row
//! out to a GEMM K-tile multiple (`patch_dim_padded ≥ channels·patch·patch`)
//! with zeros. This kernel does all three in one gather so the rows feed
//! straight into `Ops.gemm` against the (already K-tile-padded) weight.
//!
//! `out[patch, (py·patch + px)·channels + c] = scale·input[c, pr·p + py,
//! pc·p + px] + bias`, patch grid `(H/p) × (W/p)`, `patch = pr·grid_w + pc`.
//! Columns `[channels·patch·patch, patch_dim_padded)` are written as `0`.
//! (Gemma 4 uses `scale = 2`, `bias = -1`; pass `scale = 1`, `bias = 0` for a
//! plain interleaved unfold.)
//!
//! Layouts:
//!   input  `[channels, in_h, in_w]`              T   (NCHW, batch 1)
//!   out    `[n_patches, patch_dim_padded]`       T
//!   n_patches = grid_h · grid_w,  in_h = grid_h·patch,  in_w = grid_w·patch
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with
//! `grid_1d(n_patches · patch_dim_padded, 256)`.
//!   - `patch_dim_padded >= channels · patch · patch`.
//!   - `in_h == grid_h · patch`, `in_w == grid_w · patch`.

use metaltile::kernel;

#[kernel]
pub fn ffai_im2col_patch_interleaved<T>(
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] channels: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch: u32,
    #[constexpr] grid_w: u32,
    #[constexpr] patch_dim_padded: u32,
    #[constexpr] scale: f32,
    #[constexpr] bias: f32,
) {
    let idx = program_id::<0>();
    let col = idx % patch_dim_padded;
    let p_idx = idx / patch_dim_padded;
    let real_dim = channels * patch * patch;
    // Pad columns: write 0, and decode from a safe in-bounds index.
    let valid = col < real_dim;
    let cc = select(valid, col, 0u32);
    // Within-patch coordinate, channel innermost: (py, px, c).
    let c = cc % channels;
    let t1 = cc / channels;
    let px = t1 % patch;
    let py = t1 / patch;
    // Patch grid coordinate.
    let pc = p_idx % grid_w;
    let pr = p_idx / grid_w;
    let in_y = pr * patch + py;
    let in_x = pc * patch + px;
    let in_idx = (c * in_h + in_y) * in_w + in_x;
    let v = load(input[in_idx]).cast::<f32>();
    let mapped = v * scale + bias;
    let out_val = select(valid, mapped, 0.0f32);
    store(out[idx], out_val.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_im2col_patch_interleaved;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// NCHW → `[n_patches, patch_dim_padded]` interleaved-unfold oracle:
    /// channel-innermost columns, `scale·x + bias` re-map, zero pad tail.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        input: &[f32],
        channels: usize,
        in_h: usize,
        in_w: usize,
        patch: usize,
        grid_h: usize,
        grid_w: usize,
        patch_dim_padded: usize,
        scale: f32,
        bias: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; grid_h * grid_w * patch_dim_padded];
        for pr in 0..grid_h {
            for pc in 0..grid_w {
                let p_idx = pr * grid_w + pc;
                for py in 0..patch {
                    for px in 0..patch {
                        for c in 0..channels {
                            let col = (py * patch + px) * channels + c;
                            let in_y = pr * patch + py;
                            let in_x = pc * patch + px;
                            let v = input[(c * in_h + in_y) * in_w + in_x];
                            out[p_idx * patch_dim_padded + col] = v * scale + bias;
                        }
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        dt: DType,
        channels: usize,
        patch: usize,
        grid_h: usize,
        grid_w: usize,
        patch_dim_padded: usize,
        scale: f32,
        bias: f32,
    ) -> TestSetup {
        let in_h = grid_h * patch;
        let in_w = grid_w * patch;
        let n_out = grid_h * grid_w * patch_dim_padded;
        let input_f = ramp(channels * in_h * in_w, 17, 5.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = naive(
            &input,
            channels,
            in_h,
            in_w,
            patch,
            grid_h,
            grid_w,
            patch_dim_padded,
            scale,
            bias,
        );
        TestSetup::new(ffai_im2col_patch_interleaved::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("channels", channels as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch", patch as u32)
            .constexpr("grid_w", grid_w as u32)
            .constexpr("patch_dim_padded", patch_dim_padded as u32)
            .constexpr("scale", scale)
            .constexpr("bias", bias)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Gemma 4: 3 channels, patch 14, non-square 6×9 grid, re-centre 2·(x−0.5),
    // row padded from 3·14·14 = 588 up to 592 (K-tile 16 multiple).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_interleaved_gemma4(dt: DType) -> TestSetup { setup(dt, 3, 14, 6, 9, 592, 2.0, -1.0) }

    // No padding, identity re-map (scale 1, bias 0): a plain interleaved unfold.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-4, 1e-4])]
    fn test_interleaved_nopad(dt: DType) -> TestSetup { setup(dt, 3, 4, 2, 3, 48, 1.0, 0.0) }
}

/// New-syntax bench: Gemma 4 patch-unfold (3 ch, patch 14, 28×28 grid).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_im2col_patch_interleaved;

    #[bench(name = "ffai/im2col/im2col_patch_interleaved", dtypes = [f32, f16, bf16])]
    fn bench_interleaved(dt: DType) -> BenchSetup {
        let (channels, patch, grid_h, grid_w, patch_dim_padded) =
            (3usize, 14usize, 28usize, 28usize, 592usize);
        let in_h = grid_h * patch;
        let in_w = grid_w * patch;
        let n_out = grid_h * grid_w * patch_dim_padded;
        BenchSetup::new(ffai_im2col_patch_interleaved::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", channels * in_h * in_w, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("channels", channels as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch", patch as u32)
            .constexpr("grid_w", grid_w as u32)
            .constexpr("patch_dim_padded", patch_dim_padded as u32)
            .constexpr("scale", 2.0f32)
            .constexpr("bias", -1.0f32)
            .grid_1d(n_out, 256)
            .bytes_moved((2 * n_out * dt.size_bytes()) as u64)
    }
}

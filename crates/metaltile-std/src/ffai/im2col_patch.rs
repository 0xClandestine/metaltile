//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Patch im2col — unfold an NCHW image into one row per (non-overlapping)
//! `patch × patch` block, flattened `(c, py, px)`, ready for the patch-embed
//! GEMM.
//!
//! The gather every patch-embed vision tower does before its embed matmul:
//! `out[patch, (c·p + py)·p + px] = input[c, pr·p + py, pc·p + px]` where the
//! patch grid is `grid_h × grid_w = (H/p) × (W/p)` and `patch = pr·grid_w +
//! pc`. Replaces the scalar CPU unfold loops (Gemma 4 `unfoldPatches`, and
//! any patch-embed tower) with a single GPU gather, so the resulting
//! `[n_patches, channels·p·p]` rows feed straight into `Ops.gemm`.
//!
//! Non-overlapping (stride == patch, no padding) — the patch-embed case. For
//! the temporal / K-tile-padded row widths some towers want, the caller
//! sizes `patch_dim` accordingly and zero-fills the pad columns separately.
//!
//! Layouts:
//!   input  `[channels, in_h, in_w]`                     T   (NCHW, batch 1)
//!   out    `[n_patches, channels·patch·patch]`          T
//!   n_patches = grid_h · grid_w,  in_h = grid_h·patch,  in_w = grid_w·patch
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with
//! `grid_1d(n_patches · channels · patch · patch, 256)`.

use metaltile::kernel;

#[kernel]
pub fn im2col_patch<T>(
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] channels: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch: u32,
    #[constexpr] grid_w: u32,
) {
    let idx = program_id::<0>();
    let patch_dim = channels * patch * patch;
    let col = idx % patch_dim;
    let p_idx = idx / patch_dim;
    // Within-patch coordinate: (c, py, px), px fastest.
    let px = col % patch;
    let t1 = col / patch;
    let py = t1 % patch;
    let c = t1 / patch;
    // Patch grid coordinate.
    let pc = p_idx % grid_w;
    let pr = p_idx / grid_w;
    let in_y = pr * patch + py;
    let in_x = pc * patch + px;
    let in_idx = (c * in_h + in_y) * in_w + in_x;
    store(out[idx], load(input[in_idx]));
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::im2col_patch;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// NCHW → `[n_patches, channels·p·p]` patch-unfold oracle.
    fn naive(
        input: &[f32],
        channels: usize,
        in_h: usize,
        in_w: usize,
        patch: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Vec<f32> {
        let patch_dim = channels * patch * patch;
        let mut out = vec![0.0f32; grid_h * grid_w * patch_dim];
        for pr in 0..grid_h {
            for pc in 0..grid_w {
                let p_idx = pr * grid_w + pc;
                for c in 0..channels {
                    for py in 0..patch {
                        for px in 0..patch {
                            let col = (c * patch + py) * patch + px;
                            let in_y = pr * patch + py;
                            let in_x = pc * patch + px;
                            out[p_idx * patch_dim + col] = input[(c * in_h + in_y) * in_w + in_x];
                        }
                    }
                }
            }
        }
        out
    }

    fn setup(dt: DType, channels: usize, patch: usize, grid_h: usize, grid_w: usize) -> TestSetup {
        let in_h = grid_h * patch;
        let in_w = grid_w * patch;
        let n_out = grid_h * grid_w * channels * patch * patch;
        let input_f = ramp(channels * in_h * in_w, 17, 5.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = naive(&input, channels, in_h, in_w, patch, grid_h, grid_w);
        TestSetup::new(im2col_patch::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("channels", channels as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch", patch as u32)
            .constexpr("grid_w", grid_w as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // SigLIP/Gemma-style 14-patch, 3 channels, 8×8 patch grid (112×112 px).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-4, 1e-4])]
    fn test_im2col_patch_p14(dt: DType) -> TestSetup { setup(dt, 3, 14, 8, 8) }

    // Non-square grid (variable resolution): 6×9 patch grid, patch 16.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-4, 1e-4])]
    fn test_im2col_patch_nonsquare(dt: DType) -> TestSetup { setup(dt, 3, 16, 6, 9) }
}

/// New-syntax bench: Gemma 4 patch-unfold (3 ch, patch 14, 28×28 grid).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::im2col_patch;

    #[bench(name = "ffai/im2col/im2col_patch", dtypes = [f32, f16, bf16])]
    fn bench_im2col_patch(dt: DType) -> BenchSetup {
        let (channels, patch, grid_h, grid_w) = (3usize, 14usize, 28usize, 28usize);
        let in_h = grid_h * patch;
        let in_w = grid_w * patch;
        let n_out = grid_h * grid_w * channels * patch * patch;
        BenchSetup::new(im2col_patch::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", channels * in_h * in_w, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("channels", channels as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch", patch as u32)
            .constexpr("grid_w", grid_w as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((2 * n_out * dt.size_bytes()) as u64)
    }
}

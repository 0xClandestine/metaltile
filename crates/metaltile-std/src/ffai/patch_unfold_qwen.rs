//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Qwen-VL patch unfold — temporal-group + spatial-merge-block patch raster
//! with `(t, c, py, px)` columns. The gather Qwen2-VL / Qwen2.5-VL / Qwen3-VL
//! vision towers do before their patch-embed matmul.
//!
//! Qwen differs from the plain [`super::im2col_patch`] in three ways:
//!   1. Each patch row stacks `temporal_patch` frames, then channel, then the
//!      `patch × patch` block: column order `(t_within, c, py, px)` (px
//!      fastest), width `temporal_patch · channels · patch · patch`.
//!   2. Patches are emitted in a **merge-block raster**: the `side × side`
//!      patch grid is tiled into `merge_blocks × merge_blocks` blocks of
//!      `merge × merge` patches; the row index walks
//!      `(t_group, block_row, block_col, in_row, in_col)`.
//!   3. Multiple frames feed one row group. `frames` is a contiguous
//!      `[n_frames, channels, img_side, img_side]` buffer; for a still image
//!      (`is_image = 1`) the single frame is reused for every `t_within`,
//!      otherwise frame `t_group · temporal_patch + t_within` is read.
//!
//! Columns `[temporal_patch·channels·patch·patch, patch_dim_padded)` are
//! written as `0` (K-tile pad). No pixel re-map (Qwen feeds raw normalized
//! pixels).
//!
//! Layouts:
//!   frames `[n_frames, channels, img_side, img_side]`   T   (img_side = side·patch)
//!   out    `[n_patches, patch_dim_padded]`              T
//!   n_patches = grid_t · side · side,  side = merge_blocks · merge
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with
//! `grid_1d(n_patches · patch_dim_padded, 256)`.
//!   - `patch_dim_padded >= temporal_patch · channels · patch · patch`.
//!   - `img_side == merge_blocks · merge · patch`.
//!   - `is_image ∈ {0, 1}`; if `1`, `n_frames == 1`, else
//!     `n_frames == grid_t · temporal_patch`.

use metaltile::kernel;

#[kernel]
pub fn ffai_patch_unfold_qwen<T>(
    frames: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] channels: u32,
    #[constexpr] patch: u32,
    #[constexpr] temporal_patch: u32,
    #[constexpr] merge: u32,
    #[constexpr] merge_blocks: u32,
    #[constexpr] img_side: u32,
    #[constexpr] patch_dim_padded: u32,
    #[constexpr] is_image: u32,
) {
    let idx = program_id::<0>();
    let col = idx % patch_dim_padded;
    let p_idx = idx / patch_dim_padded;
    let real_dim = temporal_patch * channels * patch * patch;
    let valid = col < real_dim;
    let cc = select(valid, col, 0u32);
    // Within-row column order (t_within, c, py, px), px fastest.
    let px = cc % patch;
    let t1 = cc / patch;
    let py = t1 % patch;
    let t2 = t1 / patch;
    let c = t2 % channels;
    let t_within = t2 / channels;
    // Merge-block patch raster: (t_group, block_row, block_col, in_row, in_col).
    let ic = p_idx % merge;
    let s1 = p_idx / merge;
    let ir = s1 % merge;
    let s2 = s1 / merge;
    let bc = s2 % merge_blocks;
    let s3 = s2 / merge_blocks;
    let br = s3 % merge_blocks;
    let t_group = s3 / merge_blocks;
    let pr = br * merge + ir;
    let pc = bc * merge + ic;
    // Frame select: a still image reuses frame 0 for every temporal slot.
    let frame_idx = select(is_image == 1u32, 0u32, t_group * temporal_patch + t_within);
    let yy = pr * patch + py;
    let xx = pc * patch + px;
    let frame_stride = channels * img_side * img_side;
    let in_idx = frame_idx * frame_stride + (c * img_side + yy) * img_side + xx;
    let v = load(frames[in_idx]).cast::<f32>();
    let out_val = select(valid, v, 0.0f32);
    store(out[idx], out_val.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_patch_unfold_qwen;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// CPU oracle — mirrors FFAI's Qwen `unfoldPatches` exactly.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        frames: &[f32],
        channels: usize,
        patch: usize,
        temporal_patch: usize,
        merge: usize,
        merge_blocks: usize,
        img_side: usize,
        patch_dim_padded: usize,
        grid_t: usize,
        is_image: bool,
    ) -> Vec<f32> {
        let side = merge_blocks * merge;
        let n_patches = grid_t * side * side;
        let frame_stride = channels * img_side * img_side;
        let mut out = vec![0.0f32; n_patches * patch_dim_padded];
        let mut p_idx = 0usize;
        for t_group in 0..grid_t {
            for br in 0..merge_blocks {
                for bc in 0..merge_blocks {
                    for ir in 0..merge {
                        for ic in 0..merge {
                            let pr = br * merge + ir;
                            let pc = bc * merge + ic;
                            let mut col = 0usize;
                            for t_within in 0..temporal_patch {
                                let frame_idx =
                                    if is_image { 0 } else { t_group * temporal_patch + t_within };
                                for c in 0..channels {
                                    for py in 0..patch {
                                        let yy = pr * patch + py;
                                        for px in 0..patch {
                                            let xx = pc * patch + px;
                                            let v = frames[frame_idx * frame_stride
                                                + (c * img_side + yy) * img_side
                                                + xx];
                                            out[p_idx * patch_dim_padded + col] = v;
                                            col += 1;
                                        }
                                    }
                                }
                            }
                            p_idx += 1;
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
        temporal_patch: usize,
        merge: usize,
        merge_blocks: usize,
        patch_dim_padded: usize,
        grid_t: usize,
        is_image: bool,
    ) -> TestSetup {
        let side = merge_blocks * merge;
        let img_side = side * patch;
        let n_frames = if is_image { 1 } else { grid_t * temporal_patch };
        let n_patches = grid_t * side * side;
        let n_out = n_patches * patch_dim_padded;
        let frames_f = ramp(n_frames * channels * img_side * img_side, 23, 4.0);
        let frames = unpack_f32(&pack_f32(&frames_f, dt), dt);
        let expected = naive(
            &frames,
            channels,
            patch,
            temporal_patch,
            merge,
            merge_blocks,
            img_side,
            patch_dim_padded,
            grid_t,
            is_image,
        );
        TestSetup::new(ffai_patch_unfold_qwen::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("frames", pack_f32(&frames_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("channels", channels as u32)
            .constexpr("patch", patch as u32)
            .constexpr("temporal_patch", temporal_patch as u32)
            .constexpr("merge", merge as u32)
            .constexpr("merge_blocks", merge_blocks as u32)
            .constexpr("img_side", img_side as u32)
            .constexpr("patch_dim_padded", patch_dim_padded as u32)
            .constexpr("is_image", if is_image { 1u32 } else { 0u32 })
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // Still image: 3 ch, patch 2, temporal 2 (frame reused), merge 2, 2 blocks
    // (side 4, img 8×8), grid_t 1. real_dim = 2·3·4 = 24, padded to 32.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_qwen_image(dt: DType) -> TestSetup { setup(dt, 3, 2, 2, 2, 2, 32, 1, true) }

    // Video: distinct frames, grid_t 2 (4 frames), same geometry.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_qwen_video(dt: DType) -> TestSetup { setup(dt, 3, 2, 2, 2, 2, 32, 2, false) }
}

/// New-syntax bench: Qwen2.5-VL image unfold (3 ch, patch 14, temporal 2,
/// merge 2, 8 blocks → 16×16 patch grid, 224×224 px).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_patch_unfold_qwen;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_patch_unfold_qwen(dt: DType) -> BenchSetup {
        let (channels, patch, temporal_patch, merge, merge_blocks) =
            (3usize, 14usize, 2usize, 2usize, 8usize);
        let side = merge_blocks * merge;
        let img_side = side * patch;
        let patch_dim = temporal_patch * channels * patch * patch;
        let patch_dim_padded = patch_dim.div_ceil(16) * 16;
        let n_patches = side * side;
        let n_out = n_patches * patch_dim_padded;
        BenchSetup::new(ffai_patch_unfold_qwen::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("frames", channels * img_side * img_side, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("channels", channels as u32)
            .constexpr("patch", patch as u32)
            .constexpr("temporal_patch", temporal_patch as u32)
            .constexpr("merge", merge as u32)
            .constexpr("merge_blocks", merge_blocks as u32)
            .constexpr("img_side", img_side as u32)
            .constexpr("patch_dim_padded", patch_dim_padded as u32)
            .constexpr("is_image", 1u32)
            .grid_1d(n_out, 256)
            .bytes_moved((2 * n_out * dt.size_bytes()) as u64)
    }
}

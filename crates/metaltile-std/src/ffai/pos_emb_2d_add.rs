//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Factorized 2D position-embedding add — gather a row/column position
//! vector for each patch and add it to the patch token.
//!
//! The position-embed step the Gemma 4 vision tower (and other factorized-
//! 2D-pos towers) apply after the patch-embed: each patch at grid coordinate
//! `(row, col)` gets `pos_x[col]` + `pos_y[row]` added to its token. Replaces
//! the scalar CPU position loop with one GPU pass.
//!
//!   `out[patch, d] = tokens[patch, d] + pos_x[col·hidden + d]
//!                                     + pos_y[row·hidden + d]`
//!   where `col = patch % grid_w`, `row = patch / grid_w`.
//!
//! `pos_x` / `pos_y` are the (already resolution-matched) per-axis tables —
//! for a variable-resolution tower the caller interpolates the learned table
//! to `grid_w` / `grid_h` rows first (a resize), then this kernel does the
//! gather+add. `pos_x` / `pos_y` are f32 regardless of T.
//!
//! Layouts:
//!   tokens  `[n_patches, hidden]`   T
//!   pos_x   `[grid_w, hidden]`      f32
//!   pos_y   `[grid_h, hidden]`      f32
//!   out     `[n_patches, hidden]`   T     (n_patches = grid_h · grid_w)
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element `(patch, d)` — dispatch with
//! `grid_1d(n_patches · hidden, 256)`.

use metaltile::kernel;

#[kernel]
pub fn pos_emb_2d_add<T>(
    tokens: Tensor<T>,
    pos_x: Tensor<f32>,
    pos_y: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] grid_w: u32,
) {
    let idx = program_id::<0>();
    let d = idx % hidden;
    let patch = idx / hidden;
    let col = patch % grid_w;
    let row = patch / grid_w;
    let tok = load(tokens[idx]).cast::<f32>();
    let px = load(pos_x[col * hidden + d]);
    let py = load(pos_y[row * hidden + d]);
    store(out[idx], (tok + px + py).cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::pos_emb_2d_add;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    fn naive(
        tokens: &[f32],
        pos_x: &[f32],
        pos_y: &[f32],
        hidden: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; grid_h * grid_w * hidden];
        for patch in 0..grid_h * grid_w {
            let col = patch % grid_w;
            let row = patch / grid_w;
            for d in 0..hidden {
                out[patch * hidden + d] =
                    tokens[patch * hidden + d] + pos_x[col * hidden + d] + pos_y[row * hidden + d];
            }
        }
        out
    }

    fn setup(dt: DType, hidden: usize, grid_h: usize, grid_w: usize) -> TestSetup {
        let n_patches = grid_h * grid_w;
        let tokens_f = ramp(n_patches * hidden, 0.013, -0.4);
        let pos_x = ramp(grid_w * hidden, 0.011, -0.2);
        let pos_y = ramp(grid_h * hidden, 0.009, 0.1);
        let tokens = unpack_f32(&pack_f32(&tokens_f, dt), dt);
        let expected = naive(&tokens, &pos_x, &pos_y, hidden, grid_h, grid_w);
        TestSetup::new(pos_emb_2d_add::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("tokens", pack_f32(&tokens_f, dt), dt))
            .input(TestBuffer::from_vec("pos_x", pack_f32(&pos_x, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("pos_y", pack_f32(&pos_y, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n_patches * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .constexpr("grid_w", grid_w as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_patches * hidden, 256)
    }

    // Square grid, hidden 128.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 4e-3, 2e-2])]
    fn test_pos_emb_2d_add_square(dt: DType) -> TestSetup { setup(dt, 128, 8, 8) }

    // Non-square (variable resolution) grid 6×9.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 4e-3, 2e-2])]
    fn test_pos_emb_2d_add_nonsquare(dt: DType) -> TestSetup { setup(dt, 96, 6, 9) }
}

/// New-syntax bench: Gemma 4 vision pos-emb (28×28 grid, hidden 1152).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::pos_emb_2d_add;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_pos_emb_2d_add(dt: DType) -> BenchSetup {
        let (hidden, grid_h, grid_w) = (1152usize, 28usize, 28usize);
        let n_patches = grid_h * grid_w;
        BenchSetup::new(pos_emb_2d_add::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("tokens", n_patches * hidden, dt))
            .buffer(BenchBuffer::random("pos_x", grid_w * hidden, DType::F32))
            .buffer(BenchBuffer::random("pos_y", grid_h * hidden, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_patches * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .constexpr("grid_w", grid_w as u32)
            .grid_1d(n_patches * hidden, 256)
            .bytes_moved((2 * n_patches * hidden * dt.size_bytes()) as u64)
    }
}

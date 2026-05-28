//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Scatter along an axis — contiguous form of MLX's `scatter_axis`.
//!
//! `out[o, indices[o, a, i], i] = updates[o, a, i]` — each update
//! element is written to a row-`indices`-selected slot of `out`. One
//! thread per update element. `out` is pre-initialized by the caller
//! (typically a copy of the source) and the kernel overwrites the
//! scattered slots.
//!
//! Layout (row-contiguous):
//!   updates: [outer, axis_upd,  inner]  T
//!   indices: [outer, axis_upd,  inner]  u32
//!   out:     [outer, axis_size, inner]  T  (pre-initialized)
//!
//! Assignment (no-reduce) form: distinct `indices` are required for a
//! deterministic result — colliding indices race, matching MLX
//! `scatter_axis` with `reduce = None`. The general strided + reducing
//! kernel is a follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, one thread per update element over `outer*axis_upd*inner`.
//!
//! Codegen-only; correctness pinned by
//! `tests/scatter_axis_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel]
pub fn mt_scatter_axis<T>(
    updates: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] axis_upd: u32,
    #[constexpr] axis_size: u32,
    #[constexpr] inner: u32,
) {
    let idx = program_id::<0>();
    let i = idx - (idx / inner) * inner;
    let o = idx / (axis_upd * inner);
    let scattered = load(indices[idx]);
    let out_off = (o * axis_size + scattered) * inner + i;
    store(out[out_off], load(updates[idx]));
}

// ── bottom of source file ─────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    #[test_kernel(name = "mlx/scatter_axis_f32", dtypes = [f32], tol = 1e-6)]
    fn test_scatter_axis_matches_naive_f32(dt: DType) -> TestSetup {
        let (outer, axis_size, axis_upd, inner) = (3usize, 7usize, 5usize, 4usize);
        let mut indices = vec![0u32; outer * axis_upd * inner];
        for o in 0..outer {
            for i in 0..inner {
                for a in 0..axis_upd {
                    indices[(o * axis_upd + a) * inner + i] = ((a + o + i) % axis_size) as u32;
                }
            }
        }
        let updates: Vec<f32> =
            (0..outer * axis_upd * inner).map(|i| i as f32 * 0.25 + 1.0).collect();
        let init: Vec<f32> = (0..outer * axis_size * inner).map(|i| -(i as f32)).collect();
        let mut expected = init.clone();
        for o in 0..outer {
            for a in 0..axis_upd {
                for i in 0..inner {
                    let ui = (o * axis_upd + a) * inner + i;
                    let s = indices[ui] as usize;
                    expected[(o * axis_size + s) * inner + i] = updates[ui];
                }
            }
        }
        let total = outer * axis_upd * inner;
        let mut k = mt_scatter_axis::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("updates", pack_f32(&updates), dt))
            .input(TestBuffer::from_vec("indices", pack_u32(&indices), DType::U32))
            .input(TestBuffer::from_vec("out", pack_f32(&init), dt))
            .constexpr("axis_upd", axis_upd as u32)
            .constexpr("axis_size", axis_size as u32)
            .constexpr("inner", inner as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(total.div_ceil(64) as u32, 1, 1, [64, 1, 1])
    }
}

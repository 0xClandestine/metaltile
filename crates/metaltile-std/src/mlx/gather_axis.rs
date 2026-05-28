//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gather along an axis — contiguous form of MLX's `gather_axis`.
//!
//! `out[o, a, i] = src[o, indices[o, a, i], i]` — for each output
//! element, the middle (axis) coordinate is looked up from `indices`
//! while the outer/inner coordinates pass through. One thread per
//! output element.
//!
//! Layout (row-contiguous):
//!   src:     [outer, axis_size, inner]  T
//!   indices: [outer, axis_out,  inner]  u32
//!   out:     [outer, axis_out,  inner]  T
//!
//! The general MLX kernel handles arbitrary strides / non-contiguous
//! src+idx via `elem_to_loc`; this port covers the row-contiguous case
//! (the shape `ensureRowContiguous` produces).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, one thread per output element over `outer*axis_out*inner`.
//!
//! Codegen-only; correctness pinned by
//! `tests/gather_axis_gpu_correctness.rs`.

use metaltile::kernel;

#[kernel]
pub fn mt_gather_axis<T>(
    src: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] axis_out: u32,
    #[constexpr] axis_size: u32,
    #[constexpr] inner: u32,
) {
    let idx = program_id::<0>();
    // out / indices share shape [outer, axis_out, inner]; `idx` indexes
    // both directly. Only the outer coord `o` and inner coord `i` are
    // needed to re-address `src` (which has `axis_size`, not `axis_out`).
    let i = idx - (idx / inner) * inner;
    let o = idx / (axis_out * inner);
    let gathered = load(indices[idx]);
    let src_off = (o * axis_size + gathered) * inner + i;
    store(out[idx], load(src[src_off]));
}

// ── bottom of source file ─────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    #[test_kernel(name = "mlx/gather_axis_f32", dtypes = [f32], tol = 1e-6)]
    fn test_gather_axis_matches_naive_f32(dt: DType) -> TestSetup {
        let (outer, axis_size, axis_out, inner) = (3usize, 7usize, 5usize, 4usize);
        let src: Vec<f32> = (0..outer * axis_size * inner).map(|i| i as f32 * 0.5 - 2.0).collect();
        let indices: Vec<u32> =
            (0..outer * axis_out * inner).map(|i| ((i * 3 + 1) % axis_size) as u32).collect();
        let mut expected = vec![0.0_f32; outer * axis_out * inner];
        for o in 0..outer {
            for a in 0..axis_out {
                for i in 0..inner {
                    let oi = (o * axis_out + a) * inner + i;
                    let g = indices[oi] as usize;
                    expected[oi] = src[(o * axis_size + g) * inner + i];
                }
            }
        }
        let total = outer * axis_out * inner;
        let mut k = mt_gather_axis::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("src", pack_f32(&src), dt))
            .input(TestBuffer::from_vec("indices", pack_u32(&indices), DType::U32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; total]), dt))
            .constexpr("axis_out", axis_out as u32)
            .constexpr("axis_size", axis_size as u32)
            .constexpr("inner", inner as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(total.div_ceil(64) as u32, 1, 1, [64, 1, 1])
    }
}

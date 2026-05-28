//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Embedding-table gather. For each output element `(token, d)`: copy
//! `table[indices[token], d]`. One thread per output element.
//!
//! Bare-tensor (non-quantized) variant for embedding lookups.
//! Quantized embeddings live in `dequant_gather.rs`.
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::kernel;

#[kernel]
pub fn ffai_gather<T>(
    table: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] dim: u32,
) {
    let idx = program_id::<0>();
    let token = idx / dim;
    let d = idx - token * dim;
    let token_id = load(indices[token]);
    let src = token_id * dim + d;
    store(out[idx], load(table[src]));
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

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    #[test_kernel(name = "ffai/gather_f32", dtypes = [f32], tol = 1e-6)]
    fn test_gather_copies_correct_rows_f32(dt: DType) -> TestSetup {
        let vocab = 17usize;
        let dim = 8usize;
        let n_tokens = 6usize;
        let table: Vec<f32> =
            (0..vocab * dim).map(|i| ((i / dim) * 1000 + (i % dim)) as f32).collect();
        let indices: Vec<u32> = vec![3, 0, 11, 7, 11, 16];
        let mut expected = vec![0.0f32; n_tokens * dim];
        for (token_i, &id) in indices.iter().enumerate() {
            for d in 0..dim {
                expected[token_i * dim + d] = (id as usize * 1000 + d) as f32;
            }
        }
        let mut k = ffai_gather::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;
        let total = n_tokens * dim;
        let tpg = 256usize;
        let groups = total.div_ceil(tpg);
        TestSetup::new(k)
            .input(TestBuffer::from_vec("table", pack(&table, dt), dt))
            .input(TestBuffer::from_vec("indices", pack_u32(&indices), DType::U32))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; total], dt), dt))
            .constexpr("dim", dim as u32)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(groups as u32, 1, 1, [tpg as u32, 1, 1])
    }

    #[test_kernel(name = "ffai/gather_f16", dtypes = [f16], tol = 1e-5)]
    fn test_gather_qwen_shape_f16(dt: DType) -> TestSetup {
        let vocab = 64usize;
        let dim = 32usize;
        let n_tokens = 4usize;
        let table: Vec<f32> = (0..vocab * dim)
            .map(|i| half::f16::from_f32((i % 257) as f32 * 0.01 - 1.0).to_f32())
            .collect();
        let indices: Vec<u32> = vec![31, 0, 63, 17];
        let mut expected = vec![0.0f32; n_tokens * dim];
        for (token_i, &id) in indices.iter().enumerate() {
            for d in 0..dim {
                expected[token_i * dim + d] = table[id as usize * dim + d];
            }
        }
        let mut k = ffai_gather::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;
        let total = n_tokens * dim;
        let tpg = 256usize;
        let groups = total.div_ceil(tpg);
        TestSetup::new(k)
            .input(TestBuffer::from_vec("table", pack(&table, dt), dt))
            .input(TestBuffer::from_vec("indices", pack_u32(&indices), DType::U32))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; total], dt), dt))
            .constexpr("dim", dim as u32)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(groups as u32, 1, 1, [tpg as u32, 1, 1])
    }

    #[test_kernel(name = "ffai/gather_bf16", dtypes = [bf16], tol = 1e-3)]
    fn test_gather_qwen_shape_bf16(dt: DType) -> TestSetup {
        let vocab = 32usize;
        let dim = 16usize;
        let n_tokens = 3usize;
        let table: Vec<f32> = (0..vocab * dim)
            .map(|i| half::bf16::from_f32((i % 257) as f32 * 0.01 - 1.0).to_f32())
            .collect();
        let indices: Vec<u32> = vec![17, 0, 31];
        let mut expected = vec![0.0f32; n_tokens * dim];
        for (token_i, &id) in indices.iter().enumerate() {
            for d in 0..dim {
                expected[token_i * dim + d] = table[id as usize * dim + d];
            }
        }
        let mut k = ffai_gather::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;
        let total = n_tokens * dim;
        let tpg = 256usize;
        let groups = total.div_ceil(tpg);
        TestSetup::new(k)
            .input(TestBuffer::from_vec("table", pack(&table, dt), dt))
            .input(TestBuffer::from_vec("indices", pack_u32(&indices), DType::U32))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; total], dt), dt))
            .constexpr("dim", dim as u32)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(groups as u32, 1, 1, [tpg as u32, 1, 1])
    }
}

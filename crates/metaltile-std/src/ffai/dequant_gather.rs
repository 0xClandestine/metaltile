//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MLX-format dequantizing gather kernels (quantized embedding tables).
//! For each output element `(token, d)`: look up the packed weight,
//! extract the right value, dequantize via `q * scale + bias`.
//!
//! Layouts (per dtype, with H = `hidden`, G = `group_size`):
//!
//!   weight   [vocab, H * bits / 32]   uint32
//!   scales   [vocab, H / G]           T
//!   biases   [vocab, H / G]           T
//!   indices  [n_tokens]               u32
//!   out      [n_tokens, H]            T
//!
//! One thread per output element.  All bit widths share one formula:
//! element `d` occupies bits `[d*bits, (d+1)*bits)` in the row's bit stream,
//! spanning at most two adjacent u32 words.
//!
//! ```text
//!   bit_off  = d * bits
//!   word_idx = bit_off / 32
//!   bit_in_w = bit_off & 31
//!   lo_bits  = min(bits, 32 - bit_in_w)        ← bits from word 0
//!   spill    = bits - lo_bits                   ← bits from word 1
//!   lo       = (w0 >> bit_in_w) & ((1 << lo_bits) - 1)
//!   hi       = (w1 & ((1 << spill) - 1)) << lo_bits
//!   q        = lo | hi
//! ```
//!
//! When `spill == 0`, `w1` loads from `word_idx` (same as w0) so the address
//! is always in-bounds; the `(1 << 0) - 1 == 0` mask zeroes `hi` regardless.
//!
//! ## Macro structure
//!
//! `dequant_gather_kernel!` emits the entire `#[kernel(bench(...))] pub fn …`
//! at module scope.  The compiler expands the outer macro before the
//! `#[kernel]` proc-macro runs, so the body parser sees concrete tokens
//! with `$bits` already substituted.  Embedding the body inside an *inner*
//! `macro_rules!` call (the previous shape of this file) silently produced
//! empty kernels — the proc-macro doesn't expand inner declarative macros.

use metaltile::kernel;

macro_rules! dequant_gather_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            indices: Tensor<u32>,
            out: Tensor<T>,
            #[constexpr] hidden: u32,
            #[constexpr] group_size: u32,
        ) {
            let idx = program_id::<0>();
            let token = idx / hidden;
            let d = idx - token * hidden;
            let token_id = load(indices[token]);

            let groups_per_row = hidden / group_size;
            let g = d / group_size;
            let u32_per_row = hidden * $bits / 32u32;
            let row_off = token_id * u32_per_row;

            let bit_off = d * $bits;
            let word_idx = bit_off / 32u32;
            let bit_in_w = bit_off & 31u32;

            let bits_in_w0 = 32u32 - bit_in_w;
            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
            let spill = $bits - lo_bits;

            let w0 = load(weight[row_off + word_idx]);
            let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
            let w1 = load(weight[row_off + w1_idx]);

            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
            let q = lo | hi;

            let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
            let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
            let w_real = q.cast::<f32>() * scale + bias;
            store(out[idx], w_real.cast::<T>());
        }
    };
}

dequant_gather_kernel!(dequant_gather_int3, 3u32, "int3");
dequant_gather_kernel!(dequant_gather_int4, 4u32, "int4");
dequant_gather_kernel!(dequant_gather_int5, 5u32, "int5");
dequant_gather_kernel!(dequant_gather_int6, 6u32, "int6");
dequant_gather_kernel!(dequant_gather_int8, 8u32, "int8");

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn quantize_row_int4(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let hidden = row.len();
        let n_groups = hidden / group_size;
        let mut packed = vec![0u32; hidden / 8];
        let mut scales = vec![0.0_f32; n_groups];
        let mut biases = vec![0.0_f32; n_groups];
        for g in 0..n_groups {
            let g_slice = &row[g * group_size..(g + 1) * group_size];
            let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / 15.0 };
            scales[g] = scale;
            biases[g] = mn;
            for (i, &v) in g_slice.iter().enumerate() {
                let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
                let d = g * group_size + i;
                packed[d / 8] |= q << ((d % 8) * 4);
            }
        }
        (packed, scales, biases)
    }

    fn naive_dequant_gather(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        indices: &[u32],
        hidden: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let n_tokens = indices.len();
        let groups_per_row = hidden / group_size;
        let u32_per_row = hidden / 8;
        let mut out = vec![0.0_f32; n_tokens * hidden];
        for token in 0..n_tokens {
            let token_id = indices[token] as usize;
            for d in 0..hidden {
                let word = weight[token_id * u32_per_row + d / 8];
                let q = ((word >> ((d % 8) * 4)) & 0xf) as f32;
                let g = d / group_size;
                out[token * hidden + d] = q * scales[token_id * groups_per_row + g]
                    + biases[token_id * groups_per_row + g];
            }
        }
        out
    }

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_u32_bytes(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    #[test_kernel(name = "ffai/dequant_gather_int4/f32", dtypes = [f32], tol = 1e-4)]
    fn test_dequant_gather_int4_f32(dt: DType) -> TestSetup {
        let vocab = 8usize;
        let hidden = 256usize;
        let group_size = 64usize;
        let n_groups = hidden / group_size;
        let mut weight: Vec<u32> = Vec::new();
        let mut scales: Vec<f32> = Vec::new();
        let mut biases: Vec<f32> = Vec::new();
        for r in 0..vocab {
            let row: Vec<f32> = (0..hidden).map(|d| (((r + d) % 17) as f32 - 8.0) * 0.05).collect();
            let (pk, sc, bs) = quantize_row_int4(&row, group_size);
            weight.extend(pk);
            scales.extend(sc);
            biases.extend(bs);
        }
        let indices: Vec<u32> = vec![3, 0, 7, 1, 4, 4];
        let n_tokens = indices.len();
        let expected =
            naive_dequant_gather(&weight, &scales, &biases, &indices, hidden, group_size);
        let kernel = dequant_gather_int4::kernel_ir_for(dt);
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("weight", pack_u32_bytes(&weight), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases), dt))
            .input(TestBuffer::from_vec("indices", pack_u32_bytes(&indices), DType::U32))
            .input(TestBuffer::from_vec("out", vec![0u8; n_tokens * hidden * 4], dt))
            .input(TestBuffer::from_vec(
                "hidden",
                (hidden as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "group_size",
                (group_size as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(n_tokens as u32, 1, 1, [hidden as u32, 1, 1])
    }
}

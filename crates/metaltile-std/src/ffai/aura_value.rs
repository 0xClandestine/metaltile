//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA compressed-domain value aggregation.
//!
//! For each (q_head, dim) output element, computes
//! `Σ_t weight[head, t] · norm[kv_head, t] · codebook[unpack(packed[t, d])]`,
//! skipping tokens whose weight is below `sparse_threshold`.
//!
//! Port of `turbo_value` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `weights   [q_heads, tokens]`                    f32   — softmax(scores).
//! - `packed    [kv_heads, tokens, packed_width]`     u32   — codebook indices.
//! - `norms     [kv_heads, tokens]`                   f32   — per-position norm.
//! - `codebook  [2**bits]`                            f32   — centroids.
//!
//! Output:
//! - `output    [q_heads, dim]`                       f32
//!
//! ## Dispatch
//!
//! Grid3D, one thread per (q_head, dim) output element.
//! `gid.x = d`, `gid.y = head_idx`.  Each thread runs a single
//! sequential loop over tokens and accumulates its dim slot's
//! contribution.  Sparsity check (`w >= sparse_threshold`) skips
//! cheap-to-zero tokens, mirroring the MLX upstream's
//! flash-pass2-style aggregation guard.

use metaltile::kernel;

#[rustfmt::skip]
macro_rules! aura_value_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weights: Tensor<T>,
            packed: Tensor<u32>,
            norms: Tensor<T>,
            codebook: Tensor<T>,
            mut output: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] sparse_threshold: f32,
        ) {
            let d = program_id::<0>();
            let head_idx = program_id::<1>();
            let kv_head = head_idx / repeat_count;
            let mask = (1u32 << $bits) - 1u32;

            // Pre-compute the bit-stream coordinates for this thread's
            // dim slot.  Same for every token — only the base packed
            // pointer changes per t.
            let bit_offset = d * $bits;
            let word_idx = bit_offset / 32u32;
            let shift = bit_offset & 31u32;
            let bits_in_w0 = 32u32 - shift;
            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
            let spill = $bits - lo_bits;

            let mut acc = 0.0f32;
            for t in range(0u32, tokens, 1u32) {
                let w = load(weights[head_idx * tokens + t]).cast::<f32>();
                if w >= sparse_threshold {
                    let norm_val = load(norms[kv_head * tokens + t]).cast::<f32>();
                    let packed_row = (kv_head * tokens + t) * packed_width;

                    let w0 = load(packed[packed_row + word_idx]);
                    let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(packed[packed_row + w1_idx]);
                    let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let value = (lo | hi) & mask;

                    let centroid = load(codebook[value]).cast::<f32>();
                    acc = acc + w * norm_val * centroid;
                }
            }

            store(output[head_idx * dim + d], acc.cast::<T>());
        }
    };
}

aura_value_kernel!(aura_value_int2, 2u32, "value_int2");
aura_value_kernel!(aura_value_int3, 3u32, "value_int3");
aura_value_kernel!(aura_value_int4, 4u32, "value_int4");
aura_value_kernel!(aura_value_int6, 6u32, "value_int6");
aura_value_kernel!(aura_value_int8, 8u32, "value_int8");

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    fn pack_int4_indices(indices: &[u32], kv_heads: usize, tokens: usize, dim: usize) -> Vec<u32> {
        let bits = 4;
        let packed_width = (dim * bits).div_ceil(32);
        let mut packed = vec![0u32; kv_heads * tokens * packed_width];
        for kvh in 0..kv_heads {
            for t in 0..tokens {
                for d in 0..dim {
                    let idx = indices[(kvh * tokens + t) * dim + d];
                    let bit_offset = d * bits;
                    let word_idx = bit_offset / 32;
                    let shift = bit_offset & 31;
                    packed[(kvh * tokens + t) * packed_width + word_idx] |= (idx & 0xf) << shift;
                }
            }
        }
        packed
    }

    fn naive_aura_value(
        weights: &[f32],
        indices: &[u32],
        norms: &[f32],
        codebook: &[f32],
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        dim: usize,
        sparse_threshold: f32,
    ) -> Vec<f32> {
        let repeat = q_heads / kv_heads;
        let mut out = vec![0.0_f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            for d in 0..dim {
                let mut acc = 0.0_f32;
                for t in 0..tokens {
                    let w = weights[qh * tokens + t];
                    if w >= sparse_threshold {
                        let norm_val = norms[kvh * tokens + t];
                        let q = indices[(kvh * tokens + t) * dim + d];
                        let centroid = codebook[q as usize];
                        acc += w * norm_val * centroid;
                    }
                }
                out[qh * dim + d] = acc;
            }
        }
        out
    }

    #[test_kernel(name = "ffai/aura/value_int4", dtypes = [f32], tol = 1e-4)]
    fn test_aura_value_int4(dt: DType) -> TestSetup {
        use super::aura_value_int4;
        let dim = 128usize;
        let bits = 4usize;
        let packed_width = (dim * bits).div_ceil(32);
        let q_heads = 4usize;
        let kv_heads = 2usize;
        let tokens = 8usize;
        let repeat = q_heads / kv_heads;
        let sparse_threshold = 0.0_f32;

        let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let indices: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 13 + 5) % 16) as u32).collect();
        let packed = pack_int4_indices(&indices, kv_heads, tokens, dim);
        let norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.4 + 0.07 * i as f32).collect();
        let weights: Vec<f32> = (0..q_heads * tokens)
            .map(|i| {
                let phase = (i * 7 + 3) % 10;
                phase as f32 * 0.04
            })
            .collect();

        let expected = naive_aura_value(
            &weights,
            &indices,
            &norms,
            &codebook,
            q_heads,
            kv_heads,
            tokens,
            dim,
            sparse_threshold,
        );

        let mut kernel_ir = aura_value_int4::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("weights", pack(&weights, dt), dt))
            .input(TestBuffer::from_vec("packed", pack_u32(&packed), DType::U32))
            .input(TestBuffer::from_vec("norms", pack(&norms, dt), dt))
            .input(TestBuffer::from_vec("codebook", pack(&codebook, dt), dt))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec("packed_width", u32_le(packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("repeat_count", u32_le(repeat as u32), DType::U32))
            .input(TestBuffer::from_vec(
                "sparse_threshold",
                sparse_threshold.to_le_bytes().to_vec(),
                DType::F32,
            ))
            .expect(TestBuffer::from_vec("output", pack(&expected, dt), dt))
            .grid_3d(1, q_heads as u32, 1, [dim as u32, 1, 1])
    }
}

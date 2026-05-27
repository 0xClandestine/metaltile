//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA compressed-domain Q · K dot-product reduction.
//!
//! For each (q_head, k_position) pair, computes the dot product of the
//! rotated query vector against the codebook-quantised key vector at
//! that position, scaled by the per-position norm-correction factor.
//!
//! Port of `turbo_score` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `q_rot     [q_heads, dim]`                       f32
//! - `packed    [kv_heads, tokens, packed_width]`     u32
//! - `norms     [kv_heads, tokens]`                   f32
//! - `codebook  [2**bits]`                            f32
//!
//! Output:
//! - `scores    [q_heads, tokens]`                    f32
//!
//! ## Dispatch
//!
//! Reduction-mode kernel.  Threadgroup = (32, 1, 1); one threadgroup
//! per (q_head, token) pair via tgid_x = q_idx, tgid_y = k_idx.
//! Each of the 32 lanes accumulates a dim-strided slice of the dot
//! product; `simd_sum` reduces across the simdgroup.
//!
//! ## Constexpr params
//!
//! - `bits`            — 2 / 3 / 4 / 8.
//! - `dim`             — vector length.
//! - `packed_width`    — `ceil(dim * bits / 32)`.
//! - `repeat_count`    — GQA repeat factor (`n_q_heads / n_kv_heads`).
//!   When 1 (MHA), `kv_idx == q_idx`.
//!
//! ## Tradeoff vs the MLX upstream
//!
//! MLX caches the codebook in a per-thread stack array
//! (`float cb[LEVELS]`) before the inner loop, amortising LEVELS
//! lookups across `dim/32` iterations.  The DSL doesn't yet expose
//! stack-allocated arrays; we re-read `codebook[value]` per lookup.
//! The codebook is small (≤ 1 KB at bits=8) and Metal L1-caches
//! tightly enough that this is functionally equivalent — re-evaluate
//! if `tile profile` shows codebook reads dominating later.

use metaltile::kernel;

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestBuffer, TestSetup}};

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32  => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16  => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _           => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

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

    fn pack_u32(vals: &[u32]) -> Vec<u8> {
        bytemuck::cast_slice::<u32, u8>(vals).to_vec()
    }

    fn naive_aura_score(
        q_rot: &[f32], indices: &[u32], norms: &[f32], codebook: &[f32],
        q_heads: usize, kv_heads: usize, tokens: usize, dim: usize,
    ) -> Vec<f32> {
        let repeat = q_heads / kv_heads;
        let mut scores = vec![0.0_f32; q_heads * tokens];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            for t in 0..tokens {
                let norm_val = norms[kvh * tokens + t];
                let mut acc = 0.0_f32;
                for d in 0..dim {
                    let q = indices[(kvh * tokens + t) * dim + d];
                    let centroid = codebook[q as usize];
                    acc += q_rot[qh * dim + d] * centroid;
                }
                scores[qh * tokens + t] = acc * norm_val;
            }
        }
        scores
    }

    #[test_kernel(name = "ffai/aura/score_int4", dtypes = [f32], tol = 1e-3)]
    fn test_aura_score_int4(dt: DType) -> TestSetup {
        use super::aura_score_int4;
        let dim = 128usize;
        let bits = 4usize;
        let packed_width = (dim * bits).div_ceil(32);
        let q_heads = 4usize;
        let kv_heads = 2usize;
        let tokens = 8usize;
        let repeat = q_heads / kv_heads;

        let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let indices: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 11 + 7) % 16) as u32).collect();
        let packed = pack_int4_indices(&indices, kv_heads, tokens, dim);
        let norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.05 * i as f32).collect();
        let q_rot: Vec<f32> =
            (0..q_heads * dim).map(|i| (((i * 13) % 21) as f32 - 10.0) * 0.02).collect();

        let expected = naive_aura_score(&q_rot, &indices, &norms, &codebook, q_heads, kv_heads, tokens, dim);

        let mut kernel_ir = aura_score_int4::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("q_rot", pack(&q_rot, dt), dt))
            .input(TestBuffer::from_vec("packed", pack_u32(&packed), DType::U32))
            .input(TestBuffer::from_vec("norms", pack(&norms, dt), dt))
            .input(TestBuffer::from_vec("codebook", pack(&codebook, dt), dt))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec("packed_width", u32_le(packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("repeat_count", u32_le(repeat as u32), DType::U32))
            .expect(TestBuffer::from_vec("scores", pack(&expected, dt), dt))
            .grid_3d(q_heads as u32, tokens as u32, 1, [32, 1, 1])
    }
}

macro_rules! aura_score_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            q_rot: Tensor<T>,
            packed: Tensor<u32>,
            norms: Tensor<T>,
            codebook: Tensor<T>,
            mut scores: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
        ) {
            let lane = tid;
            let q_idx = tgid_x;
            let k_idx = tgid_y;
            let kv_idx = q_idx / repeat_count;

            let mask = (1u32 << $bits) - 1u32;
            let q_off = q_idx * dim;
            let packed_row = (kv_idx * tokens + k_idx) * packed_width;
            let norm_val = load(norms[kv_idx * tokens + k_idx]).cast::<f32>();

            // Lane-strided accumulation over dim.  Each lane handles
            // dims `[lane, lane + 32, lane + 64, …)` so the threadgroup
            // covers the whole vector when reduced via simd_sum.
            let mut acc = 0.0f32;
            let iters = (dim + 31u32) / 32u32;
            for it in range(0u32, iters, 1u32) {
                let d = it * 32u32 + lane;
                if d < dim {
                    // Bit-stream extract.  For bits ∈ {2,4,8} the
                    // window never spills; for {3,6} it can — branch
                    // on `shift + bits > 32` and re-fetch the next
                    // word.  Same shape as dequant_gather_int{3,6}.
                    let bit_offset = d * $bits;
                    let word_idx = bit_offset / 32u32;
                    let shift = bit_offset & 31u32;
                    let bits_in_w0 = 32u32 - shift;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;

                    let w0 = load(packed[packed_row + word_idx]);
                    let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(packed[packed_row + w1_idx]);

                    let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let value = (lo | hi) & mask;

                    let centroid = load(codebook[value]).cast::<f32>();
                    let qv = load(q_rot[q_off + d]).cast::<f32>();
                    acc = acc + qv * centroid;
                }
            }

            // Reduce across the 32 lanes.  Only lane 0 writes the
            // result back, scaled by the per-position norm correction.
            let total = simd_sum(acc);
            if lane == 0u32 {
                store(scores[q_idx * tokens + k_idx], (total * norm_val).cast::<T>());
            }
        }
    };
}

aura_score_kernel!(aura_score_int2, 2u32, "score_int2");
aura_score_kernel!(aura_score_int3, 3u32, "score_int3");
aura_score_kernel!(aura_score_int4, 4u32, "score_int4");
aura_score_kernel!(aura_score_int6, 6u32, "score_int6");
aura_score_kernel!(aura_score_int8, 8u32, "score_int8");

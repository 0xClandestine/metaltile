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
/// AURA quantized value-aggregation kernel — variable bit-widths (2, 3, 4, 6, 8).
///
/// Produces kernels: `aura_value_int2`, `aura_value_int3`, `aura_value_int4`,
/// `aura_value_int6`, `aura_value_int8`.
///
/// For each (dim, head) thread: unpacks the `BITS`-wide code at position `d`
/// from the LSB-first bit-stream, fetches the codebook centroid, and accumulates
/// `w * norm * centroid` over all tokens above `sparse_threshold`.
///
/// Grid: Grid3D, `[dim, n_heads, 1]`.
#[kernel(variants(BITS = [2, 3, 4, 6, 8], suffix = "int{BITS}"))]
pub fn aura_value<T>(
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
    let mask = (1u32 << BITS) - 1u32;

    // Pre-compute the bit-stream coordinates for this thread's
    // dim slot.  Same for every token — only the base packed
    // pointer changes per t.
    let bit_offset = d * BITS;
    let word_idx = bit_offset / 32u32;
    let shift = bit_offset & 31u32;
    let bits_in_w0 = 32u32 - shift;
    let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
    let spill = BITS - lo_bits;

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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::aura_value_int4;
    use crate::utils::{pack_f32, unpack_f32};

    fn round(v: f32, dt: DType) -> f32 { unpack_f32(&pack_f32(&[v], dt), dt)[0] }
    fn pack_u32(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

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

    /// Build an int4 value-aggregation test for a given head layout and
    /// sparsity threshold. `repeat_count = q_heads / kv_heads` selects the
    /// GQA fan-out (1 = MHA), and `sparse_threshold > 0` exercises the
    /// per-token skip branch (`if w >= sparse_threshold`).
    fn value_setup(dt: DType, q_heads: usize, kv_heads: usize, sparse_threshold: f32) -> TestSetup {
        let (dim, tokens) = (128usize, 8usize);
        let repeat = q_heads / kv_heads;
        let packed_width = (dim * 4).div_ceil(32);
        let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let indices: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 13 + 5) % 16) as u32).collect();
        let packed = pack_int4_indices(&indices, kv_heads, tokens, dim);
        let norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.4 + 0.07 * i as f32).collect();
        // Weights span [0.05, 0.35]; a 0.1 threshold drops the smallest few
        // (the values 0.05 / 0.08), so the skip branch actually fires.
        let weights: Vec<f32> =
            (0..q_heads * tokens).map(|i| 0.05 + ((i * 7 % 11) as f32) * 0.03).collect();

        let codebook_r: Vec<f32> = codebook.iter().map(|&v| round(v, dt)).collect();
        let norms_r: Vec<f32> = norms.iter().map(|&v| round(v, dt)).collect();
        let weights_r: Vec<f32> = weights.iter().map(|&v| round(v, dt)).collect();

        let mut expected = vec![0.0_f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            for d in 0..dim {
                let mut acc = 0.0_f32;
                for t in 0..tokens {
                    let w = weights_r[qh * tokens + t];
                    if w >= sparse_threshold {
                        let q = indices[(kvh * tokens + t) * dim + d];
                        acc += w * norms_r[kvh * tokens + t] * codebook_r[q as usize];
                    }
                }
                expected[qh * dim + d] = acc;
            }
        }

        TestSetup::new(aura_value_int4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("weights", pack_f32(&weights_r, dt), dt))
            .input(TestBuffer::from_vec("packed", pack_u32(&packed), DType::U32))
            .input(TestBuffer::from_vec("norms", pack_f32(&norms_r, dt), dt))
            .input(TestBuffer::from_vec("codebook", pack_f32(&codebook_r, dt), dt))
            .input(TestBuffer::zeros("output", q_heads * dim, dt))
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            .constexpr("tokens", tokens as u32)
            .constexpr("repeat_count", repeat as u32)
            .constexpr("sparse_threshold", sparse_threshold)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            // tpg=1 → total threads == (dim, q_heads, 1).
            .grid_3d(dim as u32, q_heads as u32, 1, [1, 1, 1])
    }

    // GQA: 4 q-heads over 2 kv-heads (repeat 2), every token kept.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-1)]
    fn test_aura_value_int4(dt: DType) -> TestSetup { value_setup(dt, 4, 2, 0.0) }

    // MHA: q-heads == kv-heads (repeat 1) — the identity head mapping.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-1)]
    fn test_aura_value_int4_mha(dt: DType) -> TestSetup { value_setup(dt, 4, 4, 0.0) }

    // Wide GQA fan-out: 8 q-heads over 2 kv-heads (repeat 4).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-1)]
    fn test_aura_value_int4_gqa4(dt: DType) -> TestSetup { value_setup(dt, 8, 2, 0.0) }

    // Sparse threshold 0.1 skips the lowest-weight tokens — exercises the
    // `w >= sparse_threshold` skip branch the other configs leave dormant.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-1)]
    fn test_aura_value_int4_sparse(dt: DType) -> TestSetup { value_setup(dt, 4, 2, 0.1) }
}

/// New-syntax benchmarks for the AURA value family (int2/3/4/6/8) — MLX-less
/// Grid3D kernels. Decode shape: head_dim 128, 32 q-heads, 8 kv-heads, 4096 tokens.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::*;

    fn setup(
        s: BenchSetup,
        dim: usize,
        bits: usize,
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        dt: DType,
    ) -> BenchSetup {
        let packed_width = (dim * bits).div_ceil(32);
        let levels = 1usize << bits;
        let repeat = q_heads / kv_heads;
        s.mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("weights", q_heads * tokens, dt))
            .buffer(BenchBuffer::random("packed", kv_heads * tokens * packed_width, DType::U32))
            .buffer(BenchBuffer::random("norms", kv_heads * tokens, dt))
            .buffer(BenchBuffer::random("codebook", levels, dt))
            .buffer(BenchBuffer::zeros("output", q_heads * dim, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            .constexpr("tokens", tokens as u32)
            .constexpr("repeat_count", repeat as u32)
            // 0.0 → no token skipped (worst case for bandwidth measurement).
            .constexpr("sparse_threshold", 0.0_f32)
            .bytes_moved((kv_heads * tokens * packed_width * 4) as u64)
            .grid_3d(dim as u32, q_heads as u32, 1, [1, 1, 1])
    }

    #[bench(dtypes = [f32, f16, bf16],
            variants(BITS = [2, 3, 4, 6, 8], suffix = "int{BITS}"))]
    fn bench_aura_value(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_value_intBITS::kernel_ir_for(dt)), 128, BITS, 32, 8, 4096, dt)
    }
}

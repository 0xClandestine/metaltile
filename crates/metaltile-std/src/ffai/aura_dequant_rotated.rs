//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA bulk dequant — unpack codebook-quantized values into rotated
//! codec space, ready to be consumed by the AURA flash-SDPA path or
//! materialised as a fp16/bf16 tensor for downstream SDPA.
//!
//! Port of `turbo_dequant_rotated` from
//! `ekryski/mlx@alpha`/`mlx/backend/metal/kernels/turbo_quant.metal`.
//!
//! ## Layout
//!
//! Input:
//! - `packed [B*H, T, packed_width]` u32  — bit-packed codebook indices.
//!   `packed_width = ceil(dim * bits / 32)`.
//! - `norms  [B*H, T]`               T    — per-token norm correction; cast-
//!   at-load to f32 internally.
//! - `codebook [2**bits]`            T    — Lloyd-Max centroids; cast-at-load
//!   to f32 internally.
//!
//! Output:
//! - `out  [B*H, T, dim]`            T    — fp16 / bf16 / fp32 in rotated
//!   codec space; caller applies the inverse rotation (e.g. via
//!   flash-SDPA p2-with-fused-rot).
//!
//! ## Bit-extract paths
//!
//! - `bits ∈ {2, 4, 8}`: 32 / bits divides cleanly → each packed word
//!   holds exactly `32 / bits` quantized dims with no cross-word spill.
//!   Inner loop emits `DIMS_PER_WORD` outputs per thread with a single load.
//! - `bits ∈ {3, 6}`: odd-width packs straddle word boundaries.  Each
//!   per-dim emit re-fetches `packed[word_idx]` (and `packed[word_idx+1]`
//!   if spilling) to grab the bits whose absolute offset is `d * bits`.
//!
//! ## Variant axis
//!
//! `#[kernel(variants(BITS = [2, 3, 4, 6, 8], suffix = "int{BITS}"))]`
//! emits one kernel module per bit-width. A compile-time
//! `if 32u32 % BITS == 0` selects the pack-strided path (BITS ∈ {2,4,8}) or the
//! element-strided bit-stream path (BITS ∈ {3,6}). BITS=5 is absent — AURA
//! does not ship int5.

use metaltile::kernel;

/// AURA codebook dequant + norm-scale, Grid3D dispatch.
///
/// Produces kernels: `aura_dequant_rotated_int2`, `_int3`, `_int4`,
/// `_int6`, `_int8`.
///
/// Even BITS (2, 4, 8): pack-strided — one u32 load amortises across all
/// `32/BITS` values in the pack.
/// Odd BITS (3, 6): element-strided bit-stream — reads `packed[word_idx]`
/// and optionally `packed[word_idx+1]` per dim.
///
/// Grid: (packed_width, tokens, B*H), tpg=[1,1,1].
#[kernel(variants(BITS = [2, 3, 4, 6, 8], suffix = "int{BITS}"))]
pub fn aura_dequant_rotated<T>(
    packed: Tensor<u32>,
    norms: Tensor<T>,
    codebook: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] dim: u32,
    #[constexpr] packed_width: u32,
    #[constexpr] tokens: u32,
) {
    let w = program_id::<0>();
    let t = program_id::<1>();
    let bh = program_id::<2>();

    let mask = (1u32 << BITS) - 1u32;
    let base = (bh * tokens + t) * packed_width;
    let norm_val = load(norms[bh * tokens + t]).cast::<f32>();

    if 32u32 % BITS == 0 {
        // Pack-strided path: one u32 load per thread covers `32/BITS` dims.
        let dims_per_word = 32u32 / BITS;
        let word = load(packed[base + w]);
        let d_base = w * dims_per_word;
        let out_row_base = (bh * tokens + t) * dim + d_base;
        for k in range(0u32, dims_per_word, 1u32) {
            let d = d_base + k;
            if d < dim {
                let val = (word >> (k * BITS)) & mask;
                let centroid = load(codebook[val]).cast::<f32>();
                let result = centroid * norm_val;
                store(out[out_row_base + k], result.cast::<T>());
            }
        }
    } else {
        // Element-strided bit-stream path: handles cross-word spills.
        let dims_per_word = (32u32 + BITS - 1u32) / BITS;
        let d_base = w * dims_per_word;
        for k in range(0u32, dims_per_word, 1u32) {
            let d = d_base + k;
            if d < dim {
                let bit_offset = d * BITS;
                let word_idx = bit_offset / 32u32;
                let bit_in_w = bit_offset & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                let spill = BITS - lo_bits;
                let w0 = load(packed[base + word_idx]);
                let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                let w1 = load(packed[base + w1_idx]);
                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                let val = (lo | hi) & mask;
                let centroid = load(codebook[val]).cast::<f32>();
                let result = centroid * norm_val;
                store(out[(bh * tokens + t) * dim + d], result.cast::<T>());
            }
        }
    }
}

/// Correctness tests for the `aura_dequant_rotated_int{2,3,4,6,8}` family.
/// Grid3D, one thread per (packed_word, token, bh) tile. Oracle: decode each
/// codebook index from the packed bit-stream and multiply by the token norm.
/// dim=128 is u32-aligned for every supported bit-width (128×2/32=8,
/// 128×3/32=12, 128×4/32=16, 128×6/32=24, 128×8/32=32 exact).
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn pack_u32(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

    /// Bit-pack a flat `[bh, t, dim]` index array into `[bh, t, packed_width]`
    /// u32 words using `bits`-wide fields — handles cross-word spills.
    fn pack_bitstream_indices(
        indices: &[u32],
        bh: usize,
        tokens: usize,
        dim: usize,
        bits: usize,
    ) -> Vec<u32> {
        let packed_width = (dim * bits).div_ceil(32);
        let mut packed = vec![0u32; bh * tokens * packed_width];
        let mask = (1u32 << bits) - 1;
        for b in 0..bh {
            for t in 0..tokens {
                for d in 0..dim {
                    let idx = indices[(b * tokens + t) * dim + d] & mask;
                    let bit_offset = d * bits;
                    let word_idx = bit_offset / 32;
                    let shift = (bit_offset & 31) as u32;
                    let bits_in_w0 = 32 - shift as usize;
                    let row_base = (b * tokens + t) * packed_width;
                    if bits_in_w0 >= bits {
                        packed[row_base + word_idx] |= idx << shift;
                    } else {
                        packed[row_base + word_idx] |= idx << shift;
                        packed[row_base + word_idx + 1] |= idx >> bits_in_w0;
                    }
                }
            }
        }
        packed
    }

    fn dequant_setup(
        kernel: Kernel,
        bits: u32,
        dim: usize,
        bh: usize,
        tokens: usize,
        dt: DType,
    ) -> TestSetup {
        let packed_width = (dim * bits as usize).div_ceil(32);
        let levels = 1usize << bits;
        let codebook: Vec<f32> =
            (0..levels).map(|i| -1.0 + 2.0 * i as f32 / (levels - 1) as f32).collect();
        let indices: Vec<u32> =
            (0..bh * tokens * dim).map(|i| ((i * 7 + 3) as u32) % levels as u32).collect();
        let packed = pack_bitstream_indices(&indices, bh, tokens, dim, bits as usize);
        let norms: Vec<f32> = (0..bh * tokens).map(|i| 0.5 + 0.1 * i as f32).collect();
        let codebook_r = unpack_f32(&pack_f32(&codebook, dt), dt);
        let norms_r = unpack_f32(&pack_f32(&norms, dt), dt);
        let expected: Vec<f32> = (0..bh * tokens * dim)
            .map(|i| codebook_r[indices[i] as usize] * norms_r[i / dim])
            .collect();
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("packed", pack_u32(&packed), DType::U32))
            .input(TestBuffer::from_vec("norms", pack_f32(&norms, dt), dt))
            .input(TestBuffer::from_vec("codebook", pack_f32(&codebook, dt), dt))
            .input(TestBuffer::zeros("out", bh * tokens * dim, dt))
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            .constexpr("tokens", tokens as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(packed_width as u32, tokens as u32, bh as u32, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-1,
                  variants(BITS = [2, 3, 4, 6, 8], suffix = "int{BITS}"))]
    fn test_aura_dequant_rotated(dt: DType) -> TestSetup {
        dequant_setup(aura_dequant_rotated_intBITS::kernel_ir_for(dt), BITS, 128, 2, 3, dt)
    }
}

/// New-syntax benchmarks for the AURA bulk-dequant family (int2/3/4/6/8) —
/// Grid3D dispatch. Shape: dim=128, 64 tokens, 8 BH.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::*;

    fn setup(
        s: BenchSetup,
        dim: usize,
        bits: u32,
        bh: usize,
        tokens: usize,
        dt: DType,
    ) -> BenchSetup {
        let packed_width = (dim * bits as usize).div_ceil(32);
        let levels = 1usize << bits;
        s.mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("packed", bh * tokens * packed_width, DType::U32))
            .buffer(BenchBuffer::random("norms", bh * tokens, dt))
            .buffer(BenchBuffer::random("codebook", levels, dt))
            .buffer(BenchBuffer::zeros("out", bh * tokens * dim, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            .constexpr("tokens", tokens as u32)
            .bytes_moved((bh * tokens * dim * dt.size_bytes()) as u64)
            .grid_3d(packed_width as u32, tokens as u32, bh as u32, [1, 1, 1])
    }

    #[bench(dtypes = [f32, f16, bf16],
            variants(BITS = [2, 3, 4, 6, 8], suffix = "int{BITS}"))]
    fn bench_aura_dequant_rotated(dt: DType) -> BenchSetup {
        setup(
            BenchSetup::new(aura_dequant_rotated_intBITS::kernel_ir_for(dt)),
            128,
            BITS,
            8,
            64,
            dt,
        )
    }
}

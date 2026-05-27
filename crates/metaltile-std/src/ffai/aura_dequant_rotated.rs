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
//! - `norms  [B*H, T]`               f32  — per-token norm correction.
//! - `codebook [2**bits]`            f32  — Lloyd-Max centroids.
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
//!   Inner loop emits `DIMS_PER_WORD` outputs per thread with a single
//!   load.
//! - `bits ∈ {3, 5, 6}`: odd-width packs straddle word boundaries.  Each
//!   per-dim emit re-fetches `packed[word_idx]` (and `packed[word_idx+1]`
//!   if spilling) to grab the bits whose absolute offset is `d * bits`.
//!   Same logic as `dequant_gemv_int{3,5,6}` in the affine-quant path.
//!
//! ## Macro structure
//!
//! Outer `aura_dequant_rotated_clean!` (for bits ∈ {2,4,8}) and
//! `aura_dequant_rotated_odd!` (for bits=3) emit the entire
//! `#[kernel(bench(...))] pub fn …` at module scope.  Required because
//! the `#[kernel]` proc-macro doesn't expand inner `macro_rules!`
//! invocations (see CLAUDE.md note about PR #19's macro regression).

use metaltile::kernel;

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestBuffer, TestSetup}};

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }
    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn pack_int4_indices(indices: &[u32], bh: usize, tokens: usize, dim: usize) -> Vec<u32> {
        let bits = 4;
        let packed_width = (dim * bits).div_ceil(32);
        let mut packed = vec![0u32; bh * tokens * packed_width];
        for b in 0..bh {
            for t in 0..tokens {
                for d in 0..dim {
                    let idx = indices[(b * tokens + t) * dim + d];
                    let bit_offset = d * bits;
                    let word_idx = bit_offset / 32;
                    let shift = bit_offset & 31;
                    packed[(b * tokens + t) * packed_width + word_idx] |= (idx & 0xf) << shift;
                }
            }
        }
        packed
    }

    fn naive_aura_dequant(
        indices: &[u32], norms: &[f32], codebook: &[f32],
        bh: usize, tokens: usize, dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; bh * tokens * dim];
        for b in 0..bh {
            for t in 0..tokens {
                let norm_val = norms[b * tokens + t];
                for d in 0..dim {
                    let q = indices[(b * tokens + t) * dim + d];
                    out[(b * tokens + t) * dim + d] = codebook[q as usize] * norm_val;
                }
            }
        }
        out
    }

    #[test_kernel(name = "ffai/aura/dequant_rotated_int4", dtypes = [f32], tol = 1e-5)]
    fn test_aura_dequant_rotated_int4(dt: DType) -> TestSetup {
        use super::aura_dequant_rotated_int4;
        let dim = 128usize;
        let bits = 4usize;
        let packed_width = (dim * bits).div_ceil(32);
        let bh = 2usize;
        let tokens = 3usize;

        let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let indices: Vec<u32> =
            (0..bh * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
        let packed = pack_int4_indices(&indices, bh, tokens, dim);
        let norms: Vec<f32> = (0..bh * tokens).map(|i| 0.5 + 0.1 * i as f32).collect();

        let expected = naive_aura_dequant(&indices, &norms, &codebook, bh, tokens, dim);

        let mut kernel_ir = aura_dequant_rotated_int4::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("packed", pack_u32(&packed), DType::U32))
            .input(TestBuffer::from_vec("norms", pack_f32(&norms), DType::F32))
            .input(TestBuffer::from_vec("codebook", pack_f32(&codebook), DType::F32))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec("packed_width", u32_le(packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .grid_3d(1, tokens as u32, bh as u32, [packed_width as u32, 1, 1])
    }
}

// ── Clean nibble/byte path: bits ∈ {2, 4, 8} ─────────────────────────────
//
// Each thread owns one packed word w covering DIMS_PER_WORD = 32/bits
// dim slots starting at `d_base = w * DIMS_PER_WORD`.  One u32 load
// amortises across all dims in the pack.
#[rustfmt::skip]
macro_rules! aura_dequant_rotated_clean {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            packed: Tensor<u32>,
            norms: Tensor<f32>,
            codebook: Tensor<f32>,
            mut out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
        ) {
            // Dispatch grid is exactly (packed_width, tokens, B*H); Metal's
            // `dispatchThreads` doesn't pad, so the MLX-source's
            // `if (w >= packed_width) return;` guards are unnecessary
            // belt-and-suspenders.  Omitted here — the DSL has no early
            // `return`, and bounded `for k < dims_per_word` plus
            // `if d < dim` keeps any spurious thread from writing out of
            // bounds.
            let w = program_id::<0>();
            let t = program_id::<1>();
            let bh = program_id::<2>();

            let mask = (1u32 << $bits) - 1u32;
            let dims_per_word = 32u32 / $bits;

            let base = (bh * tokens + t) * packed_width;
            let word = load(packed[base + w]);
            let norm_val = load(norms[bh * tokens + t]);

            let d_base = w * dims_per_word;
            let out_row_base = (bh * tokens + t) * dim + d_base;
            for k in range(0u32, dims_per_word, 1u32) {
                let d = d_base + k;
                if d < dim {
                    let val = (word >> (k * $bits)) & mask;
                    let centroid = load(codebook[val]);
                    let result = centroid * norm_val;
                    store(out[out_row_base + k], result.cast::<T>());
                }
            }
        }
    };
}

// ── Odd-width spill path: bits ∈ {3, 5, 6} ───────────────────────────────
//
// Words straddle dim boundaries: thread `w` may need to read packed[w]
// AND packed[w+1] for any dim whose bit-range crosses word index 32.
// Same bit-stream formula as `dequant_gather_int{3,5,6}`.
//
// `ceil(32 / bits)` outputs per thread; `d_base = w * DIMS_PER_WORD`
// for the iteration but the bit-offset arithmetic is keyed on the
// absolute dim index `d`, so cross-word spills resolve correctly.
#[rustfmt::skip]
macro_rules! aura_dequant_rotated_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            packed: Tensor<u32>,
            norms: Tensor<f32>,
            codebook: Tensor<f32>,
            mut out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
            #[constexpr] tokens: u32,
        ) {
            // Dispatch grid is exactly (packed_width, tokens, B*H); Metal's
            // `dispatchThreads` doesn't pad, so the MLX-source's
            // `if (w >= packed_width) return;` guards are unnecessary
            // belt-and-suspenders.  Omitted here — the DSL has no early
            // `return`, and bounded `for k < dims_per_word` plus
            // `if d < dim` keeps any spurious thread from writing out of
            // bounds.
            let w = program_id::<0>();
            let t = program_id::<1>();
            let bh = program_id::<2>();

            let mask = (1u32 << $bits) - 1u32;
            let dims_per_word = (32u32 + $bits - 1u32) / $bits;

            let base = (bh * tokens + t) * packed_width;
            let norm_val = load(norms[bh * tokens + t]);

            let d_base = w * dims_per_word;
            for k in range(0u32, dims_per_word, 1u32) {
                let d = d_base + k;
                if d < dim {
                    let bit_offset = d * $bits;
                    let word_idx = bit_offset / 32u32;
                    let bit_in_w = bit_offset & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;

                    let w0 = load(packed[base + word_idx]);
                    let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(packed[base + w1_idx]);

                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let val = (lo | hi) & mask;

                    let centroid = load(codebook[val]);
                    let result = centroid * norm_val;
                    store(out[(bh * tokens + t) * dim + d], result.cast::<T>());
                }
            }
        }
    };
}

// Bit-width × dim instantiations.  AURA today supports kb ∈ {2,3,4,6,8}
// per the session plan (kb=5 isn't shipped); add new variants here when
// the planning doc adds another kb level.
aura_dequant_rotated_clean!(aura_dequant_rotated_int2, 2u32, "dequant_rotated_int2");
aura_dequant_rotated_clean!(aura_dequant_rotated_int4, 4u32, "dequant_rotated_int4");
aura_dequant_rotated_clean!(aura_dequant_rotated_int8, 8u32, "dequant_rotated_int8");
aura_dequant_rotated_odd!(aura_dequant_rotated_int3, 3u32, "dequant_rotated_int3");
aura_dequant_rotated_odd!(aura_dequant_rotated_int6, 6u32, "dequant_rotated_int6");

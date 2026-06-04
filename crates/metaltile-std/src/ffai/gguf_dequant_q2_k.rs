//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GGUF Q2_K block dequant — k-quant 2-bit-per-weight with two-level scales.
//!
//! Follows the canonical `dequantize_row_q2_K` reference algorithm.
//!
//! ## On-disk block layout (decomposed CPU-side at load time)
//!
//! ```text
//!   struct block_q2_K {
//!     uint8_t  scales[16];   // 16 bytes — low 4 bits = scale, high 4 bits = min
//!     uint8_t  qs[64];       // 64 bytes — 2-bit-packed quants, 4 vals per byte
//!     uint16_t d;            //  2 bytes — fp16 super-scale for scales
//!     uint16_t dmin;         //  2 bytes — fp16 super-scale for mins
//!   };                       // 84 bytes per 256 values (BPW = 2.625)
//! ```
//!
//! Per output value `i ∈ [0, 256)`:
//!
//! ```text
//!   sub        = i / 16           // 0..15, picks the (4-bit scale, 4-bit min) pair
//!   in_sub     = i & 15            // 0..15 inside the sub-block
//!   scale_byte = scales[sub]
//!   scale_4bit = scale_byte & 0xf
//!   min_4bit   = (scale_byte >> 4) & 0xf
//!   qs_byte    = qs[i / 4]
//!   shift      = (i & 3) * 2
//!   q_2bit     = (qs_byte >> shift) & 0x3
//!   out[i]     = d * scale_4bit * q_2bit - dmin * min_4bit
//! ```
//!
//! ## GPU-resident split (the loader produces these from the packed block)
//!
//! 1. `qs_packed [n_blocks * 16]`   — `u32`, the 64 packed-quant bytes
//!    per block re-laid as 16 u32 words. `qs_packed[block*16 + j]`
//!    carries 16 two-bit quants in the lower / upper bytes of each
//!    u32. Output index `i ∈ [0, 256)` → `u32 j = i / 16`, then a
//!    `(i % 16) * 2`-bit shift on the byte that holds it.
//! 2. `scales    [n_blocks * 16]`   — `u8`, the raw scale/min byte
//!    pairs (low nibble = scale, high nibble = min) — kept packed
//!    because both nibbles are used per dequant.
//! 3. `d_f32     [n_blocks]`        — `f32`, host-converted from fp16.
//! 4. `dmin_f32  [n_blocks]`        — `f32`, host-converted from fp16.
//!
//! ## Dispatch
//!
//! 1D grid: one thread per *output value*. ~6 reads (1 qs_packed + 1
//! scales + 1 each of d_f32 / dmin_f32, scales cache-multicast across
//! 16 lanes that share a sub-block) and ~4 arithmetic ops per output —
//! cleanly bandwidth-bound on Apple9.

use metaltile::kernel;

// Bare `#[kernel]` — see Q8_0 sibling for why; mixed concrete +
// generic param dtype set doesn't fit the legacy `bench(...)` shape.
#[kernel]
pub fn ffai_gguf_dequant_q2_k<T>(
    qs_packed: Tensor<u32>,
    scales: Tensor<u8>,
    d_f32: Tensor<f32>,
    dmin_f32: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    if i < n_values {
        let block = i / 256u32;
        let in_block = i - block * 256u32;
        // Canonical Q2_K block layout: the 256
        // values are NOT 4-consecutive-per-byte. They split as 2 halves of
        // 128; each half is 4 j-groups of 32; each j-group is two runs of 16
        // values that index 16 CONSECUTIVE qs bytes at a SHARED 2-bit shift
        // (shift = j*2). The naive in_block/4 mapping was wrong.
        let half = in_block / 128u32; // 0..1  → qs byte base half*32
        let yh = in_block - half * 128u32; // 0..127
        let jg = yh / 32u32; // 0..3 → shift = jg*2
        let yg = yh - jg * 32u32; // 0..31
        let sub_half = yg / 16u32; // 0..1
        let l = yg - sub_half * 16u32; // 0..15 → byte within the 16-run
        let shift = jg * 2u32;
        let q_byte = half * 32u32 + sub_half * 16u32 + l; // 0..63
        let scale_idx = half * 8u32 + jg * 2u32 + sub_half; // 0..15
        let word_idx = q_byte / 4u32;
        let byte_in_word = q_byte & 3u32;
        let word = load(qs_packed[block * 16u32 + word_idx]);
        let qs_byte = (word >> (byte_in_word * 8u32)) & 0xffu32;
        let q_2bit = (qs_byte >> shift) & 0x3u32;

        let scale_byte = load(scales[block * 16u32 + scale_idx]).cast::<u32>();
        let scale_4bit = scale_byte & 0xfu32;
        let min_4bit = (scale_byte >> 4u32) & 0xfu32;

        let d = load(d_f32[block]);
        let dmin = load(dmin_f32[block]);

        let scaled =
            d * (scale_4bit.cast::<i32>().cast::<f32>()) * (q_2bit.cast::<i32>().cast::<f32>());
        let offset = dmin * (min_4bit.cast::<i32>().cast::<f32>());
        // Store the f32 result directly: the DSL narrows f32→T implicitly
        // at the Store site (an explicit `.cast::<T>()` would emit a
        // spurious same-type MSL cast).
        store(out[i], scaled - offset);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gguf_dequant_q2_k;
    use crate::{quant::gguf, utils::pack_f32};

    fn setup(n_blocks: usize, dt: DType) -> TestSetup {
        let n = n_blocks * 256;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007 - 0.5).sin() * 1.5).collect();
        // Pack + dequant via the shared GgufFormat oracle (quant::gguf) — the
        // canonical q2_k_qpos map + two-level decode now live in one place the
        // kernel, the quantizer, and this oracle all share, so the oracle can't
        // drift from the kernel (the bug class fixed in #264).
        let p = gguf::pack_q2_k(&values);
        let dequantized = gguf::dequant_q2_k(&p);
        // Pack u32 vec as little-endian bytes for the test framework.
        let qs_bytes: Vec<u8> = p.qs_packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        TestSetup::new(ffai_gguf_dequant_q2_k::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("qs_packed", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", p.scales, DType::U8))
            .input(TestBuffer::from_vec("d_f32", pack_f32(&p.d, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("dmin_f32", pack_f32(&p.dmin, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n_values", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_single_block(dt: DType) -> TestSetup { setup(1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_many_blocks(dt: DType) -> TestSetup { setup(8, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gguf_dequant_q2_k;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_q2_k(dt: DType) -> BenchSetup {
        // Representative MoE-expert down-proj slab — 4096 × 4096.
        let n = 4096 * 4096usize;
        let n_blocks = n / 256;
        BenchSetup::new(ffai_gguf_dequant_q2_k::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("qs_packed", n_blocks * 16, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks * 16, DType::U8))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("dmin_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            // qs_packed 64 B + scales 16 B + 2*4 B per block + output T
            .bytes_moved(((n_blocks * (64 + 16 + 8)) + n * dt.size_bytes()) as u64)
    }
}

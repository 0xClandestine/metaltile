//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GGUF IQ2_XXS qs extraction: raw-bytes → packed `qs_u32`.
//!
//! Replaces the per-block CPU `memcpy(64)` loop that was eating ~26 ms
//! per expert (~470 ms / token across 18 routed experts at top-K=6 ×
//! 3 weight roles on DSv4-Flash). The extract kernel reads 4 raw
//! bytes from the on-disk 66-byte block layout, assembles one u32,
//! and writes to the packed `qs_u32` staging buffer the existing
//! `ffai_gguf_dequant_iq2_xxs` kernel expects.
//!
//! ## Block layout
//!
//! ```text
//!   struct block_iq2_xxs {
//!     uint16_t d;          // bytes 0..2 — fp16 super-scale
//!     uint8_t  qs[64];     // bytes 2..66 — packed grid index + sign payloads
//!   };                      // 66 bytes per 256 values
//! ```
//!
//! Each block contributes 16 × u32 words to the output stream:
//!
//! ```text
//!   qs_u32[block * 16 + w] = LE-u32 of raw_bytes[block * 66 + 2 + w * 4 .. +4]
//! ```
//!
//! ## Dispatch
//!
//! 1 thread per output u32. Grid: `n_blocks * 16` threads in 1D.
//! Each thread does 4 byte loads + 3 shifts + 3 ORs + 1 u32 store —
//! ~12 ops. Memory-bound on the 4 raw byte reads, but they're
//! contiguous so Apple GPU coalescing is friendly within a simdgroup.

use metaltile::kernel;

#[kernel]
pub fn ffai_gguf_iq2_xxs_extract_qs(
    raw_bytes: Tensor<u8>,
    mut qs_u32: Tensor<u32>,
    #[constexpr] n_blocks: u32,
) {
    let w = tid;
    let n_words = n_blocks * 16u32;
    if w < n_words {
        let block = w / 16u32;
        let word_in_block = w - block * 16u32;
        let byte_offset = block * 66u32 + 2u32 + word_in_block * 4u32;
        let b0 = load(raw_bytes[byte_offset]).cast::<u32>();
        let b1 = load(raw_bytes[byte_offset + 1u32]).cast::<u32>();
        let b2 = load(raw_bytes[byte_offset + 2u32]).cast::<u32>();
        let b3 = load(raw_bytes[byte_offset + 3u32]).cast::<u32>();
        let packed = b0 | (b1 << 8u32) | (b2 << 16u32) | (b3 << 24u32);
        store(qs_u32[w], packed);
    }
}

#[cfg(test)]
pub mod kernel_tests {

    use super::ffai_gguf_iq2_xxs_extract_qs;

    #[test]
    fn codegen_extract_qs_smoke() {
        let ir = ffai_gguf_iq2_xxs_extract_qs::kernel_ir_for();
        assert!(!ir.body.ops.is_empty(), "kernel body emitted no ops");
        assert!(ir.params.iter().any(|p| p.name == "raw_bytes"), "missing raw_bytes");
        assert!(ir.params.iter().any(|p| p.name == "qs_u32"), "missing qs_u32");
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gguf_iq2_xxs_extract_qs;

    #[bench(name = "ffai/gguf_iq2_xxs_extract_qs", dtypes = [u32])]
    fn bench_extract_qs(_dt: DType) -> BenchSetup {
        let n = 4096 * 4096usize;
        let n_blocks = n / 256;
        BenchSetup::new(ffai_gguf_iq2_xxs_extract_qs::kernel_ir_for())
            .buffer(BenchBuffer::random("raw_bytes", n_blocks * 66, DType::U8))
            .buffer(BenchBuffer::zeros("qs_u32", n_blocks * 16, DType::U32).output())
            .constexpr("n_blocks", n_blocks as u32)
            .grid_1d(n_blocks * 16, 256)
            .bytes_moved((n_blocks * 66 + n_blocks * 16 * 4) as u64)
    }
}

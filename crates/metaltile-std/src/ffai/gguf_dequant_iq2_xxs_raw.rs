//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GGUF IQ2_XXS dequant — raw-bytes variant.
//!
//! Same algorithm as `ffai_gguf_dequant_iq2_xxs` but reads `qs` from
//! the on-disk 66-byte block layout directly, skipping the CPU
//! preprocess that split each block into a separate `qs_u32` buffer.
//!
//! The block layout is `[u16 d (fp16); u8 qs[64]]` — 66 bytes per 256
//! values. `d_f32` is still passed as a pre-staged buffer because the
//! DSL has no `bit_cast<u32 → f32>` intrinsic for the in-kernel fp16
//! → f32 conversion; staging that single 32K-element vector on CPU
//! is ~30 ms per token vs the ~470 ms the per-block `qs` memcpy was
//! costing, so the asymmetric split is net positive.

use metaltile::kernel;

#[kernel]
pub fn ffai_gguf_dequant_iq2_xxs_raw<T>(
    raw_bytes: Tensor<u8>,
    d_f32: Tensor<f32>,
    grid: Tensor<u8>,
    signs: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    if i < n_values {
        let block = i / 256u32;
        let in_block = i - block * 256u32;
        let group = in_block / 32u32;
        let in_group = in_block & 31u32;
        let octet_within_index = in_group / 8u32;
        let lane_in_octet = in_group & 7u32;

        // Block base = block * 66, then skip the 2-byte fp16 d header
        // → qs region starts at +2. Each group occupies 8 bytes (one
        // `aux_idx` u32 + one `aux_sgn` u32).
        let qs_base = block * 66u32 + 2u32 + group * 8u32;
        let i0 = load(raw_bytes[qs_base]).cast::<u32>();
        let i1 = load(raw_bytes[qs_base + 1u32]).cast::<u32>();
        let i2 = load(raw_bytes[qs_base + 2u32]).cast::<u32>();
        let i3 = load(raw_bytes[qs_base + 3u32]).cast::<u32>();
        let aux_idx = i0 | (i1 << 8u32) | (i2 << 16u32) | (i3 << 24u32);
        let s0 = load(raw_bytes[qs_base + 4u32]).cast::<u32>();
        let s1 = load(raw_bytes[qs_base + 5u32]).cast::<u32>();
        let s2 = load(raw_bytes[qs_base + 6u32]).cast::<u32>();
        let s3 = load(raw_bytes[qs_base + 7u32]).cast::<u32>();
        let aux_sgn = s0 | (s1 << 8u32) | (s2 << 16u32) | (s3 << 24u32);

        let scale_4bit = aux_sgn >> 28u32;
        let scale_factor = (scale_4bit.cast::<i32>().cast::<f32>() + 0.5) * 0.25;
        let db = load(d_f32[block]) * scale_factor;

        let grid_key = (aux_idx >> (octet_within_index * 8u32)) & 0xffu32;
        let grid_row_base = grid_key * 8u32;
        let octet_u = load(grid[grid_row_base + lane_in_octet]).cast::<u32>();
        let octet = octet_u.cast::<i32>().cast::<f32>();

        let sign_idx = (aux_sgn >> (octet_within_index * 7u32)) & 0x7fu32;
        let sign_mask = load(signs[sign_idx]).cast::<u32>();
        let lane_bit = sign_mask & (1u32 << lane_in_octet);
        let sign = select(lane_bit != 0u32, -1.0f32, 1.0f32);

        store(out[i], db * sign * octet);
    }
}

#[cfg(test)]
pub mod kernel_tests {
    use metaltile::test::*;

    use super::ffai_gguf_dequant_iq2_xxs_raw;

    #[test]
    fn codegen_iq2_xxs_raw_smoke() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let ir = ffai_gguf_dequant_iq2_xxs_raw::kernel_ir_for(dt);
            assert!(!ir.body.ops.is_empty(), "kernel body emitted no ops for {dt:?}");
            assert!(ir.params.iter().any(|p| p.name == "raw_bytes"), "missing raw_bytes param");
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gguf_dequant_iq2_xxs_raw;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_iq2_xxs_raw(dt: DType) -> BenchSetup {
        let n = 4096 * 4096usize;
        let n_blocks = n / 256;
        BenchSetup::new(ffai_gguf_dequant_iq2_xxs_raw::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("raw_bytes", n_blocks * 66, DType::U8))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("grid", 2048, DType::U8))
            .buffer(BenchBuffer::random("signs", 128, DType::U8))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            .bytes_moved(((n_blocks * 66 + n_blocks * 4 + 2048 + 128) + n * dt.size_bytes()) as u64)
    }
}

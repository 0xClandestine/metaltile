//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! FFAI Q4 block dequant — expand a resident Q4 weight back to a dense
//! `[m, k]` f16/f32 slab for the compute-bound prefill GEMM path.
//!
//! The decode path keeps weights Q4-resident (1 nibble/weight) and runs
//! the inline-dequant `gemv_q4`, which is bandwidth-bound — perfect at
//! batch 1. Prefill is COMPUTE-bound: many tokens reuse each weight, so
//! the right move is to dequant the weight ONCE to f16 and feed the
//! tensor-core `ffai_gemm`. This kernel is that one-time expansion.
//!
//! ## Q4 block layout (matches `ffai_ops::quantize_q4`)
//!
//! Per row `r`, the `k` columns are grouped into `bpr = k/32` blocks of
//! 32 values. Each block is 4 packed u32 words (`word` 0..3), each word
//! holding 8 signed 4-bit nibbles (`i` 0..7) at bit `i*4`. The dequant
//! is `value = signed_nibble * scale`, `scale = scales[r*bpr + b]`
//! (`amax/7`, symmetric, no zero-point/bias).
//!
//! ```text
//!   qs      [m * (k/32) * 4]   u32   — 4 words/block, 8 nibbles/word
//!   scales  [m * (k/32)]       f32   — per-block scale
//!   out     [m * k]            T     — dense dequantized weight
//! ```
//!
//! ## Dispatch (1 thread per u32 WORD → 8 outputs)
//!
//! 1D grid, one thread per packed u32 word. There are `n_words = n_values/8`
//! words total (8 nibbles/word). Thread `i` owns word `i`: it does ONE word
//! load + ONE scale load and emits the 8 contiguous output values that word
//! decodes to. This amortises the global qs traffic 8× (the old 1-thread-per-
//! value layout had 8 adjacent threads reload the same word) and the scale
//! traffic 8× as well (one scale serves all 8 outputs of a word; a block's
//! 4 words still share a scale, but each word now loads it once instead of
//! once-per-nibble). Memory-bound win on every backend.
//!
//! Word `i` → row `r = i/words_per_row`, within-row word `wi = i%words_per_row`
//! (`words_per_row = (k/32)*4`); block `b = wi/4`, word-in-block `w = wi%4`.
//! The 8 outputs land at `out[r*k + b*32 + w*8 + n]` for `n` in 0..7, and the
//! block scale is `scales[blk_off + r*bpr + b]`.
//!
//! Signed-nibble reconstruction without a bit_cast/arithmetic-shift
//! intrinsic: extract the 4-bit field `nib = (word >> (n*4)) & 0xF`,
//! then `select(nib >= 8, nib - 16, nib)` (the `gguf_dequant_q8_0`
//! sign trick at 4-bit width). The constexpr-bounded `range(0,8,1)` loop
//! unrolls at codegen, so the emitted MSL/PTX is the hand-unrolled 8-store
//! form — same arithmetic as the per-value kernel, just one shared word/
//! scale load. Bit-identical to the old kernel's outputs.

use metaltile::kernel;

// Scales are f16 (the resident projection/expert weights store amax/7 as f16
// via the loader's `tb_f16`). The kernel widens to f32 for the multiply.
#[kernel]
pub fn ffai_dequant_q4<T>(
    qs: Tensor<u32>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] k_in: u32,
    // Number of packed u32 words = n_values / 8 (the dispatch grid is over words,
    // not values). Bounds the 1-thread-per-word launch.
    #[constexpr] n_words: u32,
    // Block offset into qs/scales (in 32-value Q4 blocks). Lets a caller dequant
    // a contiguous sub-slab of a larger pool (e.g. one MoE expert's rows) without
    // a separate tensor view: qs offset = blk_off*4 words, scales = blk_off.
    #[constexpr] blk_off: u32,
) {
    // Grid3D 1-D launch: program_id::<0>() = global linear thread id (gid_x).
    // One thread per u32 word; each emits the 8 values that word decodes to.
    let i = program_id::<0>();
    if i < n_words {
        let bpr = k_in / 32u32;
        let words_per_row = bpr * 4u32; // 4 packed words per 32-value block
        let r = i / words_per_row;
        let wi = i - r * words_per_row; // within-row word index 0..words_per_row-1
        let b = wi / 4u32; // block within row
        let w = wi - b * 4u32; // word within block 0..3
        let blk = blk_off + r * bpr + b;
        // One word load + one scale load, amortised over the 8 outputs below.
        let word = load(qs[blk * 4u32 + w]);
        let d = load(scales[blk]).cast::<f32>();
        // Output base for this word's 8 contiguous values.
        let out_base = r * k_in + b * 32u32 + w * 8u32;
        for n in range(0u32, 8u32, 1u32) {
            let nib = (word >> (n * 4u32)) & 0xfu32;
            // Sign-extend 4-bit: values 8..15 represent -8..-1 (nib - 16).
            let q_signed = select(nib >= 8u32, nib - 16u32, nib);
            let q = q_signed.cast::<i32>().cast::<f32>();
            store(out[out_base + n], q * d);
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dequant_q4;
    use crate::utils::pack_f32;

    /// Reference Q4 quantizer — mirrors `ffai_ops::quantize_q4` exactly:
    /// signed 4-bit, per-32-block scale = amax/7, 4 u32 words/block.
    fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
        let bpr = k / 32;
        let mut qs = vec![0u32; m * bpr * 4];
        let mut scales = vec![0f32; m * bpr];
        for r in 0..m {
            for b in 0..bpr {
                let base = r * k + b * 32;
                let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
                let d = amax / 7.0;
                scales[r * bpr + b] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for word in 0..4 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                        packed |= ((q as u32) & 0xf) << (i * 4);
                    }
                    qs[r * bpr * 4 + b * 4 + word] = packed;
                }
            }
        }
        (qs, scales)
    }

    fn cpu_dequant(qs: &[u32], scales: &[f32], m: usize, k: usize) -> Vec<f32> {
        let bpr = k / 32;
        let mut out = vec![0f32; m * k];
        for r in 0..m {
            for b in 0..bpr {
                let d = scales[r * bpr + b];
                for word in 0..4 {
                    let packed = qs[r * bpr * 4 + b * 4 + word];
                    for i in 0..8 {
                        let nib = (packed >> (i * 4)) & 0xf;
                        let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                        out[r * k + b * 32 + word * 8 + i] = q as f32 * d;
                    }
                }
            }
        }
        out
    }

    fn setup(m: usize, k: usize, dt: DType) -> TestSetup {
        let n = m * k;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 - 0.3).sin() * 2.1).collect();
        let (qs, scales) = quantize_q4(&values, m, k);
        // Scales are stored f16 (matches the resident loader). Round the oracle
        // through f16 so expected == GPU.
        let scales_f16: Vec<f32> = scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
        let dequantized = cpu_dequant(&qs, &scales_f16, m, k);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
        TestSetup::new(ffai_dequant_q4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F16), DType::F16))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("k_in", k as u32)
            .constexpr("n_words", (n / 8) as u32)
            .constexpr("blk_off", 0u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            // Grid is now over u32 words (1 thread/word → 8 outputs), not values.
            .grid_3d(((n / 8) as u32).div_ceil(256), 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_ffai_dequant_q4_single_row(dt: DType) -> TestSetup { setup(1, 64, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_ffai_dequant_q4_slab(dt: DType) -> TestSetup { setup(48, 128, dt) }
}

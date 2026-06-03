//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **dequantizing GEMM** (multi-row matmul) kernels — the qmm
//! counterpart of the GEMVs in [`super::block_scaled_matmul`]:
//! `output[m, n] = Σ_k dequant(weight[n, k]) · x[m, k]` for the spec-conformant
//! formats (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! Each `(m, n)` output element is one threadgroup that reduces over K — the
//! same proven Reduction geometry as the GEMVs, just flattened into a 1-D grid
//! of `out_dim · m_rows` threadgroups so it depends only on `program_id::<0>()`
//! (no 2-D grid assumptions). `tg → (mr = tg / out_dim, n = tg − mr·out_dim)`,
//! and `output[tg]` is exactly `output[mr·out_dim + n]`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim·m_rows, 1, 1]`, `tpg = [TPG, 1, 1]`
//!   with TPG ≥ 32 & a multiple of 32. One TG per output element.
//! - Weight/scale layouts + the `block_size | 8` packing rule are identical to
//!   the GEMVs (see [`super::block_scaled_matmul`]). `x` is `[m_rows, in_dim]`,
//!   `output` is `[m_rows, out_dim]`, both row-major.

use metaltile::kernel;

/// mxfp4 dequantizing GEMM — E2M1 weights (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = n * n_packs_per_row;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let sbits = load(scales[row_block_off + blk]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// nvfp4 dequantizing GEMM — E2M1 weights (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = n * n_packs_per_row;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = e4m3_decode(load(scales[row_block_off + blk]).cast::<u32>()) * global;
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) dequantizing GEMM — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// mxfp8 (E5M2) dequantizing GEMM — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// nvfp8 dequantizing GEMM — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 GEMMs ──────────────────
// These share the block-scaled framework but store a raw per-group FP32 scale
// (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3 +
// f32 scale), so it reuses `mt_nvfp8_qmm` — only fp4 (4-bit E2M1), fp8_e5m2
// (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 dequantizing GEMM — E2M1 weights (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp4_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = n * n_packs_per_row;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = load(scales[row_block_off + blk]);
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// Legacy fp8 (E5M2) dequantizing GEMM — 8-bit weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// Symmetric int8 dequantizing GEMM — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

// ── Symmetric sub-byte integers (int2/3/4/5/6 + MXINT2..6) ──────────────────
// qmm counterpart of the sub-byte dequant kernels in `block_scaled_dequant.rs`.
// Each `(m, n)` output element is one threadgroup reducing over K, exactly like
// the 8-bit `mt_int8_qmm` above — the only change is the element decode: instead
// of reading W as one byte per element, the N-bit signed code is extracted from
// W's tight LSB-first u32 bit-stream (straddle-aware two-word read, mirroring
// `ffai/dequant_gemv.rs`), then sign-extended in float (subtract 2^N when the
// top bit is set; `$half`/`$full` are 2^(N-1) / 2^N). A 4-bit stream is byte-
// identical to the nibble layout, so int4 rides the same path. `$half`/`$full`
// are passed as literals to keep the constexpr math out of the DSL shift operands.

/// FP32-scaled symmetric int qmm (int2/3/4/5/6): bit-stream code × per-group FP32.
macro_rules! int_qmm_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<f32>,
            x: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let tg = program_id::<0>();
            let mr = tg / out_dim;
            let n = tg - mr * out_dim;
            let row_word_off = n * (in_dim * $bits / 32u32);
            let row_block_off = n * (in_dim / block_size);
            let x_row_off = mr * in_dim;

            let mut acc = 0.0f32;
            let iters = (in_dim + lsize - 1u32) / lsize;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let bit_off = c * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let elem = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[row_block_off + c / block_size]);
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[tg], total.cast::<T>());
            }
        }
    };
}
int_qmm_f32!(mt_int2_qmm, 2u32, 2u32, 4.0f32);
int_qmm_f32!(mt_int3_qmm, 3u32, 4u32, 8.0f32);
int_qmm_f32!(mt_int4_qmm, 4u32, 8u32, 16.0f32);
int_qmm_f32!(mt_int5_qmm, 5u32, 16u32, 32.0f32);
int_qmm_f32!(mt_int6_qmm, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int qmm (MXINT2/3/4/5/6): bit-stream code × pow-2 block scale.
macro_rules! int_qmm_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<u8>,
            x: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let tg = program_id::<0>();
            let mr = tg / out_dim;
            let n = tg - mr * out_dim;
            let row_word_off = n * (in_dim * $bits / 32u32);
            let row_block_off = n * (in_dim / block_size);
            let x_row_off = mr * in_dim;

            let mut acc = 0.0f32;
            let iters = (in_dim + lsize - 1u32) / lsize;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let bit_off = c * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let elem = select(q >= $half, qf - $full, qf); // sign-extend
                    let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
                    let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[tg], total.cast::<T>());
            }
        }
    };
}
int_qmm_e8m0!(mt_mxint2_qmm, 2u32, 2u32, 4.0f32);
int_qmm_e8m0!(mt_mxint3_qmm, 3u32, 4u32, 8.0f32);
int_qmm_e8m0!(mt_mxint4_qmm, 4u32, 8u32, 16.0f32);
int_qmm_e8m0!(mt_mxint5_qmm, 5u32, 16u32, 32.0f32);
int_qmm_e8m0!(mt_mxint6_qmm, 6u32, 32u32, 64.0f32);

/// MXINT8 dequantizing GEMM — 8-bit codes (byte layout, block 32), E8M0 scale.
/// Same shape as `mt_int8_qmm`; only the scale is E8M0 (`2^(bits-127)`).
#[kernel]
pub fn mt_mxint8_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

// ── FP16-scale twins ─────────────────────────────────────────────────────────
// Near-clones of the FP32-scaled GEMMs above; the ONLY change is the scale tensor
// (`Tensor<f16>` instead of `Tensor<f32>`) and the scale read (`.cast::<f32>()` on
// the half load). Element decode (E2M1 / E4M3 / E5M2 / int bit-stream + sign-
// extend), weight indexing, dispatch geometry, staging, and reduction are all
// IDENTICAL to the FP32 twin. The GPU half load matches the host
// `f16_scale_decode`, so the oracle still holds exactly.

/// nvfp8 (FP16 scale) dequantizing GEMM — E4M3 weights (block 16), per-block FP16
/// scale. Twin of `mt_nvfp8_qmm`; also serves `fp8_e4m3_f16` (same 8-bit-E4M3 +
/// scale shape, exactly as `Fp8E4m3` reuses `mt_nvfp8_qmm`).
#[kernel]
pub fn mt_nvfp8_f16_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// fp4 (FP16 scale) dequantizing GEMM — E2M1 weights (group 32), per-group FP16
/// scale. Twin of `mt_fp4_qmm`.
#[kernel]
pub fn mt_fp4_f16_qmm<T>(
    weight: Tensor<u32>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let n_packs_per_row = in_dim / 8u32;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = n * n_packs_per_row;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let blk = pack_idx / packs_per_block;
            let scale = load(scales[row_block_off + blk]).cast::<f32>();
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xFu32;
                let val = e2m1_decode(nib);
                acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// fp8 (E5M2, FP16 scale) dequantizing GEMM — 8-bit weights (group 32), per-group
/// FP16 scale. Twin of `mt_fp8_e5m2_qmm`.
#[kernel]
pub fn mt_fp8_e5m2_f16_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

/// FP16-scaled symmetric int qmm (int2/3/4/5/6): bit-stream code × per-group FP16.
/// Twin of `int_qmm_f32!`; only the scale tensor + read change.
macro_rules! int_qmm_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<f16>,
            x: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] out_dim: u32,
            #[constexpr] block_size: u32,
        ) {
            let tg = program_id::<0>();
            let mr = tg / out_dim;
            let n = tg - mr * out_dim;
            let row_word_off = n * (in_dim * $bits / 32u32);
            let row_block_off = n * (in_dim / block_size);
            let x_row_off = mr * in_dim;

            let mut acc = 0.0f32;
            let iters = (in_dim + lsize - 1u32) / lsize;
            for it in range(0u32, iters, 1u32) {
                let c = it * lsize + tid;
                if c < in_dim {
                    let bit_off = c * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;
                    let qf = q.cast::<f32>();
                    let elem = select(q >= $half, qf - $full, qf); // sign-extend
                    let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
                    acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
                }
            }
            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[tg], total.cast::<T>());
            }
        }
    };
}
int_qmm_f16!(mt_int2_f16_qmm, 2u32, 2u32, 4.0f32);
int_qmm_f16!(mt_int3_f16_qmm, 3u32, 4u32, 8.0f32);
int_qmm_f16!(mt_int4_f16_qmm, 4u32, 8u32, 16.0f32);
int_qmm_f16!(mt_int5_f16_qmm, 5u32, 16u32, 32.0f32);
int_qmm_f16!(mt_int6_f16_qmm, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) dequantizing GEMM — 8-bit codes (byte layout, group 64),
/// per-group FP16 scale. Twin of `mt_int8_qmm`.
#[kernel]
pub fn mt_int8_f16_qmm<T>(
    weight: Tensor<u8>,
    scales: Tensor<f16>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let row_off = n * in_dim;
    let row_block_off = n * (in_dim / block_size);
    let x_row_off = mr * in_dim;

    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]).cast::<f32>();
            acc = acc + (elem * scale) * load(x[x_row_off + c]).cast::<f32>();
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Reduction-contract threadgroup width (≥ 32, multiple of 32).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` quantized weights (mixed signs).
    fn weights(out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + r * 0.25) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`.
    fn qmm_oracle(
        wdq: &[f32],
        x: &[f32],
        m_rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m_rows * out_dim];
        for mr in 0..m_rows {
            for n in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += wdq[n * in_dim + k] * x[mr * in_dim + k];
                }
                out[mr * out_dim + n] = acc;
            }
        }
        out
    }

    fn qmm_setup(
        kernel: Kernel,
        fmt: QFormat,
        m_rows: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(out_dim, in_dim);
        let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = qmm_oracle(&wdq, &x, m_rows, in_dim, out_dim);
        // 8-bit codes bind as one uchar each; everything sub-byte (E2M1 nibbles
        // + int2-6 bit-streams) binds as `DType::U32`. FP32 scales bind as f32,
        // FP16 scales as f16; E8M0/E4M3 scales as one byte.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("output", m_rows * out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            (out_dim * m_rows) as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // m_rows 3, out_dim 4, in_dim 256 (divisible by both block sizes).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp4_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp4_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp8_e4m3_qmm::kernel_ir_for(dt), QFormat::Mxfp8E4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Mxfp8E5, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 3, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // in_dim 256 is a multiple of int8's block_size (64), so all formats fit.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp4_qmm::kernel_ir_for(dt), QFormat::Fp4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int8_qmm::kernel_ir_for(dt), QFormat::Int8, 3, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0, block 32). The kernel and oracle
    // share the codec, so the GPU output matches the oracle to float precision
    // regardless of how coarse the quantization is. in_dim 256 is a multiple of
    // 32, so `in_dim * bits` is u32-aligned for every sub-byte width and is also
    // divisible by both block sizes (int 64, mxint 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int2_qmm::kernel_ir_for(dt), QFormat::Int2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int3_qmm::kernel_ir_for(dt), QFormat::Int3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int4_qmm::kernel_ir_for(dt), QFormat::Int4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int5_qmm::kernel_ir_for(dt), QFormat::Int5, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int6_qmm::kernel_ir_for(dt), QFormat::Int6, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint2_qmm::kernel_ir_for(dt), QFormat::Mxint2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint3_qmm::kernel_ir_for(dt), QFormat::Mxint3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint4_qmm::kernel_ir_for(dt), QFormat::Mxint4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint5_qmm::kernel_ir_for(dt), QFormat::Mxint5, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint6_qmm::kernel_ir_for(dt), QFormat::Mxint6, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint8_qmm::kernel_ir_for(dt), QFormat::Mxint8, 3, 4, 256, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Nvfp8F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp4_f16_qmm::kernel_ir_for(dt), QFormat::Fp4F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp8_e5m2_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int2_f16_qmm::kernel_ir_for(dt), QFormat::Int2F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int3_f16_qmm::kernel_ir_for(dt), QFormat::Int3F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int4_f16_qmm::kernel_ir_for(dt), QFormat::Int4F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int5_f16_qmm::kernel_ir_for(dt), QFormat::Int5F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int6_f16_qmm::kernel_ir_for(dt), QFormat::Int6F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int8_f16_qmm::kernel_ir_for(dt), QFormat::Int8F16, 3, 4, 256, dt)
    }
}

/// Batched-decode (m=32) GEMM benches at N=K=4096 — the compute-throughput
/// precision ranking. Random packed buffers (throughput is data-independent).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn qmm_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = out_dim * (in_dim / fmt.block_size());
        // 8-bit codes are one uchar each; sub-byte codes (E2M1 nibbles + int2-6
        // bit-streams) tight-bit-pack into u32 words (with a guard word for
        // straddling 3/5/6-bit reads).
        let n_elems = out_dim * in_dim;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_elems, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n_elems, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m * in_dim * sz
            + m * out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::zeros("output", m * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_dim * m) as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m as u64 * out_dim as u64 * in_dim as u64) // GEMM: 2·M·N·K
            .with_shape_label(format!("{} m={m} n={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(name = "ffai/block_scaled_qmm/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp4_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp4_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp8_e4m3_qmm::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp4_qmm::kernel_ir_for(dt), QFormat::Fp4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int8_qmm::kernel_ir_for(dt), QFormat::Int8, 32, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(name = "ffai/block_scaled_qmm/int2", dtypes = [f32, f16, bf16])]
    fn bench_int2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int2_qmm::kernel_ir_for(dt), QFormat::Int2, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int3", dtypes = [f32, f16, bf16])]
    fn bench_int3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int3_qmm::kernel_ir_for(dt), QFormat::Int3, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int4", dtypes = [f32, f16, bf16])]
    fn bench_int4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int4_qmm::kernel_ir_for(dt), QFormat::Int4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int5", dtypes = [f32, f16, bf16])]
    fn bench_int5_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int5_qmm::kernel_ir_for(dt), QFormat::Int5, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int6", dtypes = [f32, f16, bf16])]
    fn bench_int6_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int6_qmm::kernel_ir_for(dt), QFormat::Int6, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxint2", dtypes = [f32, f16, bf16])]
    fn bench_mxint2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint2_qmm::kernel_ir_for(dt), QFormat::Mxint2, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxint3", dtypes = [f32, f16, bf16])]
    fn bench_mxint3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint3_qmm::kernel_ir_for(dt), QFormat::Mxint3, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxint4", dtypes = [f32, f16, bf16])]
    fn bench_mxint4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint4_qmm::kernel_ir_for(dt), QFormat::Mxint4, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxint5", dtypes = [f32, f16, bf16])]
    fn bench_mxint5_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint5_qmm::kernel_ir_for(dt), QFormat::Mxint5, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxint6", dtypes = [f32, f16, bf16])]
    fn bench_mxint6_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint6_qmm::kernel_ir_for(dt), QFormat::Mxint6, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/mxint8", dtypes = [f32, f16, bf16])]
    fn bench_mxint8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint8_qmm::kernel_ir_for(dt), QFormat::Mxint8, 32, 4096, 4096, dt)
    }
    // FP16-scale twins. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel.
    #[bench(name = "ffai/block_scaled_qmm/nvfp8_f16", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Nvfp8F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp8_e4m3_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp4_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp4_f16_qmm::kernel_ir_for(dt), QFormat::Fp4F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/fp8_e5m2_f16", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp8_e5m2_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int2_f16", dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int2_f16_qmm::kernel_ir_for(dt), QFormat::Int2F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int3_f16", dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int3_f16_qmm::kernel_ir_for(dt), QFormat::Int3F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int4_f16", dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int4_f16_qmm::kernel_ir_for(dt), QFormat::Int4F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int5_f16", dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int5_f16_qmm::kernel_ir_for(dt), QFormat::Int5F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int6_f16", dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int6_f16_qmm::kernel_ir_for(dt), QFormat::Int6F16, 32, 4096, 4096, dt)
    }
    #[bench(name = "ffai/block_scaled_qmm/int8_f16", dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int8_f16_qmm::kernel_ir_for(dt), QFormat::Int8F16, 32, 4096, 4096, dt)
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Flash **block-scaled** SDPA — single-pass online-softmax attention over a
//! block-scaled-quantized K/V cache, for the spec-conformant formats
//! (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8). The block-scaled counterpart of
//! the affine `ffai/flash_quantized_sdpa.rs`: K and V are dequantized inline
//! per thread via `element_decode(code) · block_scale` (no bias) instead of the
//! affine `q·scale + bias`.
//!
//! Geometry is identical to the affine kernel (one simdgroup per query):
//!   - `program_id::<0>()` = lane ∈ [0,32), owns dim slots `lane + i·32`.
//!   - `program_id::<1>()` = query index; `kv_idx = q_idx / repeat_count`.
//!   - Grid `[1, B·nQ, 1]`, tpg `[32, 1, 1]`, Mode Grid3D.
//!
//! K/V cache layout (`N = tokens`, per `(kv_head, token)` row of `dim`):
//!   - 4-bit (mxfp4/nvfp4): `k_packed [B·nKV, N, dim/8] u32` (8 E2M1 nibbles
//!     per word), `k_scales [B·nKV, N, dim/block_size]` u8 (+ global f32 nvfp4).
//!   - 8-bit (mxfp8/nvfp8/int8): `k_packed [B·nKV, N, dim] u8`, scales u8 (E8M0)
//!     or f32 (nvfp8/int8). V mirrors K. A `dim` that isn't a multiple of
//!     `block_size` (int8 group 64 over d96) tiles with a ragged trailing block:
//!     `n_blocks = ceil(dim/block_size)`, matching the host packer.
//!
//! Each format is a whole-fn `macro_rules!` parameterized by `$dpl` (= head_dim/32,
//! the per-lane dim count). Every production head dim is generated:
//! d64 (`$dpl=2`), d96 (3), d128 (4), d256 (8), d512 (16). The dispatch geometry
//! is identical across dims (only the stack size + loop bound change), so there
//! is no new freeze surface. Codegen-only; correctness pinned by `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 flash SDPA — E2M1 K/V (block 32), E8M0 pow-2 scale.
/// `$dpl` = head_dim/32 (the per-lane dim count, a compile-time stack/loop bound).
macro_rules! mxfp4_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<u8>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<u8>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;
            let words_per_token = dim / 8u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(k_packed[k_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let ksc = exp2(
                                load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            dot_partial =
                                dot_partial + stack_load("q_vals", i) * (mt_decode_e2m1(nib) * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(v_packed[v_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let vsc = exp2(
                                load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            let prev = stack_load("o", i);
                            stack_store(
                                "o",
                                i,
                                prev * exp_diff + exp_score * (mt_decode_e2m1(nib) * vsc),
                            );
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
mxfp4_flash!(mt_mxfp4_flash_sdpa_d64, 2u32);
mxfp4_flash!(mt_mxfp4_flash_sdpa_d96, 3u32);
mxfp4_flash!(mt_mxfp4_flash_sdpa_d128, 4u32);
mxfp4_flash!(mt_mxfp4_flash_sdpa_d256, 8u32);
mxfp4_flash!(mt_mxfp4_flash_sdpa_d512, 16u32);

/// nvfp4 flash SDPA — E2M1 K/V (block 16), E4M3 micro-scale × global.
macro_rules! nvfp4_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<u8>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<u8>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
            #[constexpr] global: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;
            let words_per_token = dim / 8u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(k_packed[k_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let ksc = mt_decode_e4m3(
                                load(k_scales[k_blk_row + d / block_size]).cast::<u32>(),
                            ) * global;
                            dot_partial =
                                dot_partial + stack_load("q_vals", i) * (mt_decode_e2m1(nib) * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(v_packed[v_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let vsc = mt_decode_e4m3(
                                load(v_scales[v_blk_row + d / block_size]).cast::<u32>(),
                            ) * global;
                            let prev = stack_load("o", i);
                            stack_store(
                                "o",
                                i,
                                prev * exp_diff + exp_score * (mt_decode_e2m1(nib) * vsc),
                            );
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
nvfp4_flash!(mt_nvfp4_flash_sdpa_d64, 2u32);
nvfp4_flash!(mt_nvfp4_flash_sdpa_d96, 3u32);
nvfp4_flash!(mt_nvfp4_flash_sdpa_d128, 4u32);
nvfp4_flash!(mt_nvfp4_flash_sdpa_d256, 8u32);
nvfp4_flash!(mt_nvfp4_flash_sdpa_d512, 16u32);

/// mxfp8 (E4M3) flash SDPA — 8-bit K/V (block 32), E8M0 pow-2 scale.
macro_rules! mxfp8_e4m3_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<u8>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<u8>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_e4m3(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = exp2(
                                load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_e4m3(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = exp2(
                                load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
mxfp8_e4m3_flash!(mt_mxfp8_e4m3_flash_sdpa_d64, 2u32);
mxfp8_e4m3_flash!(mt_mxfp8_e4m3_flash_sdpa_d96, 3u32);
mxfp8_e4m3_flash!(mt_mxfp8_e4m3_flash_sdpa_d128, 4u32);
mxfp8_e4m3_flash!(mt_mxfp8_e4m3_flash_sdpa_d256, 8u32);
mxfp8_e4m3_flash!(mt_mxfp8_e4m3_flash_sdpa_d512, 16u32);

/// mxfp8 (E5M2) flash SDPA — 8-bit K/V (block 32), E8M0 pow-2 scale.
macro_rules! mxfp8_e5m2_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<u8>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<u8>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_e5m2(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = exp2(
                                load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_e5m2(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = exp2(
                                load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
mxfp8_e5m2_flash!(mt_mxfp8_e5m2_flash_sdpa_d64, 2u32);
mxfp8_e5m2_flash!(mt_mxfp8_e5m2_flash_sdpa_d96, 3u32);
mxfp8_e5m2_flash!(mt_mxfp8_e5m2_flash_sdpa_d128, 4u32);
mxfp8_e5m2_flash!(mt_mxfp8_e5m2_flash_sdpa_d256, 8u32);
mxfp8_e5m2_flash!(mt_mxfp8_e5m2_flash_sdpa_d512, 16u32);

/// nvfp8 flash SDPA — E4M3 K/V (block 16), per-block FP32 scale.
macro_rules! nvfp8_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<f32>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_e4m3(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = load(k_scales[k_blk_row + d / block_size]);
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_e4m3(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = load(v_scales[v_blk_row + d / block_size]);
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
nvfp8_flash!(mt_nvfp8_flash_sdpa_d64, 2u32);
nvfp8_flash!(mt_nvfp8_flash_sdpa_d96, 3u32);
nvfp8_flash!(mt_nvfp8_flash_sdpa_d128, 4u32);
nvfp8_flash!(mt_nvfp8_flash_sdpa_d256, 8u32);
nvfp8_flash!(mt_nvfp8_flash_sdpa_d512, 16u32);

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 flash SDPA ─────────────
// These share the block-scaled attention body but store a raw per-group FP32
// scale (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit
// E4M3 + f32 scale), so it reuses `mt_nvfp8_flash_sdpa_d128`; only fp4 (4-bit
// E2M1), fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own
// decode here.

/// Legacy fp4 flash SDPA — E2M1 K/V (group 32), per-group FP32 scale.
macro_rules! fp4_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<f32>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;
            let words_per_token = dim / 8u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(k_packed[k_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let ksc = load(k_scales[k_blk_row + d / block_size]);
                            dot_partial =
                                dot_partial + stack_load("q_vals", i) * (mt_decode_e2m1(nib) * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(v_packed[v_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let vsc = load(v_scales[v_blk_row + d / block_size]);
                            let prev = stack_load("o", i);
                            stack_store(
                                "o",
                                i,
                                prev * exp_diff + exp_score * (mt_decode_e2m1(nib) * vsc),
                            );
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
fp4_flash!(mt_fp4_flash_sdpa_d64, 2u32);
fp4_flash!(mt_fp4_flash_sdpa_d96, 3u32);
fp4_flash!(mt_fp4_flash_sdpa_d128, 4u32);
fp4_flash!(mt_fp4_flash_sdpa_d256, 8u32);
fp4_flash!(mt_fp4_flash_sdpa_d512, 16u32);

/// Legacy fp8 (E5M2) flash SDPA — 8-bit K/V (group 32), FP32 scale.
macro_rules! fp8_e5m2_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<f32>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_e5m2(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = load(k_scales[k_blk_row + d / block_size]);
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_e5m2(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = load(v_scales[v_blk_row + d / block_size]);
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
fp8_e5m2_flash!(mt_fp8_e5m2_flash_sdpa_d64, 2u32);
fp8_e5m2_flash!(mt_fp8_e5m2_flash_sdpa_d96, 3u32);
fp8_e5m2_flash!(mt_fp8_e5m2_flash_sdpa_d128, 4u32);
fp8_e5m2_flash!(mt_fp8_e5m2_flash_sdpa_d256, 8u32);
fp8_e5m2_flash!(mt_fp8_e5m2_flash_sdpa_d512, 16u32);

/// Symmetric int8 flash SDPA — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
macro_rules! int8_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<f32>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            // Round up so a head dim that isn't a multiple of the group (int8 group
            // 64 over d96 → a 64-block + a 32-block) counts the ragged tail block.
            // `dim / block_size` floors; matches the host packer's `div_ceil`.
            let n_blocks = (dim + block_size - 1u32) / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_int8(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = load(k_scales[k_blk_row + d / block_size]);
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_int8(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = load(v_scales[v_blk_row + d / block_size]);
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
int8_flash!(mt_int8_flash_sdpa_d64, 2u32);
// d96 (GPT-NeoX): int8 group 64 doesn't divide 96, so the cache tiles as a
// 64-block + a ragged 32-block (host `pack`/kernel `n_blocks` both round up).
int8_flash!(mt_int8_flash_sdpa_d96, 3u32);
int8_flash!(mt_int8_flash_sdpa_d128, 4u32);
int8_flash!(mt_int8_flash_sdpa_d256, 8u32);
int8_flash!(mt_int8_flash_sdpa_d512, 16u32);

// ── Symmetric sub-byte integers (int2/3/4/5/6 + MXINT2..6) + MXINT8 ─────────
// Cache elements are signed N-bit two's-complement codes, tight-bit-packed
// LSB-first into u32 words. The host packs the whole `[B·nKV·N, dim]` cache as
// one contiguous bit-stream (`pack` flat-indexes `(kv·N + t)·dim + d`), so a
// token row's word base is `row_word = (kv_idx·tokens + t) · (dim·BITS / 32)`.
// Every supported head dim (64/96/128/256/512) is a multiple of 32, so each
// token row starts word-aligned and `dim·BITS` is a whole number of words — no
// ragged-tail straddle (the int8 d96 ragged-block case does NOT apply here).
// Within a row, element `d` lives at `bit_off = d·BITS`; decode = straddle-aware
// two-word read + float sign-extend (mirrors `mlx/block_scaled_dequant.rs`),
// applied identically to the K score loop and the V accumulation loop — exactly
// as `int8_flash` decodes both. `$half`/`$full` are 2^(N-1) / 2^N passed as
// literals to keep the constexpr math out of the DSL shift operands.

/// FP32-scaled symmetric int flash SDPA (int2/3/4/5/6): bit-stream K/V codes
/// (group 64) × per-group FP32 scale. `$dpl` = head_dim/32 (per-lane dim count,
/// a compile-time stack/loop bound); `$bits`/`$half`/`$full` select the width.
macro_rules! int_flash_f32 {
    ($name:ident, $dpl:literal, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<f32>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            // Round up so a head dim that isn't a multiple of the group (int
            // group 64 over d96 → a 64-block + a 32-block) counts the ragged
            // tail block; matches the host packer's `div_ceil`.
            let n_blocks = (dim + block_size - 1u32) / block_size;
            // Tight bit-stream words per token row (dim is a multiple of 32, so
            // `dim·BITS` is an exact word count — each token row word-aligned).
            let words_per_token = dim * $bits / 32u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            // Straddle-aware two-word read of element `d`'s code.
                            let bit_off = d * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(k_packed[k_word_row + word_idx]);
                            let w1 = load(
                                k_packed
                                    [k_word_row + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let kelem = select(q >= $half, qf - $full, qf); // sign-extend
                            let ksc = load(k_scales[k_blk_row + d / block_size]);
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_off = d * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(v_packed[v_word_row + word_idx]);
                            let w1 = load(
                                v_packed
                                    [v_word_row + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let velem = select(q >= $half, qf - $full, qf); // sign-extend
                            let vsc = load(v_scales[v_blk_row + d / block_size]);
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
int_flash_f32!(mt_int2_flash_sdpa_d64, 2u32, 2u32, 2u32, 4.0f32);
int_flash_f32!(mt_int2_flash_sdpa_d96, 3u32, 2u32, 2u32, 4.0f32);
int_flash_f32!(mt_int2_flash_sdpa_d128, 4u32, 2u32, 2u32, 4.0f32);
int_flash_f32!(mt_int2_flash_sdpa_d256, 8u32, 2u32, 2u32, 4.0f32);
int_flash_f32!(mt_int2_flash_sdpa_d512, 16u32, 2u32, 2u32, 4.0f32);
int_flash_f32!(mt_int3_flash_sdpa_d64, 2u32, 3u32, 4u32, 8.0f32);
int_flash_f32!(mt_int3_flash_sdpa_d96, 3u32, 3u32, 4u32, 8.0f32);
int_flash_f32!(mt_int3_flash_sdpa_d128, 4u32, 3u32, 4u32, 8.0f32);
int_flash_f32!(mt_int3_flash_sdpa_d256, 8u32, 3u32, 4u32, 8.0f32);
int_flash_f32!(mt_int3_flash_sdpa_d512, 16u32, 3u32, 4u32, 8.0f32);
int_flash_f32!(mt_int4_flash_sdpa_d64, 2u32, 4u32, 8u32, 16.0f32);
int_flash_f32!(mt_int4_flash_sdpa_d96, 3u32, 4u32, 8u32, 16.0f32);
int_flash_f32!(mt_int4_flash_sdpa_d128, 4u32, 4u32, 8u32, 16.0f32);
int_flash_f32!(mt_int4_flash_sdpa_d256, 8u32, 4u32, 8u32, 16.0f32);
int_flash_f32!(mt_int4_flash_sdpa_d512, 16u32, 4u32, 8u32, 16.0f32);
int_flash_f32!(mt_int5_flash_sdpa_d64, 2u32, 5u32, 16u32, 32.0f32);
int_flash_f32!(mt_int5_flash_sdpa_d96, 3u32, 5u32, 16u32, 32.0f32);
int_flash_f32!(mt_int5_flash_sdpa_d128, 4u32, 5u32, 16u32, 32.0f32);
int_flash_f32!(mt_int5_flash_sdpa_d256, 8u32, 5u32, 16u32, 32.0f32);
int_flash_f32!(mt_int5_flash_sdpa_d512, 16u32, 5u32, 16u32, 32.0f32);
int_flash_f32!(mt_int6_flash_sdpa_d64, 2u32, 6u32, 32u32, 64.0f32);
int_flash_f32!(mt_int6_flash_sdpa_d96, 3u32, 6u32, 32u32, 64.0f32);
int_flash_f32!(mt_int6_flash_sdpa_d128, 4u32, 6u32, 32u32, 64.0f32);
int_flash_f32!(mt_int6_flash_sdpa_d256, 8u32, 6u32, 32u32, 64.0f32);
int_flash_f32!(mt_int6_flash_sdpa_d512, 16u32, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int flash SDPA (MXINT2/3/4/5/6): bit-stream K/V codes
/// (block 32) × pow-2 (E8M0) block scale `2^(bits-127)`. Same straddle-aware
/// decode as `int_flash_f32`; only the scale axis differs (one u8 exponent per
/// block instead of a raw f32).
macro_rules! int_flash_e8m0 {
    ($name:ident, $dpl:literal, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<u8>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<u8>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            // mxint block 32 divides every supported head dim, but round up to
            // mirror the host packer's `div_ceil` (no ragged tail occurs here).
            let n_blocks = (dim + block_size - 1u32) / block_size;
            let words_per_token = dim * $bits / 32u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_off = d * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(k_packed[k_word_row + word_idx]);
                            let w1 = load(
                                k_packed
                                    [k_word_row + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let kelem = select(q >= $half, qf - $full, qf); // sign-extend
                            let ksc = exp2(
                                load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_off = d * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(v_packed[v_word_row + word_idx]);
                            let w1 = load(
                                v_packed
                                    [v_word_row + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let velem = select(q >= $half, qf - $full, qf); // sign-extend
                            let vsc = exp2(
                                load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
int_flash_e8m0!(mt_mxint2_flash_sdpa_d64, 2u32, 2u32, 2u32, 4.0f32);
int_flash_e8m0!(mt_mxint2_flash_sdpa_d96, 3u32, 2u32, 2u32, 4.0f32);
int_flash_e8m0!(mt_mxint2_flash_sdpa_d128, 4u32, 2u32, 2u32, 4.0f32);
int_flash_e8m0!(mt_mxint2_flash_sdpa_d256, 8u32, 2u32, 2u32, 4.0f32);
int_flash_e8m0!(mt_mxint2_flash_sdpa_d512, 16u32, 2u32, 2u32, 4.0f32);
int_flash_e8m0!(mt_mxint3_flash_sdpa_d64, 2u32, 3u32, 4u32, 8.0f32);
int_flash_e8m0!(mt_mxint3_flash_sdpa_d96, 3u32, 3u32, 4u32, 8.0f32);
int_flash_e8m0!(mt_mxint3_flash_sdpa_d128, 4u32, 3u32, 4u32, 8.0f32);
int_flash_e8m0!(mt_mxint3_flash_sdpa_d256, 8u32, 3u32, 4u32, 8.0f32);
int_flash_e8m0!(mt_mxint3_flash_sdpa_d512, 16u32, 3u32, 4u32, 8.0f32);
int_flash_e8m0!(mt_mxint4_flash_sdpa_d64, 2u32, 4u32, 8u32, 16.0f32);
int_flash_e8m0!(mt_mxint4_flash_sdpa_d96, 3u32, 4u32, 8u32, 16.0f32);
int_flash_e8m0!(mt_mxint4_flash_sdpa_d128, 4u32, 4u32, 8u32, 16.0f32);
int_flash_e8m0!(mt_mxint4_flash_sdpa_d256, 8u32, 4u32, 8u32, 16.0f32);
int_flash_e8m0!(mt_mxint4_flash_sdpa_d512, 16u32, 4u32, 8u32, 16.0f32);
int_flash_e8m0!(mt_mxint5_flash_sdpa_d64, 2u32, 5u32, 16u32, 32.0f32);
int_flash_e8m0!(mt_mxint5_flash_sdpa_d96, 3u32, 5u32, 16u32, 32.0f32);
int_flash_e8m0!(mt_mxint5_flash_sdpa_d128, 4u32, 5u32, 16u32, 32.0f32);
int_flash_e8m0!(mt_mxint5_flash_sdpa_d256, 8u32, 5u32, 16u32, 32.0f32);
int_flash_e8m0!(mt_mxint5_flash_sdpa_d512, 16u32, 5u32, 16u32, 32.0f32);
int_flash_e8m0!(mt_mxint6_flash_sdpa_d64, 2u32, 6u32, 32u32, 64.0f32);
int_flash_e8m0!(mt_mxint6_flash_sdpa_d96, 3u32, 6u32, 32u32, 64.0f32);
int_flash_e8m0!(mt_mxint6_flash_sdpa_d128, 4u32, 6u32, 32u32, 64.0f32);
int_flash_e8m0!(mt_mxint6_flash_sdpa_d256, 8u32, 6u32, 32u32, 64.0f32);
int_flash_e8m0!(mt_mxint6_flash_sdpa_d512, 16u32, 6u32, 32u32, 64.0f32);

/// MXINT8 flash SDPA — 8-bit codes (byte layout, block 32), E8M0 pow-2 block
/// scale. Same shape as `int8_flash` (one code/byte, sign-extend → `code·scale`)
/// but the scale is an E8M0 exponent: `2^(bits-127)` instead of a raw f32.
macro_rules! mxint8_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<u8>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<u8>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            // mxint8 block 32 divides every supported head dim; round up to
            // mirror the host packer's `div_ceil` (no ragged tail occurs here).
            let n_blocks = (dim + block_size - 1u32) / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_int8(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = exp2(
                                load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_int8(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = exp2(
                                load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32,
                            );
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
mxint8_flash!(mt_mxint8_flash_sdpa_d64, 2u32);
mxint8_flash!(mt_mxint8_flash_sdpa_d96, 3u32);
mxint8_flash!(mt_mxint8_flash_sdpa_d128, 4u32);
mxint8_flash!(mt_mxint8_flash_sdpa_d256, 8u32);
mxint8_flash!(mt_mxint8_flash_sdpa_d512, 16u32);

// ── FP16-scale twins (nvfp8 / fp4 / fp8_e5m2 / int2..6 / int8) ──────────────
// These are byte-for-byte clones of the FP32-scaled kernels above, with ONE
// change: the per-block scale tensor is `Tensor<f16>` (half the scale bytes)
// and each scale read becomes `load(scales[...]).cast::<f32>()`. The element
// decode (E4M3 / E2M1 / E5M2 / int bit-stream + sign-extend), weight indexing,
// online-softmax, sinks/window, and dispatch geometry are IDENTICAL to the
// FP32 twin — only the scale precision differs. `fp8_e4m3_f16` reuses the
// `nvfp8_f16` kernel (same 8-bit-E4M3 + FP16-scale shape), exactly as
// `fp8_e4m3` reuses `nvfp8` today.

/// nvfp8 flash SDPA, FP16 scale — E4M3 K/V (block 16), per-block FP16 scale.
/// Clone of `nvfp8_flash` with the scale tensor narrowed to f16 (also serves
/// `Fp8E4m3F16`: same 8-bit-E4M3 + FP16-scale shape).
macro_rules! nvfp8_f16_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<f16>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<f16>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_e4m3(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = load(k_scales[k_blk_row + d / block_size]).cast::<f32>();
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_e4m3(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = load(v_scales[v_blk_row + d / block_size]).cast::<f32>();
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
nvfp8_f16_flash!(mt_nvfp8_f16_flash_sdpa_d64, 2u32);
nvfp8_f16_flash!(mt_nvfp8_f16_flash_sdpa_d96, 3u32);
nvfp8_f16_flash!(mt_nvfp8_f16_flash_sdpa_d128, 4u32);
nvfp8_f16_flash!(mt_nvfp8_f16_flash_sdpa_d256, 8u32);
nvfp8_f16_flash!(mt_nvfp8_f16_flash_sdpa_d512, 16u32);

/// fp4 flash SDPA, FP16 scale — E2M1 K/V (group 32), per-group FP16 scale.
/// Clone of `fp4_flash` with the scale tensor narrowed to f16.
macro_rules! fp4_f16_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<f16>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<f16>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;
            let words_per_token = dim / 8u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(k_packed[k_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let ksc = load(k_scales[k_blk_row + d / block_size]).cast::<f32>();
                            dot_partial =
                                dot_partial + stack_load("q_vals", i) * (mt_decode_e2m1(nib) * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let nib = (load(v_packed[v_word_row + d / 8u32])
                                >> ((d % 8u32) * 4u32))
                                & 0xFu32;
                            let vsc = load(v_scales[v_blk_row + d / block_size]).cast::<f32>();
                            let prev = stack_load("o", i);
                            stack_store(
                                "o",
                                i,
                                prev * exp_diff + exp_score * (mt_decode_e2m1(nib) * vsc),
                            );
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
fp4_f16_flash!(mt_fp4_f16_flash_sdpa_d64, 2u32);
fp4_f16_flash!(mt_fp4_f16_flash_sdpa_d96, 3u32);
fp4_f16_flash!(mt_fp4_f16_flash_sdpa_d128, 4u32);
fp4_f16_flash!(mt_fp4_f16_flash_sdpa_d256, 8u32);
fp4_f16_flash!(mt_fp4_f16_flash_sdpa_d512, 16u32);

/// fp8 (E5M2) flash SDPA, FP16 scale — 8-bit K/V (group 32), per-group FP16
/// scale. Clone of `fp8_e5m2_flash` with the scale tensor narrowed to f16.
macro_rules! fp8_e5m2_f16_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<f16>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<f16>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            let n_blocks = dim / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_e5m2(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = load(k_scales[k_blk_row + d / block_size]).cast::<f32>();
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_e5m2(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = load(v_scales[v_blk_row + d / block_size]).cast::<f32>();
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
fp8_e5m2_f16_flash!(mt_fp8_e5m2_f16_flash_sdpa_d64, 2u32);
fp8_e5m2_f16_flash!(mt_fp8_e5m2_f16_flash_sdpa_d96, 3u32);
fp8_e5m2_f16_flash!(mt_fp8_e5m2_f16_flash_sdpa_d128, 4u32);
fp8_e5m2_f16_flash!(mt_fp8_e5m2_f16_flash_sdpa_d256, 8u32);
fp8_e5m2_f16_flash!(mt_fp8_e5m2_f16_flash_sdpa_d512, 16u32);

/// FP16-scaled symmetric int flash SDPA (int2/3/4/5/6): bit-stream K/V codes
/// (group 64) × per-group FP16 scale. Clone of `int_flash_f32` with the scale
/// tensor narrowed to f16 (`load(...).cast::<f32>()`); the straddle-aware
/// two-word decode + sign-extend is identical to the FP32 twin.
macro_rules! int_flash_f16 {
    ($name:ident, $dpl:literal, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<f16>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<f16>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            // Round up so a head dim that isn't a multiple of the group (int
            // group 64 over d96 → a 64-block + a 32-block) counts the ragged
            // tail block; matches the host packer's `div_ceil`.
            let n_blocks = (dim + block_size - 1u32) / block_size;
            // Tight bit-stream words per token row (dim is a multiple of 32, so
            // `dim·BITS` is an exact word count — each token row word-aligned).
            let words_per_token = dim * $bits / 32u32;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            // Straddle-aware two-word read of element `d`'s code.
                            let bit_off = d * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(k_packed[k_word_row + word_idx]);
                            let w1 = load(
                                k_packed
                                    [k_word_row + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let kelem = select(q >= $half, qf - $full, qf); // sign-extend
                            let ksc = load(k_scales[k_blk_row + d / block_size]).cast::<f32>();
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_off = d * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(v_packed[v_word_row + word_idx]);
                            let w1 = load(
                                v_packed
                                    [v_word_row + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let velem = select(q >= $half, qf - $full, qf); // sign-extend
                            let vsc = load(v_scales[v_blk_row + d / block_size]).cast::<f32>();
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
int_flash_f16!(mt_int2_f16_flash_sdpa_d64, 2u32, 2u32, 2u32, 4.0f32);
int_flash_f16!(mt_int2_f16_flash_sdpa_d96, 3u32, 2u32, 2u32, 4.0f32);
int_flash_f16!(mt_int2_f16_flash_sdpa_d128, 4u32, 2u32, 2u32, 4.0f32);
int_flash_f16!(mt_int2_f16_flash_sdpa_d256, 8u32, 2u32, 2u32, 4.0f32);
int_flash_f16!(mt_int2_f16_flash_sdpa_d512, 16u32, 2u32, 2u32, 4.0f32);
int_flash_f16!(mt_int3_f16_flash_sdpa_d64, 2u32, 3u32, 4u32, 8.0f32);
int_flash_f16!(mt_int3_f16_flash_sdpa_d96, 3u32, 3u32, 4u32, 8.0f32);
int_flash_f16!(mt_int3_f16_flash_sdpa_d128, 4u32, 3u32, 4u32, 8.0f32);
int_flash_f16!(mt_int3_f16_flash_sdpa_d256, 8u32, 3u32, 4u32, 8.0f32);
int_flash_f16!(mt_int3_f16_flash_sdpa_d512, 16u32, 3u32, 4u32, 8.0f32);
int_flash_f16!(mt_int4_f16_flash_sdpa_d64, 2u32, 4u32, 8u32, 16.0f32);
int_flash_f16!(mt_int4_f16_flash_sdpa_d96, 3u32, 4u32, 8u32, 16.0f32);
int_flash_f16!(mt_int4_f16_flash_sdpa_d128, 4u32, 4u32, 8u32, 16.0f32);
int_flash_f16!(mt_int4_f16_flash_sdpa_d256, 8u32, 4u32, 8u32, 16.0f32);
int_flash_f16!(mt_int4_f16_flash_sdpa_d512, 16u32, 4u32, 8u32, 16.0f32);
int_flash_f16!(mt_int5_f16_flash_sdpa_d64, 2u32, 5u32, 16u32, 32.0f32);
int_flash_f16!(mt_int5_f16_flash_sdpa_d96, 3u32, 5u32, 16u32, 32.0f32);
int_flash_f16!(mt_int5_f16_flash_sdpa_d128, 4u32, 5u32, 16u32, 32.0f32);
int_flash_f16!(mt_int5_f16_flash_sdpa_d256, 8u32, 5u32, 16u32, 32.0f32);
int_flash_f16!(mt_int5_f16_flash_sdpa_d512, 16u32, 5u32, 16u32, 32.0f32);
int_flash_f16!(mt_int6_f16_flash_sdpa_d64, 2u32, 6u32, 32u32, 64.0f32);
int_flash_f16!(mt_int6_f16_flash_sdpa_d96, 3u32, 6u32, 32u32, 64.0f32);
int_flash_f16!(mt_int6_f16_flash_sdpa_d128, 4u32, 6u32, 32u32, 64.0f32);
int_flash_f16!(mt_int6_f16_flash_sdpa_d256, 8u32, 6u32, 32u32, 64.0f32);
int_flash_f16!(mt_int6_f16_flash_sdpa_d512, 16u32, 6u32, 32u32, 64.0f32);

/// int8 flash SDPA, FP16 scale — 8-bit codes (group 64), per-group FP16 scale.
/// Clone of `int8_flash` with the scale tensor narrowed to f16; decode is the
/// same byte-layout sign-extend → `code · scale`.
macro_rules! int8_f16_flash {
    ($name:ident, $dpl:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u8>,
            k_scales: Tensor<f16>,
            v_packed: Tensor<u8>,
            v_scales: Tensor<f16>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] block_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;
            // Round up so a head dim that isn't a multiple of the group (int8 group
            // 64 over d96 → a 64-block + a 32-block) counts the ragged tail block.
            let n_blocks = (dim + block_size - 1u32) / block_size;

            stack_alloc("q_vals", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dpl, "f32");
            for i in range(0u32, $dpl, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;
            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_row = (kv_idx * tokens + t) * dim;
                    let k_blk_row = (kv_idx * tokens + t) * n_blocks;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let kelem = mt_decode_int8(load(k_packed[k_row + d]).cast::<u32>());
                            let ksc = load(k_scales[k_blk_row + d / block_size]).cast::<f32>();
                            dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                        }
                    }
                    let score = simd_sum(dot_partial);
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);
                    let v_row = (kv_idx * tokens + t) * dim;
                    let v_blk_row = (kv_idx * tokens + t) * n_blocks;
                    for i in range(0u32, $dpl, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let velem = mt_decode_int8(load(v_packed[v_row + d]).cast::<u32>());
                            let vsc = load(v_scales[v_blk_row + d / block_size]).cast::<f32>();
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                        }
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dpl, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let oi = stack_load("o", i);
                    let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
                    store(out[q_idx * dim + d], normed.cast::<T>());
                }
            }
        }
    };
}
int8_f16_flash!(mt_int8_f16_flash_sdpa_d64, 2u32);
int8_f16_flash!(mt_int8_f16_flash_sdpa_d96, 3u32);
int8_f16_flash!(mt_int8_f16_flash_sdpa_d128, 4u32);
int8_f16_flash!(mt_int8_f16_flash_sdpa_d256, 8u32);
int8_f16_flash!(mt_int8_f16_flash_sdpa_d512, 16u32);

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn source(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 20_000) as f32 / 20_000.0 * scale - scale * 0.5
            })
            .collect()
    }

    /// Dense softmax-attention over DEQUANTIZED K/V — the flash result oracle,
    /// with optional sinks + sliding window (mirrors the affine kernel's naive).
    #[allow(clippy::too_many_arguments)]
    fn naive(
        q: &[f32],
        k_deq: &[f32],
        v_deq: &[f32],
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        dim: usize,
        scale: f32,
        sinks: &[f32],
        has_sinks: bool,
        window_size: usize,
    ) -> Vec<f32> {
        let repeat = q_heads / kv_heads;
        let mut out = vec![0.0f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            let used = |t: usize| window_size == 0 || t + window_size > tokens - 1;
            let mut scores = vec![0.0f32; tokens];
            for (t, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += scale * q[qh * dim + d] * k_deq[(kvh * tokens + t) * dim + d];
                }
                *s = dot;
            }
            let mut m = if has_sinks { sinks[qh] } else { f32::NEG_INFINITY };
            for (t, &s) in scores.iter().enumerate() {
                if used(t) {
                    m = m.max(s);
                }
            }
            let mut sum = if has_sinks { (sinks[qh] - m).exp() } else { 0.0f32 };
            let mut w = vec![0.0f32; tokens];
            for (t, &s) in scores.iter().enumerate() {
                if used(t) {
                    w[t] = (s - m).exp();
                    sum += w[t];
                }
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            for d in 0..dim {
                let mut acc = 0.0f32;
                for (t, &wt) in w.iter().enumerate() {
                    acc += wt * inv * v_deq[(kvh * tokens + t) * dim + d];
                }
                out[qh * dim + d] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn flash_setup(
        kernel: Kernel,
        fmt: QFormat,
        dim: usize,
        has_sinks: bool,
        window_size: usize,
        dt: DType,
    ) -> TestSetup {
        let (q_heads, kv_heads, tokens) = (2usize, 1usize, 8usize);
        let repeat = q_heads / kv_heads;
        let attn_scale = 1.0f32 / (dim as f32).sqrt();
        let rows = kv_heads * tokens;
        // Queries (rounded through dt), block-scaled K/V cache via the codec.
        let q = unpack_f32(&pack_f32(&source(q_heads * dim, 0x51, 2.0), dt), dt);
        let k_raw = source(rows * dim, 0x62, 3.0);
        let v_raw = source(rows * dim, 0x73, 3.0);
        let kp = crate::quant::format::pack(fmt, &k_raw, rows, dim);
        let vp = crate::quant::format::pack(fmt, &v_raw, rows, dim);
        let k_deq = crate::quant::format::dequant(fmt, &kp, rows, dim);
        let v_deq = crate::quant::format::dequant(fmt, &vp, rows, dim);
        let sinks: Vec<f32> = if has_sinks {
            (0..q_heads).map(|h| 0.5 + 0.25 * h as f32).collect()
        } else {
            vec![0.0f32; q_heads]
        };
        let expected = naive(
            &q,
            &k_deq,
            &v_deq,
            q_heads,
            kv_heads,
            tokens,
            dim,
            attn_scale,
            &sinks,
            has_sinks,
            window_size,
        );
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("queries", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k_packed", kp.codes, weight_dt))
            .input(TestBuffer::from_vec("k_scales", kp.scales, scales_dt))
            .input(TestBuffer::from_vec("v_packed", vp.codes, weight_dt))
            .input(TestBuffer::from_vec("v_scales", vp.scales, scales_dt))
            .input(TestBuffer::from_vec("sinks", pack_f32(&sinks, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", q_heads * dim, dt))
            .constexpr("dim", dim as u32)
            .constexpr("tokens", tokens as u32)
            .constexpr("repeat_count", repeat as u32)
            .constexpr("block_size", fmt.block_size() as u32)
            .constexpr("num_q_heads", q_heads as u32)
            .constexpr("has_sinks", u32::from(has_sinks))
            .constexpr("window_size", window_size as u32)
            .constexpr("scale", attn_scale);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", kp.global.max(vp.global));
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            1,
            q_heads as u32,
            1,
            [32, 1, 1],
        )
    }

    // Base (full attention, no sinks) for all 5 formats at d=128.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp4_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_nvfp4_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_nvfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp4, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp8_e4m3_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxfp8_e4m3_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp8_e5m2_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxfp8_e5m2_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_nvfp8_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_nvfp8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp8, 128, false, 0, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_fp4_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_fp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Fp4, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_fp8_e4m3_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_nvfp8_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_fp8_e5m2_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_fp8_e5m2_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_int8_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_int8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int8, 128, false, 0, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8. The kernel and oracle share the codec, so the GPU output matches
    // the host `dequant` oracle regardless of how coarse the quantization is.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_int2_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int2, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_int3_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int3, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_int4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int4, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_int5_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int5, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(mt_int6_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int6, 128, false, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxint2_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxint2,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxint3_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxint3,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxint4_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxint4,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxint5_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxint5,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxint6_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxint6,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_mxint8_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Mxint8,
            128,
            false,
            0,
            dt,
        )
    }

    // FP16-scale twins (nvfp8_f16 / fp4_f16 / fp8_e5m2_f16 / int2..6_f16 /
    // int8_f16). `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel (same 8-bit-E4M3 +
    // FP16-scale shape). Same codec is shared by kernel + oracle, so the GPU
    // output matches the host `dequant` regardless of scale precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_nvfp8_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_fp4_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Fp4F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_nvfp8_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_fp8_e5m2_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_int2_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Int2F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_int3_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Int3F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_int4_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Int4F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_int5_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Int5F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_int6_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Int6F16,
            128,
            false,
            0,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_flash_sdpa_d128(dt: DType) -> TestSetup {
        flash_setup(
            mt_int8_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Int8F16,
            128,
            false,
            0,
            dt,
        )
    }

    // Sink + sliding-window paths exercised on the mxfp4 representative.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp4_flash_sdpa_d128_sinks(dt: DType) -> TestSetup {
        flash_setup(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, true, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp4_flash_sdpa_d128_window(dt: DType) -> TestSetup {
        flash_setup(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, false, 4, dt)
    }

    // ── Other production head dims (d64/d96/d256/d512), all 9 formats ──
    // `fp8_e4m3` reuses the `nvfp8` kernel (same 8-bit-E4M3 + f32-scale shape).
    macro_rules! flash_dim_test {
        ($test:ident, $kernel:ident, $fmt:expr, $dim:literal) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
            fn $test(dt: DType) -> TestSetup {
                flash_setup($kernel::kernel_ir_for(dt), $fmt, $dim, false, 0, dt)
            }
        };
    }
    // d64
    flash_dim_test!(test_mxfp4_flash_sdpa_d64, mt_mxfp4_flash_sdpa_d64, QFormat::Mxfp4, 64);
    flash_dim_test!(test_nvfp4_flash_sdpa_d64, mt_nvfp4_flash_sdpa_d64, QFormat::Nvfp4, 64);
    flash_dim_test!(
        test_mxfp8_e4m3_flash_sdpa_d64,
        mt_mxfp8_e4m3_flash_sdpa_d64,
        QFormat::Mxfp8E4,
        64
    );
    flash_dim_test!(
        test_mxfp8_e5m2_flash_sdpa_d64,
        mt_mxfp8_e5m2_flash_sdpa_d64,
        QFormat::Mxfp8E5,
        64
    );
    flash_dim_test!(test_nvfp8_flash_sdpa_d64, mt_nvfp8_flash_sdpa_d64, QFormat::Nvfp8, 64);
    flash_dim_test!(test_fp4_flash_sdpa_d64, mt_fp4_flash_sdpa_d64, QFormat::Fp4, 64);
    flash_dim_test!(test_fp8_e4m3_flash_sdpa_d64, mt_nvfp8_flash_sdpa_d64, QFormat::Fp8E4m3, 64);
    flash_dim_test!(test_fp8_e5m2_flash_sdpa_d64, mt_fp8_e5m2_flash_sdpa_d64, QFormat::Fp8E5m2, 64);
    flash_dim_test!(test_int8_flash_sdpa_d64, mt_int8_flash_sdpa_d64, QFormat::Int8, 64);
    flash_dim_test!(test_int2_flash_sdpa_d64, mt_int2_flash_sdpa_d64, QFormat::Int2, 64);
    flash_dim_test!(test_int3_flash_sdpa_d64, mt_int3_flash_sdpa_d64, QFormat::Int3, 64);
    flash_dim_test!(test_int4_flash_sdpa_d64, mt_int4_flash_sdpa_d64, QFormat::Int4, 64);
    flash_dim_test!(test_int5_flash_sdpa_d64, mt_int5_flash_sdpa_d64, QFormat::Int5, 64);
    flash_dim_test!(test_int6_flash_sdpa_d64, mt_int6_flash_sdpa_d64, QFormat::Int6, 64);
    flash_dim_test!(test_mxint2_flash_sdpa_d64, mt_mxint2_flash_sdpa_d64, QFormat::Mxint2, 64);
    flash_dim_test!(test_mxint3_flash_sdpa_d64, mt_mxint3_flash_sdpa_d64, QFormat::Mxint3, 64);
    flash_dim_test!(test_mxint4_flash_sdpa_d64, mt_mxint4_flash_sdpa_d64, QFormat::Mxint4, 64);
    flash_dim_test!(test_mxint5_flash_sdpa_d64, mt_mxint5_flash_sdpa_d64, QFormat::Mxint5, 64);
    flash_dim_test!(test_mxint6_flash_sdpa_d64, mt_mxint6_flash_sdpa_d64, QFormat::Mxint6, 64);
    flash_dim_test!(test_mxint8_flash_sdpa_d64, mt_mxint8_flash_sdpa_d64, QFormat::Mxint8, 64);
    // d96
    flash_dim_test!(test_mxfp4_flash_sdpa_d96, mt_mxfp4_flash_sdpa_d96, QFormat::Mxfp4, 96);
    flash_dim_test!(test_nvfp4_flash_sdpa_d96, mt_nvfp4_flash_sdpa_d96, QFormat::Nvfp4, 96);
    flash_dim_test!(
        test_mxfp8_e4m3_flash_sdpa_d96,
        mt_mxfp8_e4m3_flash_sdpa_d96,
        QFormat::Mxfp8E4,
        96
    );
    flash_dim_test!(
        test_mxfp8_e5m2_flash_sdpa_d96,
        mt_mxfp8_e5m2_flash_sdpa_d96,
        QFormat::Mxfp8E5,
        96
    );
    flash_dim_test!(test_nvfp8_flash_sdpa_d96, mt_nvfp8_flash_sdpa_d96, QFormat::Nvfp8, 96);
    flash_dim_test!(test_fp4_flash_sdpa_d96, mt_fp4_flash_sdpa_d96, QFormat::Fp4, 96);
    flash_dim_test!(test_fp8_e4m3_flash_sdpa_d96, mt_nvfp8_flash_sdpa_d96, QFormat::Fp8E4m3, 96);
    flash_dim_test!(test_fp8_e5m2_flash_sdpa_d96, mt_fp8_e5m2_flash_sdpa_d96, QFormat::Fp8E5m2, 96);
    // int8 d96: ragged trailing block (64 + 32) — see the kernel/packer notes.
    flash_dim_test!(test_int8_flash_sdpa_d96, mt_int8_flash_sdpa_d96, QFormat::Int8, 96);
    // int2-6 d96: int group 64 → ragged trailing block (64 + 32), same as int8;
    // mxint group 32 divides 96 evenly. The bit-stream stays word-aligned (96·BITS
    // is a whole number of u32 words for every width), so decode is unaffected.
    flash_dim_test!(test_int2_flash_sdpa_d96, mt_int2_flash_sdpa_d96, QFormat::Int2, 96);
    flash_dim_test!(test_int3_flash_sdpa_d96, mt_int3_flash_sdpa_d96, QFormat::Int3, 96);
    flash_dim_test!(test_int4_flash_sdpa_d96, mt_int4_flash_sdpa_d96, QFormat::Int4, 96);
    flash_dim_test!(test_int5_flash_sdpa_d96, mt_int5_flash_sdpa_d96, QFormat::Int5, 96);
    flash_dim_test!(test_int6_flash_sdpa_d96, mt_int6_flash_sdpa_d96, QFormat::Int6, 96);
    flash_dim_test!(test_mxint2_flash_sdpa_d96, mt_mxint2_flash_sdpa_d96, QFormat::Mxint2, 96);
    flash_dim_test!(test_mxint3_flash_sdpa_d96, mt_mxint3_flash_sdpa_d96, QFormat::Mxint3, 96);
    flash_dim_test!(test_mxint4_flash_sdpa_d96, mt_mxint4_flash_sdpa_d96, QFormat::Mxint4, 96);
    flash_dim_test!(test_mxint5_flash_sdpa_d96, mt_mxint5_flash_sdpa_d96, QFormat::Mxint5, 96);
    flash_dim_test!(test_mxint6_flash_sdpa_d96, mt_mxint6_flash_sdpa_d96, QFormat::Mxint6, 96);
    flash_dim_test!(test_mxint8_flash_sdpa_d96, mt_mxint8_flash_sdpa_d96, QFormat::Mxint8, 96);
    // d256
    flash_dim_test!(test_mxfp4_flash_sdpa_d256, mt_mxfp4_flash_sdpa_d256, QFormat::Mxfp4, 256);
    flash_dim_test!(test_nvfp4_flash_sdpa_d256, mt_nvfp4_flash_sdpa_d256, QFormat::Nvfp4, 256);
    flash_dim_test!(
        test_mxfp8_e4m3_flash_sdpa_d256,
        mt_mxfp8_e4m3_flash_sdpa_d256,
        QFormat::Mxfp8E4,
        256
    );
    flash_dim_test!(
        test_mxfp8_e5m2_flash_sdpa_d256,
        mt_mxfp8_e5m2_flash_sdpa_d256,
        QFormat::Mxfp8E5,
        256
    );
    flash_dim_test!(test_nvfp8_flash_sdpa_d256, mt_nvfp8_flash_sdpa_d256, QFormat::Nvfp8, 256);
    flash_dim_test!(test_fp4_flash_sdpa_d256, mt_fp4_flash_sdpa_d256, QFormat::Fp4, 256);
    flash_dim_test!(test_fp8_e4m3_flash_sdpa_d256, mt_nvfp8_flash_sdpa_d256, QFormat::Fp8E4m3, 256);
    flash_dim_test!(
        test_fp8_e5m2_flash_sdpa_d256,
        mt_fp8_e5m2_flash_sdpa_d256,
        QFormat::Fp8E5m2,
        256
    );
    flash_dim_test!(test_int8_flash_sdpa_d256, mt_int8_flash_sdpa_d256, QFormat::Int8, 256);
    flash_dim_test!(test_int2_flash_sdpa_d256, mt_int2_flash_sdpa_d256, QFormat::Int2, 256);
    flash_dim_test!(test_int3_flash_sdpa_d256, mt_int3_flash_sdpa_d256, QFormat::Int3, 256);
    flash_dim_test!(test_int4_flash_sdpa_d256, mt_int4_flash_sdpa_d256, QFormat::Int4, 256);
    flash_dim_test!(test_int5_flash_sdpa_d256, mt_int5_flash_sdpa_d256, QFormat::Int5, 256);
    flash_dim_test!(test_int6_flash_sdpa_d256, mt_int6_flash_sdpa_d256, QFormat::Int6, 256);
    flash_dim_test!(test_mxint2_flash_sdpa_d256, mt_mxint2_flash_sdpa_d256, QFormat::Mxint2, 256);
    flash_dim_test!(test_mxint3_flash_sdpa_d256, mt_mxint3_flash_sdpa_d256, QFormat::Mxint3, 256);
    flash_dim_test!(test_mxint4_flash_sdpa_d256, mt_mxint4_flash_sdpa_d256, QFormat::Mxint4, 256);
    flash_dim_test!(test_mxint5_flash_sdpa_d256, mt_mxint5_flash_sdpa_d256, QFormat::Mxint5, 256);
    flash_dim_test!(test_mxint6_flash_sdpa_d256, mt_mxint6_flash_sdpa_d256, QFormat::Mxint6, 256);
    flash_dim_test!(test_mxint8_flash_sdpa_d256, mt_mxint8_flash_sdpa_d256, QFormat::Mxint8, 256);
    // d512
    flash_dim_test!(test_mxfp4_flash_sdpa_d512, mt_mxfp4_flash_sdpa_d512, QFormat::Mxfp4, 512);
    flash_dim_test!(test_nvfp4_flash_sdpa_d512, mt_nvfp4_flash_sdpa_d512, QFormat::Nvfp4, 512);
    flash_dim_test!(
        test_mxfp8_e4m3_flash_sdpa_d512,
        mt_mxfp8_e4m3_flash_sdpa_d512,
        QFormat::Mxfp8E4,
        512
    );
    flash_dim_test!(
        test_mxfp8_e5m2_flash_sdpa_d512,
        mt_mxfp8_e5m2_flash_sdpa_d512,
        QFormat::Mxfp8E5,
        512
    );
    flash_dim_test!(test_nvfp8_flash_sdpa_d512, mt_nvfp8_flash_sdpa_d512, QFormat::Nvfp8, 512);
    flash_dim_test!(test_fp4_flash_sdpa_d512, mt_fp4_flash_sdpa_d512, QFormat::Fp4, 512);
    flash_dim_test!(test_fp8_e4m3_flash_sdpa_d512, mt_nvfp8_flash_sdpa_d512, QFormat::Fp8E4m3, 512);
    flash_dim_test!(
        test_fp8_e5m2_flash_sdpa_d512,
        mt_fp8_e5m2_flash_sdpa_d512,
        QFormat::Fp8E5m2,
        512
    );
    flash_dim_test!(test_int8_flash_sdpa_d512, mt_int8_flash_sdpa_d512, QFormat::Int8, 512);
    flash_dim_test!(test_int2_flash_sdpa_d512, mt_int2_flash_sdpa_d512, QFormat::Int2, 512);
    flash_dim_test!(test_int3_flash_sdpa_d512, mt_int3_flash_sdpa_d512, QFormat::Int3, 512);
    flash_dim_test!(test_int4_flash_sdpa_d512, mt_int4_flash_sdpa_d512, QFormat::Int4, 512);
    flash_dim_test!(test_int5_flash_sdpa_d512, mt_int5_flash_sdpa_d512, QFormat::Int5, 512);
    flash_dim_test!(test_int6_flash_sdpa_d512, mt_int6_flash_sdpa_d512, QFormat::Int6, 512);
    flash_dim_test!(test_mxint2_flash_sdpa_d512, mt_mxint2_flash_sdpa_d512, QFormat::Mxint2, 512);
    flash_dim_test!(test_mxint3_flash_sdpa_d512, mt_mxint3_flash_sdpa_d512, QFormat::Mxint3, 512);
    flash_dim_test!(test_mxint4_flash_sdpa_d512, mt_mxint4_flash_sdpa_d512, QFormat::Mxint4, 512);
    flash_dim_test!(test_mxint5_flash_sdpa_d512, mt_mxint5_flash_sdpa_d512, QFormat::Mxint5, 512);
    flash_dim_test!(test_mxint6_flash_sdpa_d512, mt_mxint6_flash_sdpa_d512, QFormat::Mxint6, 512);
    flash_dim_test!(test_mxint8_flash_sdpa_d512, mt_mxint8_flash_sdpa_d512, QFormat::Mxint8, 512);

    // ── FP16-scale twins across the other production head dims (d64/d96/d256/
    // d512) ── Same dims as the FP32-scaled formats, looser tol to match the
    // integer formats. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel.
    macro_rules! flash_dim_test_f16 {
        ($test:ident, $kernel:ident, $fmt:expr, $dim:literal) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
            fn $test(dt: DType) -> TestSetup {
                flash_setup($kernel::kernel_ir_for(dt), $fmt, $dim, false, 0, dt)
            }
        };
    }
    // d64
    flash_dim_test_f16!(
        test_nvfp8_f16_flash_sdpa_d64,
        mt_nvfp8_f16_flash_sdpa_d64,
        QFormat::Nvfp8F16,
        64
    );
    flash_dim_test_f16!(
        test_fp4_f16_flash_sdpa_d64,
        mt_fp4_f16_flash_sdpa_d64,
        QFormat::Fp4F16,
        64
    );
    flash_dim_test_f16!(
        test_fp8_e4m3_f16_flash_sdpa_d64,
        mt_nvfp8_f16_flash_sdpa_d64,
        QFormat::Fp8E4m3F16,
        64
    );
    flash_dim_test_f16!(
        test_fp8_e5m2_f16_flash_sdpa_d64,
        mt_fp8_e5m2_f16_flash_sdpa_d64,
        QFormat::Fp8E5m2F16,
        64
    );
    flash_dim_test_f16!(
        test_int2_f16_flash_sdpa_d64,
        mt_int2_f16_flash_sdpa_d64,
        QFormat::Int2F16,
        64
    );
    flash_dim_test_f16!(
        test_int3_f16_flash_sdpa_d64,
        mt_int3_f16_flash_sdpa_d64,
        QFormat::Int3F16,
        64
    );
    flash_dim_test_f16!(
        test_int4_f16_flash_sdpa_d64,
        mt_int4_f16_flash_sdpa_d64,
        QFormat::Int4F16,
        64
    );
    flash_dim_test_f16!(
        test_int5_f16_flash_sdpa_d64,
        mt_int5_f16_flash_sdpa_d64,
        QFormat::Int5F16,
        64
    );
    flash_dim_test_f16!(
        test_int6_f16_flash_sdpa_d64,
        mt_int6_f16_flash_sdpa_d64,
        QFormat::Int6F16,
        64
    );
    flash_dim_test_f16!(
        test_int8_f16_flash_sdpa_d64,
        mt_int8_f16_flash_sdpa_d64,
        QFormat::Int8F16,
        64
    );
    // d96 (int group 64 → ragged 64+32 tail; fp4 group 32 divides 96 evenly)
    flash_dim_test_f16!(
        test_nvfp8_f16_flash_sdpa_d96,
        mt_nvfp8_f16_flash_sdpa_d96,
        QFormat::Nvfp8F16,
        96
    );
    flash_dim_test_f16!(
        test_fp4_f16_flash_sdpa_d96,
        mt_fp4_f16_flash_sdpa_d96,
        QFormat::Fp4F16,
        96
    );
    flash_dim_test_f16!(
        test_fp8_e4m3_f16_flash_sdpa_d96,
        mt_nvfp8_f16_flash_sdpa_d96,
        QFormat::Fp8E4m3F16,
        96
    );
    flash_dim_test_f16!(
        test_fp8_e5m2_f16_flash_sdpa_d96,
        mt_fp8_e5m2_f16_flash_sdpa_d96,
        QFormat::Fp8E5m2F16,
        96
    );
    flash_dim_test_f16!(
        test_int2_f16_flash_sdpa_d96,
        mt_int2_f16_flash_sdpa_d96,
        QFormat::Int2F16,
        96
    );
    flash_dim_test_f16!(
        test_int3_f16_flash_sdpa_d96,
        mt_int3_f16_flash_sdpa_d96,
        QFormat::Int3F16,
        96
    );
    flash_dim_test_f16!(
        test_int4_f16_flash_sdpa_d96,
        mt_int4_f16_flash_sdpa_d96,
        QFormat::Int4F16,
        96
    );
    flash_dim_test_f16!(
        test_int5_f16_flash_sdpa_d96,
        mt_int5_f16_flash_sdpa_d96,
        QFormat::Int5F16,
        96
    );
    flash_dim_test_f16!(
        test_int6_f16_flash_sdpa_d96,
        mt_int6_f16_flash_sdpa_d96,
        QFormat::Int6F16,
        96
    );
    flash_dim_test_f16!(
        test_int8_f16_flash_sdpa_d96,
        mt_int8_f16_flash_sdpa_d96,
        QFormat::Int8F16,
        96
    );
    // d256
    flash_dim_test_f16!(
        test_nvfp8_f16_flash_sdpa_d256,
        mt_nvfp8_f16_flash_sdpa_d256,
        QFormat::Nvfp8F16,
        256
    );
    flash_dim_test_f16!(
        test_fp4_f16_flash_sdpa_d256,
        mt_fp4_f16_flash_sdpa_d256,
        QFormat::Fp4F16,
        256
    );
    flash_dim_test_f16!(
        test_fp8_e4m3_f16_flash_sdpa_d256,
        mt_nvfp8_f16_flash_sdpa_d256,
        QFormat::Fp8E4m3F16,
        256
    );
    flash_dim_test_f16!(
        test_fp8_e5m2_f16_flash_sdpa_d256,
        mt_fp8_e5m2_f16_flash_sdpa_d256,
        QFormat::Fp8E5m2F16,
        256
    );
    flash_dim_test_f16!(
        test_int2_f16_flash_sdpa_d256,
        mt_int2_f16_flash_sdpa_d256,
        QFormat::Int2F16,
        256
    );
    flash_dim_test_f16!(
        test_int3_f16_flash_sdpa_d256,
        mt_int3_f16_flash_sdpa_d256,
        QFormat::Int3F16,
        256
    );
    flash_dim_test_f16!(
        test_int4_f16_flash_sdpa_d256,
        mt_int4_f16_flash_sdpa_d256,
        QFormat::Int4F16,
        256
    );
    flash_dim_test_f16!(
        test_int5_f16_flash_sdpa_d256,
        mt_int5_f16_flash_sdpa_d256,
        QFormat::Int5F16,
        256
    );
    flash_dim_test_f16!(
        test_int6_f16_flash_sdpa_d256,
        mt_int6_f16_flash_sdpa_d256,
        QFormat::Int6F16,
        256
    );
    flash_dim_test_f16!(
        test_int8_f16_flash_sdpa_d256,
        mt_int8_f16_flash_sdpa_d256,
        QFormat::Int8F16,
        256
    );
    // d512
    flash_dim_test_f16!(
        test_nvfp8_f16_flash_sdpa_d512,
        mt_nvfp8_f16_flash_sdpa_d512,
        QFormat::Nvfp8F16,
        512
    );
    flash_dim_test_f16!(
        test_fp4_f16_flash_sdpa_d512,
        mt_fp4_f16_flash_sdpa_d512,
        QFormat::Fp4F16,
        512
    );
    flash_dim_test_f16!(
        test_fp8_e4m3_f16_flash_sdpa_d512,
        mt_nvfp8_f16_flash_sdpa_d512,
        QFormat::Fp8E4m3F16,
        512
    );
    flash_dim_test_f16!(
        test_fp8_e5m2_f16_flash_sdpa_d512,
        mt_fp8_e5m2_f16_flash_sdpa_d512,
        QFormat::Fp8E5m2F16,
        512
    );
    flash_dim_test_f16!(
        test_int2_f16_flash_sdpa_d512,
        mt_int2_f16_flash_sdpa_d512,
        QFormat::Int2F16,
        512
    );
    flash_dim_test_f16!(
        test_int3_f16_flash_sdpa_d512,
        mt_int3_f16_flash_sdpa_d512,
        QFormat::Int3F16,
        512
    );
    flash_dim_test_f16!(
        test_int4_f16_flash_sdpa_d512,
        mt_int4_f16_flash_sdpa_d512,
        QFormat::Int4F16,
        512
    );
    flash_dim_test_f16!(
        test_int5_f16_flash_sdpa_d512,
        mt_int5_f16_flash_sdpa_d512,
        QFormat::Int5F16,
        512
    );
    flash_dim_test_f16!(
        test_int6_f16_flash_sdpa_d512,
        mt_int6_f16_flash_sdpa_d512,
        QFormat::Int6F16,
        512
    );
    flash_dim_test_f16!(
        test_int8_f16_flash_sdpa_d512,
        mt_int8_f16_flash_sdpa_d512,
        QFormat::Int8F16,
        512
    );
}

/// Decode-shape benches: single-query attention over a block-scaled K/V cache
/// (d=128, 8 q-heads / 1 kv-head, 2048 tokens). Throughput data-independent.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn flash_bench(kernel: Kernel, fmt: QFormat, dim: usize, dt: DType) -> BenchSetup {
        let (q_heads, kv_heads, tokens) = (8usize, 1usize, 2048usize);
        let rows = kv_heads * tokens;
        let n_blocks = rows * dim.div_ceil(fmt.block_size()); // round up: ragged tail block
        // 8-bit codes are one uchar each; sub-byte codes tight-bit-pack into u32
        // words (with a guard word for straddling 3/5/6-bit reads). Both axes are
        // driven off the format so new integer formats size their buffers right.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (rows * dim, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(rows * dim, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = q_heads * dim * sz                      // queries
            + 2 * codes_len * codes_dt.size_bytes()         // K + V codes
            + 2 * n_blocks * scales_dt.size_bytes()         // K + V scales
            + q_heads * dim * sz; // out
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("queries", q_heads * dim, dt))
            .buffer(BenchBuffer::random("k_packed", codes_len, codes_dt))
            .buffer(BenchBuffer::random("k_scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("v_packed", codes_len, codes_dt))
            .buffer(BenchBuffer::random("v_scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("sinks", q_heads, DType::F32))
            .buffer(BenchBuffer::zeros("out", q_heads * dim, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("tokens", tokens as u32)
            .constexpr("repeat_count", (q_heads / kv_heads) as u32)
            .constexpr("block_size", fmt.block_size() as u32)
            .constexpr("num_q_heads", q_heads as u32)
            .constexpr("has_sinks", 0u32)
            .constexpr("window_size", 0u32)
            .constexpr("scale", 1.0f32 / (dim as f32).sqrt());
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(1, q_heads as u32, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            // QK^T + softmax·V over the cache: ~4·q_heads·tokens·dim FLOPs.
            .flops(4 * q_heads as u64 * tokens as u64 * dim as u64)
            .with_shape_label(format!("{} q={q_heads} t={tokens} d={dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp4, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxfp8_e4m3_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp8E4, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxfp8_e5m2_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp8E5, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp8, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_fp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Fp4, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Fp8E4m3, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_fp8_e5m2_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Fp8E5m2, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int8, 128, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) + MXINT8.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int2_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int2, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int3_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int3, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int4, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int5_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int5, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int6_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int6, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxint2_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxint2, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxint3_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxint3, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxint4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxint4, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxint5_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxint5, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxint6_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxint6, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxint8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxint8, 128, dt)
    }
    // FP16-scale twins (nvfp8_f16 / fp4_f16 / fp8_e5m2_f16 / int2..6_f16 /
    // int8_f16). `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp8_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp8F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_fp4_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Fp4F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp8_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(
            mt_fp8_e5m2_f16_flash_sdpa_d128::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            128,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int2_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int2F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int3_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int3F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int4_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int4F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int5_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int5F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int6_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int6F16, 128, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_int8_f16_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Int8F16, 128, dt)
    }

    // Large-head-dim perf matrix (d256 = long-context; d512 = Gemma global),
    // all 9 formats. d64/d96 are correctness-tested but follow the d128 trend.
    macro_rules! flash_dim_bench {
        ($bench:ident, $kernel:ident, $fmt:expr, $dim:literal) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $bench(dt: DType) -> BenchSetup {
                flash_bench($kernel::kernel_ir_for(dt), $fmt, $dim, dt)
            }
        };
    }
    // d256
    flash_dim_bench!(bench_mxfp4_flash_d256, mt_mxfp4_flash_sdpa_d256, QFormat::Mxfp4, 256);
    flash_dim_bench!(bench_nvfp4_flash_d256, mt_nvfp4_flash_sdpa_d256, QFormat::Nvfp4, 256);
    flash_dim_bench!(
        bench_mxfp8_e4m3_flash_d256,
        mt_mxfp8_e4m3_flash_sdpa_d256,
        QFormat::Mxfp8E4,
        256
    );
    flash_dim_bench!(
        bench_mxfp8_e5m2_flash_d256,
        mt_mxfp8_e5m2_flash_sdpa_d256,
        QFormat::Mxfp8E5,
        256
    );
    flash_dim_bench!(bench_nvfp8_flash_d256, mt_nvfp8_flash_sdpa_d256, QFormat::Nvfp8, 256);
    flash_dim_bench!(bench_fp4_flash_d256, mt_fp4_flash_sdpa_d256, QFormat::Fp4, 256);
    flash_dim_bench!(bench_fp8_e4m3_flash_d256, mt_nvfp8_flash_sdpa_d256, QFormat::Fp8E4m3, 256);
    flash_dim_bench!(bench_fp8_e5m2_flash_d256, mt_fp8_e5m2_flash_sdpa_d256, QFormat::Fp8E5m2, 256);
    flash_dim_bench!(bench_int8_flash_d256, mt_int8_flash_sdpa_d256, QFormat::Int8, 256);
    flash_dim_bench!(bench_int2_flash_d256, mt_int2_flash_sdpa_d256, QFormat::Int2, 256);
    flash_dim_bench!(bench_int3_flash_d256, mt_int3_flash_sdpa_d256, QFormat::Int3, 256);
    flash_dim_bench!(bench_int4_flash_d256, mt_int4_flash_sdpa_d256, QFormat::Int4, 256);
    flash_dim_bench!(bench_int5_flash_d256, mt_int5_flash_sdpa_d256, QFormat::Int5, 256);
    flash_dim_bench!(bench_int6_flash_d256, mt_int6_flash_sdpa_d256, QFormat::Int6, 256);
    flash_dim_bench!(bench_mxint2_flash_d256, mt_mxint2_flash_sdpa_d256, QFormat::Mxint2, 256);
    flash_dim_bench!(bench_mxint3_flash_d256, mt_mxint3_flash_sdpa_d256, QFormat::Mxint3, 256);
    flash_dim_bench!(bench_mxint4_flash_d256, mt_mxint4_flash_sdpa_d256, QFormat::Mxint4, 256);
    flash_dim_bench!(bench_mxint5_flash_d256, mt_mxint5_flash_sdpa_d256, QFormat::Mxint5, 256);
    flash_dim_bench!(bench_mxint6_flash_d256, mt_mxint6_flash_sdpa_d256, QFormat::Mxint6, 256);
    flash_dim_bench!(bench_mxint8_flash_d256, mt_mxint8_flash_sdpa_d256, QFormat::Mxint8, 256);
    // d512
    flash_dim_bench!(bench_mxfp4_flash_d512, mt_mxfp4_flash_sdpa_d512, QFormat::Mxfp4, 512);
    flash_dim_bench!(bench_nvfp4_flash_d512, mt_nvfp4_flash_sdpa_d512, QFormat::Nvfp4, 512);
    flash_dim_bench!(
        bench_mxfp8_e4m3_flash_d512,
        mt_mxfp8_e4m3_flash_sdpa_d512,
        QFormat::Mxfp8E4,
        512
    );
    flash_dim_bench!(
        bench_mxfp8_e5m2_flash_d512,
        mt_mxfp8_e5m2_flash_sdpa_d512,
        QFormat::Mxfp8E5,
        512
    );
    flash_dim_bench!(bench_nvfp8_flash_d512, mt_nvfp8_flash_sdpa_d512, QFormat::Nvfp8, 512);
    flash_dim_bench!(bench_fp4_flash_d512, mt_fp4_flash_sdpa_d512, QFormat::Fp4, 512);
    flash_dim_bench!(bench_fp8_e4m3_flash_d512, mt_nvfp8_flash_sdpa_d512, QFormat::Fp8E4m3, 512);
    flash_dim_bench!(bench_fp8_e5m2_flash_d512, mt_fp8_e5m2_flash_sdpa_d512, QFormat::Fp8E5m2, 512);
    flash_dim_bench!(bench_int8_flash_d512, mt_int8_flash_sdpa_d512, QFormat::Int8, 512);
    flash_dim_bench!(bench_int2_flash_d512, mt_int2_flash_sdpa_d512, QFormat::Int2, 512);
    flash_dim_bench!(bench_int3_flash_d512, mt_int3_flash_sdpa_d512, QFormat::Int3, 512);
    flash_dim_bench!(bench_int4_flash_d512, mt_int4_flash_sdpa_d512, QFormat::Int4, 512);
    flash_dim_bench!(bench_int5_flash_d512, mt_int5_flash_sdpa_d512, QFormat::Int5, 512);
    flash_dim_bench!(bench_int6_flash_d512, mt_int6_flash_sdpa_d512, QFormat::Int6, 512);
    flash_dim_bench!(bench_mxint2_flash_d512, mt_mxint2_flash_sdpa_d512, QFormat::Mxint2, 512);
    flash_dim_bench!(bench_mxint3_flash_d512, mt_mxint3_flash_sdpa_d512, QFormat::Mxint3, 512);
    flash_dim_bench!(bench_mxint4_flash_d512, mt_mxint4_flash_sdpa_d512, QFormat::Mxint4, 512);
    flash_dim_bench!(bench_mxint5_flash_d512, mt_mxint5_flash_sdpa_d512, QFormat::Mxint5, 512);
    flash_dim_bench!(bench_mxint6_flash_d512, mt_mxint6_flash_sdpa_d512, QFormat::Mxint6, 512);
    flash_dim_bench!(bench_mxint8_flash_d512, mt_mxint8_flash_sdpa_d512, QFormat::Mxint8, 512);

    // ── FP16-scale twins, large-head-dim perf matrix (d256 / d512) ──
    // `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel.
    // d256
    flash_dim_bench!(
        bench_nvfp8_f16_flash_d256,
        mt_nvfp8_f16_flash_sdpa_d256,
        QFormat::Nvfp8F16,
        256
    );
    flash_dim_bench!(bench_fp4_f16_flash_d256, mt_fp4_f16_flash_sdpa_d256, QFormat::Fp4F16, 256);
    flash_dim_bench!(
        bench_fp8_e4m3_f16_flash_d256,
        mt_nvfp8_f16_flash_sdpa_d256,
        QFormat::Fp8E4m3F16,
        256
    );
    flash_dim_bench!(
        bench_fp8_e5m2_f16_flash_d256,
        mt_fp8_e5m2_f16_flash_sdpa_d256,
        QFormat::Fp8E5m2F16,
        256
    );
    flash_dim_bench!(bench_int2_f16_flash_d256, mt_int2_f16_flash_sdpa_d256, QFormat::Int2F16, 256);
    flash_dim_bench!(bench_int3_f16_flash_d256, mt_int3_f16_flash_sdpa_d256, QFormat::Int3F16, 256);
    flash_dim_bench!(bench_int4_f16_flash_d256, mt_int4_f16_flash_sdpa_d256, QFormat::Int4F16, 256);
    flash_dim_bench!(bench_int5_f16_flash_d256, mt_int5_f16_flash_sdpa_d256, QFormat::Int5F16, 256);
    flash_dim_bench!(bench_int6_f16_flash_d256, mt_int6_f16_flash_sdpa_d256, QFormat::Int6F16, 256);
    flash_dim_bench!(bench_int8_f16_flash_d256, mt_int8_f16_flash_sdpa_d256, QFormat::Int8F16, 256);
    // d512
    flash_dim_bench!(
        bench_nvfp8_f16_flash_d512,
        mt_nvfp8_f16_flash_sdpa_d512,
        QFormat::Nvfp8F16,
        512
    );
    flash_dim_bench!(bench_fp4_f16_flash_d512, mt_fp4_f16_flash_sdpa_d512, QFormat::Fp4F16, 512);
    flash_dim_bench!(
        bench_fp8_e4m3_f16_flash_d512,
        mt_nvfp8_f16_flash_sdpa_d512,
        QFormat::Fp8E4m3F16,
        512
    );
    flash_dim_bench!(
        bench_fp8_e5m2_f16_flash_d512,
        mt_fp8_e5m2_f16_flash_sdpa_d512,
        QFormat::Fp8E5m2F16,
        512
    );
    flash_dim_bench!(bench_int2_f16_flash_d512, mt_int2_f16_flash_sdpa_d512, QFormat::Int2F16, 512);
    flash_dim_bench!(bench_int3_f16_flash_d512, mt_int3_f16_flash_sdpa_d512, QFormat::Int3F16, 512);
    flash_dim_bench!(bench_int4_f16_flash_d512, mt_int4_f16_flash_sdpa_d512, QFormat::Int4F16, 512);
    flash_dim_bench!(bench_int5_f16_flash_d512, mt_int5_f16_flash_sdpa_d512, QFormat::Int5F16, 512);
    flash_dim_bench!(bench_int6_f16_flash_d512, mt_int6_f16_flash_sdpa_d512, QFormat::Int6F16, 512);
    flash_dim_bench!(bench_int8_f16_flash_d512, mt_int8_f16_flash_sdpa_d512, QFormat::Int8F16, 512);
}

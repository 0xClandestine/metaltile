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
//!   - 8-bit (mxfp8/nvfp8): `k_packed [B·nKV, N, dim] u8`, scales u8 (E8M0) or
//!     f32 (nvfp8). V mirrors K. `dim` a multiple of `block_size`.
//!
//! These kernels are fixed to head-dim 128 (`dims_per_lane = 4`), the canonical
//! attention head width; other dims follow the same body with a different
//! `dims_per_lane`. Codegen-only; correctness pinned by `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 flash SDPA (d=128) — E2M1 K/V (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_flash_sdpa_d128<T>(
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

    stack_alloc("q_vals", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
        stack_store("q_vals", i, v * scale);
    }

    let sink_val = load(sinks[q_idx % num_q_heads]);
    let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
    let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
    stack_alloc("o", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        stack_store("o", i, 0.0f32);
    }

    let causal_upper = tokens - 1u32;
    for t in range(0u32, tokens, 1u32) {
        let use_key = select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
        if use_key {
            let k_word_row = (kv_idx * tokens + t) * words_per_token;
            let k_blk_row = (kv_idx * tokens + t) * n_blocks;
            let mut dot_partial = 0.0f32;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let nib =
                        (load(k_packed[k_word_row + d / 8u32]) >> ((d % 8u32) * 4u32)) & 0xFu32;
                    let ksc =
                        exp2(load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32);
                    dot_partial = dot_partial + stack_load("q_vals", i) * (e2m1_decode(nib) * ksc);
                }
            }
            let score = simd_sum(dot_partial);
            let new_m = select(m_acc > score, m_acc, score);
            let exp_diff = exp(m_acc - new_m);
            let exp_score = exp(score - new_m);
            let v_word_row = (kv_idx * tokens + t) * words_per_token;
            let v_blk_row = (kv_idx * tokens + t) * n_blocks;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let nib =
                        (load(v_packed[v_word_row + d / 8u32]) >> ((d % 8u32) * 4u32)) & 0xFu32;
                    let vsc =
                        exp2(load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32);
                    let prev = stack_load("o", i);
                    stack_store("o", i, prev * exp_diff + exp_score * (e2m1_decode(nib) * vsc));
                }
            }
            l_acc = l_acc * exp_diff + exp_score;
            m_acc = new_m;
        }
    }

    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        if d < dim {
            let oi = stack_load("o", i);
            let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
            store(out[q_idx * dim + d], normed.cast::<T>());
        }
    }
}

/// nvfp4 flash SDPA (d=128) — E2M1 K/V (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_flash_sdpa_d128<T>(
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

    stack_alloc("q_vals", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
        stack_store("q_vals", i, v * scale);
    }

    let sink_val = load(sinks[q_idx % num_q_heads]);
    let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
    let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
    stack_alloc("o", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        stack_store("o", i, 0.0f32);
    }

    let causal_upper = tokens - 1u32;
    for t in range(0u32, tokens, 1u32) {
        let use_key = select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
        if use_key {
            let k_word_row = (kv_idx * tokens + t) * words_per_token;
            let k_blk_row = (kv_idx * tokens + t) * n_blocks;
            let mut dot_partial = 0.0f32;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let nib =
                        (load(k_packed[k_word_row + d / 8u32]) >> ((d % 8u32) * 4u32)) & 0xFu32;
                    let ksc = e4m3_decode(load(k_scales[k_blk_row + d / block_size]).cast::<u32>())
                        * global;
                    dot_partial = dot_partial + stack_load("q_vals", i) * (e2m1_decode(nib) * ksc);
                }
            }
            let score = simd_sum(dot_partial);
            let new_m = select(m_acc > score, m_acc, score);
            let exp_diff = exp(m_acc - new_m);
            let exp_score = exp(score - new_m);
            let v_word_row = (kv_idx * tokens + t) * words_per_token;
            let v_blk_row = (kv_idx * tokens + t) * n_blocks;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let nib =
                        (load(v_packed[v_word_row + d / 8u32]) >> ((d % 8u32) * 4u32)) & 0xFu32;
                    let vsc = e4m3_decode(load(v_scales[v_blk_row + d / block_size]).cast::<u32>())
                        * global;
                    let prev = stack_load("o", i);
                    stack_store("o", i, prev * exp_diff + exp_score * (e2m1_decode(nib) * vsc));
                }
            }
            l_acc = l_acc * exp_diff + exp_score;
            m_acc = new_m;
        }
    }

    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        if d < dim {
            let oi = stack_load("o", i);
            let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
            store(out[q_idx * dim + d], normed.cast::<T>());
        }
    }
}

/// mxfp8 (E4M3) flash SDPA (d=128) — 8-bit K/V (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_flash_sdpa_d128<T>(
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

    stack_alloc("q_vals", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
        stack_store("q_vals", i, v * scale);
    }

    let sink_val = load(sinks[q_idx % num_q_heads]);
    let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
    let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
    stack_alloc("o", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        stack_store("o", i, 0.0f32);
    }

    let causal_upper = tokens - 1u32;
    for t in range(0u32, tokens, 1u32) {
        let use_key = select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
        if use_key {
            let k_row = (kv_idx * tokens + t) * dim;
            let k_blk_row = (kv_idx * tokens + t) * n_blocks;
            let mut dot_partial = 0.0f32;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let kelem = e4m3_decode(load(k_packed[k_row + d]).cast::<u32>());
                    let ksc =
                        exp2(load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32);
                    dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                }
            }
            let score = simd_sum(dot_partial);
            let new_m = select(m_acc > score, m_acc, score);
            let exp_diff = exp(m_acc - new_m);
            let exp_score = exp(score - new_m);
            let v_row = (kv_idx * tokens + t) * dim;
            let v_blk_row = (kv_idx * tokens + t) * n_blocks;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let velem = e4m3_decode(load(v_packed[v_row + d]).cast::<u32>());
                    let vsc =
                        exp2(load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32);
                    let prev = stack_load("o", i);
                    stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                }
            }
            l_acc = l_acc * exp_diff + exp_score;
            m_acc = new_m;
        }
    }

    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        if d < dim {
            let oi = stack_load("o", i);
            let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
            store(out[q_idx * dim + d], normed.cast::<T>());
        }
    }
}

/// mxfp8 (E5M2) flash SDPA (d=128) — 8-bit K/V (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_flash_sdpa_d128<T>(
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

    stack_alloc("q_vals", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
        stack_store("q_vals", i, v * scale);
    }

    let sink_val = load(sinks[q_idx % num_q_heads]);
    let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
    let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
    stack_alloc("o", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        stack_store("o", i, 0.0f32);
    }

    let causal_upper = tokens - 1u32;
    for t in range(0u32, tokens, 1u32) {
        let use_key = select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
        if use_key {
            let k_row = (kv_idx * tokens + t) * dim;
            let k_blk_row = (kv_idx * tokens + t) * n_blocks;
            let mut dot_partial = 0.0f32;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let kelem = e5m2_decode(load(k_packed[k_row + d]).cast::<u32>());
                    let ksc =
                        exp2(load(k_scales[k_blk_row + d / block_size]).cast::<f32>() - 127.0f32);
                    dot_partial = dot_partial + stack_load("q_vals", i) * (kelem * ksc);
                }
            }
            let score = simd_sum(dot_partial);
            let new_m = select(m_acc > score, m_acc, score);
            let exp_diff = exp(m_acc - new_m);
            let exp_score = exp(score - new_m);
            let v_row = (kv_idx * tokens + t) * dim;
            let v_blk_row = (kv_idx * tokens + t) * n_blocks;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let velem = e5m2_decode(load(v_packed[v_row + d]).cast::<u32>());
                    let vsc =
                        exp2(load(v_scales[v_blk_row + d / block_size]).cast::<f32>() - 127.0f32);
                    let prev = stack_load("o", i);
                    stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                }
            }
            l_acc = l_acc * exp_diff + exp_score;
            m_acc = new_m;
        }
    }

    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        if d < dim {
            let oi = stack_load("o", i);
            let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
            store(out[q_idx * dim + d], normed.cast::<T>());
        }
    }
}

/// nvfp8 flash SDPA (d=128) — E4M3 K/V (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_flash_sdpa_d128<T>(
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

    stack_alloc("q_vals", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
        stack_store("q_vals", i, v * scale);
    }

    let sink_val = load(sinks[q_idx % num_q_heads]);
    let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
    let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
    stack_alloc("o", 4, "f32");
    for i in range(0u32, 4u32, 1u32) {
        stack_store("o", i, 0.0f32);
    }

    let causal_upper = tokens - 1u32;
    for t in range(0u32, tokens, 1u32) {
        let use_key = select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
        if use_key {
            let k_row = (kv_idx * tokens + t) * dim;
            let k_blk_row = (kv_idx * tokens + t) * n_blocks;
            let mut dot_partial = 0.0f32;
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let kelem = e4m3_decode(load(k_packed[k_row + d]).cast::<u32>());
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
            for i in range(0u32, 4u32, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    let velem = e4m3_decode(load(v_packed[v_row + d]).cast::<u32>());
                    let vsc = load(v_scales[v_blk_row + d / block_size]);
                    let prev = stack_load("o", i);
                    stack_store("o", i, prev * exp_diff + exp_score * (velem * vsc));
                }
            }
            l_acc = l_acc * exp_diff + exp_score;
            m_acc = new_m;
        }
    }

    for i in range(0u32, 4u32, 1u32) {
        let d = lane + i * 32u32;
        if d < dim {
            let oi = stack_load("o", i);
            let normed = select(l_acc > 0.0f32, oi / l_acc, oi);
            store(out[q_idx * dim + d], normed.cast::<T>());
        }
    }
}

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
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
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

    // Sink + sliding-window paths exercised on the mxfp4 representative.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp4_flash_sdpa_d128_sinks(dt: DType) -> TestSetup {
        flash_setup(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, true, 0, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 3e-2, 1.5e-1])]
    fn test_mxfp4_flash_sdpa_d128_window(dt: DType) -> TestSetup {
        flash_setup(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, false, 4, dt)
    }
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
        let n_blocks = rows * (dim / fmt.block_size());
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (rows * dim / 8, DType::U32)
        } else {
            (rows * dim, DType::U8)
        };
        let scales_dt = if matches!(fmt, QFormat::Nvfp8) { DType::F32 } else { DType::U8 };
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

    #[bench(name = "ffai/flash_block_sdpa/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp4, 128, dt)
    }
    #[bench(name = "ffai/flash_block_sdpa/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp4_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp4, 128, dt)
    }
    #[bench(name = "ffai/flash_block_sdpa/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxfp8_e4m3_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp8E4, 128, dt)
    }
    #[bench(name = "ffai/flash_block_sdpa/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_mxfp8_e5m2_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Mxfp8E5, 128, dt)
    }
    #[bench(name = "ffai/flash_block_sdpa/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_flash(dt: DType) -> BenchSetup {
        flash_bench(mt_nvfp8_flash_sdpa_d128::kernel_ir_for(dt), QFormat::Nvfp8, 128, dt)
    }
}

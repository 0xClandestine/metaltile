//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Flash quantized SDPA — single-pass online-softmax attention over an
//! affine-quantized K/V cache. Port of `flash_quantized_sdpa.h`
//! (spec 041 phase 1.1/1.2). The affine-quant counterpart of
//! `aura_flash_sdpa`: K and V are dequantized inline per thread from
//! packed-index + per-group scale + bias triples (the layout
//! `quantized` matmul consumes), instead of an AURA codebook.
//!
//! Layout (row-contiguous, N = `tokens`, G = `group_size`):
//!   - queries:  [B*nQ, dim]              T   (caller has *not* pre-scaled)
//!   - k_packed: [B*nKV, N, dim/(32/bits)] u32
//!   - k_scales: [B*nKV, N, dim/G]        T
//!   - k_biases: [B*nKV, N, dim/G]        T
//!   - v_packed / v_scales / v_biases: same shape rule
//!   - sinks:    [num_q_heads]            f32
//!   - out:      [B*nQ, dim]              T
//!
//! `scale` (attention 1/sqrt(d)) multiplies the query once. `has_sinks`
//! (0/1) and `window_size` (0 = full causal) are constexpr. The packed
//! layout is the wasteful pack-strided form (`32/bits` values per u32,
//! no cross-word spill) — bits ∈ {4, 8} divide 32 cleanly.
//!
//! Lane `program_id::<0>()` ∈ [0,32) owns dim slots `lane + i*32`;
//! `program_id::<1>()` = query index. Single-simdgroup shape, matching
//! `aura_flash_sdpa` (token-parallelism is a perf follow-up).
//!
//! ## Mask variants
//!
//! Production attention often requires an explicit attention mask in
//! addition to the built-in causal / sliding-window guard. Two new
//! constexpr-gated kernel variants cover the MLX-upstream mask shapes:
//!
//! - **Bool mask** (`flash_quantized_sdpa_bool_mask_b{4,8}_d{64,128,256}`):
//!   takes a `mask_bool: Tensor<u32>` of shape `[B*nQ, tokens]` (packed
//!   as u32, one bit per token) — or flat byte-per-token; see note below.
//!   When `mask_bool[q_idx * tokens + t] == 0` the key at position `t`
//!   is skipped (softmax weight set to zero). Useful for segment packing
//!   and cross-sequence masking.
//!
//! - **Float mask** (`flash_quantized_sdpa_float_mask_b{4,8}_d{64,128,256}`):
//!   takes a `mask_float: Tensor<T>` of shape `[B*nQ, tokens]`.
//!   The value `mask_float[q_idx * tokens + t]` is added to the raw
//!   attention logit before the online-softmax step, enabling relative-
//!   position biases (ALiBi, T5 bias).
//!
//! Both variants are separate kernel functions (not combined into one)
//! to avoid the cost of loading an unused mask buffer on the common
//! causal-only path. The bool and float masks are composable by chaining
//! their logit modifications inside the token loop.
//!
//! The mask buffers are per-element (one f32/T or one u32 per token per
//! query), row-major `[B*nQ, tokens]`. For the bool mask, each slot is
//! a full `u32` (0 = masked, non-zero = visible) — matching the MLX
//! `mask_t` convention used in `aura_flash_sdpa`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, B*nQ, 1]`, `tg = [32, 1, 1]`.
//! - `dims_per_lane = ceil(dim / 32)`; `dim` a multiple of `32/bits`.
//!
//! Codegen-only; correctness pinned by
//! `tests/flash_quantized_sdpa_gpu_correctness.rs`.

use metaltile::kernel;

macro_rules! flash_quantized_sdpa_kernel {
    ($name:ident, $bits:literal, $dim:literal, $dims_per_lane:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<T>,
            k_biases: Tensor<T>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<T>,
            v_biases: Tensor<T>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] group_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let pack_factor = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let n_groups = dim / group_size;
            let words_per_token = dim / pack_factor;

            // Per-lane query slice, pre-scaled by the attention scale.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            // Online-softmax accumulators (sink = virtual key, value 0).
            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_grp_row = (kv_idx * tokens + t) * n_groups;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(k_packed[k_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let ksc = load(k_scales[k_grp_row + g]).cast::<f32>();
                            let kb = load(k_biases[k_grp_row + g]).cast::<f32>();
                            let kj = ksc * val.cast::<f32>() + kb;
                            dot_partial = dot_partial + stack_load("q_vals", i) * kj;
                        }
                    }
                    let score = simd_sum(dot_partial);

                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_grp_row = (kv_idx * tokens + t) * n_groups;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(v_packed[v_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let vsc = load(v_scales[v_grp_row + g]).cast::<f32>();
                            let vb = load(v_biases[v_grp_row + g]).cast::<f32>();
                            let vj = vsc * val.cast::<f32>() + vb;
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * vj);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dims_per_lane, 1u32) {
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

flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d64, 4u32, 64u32, 2u32, "b4_d64");
// d=96: GPT-NeoX head dim. dims_per_lane = ceil(96/32) = 3.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d96, 4u32, 96u32, 3u32, "b4_d96");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d128, 4u32, 128u32, 4u32, "b4_d128");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d256, 4u32, 256u32, 8u32, "b4_d256");
// d=512: Gemma 4 global-attention head dim. dims_per_lane = 512/32 = 16.
// Register pressure with 16 fp32 accumulators pushes maxTotalThreadsPerThreadgroup
// below 1024; dispatch at 256 threads/TG (8 SG) — same approach as
// ffai_sdpa_decode_d512 which also uses 16 elements/lane.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d512, 4u32, 512u32, 16u32, "b4_d512");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d64, 8u32, 64u32, 2u32, "b8_d64");
// d=96: GPT-NeoX, int8.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d96, 8u32, 96u32, 3u32, "b8_d96");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d128, 8u32, 128u32, 4u32, "b8_d128");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d256, 8u32, 256u32, 8u32, "b8_d256");
// d=512: Gemma 4 global, int8. Same 256-thread/TG constraint as b4_d512.
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d512, 8u32, 512u32, 16u32, "b8_d512");

// ── Bool-mask variants ───────────────────────────────────────────────────
//
// `mask_bool: Tensor<u32>` — shape `[B*nQ, tokens]`, one u32 per token.
// When the slot is zero the key at that position is excluded from
// attention (the online-softmax contribution is dropped). Non-zero = visible.
//
// The mask tensor is flat u32 (not bit-packed) for simplicity; one u32
// per token keeps the load a single scalar read with no shift/mask.

macro_rules! flash_quantized_sdpa_bool_mask_kernel {
    ($name:ident, $bits:literal, $dim:literal, $dims_per_lane:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<T>,
            k_biases: Tensor<T>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<T>,
            v_biases: Tensor<T>,
            sinks: Tensor<f32>,
            mask_bool: Tensor<u32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] group_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let pack_factor = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let n_groups = dim / group_size;
            let words_per_token = dim / pack_factor;

            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                // Causal / sliding-window gate (same as base kernel).
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                // Bool mask gate: skip tokens where the mask slot is 0.
                let mask_pass = load(mask_bool[q_idx * tokens + t]) != 0u32;
                if use_key & mask_pass {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_grp_row = (kv_idx * tokens + t) * n_groups;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(k_packed[k_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let ksc = load(k_scales[k_grp_row + g]).cast::<f32>();
                            let kb = load(k_biases[k_grp_row + g]).cast::<f32>();
                            let kj = ksc * val.cast::<f32>() + kb;
                            dot_partial = dot_partial + stack_load("q_vals", i) * kj;
                        }
                    }
                    let score = simd_sum(dot_partial);

                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_grp_row = (kv_idx * tokens + t) * n_groups;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(v_packed[v_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let vsc = load(v_scales[v_grp_row + g]).cast::<f32>();
                            let vb = load(v_biases[v_grp_row + g]).cast::<f32>();
                            let vj = vsc * val.cast::<f32>() + vb;
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * vj);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dims_per_lane, 1u32) {
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

flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b4_d64,
    4u32,
    64u32,
    2u32,
    "bool_mask_b4_d64"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b4_d128,
    4u32,
    128u32,
    4u32,
    "bool_mask_b4_d128"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b4_d256,
    4u32,
    256u32,
    8u32,
    "bool_mask_b4_d256"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b8_d64,
    8u32,
    64u32,
    2u32,
    "bool_mask_b8_d64"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b8_d128,
    8u32,
    128u32,
    4u32,
    "bool_mask_b8_d128"
);
flash_quantized_sdpa_bool_mask_kernel!(
    flash_quantized_sdpa_bool_mask_b8_d256,
    8u32,
    256u32,
    8u32,
    "bool_mask_b8_d256"
);

// ── Float-mask variants ──────────────────────────────────────────────────
//
// `mask_float: Tensor<T>` — shape `[B*nQ, tokens]`, one `T` per token.
// The value is added to the raw attention logit before the softmax step,
// enabling relative-position biases (ALiBi, T5 bias, etc.).

macro_rules! flash_quantized_sdpa_float_mask_kernel {
    ($name:ident, $bits:literal, $dim:literal, $dims_per_lane:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            queries: Tensor<T>,
            k_packed: Tensor<u32>,
            k_scales: Tensor<T>,
            k_biases: Tensor<T>,
            v_packed: Tensor<u32>,
            v_scales: Tensor<T>,
            v_biases: Tensor<T>,
            sinks: Tensor<f32>,
            mask_float: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] tokens: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] group_size: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
            #[constexpr] scale: f32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let pack_factor = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let n_groups = dim / group_size;
            let words_per_token = dim / pack_factor;

            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(queries[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v * scale);
            }

            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    let k_word_row = (kv_idx * tokens + t) * words_per_token;
                    let k_grp_row = (kv_idx * tokens + t) * n_groups;
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(k_packed[k_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let ksc = load(k_scales[k_grp_row + g]).cast::<f32>();
                            let kb = load(k_biases[k_grp_row + g]).cast::<f32>();
                            let kj = ksc * val.cast::<f32>() + kb;
                            dot_partial = dot_partial + stack_load("q_vals", i) * kj;
                        }
                    }
                    // Load the float mask bias and add it to the logit.
                    // The bias is a scalar per (q, t) token — all 32 lanes
                    // in the simdgroup load from the same address and obtain
                    // the same value, so the addition is uniform across lanes.
                    let bias = load(mask_float[q_idx * tokens + t]).cast::<f32>();
                    let score = simd_sum(dot_partial) + bias;

                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    let v_word_row = (kv_idx * tokens + t) * words_per_token;
                    let v_grp_row = (kv_idx * tokens + t) * n_groups;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let word_idx = d / pack_factor;
                            let shift = (d % pack_factor) * $bits;
                            let val = (load(v_packed[v_word_row + word_idx]) >> shift) & mask;
                            let g = d / group_size;
                            let vsc = load(v_scales[v_grp_row + g]).cast::<f32>();
                            let vb = load(v_biases[v_grp_row + g]).cast::<f32>();
                            let vj = vsc * val.cast::<f32>() + vb;
                            let prev = stack_load("o", i);
                            stack_store("o", i, prev * exp_diff + exp_score * vj);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            for i in range(0u32, $dims_per_lane, 1u32) {
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

flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b4_d64,
    4u32,
    64u32,
    2u32,
    "float_mask_b4_d64"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b4_d128,
    4u32,
    128u32,
    4u32,
    "float_mask_b4_d128"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b4_d256,
    4u32,
    256u32,
    8u32,
    "float_mask_b4_d256"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b8_d64,
    8u32,
    64u32,
    2u32,
    "float_mask_b8_d64"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b8_d128,
    8u32,
    128u32,
    4u32,
    "float_mask_b8_d128"
);
flash_quantized_sdpa_float_mask_kernel!(
    flash_quantized_sdpa_float_mask_b8_d256,
    8u32,
    256u32,
    8u32,
    "float_mask_b8_d256"
);

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };

    use super::*;

    fn pack_dt(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn f32_le(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn xorshift(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s % 20_000) as f32 / 20_000.0 * 2.0 - 1.0
    }

    fn source_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
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

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    /// Affine per-group quantize. Returns (packed, scales, biases, deq).
    fn quantize_kv(
        vals: &[f32],
        rows: usize,
        dim: usize,
        group_size: usize,
        bits: u32,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pack_factor = 32 / bits as usize;
        let n_groups = dim / group_size;
        let max_q = ((1u32 << bits) - 1) as f32;
        let mut packed = vec![0u32; rows * dim / pack_factor];
        let mut scales = vec![0.0_f32; rows * n_groups];
        let mut biases = vec![0.0_f32; rows * n_groups];
        let mut deq = vec![0.0_f32; rows * dim];
        for r in 0..rows {
            for g in 0..n_groups {
                let slice = &vals[r * dim + g * group_size..r * dim + (g + 1) * group_size];
                let mn = slice.iter().copied().fold(f32::INFINITY, f32::min);
                let mx = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / max_q };
                scales[r * n_groups + g] = scale;
                biases[r * n_groups + g] = mn;
                for (i, &v) in slice.iter().enumerate() {
                    let d = g * group_size + i;
                    let q = ((v - mn) / scale).round().clamp(0.0, max_q) as u32;
                    packed[(r * dim + d) / pack_factor] |= q << ((d % pack_factor) * bits as usize);
                    deq[r * dim + d] = scale * q as f32 + mn;
                }
            }
        }
        (packed, scales, biases, deq)
    }

    /// Naive causal softmax attention oracle.
    fn naive_sdpa(
        q: &[f32],
        k_deq: &[f32],
        v_deq: &[f32],
        sinks: &[f32],
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        dim: usize,
        attn_scale: f32,
        has_sinks: bool,
        window: usize,
        num_q_heads: usize,
    ) -> Vec<f32> {
        let repeat = q_heads / kv_heads;
        let causal = tokens - 1;
        let mut out = vec![0.0_f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            let mut kept: Vec<(usize, f32)> = Vec::new();
            for t in 0..tokens {
                if window == 0 || t + window > causal {
                    let dot: f32 = (0..dim)
                        .map(|d| attn_scale * q[qh * dim + d] * k_deq[(kvh * tokens + t) * dim + d])
                        .sum();
                    kept.push((t, dot));
                }
            }
            let mut m = if has_sinks { sinks[qh % num_q_heads] } else { f32::NEG_INFINITY };
            for &(_, s) in &kept {
                m = m.max(s);
            }
            let mut sum_w = if has_sinks { (sinks[qh % num_q_heads] - m).exp() } else { 0.0 };
            let mut acc = vec![0.0_f32; dim];
            for &(t, s) in &kept {
                let w = (s - m).exp();
                sum_w += w;
                for d in 0..dim {
                    acc[d] += w * v_deq[(kvh * tokens + t) * dim + d];
                }
            }
            let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
            for d in 0..dim {
                out[qh * dim + d] = acc[d] * inv;
            }
        }
        out
    }

    fn make_setup(
        kernel_ir_fn: fn(DType) -> metaltile_core::ir::Kernel,
        bits: u32,
        dim: usize,
        dt: DType,
        has_sinks: bool,
        window: usize,
        tol: f64,
    ) -> TestSetup {
        let (q_heads, kv_heads, tokens) = (2usize, 1usize, 8usize);
        let group_size = if dim % 64 == 0 { 64usize } else { 32usize };
        let attn_scale = 1.0_f32 / (dim as f32).sqrt();
        let q_raw = source_vec(q_heads * dim, 0x51, 2.0);
        let q: Vec<f32> = q_raw.iter().map(|&v| round_dt(v, dt)).collect();
        let k_raw = source_vec(kv_heads * tokens * dim, 0x62, 3.0);
        let v_raw = source_vec(kv_heads * tokens * dim, 0x73, 3.0);
        let sinks: Vec<f32> = (0..q_heads).map(|i| -0.4 + 0.3 * i as f32).collect();
        let (kp, ks, kb, k_deq) = quantize_kv(&k_raw, kv_heads * tokens, dim, group_size, bits);
        let (vp, vs, vb, v_deq) = quantize_kv(&v_raw, kv_heads * tokens, dim, group_size, bits);
        let expected = naive_sdpa(
            &q, &k_deq, &v_deq, &sinks, q_heads, kv_heads, tokens, dim, attn_scale, has_sinks,
            window, q_heads,
        );
        let repeat = q_heads / kv_heads;
        let out_size = q_heads * dim * if dt == DType::F32 { 4 } else { 2 };
        let mut kernel = kernel_ir_fn(dt);
        kernel.mode = KernelMode::Grid3D;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("queries", pack_dt(&q, dt), dt))
            .input(TestBuffer::from_vec("k_packed", pack_u32(&kp), DType::U32))
            .input(TestBuffer::from_vec("k_scales", pack_dt(&ks, dt), dt))
            .input(TestBuffer::from_vec("k_biases", pack_dt(&kb, dt), dt))
            .input(TestBuffer::from_vec("v_packed", pack_u32(&vp), DType::U32))
            .input(TestBuffer::from_vec("v_scales", pack_dt(&vs, dt), dt))
            .input(TestBuffer::from_vec("v_biases", pack_dt(&vb, dt), dt))
            .input(TestBuffer::from_vec(
                "sinks",
                bytemuck::cast_slice::<f32, u8>(&sinks).to_vec(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec("out", vec![0u8; out_size], dt))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("repeat_count", u32_le(repeat as u32), DType::U32))
            .input(TestBuffer::from_vec("group_size", u32_le(group_size as u32), DType::U32))
            .input(TestBuffer::from_vec("num_q_heads", u32_le(q_heads as u32), DType::U32))
            .input(TestBuffer::from_vec("has_sinks", u32_le(u32::from(has_sinks)), DType::U32))
            .input(TestBuffer::from_vec("window_size", u32_le(window as u32), DType::U32))
            .input(TestBuffer::from_vec("scale", f32_le(attn_scale), DType::F32))
            .expect(TestBuffer::from_vec("out", pack_dt(&expected, dt), dt))
            .grid_3d(1, q_heads as u32, 1, [32, 1, 1])
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b4_d128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_flash_quantized_sdpa_b4_d128_f32(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, 128, dt, false, 0, 1e-4)
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b4_d128_sinks_f32", dtypes = [f32], tol = 1e-4)]
    fn test_flash_quantized_sdpa_b4_d128_sinks_f32(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, 128, dt, true, 0, 1e-4)
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b4_d128_window_f32", dtypes = [f32], tol = 1e-4)]
    fn test_flash_quantized_sdpa_b4_d128_window_f32(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, 128, dt, false, 4, 1e-4)
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b8_d128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_flash_quantized_sdpa_b8_d128_f32(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b8_d128::kernel_ir_for, 8, 128, dt, false, 0, 1e-4)
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b4_d128_bf16", dtypes = [bf16], tol = 5e-2)]
    fn test_flash_quantized_sdpa_b4_d128_bf16(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b4_d128::kernel_ir_for, 4, 128, dt, false, 0, 5e-2)
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b4_d96_f32", dtypes = [f32], tol = 1e-4)]
    fn test_flash_quantized_sdpa_b4_d96_f32(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b4_d96::kernel_ir_for, 4, 96, dt, false, 0, 1e-4)
    }

    #[test_kernel(name = "ffai/flash_quantized_sdpa/b8_d96_f32", dtypes = [f32], tol = 1e-4)]
    fn test_flash_quantized_sdpa_b8_d96_f32(dt: DType) -> TestSetup {
        make_setup(flash_quantized_sdpa_b8_d96::kernel_ir_for, 8, 96, dt, false, 0, 1e-4)
    }
}

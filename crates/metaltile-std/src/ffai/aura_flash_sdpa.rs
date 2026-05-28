//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA fused single-pass SDPA — online-softmax attention over an
//! AURA/TurboQuant-compressed K/V cache, with optional attention sinks
//! and sliding-window causal masking. Port of `turbo_flash_sdpa.h`
//! (spec 041 phase 1.1, GPT-OSS sink-attention family).
//!
//! Unlike the `aura_flash_p1` + `aura_flash_pass2` pair, this does the
//! whole attention in one dispatch — one threadgroup (a single
//! 32-lane simdgroup) per query, iterating every K/V token with a
//! running online softmax, then writing the normalized output. This
//! side-steps the pass2-with-sinks graph-fusion incoherence that the
//! two-pass β-with-sinks drafts hit on GPT-OSS-20B.
//!
//! Layout (matches `aura_flash_p1`):
//!   - q_rot:        [B*nQ, dim] f32   (WHT-rotated + pre-scaled by caller)
//!   - key_packed:   [B*nKV, tokens, key_packed_width]   u32
//!   - key_norms:    [B*nKV, tokens]   f32
//!   - key_codebook: [2^key_bits]      f32
//!   - val_packed:   [B*nKV, tokens, value_packed_width] u32
//!   - val_norms:    [B*nKV, tokens]   f32
//!   - val_codebook: [2^value_bits]    f32
//!   - sinks:        [num_q_heads]     f32  (per-head sink logit)
//!   - out:          [B*nQ, dim]       T    (rotated V space)
//!
//! `has_sinks` (0/1) and `window_size` (0 = full causal) are constexpr.
//! When `has_sinks == 1` the running softmax starts at `(m = sink,
//! l = 1)` — the sink behaves as a virtual key with value 0.
//!
//! Lane `program_id::<0>()` ∈ [0,32) owns dim slots `lane + i*32`;
//! `program_id::<1>()` = query index. The MLX reference fans tokens
//! across 32 simdgroups; this port keeps the simpler single-simdgroup
//! shape of `aura_flash_p1` (correctness-equivalent; token-parallelism
//! is a perf follow-up).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, B*nQ, 1]`, `tg = [32, 1, 1]` — exactly one
//!   simdgroup per query.
//! - `dims_per_lane = ceil(dim / 32)`.
//!
//! Codegen-only; correctness pinned by
//! `tests/aura_flash_sdpa_gpu_correctness.rs`.

use metaltile::kernel;

macro_rules! aura_flash_sdpa_kernel {
    (
        $name:ident,
        $key_bits:literal,
        $value_bits:literal,
        $key_levels:literal,
        $value_levels:literal,
        $dims_per_lane:literal,
        $subop:literal
    ) => {
        #[kernel]
        pub fn $name<T>(
            q_rot: Tensor<f32>,
            key_packed: Tensor<u32>,
            key_norms: Tensor<f32>,
            key_codebook: Tensor<f32>,
            val_packed: Tensor<u32>,
            val_norms: Tensor<f32>,
            val_codebook: Tensor<f32>,
            sinks: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] key_packed_width: u32,
            #[constexpr] value_packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] num_q_heads: u32,
            #[constexpr] has_sinks: u32,
            #[constexpr] window_size: u32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let kv_idx = q_idx / repeat_count;

            let key_mask = (1u32 << $key_bits) - 1u32;
            let val_mask = (1u32 << $value_bits) - 1u32;

            // Codebook caches in per-thread stack arrays.
            stack_alloc("key_cb", $key_levels, "f32");
            for i in range(0u32, $key_levels, 1u32) {
                stack_store("key_cb", i, load(key_codebook[i]));
            }
            stack_alloc("val_cb", $value_levels, "f32");
            for i in range(0u32, $value_levels, 1u32) {
                stack_store("val_cb", i, load(val_codebook[i]));
            }

            // Per-lane slice of the rotated query, loaded once.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(q_rot[q_idx * dim + d]), 0.0f32);
                stack_store("q_vals", i, v);
            }

            // Online-softmax accumulators. With sinks, the running
            // softmax starts at (m = sink, l = 1): the sink is a virtual
            // key whose value is 0.
            let sink_val = load(sinks[q_idx % num_q_heads]);
            let mut m_acc = select(has_sinks > 0u32, sink_val, neg_infinity());
            let mut l_acc = select(has_sinks > 0u32, 1.0f32, 0.0f32);
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            // L=1 decode: the query attends K positions [0, tokens).
            let causal_upper = tokens - 1u32;

            for t in range(0u32, tokens, 1u32) {
                // Sliding-window mask: keep key `t` when window is off,
                // or when `t` is within `window_size` of the last pos.
                let use_key =
                    select(window_size == 0u32, t < tokens, t + window_size > causal_upper);
                if use_key {
                    // Q · K in the compressed domain.
                    // NOTE: row stride is `kv_stride` (cache's `maxSeq`), not
                    // `tokens` (live KV-row count). For caches that aren't
                    // fully populated yet, head 1 starts at offset
                    // `kv_stride`, NOT `tokens` — otherwise we'd read head 0's
                    // tail bytes as if they were head 1's rows.
                    let k_packed_row = (kv_idx * kv_stride + t) * key_packed_width;
                    let k_norm = load(key_norms[kv_idx * kv_stride + t]);
                    let mut dot_partial = 0.0f32;
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_offset = d * $key_bits;
                            let word_idx = bit_offset / 32u32;
                            let shift = bit_offset & 31u32;
                            let bits_in_w0 = 32u32 - shift;
                            let lo_bits = select(bits_in_w0 >= $key_bits, $key_bits, bits_in_w0);
                            let spill = $key_bits - lo_bits;
                            let w0 = load(key_packed[k_packed_row + word_idx]);
                            let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                            let w1 = load(key_packed[k_packed_row + w1_idx]);
                            let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let value = (lo | hi) & key_mask;
                            let centroid = stack_load("key_cb", value);
                            let qv = stack_load("q_vals", i);
                            dot_partial = dot_partial + qv * centroid;
                        }
                    }
                    let score = simd_sum(dot_partial) * k_norm;

                    // Online-softmax max-shift.
                    let new_m = select(m_acc > score, m_acc, score);
                    let exp_diff = exp(m_acc - new_m);
                    let exp_score = exp(score - new_m);

                    // V-side update from compressed centroids.
                    // Same `kv_stride` row stride as the K side above.
                    let v_packed_row = (kv_idx * kv_stride + t) * value_packed_width;
                    let v_norm = load(val_norms[kv_idx * kv_stride + t]);
                    for i in range(0u32, $dims_per_lane, 1u32) {
                        let d = lane + i * 32u32;
                        if d < dim {
                            let bit_offset = d * $value_bits;
                            let word_idx = bit_offset / 32u32;
                            let shift = bit_offset & 31u32;
                            let bits_in_w0 = 32u32 - shift;
                            let lo_bits =
                                select(bits_in_w0 >= $value_bits, $value_bits, bits_in_w0);
                            let spill = $value_bits - lo_bits;
                            let w0 = load(val_packed[v_packed_row + word_idx]);
                            let w1_idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                            let w1 = load(val_packed[v_packed_row + w1_idx]);
                            let lo = (w0 >> shift) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let value = (lo | hi) & val_mask;
                            let prev = stack_load("o", i);
                            let centroid = stack_load("val_cb", value);
                            let upd = prev * exp_diff + exp_score * centroid * v_norm;
                            stack_store("o", i, upd);
                        }
                    }

                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
            }

            // Normalize and write the final attention output.
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

aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb2_d128,
    4u32,
    2u32,
    16u32,
    4u32,
    4u32,
    "flash_sdpa_kb4_vb2_d128"
);
aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb4_d128,
    4u32,
    4u32,
    16u32,
    16u32,
    4u32,
    "flash_sdpa_kb4_vb4_d128"
);
aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb2_d64,
    4u32,
    2u32,
    16u32,
    4u32,
    2u32,
    "flash_sdpa_kb4_vb2_d64"
);
aura_flash_sdpa_kernel!(
    aura_flash_sdpa_kb4_vb4_d64,
    4u32,
    4u32,
    16u32,
    16u32,
    2u32,
    "flash_sdpa_kb4_vb4_d64"
);

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }
    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn pack_int_indices(
        indices: &[u32],
        kv_heads: usize,
        tokens: usize,
        dim: usize,
        bits: usize,
    ) -> Vec<u32> {
        let mask = (1u32 << bits) - 1;
        let pw = (dim * bits).div_ceil(32);
        let mut packed = vec![0u32; kv_heads * tokens * pw];
        for kvh in 0..kv_heads {
            for t in 0..tokens {
                for d in 0..dim {
                    let idx = indices[(kvh * tokens + t) * dim + d] & mask;
                    let bit = d * bits;
                    let word = bit / 32;
                    let shift = bit & 31;
                    packed[(kvh * tokens + t) * pw + word] |= idx << shift;
                    let spill = (shift + bits) as i32 - 32;
                    if spill > 0 {
                        packed[(kvh * tokens + t) * pw + word + 1] |=
                            idx >> (bits as u32 - spill as u32);
                    }
                }
            }
        }
        packed
    }

    /// CPU reference for aura_flash_sdpa (no sinks, no window).
    fn naive_aura_flash_sdpa(
        q_rot: &[f32],
        key_idx: &[u32],
        val_idx: &[u32],
        key_norms: &[f32],
        val_norms: &[f32],
        key_cb: &[f32],
        val_cb: &[f32],
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        dim: usize,
    ) -> Vec<f32> {
        let repeat = q_heads / kv_heads;
        let mut out = vec![0.0_f32; q_heads * dim];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            let mut scores = vec![0.0_f32; tokens];
            for t in 0..tokens {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    let q = key_idx[(kvh * tokens + t) * dim + d];
                    dot += q_rot[qh * dim + d] * key_cb[q as usize];
                }
                scores[t] = dot * key_norms[kvh * tokens + t];
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum_w: f32 = weights.iter().sum();
            let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
            for d in 0..dim {
                let mut acc = 0.0_f32;
                for t in 0..tokens {
                    let v = val_idx[(kvh * tokens + t) * dim + d];
                    acc += weights[t] * val_cb[v as usize] * val_norms[kvh * tokens + t];
                }
                out[qh * dim + d] = acc * inv;
            }
        }
        out
    }

    #[test_kernel(name = "ffai/aura/flash_sdpa_kb4_vb2_d128", dtypes = [f32], tol = 1e-3)]
    fn test_aura_flash_sdpa(dt: DType) -> TestSetup {
        use super::aura_flash_sdpa_kb4_vb2_d128;
        let dim = 128usize;
        let key_bits = 4usize;
        let value_bits = 2usize;
        let key_packed_width = (dim * key_bits).div_ceil(32);
        let value_packed_width = (dim * value_bits).div_ceil(32);
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let tokens = 8usize;
        let repeat = q_heads / kv_heads;
        let kv_stride = kv_heads * tokens;
        let has_sinks = 0u32;
        let window_size = 0u32; // 0 = full attention

        let key_codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();
        let val_codebook: Vec<f32> = (0..4).map(|i| -1.0 + 2.0 * i as f32 / 3.0).collect();
        let key_indices: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
        let val_indices: Vec<u32> =
            (0..kv_heads * tokens * dim).map(|i| ((i * 11 + 5) % 4) as u32).collect();
        let key_packed = pack_int_indices(&key_indices, kv_heads, tokens, dim, key_bits);
        let val_packed = pack_int_indices(&val_indices, kv_heads, tokens, dim, value_bits);
        let key_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.5 + 0.05 * i as f32).collect();
        let val_norms: Vec<f32> = (0..kv_heads * tokens).map(|i| 0.3 + 0.07 * i as f32).collect();
        let q_rot: Vec<f32> =
            (0..q_heads * dim).map(|i| (((i * 13) % 19) as f32 - 9.0) * 0.02).collect();
        let sinks: Vec<f32> = vec![0.0_f32; dim]; // no sinks

        let expected = naive_aura_flash_sdpa(
            &q_rot,
            &key_indices,
            &val_indices,
            &key_norms,
            &val_norms,
            &key_codebook,
            &val_codebook,
            q_heads,
            kv_heads,
            tokens,
            dim,
        );

        let mut kernel_ir = aura_flash_sdpa_kb4_vb2_d128::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("q_rot", pack_f32(&q_rot), DType::F32))
            .input(TestBuffer::from_vec("key_packed", pack_u32(&key_packed), DType::U32))
            .input(TestBuffer::from_vec("key_norms", pack_f32(&key_norms), DType::F32))
            .input(TestBuffer::from_vec("key_codebook", pack_f32(&key_codebook), DType::F32))
            .input(TestBuffer::from_vec("val_packed", pack_u32(&val_packed), DType::U32))
            .input(TestBuffer::from_vec("val_norms", pack_f32(&val_norms), DType::F32))
            .input(TestBuffer::from_vec("val_codebook", pack_f32(&val_codebook), DType::F32))
            .input(TestBuffer::from_vec("sinks", pack_f32(&sinks), DType::F32))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec(
                "key_packed_width",
                u32_le(key_packed_width as u32),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "value_packed_width",
                u32_le(value_packed_width as u32),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("kv_stride", u32_le(kv_stride as u32), DType::U32))
            .input(TestBuffer::from_vec("repeat_count", u32_le(repeat as u32), DType::U32))
            .input(TestBuffer::from_vec("num_q_heads", u32_le(q_heads as u32), DType::U32))
            .input(TestBuffer::from_vec("has_sinks", u32_le(has_sinks), DType::U32))
            .input(TestBuffer::from_vec("window_size", u32_le(window_size), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .grid_3d(32, q_heads as u32, 1, [32, 1, 1])
    }
}

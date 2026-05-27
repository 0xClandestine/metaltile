//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA Flash Pass 1 — per-block online-softmax over the AURA-encoded
//! K and V caches.  The hot path: runs every decode token.
//!
//! Each threadgroup processes one (q_head, k_block) pair across 32
//! lanes.  Per-lane stack arrays cache the rotated query slice and the
//! online-softmax output accumulator across the per-token inner loop;
//! a second pair of stack arrays caches the K-side and V-side codebooks
//! so the inner loop only does a table lookup, not a global memory
//! fetch.
//!
//! Port of `turbo_flash_p1` (non-causal variant) from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_flash.metal`.
//!
//! ## Layout
//!
//! Inputs:
//! - `q_rot         [q_heads, dim]`                            f32
//! - `key_packed    [kv_heads, tokens, key_packed_width]`      u32
//! - `key_norms     [kv_heads, tokens]`                        f32
//! - `key_codebook  [2**key_bits]`                             f32
//! - `val_packed    [kv_heads, tokens, val_packed_width]`      u32
//! - `val_norms     [kv_heads, tokens]`                        f32
//! - `val_codebook  [2**val_bits]`                             f32
//!
//! Outputs:
//! - `o_partials    [q_heads, num_blocks, dim]`                f32
//! - `m_partials    [q_heads, num_blocks]`                     f32
//! - `l_partials    [q_heads, num_blocks]`                     f32
//!
//! `aura_flash_pass2` later reduces the partials cross-block.
//!
//! ## Dispatch
//!
//! Grid3D: (lane, q_idx, block_idx).  Threadgroup-internal lane
//! grouping (32 lanes) provides the simdgroup that `simd_sum` reduces
//! across for the Q · K dot product.
//!
//! ## Constexpr params
//!
//! - `key_bits`        — AURA K-side bit-width (2 / 3 / 4 / 8).
//! - `value_bits`      — AURA V-side bit-width.
//! - `dim`             — head_dim (64 / 80 / 96 / 128 / 256 / 512).
//! - `key_packed_width / value_packed_width` —
//!   `ceil(dim * bits / 32)`.
//! - `key_levels / value_levels` — `1 << bits`.
//! - `dims_per_lane`   — `ceil(dim / 32)`.
//!
//! Today's instantiation: `(key_bits=4, value_bits=2, dim=128)` — the
//! `aura4v2` scheme on a Qwen3-style head_dim=128.  Extend the
//! invocations at the bottom of the file for new (kb, vb, dim) combos.
//!
//! ## Bounds checking the per-lane dim slots
//!
//! Each inner loop walks dim slots via
//! `for i in 0..dims_per_lane { let d = lane + i*32; … }`.  When dim
//! isn't a multiple of 32 (e.g. dim=80 with `dims_per_lane=3` and
//! `max_d = 31 + 2*32 = 95 > 80`), the trailing lanes must skip the
//! out-of-range dim slots.  An earlier version of this kernel dropped
//! the `if d < dim { … }` guard to work around a metaltile unroll-pass
//! bug (nested `Op::If` bodies weren't being cloned + SSA-remapped
//! per iteration), but that limited us to multiple-of-32 dims.  The
//! unroll-pass fix landed alongside this kernel, so the guards are
//! back in.

use metaltile::kernel;

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestBuffer, TestSetup}};

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }
    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn pack_int_indices(
        indices: &[u32], kv_heads: usize, tokens: usize, dim: usize, bits: usize,
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

    /// CPU online-softmax reference for P1 partials.
    fn naive_aura_flash_p1(
        q_rot: &[f32], key_idx: &[u32], val_idx: &[u32],
        key_norms: &[f32], val_norms: &[f32], key_cb: &[f32], val_cb: &[f32],
        q_heads: usize, kv_heads: usize, tokens: usize, dim: usize,
        block_size: usize, q_position: usize, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let repeat = q_heads / kv_heads;
        let num_blocks = tokens.div_ceil(block_size);
        let mut o_p = vec![0.0_f32; q_heads * num_blocks * dim];
        let mut m_p = vec![f32::NEG_INFINITY; q_heads * num_blocks];
        let mut l_p = vec![0.0_f32; q_heads * num_blocks];
        for qh in 0..q_heads {
            let kvh = qh / repeat;
            for blk in 0..num_blocks {
                let t_start = blk * block_size;
                let t_end_raw = t_start + block_size;
                let clamped = t_end_raw.min(tokens);
                let t_end = if causal { clamped.min(q_position + 1) } else { clamped };
                let mut m_acc = f32::NEG_INFINITY;
                let mut l_acc = 0.0_f32;
                let mut o_acc = vec![0.0_f32; dim];
                for t in t_start..t_end {
                    let mut dot = 0.0_f32;
                    for d in 0..dim {
                        let q = key_idx[(kvh * tokens + t) * dim + d];
                        dot += q_rot[qh * dim + d] * key_cb[q as usize];
                    }
                    let score = dot * key_norms[kvh * tokens + t];
                    let new_m = m_acc.max(score);
                    let exp_diff = (m_acc - new_m).exp();
                    let exp_score = (score - new_m).exp();
                    let v_norm = val_norms[kvh * tokens + t];
                    for d in 0..dim {
                        let v = val_idx[(kvh * tokens + t) * dim + d];
                        o_acc[d] = o_acc[d] * exp_diff + exp_score * val_cb[v as usize] * v_norm;
                    }
                    l_acc = l_acc * exp_diff + exp_score;
                    m_acc = new_m;
                }
                let base = (qh * num_blocks + blk) * dim;
                for d in 0..dim { o_p[base + d] = o_acc[d]; }
                m_p[qh * num_blocks + blk] = m_acc;
                l_p[qh * num_blocks + blk] = l_acc;
            }
        }
        (o_p, m_p, l_p)
    }

    fn build_inputs(
        q_heads: usize, kv_heads: usize, tokens: usize, dim: usize,
        key_bits: usize, value_bits: usize,
    ) -> (Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
          Vec<u32>, Vec<u32>) {
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
        (q_rot, key_packed, val_packed, key_norms, val_norms, key_codebook, val_codebook,
         key_indices, val_indices)
    }

    #[test_kernel(name = "ffai/aura/flash_p1_kb4_vb2_d128", dtypes = [f32], tol = 1e-4)]
    fn test_aura_flash_p1(dt: DType) -> TestSetup {
        use super::aura_flash_p1_kb4_vb2_d128;
        let dim = 128usize;
        let key_bits = 4usize;
        let value_bits = 2usize;
        let key_packed_width = (dim * key_bits).div_ceil(32);
        let value_packed_width = (dim * value_bits).div_ceil(32);
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let tokens = 8usize;
        let block_size = 4usize;
        let num_blocks = tokens.div_ceil(block_size);
        let repeat = q_heads / kv_heads;

        let (q_rot, key_packed, val_packed, key_norms, val_norms, key_cb, val_cb,
             key_indices, val_indices) =
            build_inputs(q_heads, kv_heads, tokens, dim, key_bits, value_bits);

        let (exp_o, exp_m, exp_l) = naive_aura_flash_p1(
            &q_rot, &key_indices, &val_indices, &key_norms, &val_norms, &key_cb, &val_cb,
            q_heads, kv_heads, tokens, dim, block_size, tokens - 1, false,
        );

        let mut kernel_ir = aura_flash_p1_kb4_vb2_d128::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("q_rot", pack_f32(&q_rot), DType::F32))
            .input(TestBuffer::from_vec("key_packed", pack_u32(&key_packed), DType::U32))
            .input(TestBuffer::from_vec("key_norms", pack_f32(&key_norms), DType::F32))
            .input(TestBuffer::from_vec("key_codebook", pack_f32(&key_cb), DType::F32))
            .input(TestBuffer::from_vec("val_packed", pack_u32(&val_packed), DType::U32))
            .input(TestBuffer::from_vec("val_norms", pack_f32(&val_norms), DType::F32))
            .input(TestBuffer::from_vec("val_codebook", pack_f32(&val_cb), DType::F32))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec("key_packed_width", u32_le(key_packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("value_packed_width", u32_le(value_packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("kv_stride", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("repeat_count", u32_le(repeat as u32), DType::U32))
            .input(TestBuffer::from_vec("num_blocks", u32_le(num_blocks as u32), DType::U32))
            .input(TestBuffer::from_vec("block_size", u32_le(block_size as u32), DType::U32))
            .input(TestBuffer::from_vec("q_position", u32_le((tokens - 1) as u32), DType::U32))
            .expect(TestBuffer::from_vec("o_partials", pack_f32(&exp_o), DType::F32))
            .expect(TestBuffer::from_vec("m_partials", pack_f32(&exp_m), DType::F32))
            .expect(TestBuffer::from_vec("l_partials", pack_f32(&exp_l), DType::F32))
            .grid_3d(1, q_heads as u32, num_blocks as u32, [32, 1, 1])
    }

    #[test_kernel(name = "ffai/aura/flash_p1_causal_kb4_vb2_d128", dtypes = [f32], tol = 1e-4)]
    fn test_aura_flash_p1_causal(dt: DType) -> TestSetup {
        use super::aura_flash_p1_causal_kb4_vb2_d128;
        let dim = 128usize;
        let key_bits = 4usize;
        let value_bits = 2usize;
        let key_packed_width = (dim * key_bits).div_ceil(32);
        let value_packed_width = (dim * value_bits).div_ceil(32);
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let tokens = 8usize;
        let block_size = 4usize;
        let num_blocks = tokens.div_ceil(block_size);
        let repeat = q_heads / kv_heads;
        // q_position = 3: block 0 (tokens 0..4) is fully visible; block 1 (tokens 4..8) is masked.
        let q_position = 3usize;

        let (q_rot, key_packed, val_packed, key_norms, val_norms, key_cb, val_cb,
             key_indices, val_indices) =
            build_inputs(q_heads, kv_heads, tokens, dim, key_bits, value_bits);

        let (exp_o, exp_m, exp_l) = naive_aura_flash_p1(
            &q_rot, &key_indices, &val_indices, &key_norms, &val_norms, &key_cb, &val_cb,
            q_heads, kv_heads, tokens, dim, block_size, q_position, true,
        );

        let mut kernel_ir = aura_flash_p1_causal_kb4_vb2_d128::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("q_rot", pack_f32(&q_rot), DType::F32))
            .input(TestBuffer::from_vec("key_packed", pack_u32(&key_packed), DType::U32))
            .input(TestBuffer::from_vec("key_norms", pack_f32(&key_norms), DType::F32))
            .input(TestBuffer::from_vec("key_codebook", pack_f32(&key_cb), DType::F32))
            .input(TestBuffer::from_vec("val_packed", pack_u32(&val_packed), DType::U32))
            .input(TestBuffer::from_vec("val_norms", pack_f32(&val_norms), DType::F32))
            .input(TestBuffer::from_vec("val_codebook", pack_f32(&val_cb), DType::F32))
            .input(TestBuffer::from_vec("dim", u32_le(dim as u32), DType::U32))
            .input(TestBuffer::from_vec("key_packed_width", u32_le(key_packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("value_packed_width", u32_le(value_packed_width as u32), DType::U32))
            .input(TestBuffer::from_vec("tokens", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("kv_stride", u32_le(tokens as u32), DType::U32))
            .input(TestBuffer::from_vec("repeat_count", u32_le(repeat as u32), DType::U32))
            .input(TestBuffer::from_vec("num_blocks", u32_le(num_blocks as u32), DType::U32))
            .input(TestBuffer::from_vec("block_size", u32_le(block_size as u32), DType::U32))
            .input(TestBuffer::from_vec("q_position", u32_le(q_position as u32), DType::U32))
            .expect(TestBuffer::from_vec("o_partials", pack_f32(&exp_o), DType::F32))
            .expect(TestBuffer::from_vec("m_partials", pack_f32(&exp_m), DType::F32))
            .expect(TestBuffer::from_vec("l_partials", pack_f32(&exp_l), DType::F32))
            .grid_3d(1, q_heads as u32, num_blocks as u32, [32, 1, 1])
    }
}

#[rustfmt::skip]
macro_rules! aura_flash_p1_kernel {
    (
        $name:ident,
        $key_bits:literal,
        $value_bits:literal,
        $key_levels:literal,
        $value_levels:literal,
        $dims_per_lane:literal,
        $causal:literal,
        $subop:literal
    ) => {
        #[kernel]
        pub fn $name<T>(
            q_rot: Tensor<T>,
            key_packed: Tensor<u32>,
            key_norms: Tensor<T>,
            key_codebook: Tensor<T>,
            val_packed: Tensor<u32>,
            val_norms: Tensor<T>,
            val_codebook: Tensor<T>,
            mut o_partials: Tensor<T>,
            mut m_partials: Tensor<T>,
            mut l_partials: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] key_packed_width: u32,
            #[constexpr] value_packed_width: u32,
            #[constexpr] tokens: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] repeat_count: u32,
            #[constexpr] num_blocks: u32,
            #[constexpr] block_size: u32,
            // Global position of this query token in the KV stream. Only
            // consulted by the causal variant (`$causal == 1`): keys at
            // token index `t > q_position` are masked out. The non-causal
            // variant ignores it (constexpr, so the dead branch is folded
            // away — no runtime cost).
            #[constexpr] q_position: u32,
        ) {
            let lane = program_id::<0>();
            let q_idx = program_id::<1>();
            let block_idx = program_id::<2>();
            let kv_idx = q_idx / repeat_count;

            let key_mask = (1u32 << $key_bits) - 1u32;
            let val_mask = (1u32 << $value_bits) - 1u32;

            let raw_end = block_idx * block_size + block_size;
            let clamped_end = select(raw_end > tokens, tokens, raw_end);
            // Causal cutoff: tokens strictly after `q_position` contribute
            // nothing, so the inner loop can stop at `q_position + 1`. For
            // the non-causal variant `$causal == 0` makes this a no-op
            // (the macro substitutes the literal at compile time).
            let causal_end = select($causal == 1u32, q_position + 1u32, clamped_end);
            let t_end = select(causal_end < clamped_end, causal_end, clamped_end);
            let t_start = block_idx * block_size;

            // ── Cache codebooks in per-thread stack arrays.  Each lane
            // touches the same codebook; the cache amortises lookups
            // across the inner per-token loop.
            stack_alloc("key_cb", $key_levels, "f32");
            for i in range(0u32, $key_levels, 1u32) {
                stack_store("key_cb", i, load(key_codebook[i]).cast::<f32>());
            }
            stack_alloc("val_cb", $value_levels, "f32");
            for i in range(0u32, $value_levels, 1u32) {
                stack_store("val_cb", i, load(val_codebook[i]).cast::<f32>());
            }

            // ── Per-lane slice of the rotated query vector — held in
            // stack registers, loaded once.  Trailing lanes whose
            // `d >= dim` get zero so the dot product treats them as a
            // no-op. Loaded as T and promoted to f32 for compute.
            stack_alloc("q_vals", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                let v = select(d < dim, load(q_rot[q_idx * dim + d]).cast::<f32>(), 0.0f32);
                stack_store("q_vals", i, v);
            }

            // ── Online-softmax accumulators.  `m` is the running max,
            // `l` the running sum_exp, `o[]` the un-normalised output
            // slice for this lane.
            let mut m_acc = neg_infinity();
            let mut l_acc = 0.0f32;
            stack_alloc("o", $dims_per_lane, "f32");
            for i in range(0u32, $dims_per_lane, 1u32) {
                stack_store("o", i, 0.0f32);
            }

            // ── Per-token inner loop ───────────────────────────────────
            for t in range(t_start, t_end, 1u32) {
                // Row stride is `kv_stride` (cache's `maxSeq`), not `tokens`
                // (live KV-row count). When the cache isn't fully populated,
                // head 1 starts at byte offset `kv_stride`, NOT `tokens` —
                // otherwise we'd read head 0's tail bytes as head 1's rows.
                let k_packed_row = (kv_idx * kv_stride + t) * key_packed_width;
                let k_norm = load(key_norms[kv_idx * kv_stride + t]).cast::<f32>();

                // Q · K via compressed-domain dot — bit-extract per dim,
                // lookup centroid in cached key_cb, accumulate against the
                // pre-loaded q_vals slice, simd_sum across the lane group.
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

                // Online-softmax max-shift identity.
                let new_m = select(m_acc > score, m_acc, score);
                let exp_diff = exp(m_acc - new_m);
                let exp_score = exp(score - new_m);

                // V-side update: bit-extract each value, look up in the
                // cached val_cb, scale by exp_score · v_norm, fold into
                // the running output via the standard online-softmax
                // rescale-then-add.
                let v_packed_row = (kv_idx * kv_stride + t) * value_packed_width;
                let v_norm = load(val_norms[kv_idx * kv_stride + t]).cast::<f32>();

                for i in range(0u32, $dims_per_lane, 1u32) {
                    let d = lane + i * 32u32;
                    if d < dim {
                        let bit_offset = d * $value_bits;
                        let word_idx = bit_offset / 32u32;
                        let shift = bit_offset & 31u32;
                        let bits_in_w0 = 32u32 - shift;
                        let lo_bits = select(bits_in_w0 >= $value_bits, $value_bits, bits_in_w0);
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

            // ── Write per-block partials (cast f32 → T on store) ───────
            let partial_base = (q_idx * num_blocks + block_idx) * dim;
            for i in range(0u32, $dims_per_lane, 1u32) {
                let d = lane + i * 32u32;
                if d < dim {
                    store(o_partials[partial_base + d], stack_load("o", i).cast::<T>());
                }
            }
            if lane == 0u32 {
                let ml_idx = q_idx * num_blocks + block_idx;
                store(m_partials[ml_idx], m_acc.cast::<T>());
                store(l_partials[ml_idx], l_acc.cast::<T>());
            }
        }
    };
}

// Production (kb, vb, dim) instantiations. The macro is parametric;
// adding a row generates one more dispatchable kernel.
//
//   dims_per_lane = ceil(dim / 32)
//   {kb,vb}_levels = 2^{kb,vb}
//
// Coverage today:
//   - head_dim=128: covers Qwen3, Llama 3.2 3B+, GPT-OSS full-attn layers
//   - head_dim=64:  covers Llama 3.2 1B and GPT-OSS sliding-window layers
//
// Symmetric (kb=vb=4) is the AURAScheme.default (aura4v4) — stability-
// first. Asymmetric kb=4 vb=2 is the production recipe aura4v2 — ~5×
// compression vs fp16 per `papers/aura-compression-algorithm.md` §2.5.
//
// Other dims (80, 96, 192, 256) + other recipes (aura8, aura3) queued
// behind a real consumer — adding more variants now is `make
// emit-all` weight bloat without a use site.
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb2_d128,
    4u32,
    2u32,
    16u32,
    4u32,
    4u32,
    0u32,
    "flash_p1_kb4_vb2_d128"
);
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb4_d128,
    4u32,
    4u32,
    16u32,
    16u32,
    4u32,
    0u32,
    "flash_p1_kb4_vb4_d128"
);
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb2_d64,
    4u32,
    2u32,
    16u32,
    4u32,
    2u32,
    0u32,
    "flash_p1_kb4_vb2_d64"
);
aura_flash_p1_kernel!(
    aura_flash_p1_kb4_vb4_d64,
    4u32,
    4u32,
    16u32,
    16u32,
    2u32,
    0u32,
    "flash_p1_kb4_vb4_d64"
);

// ── Causal variants ──────────────────────────────────────────────────────
//
// Same compressed-domain online-softmax as the non-causal kernels, with
// the per-token loop clamped at `q_position + 1` — every key strictly
// after the query token is masked out. This is the prefill / chunked
// form upstream's `turbo_flash_p1` carries as the `causal` template
// flag. The `$causal == 1` literal lets the codegen const-fold the
// `causal_end` selection, so the only runtime difference vs the
// non-causal sibling is the inner-loop trip count.
//
// Production recipe `aura4v2` (kb=4, vb=2) for the two head dims FFAI
// ships today; the symmetric `aura4v4` causal variant follows the same
// macro arm if a consumer needs it.
aura_flash_p1_kernel!(
    aura_flash_p1_causal_kb4_vb2_d128,
    4u32,
    2u32,
    16u32,
    4u32,
    4u32,
    1u32,
    "flash_p1_causal_kb4_vb2_d128"
);
aura_flash_p1_kernel!(
    aura_flash_p1_causal_kb4_vb2_d64,
    4u32,
    2u32,
    16u32,
    4u32,
    2u32,
    1u32,
    "flash_p1_causal_kb4_vb2_d64"
);

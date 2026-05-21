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
//! `aura_flash_sdpa` (token-parallelism is a perf follow-up). Bool/float
//! attention masks from the MLX kernel are not ported — causal + sinks
//! + sliding-window cover the decode shapes.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, B*nQ, 1]`, `tg = [32, 1, 1]`.
//! - `dims_per_lane = ceil(dim / 32)`; `dim` a multiple of `32/bits`.
//!
//! Codegen-only; correctness pinned by
//! `tests/flash_quantized_sdpa_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

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

        inventory::submit! {
            BenchSpec {
                op: "flash_quantized_sdpa",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[DType::F32, DType::F16, DType::BF16],
                tol: 1e-3,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Grid3D),
            }
        }
    };
}

flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d64, 4u32, 64u32, 2u32, "b4_d64");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d128, 4u32, 128u32, 4u32, "b4_d128");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b4_d256, 4u32, 256u32, 8u32, "b4_d256");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d64, 8u32, 64u32, 2u32, "b8_d64");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d128, 8u32, 128u32, 4u32, "b8_d128");
flash_quantized_sdpa_kernel!(flash_quantized_sdpa_b8_d256, 8u32, 256u32, 8u32, "b8_d256");

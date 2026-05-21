//! Batched Q/K/V 4-bit quantized GEMV — fuses the three independent
//! Q, K, V projection matvecs of a decode step into one dispatch.
//!
//! The `z` grid axis selects the matrix (`program_id::<2>()`:
//! 0 = Q, 1 = K, 2 = V); the `x` grid axis is the output row. One
//! threadgroup computes one `(matrix, row)` output. The result lands in
//! a single contiguous `y` of length `out_q + out_k + out_v`, with Q,
//! K, V concatenated in that order.
//!
//! Each matrix's GEMV is the pack-strided int4 shape of
//! `dequant_gemv_int4`. The three branches are spelled out because the
//! `#[kernel]` DSL has no function-call mechanism — the only
//! per-matrix differences are the weight/scale/bias tensors and the
//! output offset.
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   w_*       [out_*, N/8]   uint32
//!   scales_*  [out_*, N/G]   T
//!   biases_*  [out_*, N/G]   T
//!   x         [N]            T
//!   y         [out_q+out_k+out_v] T
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid: `[max(out_q, out_k, out_v), 1, 3]`** — `program_id::<0>()`
//!   = output row, `program_id::<2>()` = matrix. Rows past a matrix's
//!   `out_*` no-op.
//! - **TPG a multiple of 32, ≥ 32** (reduction kernel).
//! - `in_dim` a multiple of 8 and of `group_size`.
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_qkv_qgemv_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Fused Q/K/V int4 quantized GEMV. `program_id::<2>()` picks the matrix.
#[kernel]
pub fn ffai_batched_qkv_qgemv<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<T>,
    biases_q: Tensor<T>,
    w_k: Tensor<u32>,
    scales_k: Tensor<T>,
    biases_k: Tensor<T>,
    w_v: Tensor<u32>,
    scales_v: Tensor<T>,
    biases_v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let vals_per_pack = 8u32; // 32 / 4 bits
    let mask = 15u32;
    let n_packs = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_group_off = row * n_groups;

    if matrix == 0u32 {
        if row < out_q {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_q[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_q[row_group_off + g]).cast::<f32>();
                    let packed = load(w_q[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_q = reduce_sum(acc);
            if tid == 0u32 {
                store(out[row], total_q.cast::<T>());
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_k[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_k[row_group_off + g]).cast::<f32>();
                    let packed = load(w_k[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_k = reduce_sum(acc);
            if tid == 0u32 {
                store(out[out_q + row], total_k.cast::<T>());
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            let mut acc = 0.0f32;
            for _p in range(0u32, p_iters, 1u32) {
                let pack_idx = _p * lsize + tid;
                if pack_idx < n_packs {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales_v[row_group_off + g]).cast::<f32>();
                    let bias = load(biases_v[row_group_off + g]).cast::<f32>();
                    let packed = load(w_v[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;
                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * 4u32)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias) * load(x[p_off + i]).cast::<f32>();
                    }
                }
            }
            let total_v = reduce_sum(acc);
            if tid == 0u32 {
                store(out[out_q + out_k + row], total_v.cast::<T>());
            }
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "batched_qkv_qgemv",
        subop: "batched_qkv_qgemv",
        kernel_name: "ffai_batched_qkv_qgemv",
        kernel_ir: ffai_batched_qkv_qgemv::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

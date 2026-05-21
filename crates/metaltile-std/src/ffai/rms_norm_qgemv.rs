//! Fused RMSNorm + 4-bit quantized GEMV for decode (single-token).
//!
//! Computes `y = qmatmul(rms_norm(x) * norm_weight, W_q)` in one
//! dispatch, eliminating the global-memory round-trip of the normalized
//! activation between a standalone `rms_norm` and a quantized matvec.
//!
//! Reduction-mode: one threadgroup per output row. Phase 1 reduces
//! `sum(x²)` across the threadgroup → `inv_rms`; phase 2 is a
//! pack-strided int4 GEMV (the `dequant_gemv_int4` shape) that feeds on
//! `normed[i] = x[i] * norm_weight[i] * inv_rms` instead of raw `x`, so
//! the normalized activation never leaves registers.
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   weight  [out_dim, N/8]    uint32  (8 int4 values per u32)
//!   scales  [out_dim, N/G]    T
//!   biases  [out_dim, N/G]    T
//!   x, norm_weight  [N]       T
//!   y               [out_dim] T
//!
//! The MLX reference tiles 8 output rows per threadgroup (2 simdgroups ×
//! 4 results) to amortize the RMSNorm and the shared-memory x load. This
//! port keeps the simpler one-row-per-threadgroup shape — still removes
//! the normalized-activation round-trip; the multi-row tiling is a perf
//! follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid: 1 threadgroup per output row** — `program_id::<0>()` = row.
//! - **TPG a multiple of 32, ≥ 32** (reduction kernel).
//! - `in_dim` a multiple of 8 (int4 pack factor) and of `group_size`.
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_qgemv_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// `y[row] = Σ_i (q[row,i]·scale + bias) · (x[i]·norm_weight[i]·inv_rms)`,
/// with `inv_rms = rsqrt(mean(x²) + eps)`, weights int4-packed.
#[kernel]
pub fn ffai_rms_norm_qgemv<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();

    // Phase 1: RMSNorm — threadgroup-wide sum of squares over x.
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let tg_ssq = reduce_sum(ssq);
    let eps = load(eps_buf[0]);
    let inv_rms = rsqrt(tg_ssq / in_dim + eps);

    // Phase 2: pack-strided int4 GEMV over the normalized activation.
    let vals_per_pack = 8u32; // 32 / 4 bits
    let mask = 15u32;
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for _p in range(0u32, p_iters, 1u32) {
        let pack_idx = _p * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let xi = load(x[p_off + i]).cast::<f32>();
                let nw = load(norm_weight[p_off + i]).cast::<f32>();
                let normed = xi * nw * inv_rms;
                acc = acc + (q.cast::<f32>() * scale + bias) * normed;
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "rms_norm_qgemv",
        subop: "rms_norm_qgemv",
        kernel_name: "ffai_rms_norm_qgemv",
        kernel_ir: ffai_rms_norm_qgemv::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

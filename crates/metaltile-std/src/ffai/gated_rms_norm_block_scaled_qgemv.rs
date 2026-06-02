//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **gated-RMSNorm + block-scaled dequantizing GEMV** for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! `out = qmatmul(gated_rms_norm(y, z) · norm_weight, W_q)` in one dispatch —
//! the gated-RMSNorm staging of `ffai/gated_rms_norm_qgemv.rs` fused with the
//! block-scaled weight decode of `mlx/block_scaled_matmul.rs`.
//!
//! Phase 1 stages `inner[r,d] = y[r,d] · rsqrt(mean_d(y[r]²)+eps) ·
//! norm_weight[d] · silu(z[r,d])` into a `tg_inner` threadgroup buffer
//! (fp32), via the proven 2-simdgroup per-row scheme (`sg=0` even rows,
//! `sg=1` odd rows; one `simd_sum` per row). Phase 2 is the simple
//! one-output-row-per-TG block-scaled reduction GEMV reading `tg_inner`.
//! The staging is **weight-format-independent**, so it is identical across
//! all five formats — only phase 2's decode differs.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [64, 1, 1]`
//!   (2 simdgroups × 32 lanes — required by the phase-1 staging).
//! - `dv` a multiple of 32; `hv` even; `in_dim = hv·dv`; `in_dim` a multiple
//!   of `block_size`; 4-bit `block_size` a multiple of 8.
//! - `y` `[hv,dv]` fp32; `z` `[hv,dv]`; `norm_weight` `[dv]`; weight
//!   `[out_dim, in_dim/8]` u32 (4-bit) or `[out_dim, in_dim]` u8 (8-bit);
//!   scales `[out_dim, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!
//! Block-scaled formats carry no bias. Codegen-only; correctness pinned by
//! the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 fused gated-RMSNorm + GEMV — E2M1 weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp4_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    // Phase 1: gated RMSNorm staged into tg_inner (2-simdgroup per-row scheme).
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    // Phase 2: block-scaled E2M1 GEMV over tg_inner (one output row per TG).
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;
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
                let inner = threadgroup_load("tg_inner", p_off + i);
                acc = acc + (val * scale) * inner;
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// nvfp4 fused gated-RMSNorm + GEMV — E2M1 (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;
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
                let inner = threadgroup_load("tg_inner", p_off + i);
                acc = acc + (val * scale) * inner;
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// mxfp8 (E4M3) fused gated-RMSNorm + GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let inner = threadgroup_load("tg_inner", c);
            acc = acc + (elem * scale) * inner;
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// mxfp8 (E5M2) fused gated-RMSNorm + GEMV — 8-bit weights (block 32), E8M0 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let sbits = load(scales[row_block_off + c / block_size]).cast::<f32>();
            let scale = exp2(sbits - 127.0f32);
            let inner = threadgroup_load("tg_inner", c);
            acc = acc + (elem * scale) * inner;
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// nvfp8 fused gated-RMSNorm + GEMV — E4M3 weights (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e4m3_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            let inner = threadgroup_load("tg_inner", c);
            acc = acc + (elem * scale) * inner;
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 fused GEMVs ─────────────
// These share the fused gated-RMSNorm + block-scaled framework but store a raw
// per-group FP32 scale (no E8M0/E4M3/global). fp8_e4m3 has the same shape as
// nvfp8 (8-bit E4M3 + f32 scale), so it reuses `mt_nvfp8_gated_rms_norm_qgemv`
// — only fp4 (4-bit E2M1), fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric)
// need their own decode here. Phase 1 is identical across all formats.

/// Legacy fp4 fused gated-RMSNorm + GEMV — E2M1 weights (group 32), FP32 scale.
#[kernel]
pub fn mt_fp4_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let n_packs_per_row = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_block_off = row * n_blocks;
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
                let inner = threadgroup_load("tg_inner", p_off + i);
                acc = acc + (val * scale) * inner;
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Legacy fp8 (E5M2) fused gated-RMSNorm + GEMV — 8-bit weights (group 32),
/// per-group FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = e5m2_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            let inner = threadgroup_load("tg_inner", c);
            acc = acc + (elem * scale) * inner;
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Symmetric int8 fused gated-RMSNorm + GEMV — 8-bit codes (group 64), per-group
/// FP32 scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_gated_rms_norm_qgemv<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    let row = program_id::<0>();
    let in_dim = hv * dv;
    let row_off = row * in_dim;
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;
    let iters = (in_dim + lsize - 1u32) / lsize;
    for it in range(0u32, iters, 1u32) {
        let c = it * lsize + tid;
        if c < in_dim {
            let elem = int8_decode(load(weight[row_off + c]).cast::<u32>());
            let scale = load(scales[row_block_off + c / block_size]);
            let inner = threadgroup_load("tg_inner", c);
            acc = acc + (elem * scale) * inner;
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{quant::format::QFormat, utils::pack_f32};

    const TPG: u32 = 64;
    const EPS: f32 = 1e-5;

    /// Deterministic xorshift source (matches the int4 gated test generator).
    fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
            })
            .collect()
    }

    fn round(v: &[f32], dt: DType) -> Vec<f32> { crate::utils::unpack_f32(&pack_f32(v, dt), dt) }

    /// Gated-RMSNorm staging (CPU) → `inner[r,d]`, identical to the kernel's
    /// phase 1 (`y` fp32; `z`/`norm_weight` rounded to `dt`).
    fn stage_inner(y: &[f32], z: &[f32], nw: &[f32], hv: usize, dv: usize) -> Vec<f32> {
        let mut inner = vec![0.0f32; hv * dv];
        for r in 0..hv {
            let base = r * dv;
            let ssq: f32 = y[base..base + dv].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (ssq / dv as f32 + EPS).sqrt();
            for d in 0..dv {
                let g = z[base + d] / (1.0 + (-z[base + d]).exp());
                inner[base + d] = y[base + d] * inv_rms * nw[d] * g;
            }
        }
        inner
    }

    fn gated_setup(
        kernel: Kernel,
        fmt: QFormat,
        hv: usize,
        dv: usize,
        out_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let in_dim = hv * dv;
        // Block-scaled weights `[out_dim, in_dim]` via the shared codec.
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + r * 0.25) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, out_dim, in_dim);
        // y fp32; z / norm_weight rounded through dt (kernel loads them as T).
        let y = source(in_dim, 0xA1, 2.0, 0.1);
        let z = round(&source(in_dim, 0xD4, 1.5, 0.0), dt);
        let nw = round(&source(dv, 0xB2, 0.4, 1.0), dt);
        let inner = stage_inner(&y, &z, &nw, hv, dv);
        let expected: Vec<f32> = (0..out_dim)
            .map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * inner[c]).sum())
            .collect();
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("y", pack_f32(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack_f32(&z, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&nw, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", EPS.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", out_dim, dt))
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            out_dim as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // hv=4, dv=128, in_dim=512 (÷ 16/32), out_dim=4.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_mxfp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_nvfp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxfp8_e4m3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxfp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4, 128, 4, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_fp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_fp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int8_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8, 4, 128, 4, dt)
    }
}

/// Decode-shape benches at the Qwen3.6-A3B activation shape (hv=16, dv=128,
/// in_dim=2048, out=2048). Throughput is data-independent → random buffers.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn gated_bench(
        kernel: Kernel,
        fmt: QFormat,
        hv: usize,
        dv: usize,
        out_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let in_dim = hv * dv;
        let n_blocks = out_dim * (in_dim / fmt.block_size());
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (out_dim * in_dim / 8, DType::U32)
        } else {
            (out_dim * in_dim, DType::U8)
        };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + in_dim * 4   // y (fp32)
            + in_dim * sz  // z
            + dv * sz      // norm_weight
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("y", in_dim, DType::F32))
            .buffer(BenchBuffer::random("z", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", dv, dt))
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", out_dim, dt).output())
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64)
            .with_shape_label(format!("{} hv={hv} dv={dv} m={out_dim}", fmt.name()))
    }

    #[bench(name = "ffai/gated_rms_norm_block_qgemv/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxfp4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxfp8_e4m3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxfp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_gated(dt: DType) -> BenchSetup {
        gated_bench(mt_fp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 16, 128, 2048, dt)
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_fp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(name = "ffai/gated_rms_norm_block_qgemv/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int8,
            16,
            128,
            2048,
            dt,
        )
    }
}

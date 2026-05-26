//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-token SDPA decode for `head_dim == 512`. Parallel of
//! `ffai_sdpa_decode_d256` with 16 elements per lane and a **4-phase
//! output reduction** to stay under Apple's 32 KB threadgroup-memory
//! cap.
//!
//! Needed for Gemma 4's global (`full_attention`) layers, which use
//! `global_head_dim = 512`. The sliding layers stay on
//! `ffai_sdpa_decode_d256`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG = 512 threads** (16 simdgroups × 32 lanes), NOT 1024 like
//!   the d64/d128/d256 variants. Each lane owns 16 consecutive Q/K/V
//!   elements (`head_dim / 32 = 512 / 32 = 16`), loaded unconditionally
//!   at `lane * 16 + {0..15}`. The 16-wide per-lane footprint (16 live
//!   Q + 16 live online-softmax accumulators) pushes the kernel's
//!   `maxTotalThreadsPerThreadgroup` below 1024, so a 1024-thread
//!   dispatch silently fails (command buffer errors, output stays
//!   zero). The kernel body is fully parametric in `n_simd = TPG / 32`,
//!   so any multiple of 32 within the pipeline cap is correct; the
//!   wrapper uses 512.
//! - **`head_dim == 512`.** Wrapper-enforced.
//! - **Grid: 1 threadgroup per q_head.** Wrapper uses
//!   `grid = (nQHeads * 512, 1, 1)`, `tg = (512, 1, 1)`.
//! - **`nQHeads % nKVHeads == 0`** (GQA fan-out is integer).
//! - **`n_kv ≤ kv_stride`** (cache walk stays within capacity).
//!
//! ## Why 4-phase output reduction
//!
//! d=256 stores 8 per-lane output dims via two phases over 4 tg_outN
//! buffers of `n_lanes * (n_simd + 1) = 32 * 33 = 1056` floats each
//! (`+1` is bank-conflict padding). 4 × 1056 ≈ 16 KB.
//!
//! d=512 has 16 per-lane output dims. Materialising all 16 at once
//! would need 16 × 1056 ≈ 66 KB — far over Apple's per-kernel
//! threadgroup-memory cap. We split into four quartets (dims 0..3,
//! 4..7, 8..11, 12..15) and reuse the same 4 tg_out buffers across
//! all four phases. Six extra barriers, same ~16 KB allocation.
//!
//! Wrapping doc: see FFAI/CLAUDE.md §"Wrapping kernels in FFAI".

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sdpa",
    subop="sdpa_decode_d512",
    class=GenericEmpty,
    tol=1e-3,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_sdpa_decode_d512<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // Four-phase reduction: 4 tg_outN buffers, reused for dims (0..3),
    // (4..7), (8..11) then (12..15). 1056 = n_lanes * (n_simd + 1)
    // with bank padding.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 16u32;
    // Pre-scale this lane's 16-element Q stripe once; K/V are streamed.
    // Two banks of 8 (UnrollPass max trip count = 8).
    stack_alloc("q_s", 16, "f32");
    for _i in range(0u32, 8u32, 1u32) {
        stack_store("q_s", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
    }
    for _i in range(0u32, 8u32, 1u32) {
        stack_store("q_s", _i + 8u32, load(q[q_off + d0 + _i + 8u32]).cast::<f32>() * scale);
    }
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    stack_alloc("o", 16, "f32");
    for _i in range(0u32, 8u32, 1u32) {
        stack_store("o", _i, 0.0f32);
    }
    for _i in range(0u32, 8u32, 1u32) {
        stack_store("o", _i + 8u32, 0.0f32);
    }
    for _t in range(sg, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        let mut partial = 0.0f32;
        for _i in range(0u32, 8u32, 1u32) {
            partial = partial + stack_load("q_s", _i) * load(k[kv0 + _i]).cast::<f32>();
        }
        for _i in range(0u32, 8u32, 1u32) {
            partial =
                partial + stack_load("q_s", _i + 8u32) * load(k[kv0 + _i + 8u32]).cast::<f32>();
        }
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        for _i in range(0u32, 8u32, 1u32) {
            stack_store(
                "o",
                _i,
                stack_load("o", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
            );
        }
        for _i in range(0u32, 8u32, 1u32) {
            stack_store(
                "o",
                _i + 8u32,
                stack_load("o", _i + 8u32) * factor
                    + weight * load(v[kv0 + _i + 8u32]).cast::<f32>(),
            );
        }
    }
    // ── Cross-simdgroup reduction: max + sum_exp ────────────────────
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    // ── Cross-simdgroup output reduction — phase 1 (dims 0..3) ─────
    // Transpose layout: idx = lane * stride + sg. After barrier, sg=0
    // lanes read back ns partials each for their own (lane, d) slot.
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, stack_load("o", 0u32) * rescale);
    threadgroup_store("tg_out1", idx, stack_load("o", 1u32) * rescale);
    threadgroup_store("tg_out2", idx, stack_load("o", 2u32) * rescale);
    threadgroup_store("tg_out3", idx, stack_load("o", 3u32) * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
            so3 = so3 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
        store(out[out_off + 2u32], so2.cast::<T>());
        store(out[out_off + 3u32], so3.cast::<T>());
    }
    threadgroup_barrier();
    // ── Cross-simdgroup output reduction — phase 2 (dims 4..7) ─────
    threadgroup_store("tg_out0", idx, stack_load("o", 4u32) * rescale);
    threadgroup_store("tg_out1", idx, stack_load("o", 5u32) * rescale);
    threadgroup_store("tg_out2", idx, stack_load("o", 6u32) * rescale);
    threadgroup_store("tg_out3", idx, stack_load("o", 7u32) * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so4 = 0.0f32;
        let mut so5 = 0.0f32;
        let mut so6 = 0.0f32;
        let mut so7 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so4 = so4 + threadgroup_load("tg_out0", ri);
            so5 = so5 + threadgroup_load("tg_out1", ri);
            so6 = so6 + threadgroup_load("tg_out2", ri);
            so7 = so7 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 4u32], so4.cast::<T>());
        store(out[out_off + 5u32], so5.cast::<T>());
        store(out[out_off + 6u32], so6.cast::<T>());
        store(out[out_off + 7u32], so7.cast::<T>());
    }
    threadgroup_barrier();
    // ── Cross-simdgroup output reduction — phase 3 (dims 8..11) ────
    threadgroup_store("tg_out0", idx, stack_load("o", 8u32) * rescale);
    threadgroup_store("tg_out1", idx, stack_load("o", 9u32) * rescale);
    threadgroup_store("tg_out2", idx, stack_load("o", 10u32) * rescale);
    threadgroup_store("tg_out3", idx, stack_load("o", 11u32) * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so8 = 0.0f32;
        let mut so9 = 0.0f32;
        let mut so10 = 0.0f32;
        let mut so11 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so8 = so8 + threadgroup_load("tg_out0", ri);
            so9 = so9 + threadgroup_load("tg_out1", ri);
            so10 = so10 + threadgroup_load("tg_out2", ri);
            so11 = so11 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 8u32], so8.cast::<T>());
        store(out[out_off + 9u32], so9.cast::<T>());
        store(out[out_off + 10u32], so10.cast::<T>());
        store(out[out_off + 11u32], so11.cast::<T>());
    }
    threadgroup_barrier();
    // ── Cross-simdgroup output reduction — phase 4 (dims 12..15) ───
    threadgroup_store("tg_out0", idx, stack_load("o", 12u32) * rescale);
    threadgroup_store("tg_out1", idx, stack_load("o", 13u32) * rescale);
    threadgroup_store("tg_out2", idx, stack_load("o", 14u32) * rescale);
    threadgroup_store("tg_out3", idx, stack_load("o", 15u32) * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so12 = 0.0f32;
        let mut so13 = 0.0f32;
        let mut so14 = 0.0f32;
        let mut so15 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so12 = so12 + threadgroup_load("tg_out0", ri);
            so13 = so13 + threadgroup_load("tg_out1", ri);
            so14 = so14 + threadgroup_load("tg_out2", ri);
            so15 = so15 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 12u32], so12.cast::<T>());
        store(out[out_off + 13u32], so13.cast::<T>());
        store(out[out_off + 14u32], so14.cast::<T>());
        store(out[out_off + 15u32], so15.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_sdpa_decode_d512;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_sdpa_decode_d512::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_sdpa_decode_d512 codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_sdpa_decode_d512"),
                "MSL for {dt:?} should declare ffai_sdpa_decode_d512:\n{src}",
            );
        }
    }
}

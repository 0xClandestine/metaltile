//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused MoE decode kernel: SwiGLU + 8-way indexed down-projection + scalar-FMA chain.
//!
//! Collapses the three back-to-back dispatches FFAI's MoE GPU-router
//! decode path runs per layer (under `FFAI_MOE_GPU_ROUTER`, default-on
//! as of ITER 80) into ONE kernel launch:
//!
//!   1. `mt_swiglu` (many=8): `inner[k][d] = silu(gate[k][d]) * up[k][d]`
//!   2. `ffai_dequant_gemv_int4_expert_indexed` (many=8): per slot k,
//!       `down_out[k] = W_down[expert[k]] · inner[k]`  (out_dim = hidden)
//!   3. `mt_scalar_fma_chain8`:
//!       `acc[i] = Σ_{k=0..8} scalar[k] * down_out[k][i]`
//!
//! The three stages have a strict data dependency chain so they MUST
//! run on separate encoders today. Fusing them eliminates two encoder
//! begin/end pairs + the global memory round-trip on `inner` (8 × 768
//! floats per layer × 40 layers = ~960 KB DRAM/token at Qwen3.6-A3B)
//! AND removes the materialisation of `down_out` (8 × 2048 floats per
//! layer × 40 layers = ~2.5 MB DRAM/token).
//!
//! ## Geometry
//!
//! - One threadgroup per output row of the down projection.
//!     grid       = MTLSize(out_dim, 1, 1)        // out_dim = hidden
//!     threadgroup = MTLSize(lsize, 1, 1)         // caller picks (typ. 128)
//!
//! - Each TG iterates the 8 slots sequentially:
//!     (a) cooperatively populate `tg_inner[d] = silu(gate[k][d]) * up[k][d]`
//!         for d in `[0, in_dim)` — every thread writes a strided slice.
//!     (b) threadgroup_barrier (RAW: qmm reads `tg_inner` filled by all
//!         lanes in (a)).
//!     (c) dequant-gemv inner loop against `W_down[expert[k]][row, :]`,
//!         each thread accumulating into a per-thread `acc` that runs
//!         across ALL slots (with slot scalar baked in at accumulation).
//!     (d) threadgroup_barrier (WAR: next slot's swiglu overwrites
//!         `tg_inner` and must not race with this slot's qmm reads).
//!   After 8 slots, `reduce_sum(acc)` and lane 0 stores to `out[row]`.
//!
//! ## Threadgroup memory
//!
//! `tg_inner: [IN_DIM_MAX] f32` — staged inner activations for the
//! currently-active slot. Reused across all 8 slots (no per-slot copy).
//!
//! IN_DIM_MAX is pinned at 768 (Qwen3.6-A3B `moeIntermediate`). At f32,
//! that's 3 KiB TG memory — comfortably below the 32 KiB Apple9 cap and
//! leaves headroom for concurrent TGs on the same simdgroup-block.
//! Caller MUST validate `in_dim <= 768`; the kernel reads only the first
//! `in_dim` entries so smaller intermediates work, but larger ones
//! would scribble past the alloc.
//!
//! ## ABI
//!
//! ```text
//!   gate_0..gate_7    [in_dim]                          T
//!   up_0..up_7        [in_dim]                          T
//!   expert_indices    [8]                               u32
//!   slot_weights      [8]                               T   (routing scalars)
//!   weights_stacked   [n_experts, out_dim, in_dim / 8]  u32 (int4-packed)
//!   scales_stacked    [n_experts, out_dim, in_dim / G]  T
//!   biases_stacked    [n_experts, out_dim, in_dim / G]  T
//!   out               [out_dim]                         T
//! ```
//!
//! `expert_indices` and `slot_weights` are the contiguous outputs of
//! `mt_moe_router_topk` (k=8, packed). The 16 gate/up tensors mirror
//! FFAI's per-slot scratch caches (one `Tensor.empty([moeIntermediate])`
//! per slot, instance-cached per `MoELayer` per the ITER 32-36
//! scratch-caching rule — see CLAUDE.md / MEMORY.md).
//!
//! ## Correctness invariant
//!
//! At greedy decode this kernel is mathematically equivalent (modulo
//! floating-point reorder of the per-thread reduction) to:
//!
//!   for k in 0..8: tmp_k = mt_swiglu(gate_k, up_k)
//!   for k in 0..8: down_k = ffai_dequant_gemv_int4_expert_indexed(
//!                              W, S, B, tmp_k, expert_indices[k:k+1])
//!   out = mt_scalar_fma_chain8(slot_weights[0:1], down_0, ...,
//!                              slot_weights[7:8], down_7)
//!
//! Tolerance budget: 1e-3 (f32), 5e-2 (bf16 / f16). The reduce_sum at
//! the tail fuses 8 slots' partial-sums in ONE simd reduction (vs 8
//! separate reductions in the unfused chain), so the rounding tree is
//! shallower; we err on the side of higher precision, not lower.
//!
//! ## Why this fusion is safe
//!
//! Each threadgroup is independent (one output row each). The
//! `tg_inner` scratch is private to its TG. Cross-slot ordering inside
//! a TG is enforced by two barriers per slot (RAW on fill→read, WAR on
//! read→next-fill). Per-thread `acc` is private; the final
//! `reduce_sum` (Reduction mode) handles the cross-thread fold.
//!
//! ## Why NOT register-resident `inner`
//!
//! Tempting alternative: each thread holds its `inner_k` slice in
//! registers and reads it during qmm without TG mem. Doesn't work —
//! the qmm pack-stride pattern means thread `t` needs `inner_k[d]`
//! values at offsets `pack_idx*8 + i` for i in 0..8, which are
//! neighbours, not strided slots. Threads can't share registers.
//! TG memory is the right abstraction here. 3 KiB is cheap.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="moe_down_swiglu_accum",
    subop="int4_chain8",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_down_swiglu_accum_int4_chain8<T>(
    gate_0: Tensor<T>,
    up_0: Tensor<T>,
    gate_1: Tensor<T>,
    up_1: Tensor<T>,
    gate_2: Tensor<T>,
    up_2: Tensor<T>,
    gate_3: Tensor<T>,
    up_3: Tensor<T>,
    gate_4: Tensor<T>,
    up_4: Tensor<T>,
    gate_5: Tensor<T>,
    up_5: Tensor<T>,
    gate_6: Tensor<T>,
    up_6: Tensor<T>,
    gate_7: Tensor<T>,
    up_7: Tensor<T>,
    expert_indices: Tensor<u32>,
    slot_weights: Tensor<T>,
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<T>,
    biases_stacked: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] group_size: u32,
) {
    // Threadgroup scratch for the active slot's inner activations.
    // 768 = Qwen3.6-A3B moeIntermediate. Bump the literal (and
    // re-validate Apple9 TG-mem ceiling) if a future model needs a
    // larger intermediate. f32 stage so the qmm consumer reads at
    // accumulation precision.
    threadgroup_alloc("tg_inner", 768, "f32");

    // Int4 dequant constants — match `dequant_gemv_int4_expert_indexed`.
    let vals_per_pack = 8u32;
    let mask = 0xFu32;
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;

    // Pre-load all 8 expert indices and slot weights into registers.
    // Cheap: 8 u32 loads + 8 T loads; reused across the slot loop.
    let exp_0 = load(expert_indices[0u32]);
    let exp_1 = load(expert_indices[1u32]);
    let exp_2 = load(expert_indices[2u32]);
    let exp_3 = load(expert_indices[3u32]);
    let exp_4 = load(expert_indices[4u32]);
    let exp_5 = load(expert_indices[5u32]);
    let exp_6 = load(expert_indices[6u32]);
    let exp_7 = load(expert_indices[7u32]);
    let sw_0 = load(slot_weights[0u32]).cast::<f32>();
    let sw_1 = load(slot_weights[1u32]).cast::<f32>();
    let sw_2 = load(slot_weights[2u32]).cast::<f32>();
    let sw_3 = load(slot_weights[3u32]).cast::<f32>();
    let sw_4 = load(slot_weights[4u32]).cast::<f32>();
    let sw_5 = load(slot_weights[5u32]).cast::<f32>();
    let sw_6 = load(slot_weights[6u32]).cast::<f32>();
    let sw_7 = load(slot_weights[7u32]).cast::<f32>();

    // Running per-thread accumulator across all 8 slots. Each slot's
    // contribution = slot_weight * Σ_packs (q*s+b) * tg_inner[d].
    // Final reduce_sum fuses the 8 slots' partials in one fold.
    let mut acc = 0.0f32;

    // Iteration counts — same shape as the indexed-expert dequant-gemv.
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    let in_iters = (in_dim + lsize - 1u32) / lsize;

    // ── Slot 0 ───────────────────────────────────────────────────────
    // (a) Cooperatively fill tg_inner with silu(gate_0) * up_0.
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_0[d]).cast::<f32>();
            let u = load(up_0[d]).cast::<f32>();
            // Inline silu in f32 — same form as gated_rmsnorm.rs and
            // swiglu.rs. Avoids T→f32→T round-trip and keeps the gate
            // precise before the multiply.
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    // (b) Dequant-gemv inner loop for slot 0.
    let we_0 = exp_0 * out_dim * n_packs_per_row;
    let se_0 = exp_0 * n_groups * out_dim;
    let rpo_0 = we_0 + row * n_packs_per_row;
    let rgo_0 = se_0 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_0 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_0 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_0 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_0 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 1 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_1[d]).cast::<f32>();
            let u = load(up_1[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_1 = exp_1 * out_dim * n_packs_per_row;
    let se_1 = exp_1 * n_groups * out_dim;
    let rpo_1 = we_1 + row * n_packs_per_row;
    let rgo_1 = se_1 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_1 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_1 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_1 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_1 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 2 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_2[d]).cast::<f32>();
            let u = load(up_2[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_2 = exp_2 * out_dim * n_packs_per_row;
    let se_2 = exp_2 * n_groups * out_dim;
    let rpo_2 = we_2 + row * n_packs_per_row;
    let rgo_2 = se_2 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_2 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_2 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_2 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_2 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 3 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_3[d]).cast::<f32>();
            let u = load(up_3[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_3 = exp_3 * out_dim * n_packs_per_row;
    let se_3 = exp_3 * n_groups * out_dim;
    let rpo_3 = we_3 + row * n_packs_per_row;
    let rgo_3 = se_3 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_3 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_3 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_3 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_3 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 4 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_4[d]).cast::<f32>();
            let u = load(up_4[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_4 = exp_4 * out_dim * n_packs_per_row;
    let se_4 = exp_4 * n_groups * out_dim;
    let rpo_4 = we_4 + row * n_packs_per_row;
    let rgo_4 = se_4 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_4 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_4 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_4 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_4 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 5 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_5[d]).cast::<f32>();
            let u = load(up_5[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_5 = exp_5 * out_dim * n_packs_per_row;
    let se_5 = exp_5 * n_groups * out_dim;
    let rpo_5 = we_5 + row * n_packs_per_row;
    let rgo_5 = se_5 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_5 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_5 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_5 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_5 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 6 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_6[d]).cast::<f32>();
            let u = load(up_6[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_6 = exp_6 * out_dim * n_packs_per_row;
    let se_6 = exp_6 * n_groups * out_dim;
    let rpo_6 = we_6 + row * n_packs_per_row;
    let rgo_6 = se_6 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_6 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_6 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_6 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_6 * dq * inner_v;
            }
        }
    }
    threadgroup_barrier();

    // ── Slot 7 ───────────────────────────────────────────────────────
    for s_iter in range(0u32, in_iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < in_dim {
            let g = load(gate_7[d]).cast::<f32>();
            let u = load(up_7[d]).cast::<f32>();
            let s = g / (1.0f32 + exp(0.0f32 - g));
            threadgroup_store("tg_inner", d, s * u);
        }
    }
    threadgroup_barrier();
    let we_7 = exp_7 * out_dim * n_packs_per_row;
    let se_7 = exp_7 * n_groups * out_dim;
    let rpo_7 = we_7 + row * n_packs_per_row;
    let rgo_7 = se_7 + row * n_groups;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[rgo_7 + g]).cast::<f32>();
            let bias = load(biases_stacked[rgo_7 + g]).cast::<f32>();
            let packed = load(weights_stacked[rpo_7 + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                let dq = q.cast::<f32>() * scale + bias;
                let inner_v = threadgroup_load("tg_inner", p_off + i);
                acc = acc + sw_7 * dq * inner_v;
            }
        }
    }
    // No trailing barrier — no further tg_inner access after slot 7.

    // ── Cross-thread fold + store ────────────────────────────────────
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped block-scaled BGEMM (BM=8) — the BM=8 sibling of
//! `moe_mpp_block_scaled` (BM=16) and the block-scaled / legacy-float /
//! symmetric-int8 counterpart of `moe_mpp_bm8` (int4) and `moe_mpp_bm8_int8`.
//!
//! Geometry is **byte-identical** to `mt_moe_gather_qmm_mma_int4_bm8_mpp` /
//! `…_int8_bm8_mpp`: BM=8 / BN=32 / BK=16 with the **direct-input** `matmul2d`
//! form (M=8 → the inputs cannot be cooperative tensors on Apple's MPP path, so
//! A and B are passed as direct `metal::tensor` views over threadgroup memory),
//! the per-TG contiguous-expert sub-run walk over 8 rows, the X staging
//! (32 lanes × 4), the `coop_tile_*` ops, and the masked coop-write tail
//! (32 lanes × 8 = 256 = BM*BN). The **only** change vs. the int templates is
//! the W-dequant staging block — it emits `element_decode(code) · block_scale`
//! (no bias) per `mlx/block_scaled_qmm_mpp` instead of the affine `scale·q + bias`.
//!
//! Nineteen kernels cover all twenty formats (`fp8_e4m3` reuses the `nvfp8`
//! kernel — both are 8-bit E4M3 + f32 per-block scale):
//!
//! | kernel                                 | element | weight | scale       |
//! |----------------------------------------|---------|--------|-------------|
//! | `mt_mxfp4_moe_gather_qmm_bm8_mpp`      | E2M1    | u32    | E8M0 (u8)   |
//! | `mt_nvfp4_moe_gather_qmm_bm8_mpp`      | E2M1    | u32    | E4M3 (u8) × global |
//! | `mt_fp4_moe_gather_qmm_bm8_mpp`        | E2M1    | u32    | f32         |
//! | `mt_mxfp8_e4m3_moe_gather_qmm_bm8_mpp` | E4M3    | u8     | E8M0 (u8)   |
//! | `mt_mxfp8_e5m2_moe_gather_qmm_bm8_mpp` | E5M2    | u8     | E8M0 (u8)   |
//! | `mt_fp8_e5m2_moe_gather_qmm_bm8_mpp`   | E5M2    | u8     | f32         |
//! | `mt_nvfp8_moe_gather_qmm_bm8_mpp`      | E4M3    | u8     | f32         |
//! | `mt_int8_moe_gather_qmm_bm8_mpp`       | int8    | u8     | f32         |
//! | `mt_int{2,3,4,5,6}_moe_gather_qmm_bm8_mpp`   | intN  | u32 | f32      |
//! | `mt_mxint{2,3,4,5,6}_moe_gather_qmm_bm8_mpp` | intN  | u32 | E8M0 (u8) |
//! | `mt_mxint8_moe_gather_qmm_bm8_mpp`     | int8    | u8     | E8M0 (u8)   |
//!
//! Weight layout (stacked `[n_experts·n_out, k_in]`, packed in ONE call —
//! never per-expert pack + concatenation): 4-bit `w [·, k_in/8] u32` (8 E2M1
//! nibbles/word, LSB-first), 8-bit `w [·, k_in] u8` (one code per byte),
//! sub-byte int2/3/5/6 `w [·, k_in·BITS/32] u32` (tight LSB-first bit-stream,
//! per-row word-aligned — `k_in·BITS % 32 == 0` since k_in is a multiple of 32,
//! one guard word at the very end of the whole stack). The global stacked row
//! is `g_row = expert·n_out + n`. Scales `[n_experts, n_out, k_in/block_size]`
//! are u8 (E8M0/E4M3) or f32 (nvfp8 / legacy fp / int / mxint). No `biases`
//! param — block-scaled is scale-only.
//!
//! ## bf16 staging
//!
//! Same `coop_stage(T)` trick as the int templates: bf16 activations stage
//! through `half` so `mpp::tensor_ops::matmul2d` sees a supported
//! cooperative-tensor dtype. Accumulation is fp32.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(8, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)`, `direct_inputs=true` — identical to the int4/int8
//! BM=8 MPP descriptor; only the threadgroup W tile contents differ.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/32, ceil(m_total/8), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup — `matmul2d` is `execution_simdgroup`).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `block_size` divides `k_in`, and
//!   `block_size ≥ 16` (so the 16-element K window staged per lane per `kb`
//!   sits inside one block — one scale load per lane per `kb` is exact).
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a linkable stub.

use metaltile::kernel;

// ── 4-bit (E2M1) MoE MPP kernels — model: moe_mpp_bm8 (int4) ────────────────

/// mxfp4 MoE gather BGEMM, BM=8 / BN=32 / BK=16 — E2M1 weights, E8M0 scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]` (per-row expert id), `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    // Descriptor 8×32×16, direct-input (M=8 → not a cooperative tensor).
    // direct_inputs=true; A view = [K=16, M=8], B view = [K=16, N=32].
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    // Walk the BM=8 rows in contiguous-expert sub-runs.
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                // Stage X[m_tile_base..+8, kb..kb+16] → xs. 32 lanes × 4.
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws.
                // 32 lanes × 2 packs/lane; 8 nibbles/pack.
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane * 2u32 + _pi;
                    let w_row = pack_id / 2u32; // 0..31 (BN rows)
                    let pack_col = pack_id % 2u32; // 0..1 (BK=16 → 2 packs)
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 8u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 8u32;
                    let g = k_off / block_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    // mxfp4: E8M0 pow-2 block scale → 2^(bits-127).
                    let scale = exp2(load(scales[sb_off]).cast::<f32>() - 127.0f32);
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, e2m1_decode(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            // C [M=8, N=32] row-major → extents N,M = 32,8.
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// nvfp4 MoE gather BGEMM, BM=8 / BN=32 / BK=16 — E2M1 weights, E4M3
/// micro-scale × global FP32. `global` is the LAST constexpr.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (E4M3 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane * 2u32 + _pi;
                    let w_row = pack_id / 2u32;
                    let pack_col = pack_id % 2u32;
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 8u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 8u32;
                    let g = k_off / block_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    // nvfp4: E4M3 micro-scale × global FP32.
                    let scale = e4m3_decode(load(scales[sb_off]).cast::<u32>()) * global;
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, e2m1_decode(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// Legacy fp4 MoE gather BGEMM, BM=8 / BN=32 / BK=16 — E2M1 weights, per-group
/// FP32 scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane * 2u32 + _pi;
                    let w_row = pack_id / 2u32;
                    let pack_col = pack_id % 2u32;
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 8u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 8u32;
                    let g = k_off / block_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    // fp4: raw per-group FP32 scale.
                    let scale = load(scales[sb_off]);
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, e2m1_decode(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

// ── 8-bit (E4M3 / E5M2 / int8) MoE MPP kernels — u8 byte-strided weight ─────

/// mxfp8 (E4M3) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit E4M3 weights,
/// E8M0 pow-2 block scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E4M3, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                // Dequant W → ws. 32 lanes (lane = BN row), 16 K-elems/lane.
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // mxfp8 e4m3: E8M0 pow-2 block scale → 2^(bits-127).
                let scale = exp2(load(scales[sb_off]).cast::<f32>() - 127.0f32);
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = e4m3_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// mxfp8 (E5M2) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit E5M2 weights,
/// E8M0 pow-2 block scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E5M2, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // mxfp8 e5m2: E8M0 pow-2 block scale → 2^(bits-127).
                let scale = exp2(load(scales[sb_off]).cast::<f32>() - 127.0f32);
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = e5m2_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// Legacy fp8 (E5M2) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit E5M2
/// weights, per-group FP32 scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E5M2, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // fp8 e5m2: raw per-group FP32 scale.
                let scale = load(scales[sb_off]);
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = e5m2_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// nvfp8 MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit E4M3 weights, per-block
/// FP32 scale. Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape;
/// only the `block_size` constexpr differs).
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E4M3, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // nvfp8: raw per-block FP32 scale.
                let scale = load(scales[sb_off]);
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = e4m3_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// Symmetric int8 MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit codes,
/// per-group FP32 scale (no bias).
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (int8 codes, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // int8: raw per-group FP32 scale (symmetric, no bias).
                let scale = load(scales[sb_off]);
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = int8_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

// ── Symmetric sub-byte integer MoE MPP kernels (int2/3/4/5/6 + MXINT2..6) ───
// The element is a signed N-bit two's-complement code, tight-bit-packed
// LSB-first into u32 words. The WHOLE `[n_experts·n_out, k_in]` expert stack is
// packed in ONE call (the test builds the full stacked matrix and packs once —
// never per-expert pack + byte concatenation), so it is a single contiguous
// bit-stream with one guard word at the very end. Every weight row therefore
// stays word-aligned (k_in a multiple of 32 ⇒ `k_in·BITS % 32 == 0`), and the
// per-row word base is just `g_row · (k_in·BITS/32)` with the gather's global
// stacked row `g_row = cur_expert·n_out + (n_tile_base + w_row)` — exactly the
// `[g_row, k_col]` flat index the 8-bit kernels read, expressed as a bit
// offset. Each lane stages a 16-K-element stripe of one BN row into `ws` exactly
// as the 8-bit kernels do; only the per-element decode changes. Decode is the
// straddle-aware two-word read + float sign-extend from `block_scaled_dequant`'s
// proven `int_dequant_*` macros, multiplied by the block scale. `$half`/`$full`
// are 2^(N-1)/2^N passed as literals to keep the constexpr shift math out of the
// DSL operands. **Dispatch geometry, BM/BN/BK tile sizes, coop-tensor extents,
// TPG, grid, and the contiguous-expert sub-run walk are byte-identical to the
// 8-bit kernels above** — only the W-stage decode + scale read differ.

/// FP32-scaled symmetric int MoE gather BGEMM (int2/3/4/5/6): per-element
/// bit-stream code × per-group FP32 scale, staged into `ws` and fed to the
/// tensor engine. The lane's BN row maps to global stacked row
/// `g_row = cur_expert·n_out + (n_tile_base + w_row)`, whose tight bit-stream
/// base word is `g_row · (k_in·BITS/32)`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts·n_out, k_in·BITS/32]` (tight
/// LSB-first bit-stream, per-row word-aligned), `scales [n_experts, n_out,
/// k_in/block_size]` (f32), `indices [m_total]`, `out [m_total, n_out]`.
macro_rules! int_moe_gather_qmm_bm8_mpp_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            x: Tensor<T>,
            w: Tensor<u32>,
            scales: Tensor<f32>,
            indices: Tensor<u32>,
            mut out: Tensor<T>,
            #[constexpr] m_total: u32,
            #[constexpr] n_out: u32,
            #[constexpr] k_in: u32,
            #[constexpr] block_size: u32,
        ) {
            let n_tile_base = tgid_x * 32u32;
            let m_tile_base = tgid_y * 8u32;
            let lane = simd_lane;
            let groups_per_row = k_in / block_size;
            let words_per_row = k_in * $bits / 32u32;
            threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
            threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
            threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
            coop_tile_setup(
                "gemm",
                8,
                32,
                16, // m, n, k
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
                true, // direct_inputs
                true,
                16,
                8, // a: is_tg, ei, eo
                true,
                16,
                32, // b: is_tg, ei, eo
            );
            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 8u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
                // Find the run end — first row whose expert differs (or OOB).
                let mut sub_end = 8u32;
                let mut found = 0u32;
                for _ii in range(0u32, 8u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 8u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
                if cur_valid {
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    coop_tile_zero("gemm");
                    for kb in range(0u32, k_in, 16u32) {
                        for _e in range(0u32, 4u32, 1u32) {
                            let flat = lane * 4u32 + _e;
                            let mr = flat / 16u32;
                            let kc = flat % 16u32;
                            let gr = m_tile_base + mr;
                            let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                            let safe_g = select(in_run, gr, 0u32);
                            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                        }
                        // Dequant W → ws. 32 lanes (lane = BN row), 16 K-elems/lane.
                        let w_row = lane; // 0..31 (BN row)
                        // Global stacked row (single contiguous bit-stream).
                        let g_row = cur_expert * n_out + n_tile_base + w_row;
                        let row_word = g_row * words_per_row;
                        let g = kb / block_size;
                        let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                        // int2/3/5/6: raw per-group FP32 scale.
                        let scale = load(scales[sb_off]);
                        for kc in range(0u32, 16u32, 1u32) {
                            let bit_off = (kb + kc) * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w[row_word + word_idx]);
                            let w1 =
                                load(w[row_word + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            threadgroup_store("ws", w_row * 16u32 + kc, val * scale);
                        }
                        threadgroup_barrier();
                        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                        coop_tile_run("gemm", true);
                        threadgroup_barrier();
                    }
                    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
                    threadgroup_barrier();
                    // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
                    for _e in range(0u32, 8u32, 1u32) {
                        let flat = lane * 8u32 + _e;
                        let mr = flat / 32u32;
                        let nc = flat % 32u32;
                        let gr = m_tile_base + mr;
                        let gc = n_tile_base + nc;
                        let in_run =
                            (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                        if in_run {
                            let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                            store(out[gr * n_out + gc], v.cast::<T>());
                        }
                    }
                    threadgroup_barrier();
                }
                sub_offset = sub_end;
            }
        }
    };
}
int_moe_gather_qmm_bm8_mpp_f32!(mt_int2_moe_gather_qmm_bm8_mpp, 2u32, 2u32, 4.0f32);
int_moe_gather_qmm_bm8_mpp_f32!(mt_int3_moe_gather_qmm_bm8_mpp, 3u32, 4u32, 8.0f32);
int_moe_gather_qmm_bm8_mpp_f32!(mt_int4_moe_gather_qmm_bm8_mpp, 4u32, 8u32, 16.0f32);
int_moe_gather_qmm_bm8_mpp_f32!(mt_int5_moe_gather_qmm_bm8_mpp, 5u32, 16u32, 32.0f32);
int_moe_gather_qmm_bm8_mpp_f32!(mt_int6_moe_gather_qmm_bm8_mpp, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int MoE gather BGEMM (MXINT2/3/4/5/6): per-element
/// bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`, staged into `ws`.
/// Same straddle-aware bit-stream decode and staging path as
/// `int_moe_gather_qmm_bm8_mpp_f32`; only the scale axis differs (one u8
/// exponent per block instead of a raw f32).
///
/// Params: `x [m_total, k_in]`, `w [n_experts·n_out, k_in·BITS/32]` (tight
/// LSB-first bit-stream, per-row word-aligned), `scales [n_experts, n_out,
/// k_in/block_size]` (E8M0 byte), `indices [m_total]`, `out [m_total, n_out]`.
macro_rules! int_moe_gather_qmm_bm8_mpp_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            x: Tensor<T>,
            w: Tensor<u32>,
            scales: Tensor<u8>,
            indices: Tensor<u32>,
            mut out: Tensor<T>,
            #[constexpr] m_total: u32,
            #[constexpr] n_out: u32,
            #[constexpr] k_in: u32,
            #[constexpr] block_size: u32,
        ) {
            let n_tile_base = tgid_x * 32u32;
            let m_tile_base = tgid_y * 8u32;
            let lane = simd_lane;
            let groups_per_row = k_in / block_size;
            let words_per_row = k_in * $bits / 32u32;
            threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
            threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
            threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
            coop_tile_setup(
                "gemm",
                8,
                32,
                16, // m, n, k
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
                true, // direct_inputs
                true,
                16,
                8, // a: is_tg, ei, eo
                true,
                16,
                32, // b: is_tg, ei, eo
            );
            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 8u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
                // Find the run end — first row whose expert differs (or OOB).
                let mut sub_end = 8u32;
                let mut found = 0u32;
                for _ii in range(0u32, 8u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 8u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
                if cur_valid {
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    coop_tile_zero("gemm");
                    for kb in range(0u32, k_in, 16u32) {
                        for _e in range(0u32, 4u32, 1u32) {
                            let flat = lane * 4u32 + _e;
                            let mr = flat / 16u32;
                            let kc = flat % 16u32;
                            let gr = m_tile_base + mr;
                            let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                            let safe_g = select(in_run, gr, 0u32);
                            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                        }
                        // Dequant W → ws. 32 lanes (lane = BN row), 16 K-elems/lane.
                        let w_row = lane; // 0..31 (BN row)
                        // Global stacked row (single contiguous bit-stream).
                        let g_row = cur_expert * n_out + n_tile_base + w_row;
                        let row_word = g_row * words_per_row;
                        let g = kb / block_size;
                        let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                        // MXINT2/3/5/6: E8M0 pow-2 block scale → 2^(bits-127).
                        let scale = exp2(load(scales[sb_off]).cast::<f32>() - 127.0f32);
                        for kc in range(0u32, 16u32, 1u32) {
                            let bit_off = (kb + kc) * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w[row_word + word_idx]);
                            let w1 =
                                load(w[row_word + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            threadgroup_store("ws", w_row * 16u32 + kc, val * scale);
                        }
                        threadgroup_barrier();
                        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                        coop_tile_run("gemm", true);
                        threadgroup_barrier();
                    }
                    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
                    threadgroup_barrier();
                    // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
                    for _e in range(0u32, 8u32, 1u32) {
                        let flat = lane * 8u32 + _e;
                        let mr = flat / 32u32;
                        let nc = flat % 32u32;
                        let gr = m_tile_base + mr;
                        let gc = n_tile_base + nc;
                        let in_run =
                            (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                        if in_run {
                            let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                            store(out[gr * n_out + gc], v.cast::<T>());
                        }
                    }
                    threadgroup_barrier();
                }
                sub_offset = sub_end;
            }
        }
    };
}
int_moe_gather_qmm_bm8_mpp_e8m0!(mt_mxint2_moe_gather_qmm_bm8_mpp, 2u32, 2u32, 4.0f32);
int_moe_gather_qmm_bm8_mpp_e8m0!(mt_mxint3_moe_gather_qmm_bm8_mpp, 3u32, 4u32, 8.0f32);
int_moe_gather_qmm_bm8_mpp_e8m0!(mt_mxint4_moe_gather_qmm_bm8_mpp, 4u32, 8u32, 16.0f32);
int_moe_gather_qmm_bm8_mpp_e8m0!(mt_mxint5_moe_gather_qmm_bm8_mpp, 5u32, 16u32, 32.0f32);
int_moe_gather_qmm_bm8_mpp_e8m0!(mt_mxint6_moe_gather_qmm_bm8_mpp, 6u32, 32u32, 64.0f32);

/// MXINT8 MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit symmetric codes (byte
/// layout, block 32), E8M0 pow-2 block scale `2^(bits-127)`. Byte-strided
/// staging like the int8 / mxfp8 kernels (one byte per code); decode is
/// `int8_decode → val · scale`. Geometry and coop-tensor extents are
/// byte-identical to the int8 kernel above.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (int8 codes, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // mxint8: E8M0 pow-2 block scale → 2^(bits-127).
                let scale = exp2(load(scales[sb_off]).cast::<f32>() - 127.0f32);
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = int8_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

// ── FP16-scale twins (nvfp8 / fp4 / fp8_e5m2 / int2..6 / int8) ──────────────
// Each kernel below is a byte-for-byte clone of its FP32-scaled sibling above
// with ONLY the scale axis changed: the `scales` tensor binds as `Tensor<f16>`
// (two bytes per group) and every scale read becomes
// `load(scales[…]).cast::<f32>()` so the dequant math stays in fp32. The
// element decode (E4M3 / E2M1 / E5M2 / int bit-stream + sign-extend), weight
// indexing, X staging, coop-tensor extents, descriptor, grid/TPG, KernelMode,
// and the contiguous-expert sub-run walk are **identical** to the FP32 twins —
// nothing about the MPP/coop geometry changes. The proven decode + f16 scale
// read mirrors `mlx/block_scaled_dequant`'s `mt_*_f16_dequant` references.

/// nvfp8 (FP16 scale) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit E4M3
/// weights, per-block FP16 scale. Clone of `mt_nvfp8_moe_gather_qmm_bm8_mpp`
/// with the scale axis as f16. Also serves **fp8_e4m3_f16** (same 8-bit-E4M3 +
/// f16-scale shape; only the `block_size` constexpr differs).
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E4M3, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<f16>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // nvfp8 f16: raw per-block FP16 scale (read as f16, math in f32).
                let scale = load(scales[sb_off]).cast::<f32>();
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = e4m3_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// Legacy fp4 (FP16 scale) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — E2M1
/// weights, per-group FP16 scale. Clone of `mt_fp4_moe_gather_qmm_bm8_mpp` with
/// the scale axis as f16.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<f16>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane * 2u32 + _pi;
                    let w_row = pack_id / 2u32;
                    let pack_col = pack_id % 2u32;
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 8u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 8u32;
                    let g = k_off / block_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    // fp4 f16: raw per-group FP16 scale (read as f16, math in f32).
                    let scale = load(scales[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, e2m1_decode(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// Legacy fp8 (E5M2, FP16 scale) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit
/// E5M2 weights, per-group FP16 scale. Clone of
/// `mt_fp8_e5m2_moe_gather_qmm_bm8_mpp` with the scale axis as f16.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E5M2, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<f16>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // fp8 e5m2 f16: raw per-group FP16 scale (read as f16, math in f32).
                let scale = load(scales[sb_off]).cast::<f32>();
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = e5m2_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

/// FP16-scaled symmetric int MoE gather BGEMM (int2/3/4/5/6, group 64): per-row
/// bit-stream code × per-group FP16 scale. Byte-for-byte clone of
/// `int_moe_gather_qmm_bm8_mpp_f32` with the scale axis as f16 — the global
/// stacked row, tight bit-stream base word, straddle-aware decode + sign-extend,
/// and all geometry are identical; only `scales` binds as f16 and the scale read
/// gains `.cast::<f32>()`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts·n_out, k_in·BITS/32]` (tight
/// LSB-first bit-stream, per-row word-aligned), `scales [n_experts, n_out,
/// k_in/block_size]` (f16), `indices [m_total]`, `out [m_total, n_out]`.
macro_rules! int_moe_gather_qmm_bm8_mpp_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            x: Tensor<T>,
            w: Tensor<u32>,
            scales: Tensor<f16>,
            indices: Tensor<u32>,
            mut out: Tensor<T>,
            #[constexpr] m_total: u32,
            #[constexpr] n_out: u32,
            #[constexpr] k_in: u32,
            #[constexpr] block_size: u32,
        ) {
            let n_tile_base = tgid_x * 32u32;
            let m_tile_base = tgid_y * 8u32;
            let lane = simd_lane;
            let groups_per_row = k_in / block_size;
            let words_per_row = k_in * $bits / 32u32;
            threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
            threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
            threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
            coop_tile_setup(
                "gemm",
                8,
                32,
                16, // m, n, k
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
                true, // direct_inputs
                true,
                16,
                8, // a: is_tg, ei, eo
                true,
                16,
                32, // b: is_tg, ei, eo
            );
            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 8u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
                // Find the run end — first row whose expert differs (or OOB).
                let mut sub_end = 8u32;
                let mut found = 0u32;
                for _ii in range(0u32, 8u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 8u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
                if cur_valid {
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    coop_tile_zero("gemm");
                    for kb in range(0u32, k_in, 16u32) {
                        for _e in range(0u32, 4u32, 1u32) {
                            let flat = lane * 4u32 + _e;
                            let mr = flat / 16u32;
                            let kc = flat % 16u32;
                            let gr = m_tile_base + mr;
                            let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                            let safe_g = select(in_run, gr, 0u32);
                            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                        }
                        // Dequant W → ws. 32 lanes (lane = BN row), 16 K-elems/lane.
                        let w_row = lane; // 0..31 (BN row)
                        // Global stacked row (single contiguous bit-stream).
                        let g_row = cur_expert * n_out + n_tile_base + w_row;
                        let row_word = g_row * words_per_row;
                        let g = kb / block_size;
                        let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                        // int2/3/5/6 f16: raw per-group FP16 scale (read as f16, math in f32).
                        let scale = load(scales[sb_off]).cast::<f32>();
                        for kc in range(0u32, 16u32, 1u32) {
                            let bit_off = (kb + kc) * $bits;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                            let spill = $bits - lo_bits;
                            let w0 = load(w[row_word + word_idx]);
                            let w1 =
                                load(w[row_word + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                            let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                            let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                            let q = lo | hi;
                            let qf = q.cast::<f32>();
                            let val = select(q >= $half, qf - $full, qf); // sign-extend
                            threadgroup_store("ws", w_row * 16u32 + kc, val * scale);
                        }
                        threadgroup_barrier();
                        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                        coop_tile_run("gemm", true);
                        threadgroup_barrier();
                    }
                    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
                    threadgroup_barrier();
                    // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
                    for _e in range(0u32, 8u32, 1u32) {
                        let flat = lane * 8u32 + _e;
                        let mr = flat / 32u32;
                        let nc = flat % 32u32;
                        let gr = m_tile_base + mr;
                        let gc = n_tile_base + nc;
                        let in_run =
                            (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                        if in_run {
                            let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                            store(out[gr * n_out + gc], v.cast::<T>());
                        }
                    }
                    threadgroup_barrier();
                }
                sub_offset = sub_end;
            }
        }
    };
}
int_moe_gather_qmm_bm8_mpp_f16!(mt_int2_f16_moe_gather_qmm_bm8_mpp, 2u32, 2u32, 4.0f32);
int_moe_gather_qmm_bm8_mpp_f16!(mt_int3_f16_moe_gather_qmm_bm8_mpp, 3u32, 4u32, 8.0f32);
int_moe_gather_qmm_bm8_mpp_f16!(mt_int4_f16_moe_gather_qmm_bm8_mpp, 4u32, 8u32, 16.0f32);
int_moe_gather_qmm_bm8_mpp_f16!(mt_int5_f16_moe_gather_qmm_bm8_mpp, 5u32, 16u32, 32.0f32);
int_moe_gather_qmm_bm8_mpp_f16!(mt_int6_f16_moe_gather_qmm_bm8_mpp, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) MoE gather BGEMM, BM=8 / BN=32 / BK=16 — 8-bit codes (byte
/// layout, group 64), per-group FP16 scale (no bias). Clone of
/// `mt_int8_moe_gather_qmm_bm8_mpp` with the scale axis as f16.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (int8 codes, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_moe_gather_qmm_bm8_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u8>,
    scales: Tensor<f16>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 256, f32); // 8 × 32
    coop_tile_setup(
        "gemm",
        8,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true, // direct_inputs
        true,
        16,
        8, // a: is_tg, ei, eo
        true,
        16,
        32, // b: is_tg, ei, eo
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 8u32;
        let mut found = 0u32;
        for _ii in range(0u32, 8u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 8u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 8u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 8u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 4u32, 1u32) {
                    let flat = lane * 4u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                let w_row = lane; // 0..31 (BN row)
                let g = kb / block_size;
                let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                // int8 f16: raw per-group FP16 scale (read as f16, math in f32).
                let scale = load(scales[sb_off]).cast::<f32>();
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = int8_decode(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32, true);
                coop_tile_run("gemm", true);
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 8);
            threadgroup_barrier();
            // Coop-write out_scratch → out. 32 lanes × 8 elems = 256 = BM*BN.
            for _e in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

#[cfg(test)]
mod tests {
    use metaltile::{
        codegen::msl::MslGenerator,
        core::{DType, ir::Op},
    };

    use super::*;

    /// Every block-scaled MoE kernel (BM=8) builds, drops to `CoopTile*` ops (no
    /// raw inline MSL), and has the 5-tensor / 4-constexpr ABI (nvfp4 adds the
    /// `global` constexpr → 5).
    #[test]
    fn kernels_construct_and_use_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let kernels = [
                ("mt_mxfp4_moe_gather_qmm_bm8_mpp", 4usize),
                ("mt_fp4_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxfp8_e4m3_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxfp8_e5m2_moe_gather_qmm_bm8_mpp", 4),
                ("mt_fp8_e5m2_moe_gather_qmm_bm8_mpp", 4),
                ("mt_nvfp8_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int8_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int2_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int3_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int4_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int5_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int6_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxint2_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxint3_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxint4_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxint5_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxint6_moe_gather_qmm_bm8_mpp", 4),
                ("mt_mxint8_moe_gather_qmm_bm8_mpp", 4),
                ("mt_nvfp8_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_fp4_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_fp8_e5m2_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int2_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int3_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int4_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int5_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int6_f16_moe_gather_qmm_bm8_mpp", 4),
                ("mt_int8_f16_moe_gather_qmm_bm8_mpp", 4),
            ];
            let irs = [
                mt_mxfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_fp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxfp8_e4m3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxfp8_e5m2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_fp8_e5m2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_nvfp8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int5_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int6_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxint2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxint3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxint4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxint5_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxint6_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_mxint8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_nvfp8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_fp4_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_fp8_e5m2_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int2_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int3_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int4_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int5_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int6_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
                mt_int8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            ];
            for ((name, n_const), k) in kernels.iter().zip(irs.iter()) {
                assert_eq!(&k.name, name);
                assert_eq!(k.params.len(), 5, "{name}: 5 tensor params (no biases)");
                assert!(k.params[4].is_output, "{name}: out is last param");
                assert_eq!(k.constexprs.len(), *n_const, "{name}: constexpr count");
                let all_ops =
                    || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
                assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })), "{name}: no MSL");
                assert!(
                    all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })),
                    "{name}: setup"
                );
                assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })), "{name}: run");
            }
        }
    }

    /// nvfp4 carries the extra `global` FP32 constexpr (LAST), giving 5.
    #[test]
    fn nvfp4_has_global_constexpr_last() {
        let k = mt_nvfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(DType::F32);
        assert_eq!(k.params.len(), 5, "5 tensor params (no biases)");
        assert_eq!(k.constexprs.len(), 5, "m_total/n_out/k_in/block_size/global");
        assert_eq!(k.constexprs[4].name.name(), "global", "global is the last constexpr");
    }

    /// bf16 must stage through `half`: the `coop_stage(T)` tiles and
    /// cooperative tensors resolve to `half`, never `bfloat`.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_int8_moe_gather_qmm_bm8_mpp::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half for matmul2d");
    }

    /// Codegen sanity — the MPP header + descriptor land in the MSL.
    #[test]
    fn codegen_emits_mpp_include() {
        let mut k = mt_mxfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(DType::F32);
        k.name = "mt_mxfp4_moe_gather_qmm_bm8_mpp_f32".into();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_mxfp4_moe_gather_qmm_bm8_mpp_f32"));
    }
}

/// New-syntax correctness tests for the MPP block-scaled MoE BGEMM (BM=8).
///
/// Oracle is the clean per-row-`indices` block-scaled dequant-then-matmul: each
/// row `t` resolves its expert from `indices[t]`, dequantizes that expert's
/// `[n_out, k_in]` weight slab via the shared `quant::format::dequant`, and dots
/// against the row's input. Inputs are dtype-rounded so the GPU sees exactly
/// what the oracle computes; tolerance is wide because the MPP cooperative-tensor
/// accumulator reorders the K reduction. `fp8_e4m3` dispatches the `nvfp8`
/// kernel with `QFormat::Fp8E4m3` (same 8-bit-E4M3 + f32-scale shape).
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(n_out/32, ceil(m_total/8), 1, [32,1,1])`.
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Test shape for a block-scaled MoE variant (all clean multiples).
    struct BlockTestShape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    /// Build a `TestSetup` for a block-scaled indexed-MoE-MPP kernel (BM=8).
    /// Mirrors `int8_indexed_setup`: per-row expert routing, dtype-rounded x,
    /// oracle = `Σ_k x[t,k] · dequant(W_expert)[nc,k]`. Differs in that the
    /// whole `[n_experts·n_out, k_in]` expert stack is packed in ONE
    /// `quant::format::pack` call (no biases buffer; scale dtype per format;
    /// weight dtype U32 for sub-byte bit-streams / U8 for 8-bit). `block_size`
    /// and the nvfp4 `global` constexpr come from the packed tensor. The BM=8
    /// m-tile height drives `ceil(m_total/8)` m-tiles.
    #[allow(clippy::too_many_arguments)]
    fn block_indexed_setup(
        kernel: Kernel,
        fmt: QFormat,
        shape: BlockTestShape,
        dt: DType,
    ) -> TestSetup {
        let BlockTestShape { n_experts, m_total, n_out, k_in } = shape;
        let block_size = fmt.block_size();
        let stack_rows = n_experts * n_out;

        // Per-row expert indices, sorted (post-permute layout).
        let indices: Vec<u32> = (0..m_total).map(|r| (r / (m_total / n_experts)) as u32).collect();

        // Build the FULL `[n_experts·n_out, k_in]` stacked weight matrix (all
        // experts stacked along rows) and pack it in ONE call — never per-expert
        // packing + byte concatenation. For sub-byte widths (3/5/6-bit) `pack`
        // appends a single guard word at the very end of the contiguous
        // bit-stream; concatenating per-expert buffers would instead inject a
        // guard word mid-stream and misalign every expert after the first. One
        // stacked pack is byte-identical to the old per-expert concat for the
        // 4-bit/8-bit formats (those widths divide 32 ⇒ exact word count, no
        // guard word) and correct for every sub-byte width. `k_in` is a multiple
        // of 32, so each row's bit-stream is word-aligned for every width. The
        // magnitude pattern mirrors the non-MoE block-scaled test, keyed off the
        // global stacked row (expert folded into the row index).
        let stacked: Vec<f32> = (0..stack_rows * k_in)
            .map(|i| {
                let g_row = i / k_in;
                let e = (g_row / n_out) as f32;
                let r = (g_row % n_out) as f32;
                let c = (i % k_in) as f32;
                let mag = (0.4 + ((r + e) % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15);
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::quant::format::pack(fmt, &stacked, stack_rows, k_in);
        let global = p.global;
        // Dequant the full stack once; row `expert·n_out + nc` is expert `e`'s
        // output row `nc`.
        let wdq = crate::quant::format::dequant(fmt, &p, stack_rows, k_in);

        // Activations: dtype-rounded so the GPU sees exactly the oracle's x.
        let x_f: Vec<f32> = (0..m_total * k_in).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);

        // Oracle: out[t, nc] = Σ_k x[t, k] · dequant(W)[expert(t)·n_out + nc, k].
        let mut expected = vec![0.0f32; m_total * n_out];
        for t in 0..m_total {
            let base = indices[t] as usize * n_out;
            for nc in 0..n_out {
                let mut acc = 0.0f32;
                for kk in 0..k_in {
                    acc += x[t * k_in + kk] * wdq[(base + nc) * k_in + kk];
                }
                expected[t * n_out + nc] = acc;
            }
        }

        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16 (two bytes); E8M0/E4M3 scales as
        // one byte. Both axes are driven off the format so new integer / fp16
        // formats pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };

        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            n_out as u32 / 32,
            (m_total as u32).div_ceil(8),
            1,
            [32, 1, 1],
        )
    }

    // n_experts=4, m_total=64, n_out=64, k_in=64 (divisible by 16/32/64).
    // BM=8 → ceil(64/8)=8 m-tiles, BN=32 → 64/32=2 n-tiles.
    const SHAPE: BlockTestShape = BlockTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp4_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxfp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Nvfp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp8_e4m3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp8_e5m2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp8_e5m2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8,
            SHAPE,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int8,
            SHAPE,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). k_in=64 is a multiple of 32, so
    // each weight row's bit-stream is word-aligned for every width, and the
    // whole `[n_experts·n_out, k_in]` stack is packed once → one contiguous
    // bit-stream (guard word at the very end). Kernel + oracle share the codec.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int2_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int3_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int4_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int5_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int5_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int6_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int6_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int6,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint2_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxint2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint3_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxint3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint4_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxint4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint5_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint5_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxint5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint6_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint6_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxint6,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint8_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxint8,
            SHAPE,
            dt,
        )
    }

    // FP16-scale twins: same element packing as their FP32 siblings, only the
    // scale buffer is f16 (driven through the `scales_dt` match). `fp8_e4m3_f16`
    // reuses the `nvfp8_f16` kernel (8-bit E4M3 + f16 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp4_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp4F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp8_e5m2_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int2_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int2_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int2F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int3_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int3_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int3F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int4_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int4_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int4F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int5_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int5_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int5F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int6_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int6_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int6F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_f16_moe_gather_qmm_bm8_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int8F16,
            SHAPE,
            dt,
        )
    }
}

/// New-syntax benchmarks for the MPP block-scaled MoE BGEMM (BM=8). Random
/// buffers; `flops = 2·m_total·n_out·k_in` (the gather does a full matmul per
/// row's expert — dense-equivalent FLOPs).
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(n_out/32, m_total.div_ceil(8), 1, [32,1,1])`.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    struct BlockBenchShape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    fn block_bench(kernel: Kernel, fmt: QFormat, shape: BlockBenchShape, dt: DType) -> BenchSetup {
        let BlockBenchShape { n_experts, m_total, n_out, k_in } = shape;
        let block_size = fmt.block_size();
        let groups_per_row = k_in / block_size;
        // The whole `[n_experts·n_out, k_in]` expert stack is one contiguous
        // bit-stream (single pack), so its code length is `bitstream_words` over
        // the *total* element count (one guard word for the whole stack). 8-bit
        // codes are one uchar each; every sub-byte width (4-bit nibble packs +
        // int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words
        // (`bitstream_words` collapses to the old `n·k/8` for the 4-bit case).
        // Both axes are driven off the format so new integer formats pick up the
        // right buffer types with no regression for the pre-existing formats.
        let stack_n = n_experts * n_out * k_in;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (stack_n, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(stack_n, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = n_experts * n_out * groups_per_row;
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m_total * k_in * sz
            + m_total * n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("indices", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(n_out as u32 / 32, (m_total as u32).div_ceil(8), 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            // MoE gather_qmm indexed: 2 * m_total * n_out * k_in (dense-equivalent).
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
            .with_shape_label(format!(
                "{} M{m_total} N{n_out} K{k_in} E{n_experts} {}",
                fmt.name(),
                crate::utils::dtype_label(dt)
            ))
    }

    // n_experts=8, m_total=512, n_out=4096, k_in=4096.
    const SHAPE: BlockBenchShape =
        BlockBenchShape { n_experts: 8, m_total: 512, n_out: 4096, k_in: 4096 };

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4(dt: DType) -> BenchSetup {
        block_bench(mt_mxfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxfp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Nvfp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4(dt: DType) -> BenchSetup {
        block_bench(mt_fp4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Fp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxfp8_e4m3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxfp8_e5m2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp8_e5m2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Nvfp8, SHAPE, dt)
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Fp8E4m3, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        block_bench(mt_int8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Int8, SHAPE, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). k_in=4096 is a multiple of 32 → word-aligned per
    // width; the whole expert stack is one contiguous bit-stream (single pack).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2(dt: DType) -> BenchSetup {
        block_bench(mt_int2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Int2, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3(dt: DType) -> BenchSetup {
        block_bench(mt_int3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Int3, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4(dt: DType) -> BenchSetup {
        block_bench(mt_int4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Int4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5(dt: DType) -> BenchSetup {
        block_bench(mt_int5_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Int5, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6(dt: DType) -> BenchSetup {
        block_bench(mt_int6_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Int6, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2(dt: DType) -> BenchSetup {
        block_bench(mt_mxint2_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxint2, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3(dt: DType) -> BenchSetup {
        block_bench(mt_mxint3_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxint3, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4(dt: DType) -> BenchSetup {
        block_bench(mt_mxint4_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxint4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5(dt: DType) -> BenchSetup {
        block_bench(mt_mxint5_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxint5, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6(dt: DType) -> BenchSetup {
        block_bench(mt_mxint6_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxint6, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8(dt: DType) -> BenchSetup {
        block_bench(mt_mxint8_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt), QFormat::Mxint8, SHAPE, dt)
    }
    // FP16-scale twins: same element packing as their FP32 siblings; only the
    // scale buffer is f16 (driven through the `scales_dt` match). fp8_e4m3_f16
    // reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp4_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp4F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp8_e5m2_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int2_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int2F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int3_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int3F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int4_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int4F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int5_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int5F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int6_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int6F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int8_f16_moe_gather_qmm_bm8_mpp::kernel_ir_for(dt),
            QFormat::Int8F16,
            SHAPE,
            dt,
        )
    }
}

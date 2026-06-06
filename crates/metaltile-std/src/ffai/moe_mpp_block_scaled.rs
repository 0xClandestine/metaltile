//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped block-scaled BGEMM — the block-scaled / legacy-float
//! / symmetric-int8 counterpart of `moe_mpp` (int4) and `moe_mpp_int8`.
//!
//! Algorithmically **byte-identical** to `mt_moe_gather_qmm_mma_int4_bm16_mpp`:
//! BM=16 / BN=32 / BK=16 MPP cooperative-tensor tiling, the per-TG
//! contiguous-expert sub-run walk, the X staging, the `coop_tile_*` ops, and the
//! masked coop-write tail. The **only** change is the W-dequant staging block —
//! it emits `element_decode(code) · block_scale` (no bias) per
//! `mlx/block_scaled_qmm_mpp` instead of the affine `scale·q + bias`.
//!
//! Float formats: eight kernels cover all nine (`fp8_e4m3` reuses the `nvfp8`
//! kernel — both are 8-bit E4M3 + f32 per-block scale):
//!
//! | kernel                              | element | weight | scale       |
//! |-------------------------------------|---------|--------|-------------|
//! | `mt_mxfp4_moe_gather_qmm_bm16_mpp`  | E2M1    | u32    | E8M0 (u8)   |
//! | `mt_nvfp4_moe_gather_qmm_bm16_mpp`  | E2M1    | u32    | E4M3 (u8) × global |
//! | `mt_fp4_moe_gather_qmm_bm16_mpp`    | E2M1    | u32    | f32         |
//! | `mt_mxfp8_e4m3_moe_gather_qmm_bm16_mpp` | E4M3 | u8   | E8M0 (u8)   |
//! | `mt_mxfp8_e5m2_moe_gather_qmm_bm16_mpp` | E5M2 | u8   | E8M0 (u8)   |
//! | `mt_fp8_e5m2_moe_gather_qmm_bm16_mpp`   | E5M2 | u8   | f32         |
//! | `mt_nvfp8_moe_gather_qmm_bm16_mpp`  | E4M3    | u8     | f32         |
//! | `mt_int8_moe_gather_qmm_bm16_mpp`   | int8    | u8     | f32         |
//!
//! Symmetric integer formats add eleven more kernels — two `macro_rules!`
//! families for the sub-byte bit-stream widths plus one 8-bit MXINT8 kernel:
//!
//! | kernel(s)                                    | element | weight | scale     |
//! |----------------------------------------------|---------|--------|-----------|
//! | `mt_int{2,3,4,5,6}_moe_gather_qmm_bm16_mpp`  | intN    | u32    | f32       |
//! | `mt_mxint{2,3,4,5,6}_moe_gather_qmm_bm16_mpp`| intN    | u32    | E8M0 (u8) |
//! | `mt_mxint8_moe_gather_qmm_bm16_mpp`          | int8    | u8     | E8M0 (u8) |
//!
//! Weight layout for the whole `[n_experts·n_out, k_in]` expert stack (packed in
//! ONE call so the bit-stream stays contiguous with one trailing guard word):
//! 4-bit `w [k_in/8] u32`/row (8 E2M1 nibbles/word, LSB-first), sub-byte intN
//! `w [k_in·bits/32] u32`/row (tight LSB-first bit-stream, per-row word-aligned
//! since `k_in % 32 == 0`), 8-bit `w [k_in] u8`/row (one code per byte). Scales
//! `[n_experts, n_out, k_in/block_size]` are u8 (E8M0/E4M3) or f32 (nvfp8 /
//! legacy fp / int8). No `biases` param — block-scaled is scale-only.
//!
//! ## bf16 staging
//!
//! Same `coop_stage(T)` trick as the int templates: bf16 activations stage
//! through `half` so `mpp::tensor_ops::matmul2d` sees a supported
//! cooperative-tensor dtype. Accumulation is fp32.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — identical to the int4/int8 MPP descriptor; only the
//! threadgroup W tile contents differ.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/32, ceil(m_total/16), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup — `matmul2d` is `execution_simdgroup`).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `block_size` divides `k_in`, and
//!   `block_size ≥ 16` (so the 16-element K window staged per lane per `kb`
//!   sits inside one block — one scale load per lane per `kb` is exact).
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a linkable stub.

use metaltile::kernel;

// ── 4-bit (E2M1) MoE MPP kernels — model: moe_mpp (int4) ───────────────────

/// mxfp4 MoE gather BGEMM, BM=16 / BN=32 / BK=16 — E2M1 weights, E8M0 scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]` (per-row expert id), `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    // Walk the BM=16 rows in contiguous-expert sub-runs.
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                // Stage X[m_tile_base..+16, kb..kb+16] → xs. 32 lanes × 8.
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                        threadgroup_store("ws", dst + _j, mt_decode_e2m1(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// nvfp4 MoE gather BGEMM, BM=16 / BN=32 / BK=16 — E2M1 weights, E4M3
/// micro-scale × global FP32. `global` is the LAST constexpr.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (E4M3 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let scale = mt_decode_e4m3(load(scales[sb_off]).cast::<u32>()) * global;
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, mt_decode_e2m1(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// Legacy fp4 MoE gather BGEMM, BM=16 / BN=32 / BK=16 — E2M1 weights, per-group
/// FP32 scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                        threadgroup_store("ws", dst + _j, mt_decode_e2m1(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// mxfp8 (E4M3) MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit E4M3 weights,
/// E8M0 pow-2 block scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E4M3, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let elem = mt_decode_e4m3(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// mxfp8 (E5M2) MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit E5M2 weights,
/// E8M0 pow-2 block scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E5M2, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let elem = mt_decode_e5m2(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// Legacy fp8 (E5M2) MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit E5M2
/// weights, per-group FP32 scale.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E5M2, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let elem = mt_decode_e5m2(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// nvfp8 MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit E4M3 weights, per-block
/// FP32 scale. Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape;
/// only the `block_size` constexpr differs).
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E4M3, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let elem = mt_decode_e4m3(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// Symmetric int8 MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit codes,
/// per-group FP32 scale (no bias).
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (int8 codes, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let elem = mt_decode_int8(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

// ── FP16-scale float twins (nvfp8_f16 / fp4_f16 / fp8_e5m2_f16) ─────────────
// Near-clones of the FP32-scaled float kernels above: the element decode
// (E4M3 / E2M1 / E5M2), weight indexing, dispatch geometry, staging, and
// reduction are IDENTICAL. The ONLY change is the per-block scale tensor —
// `scales: Tensor<f16>` read as `load(scales[..]).cast::<f32>()` (the native
// `half` load matches `quant::format::dequant`'s FP16 decode). See
// `mlx/block_scaled_dequant.rs`'s `mt_nvfp8_f16_dequant` etc. for the proven
// scale-read pattern.

/// nvfp8 (FP16-scale) MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit E4M3
/// weights, per-block FP16 scale. Twin of `mt_nvfp8_moe_gather_qmm_bm16_mpp`;
/// also serves **fp8_e4m3_f16** (same 8-bit-E4M3 + f16-scale shape; only the
/// `block_size` constexpr differs).
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E4M3, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                // nvfp8 f16: per-block FP16 scale (native half load → f32).
                let scale = load(scales[sb_off]).cast::<f32>();
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = mt_decode_e4m3(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// Legacy fp4 (FP16-scale) MoE gather BGEMM, BM=16 / BN=32 / BK=16 — E2M1
/// weights, per-group FP16 scale. Twin of `mt_fp4_moe_gather_qmm_bm16_mpp`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (E2M1, 8
/// nibbles/u32), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    // fp4 f16: per-group FP16 scale (native half load → f32).
                    let scale = load(scales[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, mt_decode_e2m1(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// Legacy fp8 (E5M2, FP16-scale) MoE gather BGEMM, BM=16 / BN=32 / BK=16 —
/// 8-bit E5M2 weights, per-group FP16 scale. Twin of
/// `mt_fp8_e5m2_moe_gather_qmm_bm16_mpp`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (E5M2, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                // fp8 e5m2 f16: per-group FP16 scale (native half load → f32).
                let scale = load(scales[sb_off]).cast::<f32>();
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = mt_decode_e5m2(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

// ── Symmetric sub-byte integer MoE MPP kernels (int2/3/4/5/6 + MXINT2..6) ────
// The element is a signed N-bit two's-complement code, tight-bit-packed LSB-first
// into u32 words. The WHOLE `[n_experts·n_out, k_in]` expert stack is packed in
// ONE call (the test/bench build the full stacked matrix and pack once — never
// per-expert concatenation), so it is a single contiguous bit-stream with one
// guard word at the very end. Every weight row therefore stays word-aligned
// (`k_in` a multiple of 32 ⇒ `k_in · bits % 32 == 0`), and the W global row is
// `g_row = cur_expert · n_out + (n_tile_base + w_row)` — the same flat row the
// 8-bit kernels read, just with a tight-bit-stream word base
// `g_row · (k_in · bits / 32)`. The 4-bit kernel's lane layout is reused exactly
// (32 lanes × 2 stripes/lane, 8 K-elems/stripe → BN=32 rows × BK=16 K-window);
// only the per-element decode + scale read differ. Decode is the straddle-aware
// two-word read + float sign-extend from `mlx/block_scaled_qmm_mpp`'s proven
// `int_qmm_mma_mpp_*` macros, multiplied by the block scale. `$half`/`$full` are
// 2^(N-1) / 2^N passed as literals to keep the constexpr shift math out of the
// DSL operands. **Dispatch geometry, tile sizes, coop-tensor extents, TPG, and
// grid are byte-identical to the 4-bit / 8-bit kernels above** — only the
// W-stage decode + scale read differ.

/// FP32-scaled symmetric int MoE gather BGEMM (int2/3/4/5/6), BM=16 / BN=32 /
/// BK=16 — per-element bit-stream code × per-group FP32 scale, staged into `ws`
/// and fed to the tensor engine. `w_word_row_base` indexes the gathered W row's
/// tight bit-stream (`k_in · bits / 32` u32 words per row,
/// `g_row = cur_expert · n_out + (n_tile_base + w_row)`).
///
/// Params: `x [m_total, k_in]`, `w [n_experts·n_out · k_in·bits/32]` (tight
/// bit-stream, LSB-first), `scales [n_experts, n_out, k_in/block_size]` (f32),
/// `indices [m_total]`, `out [m_total, n_out]`.
macro_rules! int_moe_gather_qmm_bm16_mpp_f32 {
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
            let m_tile_base = tgid_y * 16u32;
            let lane = simd_lane;
            let words_per_row = k_in * $bits / 32u32;
            let groups_per_row = k_in / block_size;
            threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
            threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
            threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
            coop_tile_setup(
                "gemm",
                16,
                32,
                16, // m, n, k
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
            );
            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 16u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
                // Find the run end — first row whose expert differs (or OOB).
                let mut sub_end = 16u32;
                let mut found = 0u32;
                for _ii in range(0u32, 16u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 16u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
                if cur_valid {
                    // Per-expert row base in the single stacked bit-stream pack.
                    let w_expert_row_base = cur_expert * n_out;
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    coop_tile_zero("gemm");
                    for kb in range(0u32, k_in, 16u32) {
                        // Stage X[m_tile_base..+16, kb..kb+16] → xs. 32 lanes × 8.
                        for _e in range(0u32, 8u32, 1u32) {
                            let flat = lane * 8u32 + _e;
                            let mr = flat / 16u32;
                            let kc = flat % 16u32;
                            let gr = m_tile_base + mr;
                            let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                            let safe_g = select(in_run, gr, 0u32);
                            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                        }
                        // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws.
                        // 32 lanes × 2 stripes/lane; 8 K-elems/stripe (same lane
                        // layout as the 4-bit kernel's 2 packs/lane).
                        for _pi in range(0u32, 2u32, 1u32) {
                            let pack_id = lane * 2u32 + _pi;
                            let w_row = pack_id / 2u32; // 0..31 (BN rows)
                            let stripe = pack_id % 2u32; // 0..1 (BK=16 → 2 stripes)
                            let g_row = w_expert_row_base + n_tile_base + w_row;
                            let w_word_row_base = g_row * words_per_row;
                            let k_off = kb + stripe * 8u32;
                            let g = k_off / block_size;
                            let sb_off =
                                sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                            // int: raw per-group FP32 scale.
                            let scale = load(scales[sb_off]);
                            let dst = w_row * 16u32 + stripe * 8u32;
                            for _j in range(0u32, 8u32, 1u32) {
                                let bit_off = (k_off + _j) * $bits;
                                let word_idx = bit_off / 32u32;
                                let bit_in_w = bit_off & 31u32;
                                let bits_in_w0 = 32u32 - bit_in_w;
                                let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                                let spill = $bits - lo_bits;
                                let w0 = load(w[w_word_row_base + word_idx]);
                                let w1 = load(
                                    w[w_word_row_base
                                        + select(spill > 0u32, word_idx + 1u32, word_idx)],
                                );
                                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                                let q = lo | hi;
                                let qf = q.cast::<f32>();
                                let val = select(q >= $half, qf - $full, qf); // sign-extend
                                threadgroup_store("ws", dst + _j, val * scale);
                            }
                        }
                        threadgroup_barrier();
                        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                        coop_tile_run("gemm");
                        threadgroup_barrier();
                    }
                    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
                    threadgroup_barrier();
                    // Coop-write out_scratch → out with the per-row expert mask.
                    // 32 lanes × 16 elems = 512 = BM*BN.
                    for _e in range(0u32, 16u32, 1u32) {
                        let flat = lane * 16u32 + _e;
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
int_moe_gather_qmm_bm16_mpp_f32!(mt_int2_moe_gather_qmm_bm16_mpp, 2u32, 2u32, 4.0f32);
int_moe_gather_qmm_bm16_mpp_f32!(mt_int3_moe_gather_qmm_bm16_mpp, 3u32, 4u32, 8.0f32);
int_moe_gather_qmm_bm16_mpp_f32!(mt_int4_moe_gather_qmm_bm16_mpp, 4u32, 8u32, 16.0f32);
int_moe_gather_qmm_bm16_mpp_f32!(mt_int5_moe_gather_qmm_bm16_mpp, 5u32, 16u32, 32.0f32);
int_moe_gather_qmm_bm16_mpp_f32!(mt_int6_moe_gather_qmm_bm16_mpp, 6u32, 32u32, 64.0f32);

/// FP16-scaled symmetric int MoE gather BGEMM (int2/3/4/5/6), BM=16 / BN=32 /
/// BK=16 — FP16-scale twin of `int_moe_gather_qmm_bm16_mpp_f32`. The
/// straddle-aware bit-stream decode, weight indexing, staging, geometry, and
/// reduction are IDENTICAL; only the scale tensor changes to `Tensor<f16>`,
/// read as `load(scales[..]).cast::<f32>()` (native half load → f32, matching
/// `quant::format::dequant`). Mirrors `mlx/block_scaled_dequant.rs`'s
/// `int_dequant_f16!`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts·n_out · k_in·bits/32]` (tight
/// bit-stream, LSB-first), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
macro_rules! int_moe_gather_qmm_bm16_mpp_f16 {
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
            let m_tile_base = tgid_y * 16u32;
            let lane = simd_lane;
            let words_per_row = k_in * $bits / 32u32;
            let groups_per_row = k_in / block_size;
            threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
            threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
            threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
            coop_tile_setup(
                "gemm",
                16,
                32,
                16, // m, n, k
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
            );
            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 16u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
                // Find the run end — first row whose expert differs (or OOB).
                let mut sub_end = 16u32;
                let mut found = 0u32;
                for _ii in range(0u32, 16u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 16u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
                if cur_valid {
                    // Per-expert row base in the single stacked bit-stream pack.
                    let w_expert_row_base = cur_expert * n_out;
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    coop_tile_zero("gemm");
                    for kb in range(0u32, k_in, 16u32) {
                        // Stage X[m_tile_base..+16, kb..kb+16] → xs. 32 lanes × 8.
                        for _e in range(0u32, 8u32, 1u32) {
                            let flat = lane * 8u32 + _e;
                            let mr = flat / 16u32;
                            let kc = flat % 16u32;
                            let gr = m_tile_base + mr;
                            let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                            let safe_g = select(in_run, gr, 0u32);
                            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                        }
                        // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws.
                        // 32 lanes × 2 stripes/lane; 8 K-elems/stripe (same lane
                        // layout as the 4-bit kernel's 2 packs/lane).
                        for _pi in range(0u32, 2u32, 1u32) {
                            let pack_id = lane * 2u32 + _pi;
                            let w_row = pack_id / 2u32; // 0..31 (BN rows)
                            let stripe = pack_id % 2u32; // 0..1 (BK=16 → 2 stripes)
                            let g_row = w_expert_row_base + n_tile_base + w_row;
                            let w_word_row_base = g_row * words_per_row;
                            let k_off = kb + stripe * 8u32;
                            let g = k_off / block_size;
                            let sb_off =
                                sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                            // int f16: per-group FP16 scale (native half load → f32).
                            let scale = load(scales[sb_off]).cast::<f32>();
                            let dst = w_row * 16u32 + stripe * 8u32;
                            for _j in range(0u32, 8u32, 1u32) {
                                let bit_off = (k_off + _j) * $bits;
                                let word_idx = bit_off / 32u32;
                                let bit_in_w = bit_off & 31u32;
                                let bits_in_w0 = 32u32 - bit_in_w;
                                let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                                let spill = $bits - lo_bits;
                                let w0 = load(w[w_word_row_base + word_idx]);
                                let w1 = load(
                                    w[w_word_row_base
                                        + select(spill > 0u32, word_idx + 1u32, word_idx)],
                                );
                                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                                let q = lo | hi;
                                let qf = q.cast::<f32>();
                                let val = select(q >= $half, qf - $full, qf); // sign-extend
                                threadgroup_store("ws", dst + _j, val * scale);
                            }
                        }
                        threadgroup_barrier();
                        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                        coop_tile_run("gemm");
                        threadgroup_barrier();
                    }
                    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
                    threadgroup_barrier();
                    // Coop-write out_scratch → out with the per-row expert mask.
                    // 32 lanes × 16 elems = 512 = BM*BN.
                    for _e in range(0u32, 16u32, 1u32) {
                        let flat = lane * 16u32 + _e;
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
int_moe_gather_qmm_bm16_mpp_f16!(mt_int2_f16_moe_gather_qmm_bm16_mpp, 2u32, 2u32, 4.0f32);
int_moe_gather_qmm_bm16_mpp_f16!(mt_int3_f16_moe_gather_qmm_bm16_mpp, 3u32, 4u32, 8.0f32);
int_moe_gather_qmm_bm16_mpp_f16!(mt_int4_f16_moe_gather_qmm_bm16_mpp, 4u32, 8u32, 16.0f32);
int_moe_gather_qmm_bm16_mpp_f16!(mt_int5_f16_moe_gather_qmm_bm16_mpp, 5u32, 16u32, 32.0f32);
int_moe_gather_qmm_bm16_mpp_f16!(mt_int6_f16_moe_gather_qmm_bm16_mpp, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int MoE gather BGEMM (MXINT2/3/4/5/6), BM=16 / BN=32 /
/// BK=16 — per-element bit-stream code × pow-2 (E8M0) block scale `2^(bits-127)`,
/// staged into `ws`. Same straddle-aware bit-stream decode and staging path as
/// `int_moe_gather_qmm_bm16_mpp_f32`; only the scale axis differs (one u8
/// exponent per block instead of a raw f32).
///
/// Params: `x [m_total, k_in]`, `w [n_experts·n_out · k_in·bits/32]` (tight
/// bit-stream, LSB-first), `scales [n_experts, n_out, k_in/block_size]` (E8M0
/// byte), `indices [m_total]`, `out [m_total, n_out]`.
macro_rules! int_moe_gather_qmm_bm16_mpp_e8m0 {
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
            let m_tile_base = tgid_y * 16u32;
            let lane = simd_lane;
            let words_per_row = k_in * $bits / 32u32;
            let groups_per_row = k_in / block_size;
            threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
            threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
            threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
            coop_tile_setup(
                "gemm",
                16,
                32,
                16, // m, n, k
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
            );
            let mut sub_offset = 0u32;
            for _sub_iter in range(0u32, 16u32, 1u32) {
                let cur_row = m_tile_base + sub_offset;
                let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
                let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
                // Find the run end — first row whose expert differs (or OOB).
                let mut sub_end = 16u32;
                let mut found = 0u32;
                for _ii in range(0u32, 16u32, 1u32) {
                    let probe = sub_offset + 1u32 + _ii;
                    let probe_row = m_tile_base + probe;
                    let probe_in_range = (probe < 16u32) & (probe_row < m_total);
                    if probe_in_range & (found == 0u32) {
                        let e = load(indices[probe_row]);
                        if e != cur_expert {
                            sub_end = probe;
                            found = 1u32;
                        }
                    }
                    if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                        sub_end = probe;
                        found = 1u32;
                    }
                }
                let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
                if cur_valid {
                    // Per-expert row base in the single stacked bit-stream pack.
                    let w_expert_row_base = cur_expert * n_out;
                    let sb_expert_base = cur_expert * n_out * groups_per_row;
                    coop_tile_zero("gemm");
                    for kb in range(0u32, k_in, 16u32) {
                        for _e in range(0u32, 8u32, 1u32) {
                            let flat = lane * 8u32 + _e;
                            let mr = flat / 16u32;
                            let kc = flat % 16u32;
                            let gr = m_tile_base + mr;
                            let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                            let safe_g = select(in_run, gr, 0u32);
                            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                        }
                        // 32 lanes × 2 stripes/lane; 8 K-elems/stripe.
                        for _pi in range(0u32, 2u32, 1u32) {
                            let pack_id = lane * 2u32 + _pi;
                            let w_row = pack_id / 2u32; // 0..31 (BN rows)
                            let stripe = pack_id % 2u32; // 0..1 (BK=16 → 2 stripes)
                            let g_row = w_expert_row_base + n_tile_base + w_row;
                            let w_word_row_base = g_row * words_per_row;
                            let k_off = kb + stripe * 8u32;
                            let g = k_off / block_size;
                            let sb_off =
                                sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                            // mxint: E8M0 pow-2 block scale → 2^(bits-127).
                            let scale = exp2(load(scales[sb_off]).cast::<f32>() - 127.0f32);
                            let dst = w_row * 16u32 + stripe * 8u32;
                            for _j in range(0u32, 8u32, 1u32) {
                                let bit_off = (k_off + _j) * $bits;
                                let word_idx = bit_off / 32u32;
                                let bit_in_w = bit_off & 31u32;
                                let bits_in_w0 = 32u32 - bit_in_w;
                                let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                                let spill = $bits - lo_bits;
                                let w0 = load(w[w_word_row_base + word_idx]);
                                let w1 = load(
                                    w[w_word_row_base
                                        + select(spill > 0u32, word_idx + 1u32, word_idx)],
                                );
                                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                                let q = lo | hi;
                                let qf = q.cast::<f32>();
                                let val = select(q >= $half, qf - $full, qf); // sign-extend
                                threadgroup_store("ws", dst + _j, val * scale);
                            }
                        }
                        threadgroup_barrier();
                        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                        coop_tile_run("gemm");
                        threadgroup_barrier();
                    }
                    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
                    threadgroup_barrier();
                    for _e in range(0u32, 16u32, 1u32) {
                        let flat = lane * 16u32 + _e;
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
int_moe_gather_qmm_bm16_mpp_e8m0!(mt_mxint2_moe_gather_qmm_bm16_mpp, 2u32, 2u32, 4.0f32);
int_moe_gather_qmm_bm16_mpp_e8m0!(mt_mxint3_moe_gather_qmm_bm16_mpp, 3u32, 4u32, 8.0f32);
int_moe_gather_qmm_bm16_mpp_e8m0!(mt_mxint4_moe_gather_qmm_bm16_mpp, 4u32, 8u32, 16.0f32);
int_moe_gather_qmm_bm16_mpp_e8m0!(mt_mxint5_moe_gather_qmm_bm16_mpp, 5u32, 16u32, 32.0f32);
int_moe_gather_qmm_bm16_mpp_e8m0!(mt_mxint6_moe_gather_qmm_bm16_mpp, 6u32, 32u32, 64.0f32);

/// MXINT8 MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit symmetric codes (byte
/// layout, block 32), E8M0 pow-2 block scale `2^(bits-127)`. Byte-strided
/// staging like the int8 / mxfp8 kernels (one byte per code); decode is
/// `mt_decode_int8 → val · scale`. Geometry and coop-tensor extents are
/// byte-identical to the int8 kernel.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (int8 codes, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (E8M0 byte),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                    let elem = mt_decode_int8(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

/// Symmetric int8 (FP16-scale) MoE gather BGEMM, BM=16 / BN=32 / BK=16 — 8-bit
/// codes (byte layout, group 64), per-group FP16 scale (no bias). FP16-scale
/// twin of `mt_int8_moe_gather_qmm_bm16_mpp`; identical byte-strided staging and
/// `mt_decode_int8`, only the scale tensor is `Tensor<f16>` (native half load →
/// f32). Mirrors `mlx/block_scaled_dequant.rs`'s `mt_int8_f16_dequant`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in]` (int8 codes, 1
/// byte/elem), `scales [n_experts, n_out, k_in/block_size]` (f16),
/// `indices [m_total]`, `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_moe_gather_qmm_bm16_mpp<T>(
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
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let groups_per_row = k_in / block_size;
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base_8 = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
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
                // int8 f16: per-group FP16 scale (native half load → f32).
                let scale = load(scales[sb_off]).cast::<f32>();
                let w_dev = w_expert_base_8 + (n_tile_base + w_row) * k_in + kb;
                for kc in range(0u32, 16u32, 1u32) {
                    let elem = mt_decode_int8(load(w[w_dev + kc]).cast::<u32>());
                    threadgroup_store("ws", w_row * 16u32 + kc, elem * scale);
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
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

    /// Every block-scaled MoE kernel builds, drops to `CoopTile*` ops (no raw
    /// inline MSL), and has the 5-tensor / 4-constexpr ABI (nvfp4 adds the
    /// `global` constexpr → 5).
    #[test]
    fn kernels_construct_and_use_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let kernels = [
                ("mt_mxfp4_moe_gather_qmm_bm16_mpp", 4usize),
                ("mt_fp4_moe_gather_qmm_bm16_mpp", 4),
                ("mt_mxfp8_e4m3_moe_gather_qmm_bm16_mpp", 4),
                ("mt_mxfp8_e5m2_moe_gather_qmm_bm16_mpp", 4),
                ("mt_fp8_e5m2_moe_gather_qmm_bm16_mpp", 4),
                ("mt_nvfp8_moe_gather_qmm_bm16_mpp", 4),
                ("mt_int8_moe_gather_qmm_bm16_mpp", 4),
            ];
            let irs = [
                mt_mxfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
                mt_fp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
                mt_mxfp8_e4m3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
                mt_mxfp8_e5m2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
                mt_fp8_e5m2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
                mt_nvfp8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
                mt_int8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
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
        let k = mt_nvfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(DType::F32);
        assert_eq!(k.params.len(), 5, "5 tensor params (no biases)");
        assert_eq!(k.constexprs.len(), 5, "m_total/n_out/k_in/block_size/global");
        assert_eq!(k.constexprs[4].name.name(), "global", "global is the last constexpr");
    }

    /// bf16 must stage through `half`: the `coop_stage(T)` tiles and
    /// cooperative tensors resolve to `half`, never `bfloat`.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_int8_moe_gather_qmm_bm16_mpp::kernel_ir_for(DType::BF16);
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
        let mut k = mt_mxfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(DType::F32);
        k.name = "mt_mxfp4_moe_gather_qmm_bm16_mpp_f32".into();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_mxfp4_moe_gather_qmm_bm16_mpp_f32"));
    }
}

/// New-syntax correctness tests for the MPP block-scaled MoE BGEMM (BM=16).
///
/// Oracle is the clean per-row-`indices` block-scaled dequant-then-matmul: each
/// row `t` resolves its expert from `indices[t]`, dequantizes that expert's
/// `[n_out, k_in]` weight slab via the shared `quant::format::dequant`, and dots
/// against the row's input. Inputs are dtype-rounded so the GPU sees exactly
/// what the oracle computes; tolerance is wide because the MPP cooperative-tensor
/// accumulator reorders the K reduction. `fp8_e4m3` dispatches the `nvfp8`
/// kernel with `QFormat::Fp8E4m3` (same 8-bit-E4M3 + f32-scale shape).
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(n_out/32, ceil(m_total/16), 1, [32,1,1])`.
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

    /// Build a `TestSetup` for a block-scaled indexed-MoE-MPP kernel. Mirrors
    /// `int8_indexed_setup`: per-row expert routing, dtype-rounded x, oracle =
    /// `Σ_k x[t,k] · dequant(W_expert)[nc,k]`. The whole `[n_experts·n_out, k_in]`
    /// expert stack is packed in ONE call (no biases buffer; scale dtype per
    /// format; weight dtype U32 for sub-byte bit-streams / U8 for 8-bit codes).
    /// `block_size` and the nvfp4 `global` constexpr come from the packed tensors.
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
        // 4-bit / 8-bit formats (those widths divide 32 ⇒ exact word count, no
        // guard word) and correct for every sub-byte width. `k_in` is a multiple
        // of 32, so each row's bit-stream is word-aligned for every width.
        // Mirrors the magnitude pattern used by the non-MoE block-scaled test,
        // keyed off the stacked global row's expert index `r / n_out`.
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
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
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
            (m_total as u32).div_ceil(16),
            1,
            [32, 1, 1],
        )
    }

    // n_experts=4, m_total=64, n_out=64, k_in=64 (divisible by 16/32/64).
    const SHAPE: BlockTestShape = BlockTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp4_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxfp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Nvfp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp8_e4m3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp8_e5m2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp8_e5m2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8,
            SHAPE,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int8,
            SHAPE,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). k_in=64 is a multiple of 32, so
    // `k_in*bits % 32 == 0` for every width and each stacked weight row's
    // bit-stream is word-aligned. The whole `[n_experts·n_out, k_in]` stack is
    // packed once, so the bit-stream stays contiguous (one guard word at the very
    // end) and kernel/oracle share the codec — the GPU output tracks the
    // dequant-then-matmul reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int2_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int3_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int4_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int5_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int5_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int6_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int6_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int6,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint2_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint3_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint4_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint5_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint5_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint6_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint6_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint6,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint8_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint8,
            SHAPE,
            dt,
        )
    }

    // FP16-scale twins of the FP32-scaled float + int formats. Same element
    // packing (codes dtype unchanged); the scale buffer is f16. `fp8_e4m3_f16`
    // reuses the `nvfp8_f16` kernel (8-bit E4M3 + f16 scale; only block_size
    // differs).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            SHAPE,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp4_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp4F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp8_e5m2_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int2_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int2_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int2F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int3_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int3_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int3F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int4_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int4_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int4F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int5_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int5_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int5F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int6_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int6_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int6F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_f16_moe_gather_qmm_bm16_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int8_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int8F16,
            SHAPE,
            dt,
        )
    }
}

/// New-syntax benchmarks for the MPP block-scaled MoE BGEMM (BM=16). Random
/// buffers; `flops = 2·m_total·n_out·k_in` (the gather does a full matmul per
/// row's expert — dense-equivalent FLOPs).
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(n_out/32, m_total.div_ceil(16), 1, [32,1,1])`.
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
        // (`bitstream_words` collapses to the old `n/8` for the 4-bit case). Both
        // axes are driven off the format.
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
        s.grid_3d(n_out as u32 / 32, (m_total as u32).div_ceil(16), 1, [32, 1, 1])
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
        block_bench(mt_mxfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Mxfp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Nvfp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4(dt: DType) -> BenchSetup {
        block_bench(mt_fp4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Fp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxfp8_e4m3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxfp8_e5m2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp8_e5m2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Nvfp8, SHAPE, dt)
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        block_bench(mt_int8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Int8, SHAPE, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). k_in=4096 is a multiple of 32 → word-aligned per width.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2(dt: DType) -> BenchSetup {
        block_bench(mt_int2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Int2, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3(dt: DType) -> BenchSetup {
        block_bench(mt_int3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Int3, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4(dt: DType) -> BenchSetup {
        block_bench(mt_int4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Int4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5(dt: DType) -> BenchSetup {
        block_bench(mt_int5_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Int5, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6(dt: DType) -> BenchSetup {
        block_bench(mt_int6_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt), QFormat::Int6, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint2_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint2,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint3_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint3,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint4_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint4,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint5_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint5,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint6_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint6,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint8_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Mxint8,
            SHAPE,
            dt,
        )
    }
    // FP16-scale twins (f16 scale buffer; codes packing unchanged). fp8_e4m3_f16
    // reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp4_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp4F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp8_e5m2_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int2_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int2F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int3_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int3F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int4_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int4F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int5_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int5F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int6_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int6F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int8_f16_moe_gather_qmm_bm16_mpp::kernel_ir_for(dt),
            QFormat::Int8F16,
            SHAPE,
            dt,
        )
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm64_mpp`.
//!
//! BM=BN=64, BK=32 variant of the MPP MoE kernel. Where `…_bm16_mpp`
//! runs one simdgroup over a 16×32 tile, this runs **4 simdgroups** in a
//! 2×2 warp grid over a 64×64 tile — each SG owns a 32×32 sub-tile and a
//! 32×32×32 `matmul2d`. For long-context prefill the larger tile amortizes
//! the int4 dequant across more output.
//!
//! Expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics —
//! no `Op::InlineMsl`. Each SG's `coop_tile_load_*` / `coop_tile_store_c`
//! takes a per-SG offset into the shared `Xs` / `Ws` / `OutScratch`
//! threadgroup buffers.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(32, 32, 32, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — all dims 32, so the inputs are cooperative
//! tensors (not the direct-input path the `…_bm8` variant needs).
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for `T = bf16`, else `T`. Apple's `matmul2d`
//! mishandles `bfloat` cooperative tensors; `half` losslessly covers
//! bf16's mantissa. Accumulation is fp32.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/64, ceil(M/64), 1]`; threadgroup
//!   `[128, 1, 1]` (4 simdgroups, 2×2 warp grid).
//! - `k_in % 32 == 0`, `n_out % 64 == 0`, `group_size` divides `k_in`.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_bm64_correctness.rs`.

use metaltile::kernel;

/// MPP MoE int4 grouped BGEMM, BM=BN=64 / BK=32, 4 simdgroups (2×2).
/// Signature matches `…_bm16_mpp`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int4_bm64_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    // 2×2 warp grid: sm/sn select this SG's 32×32 sub-tile.
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    // X coop-load: 128 lanes × 16 contiguous K = 2048 = BM(64)×TG_LD(32).
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    threadgroup_alloc("Xs", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("Ws", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("OutScratch", 4096, f32); // 4 SG × 32 × 32
    // Descriptor 32×32×32, cooperative-tensor inputs, accumulate.
    coop_tile_setup(
        "gemm",
        32,
        32,
        32, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 64u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 64u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 64u32;
        let mut found = 0u32;
        for _ii in range(0u32, 64u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 64u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 64u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 64u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                // Stage X[m_tile_base..+64, kb..kb+32] → Xs. 128 lanes × 16.
                let gr_x = m_tile_base + x_m_row;
                let in_run_x = (x_m_row >= sub_offset) & (x_m_row < sub_end) & (gr_x < m_total);
                let safe_gr_x = select(in_run_x, gr_x, 0u32);
                let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
                let x_ws_base = x_m_row * 32u32 + x_k_base;
                for _i in range(0u32, 16u32, 1u32) {
                    let xv = load(x[x_dev_base + _i]).cast::<f32>();
                    threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
                }
                // Dequant W → Ws. 128 lanes × 2 packs/lane = 256 packs.
                for _pi in range(0u32, 2u32, 1u32) {
                    let pack_id = lane_in_tg * 2u32 + _pi;
                    let w_row = pack_id / 4u32; // 0..63 (BN rows)
                    let pack_in_row = pack_id & 3u32; // 0..3 (BK=32 → 4 packs)
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 8u32
                        + pack_in_row;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_in_row * 8u32;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let ws_base = w_row * 32u32 + pack_in_row * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let q = ((packed >> (_j * 4u32)) & 15u32).cast::<f32>();
                        threadgroup_store("Ws", ws_base + _j, s * q + b);
                    }
                }
                threadgroup_barrier();
                // Per-SG 32×32 sub-tile views into Xs / Ws (offset by the
                // SG's 32-row span × TG_LD=32). extents<32, 32> = K-inner.
                coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
                coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            // Store this SG's 32×32 fp32 result into its OutScratch slot.
            coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
            threadgroup_barrier();
            // Coop-write OutScratch → out. 128 lanes × 32 = 4096 = BM*BN.
            // Each (mr, nc) lives in SG `(mr/32)*2 + (nc/32)`'s scratch.
            for _e in range(0u32, 32u32, 1u32) {
                let flat = lane_in_tg * 32u32 + _e;
                let mr = flat / 64u32;
                let nc = flat & 63u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
                    let v = threadgroup_load(
                        "OutScratch",
                        src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
                    );
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
    use metaltile_core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_moe_gather_qmm_mma_int4_bm64_mpp::kernel_ir_for(dt);
            assert_eq!(k.params.len(), 6);
            assert_eq!(k.constexprs.len(), 4);
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    fn bf16_stages_through_half() {
        let k = mt_moe_gather_qmm_mma_int4_bm64_mpp::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half");
    }
}

#[cfg(target_os = "macos")]
pub mod tests_support {
    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::{super::moe::mt_moe_gather_qmm_int4, mt_moe_gather_qmm_mma_int4_bm64_mpp};

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    #[derive(Clone, Copy, Debug)]
    enum Dt {
        F32,
        Bf16,
    }
    impl Dt {
        fn to_dtype(self) -> DType {
            match self {
                Dt::F32 => DType::F32,
                Dt::Bf16 => DType::BF16,
            }
        }
    }
    fn pack_bytes_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_bytes_bf16(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect()
    }
    fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
        match dt {
            Dt::F32 => pack_bytes_f32(vals),
            Dt::Bf16 => pack_bytes_bf16(vals),
        }
    }
    fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            Dt::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        }
    }
    fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
        assert!(weights.len() % 8 == 0);
        weights
            .chunks_exact(8)
            .map(|chunk| {
                let mut packed = 0u32;
                for (i, &q) in chunk.iter().enumerate() {
                    packed |= (q & 0xf) << (i * 4);
                }
                packed
            })
            .collect()
    }
    fn skip_unless_apple10(test_name: &str) -> Option<Context> {
        let ctx = Context::new().expect("Context::new");
        let family = ctx.chip_family();
        if family.is_none_or(|lvl| lvl < 10) {
            eprintln!("skip {test_name}: needs Apple10+ GPU (chip_family={family:?})");
            return None;
        }
        Some(ctx)
    }

    fn run_m1(
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        expert_offsets: &[u32],
        n_out: usize,
        t_rows: usize,
        k_in: usize,
        group_size: usize,
        n_experts: usize,
    ) -> Vec<f32> {
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("x".into(), pack_bytes_f32(x));
        b.insert(
            "weight_packed".into(),
            weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
        );
        b.insert("scales".into(), pack_bytes_f32(scales));
        b.insert("biases".into(), pack_bytes_f32(biases));
        b.insert(
            "expert_offsets".into(),
            expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        );
        b.insert("out".into(), pack_bytes_f32(&vec![0.0_f32; t_rows * n_out]));
        b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        b.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
        b.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
        b.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_int4::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(&k, &b, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1])
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
    }

    fn run_bm64(
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        indices: &[u32],
        n_out: usize,
        t_rows: usize,
        k_in: usize,
        group_size: usize,
        dt: Dt,
    ) -> Vec<f32> {
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("x".into(), pack_bytes(x, dt));
        b.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
        b.insert("scales".into(), pack_bytes(scales, dt));
        b.insert("biases".into(), pack_bytes(biases, dt));
        b.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], dt));
        b.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
        b.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
        b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
        b.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().unwrap();
        let mut k = mt_moe_gather_qmm_mma_int4_bm64_mpp::kernel_ir_for(dt.to_dtype());
        k.mode = KernelMode::Reduction;
        let r = ctx
            .dispatch_with_grid(
                &k,
                &b,
                &BTreeMap::new(),
                [n_out.div_ceil(64), t_rows.div_ceil(64), 1],
                [128, 1, 1],
            )
            .unwrap();
        unpack_bytes(r.outputs.get("out").unwrap(), dt)
    }

    fn cosine_f64(a: &[f32], b: &[f32]) -> (f64, usize) {
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        let mut nan_count = 0usize;
        for (x, y) in a.iter().zip(b) {
            if !x.is_finite() || !y.is_finite() {
                nan_count += 1;
                continue;
            }
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12), nan_count)
    }

    #[test]
    fn moe_gather_qmm_mma_int4_bm64_mpp_matches_m1_clean_tile() {
        let _g = gpu_lock();
        let Some(_ctx) = skip_unless_apple10("bm64_mpp_clean_tile") else { return };
        let n_experts = 4usize;
        let k_in = 64usize;
        let n_out = 64usize;
        let group_size = 32usize;
        let t_rows = 64usize;
        let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
        let total_weights = n_experts * n_out * k_in;
        let weight_unpacked: Vec<u32> =
            (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
        let weight_packed: Vec<u32> =
            weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
        let groups_total = n_experts * n_out * (k_in / group_size);
        let scales: Vec<f32> =
            (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
        let biases: Vec<f32> =
            (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
        let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();
        let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
        for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
            *off = indices
                .iter()
                .position(|&e| e as usize >= e_idx)
                .map(|p| p as u32)
                .unwrap_or(t_rows as u32);
        }
        expert_offsets[n_experts] = t_rows as u32;
        let y_m1 = run_m1(
            &weight_packed,
            &scales,
            &biases,
            &x,
            &expert_offsets,
            n_out,
            t_rows,
            k_in,
            group_size,
            n_experts,
        );
        let y_mpp = run_bm64(
            &weight_packed,
            &scales,
            &biases,
            &x,
            &indices,
            n_out,
            t_rows,
            k_in,
            group_size,
            Dt::F32,
        );
        let (cos, nan_count) = cosine_f64(&y_m1, &y_mpp);
        assert_eq!(nan_count, 0, "MPP BM=64 kernel produced non-finite values");
        assert!(cos >= 0.999, "MPP MoE BM=64 vs m1 cosine = {cos:.6} (want ≥ 0.999)");
    }

    #[test]
    fn moe_gather_qmm_mma_int4_bm64_mpp_matches_m1_multi_tile() {
        let _g = gpu_lock();
        let Some(_ctx) = skip_unless_apple10("bm64_mpp_multi_tile") else { return };
        let n_experts = 8usize;
        let k_in = 128usize;
        let n_out = 128usize;
        let group_size = 64usize;
        let t_rows = 128usize;
        let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
        let total_weights = n_experts * n_out * k_in;
        let weight_unpacked: Vec<u32> =
            (0..total_weights).map(|i| ((i as u32) * 11 + 5) & 0xf).collect();
        let weight_packed: Vec<u32> =
            weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
        let groups_total = n_experts * n_out * (k_in / group_size);
        let scales: Vec<f32> =
            (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.07).sin()).collect();
        let biases: Vec<f32> =
            (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.11).cos()).collect();
        let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.017).sin()).collect();
        let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
        for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
            *off = indices
                .iter()
                .position(|&e| e as usize >= e_idx)
                .map(|p| p as u32)
                .unwrap_or(t_rows as u32);
        }
        expert_offsets[n_experts] = t_rows as u32;
        let y_m1 = run_m1(
            &weight_packed,
            &scales,
            &biases,
            &x,
            &expert_offsets,
            n_out,
            t_rows,
            k_in,
            group_size,
            n_experts,
        );
        let y_mpp = run_bm64(
            &weight_packed,
            &scales,
            &biases,
            &x,
            &indices,
            n_out,
            t_rows,
            k_in,
            group_size,
            Dt::F32,
        );
        let (cos, nan_count) = cosine_f64(&y_m1, &y_mpp);
        assert_eq!(nan_count, 0, "MPP BM=64 kernel produced non-finite values (multi-tile)");
        assert!(cos >= 0.999, "MPP MoE BM=64 vs m1 cosine = {cos:.6} (want ≥ 0.999) (multi-tile)");
    }

    #[test]
    fn moe_gather_qmm_mma_int4_bm64_mpp_bf16_matches_m1_clean_tile() {
        let _g = gpu_lock();
        let Some(_ctx) = skip_unless_apple10("bm64_mpp_bf16_clean_tile") else { return };
        let n_experts = 4usize;
        let k_in = 64usize;
        let n_out = 64usize;
        let group_size = 32usize;
        let t_rows = 64usize;
        let indices: Vec<u32> = (0..t_rows).map(|r| (r / (t_rows / n_experts)) as u32).collect();
        let total_weights = n_experts * n_out * k_in;
        let weight_unpacked: Vec<u32> =
            (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
        let weight_packed: Vec<u32> =
            weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
        let groups_total = n_experts * n_out * (k_in / group_size);
        let scales: Vec<f32> =
            (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin()).collect();
        let biases: Vec<f32> =
            (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos()).collect();
        let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();
        let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
        for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
            *off = indices
                .iter()
                .position(|&e| e as usize >= e_idx)
                .map(|p| p as u32)
                .unwrap_or(t_rows as u32);
        }
        expert_offsets[n_experts] = t_rows as u32;
        let y_m1 = run_m1(
            &weight_packed,
            &scales,
            &biases,
            &x,
            &expert_offsets,
            n_out,
            t_rows,
            k_in,
            group_size,
            n_experts,
        );
        let y_mpp = run_bm64(
            &weight_packed,
            &scales,
            &biases,
            &x,
            &indices,
            n_out,
            t_rows,
            k_in,
            group_size,
            Dt::Bf16,
        );
        let (cos, nan_count) = cosine_f64(&y_m1, &y_mpp);
        assert_eq!(nan_count, 0, "MPP BM=64 bf16 kernel produced non-finite values");
        assert!(cos >= 0.997, "MPP MoE BM=64 bf16 vs m1 cosine = {cos:.6} (want ≥ 0.997)");
    }
}

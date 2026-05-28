//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped int4 BGEMM — `mt_moe_gather_qmm_mma_int4_bm16_mpp`.
//!
//! Routes the per-tile matmul through Apple's MetalPerformancePrimitives
//! `mpp::tensor_ops::matmul2d`. Algorithmically mirrors
//! `mt_moe_gather_qmm_mma_int4_bm16` (BM=16, BN=32, per-TG expert
//! sub-runs, per-row expert dispatch); the inner `simdgroup_matmul`
//! 8×8 frags are replaced by a single `16×32×16` MPP descriptor.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`. The `coop_tile_*` ops lower to the
//! `mpp::tensor_ops::matmul2d` cooperative-tensor calls; codegen emits
//! the framework include automatically.
//!
//! ## bf16 staging
//!
//! Apple's `matmul2d` mishandles `bfloat` cooperative tensors, so bf16
//! activations are staged through `half` (10-bit mantissa losslessly
//! covers bf16's 7; accumulation is fp32 regardless). The DSL
//! `coop_stage(T)` form yields `half` for `T = bf16` and `T` otherwise —
//! the kernel stays generic over `T` while its threadgroup tiles and
//! cooperative tensors pick up the staged type.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — `N=32` satisfies Apple's "at least one of
//! M/N/K = 32" rule; `tb=true` reads W in its native `[N, K]` layout;
//! `multiply_accumulate` spans the K loop without an explicit add.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/32, ceil(M/16), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup — `matmul2d` is `execution_simdgroup`).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`.
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a
//!   linkable stub.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_correctness.rs`
//! (cosine ≥ 0.999 vs the m1 scalar oracle).

use metaltile::kernel;

/// MPP MoE int4 grouped BGEMM, BM=16 / BN=32 / BK=16, one simdgroup.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in/8]` (int4
/// packed, 8 nibbles/uint32), `scales`/`biases [n_experts, n_out,
/// k_in/group]`, `indices [m_total]` (per-row expert id), `out
/// [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int4_bm16_mpp<T>(
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
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    // Threadgroup staging tiles. `coop_stage(T)` = half for bf16, else T —
    // the matmul reads these as cooperative tensors. `out_scratch` is
    // fp32: `coop_tile_store_c` requires the destination elem-type to
    // match the accumulator.
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    // MPP descriptor 16×32×16, ta=false tb=true tc=false, accumulate.
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
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let q = ((packed >> (_j * 4u32)) & 15u32).cast::<f32>();
                        threadgroup_store("ws", dst + _j, s * q + b);
                    }
                }
                threadgroup_barrier();
                // A = xs [M=16, K=16] (ta=false → extents K,M = 16,16).
                // B = ws [N=32, K=16] (tb=true  → extents K,N = 16,32).
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            // C [M=16, N=32] row-major → extents N,M = 32,16.
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
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_moe_gather_qmm_mma_int4_bm16_mpp");
            assert_eq!(k.params.len(), 6);
            assert!(k.params[5].is_output);
            assert_eq!(k.constexprs.len(), 4);
            // No raw inline MSL — the matmul is CoopTile* ops.
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    /// bf16 must stage through `half`: the `coop_stage(T)` tiles and
    /// cooperative tensors resolve to `half`, never `bfloat`.
    #[test]
    fn bf16_stages_through_half() {
        let k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::BF16);
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
        let mut k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::F32);
        k.name = "mt_moe_gather_qmm_mma_int4_bm16_mpp_f32".into();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_moe_gather_qmm_mma_int4_bm16_mpp_f32"));
    }
}

#[cfg(target_os = "macos")]
pub mod tests_support {
    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::{super::moe::mt_moe_gather_qmm_int4, mt_moe_gather_qmm_mma_int4_bm16_mpp};

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    #[derive(Clone, Copy, Debug)]
    enum Dt {
        F32,
        F16,
        Bf16,
    }
    impl Dt {
        fn to_dtype(self) -> DType {
            match self {
                Dt::F32 => DType::F32,
                Dt::F16 => DType::F16,
                Dt::Bf16 => DType::BF16,
            }
        }
    }
    fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
        }
    }
    fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            Dt::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
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

    #[test]
    fn moe_gather_qmm_mma_int4_bm16_mpp_matches_m1_clean_tile() {
        let _g = gpu_lock();
        let probe = Context::new().expect("Context::new");
        let family = probe.chip_family();
        if family.is_none_or(|lvl| lvl < 10) {
            eprintln!("skip bm16_mpp_clean_tile: needs Apple10+ GPU (chip_family={family:?})");
            return;
        }
        drop(probe);
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
        let y_m1 = {
            let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            b.insert("x".into(), pack_bytes(&x, Dt::F32));
            b.insert(
                "weight_packed".into(),
                weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
            );
            b.insert("scales".into(), pack_bytes(&scales, Dt::F32));
            b.insert("biases".into(), pack_bytes(&biases, Dt::F32));
            b.insert(
                "expert_offsets".into(),
                expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
            );
            b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
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
        };
        let y_mpp = {
            let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            b.insert("x".into(), pack_bytes(&x, Dt::F32));
            b.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
            b.insert("scales".into(), pack_bytes(&scales, Dt::F32));
            b.insert("biases".into(), pack_bytes(&biases, Dt::F32));
            b.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
            b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
            b.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
            b.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
            b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
            b.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
            let ctx = Context::new().unwrap();
            let mut k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::F32);
            k.mode = KernelMode::Reduction;
            let r = ctx
                .dispatch_with_grid(
                    &k,
                    &b,
                    &BTreeMap::new(),
                    [n_out / 32, t_rows.div_ceil(16), 1],
                    [32, 1, 1],
                )
                .unwrap();
            unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
        };
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        let mut nan_count = 0usize;
        for (a, b) in y_m1.iter().zip(&y_mpp) {
            if !a.is_finite() || !b.is_finite() {
                nan_count += 1;
                continue;
            }
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert_eq!(nan_count, 0, "MPP kernel produced non-finite values");
        assert!(cos >= 0.999, "MPP MoE vs m1 cosine = {cos:.6} (want ≥ 0.999)");
    }

    #[test]
    fn moe_gather_qmm_mma_int4_bm16_mpp_bf16_matches_m1_clean_tile() {
        let _g = gpu_lock();
        let probe = Context::new().expect("Context::new");
        let family = probe.chip_family();
        if family.is_none_or(|lvl| lvl < 10) {
            eprintln!("skip bm16_mpp_bf16_clean_tile: needs Apple10+ GPU (chip_family={family:?})");
            return;
        }
        drop(probe);
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
        let y_m1 = {
            let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            b.insert("x".into(), pack_bytes(&x, Dt::F32));
            b.insert(
                "weight_packed".into(),
                weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
            );
            b.insert("scales".into(), pack_bytes(&scales, Dt::F32));
            b.insert("biases".into(), pack_bytes(&biases, Dt::F32));
            b.insert(
                "expert_offsets".into(),
                expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
            );
            b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
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
        };
        let y_mpp = {
            let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            b.insert("x".into(), pack_bytes(&x, Dt::Bf16));
            b.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
            b.insert("scales".into(), pack_bytes(&scales, Dt::Bf16));
            b.insert("biases".into(), pack_bytes(&biases, Dt::Bf16));
            b.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
            b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::Bf16));
            b.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
            b.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
            b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
            b.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
            let ctx = Context::new().unwrap();
            let mut k = mt_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::BF16);
            k.mode = KernelMode::Reduction;
            let r = ctx
                .dispatch_with_grid(
                    &k,
                    &b,
                    &BTreeMap::new(),
                    [n_out / 32, t_rows.div_ceil(16), 1],
                    [32, 1, 1],
                )
                .unwrap();
            unpack_bytes(r.outputs.get("out").unwrap(), Dt::Bf16)
        };
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        let mut nan_count = 0usize;
        for (a, b) in y_m1.iter().zip(&y_mpp) {
            if !a.is_finite() || !b.is_finite() {
                nan_count += 1;
                continue;
            }
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert_eq!(nan_count, 0, "MPP bf16 kernel produced non-finite values");
        assert!(cos >= 0.997, "MPP MoE bf16 vs m1 cosine = {cos:.6} (want ≥ 0.997)");
    }
}

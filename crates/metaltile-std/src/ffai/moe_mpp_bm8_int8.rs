//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped int8 BGEMM — `mt_moe_gather_qmm_mma_int8_bm8_mpp`.
//!
//! BM=8 int8 sibling of `mt_moe_gather_qmm_mma_int4_bm8_mpp`. Same algorithm
//! and call-site signature; the weight layout changes from int4 (8 nibbles/u32)
//! to int8 (4 bytes/u32), doubling the number of weight u32s per row but
//! halving the packing inner-loop work per word.
//!
//! ## Direct-input matmul2d
//!
//! Descriptor `matmul2d_descriptor(8, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)`. With M=8 the inputs cannot be cooperative tensors
//! (Apple's MPP path requires at least one of M/N/K ≥ 16 for cooperative
//! tensor descriptors), so A and B are passed as **direct** `metal::tensor`
//! views over threadgroup memory — the `direct_inputs` form.
//!
//! ## int4 → int8 lane mapping (BM=8)
//!
//! W tile size: BN(32) × BK(16) = 512 elements.
//!
//! - **int4**: 32 lanes × 2 packs/lane × 8 nibbles/pack = 512 ✓
//!   - pack_id = lane*2 + _pi; w_row = pack_id/2; pack_col = pack_id%2
//!   - k_off = kb + pack_col*8; dst = w_row*16 + pack_col*8
//!   - Extracts 8 nibbles: `(packed >> (j*4)) & 0xf`
//!
//! - **int8**: 32 lanes × 4 packs/lane × 4 bytes/pack = 512 ✓
//!   - pack_id = lane*4 + _pi; w_row = pack_id/4; pack_col = pack_id%4
//!   - k_off = kb + pack_col*4; dst = w_row*16 + pack_col*4
//!   - Extracts 4 bytes: `(packed >> (j*8)) & 0xff`
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for `T = bf16`, else `T` — Apple's `matmul2d`
//! mishandles `bfloat` operands, and `half` losslessly covers bf16's
//! mantissa. Accumulation is fp32.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/32, ceil(M/8), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup).
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_bm8_int8_correctness.rs`.

use metaltile::kernel;

/// MPP MoE int8 grouped BGEMM, BM=8 / BN=32 / BK=16, one simdgroup,
/// direct-input `matmul2d`. Signature matches `…_int4_bm8_mpp`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_moe_gather_qmm_mma_int8_bm8_mpp<T>(
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
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    // int8: 4 bytes per u32 → k_in / 4 packs per weight row.
    let packs_per_row = k_in / 4u32;
    let groups_per_row = k_in / group_size;
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
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 8u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 8u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Walk forward to find the first row whose expert differs, clamping
        // sub_end at the tile boundary or at m_total.
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
                // Dequant W → ws.
                //
                // int8 lane mapping: 32 lanes × 4 packs/lane × 4 bytes/pack
                //   = 512 = BN(32) × BK(16).
                //
                // pack_id = lane*4 + _pi   (0..127 — covers 32 w_rows × 4 packs/row)
                // w_row   = pack_id / 4    (0..31 = BN rows)
                // pack_col= pack_id % 4    (0..3 — selects which of the 4 u32s in BK)
                //
                // k_off = kb + pack_col*4  (byte-offset of this pack's first element)
                // dst   = w_row*16 + pack_col*4 (flat index into ws threadgroup buf)
                //
                // Each pack holds 4 bytes (one per K-element); inner _j in 0..4
                // extracts byte j via (packed >> (j*8)) & 0xff.
                for _pi in range(0u32, 4u32, 1u32) {
                    let pack_id = lane * 4u32 + _pi;
                    let w_row = pack_id / 4u32;
                    let pack_col = pack_id % 4u32;
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / 4u32
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * 4u32;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * 4u32;
                    for _j in range(0u32, 4u32, 1u32) {
                        let q = ((packed >> (_j * 8u32)) & 255u32).cast::<f32>();
                        threadgroup_store("ws", dst + _j, s * q + b);
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

#[cfg(test)]
mod tests {
    use metaltile_core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for(dt);
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
        let k = mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for(DType::BF16);
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

    use super::mt_moe_gather_qmm_mma_int8_bm8_mpp;
    use super::super::moe::mt_moe_gather_qmm_b8;

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    #[derive(Clone, Copy, Debug)]
    enum Dt { F32, F16, Bf16 }
    impl Dt {
        fn to_dtype(self) -> DType {
            match self { Dt::F32 => DType::F32, Dt::F16 => DType::F16, Dt::Bf16 => DType::BF16 }
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
            Dt::F16 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
            Dt::Bf16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect(),
        }
    }
    fn pack_int8_row(weights: &[u32]) -> Vec<u32> {
        assert!(weights.len() % 4 == 0);
        weights.chunks_exact(4).map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() { packed |= (q & 0xff) << (i * 8); }
            packed
        }).collect()
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

    struct Case {
        n_experts: usize,
        t_rows: usize,
        n_out: usize,
        k_in: usize,
        group_size: usize,
        dt: Dt,
        indices: Option<Vec<u32>>,
        min_cos: f64,
        label: &'static str,
    }
    impl Case {
        fn new(label: &'static str, n_experts: usize, t_rows: usize, n_out: usize, k_in: usize, group_size: usize, dt: Dt) -> Self {
            Self { n_experts, t_rows, n_out, k_in, group_size, dt, indices: None, min_cos: 0.999, label }
        }
    }

    fn run_case(case: &Case) {
        let _g = gpu_lock();
        let Some(_ctx) = skip_unless_apple10(case.label) else { return };
        let Case { n_experts, t_rows, n_out, k_in, group_size, dt, .. } = *case;
        let indices: Vec<u32> = case.indices.clone().unwrap_or_else(|| (0..t_rows).map(|r| ((r * n_experts) / t_rows) as u32).collect());
        let total_weights = n_experts * n_out * k_in;
        let weight_unpacked: Vec<u32> = (0..total_weights).map(|i| ((i as u32).wrapping_mul(13).wrapping_add(7)) & 0xff).collect();
        let weight_packed: Vec<u32> = weight_unpacked.chunks_exact(k_in).flat_map(pack_int8_row).collect();
        let groups_total = n_experts * n_out * (k_in / group_size);
        let scales: Vec<f32> = (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.041).sin()).collect();
        let biases: Vec<f32> = (0..groups_total).map(|i| -0.02 + 0.005 * (i as f32 * 0.083).cos()).collect();
        let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * (i as f32 * 0.019).sin()).collect();
        let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
        for (e_idx, off) in expert_offsets.iter_mut().enumerate().take(n_experts + 1) {
            *off = indices.iter().position(|&e| e as usize >= e_idx).map(|p| p as u32).unwrap_or(t_rows as u32);
        }
        expert_offsets[n_experts] = t_rows as u32;
        let y_ref: Vec<f32> = {
            let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            b.insert("x".into(), pack_bytes(&x, Dt::F32));
            b.insert("weight_packed".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
            b.insert("scales".into(), pack_bytes(&scales, Dt::F32));
            b.insert("biases".into(), pack_bytes(&biases, Dt::F32));
            b.insert("expert_offsets".into(), expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect());
            b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], Dt::F32));
            b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
            b.insert("m_out".into(), (n_out as u32).to_le_bytes().to_vec());
            b.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
            b.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
            let ctx = Context::new().unwrap();
            let mut k = mt_moe_gather_qmm_b8::kernel_ir_for(DType::F32);
            k.mode = KernelMode::Reduction;
            let r = ctx.dispatch_with_grid(&k, &b, &BTreeMap::new(), [n_out, t_rows, 1], [32, 1, 1]).unwrap();
            unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32)
        };
        let y_mpp: Vec<f32> = {
            let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            b.insert("x".into(), pack_bytes(&x, dt));
            b.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
            b.insert("scales".into(), pack_bytes(&scales, dt));
            b.insert("biases".into(), pack_bytes(&biases, dt));
            b.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
            b.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * n_out], dt));
            b.insert("m_total".into(), (t_rows as u32).to_le_bytes().to_vec());
            b.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
            b.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
            b.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());
            let ctx = Context::new().unwrap();
            let mut k = mt_moe_gather_qmm_mma_int8_bm8_mpp::kernel_ir_for(dt.to_dtype());
            k.mode = KernelMode::Reduction;
            let r = ctx.dispatch_with_grid(&k, &b, &BTreeMap::new(), [n_out.div_ceil(32), t_rows.div_ceil(8), 1], [32, 1, 1]).unwrap();
            unpack_bytes(r.outputs.get("out").unwrap(), dt)
        };
        let mut dot = 0.0_f64; let mut na = 0.0_f64; let mut nb = 0.0_f64; let mut nan_count = 0usize;
        for (a, b) in y_ref.iter().zip(&y_mpp) {
            if !a.is_finite() || !b.is_finite() { nan_count += 1; continue; }
            dot += (*a as f64) * (*b as f64); na += (*a as f64) * (*a as f64); nb += (*b as f64) * (*b as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert_eq!(nan_count, 0, "[{}] MPP BM=8 int8 produced non-finite values", case.label);
        assert!(cos >= case.min_cos, "[{}] MPP MoE BM=8 int8 vs b8 cosine = {:.6} (want ≥ {:.3})", case.label, cos, case.min_cos);
    }

    #[test]
    fn bm8_int8_f32_small_t8() { run_case(&Case::new("f32_small_t8", 4, 8, 64, 64, 32, Dt::F32)); }
    #[test]
    fn bm8_int8_f16_small_t8() { run_case(&Case::new("f16_small_t8", 4, 8, 64, 64, 32, Dt::F16)); }
    #[test]
    fn bm8_int8_bf16_small_t8() {
        let mut case = Case::new("bf16_small_t8", 4, 8, 64, 64, 32, Dt::Bf16);
        case.min_cos = 0.997;
        run_case(&case);
    }
    #[test]
    fn bm8_int8_f16_multi_tile() { run_case(&Case::new("f16_multi_tile", 8, 16, 128, 128, 64, Dt::F16)); }
    #[test]
    fn bm8_int8_f16_ragged_t5() {
        let mut case = Case::new("f16_ragged_t5", 3, 5, 64, 64, 32, Dt::F16);
        case.indices = Some(vec![0, 0, 1, 1, 2]);
        run_case(&case);
    }
    #[test]
    fn bm8_int8_f16_production_shape() {
        let mut case = Case::new("f16_production_shape", 128, 8, 512, 2048, 64, Dt::F16);
        case.indices = Some(vec![3, 17, 42, 55, 71, 88, 99, 120]);
        run_case(&case);
    }
    #[test]
    fn bm8_int8_bf16_production_shape() {
        let mut case = Case::new("bf16_production_shape", 128, 8, 512, 2048, 64, Dt::Bf16);
        case.indices = Some(vec![3, 17, 42, 55, 71, 88, 99, 120]);
        case.min_cos = 0.997;
        run_case(&case);
    }
}

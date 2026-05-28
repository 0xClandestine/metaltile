//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `mt_qmm_nax` — production int4 quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! This is the MPP (MetalPerformancePrimitives) counterpart of
//! `mt_qmm_mma` (the simdgroup-ladder variant). It mirrors the same
//! algorithm — int4 weights dequantized into threadgroup memory once
//! per K-block, then a per-simdgroup matmul against the fp T X-tile —
//! but replaces the manual 8×8 `simdgroup_matmul` ladder with one
//! cooperative `matmul2d` per SG per K-block.
//!
//! Expressed entirely in the `#[kernel]` DSL via the `coop_tile_*`
//! intrinsics — no `Op::InlineMsl`. Algorithmically identical to
//! `mt_qmm_mma_mpp`; the two co-exist so consumers can pick the
//! `_nax` vs `_mpp` name in their own dispatch tables.
//!
//! ## bf16 staging
//!
//! `coop_stage(T)` = `half` for bf16 (Apple `matmul2d` mishandles
//! `bfloat` cooperative tensors), else `T`. Accumulation is fp32.
//!
//! ## Geometry
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM = WN = 2). Fixed.
//! - **BM = BN = BK = 32** → 32×32 output tile (1024 outputs/TG).
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **2×2 warp grid**: each SG owns a 16×16 sub-tile + one 16×16×32 MMA
//!   per K-block (acc-mode `multiply_accumulate`).
//! - **TG row stride = BK + 4 (skew) = 36** — bank-conflict avoidance.
//! - **Group size baked at 64** — Qwen3.6-A3B default.
//! - **`KernelMode::Reduction`**.

use metaltile::kernel;

/// MPP int4 quantized matmul `Out = X · dequant(W)`. Same shape as
/// `mt_qmm_mma_mpp`; both kernels co-exist for naming compatibility.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_qmm_nax<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("Ws", 1152u32, coop_stage(T)); // 32 × 36
    threadgroup_alloc("OutScratch", 1024u32, f32); // 4 SG × 16 × 16
    coop_tile_setup(
        "gemm",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let packs_per_row = k / 8u32;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    let w_pack_row_base = wn_plus_wr * packs_per_row;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        let pack_dev = w_pack_row_base + kb / 8u32 + x_k_quad;
        let packed = load(w[pack_dev]);
        let k_off = kb + x_k_quad * 8u32;
        let g = k_off / 64u32;
        let sb_off = sb_base + g;
        let scale = load(scales[sb_off]).cast::<f32>();
        let bias = load(biases[sb_off]).cast::<f32>();
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = ((packed >> (_ni * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", x_ws_base + _ni, scale * nib + bias);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::{dtype::DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_qmm_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_qmm_nax");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.constexprs.len(), 3);

            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    fn bf16_stages_through_half() {
        let k = mt_qmm_nax::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 must stage as half");
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_qmm_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => unreachable!(),
            };
            k.name = format!("mt_qmm_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_qmm_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
        }
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn cpu_qmm_reference(
        w: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for m_row in 0..m {
            for n_col in 0..n {
                let mut acc = 0.0f32;
                for g in 0..gs_per_row {
                    let s = scales[n_col * gs_per_row + g];
                    let bias = biases[n_col * gs_per_row + g];
                    let mut q_dot = 0.0f32;
                    let mut x_sum = 0.0f32;
                    for p in 0..8usize {
                        let packed = w[n_col * k / 8 + g * 8 + p];
                        for bit in 0..8u32 {
                            let q = ((packed >> (bit * 4)) & 0xF) as f32;
                            let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                            q_dot += q * xv;
                            x_sum += xv;
                        }
                    }
                    acc += s * q_dot + bias * x_sum;
                }
                out[m_row * n + n_col] = acc;
            }
        }
        out
    }

    fn build_quant_inputs(
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let w: Vec<u32> = (0..n * k / 8)
            .map(|i| {
                let mut v = 0u32;
                for bit in 0..8u32 {
                    v |= ((i as u32 + bit) & 0xF) << (bit * 4);
                }
                v
            })
            .collect();
        let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
        (w, scales, biases, x)
    }

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_u32_bytes(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    fn pack_f16(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += *x as f64 * *y as f64;
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32
    }

    #[test_kernel(name = "mlx/qmm_nax/f32_small", dtypes = [f32], tol = 0.001)]
    fn test_qmm_nax_f32_small(dt: DType) -> TestSetup {
        let (m, n, k, group_size) = (32usize, 32usize, 64usize, 64usize);
        let gs_per_row = k / group_size;
        let (w, scales, biases, x) = build_quant_inputs(m, n, k, gs_per_row);
        let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);
        let kernel = mt_qmm_nax::kernel_ir_for(dt);
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("w", pack_u32_bytes(&w), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases), dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x), dt))
            .input(TestBuffer::from_vec("out", vec![0u8; m * n * 4], dt))
            .input(TestBuffer::from_vec("k", (k as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec(
                "gs_per_row",
                (gs_per_row as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), dt))
            .grid_3d(n / 32, m / 32, 1, [128, 1, 1])
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Dense Q4 GEMM via cooperative-tensor MMA — the tensor-core projection GEMM
//! for Nemotron prefill. The MMA twin of `ffai_gemm_q8_mpp`, but reading the
//! bench's Q4 weight layout (signed 4-bit, per-32-block scale `amax/7` stored
//! f16) instead of Q8_0. Compute-bound prefill projections (q/k/v/o, mamba
//! in/out_proj, shared experts, lm_head) run on this instead of the f32
//! scalar `ffai_gemm` (which sat at ~0.1% of the tensor-core peak).
//!
//! Same 64×64×32 coop_tile geometry as `ffai_gemm_q8_mpp` (4 simdgroups,
//! 2×2 warp grid, 128 threads/tg). Only the weight-dequant block differs.
//!
//! ## Q4 weight layout (matches `ffai_ops::quantize_q4`)
//!   qs     [out_dim * (k_in/32) * 4]   u32  — 4 words/block, 8 signed nibbles/word
//!   scales [out_dim * (k_in/32)]       f16  — per-32-block scale (amax/7)
//!   value[r,k] = signed_nibble * scale[r, k/32]
//!
//! Weight is `[out_dim, k_in]`; x `[n_rows, k_in]`; out `[n_rows, out_dim]`.
//! out[r,o] = Σ_k W[o,k]·x[r,k]. Name has `_mpp_` so the MMA path is taken.
//!
//! grid (threadgroups) = [ceil(out_dim/64), ceil(n_rows/64), 1], tg [128,1,1].

use metaltile::kernel;

#[kernel]
pub fn ffai_gemm_q4_mpp<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    scales: Tensor<f16>,
    mut out: Tensor<T>,
    #[constexpr] n_rows: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] k_in: u32,
) {
    let n_tile_base = tgid_x * 64u32; // output-feature tile (N dim)
    let m_tile_base = tgid_y * 64u32; // token tile (M dim)
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    let bpr = k_in / 32u32; // Q4 blocks per row
    threadgroup_alloc("Xs", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("Ws", 2048, coop_stage(T)); // 64 × 32
    threadgroup_alloc("OutScratch", 4096, f32); // 4 SG × 32 × 32
    coop_tile_setup(
        "gemm",
        32,
        32,
        32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm");
    for kb in range(0u32, k_in, 32u32) {
        // Stage X[m_tile_base..+64, kb..kb+32] → Xs. 128 lanes × 16.
        let gr_x = m_tile_base + x_m_row;
        let in_run_x = gr_x < n_rows;
        let safe_gr_x = select(in_run_x, gr_x, 0u32);
        let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
        let x_ws_base = x_m_row * 32u32 + x_k_base;
        for _i in range(0u32, 16u32, 1u32) {
            let xv = load(x[x_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
        }
        // Dequant W[n_tile_base..+64, kb..kb+32] → Ws via Q4. 128 lanes × 16.
        for _i in range(0u32, 16u32, 1u32) {
            let flat = lane_in_tg * 16u32 + _i;
            let w_row = flat / 32u32; // 0..63 (output feature within tile)
            let k_local = flat & 31u32; // 0..31 (BK)
            let global_col = n_tile_base + w_row;
            let k = kb + k_local;
            // Q4: block = (global_col*bpr) + k/32; within-block lane = k%32;
            // word = lane/8 (0..3); nibble = lane%8.
            let blk = global_col * bpr + k / 32u32;
            let lane = k & 31u32;
            let word = load(qs[blk * 4u32 + lane / 8u32]);
            let nib = (word >> ((lane & 7u32) * 4u32)) & 0xfu32;
            // Sign-extend 4-bit: 8..15 represent -8..-1 (nib - 16).
            let q_signed = select(nib >= 8u32, nib - 16u32, nib);
            let qf = q_signed.cast::<i32>().cast::<f32>();
            let sc = load(scales[blk]).cast::<f32>();
            let w = (sc * qf).cast::<T>().cast::<f32>();
            threadgroup_store("Ws", w_row * 32u32 + k_local, w);
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
    threadgroup_barrier();
    for _e in range(0u32, 32u32, 1u32) {
        let flat = lane_in_tg * 32u32 + _e;
        let mr = flat / 64u32;
        let nc = flat & 63u32;
        let gr = m_tile_base + mr;
        let gc = n_tile_base + nc;
        let in_run = (gr < n_rows) & (gc < out_dim);
        if in_run {
            let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
            let v = threadgroup_load(
                "OutScratch",
                src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
            );
            store(out[gr * out_dim + gc], v.cast::<T>());
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gemm_q4_mpp;

    // q_proj-like shape: in=2688, out=4096, 256 tokens.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemm_q4_mpp(dt: DType) -> BenchSetup {
        let n_rows = 256usize;
        let out_dim = 4096usize;
        let k_in = 2688usize;
        let n_blocks = out_dim * k_in / 32;
        BenchSetup::new(ffai_gemm_q4_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", n_rows * k_in, dt))
            .buffer(BenchBuffer::random("qs", n_blocks * 4, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks, DType::F16))
            .buffer(BenchBuffer::zeros("out", n_rows * out_dim, dt).output())
            .constexpr("n_rows", n_rows as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("k_in", k_in as u32)
            .grid_3d((out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved((n_blocks * 18 + n_rows * k_in * dt.size_bytes()) as u64)
    }
}

#[cfg(test)]
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gemm_q4_mpp;
    use crate::utils::pack_f32;

    fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
        let bpr = k / 32;
        let mut qs = vec![0u32; m * bpr * 4];
        let mut scales = vec![0f32; m * bpr];
        for r in 0..m {
            for b in 0..bpr {
                let base = r * k + b * 32;
                let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
                let d = amax / 7.0;
                scales[r * bpr + b] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for word in 0..4 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                        packed |= ((q as u32) & 0xf) << (i * 4);
                    }
                    qs[r * bpr * 4 + b * 4 + word] = packed;
                }
            }
        }
        (qs, scales)
    }

    // out[r,o] = Σ_k dequant(W[o,k]) · x[r,k]; scales rounded through f16.
    fn naive(x: &[f32], qs: &[u32], scales_f16: &[f32], n_rows: usize, out_dim: usize, k_in: usize) -> Vec<f32> {
        let bpr = k_in / 32;
        let mut out = vec![0f32; n_rows * out_dim];
        for r in 0..n_rows {
            for o in 0..out_dim {
                let mut acc = 0f32;
                for kk in 0..k_in {
                    let blk = o * bpr + kk / 32;
                    let lane = kk % 32;
                    let word = qs[blk * 4 + lane / 8];
                    let nib = (word >> ((lane % 8) * 4)) & 0xf;
                    let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                    acc += (q as f32 * scales_f16[blk]) * x[r * k_in + kk];
                }
                out[r * out_dim + o] = acc;
            }
        }
        out
    }

    fn setup(n_rows: usize, out_dim: usize, k_in: usize, dt: DType) -> TestSetup {
        let xv: Vec<f32> = (0..n_rows * k_in).map(|i| (i as f32 * 0.011 - 0.5).sin() * 1.3).collect();
        let wv: Vec<f32> = (0..out_dim * k_in).map(|i| (i as f32 * 0.017 - 0.3).cos() * 0.9).collect();
        let (qs, scales) = quantize_q4(&wv, out_dim, k_in);
        let scales_f16: Vec<f32> = scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
        let expected = naive(&xv, &qs, &scales_f16, n_rows, out_dim, k_in);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
        TestSetup::new(ffai_gemm_q4_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&xv, dt), dt))
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F16), DType::F16))
            .input(TestBuffer::zeros("out", n_rows * out_dim, dt))
            .constexpr("n_rows", n_rows as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("k_in", k_in as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1, [128, 1, 1])
    }

    // 64×64 single-tile, k=64 (2 Q4 blocks). bf16 tol widened for MMA accum.
    #[test_kernel(dtypes = [f16, bf16], tol = [3e-2, 2e-1])]
    fn test_gemm_q4_mpp_tile(dt: DType) -> TestSetup { setup(64, 64, 64, dt) }

    // Non-tile-aligned edges: 40 rows, 96 out, k=128.
    #[test_kernel(dtypes = [f16, bf16], tol = [3e-2, 2e-1])]
    fn test_gemm_q4_mpp_edges(dt: DType) -> TestSetup { setup(40, 96, 128, dt) }
}

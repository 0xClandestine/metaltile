//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled simdgroup-matrix (MMA) dequantizing GEMM — the M ≥ 32
//! ALU-throughput path for the spec-conformant formats. This is a direct
//! adaptation of `mlx/quantized.rs::mt_qmm_mma` (the int4 affine MMA): the
//! **dispatch geometry, threadgroup-memory layout, 8×8 frag mapping, and MMA
//! inner loop are copied verbatim** — only the per-pack weight *dequant*
//! staging changes (E2M1 codebook × E8M0 pow-2 scale instead of int4 affine).
//! Reusing the proven geometry keeps it off the reduction freeze-hazard surface.
//!
//! ## DISPATCH INVARIANTS (identical to `mt_qmm_mma`)
//!
//! - **Mode: Reduction**, `grid = [n/32, m/32, 1]`, `tpg = [128, 1, 1]`
//!   (4 simdgroups × 32 lanes, WM=WN=2). `m`, `n`, `k` all multiples of 32.
//! - BM = BN = BK = 32, output tile 32×32. TG memory `xs`/`ws` are `32×36`
//!   (skew 4 to break bank conflicts; 36 is correct for every dtype).
//! - weight `[n, k/8]` u32 (8 E2M1 nibbles/word); scales `[n, k/block_size]` u8
//!   (E8M0); `block_size` a multiple of 8. x `[m, k]`, out `[m, n]`, row-major.

use metaltile::kernel;

/// mxfp4 simdgroup-matrix dequantizing GEMM (E2M1 weights, E8M0 pow-2 scale).
#[kernel]
pub fn mt_mxfp4_qmm_mma<T>(
    w: Tensor<u32>,
    scales: Tensor<u8>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let packs_per_row = k / 8u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — each lane loads 1 pack → 8 fp T (mxfp4) ──
        let pack_k_off = kb / 8u32 + pack_in_row;
        let pack = load(w[w_pack_row_base + pack_k_off]);
        let k_off = kb + pack_in_row * 8u32;
        let g = k_off / block_size; // E8M0 block index (one per BK for bs=32)
        let sbits = load(scales[sb_base + g]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        let ws_base = w_row * ws_ld + pack_in_row * 8u32;
        for i in range(0u32, 8u32, 1u32) {
            let nib = (pack >> (i * 4u32)) & 0xFu32;
            let m = nib & 0x7u32;
            let mag = select(
                m < 1u32,
                0.0f32,
                select(
                    m < 2u32,
                    0.5f32,
                    select(
                        m < 3u32,
                        1.0f32,
                        select(
                            m < 4u32,
                            1.5f32,
                            select(
                                m < 5u32,
                                2.0f32,
                                select(m < 6u32, 3.0f32, select(m < 7u32, 4.0f32, 6.0f32)),
                            ),
                        ),
                    ),
                ),
            );
            let val = select((nib & 0x8u32) > 0u32, -mag, mag);
            threadgroup_store("ws", ws_base + i, (val * scale).cast::<T>());
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 1
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 2
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 3
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    // ── 4. Write 4 C frags to global out ──
    let out_m_base = m_tile * 32u32 + sm * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Deterministic `[n, k]` weights (mixed signs, per-block magnitude).
    fn weights(n: usize, k: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| {
                let r = (i / k) as f32;
                let c = (i % k) as f32;
                let mag = (0.4 + (r % 7.0) * 0.15) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`.
    fn qmm_oracle(wdq: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for mr in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += wdq[nn * k + kk] * x[mr * k + kk];
                }
                out[mr * n + nn] = acc;
            }
        }
        out
    }

    /// m, n multiples of 32; k a multiple of 32 (and of block_size).
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(n, k);
        let p = crate::quant::format::pack(fmt, &w, n, k);
        let wdq = crate::quant::format::dequant(fmt, &p, n, k);
        let x_f: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = qmm_oracle(&wdq, &x, m, k, n);
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("w", p.codes, DType::U32))
            .input(TestBuffer::from_vec("scales", p.scales, DType::U8))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n / 32) as u32, (m / 32) as u32, 1, [128, 1, 1])
    }

    // 32×32 output tile, K=64 (2 K-blocks).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp4_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp4_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp4, 32, 32, 64, dt)
    }
}

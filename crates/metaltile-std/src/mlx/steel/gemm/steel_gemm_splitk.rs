//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Steel split-K GEMM — #[kernel] DSL vs MLX
//! `metal/steel/gemm/kernels/steel_gemm_splitk.metal`.
//!
//! GEMM that partitions the K dimension across threadgroups so a
//! skinny-M / skinny-N matmul with a very large K still saturates the
//! GPU. It is a **two-kernel** dispatch:
//!
//!   1. `mt_steel_gemm_splitk_*` — each K-split computes a partial
//!      `[M, N]` product over its slice of K and writes it to a
//!      `[n_splits, M, N]` fp32 partials buffer.
//!   2. `mt_steel_gemm_splitk_accum*` — reduces the `n_splits` partial
//!      `[M, N]` matrices into the final `[M, N]` output. The plain
//!      `accum` form is a straight sum; the `axpby` form computes
//!      `α·(Σ partials) + β·C_in` for the fused-bias / residual case.
//!
//! ## How the split-K handoff is expressed
//!
//! The DSL needs no "split-K scheduling primitive" — the partition is
//! just a 3-D grid (`program_id<2>` = K-split index) plus a K-loop
//! whose `[k_start, k_end)` bounds are derived from the split index
//! and a per-split `k_per_split` constexpr. The inter-kernel handoff
//! is an ordinary device buffer: kernel 1 writes the partials, kernel
//! 2 reads them. Two separate `#[kernel]` dispatches, sequenced by the
//! caller — exactly the MLX two-pass pattern.
//!
//! The partials buffer is always **fp32** (the accumulator dtype) so
//! the cross-split sum keeps full precision even for f16 / bf16
//! inputs — mirroring MLX's `AccumType = float`.
//!
//! ## DISPATCH INVARIANTS — split-K kernel
//!
//! - **TPG: `WM*WN*32` threads** (one simdgroup per sub-tile).
//!   `64×64 / 2×2` ⇒ 4 simdgroups ⇒ `tpg = 128`.
//! - **Grid: 3-D — `program_id<0>` = N-block, `program_id<1>` = M-block,
//!   `program_id<2>` = K-split index** (`0 ≤ split < n_splits`).
//! - **`m % BM == 0`, `n % BN == 0`.** `k_per_split` is a multiple of
//!   16 and `n_splits * k_per_split == k`. The last split may legally
//!   run past `k`; the K-loop is clamped to `k`.
//! - **`partials` is fp32, length `n_splits * m * n`**, laid out
//!   `[split, M, N]` row-major. The split-K kernel is itself a `T`
//!   kernel for the A / B operands but writes f32 partials — the
//!   `partials` tensor is declared `Tensor<f32>` regardless of `T`.
//! - **`KernelMode::SimdGroup2D`.**
//!
//! ## DISPATCH INVARIANTS — accum kernel
//!
//! - **Elementwise / Grid3D — one thread per `[M, N]` output element.**
//! - **`partials` length `n_splits * m * n` (fp32)**, `out` length
//!   `m * n`. The `axpby` form additionally reads a `c_in` `[M, N]`
//!   operand and two scalar constexprs `alpha` / `beta`.

use metaltile::kernel;

// ── Pass 1 — split-K partial GEMM ───────────────────────────────────────

/// Expand one `(BM, BN, WM, WN)` block-shape instantiation of the
/// split-K partial GEMM. The outer `macro_rules!` substitutes the
/// literals before the `#[kernel]` body parser runs.
#[rustfmt::skip]
macro_rules! steel_gemm_splitk_kernel {
    ($name:ident, $bm:literal, $bn:literal, $wm:literal, $wn:literal, $tpg:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            a: Tensor<T>,
            b: Tensor<T>,
            mut partials: Tensor<f32>,
            #[constexpr] m: u32,
            #[constexpr] n: u32,
            #[constexpr] k: u32,
            #[constexpr] k_per_split: u32,
        ) {
            // ── Block / simdgroup geometry (identical to steel_gemm_fused) ──
            let bm = $bm;
            let bn = $bn;
            let wm = $wm;
            let wn = $wn;
            let sub_m = bm / wm;
            let sub_n = bn / wn;
            let n_fm = sub_m / 8u32;
            let n_fn = sub_n / 8u32;
            let n_kf = 2u32; // BK = 16 ⇒ two 8×8 K-fragments per K-step.

            let tg_col = program_id::<0>(); // N-block index
            let tg_row = program_id::<1>(); // M-block index
            let split = program_id::<2>(); // K-split index
            let sg_id = simd_group_id();
            let sg_m = sg_id / wn;
            let sg_n = sg_id % wn;
            let lane = simd_lane_id();

            // Apple 8×8 fragment lane mapping.
            let qid = lane / 4u32;
            let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
            let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
            let fn1 = fn0 + 1u32;

            let sub_m0 = sg_m * sub_m;
            let sub_n0 = sg_n * sub_n;
            let block_m0 = tg_row * bm;
            let block_n0 = tg_col * bn;

            // ── This split's K-range ──
            // [k_start, k_end) — the last split is clamped to `k`.
            let k_start = split * k_per_split;
            let k_end_raw = k_start + k_per_split;
            let k_end = select(k_end_raw < k, k_end_raw, k);
            // Partial-output base offset for this split: [split, M, N].
            let part_base = split * m * n;

            for _fm_i in range(0, n_fm, 1) {
                for _fn_i in range(0, n_fn, 1) {
                    let acc = simdgroup_alloc::<f32, 8, 8>();
                    simdgroup_elem_store(acc, 0, 0.0f32);
                    simdgroup_elem_store(acc, 1, 0.0f32);

                    let m_row = block_m0 + sub_m0 + _fm_i * 8u32;
                    let n_col = block_n0 + sub_n0 + _fn_i * 8u32;

                    for kb in range(k_start, k_end, 16) {
                        for _kf in range(0, n_kf, 1) {
                            let kf = kb + _kf * 8u32;
                            let sub_a = simdgroup_alloc::<T, 8, 8>();
                            let sub_b = simdgroup_alloc::<T, 8, 8>();

                            simdgroup_elem_store(
                                sub_a,
                                0,
                                load(a[(m_row + fm) * k + kf + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_a,
                                1,
                                load(a[(m_row + fm) * k + kf + fn1]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_b,
                                0,
                                load(b[(kf + fm) * n + n_col + fn0]).cast::<T>(),
                            );
                            simdgroup_elem_store(
                                sub_b,
                                1,
                                load(b[(kf + fm) * n + n_col + fn1]).cast::<T>(),
                            );

                            simdgroup_matmul(sub_a, sub_b, acc);
                        }
                    }

                    // Write this split's fp32 partial — no cast, the
                    // partials buffer is the f32 accumulator dtype.
                    let r0 = simdgroup_elem_load(acc, 0);
                    let r1 = simdgroup_elem_load(acc, 1);
                    store(partials[part_base + (m_row + fm) * n + n_col + fn0], r0);
                    store(partials[part_base + (m_row + fm) * n + n_col + fn1], r1);
                }
            }
        }
    };
}

// 64×64×16 / 2×2 — the canonical large-tile shape (4 simdgroups).
steel_gemm_splitk_kernel!(
    mt_steel_gemm_splitk_64x64x16_2x2,
    64u32,
    64u32,
    2u32,
    2u32,
    128u32,
    "bm64_bn64_bk16_wm2_wn2"
);
// 32×32×16 / 2×2 — small-tile shape (4 simdgroups) — split-K is most
// useful exactly here: skinny M/N with a large K.
steel_gemm_splitk_kernel!(
    mt_steel_gemm_splitk_32x32x16_2x2,
    32u32,
    32u32,
    2u32,
    2u32,
    128u32,
    "bm32_bn32_bk16_wm2_wn2"
);

// ── Pass 2 — partial-sum reduction ──────────────────────────────────────

/// Split-K accumulation: reduce `n_splits` partial `[M, N]` matrices
/// (fp32) into the final `[M, N]` output. One thread per output
/// element. This is the plain-sum form of MLX's
/// `steel_gemm_splitk_accum`.
#[kernel]
pub fn mt_steel_gemm_splitk_accum<T>(
    partials: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] m: u32,
    #[constexpr] n: u32,
    #[constexpr] n_splits: u32,
) {
    // One thread per [M, N] output element — `program_id::<0>()` is the
    // global flat index, the grid is sized to `m * n` by the dispatch.
    let idx = program_id::<0>();
    let total = m * n;
    // Sum this element across every K-split.
    let mut acc = 0.0f32;
    for s in range(0u32, n_splits, 1u32) {
        acc = acc + load(partials[s * total + idx]);
    }
    store(out[idx], acc.cast::<T>());
}

/// Split-K accumulation, `axpby` form: `out = α·(Σ partials) + β·c_in`.
/// The fused-bias / residual variant of MLX's
/// `steel_gemm_splitk_accum_*_axbpy`. One thread per output element.
#[kernel]
pub fn mt_steel_gemm_splitk_accum_axpby<T>(
    partials: Tensor<f32>,
    c_in: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] m: u32,
    #[constexpr] n: u32,
    #[constexpr] n_splits: u32,
    #[constexpr] alpha: f32,
    #[constexpr] beta: f32,
) {
    let idx = program_id::<0>();
    let total = m * n;
    let mut acc = 0.0f32;
    for s in range(0u32, n_splits, 1u32) {
        acc = acc + load(partials[s * total + idx]);
    }
    // α·(Σ partials) + β·c_in.
    let prev = load(c_in[idx]).cast::<f32>();
    let res = alpha * acc + beta * prev;
    store(out[idx], res.cast::<T>());
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };

    use super::*;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => panic!(),
        }
    }

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    /// Naive matmul over a K-slice `[k_start, k_end)`.
    fn naive_partial_matmul(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        k_start: usize,
        k_end: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in k_start..k_end {
                    acc += a[mi * k + ki] * b[ki * n + ni];
                }
                out[mi * n + ni] = acc;
            }
        }
        out
    }

    const BM: usize = 64;
    const BN: usize = 64;
    const M: usize = BM * 2;
    const N: usize = BN * 2;

    // ── Test: accum kernel alone — sum of known fp32 partials ───────────

    fn make_accum_setup(dt: DType, n_splits: usize, m: usize, n: usize) -> TestSetup {
        // Build known partials: split s has all elements = (s+1) as f32.
        let partials: Vec<f32> =
            (0..n_splits * m * n).map(|i| ((i / (m * n) + 1) as f32) * 0.01).collect();
        let expected: Vec<f32> =
            (0..m * n).map(|_| (1..=n_splits).map(|s| s as f32 * 0.01).sum::<f32>()).collect();
        let mut kernel = mt_steel_gemm_splitk_accum::kernel_ir_for(dt);
        kernel.mode = KernelMode::Elementwise;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("partials", pack_f32(&partials), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("n_splits", n_splits as u32)
            .grid_1d(m * n, 256)
    }

    #[test_kernel(name = "steel/gemm_splitk_accum_f32", dtypes = [f32], tol = 1e-5)]
    fn test_gemm_splitk_accum_f32(dt: DType) -> TestSetup { make_accum_setup(dt, 3, M, N) }

    #[test_kernel(name = "steel/gemm_splitk_accum_f16", dtypes = [f16], tol = 1e-3)]
    fn test_gemm_splitk_accum_f16(dt: DType) -> TestSetup { make_accum_setup(dt, 3, M, N) }

    // ── Test: accum_axpby kernel — α·Σ + β·c_in ─────────────────────────

    #[test_kernel(name = "steel/gemm_splitk_accum_axpby_f32", dtypes = [f32], tol = 1e-4)]
    fn test_gemm_splitk_accum_axpby_f32(dt: DType) -> TestSetup {
        let (n_splits, m, n) = (2usize, M, N);
        let (alpha, beta) = (0.5f32, 2.0f32);
        let partials: Vec<f32> =
            (0..n_splits * m * n).map(|i| ((i / (m * n) + 1) as f32) * 0.1).collect();
        let c_in: Vec<f32> = (0..m * n).map(|i| (i as f32 % 7.0 - 3.0) * 0.05).collect();
        let partial_sum: Vec<f32> = (0..m * n)
            .map(|j| (0..n_splits).map(|s| partials[s * m * n + j]).sum::<f32>())
            .collect();
        let expected: Vec<f32> =
            partial_sum.iter().zip(c_in.iter()).map(|(&p, &c)| alpha * p + beta * c).collect();
        let mut kernel = mt_steel_gemm_splitk_accum_axpby::kernel_ir_for(dt);
        kernel.mode = KernelMode::Elementwise;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("partials", pack_f32(&partials), DType::F32))
            .input(TestBuffer::from_vec("c_in", pack(&c_in, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("n_splits", n_splits as u32)
            .constexpr("alpha", alpha)
            .constexpr("beta", beta)
            .grid_1d(m * n, 256)
    }

    // ── Test: splitk pass1 — check one split's partial product ──────────

    fn make_pass1_setup(dt: DType, m: usize, n: usize, k: usize, n_splits: usize) -> TestSetup {
        let k_per_split = k / n_splits;
        let a_raw = ramp(m * k, 19, 7.0);
        let b_raw = ramp(k * n, 23, 9.0);
        let a: Vec<f32> = a_raw.iter().map(|&v| round_dt(v, dt)).collect();
        let b: Vec<f32> = b_raw.iter().map(|&v| round_dt(v, dt)).collect();
        // Expected: [n_splits, m, n] — each split's partial product.
        let mut expected = vec![0.0f32; n_splits * m * n];
        for s in 0..n_splits {
            let partial =
                naive_partial_matmul(&a, &b, m, k, n, s * k_per_split, (s + 1) * k_per_split);
            expected[s * m * n..(s + 1) * m * n].copy_from_slice(&partial);
        }
        let mut kernel = mt_steel_gemm_splitk_64x64x16_2x2::kernel_ir_for(dt);
        kernel.mode = KernelMode::SimdGroup2D;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("a", pack(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b_raw, dt), dt))
            .expect(TestBuffer::from_vec("partials", pack_f32(&expected), DType::F32))
            .constexpr("m", m as u32)
            .constexpr("n", n as u32)
            .constexpr("k", k as u32)
            .constexpr("k_per_split", k_per_split as u32)
            .grid_3d((n / BN) as u32, (m / BM) as u32, n_splits as u32, [128, 1, 1])
    }

    #[test_kernel(name = "steel/gemm_splitk_pass1_2way_f32", dtypes = [f32], tol = 3e-3)]
    fn test_gemm_splitk_pass1_2way_f32(dt: DType) -> TestSetup { make_pass1_setup(dt, M, N, 64, 2) }

    #[test_kernel(name = "steel/gemm_splitk_pass1_2way_f16", dtypes = [f16], tol = 8e-2)]
    fn test_gemm_splitk_pass1_2way_f16(dt: DType) -> TestSetup { make_pass1_setup(dt, M, N, 64, 2) }
}

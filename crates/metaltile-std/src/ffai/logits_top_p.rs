//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Top-p (nucleus) logits filter for the sampling pipeline.
//!
//! Top-p sampling keeps the smallest set of most-likely tokens whose
//! cumulative probability reaches `top_p`, and masks the rest. The
//! reference definition sorts the probabilities descending and walks
//! the prefix until the running sum clears `top_p`. Equivalently — and
//! without a sort — there is a probability cutoff `c` such that the
//! kept set is exactly `{ i : P(i) ≥ c }`, and that set's mass is the
//! smallest that reaches `top_p`. This kernel finds `c` directly.
//!
//! Working in logit space avoids a full softmax: for any shift, the
//! unnormalised weight of token `i` is `w_i = exp(logit_i − logit_max)`
//! and `Z = Σ w_i`. The keep test `P(i) ≥ c` becomes `w_i ≥ c·Z`, so
//! the cutoff search runs entirely on `w ∈ (0, 1]`.
//!
//! `w` is not sorted, so `c` is found by **bisection**: the kept mass
//! `S(t) = Σ_{w_i ≥ t} w_i` is monotonically non-increasing in `t`, so
//! a binary search on `t ∈ [0, 1]` converges on the threshold where
//! `S(t)` just reaches `top_p·Z`. 24 halvings pin `t` to a `2⁻²⁴`
//! (≈ 6e-8) interval — far finer than the gap between adjacent token
//! weights near any realistic cutoff. A final pass masks every logit
//! whose weight is below the converged floor to `-INFINITY`, so the
//! downstream `softmax_categorical_sample` sees `exp(-inf) = 0`.
//!
//! This is the iterative-search sibling of `logits_min_p_mask`: min-p's
//! cutoff is a closed form of the row max (one reduction), but top-p's
//! cutoff depends on the whole mass profile, so it costs one reduction
//! per bisection step. The whole filter is still a single self-contained
//! GPU kernel — one threadgroup per row, no host round-trip, no sort.
//!
//! Reduction-mode, generic over T; the max, the partition function and
//! every kept-mass sum are computed in f32 so f16/bf16 logits don't
//! drift. One threadgroup per row; `n` is the vocab length, looped so
//! any `n` works at any (multiple-of-32) threadgroup size.
//!
//! Caller contract: `0 < top_p < 1`. As `top_p → 0` only the argmax
//! survives; as `top_p → 1` nothing is masked. A typical serving value
//! is 0.9–0.95.

use metaltile::kernel;

#[kernel]
pub fn logits_top_p_mask<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] top_p: f32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    // Pass 1: one streaming pass for both the row max and the partition
    // function Z = Σ exp(logit − row_max). Each lane keeps a running
    // (max, sum) pair in online-softmax form; the pair is then merged
    // across the threadgroup. This mirrors `mt_softmax`'s looped path.
    let mut lm = neg_infinity();
    let mut ls = 0.0f32;
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        let nm = max(lm, v);
        ls = ls * exp(lm - nm) + exp(v - nm);
        lm = nm;
    }
    let row_max = reduce_max(lm);
    // Rescale every lane's partial sum to the common max before reducing.
    let z = reduce_sum(ls * exp(lm - row_max));
    // Bisection: find the largest weight threshold `t` whose kept mass
    // S(t) = Σ_{w_i ≥ t} w_i still reaches `target`. `lo` is the highest
    // threshold known to keep enough mass, `hi` the lowest known to keep
    // too little; the kept set shrinks as the threshold rises.
    // 24 halvings of `t ∈ [0, 1]` pin the cutoff to a ≈ 6e-8 interval;
    // each step costs one threadgroup reduction over the row.
    let target = top_p * z;
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _k in range(0u32, 24u32, 1u32) {
        let mid = (lo + hi) * 0.5f32;
        let mut partial = 0.0f32;
        for _i in range(rs + tid, re, lsize) {
            let w = exp(load(inp[_i]).cast::<f32>() - row_max);
            partial = partial + select(w >= mid, w, 0.0f32);
        }
        let kept_mass = reduce_sum(partial);
        // S is non-increasing in the threshold: if `mid` still keeps
        // enough mass we can raise the floor, otherwise we must lower it.
        let enough = kept_mass >= target;
        lo = select(enough, mid, lo);
        hi = select(enough, hi, mid);
    }
    // Pass 2: keep a logit iff its weight clears the converged floor
    // `lo`, else -inf. `lo` starts at 0, so a token whose weight equals
    // the floor is kept — the argmax (weight 1) therefore always survives.
    let neg_inf = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        store(out[_i], select(exp(v - row_max) >= lo, v, neg_inf).cast::<T>());
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };
    use metaltile::test_kernel;

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    /// CPU oracle: replays the kernel's sort-free bisection (24 halvings).
    fn cpu_top_p_mask(logits: &[f32], n: usize, rows: usize, top_p: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let base = r * n;
            let row = &logits[base..base + n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let w: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = w.iter().sum();
            let target = top_p * z;
            let mut lo = 0.0f32;
            let mut hi = 1.0f32;
            for _ in 0..24 {
                let mid = (lo + hi) * 0.5;
                let kept: f32 = w.iter().filter(|&&x| x >= mid).sum();
                if kept >= target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            for (i, &wi) in w.iter().enumerate() {
                out[base + i] = if wi >= lo { row[i] } else { f32::NEG_INFINITY };
            }
        }
        out
    }

    #[test_kernel(name = "logits/top_p_mask_mid_range", dtypes = [f32, f16, bf16], tol = 1e-4)]
    fn test_top_p_mask_mid_range(dt: DType) -> TestSetup {
        let n = 320usize;
        let rows = 4usize;
        let top_p = 0.9f32;
        let logits_raw: Vec<f32> = (0..n * rows).map(|i| (i % 53) as f32 * 0.2 - 5.0).collect();
        let logits: Vec<f32> = logits_raw.iter().map(|&v| round(v, dt)).collect();
        let expected = cpu_top_p_mask(&logits, n, rows, top_p);

        let mut k = logits_top_p_mask::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Reduction;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("top_p", top_p.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(name = "logits/top_p_mask_near_zero", dtypes = [f32], tol = 1e-4)]
    fn test_top_p_mask_near_zero(dt: DType) -> TestSetup {
        // Strictly increasing logits → unique argmax at last index, top_p=0.01.
        let n = 64usize;
        let rows = 1usize;
        let top_p = 0.01f32;
        let logits: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let expected = cpu_top_p_mask(&logits, n, rows, top_p);

        let mut k = logits_top_p_mask::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Reduction;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("top_p", top_p.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }
}

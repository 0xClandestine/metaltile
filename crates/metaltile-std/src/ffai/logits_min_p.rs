//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Min-p (minimum-probability) logits filter for the sampling pipeline.
//!
//! Min-p sampling keeps every token whose probability is at least
//! `min_p` times the probability of the most-likely token, and masks
//! the rest:
//!
//!   keep token i  ⇔  P(i) ≥ min_p · P_max
//!
//! Working in logit space avoids a full softmax. For any shift `C`,
//! `P(i) / P_max = exp(logit_i − logit_max)`, so the keep test is
//! simply `exp(logit_i − logit_max) ≥ min_p`. The kernel finds the
//! row max with one threadgroup reduction, then masks every logit
//! below the cutoff to `-INFINITY` in a second pass. Downstream
//! `softmax_categorical_sample` sees `exp(-inf) = 0`, so masked tokens
//! contribute zero probability.
//!
//! This is the reduction-mode sibling of `logits_topk_mask`: top-K
//! needs a host-computed K-th-largest threshold, but min-p's cutoff is
//! defined purely by the row max, so the whole filter fits in one
//! self-contained GPU kernel — no host round-trip, no sort.
//!
//! Reduction-mode, generic over T; the max and the ratio are computed
//! in f32 so f16/bf16 logits don't drift. One threadgroup per row;
//! `n` is the vocab length, looped so any `n` works at any
//! (multiple-of-32) threadgroup size.
//!
//! Caller contract: `0 < min_p < 1`. As `min_p → 0` nothing is masked;
//! as `min_p → 1` only the argmax (and exact ties) survive. A typical
//! serving value is 0.05–0.1.

use metaltile::kernel;

#[kernel]
pub fn logits_min_p_mask<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] min_p: f32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    // Pass 1: threadgroup-wide max of the row's logits.
    let mut lm = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        lm = max(lm, load(inp[_i]).cast::<f32>());
    }
    let row_max = reduce_max(lm);
    // Pass 2: keep a logit iff exp(logit - row_max) >= min_p, else -inf.
    let neg_inf = neg_infinity();
    for _i in range(rs + tid, re, lsize) {
        let v = load(inp[_i]).cast::<f32>();
        store(out[_i], select(exp(v - row_max) >= min_p, v, neg_inf).cast::<T>());
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };
    use metaltile_macros::test_kernel;

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
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    /// CPU oracle: keep `v` iff `exp(v − row_max) ≥ min_p`, else `-inf`.
    fn cpu_min_p_mask(logits: &[f32], n: usize, rows: usize, min_p: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let base = r * n;
            let row = &logits[base..base + n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (i, &v) in row.iter().enumerate() {
                out[base + i] = if (v - m).exp() >= min_p { v } else { f32::NEG_INFINITY };
            }
        }
        out
    }

    #[test_kernel(name = "logits/min_p_mask_mid_range", dtypes = [f32, f16, bf16], tol = 1e-4)]
    fn test_min_p_mask_mid_range(dt: DType) -> TestSetup {
        let n = 320usize;
        let rows = 4usize;
        let min_p = 0.1f32;
        let logits_raw: Vec<f32> = (0..n * rows).map(|i| (i % 53) as f32 * 0.2 - 5.0).collect();
        let logits: Vec<f32> = logits_raw.iter().map(|&v| round(v, dt)).collect();
        let expected = cpu_min_p_mask(&logits, n, rows, min_p);

        let mut k = logits_min_p_mask::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("min_p", min_p.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(name = "logits/min_p_mask_near_zero", dtypes = [f32], tol = 1e-4)]
    fn test_min_p_mask_near_zero(dt: DType) -> TestSetup {
        let n = 320usize;
        let rows = 3usize;
        let min_p = 1e-6f32;
        let logits: Vec<f32> = (0..n * rows).map(|i| (i % 53) as f32 * 0.2 - 5.0).collect();
        let expected = cpu_min_p_mask(&logits, n, rows, min_p);

        let mut k = logits_min_p_mask::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("min_p", min_p.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(name = "logits/min_p_mask_near_one", dtypes = [f32], tol = 1e-4)]
    fn test_min_p_mask_near_one(dt: DType) -> TestSetup {
        let n = 64usize;
        let rows = 1usize;
        let min_p = 0.999f32;
        let logits: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let expected = cpu_min_p_mask(&logits, n, rows, min_p);

        let mut k = logits_min_p_mask::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::from_vec("min_p", min_p.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }
}

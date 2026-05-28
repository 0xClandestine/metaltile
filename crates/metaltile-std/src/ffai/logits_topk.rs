//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Top-K filter — masking variant.
//!
//! The full top-K filter pipeline is:
//!
//!   1. Find the K-th largest logit value: `threshold = argpartition(logits, -K)`
//!   2. For every logit, if `logit >= threshold` keep it; else set to `-inf`
//!
//! Step 1 is a selection / partial-sort. On GPU at typical serving K (50, 100)
//! and Qwen-scale vocab (152K) the per-call cost is dominated by Metal command-
//! buffer overhead, not arithmetic — a CPU argpartition + threshold-pass is
//! roughly the same wall-clock as a GPU select kernel and one less dispatch.
//! This file ships the GPU mask kernel and leaves threshold computation to
//! the caller. A future PR can add a GPU-side selection kernel when serving
//! batch sizes make a single fused dispatch pull ahead.
//!
//! Caller contract:
//!   - Compute `threshold` = the K-th largest value (descending) on the host.
//!   - Pass it as the constexpr `threshold` parameter.
//!   - Logits below `threshold` are replaced with `-INFINITY` (this is the
//!     standard sentinel — downstream softmax sees `exp(-inf) = 0` and the
//!     filtered tokens contribute zero probability).
//!
//! Generic over T. Grid3D one-thread-per-vocab-position.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Grid3D.** One thread per vocab position.
//! - **Grid: `[ceil(n / TPG), 1, 1]`, TG: `[TPG, 1, 1]`** (TPG = 256 is the
//!   tested geometry; the kernel is pure elementwise so any TPG works).
//! - **`n = grid.x * tg.x`** — caller sizes the dispatch so the total
//!   thread count exactly matches the vocab length. Threads past `n`
//!   would read/write out of bounds; the runtime should not overshoot.
//! - **No `threadgroup_*` / `simd_*` cooperation** — every thread is
//!   independent. The only invariant is the threshold semantic above.

use metaltile::kernel;

#[kernel]
pub fn logits_topk_mask<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] threshold: f32) {
    let i = program_id::<0>();
    let v = load(inp[i]).cast::<f32>();
    // `select(cond, lhs, rhs)` returns lhs when cond is true.
    // Keep value when v >= threshold; otherwise sentinel to -inf.
    let neg_inf = neg_infinity();
    let masked = select(v >= threshold, v, neg_inf);
    store(out[i], masked.cast::<T>());
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

    fn kth_largest(logits: &[f32], k: usize) -> f32 {
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted[k - 1]
    }

    fn cpu_topk_mask(logits: &[f32], threshold: f32) -> Vec<f32> {
        logits.iter().map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY }).collect()
    }

    #[test_kernel(name = "logits/topk_mask_k50_f32", dtypes = [f32], tol = 1e-5)]
    fn test_topk_mask_k50(dt: DType) -> TestSetup {
        let n = 1024usize;
        let k = 50usize;
        let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin() * 5.0).collect();
        let threshold = kth_largest(&logits, k);
        let expected = cpu_topk_mask(&logits, threshold);

        let mut ker = logits_topk_mask::kernel_ir_for(dt);
        ker.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(ker)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("threshold", threshold.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "logits/topk_mask_neg_inf_keeps_all", dtypes = [f32], tol = 1e-5)]
    fn test_topk_mask_neg_inf(dt: DType) -> TestSetup {
        let n = 256usize;
        let logits: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let threshold = f32::NEG_INFINITY;
        let expected = cpu_topk_mask(&logits, threshold);

        let mut ker = logits_topk_mask::kernel_ir_for(dt);
        ker.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(ker)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("threshold", threshold.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "logits/topk_mask_f16", dtypes = [f16], tol = 1e-2)]
    fn test_topk_mask_f16(dt: DType) -> TestSetup {
        let n = 1024usize;
        let k = 50usize;
        let logits: Vec<f32> =
            (0..n).map(|i| round(((i as f32) * 0.0173).sin() * 5.0, dt)).collect();
        let threshold = kth_largest(&logits, k);
        let expected = cpu_topk_mask(&logits, threshold);

        let mut ker = logits_topk_mask::kernel_ir_for(dt);
        ker.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(ker)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("threshold", threshold.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MoE router — sigmoid + bias + (optional) top-k normalize + scale.
//!
//! The DeepSeek-V3 routing pattern, adopted by StepFun's Step-3 family:
//!
//! ```text
//!   gates  = sigmoid(W_router · x) + router_bias        [n_experts]
//!   topk   = argpartition(-gates, k)[:, :k]             [k]   (indices)
//!   weights = gates.gather(topk)                        [k]
//!   if norm_topk_prob:  weights = weights / weights.sum()
//!   weights = weights * scaling_factor
//! ```
//!
//! Distinct from the more common softmax-routing pattern Qwen3 / Llama /
//! GPT-OSS use: the sigmoid + bias factorisation lets each expert score
//! be evaluated independently (no softmax denominator coupling), and
//! the per-expert bias is what the upstream "noisy gating with auxiliary
//! load-balancing loss" line learns.
//!
//! This file ships the **score path only** — `sigmoid(logits) + bias`
//! into a `[n_experts]` scores tensor. Top-k selection + normalize +
//! scale are kept as a separate kernel because the same selection
//! pipeline is shared with the (existing) softmax-routing path; only
//! the score producer changes per family.
//!
//! ## ABI
//!
//! ```text
//!   logits     [n_experts] f32      — pre-sigmoid router output
//!                                       (`W_router · x`)
//!   bias       [n_experts] f32      — per-expert routing bias
//!   scores     [n_experts] f32      — out: `sigmoid(logits) + bias`
//! ```
//!
//! Grid is 1D elementwise: one thread per expert. Caller drives
//! `grid_1d(n_experts, 64)` — n_experts is typically 64-288 across
//! the production checkpoints (DeepSeek-V3 = 256, Step-3 = 288).

use metaltile::kernel;

// Bare `#[kernel]` — non-generic, all-`Tensor<f32>` kernel; the legacy
// `bench(...)` registration expects a generic-T shape it can't bind to
// a no-`<T>` signature. The new declarative `#[bench]` on
// `kernel_benches::bench_router` below handles registration directly.
#[kernel]
pub fn ffai_moe_router_sigmoid_bias(
    logits: Tensor<f32>,
    bias: Tensor<f32>,
    mut scores: Tensor<f32>,
) {
    let idx = tid;
    let l = load(logits[idx]);
    let b = load(bias[idx]);
    // Free-function `exp` + f32-suffixed literals: the method form
    // `(-l).exp()` nested in a larger expression elides its binding in
    // the DSL codegen mangler (same fix as moe_router_sqrtsoftplus).
    let s = 1.0f32 / (1.0f32 + exp(0.0f32 - l));
    store(scores[idx], s + b);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_moe_router_sigmoid_bias;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n_experts: usize) -> TestSetup {
        let dt = DType::F32;
        let logits: Vec<f32> = (0..n_experts).map(|i| (i % 31) as f32 * 0.1 - 1.5).collect();
        let bias: Vec<f32> = (0..n_experts).map(|i| (i % 7) as f32 * 0.05 - 0.15).collect();
        let l_dt = unpack_f32(&pack_f32(&logits, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&bias, dt), dt);
        let expected: Vec<f32> =
            l_dt.iter().zip(&b_dt).map(|(&l, &b)| 1.0_f32 / (1.0 + (-l).exp()) + b).collect();
        TestSetup::new(ffai_moe_router_sigmoid_bias::kernel_ir())
            .input(TestBuffer::from_vec("logits", pack_f32(&logits, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("scores", n_experts, dt))
            .expect(TestBuffer::from_vec("scores", pack_f32(&expected, dt), dt))
            .grid_1d(n_experts, 64)
    }

    // tol 2e-4: sigmoid routes through `exp`, so the GPU↔CPU-oracle gap
    // is dominated by transcendental fast-math and varies by GPU family
    // — same rationale as the sister moe_router_sqrtsoftplus. Still far
    // tighter than a logic error would ever land.
    #[test_kernel(dtypes = [f32], tol = [2e-4])]
    fn test_router_sigmoid_bias_step3(_dt: DType) -> TestSetup { setup(288) }

    /// DeepSeek-V3 shape — same router pattern at a different expert count.
    #[test_kernel(dtypes = [f32], tol = [2e-4])]
    fn test_router_sigmoid_bias_dsv3(_dt: DType) -> TestSetup { setup(256) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_router_sigmoid_bias;

    #[bench(name = "ffai/moe_router_sigmoid_bias", dtypes = [f32])]
    fn bench_router(_dt: DType) -> BenchSetup {
        let dt = DType::F32;
        let n_experts = 288usize;
        BenchSetup::new(ffai_moe_router_sigmoid_bias::kernel_ir())
            .buffer(BenchBuffer::random("logits", n_experts, dt))
            .buffer(BenchBuffer::random("bias", n_experts, dt))
            .buffer(BenchBuffer::zeros("scores", n_experts, dt).output())
            .grid_1d(n_experts, 64)
            .bytes_moved((3 * n_experts * dt.size_bytes()) as u64)
    }
}

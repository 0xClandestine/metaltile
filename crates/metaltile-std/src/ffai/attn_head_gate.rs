//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Head-wise attention output gate — `attn_out[h, d] *= sigmoid(g[h])`.
//!
//! Applied between the SDPA output and `o_proj` for checkpoints that
//! ship a per-head scalar gate (`use_head_wise_attn_gate` in upstream
//! configs). StepFun's Step-3.7-Flash + Step-3.5-Flash are the current
//! consumers; other gated-attention variants in the literature fit the
//! same shape.
//!
//! The pattern in PyTorch reference impls is:
//!
//! ```python
//!   g = self.g_proj(x)              # [batch, n_heads] — per-head scalar
//!   attn = attn * sigmoid(g)[..., None]
//! ```
//!
//! `g_proj` is a `Linear(hidden, n_heads)` projection of the layer's
//! input. The fused kernel below does the `sigmoid(g) × attn` step in
//! one pass; computing `g` itself stays as a separate gemv (it's just
//! `hidden → n_heads` and reuses the existing dense matmul path).
//!
//! ## ABI
//!
//! ```text
//!   attn    [n_heads, head_dim] T       — SDPA output, modified in place
//!   gate    [n_heads]            T      — pre-sigmoid logits (g_proj(x))
//!   out     [n_heads, head_dim] T       — gated SDPA output
//!   head_dim u32                        — constexpr, distinguishes
//!                                          per-head row size
//! ```
//!
//! Grid is 1D elementwise over the flat `[n_heads * head_dim]` buffer;
//! each thread derives its owning head as `idx / head_dim` and reuses
//! that head's single sigmoid. Caller drives `grid_1d(n, 256)`.

use metaltile::kernel;

// Bare `#[kernel]` — the legacy `bench(...)` registration doesn't
// fit this 3-input + 1-constexpr shape; the new declarative `#[bench]`
// on `kernel_benches::bench_attn_head_gate` below handles registration.
#[kernel]
pub fn ffai_attn_head_gate<T>(
    attn: Tensor<T>,
    gate: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
) {
    // 1D elementwise over the flat [n_heads * head_dim] attn buffer; the
    // owning head is `idx / head_dim` so the per-head gate is shared
    // across its head_dim lanes.
    let idx = tid;
    let h = idx / head_dim;
    // sigmoid(g[h]) is the per-head scalar. Computed in f32 so the
    // small-magnitude tail (`g ≈ -10`) doesn't underflow on bf16.
    // Free-function `exp` + f32 literals so the binding isn't elided.
    let g_raw = load(gate[h]).cast::<f32>();
    let s = 1.0f32 / (1.0f32 + exp(0.0f32 - g_raw));
    let a = load(attn[idx]).cast::<f32>();
    store(out[idx], (a * s).cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_attn_head_gate;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n_heads: usize, head_dim: usize, dt: DType) -> TestSetup {
        let n = n_heads * head_dim;
        let attn: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.1 - 1.0).collect();
        let gate: Vec<f32> = (0..n_heads).map(|h| (h % 7) as f32 * 0.5 - 1.5).collect();
        let a_dt = unpack_f32(&pack_f32(&attn, dt), dt);
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let mut expected: Vec<f32> = Vec::with_capacity(n);
        for h in 0..n_heads {
            let s = 1.0_f32 / (1.0 + (-g_dt[h]).exp());
            for d in 0..head_dim {
                expected.push(a_dt[h * head_dim + d] * s);
            }
        }
        TestSetup::new(ffai_attn_head_gate::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("attn", pack_f32(&attn, dt), dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("head_dim", head_dim as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_ffai_attn_head_gate_step3_full(dt: DType) -> TestSetup { setup(64, 128, dt) }

    /// SWA-layer shape variant: Step-3 uses 96 q-heads on
    /// sliding-attention layers. Validates the kernel doesn't bake the
    /// full-attn 64-head count in.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_ffai_attn_head_gate_step3_swa(dt: DType) -> TestSetup { setup(96, 128, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_attn_head_gate;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_attn_head_gate(dt: DType) -> BenchSetup {
        // Step-3 full-attn shape.
        let (n_heads, head_dim) = (64usize, 128usize);
        let n = n_heads * head_dim;
        BenchSetup::new(ffai_attn_head_gate::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("attn", n, dt))
            .buffer(BenchBuffer::random("gate", n_heads, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .grid_1d(n, 256)
            .bytes_moved(((2 * n + n_heads) * dt.size_bytes()) as u64)
    }
}

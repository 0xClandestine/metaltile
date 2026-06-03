//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Clamped SwiGLU activation — `clip(silu(gate), max=L) * clip(up, -L, L)`.
//!
//! Drop-in replacement for [`mt_swiglu`](super::swiglu::mt_swiglu) for
//! checkpoints that ship per-layer activation-clipping limits. Pattern:
//!
//! ```text
//!   out[i] = clip(silu(gate[i]), max=L) * clip(up[i], -L, L)
//! ```
//!
//! Two checkpoints in the wild use this shape — gpt-oss-120B and
//! StepFun's Step-3.7-Flash (per-layer `swiglu_limits` / `swiglu_limits_shared`
//! lists that are non-zero on a small subset of layers, zero elsewhere).
//! For layers whose limit is zero the caller should dispatch the plain
//! [`mt_swiglu`](super::swiglu::mt_swiglu); the clamped variant is the
//! one to reach for on the marked layers.
//!
//! Clipping happens on the f32 intermediates inside the kernel, before
//! the final cast back to `T`, so quant-stats fit the clipped range
//! regardless of activation dtype.
//!
//! ## ABI
//!
//! ```text
//!   gate  [N] T   — w_gate · x
//!   up    [N] T   — w_up · x
//!   out   [N] T   — clipped SwiGLU output
//!   limit f32     — constexpr; non-negative clip bound. `limit <= 0`
//!                    collapses to plain SwiGLU (no clip).
//! ```
//!
//! Grid is 1D elementwise: one thread per output position. Caller
//! drives `grid_1d(n, 256)`.

use metaltile::kernel;

// Bare `#[kernel]` — the legacy `bench(...)` registration's `Binary`
// class can't represent the extra `limit: f32` runtime scalar; the new
// declarative `#[bench]` attribute on `kernel_benches::bench_clamped_swiglu`
// below registers the kernel for `tile bench` directly.
#[kernel]
pub fn mt_clamped_swiglu<T>(
    gate: Tensor<T>,
    up: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] limit: f32,
) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // silu(g) = g * sigmoid(g). Free-function `exp` + f32 literals so the
    // binding isn't elided (the method form `(-g).exp()` nested in a
    // larger expr drops out of codegen — see moe_router_sqrtsoftplus).
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - g));
    let s_raw = g * sig;
    // Clip via `select`, not `min`/`max`: the DSL's min/max overloads
    // are ambiguous on mixed int/float operands (the dsv4_swiglu_limit
    // sibling clamps the same way). `limit <= 0` collapses to plain
    // SwiGLU. silu's upper tail is clipped one-sided; `up` two-sided.
    let active = limit > 0.0f32;
    let neg = 0.0f32 - limit;
    let s_clipped = select(active, select(s_raw > limit, limit, s_raw), s_raw);
    let u_hi = select(u > limit, limit, u);
    let u_lo = select(u_hi < neg, neg, u_hi);
    let u_clipped = select(active, u_lo, u);
    store(out[idx], (s_clipped * u_clipped).cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_clamped_swiglu;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, limit: f32, dt: DType) -> TestSetup {
        let gate: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.2 - 1.0).collect();
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let u_dt = unpack_f32(&pack_f32(&up, dt), dt);
        let expected: Vec<f32> = g_dt
            .iter()
            .zip(&u_dt)
            .map(|(&g, &u)| {
                let s = g / (1.0 + (-g).exp()); // silu(g) = g * sigmoid(g)
                let (s_c, u_c) =
                    if limit > 0.0 { (s.min(limit), u.max(-limit).min(limit)) } else { (s, u) };
                s_c * u_c
            })
            .collect();
        TestSetup::new(mt_clamped_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(&up, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("limit", limit)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_clamped_swiglu_active(dt: DType) -> TestSetup { setup(1024, 7.0, dt) }

    /// `limit == 0` collapses to plain SwiGLU — equivalence with
    /// [`mt_swiglu`](super::super::swiglu::mt_swiglu) is the
    /// invariant we ship the per-layer dispatch on.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_clamped_swiglu_zero_limit_equals_plain(dt: DType) -> TestSetup {
        setup(1024, 0.0, dt)
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_clamped_swiglu;

    #[bench(name = "mlx/clamped_swiglu", dtypes = [f32, f16, bf16])]
    fn bench_clamped_swiglu(dt: DType) -> BenchSetup {
        let n = 1024 * 1024usize;
        BenchSetup::new(mt_clamped_swiglu::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("limit", 7.0f32)
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }
}

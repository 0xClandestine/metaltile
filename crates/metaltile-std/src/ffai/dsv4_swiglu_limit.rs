//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 SwiGLU with `swiglu_limit` clamp — `silu(min(gate, limit)) *
//! clip(up, -limit, +limit)`. DeepSeek-V4 trains the gated MLP / MoE
//! experts with this clamp (swiglu_limit=10); unclamped `silu(gate)*up`
//! overflows fp16 in the deep, high-magnitude layers and makes the
//! batched-prefill vs sequential-decode paths diverge (large values
//! round differently across kernels — the clamp pins them identical).
//!
//! Drop-in replacement for `mt_swiglu` on the DSv4 gate/up → inner path.

use metaltile::kernel;

#[kernel]
pub fn ffai_dsv4_swiglu_limit<T>(
    gate: Tensor<T>,
    up: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] limit: f32,
) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // silu(min(gate, limit))
    let g_lim = select(g > limit, limit, g);
    let s = g_lim / (1.0f32 + exp(0.0f32 - g_lim));
    // clip(up, -limit, +limit)
    let neg = 0.0f32 - limit;
    let u_lo = select(u < neg, neg, u);
    let u_lim = select(u_lo > limit, limit, u_lo);
    store(out[idx], (s * u_lim).cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_swiglu_limit;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, dt: DType) -> TestSetup {
        let limit = 10.0f32;
        // Span values that exercise both clamp branches (|x| > 10).
        let gate: Vec<f32> = (0..n).map(|i| (i % 41) as f32 * 0.8 - 16.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 37) as f32 * 0.9 - 16.0).collect();
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let u_dt = unpack_f32(&pack_f32(&up, dt), dt);
        let expected: Vec<f32> = g_dt
            .iter()
            .zip(&u_dt)
            .map(|(&g, &u)| {
                let gl = g.min(limit);
                let s = gl / (1.0 + (-gl).exp());
                let ul = u.clamp(-limit, limit);
                s * ul
            })
            .collect();
        TestSetup::new(ffai_dsv4_swiglu_limit::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(&up, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("limit", limit)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // f16/bf16 tol mirrors the sister gated-product kernel
    // `moe_down_swiglu_accum` (also `silu(gate)*up`). The clamp pins the
    // product to |·| ≤ limit² (=100), and `silu` routes through GPU
    // fast-`exp`, so the f16 output round lands ~15 ulp off the libm CPU
    // oracle (err 7.8e-3 on M5; wider on older fast-math). f32 stays tight
    // at 1e-4 — it proves the logic; the half tols absorb fast-math+round.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-2, 2e-1])]
    fn test_ffai_dsv4_swiglu_limit(dt: DType) -> TestSetup { setup(1024, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_swiglu_limit;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_dsv4_swiglu_limit(dt: DType) -> BenchSetup {
        let n = 1024 * 1024usize;
        BenchSetup::new(ffai_dsv4_swiglu_limit::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("limit", 10.0f32)
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }
}

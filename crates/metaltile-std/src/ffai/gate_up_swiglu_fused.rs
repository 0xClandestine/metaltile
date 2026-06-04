//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused per-expert (gate gemv + up gemv + SwiGLU): `inner[r] =
//! silu(gate_w[r] · x) * (up_w[r] · x)`. Single dispatch replaces
//! gate_gemv → up_gemv → swiglu (3 dispatches).
//!
//! For MoE decode where each routed expert needs gate/up before
//! down, this fuses the 3-dispatch prelude into one. Saves 2
//! commands per expert × top-K per layer.
//!
//! Geometry: one threadgroup per inner row (`intermediate = mat
//! shape[0]`). Each TG does TWO `strided_reduce_dot` + `reduce_sum`
//! passes (one for gate, one for up), then a per-row silu+mul +
//! store of inner.

use metaltile::kernel;

#[kernel]
pub fn ffai_gate_up_swiglu_fused<T>(
    gate_w: Tensor<T>,
    up_w: Tensor<T>,
    x: Tensor<T>,
    mut inner: Tensor<T>,
    #[constexpr] k: u32,
) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let g_dot = strided_reduce_dot(gate_w, x, rs, rs, re);
    let g_full = reduce_sum(g_dot);
    let u_dot = strided_reduce_dot(up_w, x, rs, rs, re);
    let u_full = reduce_sum(u_dot);
    // Match the unfused gate-gemv → silu → mul chain's precision:
    // gate-gemv stores its result as T (f16/bf16 round-trip) before
    // mt_swiglu's silu reads it. Round g + u back to T then to f32 so
    // accumulated drift in long decode runs matches greedy on the
    // unfused path exactly.
    let g_t = g_full.cast::<T>();
    let u_t = u_full.cast::<T>();
    let g = g_t.cast::<f32>();
    let u = u_t.cast::<f32>();
    store(inner[row], silu(g) * u);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_gate_up_swiglu_fused;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(m: usize, k: usize, dt: DType) -> TestSetup {
        let gate: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let up: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let x: Vec<f32> = (0..k).map(|j| ((j % 11) as f32 - 5.0) * 0.03).collect();
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let u_dt = unpack_f32(&pack_f32(&up, dt), dt);
        let x_dt = unpack_f32(&pack_f32(&x, dt), dt);
        let expected: Vec<f32> = (0..m)
            .map(|r| {
                let g_val: f32 = (0..k).map(|j| g_dt[r * k + j] * x_dt[j]).sum();
                let u_val: f32 = (0..k).map(|j| u_dt[r * k + j] * x_dt[j]).sum();
                let sig = g_val / (1.0 + (-g_val).exp());
                sig * u_val
            })
            .collect();
        TestSetup::new(ffai_gate_up_swiglu_fused::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("gate_w", pack_f32(&gate, dt), dt))
            .input(TestBuffer::from_vec("up_w", pack_f32(&up, dt), dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::zeros("inner", m, dt))
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("inner", pack_f32(&expected, dt), dt))
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-1, 1.0])]
    fn test_gus_small(dt: DType) -> TestSetup { setup(16, 256, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_gate_up_swiglu_fused;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gus(dt: DType) -> BenchSetup {
        let (m, k) = (2048usize, 4096usize);
        BenchSetup::new(ffai_gate_up_swiglu_fused::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("gate_w", m * k, dt))
            .buffer(BenchBuffer::random("up_w", m * k, dt))
            .buffer(BenchBuffer::random("x", k, dt))
            .buffer(BenchBuffer::zeros("inner", m, dt).output())
            .constexpr("k", k as u32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * m * k * dt.size_bytes() + k * dt.size_bytes()) as u64)
    }
}

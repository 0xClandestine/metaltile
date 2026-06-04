//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused multi-expert (top-K=6) down gemv + weighted accumulate.
//!
//! For routed experts with PRE-DEQUANTED f16 down weights and
//! PRE-COMPUTED inner activations (post-swiglu), this kernel does the
//! full 6-way weighted sum into moeAccum in ONE dispatch:
//!
//! ```text
//!   moeAccum[r] += Σ_{k=0..5} w[k] * (down_w_k[r] · inner_k)
//! ```
//!
//! Replaces 18 dispatches per layer (6 gemv + 6 mul + 6 add) with 1.
//! Saves ~17 × 100 µs encoding overhead per layer × 43 layers ≈
//! 72 ms per token — the bulk of the dispatch-count overhead.
//!
//! Geometry: one threadgroup per output row of moeAccum. Each TG
//! sequentially walks the 6 routed slots, stages that slot's inner
//! activation in threadgroup memory, accumulates that slot's dot
//! product in a per-thread f32 register, then manually reduces the
//! six dot products via simd/threadgroup scratch.
//! This mirrors the barrier discipline used by
//! `moe_down_swiglu_accum_int4_chain8` and avoids chaining multiple
//! reduction helpers in one kernel body.

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_down_weighted_sum_6<T>(
    down_0: Tensor<T>,
    inner_0: Tensor<T>,
    down_1: Tensor<T>,
    inner_1: Tensor<T>,
    down_2: Tensor<T>,
    inner_2: Tensor<T>,
    down_3: Tensor<T>,
    inner_3: Tensor<T>,
    down_4: Tensor<T>,
    inner_4: Tensor<T>,
    down_5: Tensor<T>,
    inner_5: Tensor<T>,
    weights: Tensor<f32>,
    mut accum: Tensor<T>,
    #[constexpr] k: u32,
) {
    // DSv4 Flash uses moeIntermediate=2048. Keep the alloc literal fixed
    // so the generated MSL has static TG memory; Swift preconditions the
    // production wrapper to k <= 2048.
    threadgroup_alloc("tg_inner", 2048, "f32");
    threadgroup_alloc("sgs", 8, "f32");
    threadgroup_alloc("dots", 6, "f32");

    let row = program_id::<0>();
    let row_base = row * k;
    let w0 = load(weights[0u32]);
    let w1 = load(weights[1u32]);
    let w2 = load(weights[2u32]);
    let w3 = load(weights[3u32]);
    let w4 = load(weights[4u32]);
    let w5 = load(weights[5u32]);
    let iters = (k + lsize - 1u32) / lsize;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;

    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            threadgroup_store("tg_inner", d, load(inner_0[d]).cast::<f32>());
        }
    }
    threadgroup_barrier();
    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            acc0 =
                acc0 + load(down_0[row_base + d]).cast::<f32>() * threadgroup_load("tg_inner", d);
        }
    }
    threadgroup_barrier();

    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            threadgroup_store("tg_inner", d, load(inner_1[d]).cast::<f32>());
        }
    }
    threadgroup_barrier();
    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            acc1 =
                acc1 + load(down_1[row_base + d]).cast::<f32>() * threadgroup_load("tg_inner", d);
        }
    }
    threadgroup_barrier();

    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            threadgroup_store("tg_inner", d, load(inner_2[d]).cast::<f32>());
        }
    }
    threadgroup_barrier();
    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            acc2 =
                acc2 + load(down_2[row_base + d]).cast::<f32>() * threadgroup_load("tg_inner", d);
        }
    }
    threadgroup_barrier();

    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            threadgroup_store("tg_inner", d, load(inner_3[d]).cast::<f32>());
        }
    }
    threadgroup_barrier();
    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            acc3 =
                acc3 + load(down_3[row_base + d]).cast::<f32>() * threadgroup_load("tg_inner", d);
        }
    }
    threadgroup_barrier();

    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            threadgroup_store("tg_inner", d, load(inner_4[d]).cast::<f32>());
        }
    }
    threadgroup_barrier();
    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            acc4 =
                acc4 + load(down_4[row_base + d]).cast::<f32>() * threadgroup_load("tg_inner", d);
        }
    }
    threadgroup_barrier();

    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            threadgroup_store("tg_inner", d, load(inner_5[d]).cast::<f32>());
        }
    }
    threadgroup_barrier();
    for s_iter in range(0u32, iters, 1u32) {
        let d = s_iter * lsize + tid;
        if d < k {
            acc5 =
                acc5 + load(down_5[row_base + d]).cast::<f32>() * threadgroup_load("tg_inner", d);
        }
    }

    let sg0 = simd_sum(acc0);
    if simd_lane == 0u32 {
        threadgroup_store("sgs", simd_id, sg0);
    }
    threadgroup_barrier();
    if simd_id == 0u32 {
        let v = select(simd_lane < n_simd, threadgroup_load("sgs", simd_lane), 0.0f32);
        let total = simd_sum(v);
        if simd_lane == 0u32 {
            threadgroup_store("dots", 0u32, total);
        }
    }
    threadgroup_barrier();

    let sg1 = simd_sum(acc1);
    if simd_lane == 0u32 {
        threadgroup_store("sgs", simd_id, sg1);
    }
    threadgroup_barrier();
    if simd_id == 0u32 {
        let v = select(simd_lane < n_simd, threadgroup_load("sgs", simd_lane), 0.0f32);
        let total = simd_sum(v);
        if simd_lane == 0u32 {
            threadgroup_store("dots", 1u32, total);
        }
    }
    threadgroup_barrier();

    let sg2 = simd_sum(acc2);
    if simd_lane == 0u32 {
        threadgroup_store("sgs", simd_id, sg2);
    }
    threadgroup_barrier();
    if simd_id == 0u32 {
        let v = select(simd_lane < n_simd, threadgroup_load("sgs", simd_lane), 0.0f32);
        let total = simd_sum(v);
        if simd_lane == 0u32 {
            threadgroup_store("dots", 2u32, total);
        }
    }
    threadgroup_barrier();

    let sg3 = simd_sum(acc3);
    if simd_lane == 0u32 {
        threadgroup_store("sgs", simd_id, sg3);
    }
    threadgroup_barrier();
    if simd_id == 0u32 {
        let v = select(simd_lane < n_simd, threadgroup_load("sgs", simd_lane), 0.0f32);
        let total = simd_sum(v);
        if simd_lane == 0u32 {
            threadgroup_store("dots", 3u32, total);
        }
    }
    threadgroup_barrier();

    let sg4 = simd_sum(acc4);
    if simd_lane == 0u32 {
        threadgroup_store("sgs", simd_id, sg4);
    }
    threadgroup_barrier();
    if simd_id == 0u32 {
        let v = select(simd_lane < n_simd, threadgroup_load("sgs", simd_lane), 0.0f32);
        let total = simd_sum(v);
        if simd_lane == 0u32 {
            threadgroup_store("dots", 4u32, total);
        }
    }
    threadgroup_barrier();

    let sg5 = simd_sum(acc5);
    if simd_lane == 0u32 {
        threadgroup_store("sgs", simd_id, sg5);
    }
    threadgroup_barrier();
    if simd_id == 0u32 {
        let v = select(simd_lane < n_simd, threadgroup_load("sgs", simd_lane), 0.0f32);
        let total = simd_sum(v);
        if simd_lane == 0u32 {
            threadgroup_store("dots", 5u32, total);
        }
    }
    threadgroup_barrier();

    let prev = load(accum[row]).cast::<f32>();
    let d0 = threadgroup_load("dots", 0u32).cast::<T>().cast::<f32>();
    let c0 = (w0 * d0).cast::<T>().cast::<f32>();
    let a0 = (prev + c0).cast::<T>().cast::<f32>();
    let d1 = threadgroup_load("dots", 1u32).cast::<T>().cast::<f32>();
    let c1 = (w1 * d1).cast::<T>().cast::<f32>();
    let a1 = (a0 + c1).cast::<T>().cast::<f32>();
    let d2 = threadgroup_load("dots", 2u32).cast::<T>().cast::<f32>();
    let c2 = (w2 * d2).cast::<T>().cast::<f32>();
    let a2 = (a1 + c2).cast::<T>().cast::<f32>();
    let d3 = threadgroup_load("dots", 3u32).cast::<T>().cast::<f32>();
    let c3 = (w3 * d3).cast::<T>().cast::<f32>();
    let a3 = (a2 + c3).cast::<T>().cast::<f32>();
    let d4 = threadgroup_load("dots", 4u32).cast::<T>().cast::<f32>();
    let c4 = (w4 * d4).cast::<T>().cast::<f32>();
    let a4 = (a3 + c4).cast::<T>().cast::<f32>();
    let d5 = threadgroup_load("dots", 5u32).cast::<T>().cast::<f32>();
    let c5 = (w5 * d5).cast::<T>().cast::<f32>();
    let a5 = (a4 + c5).cast::<T>().cast::<f32>();
    store(accum[row], a5);
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_moe_down_weighted_sum_6;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(m: usize, k: usize, dt: DType) -> TestSetup {
        let weights = [0.20f32, 0.18, 0.17, 0.16, 0.15, 0.14];
        let mut dws: Vec<Vec<f32>> = Vec::new();
        let mut inns: Vec<Vec<f32>> = Vec::new();
        for slot in 0..6 {
            let dw: Vec<f32> =
                (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01 * (slot as f32 + 1.0)).collect();
            let inn: Vec<f32> = (0..k).map(|j| ((j % 13) as f32 - 6.0) * 0.02).collect();
            dws.push(dw);
            inns.push(inn);
        }
        let accum_in: Vec<f32> = (0..m).map(|r| (r as f32 % 5.0) * 0.1).collect();
        let dws_dt: Vec<Vec<f32>> = dws.iter().map(|d| unpack_f32(&pack_f32(d, dt), dt)).collect();
        let inns_dt: Vec<Vec<f32>> =
            inns.iter().map(|i| unpack_f32(&pack_f32(i, dt), dt)).collect();
        let accum_dt = unpack_f32(&pack_f32(&accum_in, dt), dt);
        let expected: Vec<f32> = (0..m)
            .map(|r| {
                let mut s = accum_dt[r];
                for slot in 0..6 {
                    let dot: f32 = (0..k).map(|j| dws_dt[slot][r * k + j] * inns_dt[slot][j]).sum();
                    s += weights[slot] * dot;
                }
                s
            })
            .collect();
        TestSetup::new(ffai_moe_down_weighted_sum_6::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("down_0", pack_f32(&dws[0], dt), dt))
            .input(TestBuffer::from_vec("inner_0", pack_f32(&inns[0], dt), dt))
            .input(TestBuffer::from_vec("down_1", pack_f32(&dws[1], dt), dt))
            .input(TestBuffer::from_vec("inner_1", pack_f32(&inns[1], dt), dt))
            .input(TestBuffer::from_vec("down_2", pack_f32(&dws[2], dt), dt))
            .input(TestBuffer::from_vec("inner_2", pack_f32(&inns[2], dt), dt))
            .input(TestBuffer::from_vec("down_3", pack_f32(&dws[3], dt), dt))
            .input(TestBuffer::from_vec("inner_3", pack_f32(&inns[3], dt), dt))
            .input(TestBuffer::from_vec("down_4", pack_f32(&dws[4], dt), dt))
            .input(TestBuffer::from_vec("inner_4", pack_f32(&inns[4], dt), dt))
            .input(TestBuffer::from_vec("down_5", pack_f32(&dws[5], dt), dt))
            .input(TestBuffer::from_vec("inner_5", pack_f32(&inns[5], dt), dt))
            .input(TestBuffer::from_vec("weights", pack_f32(&weights, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("accum", pack_f32(&accum_in, dt), dt))
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("accum", pack_f32(&expected, dt), dt))
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-1, 2.0])]
    fn test_mds_small(dt: DType) -> TestSetup { setup(16, 256, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_moe_down_weighted_sum_6;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mds(dt: DType) -> BenchSetup {
        let (m, k) = (4096usize, 2048usize);
        BenchSetup::new(ffai_moe_down_weighted_sum_6::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("down_0", m * k, dt))
            .buffer(BenchBuffer::random("inner_0", k, dt))
            .buffer(BenchBuffer::random("down_1", m * k, dt))
            .buffer(BenchBuffer::random("inner_1", k, dt))
            .buffer(BenchBuffer::random("down_2", m * k, dt))
            .buffer(BenchBuffer::random("inner_2", k, dt))
            .buffer(BenchBuffer::random("down_3", m * k, dt))
            .buffer(BenchBuffer::random("inner_3", k, dt))
            .buffer(BenchBuffer::random("down_4", m * k, dt))
            .buffer(BenchBuffer::random("inner_4", k, dt))
            .buffer(BenchBuffer::random("down_5", m * k, dt))
            .buffer(BenchBuffer::random("inner_5", k, dt))
            .buffer(BenchBuffer::random("weights", 6, DType::F32))
            .buffer(BenchBuffer::random("accum", m, dt).output())
            .constexpr("k", k as u32)
            .grid_3d(m as u32, 1, 1, [256, 1, 1])
            .bytes_moved(
                (6 * m * k * dt.size_bytes() + 6 * k * dt.size_bytes() + m * dt.size_bytes())
                    as u64,
            )
    }
}

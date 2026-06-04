//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Multi-query causal sliding-window SDPA for head_dim=512 with attention
//! sink — the PREFILL counterpart of `ffai_sdpa_decode_d512_sink`. Each
//! (q_head, q_pos) threadgroup runs the same flash online-softmax over the
//! KV cache, but bounded causally: query at absolute position
//! `p = kv_base + q_pos` attends KV `[max(0, p+1-window) .. p]`. For the
//! DSv4 full-attn (compress_ratio=0) layers, `window=128`; pass a huge
//! window for full causal. MQA (n_kv_heads via heads_per_group).
//!
//! q/out are `[n_query, n_q_heads, head_dim]`; k/v are `[kv_stride,
//! head_dim]` per kv-head (the resident/prefill KV cache, absolute pos).

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_sdpa_prefill_d512_sink<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    sink_logit: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] window: u32,
    #[constexpr] kv_base: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let q_pos = tgid_y;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    threadgroup_alloc("tg_out3", 1056);
    let q_off = (q_pos * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 16u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let q4 = load(q[q_off + d0 + 4u32]).cast::<f32>() * scale;
    let q5 = load(q[q_off + d0 + 5u32]).cast::<f32>() * scale;
    let q6 = load(q[q_off + d0 + 6u32]).cast::<f32>() * scale;
    let q7 = load(q[q_off + d0 + 7u32]).cast::<f32>() * scale;
    let q8 = load(q[q_off + d0 + 8u32]).cast::<f32>() * scale;
    let q9 = load(q[q_off + d0 + 9u32]).cast::<f32>() * scale;
    let q10 = load(q[q_off + d0 + 10u32]).cast::<f32>() * scale;
    let q11 = load(q[q_off + d0 + 11u32]).cast::<f32>() * scale;
    let q12 = load(q[q_off + d0 + 12u32]).cast::<f32>() * scale;
    let q13 = load(q[q_off + d0 + 13u32]).cast::<f32>() * scale;
    let q14 = load(q[q_off + d0 + 14u32]).cast::<f32>() * scale;
    let q15 = load(q[q_off + d0 + 15u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    let mut o4 = 0.0f32;
    let mut o5 = 0.0f32;
    let mut o6 = 0.0f32;
    let mut o7 = 0.0f32;
    let mut o8 = 0.0f32;
    let mut o9 = 0.0f32;
    let mut o10 = 0.0f32;
    let mut o11 = 0.0f32;
    let mut o12 = 0.0f32;
    let mut o13 = 0.0f32;
    let mut o14 = 0.0f32;
    let mut o15 = 0.0f32;
    // Causal sliding-window KV range for this query's absolute position.
    let p1 = kv_base + q_pos + 1u32;
    let lo = select(p1 > window, p1 - window, 0u32);
    for _t in range(lo + sg, p1, ns) {
        let base = kv_head_base + _t * head_dim;
        let kv0 = base + d0;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv0 + 1u32]).cast::<f32>();
        let k2 = load(k[kv0 + 2u32]).cast::<f32>();
        let k3 = load(k[kv0 + 3u32]).cast::<f32>();
        let k4 = load(k[kv0 + 4u32]).cast::<f32>();
        let k5 = load(k[kv0 + 5u32]).cast::<f32>();
        let k6 = load(k[kv0 + 6u32]).cast::<f32>();
        let k7 = load(k[kv0 + 7u32]).cast::<f32>();
        let k8 = load(k[kv0 + 8u32]).cast::<f32>();
        let k9 = load(k[kv0 + 9u32]).cast::<f32>();
        let k10 = load(k[kv0 + 10u32]).cast::<f32>();
        let k11 = load(k[kv0 + 11u32]).cast::<f32>();
        let k12 = load(k[kv0 + 12u32]).cast::<f32>();
        let k13 = load(k[kv0 + 13u32]).cast::<f32>();
        let k14 = load(k[kv0 + 14u32]).cast::<f32>();
        let k15 = load(k[kv0 + 15u32]).cast::<f32>();
        let partial = q0 * k0
            + q1 * k1
            + q2 * k2
            + q3 * k3
            + q4 * k4
            + q5 * k5
            + q6 * k6
            + q7 * k7
            + q8 * k8
            + q9 * k9
            + q10 * k10
            + q11 * k11
            + q12 * k12
            + q13 * k13
            + q14 * k14
            + q15 * k15;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv0 + 1u32]).cast::<f32>();
        let v2 = load(v[kv0 + 2u32]).cast::<f32>();
        let v3 = load(v[kv0 + 3u32]).cast::<f32>();
        let v4 = load(v[kv0 + 4u32]).cast::<f32>();
        let v5 = load(v[kv0 + 5u32]).cast::<f32>();
        let v6 = load(v[kv0 + 6u32]).cast::<f32>();
        let v7 = load(v[kv0 + 7u32]).cast::<f32>();
        let v8 = load(v[kv0 + 8u32]).cast::<f32>();
        let v9 = load(v[kv0 + 9u32]).cast::<f32>();
        let v10 = load(v[kv0 + 10u32]).cast::<f32>();
        let v11 = load(v[kv0 + 11u32]).cast::<f32>();
        let v12 = load(v[kv0 + 12u32]).cast::<f32>();
        let v13 = load(v[kv0 + 13u32]).cast::<f32>();
        let v14 = load(v[kv0 + 14u32]).cast::<f32>();
        let v15 = load(v[kv0 + 15u32]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
        o4 = o4 * factor + weight * v4;
        o5 = o5 * factor + weight * v5;
        o6 = o6 * factor + weight * v6;
        o7 = o7 * factor + weight * v7;
        o8 = o8 * factor + weight * v8;
        o9 = o9 * factor + weight * v9;
        o10 = o10 * factor + weight * v10;
        o11 = o11 * factor + weight * v11;
        o12 = o12 * factor + weight * v12;
        o13 = o13 * factor + weight * v13;
        o14 = o14 * factor + weight * v14;
        o15 = o15 * factor + weight * v15;
    }
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    let g_max0 = threadgroup_load("tg_max", 0);
    let g_sum0 = threadgroup_load("tg_sum", 0);
    let sink = load(sink_logit[q_head]);
    let g_max = select(sink > g_max0, sink, g_max0);
    let g_sum = g_sum0 * exp(g_max0 - g_max) + exp(sink - g_max);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
            so3 = so3 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
        store(out[out_off + 2u32], so2.cast::<T>());
        store(out[out_off + 3u32], so3.cast::<T>());
    }
    threadgroup_barrier();
    threadgroup_store("tg_out0", idx, o4 * rescale);
    threadgroup_store("tg_out1", idx, o5 * rescale);
    threadgroup_store("tg_out2", idx, o6 * rescale);
    threadgroup_store("tg_out3", idx, o7 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so4 = 0.0f32;
        let mut so5 = 0.0f32;
        let mut so6 = 0.0f32;
        let mut so7 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so4 = so4 + threadgroup_load("tg_out0", ri);
            so5 = so5 + threadgroup_load("tg_out1", ri);
            so6 = so6 + threadgroup_load("tg_out2", ri);
            so7 = so7 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 4u32], so4.cast::<T>());
        store(out[out_off + 5u32], so5.cast::<T>());
        store(out[out_off + 6u32], so6.cast::<T>());
        store(out[out_off + 7u32], so7.cast::<T>());
    }
    threadgroup_barrier();
    threadgroup_store("tg_out0", idx, o8 * rescale);
    threadgroup_store("tg_out1", idx, o9 * rescale);
    threadgroup_store("tg_out2", idx, o10 * rescale);
    threadgroup_store("tg_out3", idx, o11 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so8 = 0.0f32;
        let mut so9 = 0.0f32;
        let mut so10 = 0.0f32;
        let mut so11 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so8 = so8 + threadgroup_load("tg_out0", ri);
            so9 = so9 + threadgroup_load("tg_out1", ri);
            so10 = so10 + threadgroup_load("tg_out2", ri);
            so11 = so11 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 8u32], so8.cast::<T>());
        store(out[out_off + 9u32], so9.cast::<T>());
        store(out[out_off + 10u32], so10.cast::<T>());
        store(out[out_off + 11u32], so11.cast::<T>());
    }
    threadgroup_barrier();
    threadgroup_store("tg_out0", idx, o12 * rescale);
    threadgroup_store("tg_out1", idx, o13 * rescale);
    threadgroup_store("tg_out2", idx, o14 * rescale);
    threadgroup_store("tg_out3", idx, o15 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so12 = 0.0f32;
        let mut so13 = 0.0f32;
        let mut so14 = 0.0f32;
        let mut so15 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so12 = so12 + threadgroup_load("tg_out0", ri);
            so13 = so13 + threadgroup_load("tg_out1", ri);
            so14 = so14 + threadgroup_load("tg_out2", ri);
            so15 = so15 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off + 12u32], so12.cast::<T>());
        store(out[out_off + 13u32], so13.cast::<T>());
        store(out[out_off + 14u32], so14.cast::<T>());
        store(out[out_off + 15u32], so15.cast::<T>());
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_prefill_d512_sink;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_sdpa_prefill_d512_sink(dt: DType) -> BenchSetup {
        let hd = 512usize;
        let n_q = 64usize;
        let n_query = 256usize;
        let kv = 256usize;
        BenchSetup::new(ffai_sdpa_prefill_d512_sink::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_query * n_q * hd, dt))
            .buffer(BenchBuffer::random("k", kv * hd, dt))
            .buffer(BenchBuffer::random("v", kv * hd, dt))
            .buffer(BenchBuffer::random("sink_logit", n_q, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_query * n_q * hd, dt).output())
            .constexpr("head_dim", hd as u32)
            .constexpr("n_q_heads", n_q as u32)
            .constexpr("kv_stride", kv as u32)
            .constexpr("heads_per_group", n_q as u32)
            .constexpr("window", 128u32)
            .constexpr("kv_base", 0u32)
            .constexpr("scale", 0.044194174f32)
            .grid_3d(n_q as u32, n_query as u32, 1, [32, 1, 1])
            .bytes_moved(((kv * hd * 2 + n_query * n_q * hd) * dt.size_bytes()) as u64)
    }
}

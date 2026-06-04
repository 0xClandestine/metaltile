//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-query decode SDPA with a **per-head learned attention sink read
//! from a buffer** — the GPT-OSS-20B variant of [`super::sdpa_decode`].
//!
//! `sdpa_decode` carries the sink as a *scalar constexpr* (`sink_logit`),
//! so the host must dispatch (or specialize) per routed head to vary it —
//! and GPT-OSS has a **distinct learned sink per head**. This variant takes
//! `sink` as a `[n_q_heads]` f32 buffer and loads `sink[q_head]` inside the
//! kernel, so all heads run in one dispatch (grid = one threadgroup per
//! q_head, `tgid_x = q_head`) and the host-side per-layer sink sync in
//! `GPTOSSAttention.forward` goes away.
//!
//! The sink is a virtual key with score `sink[q_head]` and value 0: it
//! contributes `exp(sink[q_head] − g_max)` to the softmax denominator but
//! nothing to the output accumulator (so a confidently-attending head can
//! down-weight all real keys). Everything else — GQA, the sink-token range
//! `[0, sink_end)` + sliding window `[window_start, n_kv)`, the online
//! softmax and cross-simdgroup reduction — is byte-identical to
//! `sdpa_decode`.
//!
//! Q / out `[n_q_heads, head_dim]`; K / V `[n_kv_heads, kv_stride, head_dim]`;
//! `sink` `[n_q_heads]` f32.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction, TPG=1024, one threadgroup per q_head — dispatch with
//! `grid_3d(n_q_heads, 1, 1, [1024, 1, 1])`. `head_dim` a multiple of 4
//! (4 elements per lane). `window_start >= sink_end` (sparse path).

use metaltile::kernel;

#[kernel]
pub fn ffai_sdpa_decode_sink_buf<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    sink: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] sink_end: u32,
    #[constexpr] window_start: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
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
    let q_off = q_head * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    // Per-head learned sink logit (this threadgroup's q_head).
    let sink_logit = load(sink[q_head]);
    // 4 elements per lane; bounds-masked so any head_dim that is a multiple
    // of 4 and ≤ 128 works (GPT-OSS d64 → lanes 16..31 idle, d128 → all
    // lanes participate). Out-of-range lanes load 0 and skip their store.
    let d0 = lane * 4u32;
    let d1 = d0 + 1u32;
    let d2 = d0 + 2u32;
    let d3 = d0 + 3u32;
    let d0s = select(d0 < head_dim, d0, 0u32);
    let d1s = select(d1 < head_dim, d1, 0u32);
    let d2s = select(d2 < head_dim, d2, 0u32);
    let d3s = select(d3 < head_dim, d3, 0u32);
    let q0 = select(d0 < head_dim, load(q[q_off + d0s]).cast::<f32>() * scale, 0.0f32);
    let q1 = select(d1 < head_dim, load(q[q_off + d1s]).cast::<f32>() * scale, 0.0f32);
    let q2 = select(d2 < head_dim, load(q[q_off + d2s]).cast::<f32>() * scale, 0.0f32);
    let q3 = select(d3 < head_dim, load(q[q_off + d3s]).cast::<f32>() * scale, 0.0f32);
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    // Sink-token range pass `[0, sink_end)` (collapses when sink_end == 0).
    for _t in range(sg, sink_end, ns) {
        let base = kv_head_base + _t * head_dim;
        // Clamped element indices keep every load in-bounds; idle lanes
        // (d ≥ head_dim) have q = 0 so they add nothing to the score, and
        // their V accumulation is never stored.
        let kv0 = base + d0s;
        let kv1 = base + d1s;
        let kv2 = base + d2s;
        let kv3 = base + d3s;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let k2 = load(k[kv2]).cast::<f32>();
        let k3 = load(k[kv3]).cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
        let v3 = load(v[kv3]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
    }
    // Window pass `[window_start, n_kv)` (dense path: window_start == 0).
    for _t in range(sg + window_start, n_kv, ns) {
        let base = kv_head_base + _t * head_dim;
        // Clamped element indices keep every load in-bounds; idle lanes
        // (d ≥ head_dim) have q = 0 so they add nothing to the score, and
        // their V accumulation is never stored.
        let kv0 = base + d0s;
        let kv1 = base + d1s;
        let kv2 = base + d2s;
        let kv3 = base + d3s;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let k2 = load(k[kv2]).cast::<f32>();
        let k3 = load(k[kv3]).cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        let v2 = load(v[kv2]).cast::<f32>();
        let v3 = load(v[kv3]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
    }
    // ── Cross-simdgroup reduction: max + sum_exp (incl. the sink) ──
    if lane == 0u32 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0u32 {
        let g_max_raw = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        // The sink is a virtual key — its score must enter the global max.
        // Carry it on lane 0 only so simd_max sees it exactly once.
        let g_max_in =
            select(lane == 0u32, select(g_max_raw > sink_logit, g_max_raw, sink_logit), g_max_raw);
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_raw - g_max), 0.0f32);
        // Sink contributes exp(sink_logit − g_max) to the denominator
        // (value 0 → nothing to the output), counted once on lane 0.
        let sink_sum = exp(sink_logit - g_max);
        let g_sum = simd_sum(g_sum_in + select(lane == 0u32, sink_sum, 0.0f32));
        if lane == 0u32 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
    let stride = ns + 1u32;
    let idx = lane * stride + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();
    if sg == 0u32 {
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
        if d0 < head_dim {
            store(out[out_off], so0.cast::<T>());
        }
        if d1 < head_dim {
            store(out[out_off + 1u32], so1.cast::<T>());
        }
        if d2 < head_dim {
            store(out[out_off + 2u32], so2.cast::<T>());
        }
        if d3 < head_dim {
            store(out[out_off + 3u32], so3.cast::<T>());
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_decode_sink_buf;
    use crate::utils::{pack_f32, unpack_f32};

    /// Single-query decode oracle with a per-head sink (value 0) folded
    /// into the softmax denominator. Dense path (sink_end=0, window_start=0).
    #[allow(clippy::too_many_arguments)]
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sink: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv: usize,
        kv_stride: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_q_heads * head_dim];
        // `qh` indexes several arrays (out/q/sink) with different strides, so
        // a single `enumerate()` would not replace the range loop cleanly.
        #[allow(clippy::needless_range_loop)]
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = qh * head_dim;
            let kv_slab = kvh * kv_stride * head_dim;
            let mut scores = vec![0.0f32; n_kv];
            for (t, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[kv_slab + t * head_dim + d];
                }
                *s = dot * scale;
            }
            // Global max includes the sink logit.
            let mut m = sink[qh];
            for &s in scores.iter() {
                if s > m {
                    m = s;
                }
            }
            let mut denom = (sink[qh] - m).exp();
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                denom += *s;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &s) in scores.iter().enumerate() {
                    acc += s * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    fn setup(dt: DType, head_dim: usize) -> TestSetup {
        let (n_q_heads, n_kv_heads) = (8usize, 2usize); // GQA group 4 — GPT-OSS.
        let n_kv = 40usize;
        let kv_stride = n_kv;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        // Distinct per-head sinks (some large enough to dominate).
        let sink: Vec<f32> = (0..n_q_heads).map(|i| (i as f32 - 3.5) * 0.8).collect();
        let expected =
            naive(&q, &k, &v, &sink, n_q_heads, n_kv_heads, head_dim, n_kv, kv_stride, scale);

        TestSetup::new(ffai_sdpa_decode_sink_buf::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec(
                "sink",
                sink.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("sink_end", 0u32)
            .constexpr("window_start", 0u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
    }

    // GPT-OSS head_dim 64 (ragged: lanes 16..31 idle, bounds-masked).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_decode_sink_buf_d64(dt: DType) -> TestSetup { setup(dt, 64) }

    // head_dim 128 (every lane participates — the dense decode width).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa_decode_sink_buf_d128(dt: DType) -> TestSetup { setup(dt, 128) }
}

/// New-syntax bench: GPT-OSS decode step, 8 heads, head_dim 64, 2k cache.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_decode_sink_buf;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_sink_buf(dt: DType) -> BenchSetup {
        let head_dim = 64usize;
        let (n_q_heads, n_kv_heads) = (8usize, 2usize);
        let n_kv = 2048usize;
        let kv_stride = n_kv;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bytes = (n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_sink_buf::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("sink", n_q_heads, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("sink_end", 0u32)
            .constexpr("window_start", 0u32)
            .constexpr("scale", scale)
            .grid_3d(n_q_heads as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
            // Single-query decode: 4·H·Nkv·D (QK + softmax·V over the cache).
            .flops(4 * (n_q_heads as u64) * (n_kv as u64) * (head_dim as u64))
    }
}

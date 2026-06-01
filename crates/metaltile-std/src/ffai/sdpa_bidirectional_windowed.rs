//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! **Windowed** (block-diagonal) bidirectional SDPA — each query attends
//! only the keys in *its own window*, a contiguous `[seg_start, seg_start +
//! seg_len)` slice of the K/V sequence, instead of the full `[0, n_kv)`.
//!
//! This is the variant the **Qwen2.5-VL** vision tower's windowed-attention
//! blocks (and the **MiniCPM-V** resampler) need. Those towers run most
//! blocks under window attention (a token attends only within its spatial
//! window), with only a few full-attention blocks. The reference permutes
//! tokens into window order so each window is a contiguous run; the caller
//! does the same and passes, per query, the `[seg_start, seg_len)` of its
//! window in the (permuted) K/V sequence. With windows contiguous, this is
//! exactly [`super::sdpa_bidirectional`] with the K walk bounded to one
//! segment — same online-softmax + cross-simdgroup reduction, same
//! TPG=1024 reduction dispatch, so no machine-freeze hazard is introduced.
//!
//! `head_dim == 80` (Qwen2.5-VL: hidden 1280 / 16 heads). Ragged 3-element
//! layout (lanes 0..26 own indices 0..79, lane 26's 3rd element and lanes
//! 27..31 are bounds-masked) — identical to `ffai_sdpa_bidirectional_d80`.
//!
//! ## Window contract
//!
//! `seg_start` / `seg_len` are `[n_query]` u32 vectors. Query row `i`
//! attends keys `j ∈ [seg_start[i], seg_start[i] + seg_len[i])`. The caller
//! guarantees that range lies in `[0, n_kv)` and that q/k/v are already in
//! window-contiguous order (so a window is one slice in both q and kv).
//!
//! ## DISPATCH INVARIANTS
//!
//! Identical geometry to `sdpa_bidirectional` (Reduction, TPG=1024, one
//! threadgroup per `(query, q_head)`, grid `(n_q_heads*n_query*1024, 1, 1)`
//! via `grid_3d((n_q_heads*n_query), 1, 1, [1024,1,1])`), `head_dim == 80`.
//!
//! Q / `out` layout: `[n_query, n_q_heads, head_dim]` row-major.
//! K / V layout:     `[n_kv_heads, kv_stride, head_dim]` row-major.

use metaltile::kernel;

#[kernel]
pub fn ffai_sdpa_bidirectional_windowed_d80<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    seg_start: Tensor<u32>,
    seg_len: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] scale: f32,
) {
    let tg = tgid_x;
    let query_idx = tg / n_q_heads;
    let q_head = tg % n_q_heads;
    let kv_head = q_head / heads_per_group;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    // This query's window in the (permuted) K/V sequence.
    let win_start = load(seg_start[query_idx]);
    let win_end = win_start + load(seg_len[query_idx]);
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1056);
    threadgroup_alloc("tg_out1", 1056);
    threadgroup_alloc("tg_out2", 1056);
    let q_off = (query_idx * n_q_heads + q_head) * head_dim;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 3u32;
    let d1 = d0 + 1u32;
    let d2 = d0 + 2u32;
    let d0s = select(d0 < head_dim, d0, 0u32);
    let d1s = select(d1 < head_dim, d1, 0u32);
    let d2s = select(d2 < head_dim, d2, 0u32);
    let q0 = select(d0 < head_dim, load(q[q_off + d0s]).cast::<f32>() * scale, 0.0f32);
    let q1 = select(d1 < head_dim, load(q[q_off + d1s]).cast::<f32>() * scale, 0.0f32);
    let q2 = select(d2 < head_dim, load(q[q_off + d2s]).cast::<f32>() * scale, 0.0f32);
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    // Bounded K walk: only this query's window, distributed across the
    // simdgroups (each sg strides by ns within [win_start, win_end)).
    for _t in range(win_start + sg, win_end, ns) {
        let base = kv_head_base + _t * head_dim;
        let k0 = select(d0 < head_dim, load(k[base + d0s]).cast::<f32>(), 0.0f32);
        let k1 = select(d1 < head_dim, load(k[base + d1s]).cast::<f32>(), 0.0f32);
        let k2 = select(d2 < head_dim, load(k[base + d2s]).cast::<f32>(), 0.0f32);
        let partial = q0 * k0 + q1 * k1 + q2 * k2;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = select(d0 < head_dim, load(v[base + d0s]).cast::<f32>(), 0.0f32);
        let v1 = select(d1 < head_dim, load(v[base + d1s]).cast::<f32>(), 0.0f32);
        let v2 = select(d2 < head_dim, load(v[base + d2s]).cast::<f32>(), 0.0f32);
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
    }
    if lane == 0u32 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0u32 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
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
    threadgroup_barrier();
    if sg == 0u32 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * stride + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
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
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_sdpa_bidirectional_windowed_d80;
    use crate::utils::{pack_f32, unpack_f32};

    // Per (query, q_head): softmax(scale·Q·Kᵀ)·V over THIS query's window
    // `[seg_start[i], seg_start[i] + seg_len[i])`. Q/out `[n_query,
    // n_q_heads, head_dim]`, K/V `[n_kv_heads, kv_stride, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_windowed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seg_start: &[u32],
        seg_len: &[u32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_query: usize,
        kv_stride: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
        for r in 0..n_query {
            let ws = seg_start[r] as usize;
            let we = ws + seg_len[r] as usize;
            for qh in 0..n_q_heads {
                let kvh = qh / gqa;
                let q_off = (r * n_q_heads + qh) * head_dim;
                let kv_slab = kvh * kv_stride * head_dim;
                let mut scores = vec![0.0f32; we - ws];
                for (idx, t) in (ws..we).enumerate() {
                    let k_off = kv_slab + t * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    scores[idx] = dot * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (idx, t) in (ws..we).enumerate() {
                        acc += scores[idx] * inv * v[kv_slab + t * head_dim + d];
                    }
                    out[q_off + d] = acc;
                }
            }
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    /// Build a uniform-or-ragged window partition: windows of `win` keys,
    /// the last one short if `n_query` isn't a multiple of `win`.
    fn windows(n_query: usize, win: usize) -> (Vec<u32>, Vec<u32>) {
        let mut start = vec![0u32; n_query];
        let mut len = vec![0u32; n_query];
        for i in 0..n_query {
            let ws = (i / win) * win;
            let we = (ws + win).min(n_query);
            start[i] = ws as u32;
            len[i] = (we - ws) as u32;
        }
        (start, len)
    }

    fn setup(dt: DType, n_query: usize, win: usize) -> TestSetup {
        let head_dim = 80usize;
        let (n_q_heads, n_kv_heads) = (4usize, 4usize);
        let kv_stride = n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (seg_start, seg_len) = windows(n_query, win);

        let q = unpack_f32(&pack_f32(&ramp(n_query * n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected = naive_windowed(
            &q, &k, &v, &seg_start, &seg_len, n_q_heads, n_kv_heads, head_dim, n_query, kv_stride,
            scale,
        );

        TestSetup::new(ffai_sdpa_bidirectional_windowed_d80::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec(
                "seg_start",
                seg_start.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "seg_len",
                seg_len.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::U32,
            ))
            .input(TestBuffer::zeros("out", n_query * n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
    }

    // Uniform windows: 32 queries, window 8 → 4 equal windows.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_windowed_d80_uniform(dt: DType) -> TestSetup { setup(dt, 32, 8) }

    // Ragged: 20 queries, window 8 → [0,8),[8,16),[16,20) (last short).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_windowed_d80_ragged(dt: DType) -> TestSetup { setup(dt, 20, 8) }
}

/// New-syntax bench: Qwen2.5-VL windowed block at a native-resolution-ish
/// shape — 1024 patches, 16 heads, window 64.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_sdpa_bidirectional_windowed_d80;

    #[bench(name = "ffai/sdpa_bidirectional_windowed_d80", dtypes = [f32, f16, bf16])]
    fn bench_windowed_d80(dt: DType) -> BenchSetup {
        let head_dim = 80usize;
        let (n_q_heads, n_kv_heads) = (16usize, 16usize);
        let n_query = 1024usize;
        let kv_stride = n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let win = 64usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut seg_start = vec![0u32; n_query];
        let mut seg_len = vec![0u32; n_query];
        for i in 0..n_query {
            let ws = (i / win) * win;
            seg_start[i] = ws as u32;
            seg_len[i] = ((ws + win).min(n_query) - ws) as u32;
        }
        let bytes = (2 * n_query * n_q_heads * head_dim + 2 * n_kv_heads * n_query * head_dim)
            * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_bidirectional_windowed_d80::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_query * n_q_heads * head_dim, dt))
            .buffer(BenchBuffer::random("k", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::random("v", n_kv_heads * kv_stride * head_dim, dt))
            .buffer(BenchBuffer::from_vec(
                "seg_start",
                seg_start.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "seg_len",
                seg_len.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", n_query * n_q_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("scale", scale)
            .grid_3d((n_q_heads * n_query) as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
            // Windowed: each query attends only its `win` keys ⇒ useful work
            // is 4·H·Nq·win·D (QK + softmax·V), not the dense Nq² form.
            .flops(4 * (n_q_heads as u64) * (n_query as u64) * (win as u64) * (head_dim as u64))
    }
}

//! Decode-form scaled dot-product attention — `mt_sdpa_vector`.
//!
//! Faithful port of MLX `sdpa_vector<T, D, V=D>` template instantiation
//! `sdpa_vector_{tname}_128_128`. One threadgroup per Q head, 1024
//! threads = `BN × BD = 32 simdgroups × 32 lanes`. Each simdgroup walks
//! a stride-`BN` slice of `n_kv` positions, then a two-step cross-
//! simdgroup reduction combines the partial online-softmax results.
//!
//! Differs from `mt_sdpa` (same file family) only by adding **GQA**
//! support: `kv_head = q_head / gqa_factor`. When `gqa_factor = 1`
//! this is exactly `mt_sdpa` semantically — but with the `mlx`-side
//! comparison wired through the `SdpaVector` dispatch, which handles
//! the parameterised K/V head count and the per-Q-head dispatch shape
//! the GQA case needs.
//!
//! `head_dim` is hardcoded to 128: each lane owns `head_dim / BD = 4`
//! consecutive Q/K/V quartiles, the dot-product across `head_dim`
//! reduces via `simd_sum`, and the V accumulator stays in 4 thread-
//! local f32 registers throughout the n_kv walk.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sdpa",
    subop="sdpa_vector",
    class=SdpaVector,
    h=128,        // head_dim
    n_kv=4096,
    n_heads=32,   // n_q_heads
    gqa_factor=4, // 32 Q heads grouped onto 8 KV heads
    batch=1,
    tpg=1024,     // BN × BD = 32 × 32
    tol=1e-3,
    metal_file="scaled_dot_product_attention.metal",
)]
#[kernel]
pub fn mt_sdpa_vector<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    // Safety guard: wrong TPG (<32) makes n_simd=0 → infinite GPU loop.
    // Write zeros and return rather than freezing the machine.
    if ns == 0u32 {
        let out_off = q_head * head_dim;
        for d in range(0u32, head_dim, 1u32) {
            store(out[out_off + d], 0.0f32.cast::<T>());
        }
        return;
    }

    // 32-slot scalars for the cross-simdgroup max/sum + a 1024-slot output
    // buffer reused 4× in the reduction loop below. Matches MLX's layout:
    // 4 KB tg memory total. On M2 (32 KB tg/SM) that's 7 concurrent TGs/SM
    // vs the 2 we got from the old 16 KB / 4-array layout — the missing
    // occupancy factor that capped bf16 at 62% MT despite vectorized loads.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);

    let q_off = q_head * head_dim;
    // kv_stride is the allocated slots per KV head (= max_seq_len for a
    // fixed-size KV cache, = n_kv for the bench case where stride == count).
    let kv_base = kv_head * kv_stride * head_dim;
    let d0 = lane * 4u32;

    // Each lane pre-scales its 4 query elements once. K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;

    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;

    // Per iter: dot Q with one K row → online-softmax update → accumulate V row.
    // Pre-computing the 4 KV indices and issuing the 4 loads as a single run
    // (no BinOp/Cast interleaved) is what lets the vectorize pass collapse
    // them into one bfloat4 / half4 / float4 load — same shape as
    // `sdpa_decode_2pass_pass1`. Inline'd loads + casts broke the run before.
    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let kv2 = kv0 + 2u32;
        let kv3 = kv0 + 3u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k3_raw = load(k[kv3]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let k3 = k3_raw.cast::<f32>();
        let score = simd_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
    }

    // ── Cross-simdgroup reduction: max + sum_exp ───────────────────
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

    // ── Cross-simdgroup reduction: outputs ─────────────────────────
    //
    // Per output element: write per-(lane, sg) partial, barrier, transpose-
    // load (sg*ns + lane) × per-sg `factor_g`, simd_sum across the 32 lanes.
    // lane 0 of each sg then holds the reduced value for output position
    // `sg*4 + i`. Reuses the single 1 KB `tg_out` array for all 4 iters —
    // see the `threadgroup_alloc` comment above for the occupancy rationale.
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    // Scale this simdgroup's partial sums by its own correction factor before
    // contributing to the cross-simdgroup sum.  Each simdgroup has a different
    // run_max, so the factor must be applied *per-simdgroup* (not per-reading-
    // thread).  Writing the pre-scaled values lets the simd_sum below simply
    // add them without any additional scaling.
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);

    threadgroup_store("tg_out", lane * ns + sg, o0 * factor_g);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane)) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o1 * factor_g);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane)) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o2 * factor_g);
    threadgroup_barrier();
    let red2 = simd_sum(threadgroup_load("tg_out", sg * ns + lane)) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o3 * factor_g);
    threadgroup_barrier();
    let red3 = simd_sum(threadgroup_load("tg_out", sg * ns + lane)) * inv_sum;

    // lane 0 of each simdgroup writes its 4 elements at `q_off + sg*4`.
    // Output assignment is sg-indexed (was lane-indexed pre-occupancy fix),
    // matching MLX. f32→T narrowing is implicit at the MSL Store — adding
    // a `.cast::<T>()` would break the 4-consecutive-Store vectorize window
    // and double-wrap bf16 (`bfloat(bfloat(val))`).
    if lane == 0u32 {
        let out_off = q_off + sg * 4u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
        store(out[out_off + 2u32], red2);
        store(out[out_off + 3u32], red3);
    }
}

/// SDPA decode for small head dims (head_dim = 64, 2 elems per lane).
///
/// Same cross-simdgroup reduction as `mt_sdpa_vector` but each of the 32
/// lanes handles 2 consecutive Q/K/V elements (`d0 = lane*2`) instead of 4.
/// Used when `head_dim = 64` (e.g. Llama 3.2 1B / older 7B-class models).
/// Dispatch: `[n_q_heads, 1, 1] × [1024, 1, 1]` (32 simdgroups × 32 lanes).
#[bench_kernel(
    op="sdpa",
    subop="sdpa_vector_hd64",
    class=SdpaVector,
    h=64,
    n_kv=4096,
    n_heads=32,
    gqa_factor=4,
    batch=1,
    tpg=1024,
    tol=1e-3,
    metal_file="scaled_dot_product_attention.metal",
)]
#[kernel]
pub fn mt_sdpa_vector_hd64<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    // Safety guard: wrong TPG (<32) makes n_simd=0 → infinite GPU loop.
    // Write zeros and return rather than freezing the machine.
    if ns == 0u32 {
        let out_off = q_head * head_dim;
        for d in range(0u32, head_dim, 1u32) {
            store(out[out_off + d], 0.0f32.cast::<T>());
        }
        return;
    }

    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);

    let q_off = q_head * head_dim;
    let kv_base = kv_head * kv_stride * head_dim;
    // 2 elements per lane (head_dim=64, BD=32 → 64/32 = 2)
    let d0 = lane * 2u32;

    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;

    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;

    for t in range(sg, n_kv, ns) {
        let kv0 = kv_base + t * head_dim + d0;
        let kv1 = kv0 + 1u32;
        let k0 = load(k[kv0]).cast::<f32>();
        let k1 = load(k[kv1]).cast::<f32>();
        let score = simd_sum(q0 * k0 + q1 * k1);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0 = load(v[kv0]).cast::<f32>();
        let v1 = load(v[kv1]).cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
    }

    // Cross-simdgroup reduction (identical to 4-elem variant)
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

    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    // Pre-scale by this simdgroup's own correction factor so the cross-sg sum
    // can simply add already-corrected values.
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);

    threadgroup_store("tg_out", lane * ns + sg, o0 * factor_g);
    threadgroup_barrier();
    let red0 = simd_sum(threadgroup_load("tg_out", sg * ns + lane)) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o1 * factor_g);
    threadgroup_barrier();
    let red1 = simd_sum(threadgroup_load("tg_out", sg * ns + lane)) * inv_sum;

    // lane 0 of each simdgroup writes 2 elements at q_off + sg*2
    if lane == 0u32 {
        let out_off = q_off + sg * 2u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
    }
}

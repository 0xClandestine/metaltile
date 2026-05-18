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
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    // 32 slots per simdgroup-reduction array; 1024 slots per output
    // accumulator (one per thread). Matches mt_sdpa exactly.
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1024);
    threadgroup_alloc("tg_out1", 1024);
    threadgroup_alloc("tg_out2", 1024);
    threadgroup_alloc("tg_out3", 1024);

    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
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

    // Each simdgroup walks every (n_simd)th KV position. Per iteration:
    // simd_sum reduces the per-lane quartile dot product into a full
    // score, then we apply the online-softmax update and accumulate
    // the V quartile.
    for _t in range(sg, n_kv, ns) {
        let base = kv_base + _t * head_dim;
        let partial = q0 * load(k[base + d0]).cast::<f32>()
            + q1 * load(k[base + d0 + 1u32]).cast::<f32>()
            + q2 * load(k[base + d0 + 2u32]).cast::<f32>()
            + q3 * load(k[base + d0 + 3u32]).cast::<f32>();
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        o0 = o0 * factor + weight * load(v[base + d0]).cast::<f32>();
        o1 = o1 * factor + weight * load(v[base + d0 + 1u32]).cast::<f32>();
        o2 = o2 * factor + weight * load(v[base + d0 + 2u32]).cast::<f32>();
        o3 = o3 * factor + weight * load(v[base + d0 + 3u32]).cast::<f32>();
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
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = exp(run_max - g_max) / g_sum;
    let idx = lane * ns + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();

    // Simdgroup 0 sums the per-simdgroup contributions (one accumulator
    // per lane, n_simd entries each) and writes the final quartile.
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * ns + _g;
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
}

//! RMS normalization benchmark — #[kernel] DSL vs MLX metal/rms_norm.metal
//!
//! The kernel is generic over `N = tpg * 4` — each thread owns 4
//! consecutive elements, the partial sum-of-squares reduces across
//! the threadgroup. The bench wires `n=4096, tpg=1024` for the
//! hidden-axis case. For per-head normalisation (Qwen3-style q_norm
//! / k_norm pre-RoPE), the same kernel is dispatched as one
//! threadgroup per `(batch*token*n_heads)` row at `tpg = head_dim/4`
//! with the per-head_dim weight broadcast across all rows. The
//! per-head contract is pinned by
//! `tests/rms_norm_per_head_gpu.rs`.
//!
//! ## Dispatch constraint: N ≥ 128
//!
//! `mt_rms_norm`'s 4-elements-per-thread layout requires tpg ≥ 32 so
//! `reduce_sum(partial_ssq)` lowers to a well-defined `simd_sum`
//! (inactive lanes at tpg < 32 contribute undefined partials and the
//! ssq blows up). Models with head_dim < 128 (older 7B-class and
//! older 7B-class, head_dim=64) dispatch [`mt_rms_norm_small`] instead,
//! which uses a 2-elements-per-thread layout so head_dim=64 hits the
//! tpg=32 minimum.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rms_norm",
    subop="rms_norm",
    class=RowNorm,
    b=1024,
    n=4096,
    tpg=1024,
    reads=2,
    pre_weight=1.0,
    post_eps=1e-5,
    tol=1e-4,
    mlx="rms{tn}",
    metal_file="rms_norm.metal",
)]
#[kernel]
pub fn mt_rms_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // Each thread owns exactly 4 consecutive elements (N = TPG * 4).
    // Read x once, cache in registers, reuse for both ssq and output — 3 reads total.
    let base = rs + tid * 4u32;
    let col = tid * 4u32;
    let x0 = load(x[base]).cast::<f32>();
    let x1 = load(x[base + 1u32]).cast::<f32>();
    let x2 = load(x[base + 2u32]).cast::<f32>();
    let x3 = load(x[base + 3u32]).cast::<f32>();
    let partial_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
    store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
    store(out[base + 2u32], (x2 * rms * load(w[col + 2u32]).cast::<f32>()).cast::<T>());
    store(out[base + 3u32], (x3 * rms * load(w[col + 3u32]).cast::<f32>()).cast::<T>());
}

/// Small-head RMSNorm — 2 consecutive elements per thread, so
/// `N = tpg * 2`. Covers per-head dispatch at head_dim ∈ {64, 128,
/// 192, 256} (head_dim=64 → tpg=32 hits the single-simdgroup
/// minimum that the 4-element variant misses). At head_dim ≥ 128
/// the 4-element [`mt_rms_norm`] has better ILP per lane and is
/// preferred; this variant exists to cover the small-head_dim
/// regime (older 7B-class architectures) without a dispatch-time
/// fallback.
///
/// Algorithm-identical to `mt_rms_norm`: f32 accumulator for the
/// sum-of-squares, threadgroup-wide `reduce_sum`, `rsqrt(ssq/n + eps)`
/// scaling, per-element output store rounded through `T`.
#[kernel]
pub fn mt_rms_norm_small<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // 2 elements per thread → tpg = n / 2. The minimum supported is
    // tpg = 32 (one full simdgroup) → n ≥ 64.
    let base = rs + tid * 2u32;
    let col = tid * 2u32;
    let x0 = load(x[base]).cast::<f32>();
    let x1 = load(x[base + 1u32]).cast::<f32>();
    let partial_ssq = x0 * x0 + x1 * x1;
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
    store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
}

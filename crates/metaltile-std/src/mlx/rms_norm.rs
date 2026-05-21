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
//! Models with head_dim < 128 (older 7B-class, head_dim=64) dispatch
//! [`mt_rms_norm_small`] instead, which uses a 2-elements-per-thread
//! layout so head_dim=64 still hits the tpg=32 minimum.
//!
//! ## DISPATCH INVARIANTS
//!
//! This kernel is reduction-mode and has STRICT threadgroup-geometry
//! requirements. Violating any of these silently miscomputes the
//! output (best case) or pins the GPU in an infinite loop (worst
//! case — see FFAI post-mortem 2026-05-19). Consumers MUST encode
//! these as preconditions in their wrappers.
//!
//! - **`N = TPG * 4`.** Each thread owns exactly 4 consecutive
//!   elements of the row, loaded unconditionally at offsets
//!   `tid*4 + {0..3}`. The wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full Apple simdgroup).
//!   The cross-simdgroup combine reads `n_simd = TPG / 32` slots
//!   from threadgroup memory; with `TPG < 32` the combine reads
//!   zero everywhere and `tg_ssq` silently collapses to 0.
//! - **`TPG ≤ 1024`** (Apple's max-threads-per-threadgroup cap on
//!   M-series). Combined with `N = TPG*4`, this means `N ≤ 4096`;
//!   larger rows need the multi-row dispatch variant + chunking.
//! - **Combined**: `n` must be a multiple of 128 and `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row.** Multi-row dispatch uses
//!   `grid = (nRows * TPG, 1, 1)`, `tg = (TPG, 1, 1)`; Metal slices
//!   that into `nRows` threadgroups of `TPG` threads each.

use metaltile::{bench_kernel, kernel};

/// Cross-kernel callee: threadgroup-wide RMS inverse.
///
/// Given each thread's pre-computed `partial_ssq` (sum of squares for its
/// slice of the row), reduces across the threadgroup and returns:
///
/// ```text
///   rsqrt(reduce_sum(partial_ssq) / n + eps)
/// ```
///
/// This kernel exists **only** as a cross-kernel callee. Kernels that fuse
/// RMSNorm with a second operation (residual add, RoPE, quantized GEMV) call
/// it via the DSL cross-kernel syntax so that the reduction + rsqrt body is
/// expressed once and inlined by `KernelInlinePass` rather than copy-pasted.
///
/// ## Calling convention
///
/// ```rust
/// // In the caller kernel body (after computing per-thread partial_ssq):
/// let inv_rms = mt_rms_inv_scalar(partial_ssq, eps_buf, n);
/// ```
///
/// - `partial_ssq` → `KernelCallArg::Value`: the callee's param-load is
///   replaced by the caller's pre-computed scalar. No memory round-trip.
/// - `eps_buf`, `n` → `KernelCallArg::Tensor`: the callee's loads are kept
///   but renamed to the caller's buffer/constexpr names, so the inlined code
///   reads the correct per-kernel eps and row length.
/// - The output param `out` receives no arg; its store is skipped and the
///   stored `inv_rms` value is returned as the call result.
///
/// The index `0u32` in `load(partial_ssq[0u32])` is a placeholder —
/// `KernelInlinePass` skips this load entirely (Value arg) so it is never
/// emitted to MSL.
#[kernel]
pub fn mt_rms_inv_scalar(
    partial_ssq: Tensor<f32>,
    eps_buf: Tensor<f32>,
    mut out: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let v = load(partial_ssq[0u32]); // replaced by Value arg at inline time
    let tg_ssq = reduce_sum(v);
    let eps = load(eps_buf[0u32]);
    store(out[0u32], rsqrt(tg_ssq / n + eps));
}

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
    // The wrapper enforces this — but as belt-and-braces (the original
    // 2026-05-19 freeze came from a wrong-TPG dispatch in a sibling
    // kernel), clamp the load base for OOB threads and mask their SSQ
    // contribution + skip their stores. Threads with `col >= n` re-read
    // row[0..3] (benign, since `partial_ssq` for them is forced to 0),
    // participate in `reduce_sum` (required — Apple simdgroup
    // primitives need all lanes active), and skip their stores so
    // they don't trample a neighbouring row.
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col; // only used inside the in_bounds-guarded store block.
    // Read x once, cache in registers, reuse for both ssq and output — 3 reads total.
    let x0 = load(x[safe_base]).cast::<f32>();
    let x1 = load(x[safe_base + 1u32]).cast::<f32>();
    let x2 = load(x[safe_base + 2u32]).cast::<f32>();
    let x3 = load(x[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    // Mask OOB lanes to 0 contribution so `mean(x²) = tg_ssq / n` stays
    // correct: in-bounds lanes contribute their real x² values, the
    // sum/n divisor is unchanged. Only valid when the wrapper has
    // ensured the in-bounds lanes cover the full row exactly once;
    // duplicate / missing coverage is a wrapper bug we can't repair here.
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
        store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
        store(out[base + 2u32], (x2 * rms * load(w[col + 2u32]).cast::<f32>()).cast::<T>());
        store(out[base + 3u32], (x3 * rms * load(w[col + 3u32]).cast::<f32>()).cast::<T>());
    }
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
#[bench_kernel(
    op="rms_norm",
    subop="rms_norm_small",
    class=RowNorm,
    // Per-head dispatch shape: head_dim=64 row count tuned so the bench
    // walks a representative batched-prefill workload (4 batches × 16
    // tokens × 16 q heads at head_dim=64 = 1024 rows). Same `n × b`
    // total element count as the parent `mt_rms_norm` bench so the
    // GB/s comparison is apples-to-apples.
    b=1024,
    n=64,
    tpg=32,
    reads=2,
    pre_weight=1.0,
    post_eps=1e-5,
    tol=1e-4,
    mlx="rms{tn}",
    metal_file="rms_norm.metal",
)]
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

//! Fused RMSNorm + RoPE — normalizes a Q/K head then applies the
//! rotary position embedding, in one dispatch. Saves a kernel launch
//! per Q and per K vs separate `rms_norm` + `rope` calls (the
//! post-projection q_norm/k_norm path in Qwen3-style models).
//!
//! Non-traditional (paired) RoPE layout: element `i` rotates with
//! element `i + half`. One threadgroup per row — a row is one
//! `(batch, seq_pos, head)` slice of length `axis_size`; thread `lid`
//! owns the pair `(lid, lid + half)`.
//!
//! Phase 1 — `inv_rms = rsqrt(mean(x²) + eps)` via threadgroup
//! `reduce_sum` over both halves.
//! Phase 2 — `normed = w * x * inv_rms`, then rotate:
//!   `out[lid]      = normed_a·cos θ − normed_b·sin θ`
//!   `out[lid+half] = normed_a·sin θ + normed_b·cos θ`
//! where `θ = pos · inv_freqs[lid]` and the row's position is
//! `pos = offset + (row / n_heads) mod seq_len`.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel.
//!
//! - **`TPG = axis_size / 2`** — one thread per rotation pair.
//! - **`axis_size` must be a multiple of 64** so `TPG` is a multiple
//!   of 32 (a whole number of simdgroups) and `TPG ≥ 32`. Common head
//!   dims 64 / 128 / 256 satisfy this.
//! - **`TPG ≤ 1024`** → `axis_size ≤ 2048`.
//! - **Grid: 1 threadgroup per row**, `program_id::<0>()` = row index.
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_rope_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Fused RMSNorm + paired-layout RoPE for one Q/K head per threadgroup.
#[kernel]
pub fn ffai_rms_norm_rope<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    inv_freqs: Tensor<f32>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] axis_size: u32,
    #[constexpr] offset: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] seq_len: u32,
) {
    let row = program_id::<0>();
    let half = axis_size / 2u32;
    let rs = row * axis_size;
    let lid = tid;

    // Phase 1: per-thread pair → threadgroup-wide sum of squares.
    let v1 = load(x[rs + lid]).cast::<f32>();
    let v2 = load(x[rs + lid + half]).cast::<f32>();
    let tg_ssq = reduce_sum(v1 * v1 + v2 * v2);
    let eps = load(eps_buf[0]);
    let inv_rms = rsqrt(tg_ssq / axis_size + eps);

    // Phase 2: weight scale + RoPE rotation.
    let l = (row / n_heads) % seq_len;
    let pos = (offset + l).cast::<f32>();
    let theta = pos * load(inv_freqs[lid]);
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    let normed_a = v1 * load(w[lid]).cast::<f32>() * inv_rms;
    let normed_b = v2 * load(w[lid + half]).cast::<f32>() * inv_rms;

    store(out[rs + lid], (normed_a * cos_t - normed_b * sin_t).cast::<T>());
    store(out[rs + lid + half], (normed_a * sin_t + normed_b * cos_t).cast::<T>());
}

inventory::submit! {
    BenchSpec {
        op: "rms_norm_rope",
        subop: "rms_norm_rope",
        kernel_name: "ffai_rms_norm_rope",
        kernel_ir: ffai_rms_norm_rope::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-4,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

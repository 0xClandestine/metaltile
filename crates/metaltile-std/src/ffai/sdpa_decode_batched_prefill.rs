//! M7 K=8/16 batched-Q SDPA decode via `mt_sdpa_prefill_mma` reuse.
//!
//! No new MSL — these `BenchSpec` rows wire the existing FA-2
//! simdgroup-matrix prefill tile (`mt_sdpa_prefill_mma` from PR #47/#52)
//! through the `SdpaBatchedDecode { variant: PrefillTile, .. }` dispatch.
//! The runner (`run_sdpa_batched_decode_prefill_tile` in `run_spec.rs`)
//! pads Q up to BQ=32 rows and K/V up to `n_kv + 32` slots so the
//! prefill kernel's hardcoded causal mask gives Q[i] for i in 0..K the
//! speculative-decode-verify semantics:
//!
//! ```text
//! attended_kv(i) = [0, n_kv + i + 1)   for i in 0..K
//! ```
//!
//! i.e. prefix of size `n_kv` plus the K-1 preceding candidates (causal
//! across candidates — exactly what DFlash verify wants, mirroring
//! dflash-mlx's `verify_qmm`).
//!
//! Wasted Q work scales with `(BQ - K) / BQ` — 50% at K=16, 75% at
//! K=8. If the bench shows this killing the amortization win, a
//! hand-rolled BQ=K variant of the MMA kernel is the follow-up.
//!
//! Semantics divergence from K=2/4: the decode-form K=2/4 kernels
//! (`sdpa_decode_batched_q2` / `_q4`) implement **flat** batched-Q
//! decode — every Q[i] attends to the same `[0, n_kv)` range. K=8/16
//! via this prefill-tile reuse implements **causal** batched-Q decode.
//! Both are useful: flat for parallel verifier-only workloads, causal
//! for spec-decode verify. Consumers pick the right `BatchedDecodeVariant`
//! based on the semantics they need.

use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    mlx::steel::attn::steel_attention_mma::mt_sdpa_prefill_mma,
    spec::{BatchedDecodeVariant, BenchDispatch, BenchSpec},
};

// K=8 — 75% wasted Q rows inside the BQ=32 tile, but the K and V loads
// still amortize across 8 streams. Compares against
// `naive_sdpa_causal_prefix_f32` (q_len_padded=32, q_len_off=n_kv).
inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "sdpa_decode_batched_q8",
        kernel_name: "mt_sdpa_prefill_mma",
        kernel_ir: mt_sdpa_prefill_mma::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 2e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::SdpaBatchedDecode {
            head_dim: 128,
            n_kv: 4096,
            n_q_heads: 32,
            gqa_factor: 4,
            batch_q: 8,
            variant: BatchedDecodeVariant::PrefillTile { bq: 32, bk: 16, wm: 4, wn: 1 },
            tpg: 128,
        },
        kernel_mode: Some(KernelMode::SimdGroup2D),
    }
}

// K=16 — 50% wasted Q rows. This is the row that's closest to the
// DFlash hero path (BM=16, BN=16 per the dflash-mlx verify_qmm
// kernel), and where the prefill-tile reuse provides the most
// compelling K-vs-overhead ratio inside the existing kernel.
inventory::submit! {
    BenchSpec {
        op: "sdpa",
        subop: "sdpa_decode_batched_q16",
        kernel_name: "mt_sdpa_prefill_mma",
        kernel_ir: mt_sdpa_prefill_mma::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 2e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::SdpaBatchedDecode {
            head_dim: 128,
            n_kv: 4096,
            n_q_heads: 32,
            gqa_factor: 4,
            batch_q: 16,
            variant: BatchedDecodeVariant::PrefillTile { bq: 32, bk: 16, wm: 4, wn: 1 },
            tpg: 128,
        },
        kernel_mode: Some(KernelMode::SimdGroup2D),
    }
}

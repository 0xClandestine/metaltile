# metaltile kernel-op coverage audit

Generated: 2026-05-18 · Refreshed: 2026-05-20
Sources surveyed:
- MLX upstream `ml-explore/mlx@main` (commit `2414e5df`)
- MLX fork `ekryski/mlx@alpha` (commit `4919270e`)
- metaltile `thewafflehaus/metaltile:ek/aura-port` (commit `141a60b`)

## Summary

- Total kernel-op rows in this audit (union): **74**
- metaltile-ported kernel ops: **45 / 74 = 61 %** — 35 full ✓ (47 %), 10 partial ~ (14 %)
- **Still to cover: 29 ops not ported (✗)**, plus **10 partial ports** still to finish
- Upstream MLX kernel ops in the union: **50**; ekryski/alpha-only delta: **18** (of which **6** are FFAI-only — in neither MLX tree)

> **Note on the refresh.** The previous summary (54 rows / 27 ported / 50 %)
> was stale: it predated table rows added in later passes and undercounted the
> partial-port rows. The figures above are recounted directly from the 74-row
> table below at metaltile commit `141a60b`. The MLX-upstream and
> MLX-alpha columns were not re-verified against those repos (not checked out);
> only the metaltile column was re-surveyed against source.

## Op coverage table

| Op | MLX (upstream) | MLX (ekryski@alpha) | metaltile | Notes |
|---|---|---|---|---|
| arange | ✓ | ✓ | ✓ | `mlx/arange.rs` → `mt_arange`. Generic `T`. Direct port. |
| arg_reduce (argmax/argmin → float) | ✓ | ✓ | ~ | `mlx/arg_reduce.rs` → `mt_argmax_f32` only. f32 argmax only; argmin and bf16/f16 not yet. |
| arg_reduce (argmax → u32 index) | ✗ | ✗ | ✓ | `ffai/arg_reduce.rs` → `ffai_argmax<T>`. FFAI-only; integer-index sampler workhorse. |
| binary (elementwise add/sub/mul/div/min/max) | ✓ | ✓ | ✓ | `mlx/binary.rs` → 6 kernels. Generic `T`. Direct port. |
| binary_two (fused two-output elementwise) | ✓ | ✓ | ✓ | `mlx/binary_two.rs` → `mt_binary_two<T>`. |
| copy (contiguous) | ✓ | ✓ | ✓ | `mlx/copy.rs` → `mt_copy<T>`. |
| copy (strided / general) | ✓ | ✓ | ~ | `mlx/strided.rs` → `mt_strided_copy`. Limited stride dimensionality. |
| ternary (select) | ✓ | ✓ | ✓ | `mlx/ternary.rs` → `mt_select<T>`. |
| unary (exp/log/sqrt/rsqrt/abs/silu/etc.) | ✓ | ✓ | ✓ | `mlx/unary.rs` → 7+ kernels including `mt_silu`. |
| random (key hash → u32) | ✓ | ✓ | ✓ | `mlx/random.rs` → `mt_random_hash`. |
| reduce (sum/prod/max/min — all + row + col) | ✓ | ✓ | ~ | `mlx/reduce.rs` covers `all_reduce*` and `row_reduce`. Column-reduce partial; segmented-reduce missing. |
| sort | ✓ | ✓ | ~ | `mlx/sort.rs` → `mt_sort<T>`. Single-block path only; multi-block / segmented not yet. |
| scan (prefix sum) | ✓ | ✓ | ~ | `mlx/scan.rs` → `mt_scan<T>`. Inclusive sum only; exclusive / multi-op not yet. |
| softmax | ✓ | ✓ | ✓ | `mlx/softmax.rs` → `mt_softmax<T>` (looped + single-row collapsed). |
| logsumexp | ✓ | ✓ | ✓ | `mlx/logsumexp.rs` → `mt_logsumexp<T>`. |
| layer_norm | ✓ | ✓ | ✓ | `mlx/layer_norm.rs` → `mt_layer_norm<T>`. |
| rms_norm | ✓ | ✓ | ✓ | `mlx/rms_norm.rs` → `mt_rms_norm<T>` plus `mt_rms_norm_small<T>` (2-elem/thread small-head_dim variant for the per-head q_norm/k_norm dispatch). |
| rope (standard) | ✓ | ✓ | ✓ | `mlx/rope.rs` → `mt_rope` (fp16 only). |
| rope (Llama-3 banded) | ✗ | ✗ | ✓ | `ffai/rope_llama.rs` → `ffai_rope_llama<T>`. Decode-form, generic dtype, optional Llama-3 frequency-band scaling. No MLX counterpart. |
| sdpa_vector (prefill / generic) | ✓ | ✓ | ✓ | `mlx/scaled_dot_product_attention.rs` → `mt_sdpa<T>`. Scalar SDPA — sufficient for short sequences. |
| sdpa_vector (GQA decode, single pass) | ✓ | ✓ | ✓ | `mlx/sdpa_vector.rs` → `mt_sdpa_vector<T>`. head_dim=128 only; covers f32/f16/bf16. |
| sdpa_vector_2pass | ✓ | ✓ | ✓ | `ffai/sdpa_decode_2pass.rs`. head_dim=128 only. Upstream supports {64,96,128,256}. |
| sdpa_decode (FFAI production decode, decoupled `kv_stride`) | ✗ | ✗ | ✓ | `ffai/sdpa_decode.rs` → `ffai_sdpa_decode<T>`, plus `ffai/sdpa_decode_d64.rs` / `sdpa_decode_d256.rs` for head_dim {64, 256}. FFAI-only variant with `kv_stride` ≠ `n_kv` (pre-allocated max-seq cache); now covers head_dim ∈ {64, 128, 256} and a sliding-window + sink-token path (`sink_end` / `window_start` constexprs). |
| steel_attention (Flash, prefill) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention.rs` → `mt_sdpa_prefill<T>`. Scalar-flash prefill (BQ=4, online softmax, causal), generic `T`, head_dim=128. The old "`Op::FlashAttention` lowers to an error placeholder" blocker is resolved. |
| steel_attention_mma (Flash prefill, simdgroup-MMA) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_mma.rs` → `mt_sdpa_prefill_mma<T>`. Real simdgroup-matrix MMA path; generic `T`, validated f32/f16/bf16, head_dim=128. A pre-M3 bf16-tuned sibling `mt_sdpa_prefill_mma_bf16` (`steel_attention_mma_bf16.rs`) is selected by `sdpa_prefill_mma_for()` — a perf specialization, not a separate op. |
| steel_attention_nax | ✓ | ✓ | ✗ | Header-only stub + `nax` feature gate. |
| steel_gemm_fused | ✓ | ✓ | ~ | `mlx/steel/gemm/steel_gemm_fused.rs` → `mt_steel_gemm_64x64x16_2x2<T>`. One block-shape variant; upstream has many. |
| steel_gemm_fused_nax | ✓ | ✓ | ✗ | Blocker: `nax` feature gate. (Simdgroup-matrix primitive now exists — see `steel_attention_mma`.) |
| steel_gemm_gather | ✓ | ✓ | ✗ | Blocker: indirect (gather) indexing of the matmul operands. |
| steel_gemm_gather_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_gemm_masked | ✓ | ✓ | ✗ | Blocker: block-level predication. |
| steel_gemm_segmented | ✓ | ✓ | ✗ | Blocker: ragged batched matmul. |
| steel_gemm_splitk + accum | ✓ | ✓ | ✗ | Blocker: two-kernel split-K dispatch + accumulator pass. |
| steel_gemm_splitk_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_conv 2D (implicit-GEMM) | ✓ | ✓ | ✗ | Blocker: im2col primitives missing. |
| steel_conv 3D | ✓ | ✓ | ✗ | Same blocker + 3D `MLXConvParams<3>` indexing. |
| steel_conv_general (strides/dilation/groups) | ✓ | ✓ | ✗ | Same blockers as steel_conv. |
| conv (winograd + naive_unfold + depthwise) | ✓ | ✓ | ✗ | `crates/metaltile-std/src/mlx/conv.rs` is a stub left from the old bench crate, not declared in `mod.rs`. No DSL port. |
| gemv | ✓ | ✓ | ✓ | `mlx/gemv.rs` → `mt_gemv<T>`. |
| gemv_masked | ✓ | ✓ | ✓ | `mlx/gemv_masked.rs` → `mt_gemv_masked<T>` (no MLX comparison wired). |
| quantized (affine_quantize / affine_dequantize) | ✓ | ✓ | ~ | `mlx/quantized.rs` → quantize **and** dequantize for int4/int8, plus dequantize for int3/int5/int6 (`mt_affine_{quantize,dequantize}_int{3,4,5,6,8}`). Gap: int2, and the quantize side of int3/5/6. |
| quantized (affine_qmv / qvm / qmm — matvec / matmul) | ✓ | ✓ | ~ | `mlx/quantized.rs` → `mt_qmv` + `mt_qmm` / `mt_qmm_bm2` / `mt_qmm_bm4` (3 M-batch tiles) with an `mt_qmm_for` selector, all f32+f16, int4. Gap: `qvm` absent, bit-widths other than int4 absent, bf16 absent. |
| quantized (gather_qmv / gather_qmm — gather variants) | ✓ | ✓ | ✗ | Affine gather-qmm/qvm absent. Bare-tensor `ffai/gather.rs` exists but is non-quantized. |
| dequant_gather (quantized embedding-table gather) | ✗ | ✗ | ✓ | `ffai/dequant_gather.rs`. int{3,4,5,6,8} all bit-widths. FFAI-specific, no MLX counterpart. |
| dequant_gemv (quantized GEMV, FFAI flavour) | ~ (subset of `quantized.metal`) | ~ | ✓ | `ffai/dequant_gemv.rs`. int{3,4,5,6,8}, generic `T`. Coexists with the partial `mt_qmv_f32` port; FFAI-tuned shape. |
| fp_quantized (fp4/fp8 quant + dequant) | ✓ | ✓ | ~ | `mlx/fp_quantized.rs` → `mt_fp4_quant_dequant` (f32 only). fp8 path and other dtypes missing. |
| fp_quantized_nax | ✓ | ✓ | ✗ | Module file present but empty (no `#[kernel]` defs). NAX-gated. |
| quantized_nax | ✓ | ✓ | ✗ | Module file present but empty (no `#[kernel]` defs). NAX-gated. |
| fft (radix + readwrite) | ✓ | ✓ | ✗ | Stub file in repo, not declared. No DSL port. |
| hadamard (hadamard_n + hadamard_m) | ✓ | ✓ | ✗ | Not ported. Used by Walsh-Hadamard quant path; could matter for AURA rotations longer-term. |
| fence | ✓ | ✓ | ✗ | Stub file in repo, not declared. Synchronization primitive. |
| gather (bare-tensor embedding lookup) | ✓ (via indexing/) | ✓ | ✓ | `ffai/gather.rs` → `ffai_gather<T>`. FFAI's embedding-table gather. |
| indexing (scatter, scatter_axis, gather_axis, gather_front, masked_scatter) | ✓ | ✓ | ✗ | Header-only family in MLX; metaltile only covers bare gather today. scatter/scatter_axis/masked_scatter all absent. |
| aura_encode (codebook quantize, fused) | ✗ | ✓ (`turbo_fused_encode` in `turbo_quant.metal`) | ✓ | `ffai/aura_encode.rs`. Bit-widths 2/3/4/8. Renamed turbo_*→aura_*. |
| aura_dequant_rotated (bulk dequant to rotated codec space) | ✗ | ✓ (`turbo_dequant_rotated` in `turbo_quant.metal`) | ✓ | `ffai/aura_dequant_rotated.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_score (compressed-domain Q·K) | ✗ | ✓ (`turbo_score`) | ✓ | `ffai/aura_score.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_value (compressed-domain value aggregation) | ✗ | ✓ (`turbo_value` in `turbo_quant.metal`) | ✓ | `ffai/aura_value.rs`. Sparsity-threshold guard mirrors MLX upstream. Renamed. |
| aura_flash_p1 (compressed-domain flash pass 1) | ✗ | ✓ (`turbo_flash_p1` in `turbo_flash.metal`) | ~ | `ffai/aura_flash_p1.rs`. Only the `(kb=4, vb=2, dim=128)` aura4v2/Qwen3-128 instantiation today; causal-variant from upstream not ported. |
| aura_flash_pass2 (cross-block online-softmax merge) | ✗ | ✓ (`turbo_flash_pass2`) | ✓ | `ffai/aura_flash_pass2.rs`. fp32 accums → bf16 final. Renamed. |
| turbo_flash_sdpa (fused single-pass SDPA, sinks variant) | ✗ | ✓ (`turbo_flash_sdpa.metal`) | ✗ | NOT PORTED. Sinks-using models (spec 041 phase 1.1) — needed for GPT-OSS / sink-attention configs. |
| flash_quantized_sdpa (single-pass quantized SDPA, affine cache) | ✗ | ✓ (`flash_quantized_sdpa.metal`) | ✗ | NOT PORTED. bits ∈ {4,8}, head_dim ∈ {64,96,128,256,512}. Direct competitor to `sdpa_decode_2pass` over affine-quant KV caches. |
| gated_delta (GatedDeltaNet recurrence) | ✗ | ✓ (`gated_delta.metal`) | ✗ | NOT PORTED. Required for GDN-bearing models (Qwen 3.5 / 3.6 hybrid). Two variants in upstream: standard + fused. |
| gated_delta_replay (tape capture + state replay) | ✗ | ✓ (`gated_delta_replay.metal`) | ✗ | NOT PORTED. Spec 020 phase 2 — speculative decoding rollback on GDN. |
| ssm_step (Mamba 2 SSD single-token decode) | ✗ | ✓ (`ssm.metal`) | ✓ | `ffai/ssm.rs` → `ssm_step<T>`, `mt_ssm_step<T>`. Faithful port; `mlx_src: None` because pinned MLX upstream doesn't ship `ssm.metal`. Will graduate to `mlx/` when pin moves. |
| conv1d_causal_step (depthwise SSM conv stream) | ✗ | partial (subset of SSM toolchain) | ✓ | `ffai/ssm.rs` → `conv1d_causal_step<T>`. fp32 state recurrence. |
| ssm_replay (sequential tape capture + replay) | ✗ | ✓ (`ssm_replay.metal`) | ✗ | NOT PORTED. Spec 040 — Mamba/Mamba2 state replay for speculative decoding. |
| fused_gate_activation (silu/gelu × up gate) | ✗ | ✓ (`fused_gate_activation.metal`) | ✗ | NOT PORTED. Single-row + looped variants; replaces split+act+mul (≥2 dispatches → 1). Hot path in every FFN. |
| rms_norm_residual (RMSNorm + residual add fused) | ✗ | ✓ (`rms_norm_residual.metal`) | ✗ | NOT PORTED. ~90 saved dispatches/token on Gemma4-30 type configs. |
| rms_norm_rope (RMSNorm + RoPE fused) | ✗ | ✓ (`rms_norm_rope.metal`) | ✗ | NOT PORTED. Q/K post-projection norm+rope in one dispatch. |
| rms_norm_qgemv (RMSNorm + 4-bit quantized GEMV fused) | ✗ | ✓ (`rms_norm_qgemv.metal`) | ✗ | NOT PORTED. Eliminates global RT between norm and qmatmul. |
| batched_qkv_qgemv (Q/K/V 4-bit qGEMV → 1 dispatch) | ✗ | ✓ (`batched_qkv_qgemv.metal`) | ✗ | NOT PORTED. Decode-form fused QKV projection over int4 weights. |
| kv_cache_update (raw bf16/fp16 single-token append) | ✗ | ✗ | ✓ | `ffai/kv_cache.rs` → `kv_cache_update<T>`. FFAI-only; raw cache append. |
| kv_cache (affine-quant int4/int8 quantize + bulk dequant) | ~ (via `quantized.metal` affine_quantize) | ~ | ✓ | `ffai/kv_cache.rs` — `quantize_kv` + `bulk_dequant_kv` for int4/int8. FFAI-specific cache layout. |
| sampling (softmax + categorical inverse-CDF) | ✗ | ✗ | ✓ | `ffai/sampling.rs` → `softmax_categorical_sample`. Companion to `ffai_argmax` for `T > 0` decode. |

## Notes on counting decisions

A few rows mix multiple `.metal` files into one op or split one file into multiple ops:

- **`sdpa_vector*` rows.** Upstream `sdpa_vector.h` defines `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`. Counted as two ops: `sdpa_vector` (single pass) + `sdpa_vector_2pass` (two-pass pair).
- **AURA stack.** Each codec stage (`encode`, `dequant_rotated`, `score`, `value`, `flash_p1`, `flash_pass2`) is a separate row — they're separately compiled kernels with their own dispatch shapes. The unported `turbo_flash_sdpa` (sinks-fused single-pass) is also its own row.
- **`steel/` family.** Each kernel file in `steel/{attn,conv,gemm}/kernels/` becomes one op row; per-block-shape instantiations are not counted separately. `steel_attention` (scalar-flash) and `steel_attention_mma` (simdgroup-MMA) are counted as two rows because they are separately compiled kernels with different lowering strategies; the bf16-tuned `mt_sdpa_prefill_mma_bf16` is folded into the MMA row as a perf specialization.
- **`quantized.metal`.** Split into three rows by semantic operation (quant/dequant, qmv/qvm/qmm matmul, gather-qmv/qmm) rather than by template instantiation. Quantized-NAX and FP-quantized-NAX are separate rows because the metaltile modules exist (empty) and have separate feature gates.
- **`indexing/`** is one row covering scatter / scatter_axis / gather_axis / gather_front / masked_scatter. Bare `gather` is its own row because metaltile has a dedicated FFAI port.
- **Cells marked `~`** indicate metaltile has a partial port — typically one bit-width, one dtype, or one block shape where upstream has many. Read the notes column for the specific gap.

## Highest-value un-ported ops (next-up recommendations)

Roughly ordered by FFAI-impact × tractability. The fused-norm/-act/-qgemv family is the biggest collective win — each saves a per-layer dispatch on decode, and they're all DSL-tractable (no new simdgroup-matrix primitive needed). Note the DSL now demonstrably *has* a simdgroup-matrix MMA path (`steel_attention_mma`, and the `probe/mma_layout_probe.rs` layout probe), so the remaining `steel_gemm_*` / `steel_conv*` rows are no longer blocked on the primitive itself — only on the gather / masked / split-K / im2col logic layered on top.

1. **`flash_quantized_sdpa`** — direct upgrade path over `sdpa_decode_2pass` for affine-quant KV caches. Covers head_dim {64, 96, 128, 256, 512} and bits {4, 8}. Biggest single-decode latency win for production FFAI configs.
2. **`turbo_flash_sdpa` (→ `aura_flash_sdpa`)** — fused single-pass AURA SDPA with sinks support. Needed for GPT-OSS sink-attention; closes the last AURA gap.
3. **`gated_delta` + `gated_delta_replay`** — blocker for Qwen 3.5 / 3.6 hybrid GDN+attn models, including speculative-decode rollback. Two kernels, well-specified upstream.
4. **`ssm_replay`** — completes the Mamba/Mamba2 speculative-decode story. SSM step is already ported; this is the tape-capture/replay companion.
5. **`rms_norm_residual` + `rms_norm_rope` + `rms_norm_qgemv`** — three fused norm kernels. Saves ~90 dispatches/token on Gemma4-class configs; rms_norm_qgemv eliminates a global memory round-trip.
6. **`fused_gate_activation`** — silu/gelu × up-gate in one dispatch. Hot path in every FFN; trivial to port (elementwise).
7. **`batched_qkv_qgemv`** — fuses Q/K/V int4 projections into one dispatch. Decode hot path.
8. **`steel_gemm_fused` shape coverage** — only `64×64×16` is wired today; prefill perf needs more block shapes. Unblocks longer-context prefill paths even before simdgroup-matrix lands more broadly.
9. **`hadamard`** — Walsh-Hadamard rotation. Relevant if AURA's rotation matrix is ever swapped for the orthonormal-Hadamard variant.
10. **`indexing` (scatter, scatter_axis, masked_scatter)** — missing for any cache update path that isn't a simple append (e.g. sliding-window evict, prefix-cache splice, batched scatter).

## Open uncertainties / counting caveats

- `quantized_nax.rs` and `fp_quantized_nax.rs` were re-checked at `141a60b`: both are still empty (TODO comment only, zero `#[kernel]`) and both are `#[cfg(feature = "nax")]`-gated in `mlx/mod.rs`. Counted as `✗` for metaltile.
- `mlx/strided.rs` (`mt_strided_copy`) covers strided copy but I didn't audit which stride dimensionalities — marked `~` defensively. Upstream `copy.metal` has multiple `copy_g_nd*` shapes.
- `ffai/sdpa_decode.rs` is FFAI-specific (`✗ / ✗ / ✓`) — it's not a port of an upstream MLX kernel; it's a derivative of `mt_sdpa_vector` with a decoupled `kv_stride` parameter for pre-allocated caches. Worth raising whether this should live in `mlx/` once we propose decoupled-stride upstream.
- `ffai/aura_flash_p1.rs` is marked `~` (partial) because only the `(kb=4, vb=2, dim=128)` instantiation is registered; the causal variant from `turbo_flash.metal` and other (kb, vb, dim) combos aren't ported yet.
- Coverage % treats the alpha-only kernels as in-scope (we maintain the fork, so they count toward the union). If you want the upstream-only metric, that's 21 / 41 = 51 %.

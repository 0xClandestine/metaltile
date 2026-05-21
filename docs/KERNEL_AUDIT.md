# metaltile kernel-op coverage audit

Generated: 2026-05-18 · Refreshed: 2026-05-21 (consolidation pass +
Vision / STT / TTS front-end kernels)
Sources surveyed:
- MLX upstream `ml-explore/mlx@main` (commit `2414e5df`)
- MLX fork `ekryski/mlx@alpha` (commit `4919270e`)
- metaltile `thewafflehaus/metaltile:ek/aura-port` (the consolidated branch —
  `origin/dev` plus the Gemma / Nemotron-H / GPT-OSS-20B kernel work)

## Summary

- Total kernel-op rows in this audit (union): **89**
- metaltile-ported kernel ops: **65 / 89 = 73 %** — 54 full ✓ (61 %), 11 partial ~ (12 %)
- **Still to cover: 24 ops not ported (✗)**, plus **11 partial ports** still to finish
- The 6 Vision / STT / TTS front-end kernels (Phase 6.5 / 7) — `conv2d`,
  `patch_embed`, `rope_2d`, `mel_spectrogram`, `audio_conv1d`,
  `vocoder/iSTFT` — are now ported (✓ rows below).
- 3 in-flight kernel families have an **open PR** (not yet landed) — see
  [Kernels with open PRs](#kernels-with-open-prs).

> **Note on the 2026-05-21 consolidation pass.** The Gemma / Nemotron-H /
> GPT-OSS-20B kernel work, previously spread across separate worktrees, is now
> consolidated onto `ek/aura-port`. Two Gemma kernels — `sdpa_decode_d512` and
> `rms_norm_wide` — are added as ✓ rows. A model-side review of FFAI's decode
> path also surfaced several **host-side compute fallbacks** that exist only
> because a GPU kernel is missing; the kernels that would close them
> (`gated_rmsnorm`, the `sdpa_decode` learned-sink term, a 2D-`A_log` `ssm_step`
> variant) are recorded below, and the still-needed **Vision / STT / TTS**
> front-end kernels (`conv2d`, `patch_embed`, `rope_2d`, `mel_spectrogram`,
> `audio_conv1d`, `vocoder/iSTFT`) are added as ✗ rows for Phase 6.5 / 7.
> The MLX-upstream and MLX-alpha columns were **not** re-verified against those
> repos (not checked out) — only the metaltile column was re-surveyed.

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
| swiglu (`silu(gate)·up` fused MLP activation) | ✗ | ✗ | ✓ | `mlx/swiglu.rs` → `mt_swiglu<T>`. Fused element-wise `silu(gate) * up` — the standard modern-transformer MLP activation (Llama 4, Qwen3 dense + MoE, Gemma, Mistral). metaltile fuses what MLX expresses as separate `silu` + `mul` ops; no dedicated MLX kernel. The broader `fused_gate_activation` (gelu / clipped-swiglu variants) is still a separate ✗ row below. |
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
| sdpa_decode_batched (speculative-decode batched-Q decode) | ✗ | ✗ | ✓ | `ffai/sdpa_decode_batched.rs` → `sdpa_decode_batched_q{2,4}<T>` (+ `sdpa_decode_batched_prefill.rs`). K query positions share one KV walk per dispatch (M7 speculative decoding), amortizing KV memory bandwidth K× vs. K independent single-Q `sdpa_decode` dispatches. FFAI-only. |
| steel_attention (Flash, prefill) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention.rs` → `mt_sdpa_prefill<T>`. Scalar-flash prefill (BQ=4, online softmax, causal), generic `T`, head_dim=128. The old "`Op::FlashAttention` lowers to an error placeholder" blocker is resolved. |
| steel_attention_mma (Flash prefill, simdgroup-MMA) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_mma.rs` → `mt_sdpa_prefill_mma<T>`. Real simdgroup-matrix MMA path; generic `T`, validated f32/f16/bf16, head_dim=128. A pre-M3 bf16-tuned sibling `mt_sdpa_prefill_mma_bf16` (`steel_attention_mma_bf16.rs`) is selected by `sdpa_prefill_mma_for()` — a perf specialization, not a separate op. |
| steel_attention_nax | ✓ | ✓ | ✗ | Header-only stub + `nax` feature gate. |
| steel_gemm_fused | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused.rs` → `mt_steel_gemm_{64x64x16_2x2,32x32x16_2x2,64x64x16_1x2,32x64x16_1x2}<T>`. Plain row-major `C = A·B` via Apple 8×8 simdgroup-matrix MMA; four block-shape instantiations (each mirrors an MLX `instantiate_gemm_shapes_helper` shape). Fixed a transposed-B fragment-load bug in the original `64×64×16_2x2` kernel (it loaded `B` with the `(fn,fm)` GEMM-transposed lane convention, shipping `Bᵀ`-shaped output) plus a missing K-accumulation loop (only summed K∈[0,16)). Verified by `steel_gemm_gpu_correctness` (all four transpose modes, f32/f16/bf16). |
| steel_gemm_fused_nax | ✓ | ✓ | ✗ | Blocker: `nax` feature gate. (Simdgroup-matrix primitive now exists — see `steel_attention_mma`.) |
| steel_gemm_gather | ✓ | ✓ | ✗ | Blocker: indirect (gather) indexing of the matmul operands. |
| steel_gemm_gather_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_gemm_masked | ✓ | ✓ | ✗ | Blocker: block-level predication. |
| steel_gemm_segmented | ✓ | ✓ | ✗ | Blocker: ragged batched matmul. |
| steel_gemm_splitk + accum | ✓ | ✓ | ✗ | Blocker: two-kernel split-K dispatch + accumulator pass. |
| steel_gemm_splitk_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_conv 2D (implicit-GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` / `conv2d_generic`. 2D convolution as a direct conv (implicit im2col, one thread per output) rather than MLX's explicit-im2col tiled GEMM — equivalent result, no im2col staging buffer. Covers fixed-patch and runtime-stride/pad configs. The MMA-tiled implicit-GEMM is a perf follow-up. Verified by `conv2d_gpu_correctness`. |
| steel_conv 3D | ✓ | ✓ | ✓ | `ffai/conv3d.rs` → `conv3d_generic` (strided / padded dense 3D conv) + `conv3d_grouped` (adds dilation + grouped channels; `groups == in_ch` is depthwise). 5D NCDHW input, OIDHW weight — the volumetric counterpart of `conv2d.rs`: direct conv (implicit im2col), one thread per output voxel, fp32 accumulation, padding taps masked in the padded-input frame. Generic `T` (f32/f16/bf16). The MMA-tiled implicit-GEMM is a perf follow-up. Verified by `conv3d_gpu_correctness`. |
| steel_conv_general (strides/dilation/groups) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_grouped<T>`. Fully general 2D conv: strides, dilation (atrous), padding, and grouped channels (`groups == in_ch` is depthwise). NCHW input, OIHW weight with the I dimension = `in_ch/groups`. Direct conv, one thread per output, fp32 accumulation. Verified by `conv2d_gpu_correctness`. |
| conv (winograd + naive_unfold + depthwise) | ✓ | ✓ | ~ | The `naive_unfold` + depthwise cases are covered for **both 2D and 3D** — `ffai/conv2d.rs` (`conv2d_generic` + `conv2d_grouped`) and `ffai/conv3d.rs` (`conv3d_generic` + `conv3d_grouped`); the `_grouped` kernels handle depthwise via `groups == in_ch` and dilation (atrous). The Winograd fast-conv path is not ported (a perf-only specialization for 3×3 stride-1 convs). The old `mlx/conv.rs` bench-crate stub is superseded. |
| gemv | ✓ | ✓ | ✓ | `mlx/gemv.rs` → `mt_gemv<T>`. |
| gemv_masked | ✓ | ✓ | ✓ | `mlx/gemv_masked.rs` → `mt_gemv_masked<T>` (no MLX comparison wired). |
| quantized (affine_quantize / affine_dequantize) | ✓ | ✓ | ~ | `mlx/quantized.rs` → quantize **and** dequantize for int4/int8, plus dequantize for int3/int5/int6 (`mt_affine_{quantize,dequantize}_int{3,4,5,6,8}`). Gap: int2, and the quantize side of int3/5/6. |
| quantized (affine_qmv / qvm / qmm — matvec / matmul) | ✓ | ✓ | ~ | `mlx/quantized.rs` → `mt_qmv` + `mt_qmm` / `mt_qmm_bm2` / `mt_qmm_bm4` (3 M-batch tiles) with an `mt_qmm_for` selector, all f32+f16, int4. Gap: `qvm` absent, bit-widths other than int4 absent, bf16 absent. An int4 qmm via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores, MLX-parity) is **open in PR [#137](https://github.com/0xClandestine/metaltile/pull/137)** (`mt_qmm_mma_mpp`). |
| quantized (gather_qmv / gather_qmm — gather variants) | ✓ | ✓ | ✗ | Affine gather-qmm/qvm absent. Bare-tensor `ffai/gather.rs` exists but is non-quantized. The MoE grouped-gather quantized matmul is **open in PR [#125](https://github.com/0xClandestine/metaltile/pull/125) / [#136](https://github.com/0xClandestine/metaltile/pull/136)** (`mt_moe_gather_qmm_int4` + MMA / MPP-NAX variants). |
| moe (router top-k + permute + unpermute orchestration) | ✗ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_router_topk<T>`, `mt_moe_permute<T>`, `mt_moe_unpermute<T>`. MoE expert-routing orchestration for Qwen3.6-35B-A3B / Qwen3-Coder-30B-A3B end-to-end serving. The grouped quantized BGEMM that fuses the per-expert FFN matmuls into one dispatch is **open in PR [#125](https://github.com/0xClandestine/metaltile/pull/125) / [#136](https://github.com/0xClandestine/metaltile/pull/136)**. |
| dequant_gather (quantized embedding-table gather) | ✗ | ✗ | ✓ | `ffai/dequant_gather.rs`. int{3,4,5,6,8} all bit-widths. FFAI-specific, no MLX counterpart. |
| dequant_gemv (quantized GEMV, FFAI flavour) | ~ (subset of `quantized.metal`) | ~ | ✓ | `ffai/dequant_gemv.rs`. int{3,4,5,6,8}, generic `T`. Coexists with the partial `mt_qmv_f32` port; FFAI-tuned shape. |
| fp_quantized (fp4/fp8 quant + dequant) | ✓ | ✓ | ~ | `mlx/fp_quantized.rs` → `mt_fp4_quant_dequant` (f32 only). fp8 path and other dtypes missing. |
| fp_quantized_nax | ✓ | ✓ | ✗ | Module file present but empty (no `#[kernel]` defs). NAX-gated. |
| quantized_nax | ✓ | ✓ | ✗ | Module file present but empty (no `#[kernel]` defs). NAX-gated. |
| fft (radix + readwrite) | ✓ | ✓ | ✗ | Stub file in repo, not declared. No DSL port. |
| hadamard (hadamard_n + hadamard_m) | ✓ | ✓ | ~ | `mlx/hadamard.rs` → `mt_hadamard_n{64,128,256,512,1024}<T>`. Power-of-2 FWHT via log2(N) butterfly passes. The non-power-of-2 `hadamard_m` factor (M ∈ {12,20,28}) is a follow-up. |
| fence | ✓ | ✓ | ✗ | Stub file in repo, not declared. Synchronization primitive. |
| gather (bare-tensor embedding lookup) | ✓ (via indexing/) | ✓ | ✓ | `ffai/gather.rs` → `ffai_gather<T>`. FFAI's embedding-table gather. |
| indexing (scatter, scatter_axis, gather_axis, gather_front, masked_scatter) | ✓ | ✓ | ✓ | `mlx/gather_axis.rs` + `mlx/scatter_axis.rs` → `mt_gather_axis` / `mt_scatter_axis` (contiguous along-axis); `mlx/indexing.rs` → `mt_gather_front` (first-axis row gather), `mt_scatter` (first-axis row scatter, no-reduce assignment form), `mt_masked_scatter` (per-element masked gather-scatter). All five are one-thread-per-output Grid3D with an `n_elems` bounds guard. Verified by `gather_axis_gpu_correctness` / `scatter_axis_gpu_correctness` / `indexing_gpu_correctness`. |
| aura_encode (codebook quantize, fused) | ✗ | ✓ (`turbo_fused_encode` in `turbo_quant.metal`) | ✓ | `ffai/aura_encode.rs`. Bit-widths 2/3/4/8. Renamed turbo_*→aura_*. |
| aura_dequant_rotated (bulk dequant to rotated codec space) | ✗ | ✓ (`turbo_dequant_rotated` in `turbo_quant.metal`) | ✓ | `ffai/aura_dequant_rotated.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_score (compressed-domain Q·K) | ✗ | ✓ (`turbo_score`) | ✓ | `ffai/aura_score.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_value (compressed-domain value aggregation) | ✗ | ✓ (`turbo_value` in `turbo_quant.metal`) | ✓ | `ffai/aura_value.rs`. Sparsity-threshold guard mirrors MLX upstream. Renamed. |
| aura_flash_p1 (compressed-domain flash pass 1) | ✗ | ✓ (`turbo_flash_p1` in `turbo_flash.metal`) | ~ | `ffai/aura_flash_p1.rs`. Only the `(kb=4, vb=2, dim=128)` aura4v2/Qwen3-128 instantiation today; causal-variant from upstream not ported. |
| aura_flash_pass2 (cross-block online-softmax merge) | ✗ | ✓ (`turbo_flash_pass2`) | ✓ | `ffai/aura_flash_pass2.rs`. fp32 accums → bf16 final. Renamed. |
| turbo_flash_sdpa (fused single-pass SDPA, sinks variant) | ✗ | ✓ (`turbo_flash_sdpa.metal`) | ✓ | `ffai/aura_flash_sdpa.rs` → `aura_flash_sdpa_kb*_vb*_d*<T>`. Single-pass online-softmax over compressed K/V with attention sinks + sliding-window causal mask. Single-simdgroup shape (token-parallelism a perf follow-up). |
| flash_quantized_sdpa (single-pass quantized SDPA, affine cache) | ✗ | ✓ (`flash_quantized_sdpa.metal`) | ✓ | `ffai/flash_quantized_sdpa.rs` → `flash_quantized_sdpa_b{4,8}_d{64,128,256}<T>`. Single-pass online-softmax SDPA over affine-quant KV, with sinks + sliding-window. head_dim {96,512} and bool/float masks are a follow-up. |
| gated_delta (GatedDeltaNet recurrence) | ✗ | ✓ (`gated_delta.metal`) | ✓ | `ffai/gated_delta.rs` → `mt_gated_delta_step<T>` (single-token decode) + `mt_gated_delta_chunk<T>` (chunked-prefill). GDN linear-attention for the Qwen3.5 / 3.6 / 3.6-MoE hybrid models (≈75 % of layers). The MMA-tiled chunked-WY prefill perf variant (`mt_gated_delta_wy_chunk`) is **open in PR [#115](https://github.com/0xClandestine/metaltile/pull/115)**. |
| gated_delta_replay (tape capture + state replay) | ✗ | ✓ (`gated_delta_replay.metal`) | ✓ | `ffai/gated_delta_replay.rs` → `gated_delta_step_record<T>` (forward + delta-tape) + `state_replay<T>` (branchless accepted-prefix re-fold). Speculative-decode rollback on GDN. |
| ssm_step (Mamba 2 SSD single-token decode) | ✗ | ✓ (`ssm.metal`) | ✓ | `ffai/ssm.rs` → `ssm_step<T>`, `mt_ssm_step<T>`. Faithful port; `mlx_src: None` because pinned MLX upstream doesn't ship `ssm.metal`. Will graduate to `mlx/` when pin moves. |
| conv1d_causal_step (depthwise SSM conv stream) | ✗ | partial (subset of SSM toolchain) | ✓ | `ffai/ssm.rs` → `conv1d_causal_step<T>`. fp32 state recurrence. |
| ssm_replay (sequential tape capture + replay) | ✗ | ✓ (`ssm_replay.metal`) | ✓ | `ffai/ssm_replay.rs` → `ssm_step_record<T>` (SSD forward + dA/dBx tape) + `ssm_replay<T>` (re-fold first k entries). Spec 040 Mamba/Mamba2 state replay. |
| fused_gate_activation (silu/gelu × up gate) | ✗ | ✓ (`fused_gate_activation.metal`) | ✓ | `mlx/fused_gate_activation.rs` → `mt_fused_gate_gelu` (gelu-tanh approximation) + `mt_fused_gate_clipped_swiglu` (GPT-OSS clipped variant — `[-7,7]` clamp, `sigmoid(1.702·g)` gate, `+1` up bias). The `silu` variant ships separately as `mlx/swiglu.rs` (see the `swiglu` row). One-thread-per-output Grid3D; the MLX `single_row` / `looped` threadgroup-tiling split is a perf detail, not a separate op. Verified by `fused_gate_activation_gpu_correctness`. |
| rms_norm_residual (RMSNorm + residual add fused) | ✗ | ✓ (`rms_norm_residual.metal`) | ✓ | `ffai/rms_norm_residual.rs` → `ffai_rms_norm_residual<T>`. Reduction-mode, `N = TPG*4`; mirrors `mt_rms_norm` + a residual-add input. ~90 saved dispatches/token on Gemma4-30 type configs. |
| rms_norm_rope (RMSNorm + RoPE fused) | ✗ | ✓ (`rms_norm_rope.metal`) | ✓ | `ffai/rms_norm_rope.rs` → `ffai_rms_norm_rope<T>`. Reduction-mode, paired-layout RoPE; `TPG = axis_size/2`. Q/K post-projection norm+rope in one dispatch. |
| rms_norm_qgemv (RMSNorm + 4-bit quantized GEMV fused) | ✗ | ✓ (`rms_norm_qgemv.metal`) | ✓ | `ffai/rms_norm_qgemv.rs` → `ffai_rms_norm_qgemv<T>`. Reduction-mode, int4, one row/threadgroup; eliminates the global RT of the normalized activation. MLX's 8-row-per-TG tiling is a perf follow-up. |
| batched_qkv_qgemv (Q/K/V 4-bit qGEMV → 1 dispatch) | ✗ | ✓ (`batched_qkv_qgemv.metal`) | ✓ | `ffai/batched_qkv_qgemv.rs` → `ffai_batched_qkv_qgemv<T>`. Reduction-mode, int4; `program_id::<2>()` selects Q/K/V, output concatenated `[Q\|K\|V]`. Decode-form fused QKV projection. |
| kv_cache_update (raw bf16/fp16 single-token append) | ✗ | ✗ | ✓ | `ffai/kv_cache.rs` → `kv_cache_update<T>`. FFAI-only; raw cache append. |
| kv_cache (affine-quant int4/int8 quantize + bulk dequant) | ~ (via `quantized.metal` affine_quantize) | ~ | ✓ | `ffai/kv_cache.rs` — `quantize_kv` + `bulk_dequant_kv` for int4/int8. FFAI-specific cache layout. |
| sampling (softmax + categorical inverse-CDF) | ✗ | ✗ | ✓ | `ffai/sampling.rs` → `softmax_categorical_sample`. Companion to `ffai_argmax` for `T > 0` decode. |
| logits processors (temperature, repetition penalty, top-k / top-p / min-p masks) | ✗ | ✗ | ✓ | `ffai/logits_{processors,topk,top_p,min_p}.rs` → `logits_temperature`, `logits_repetition_penalty`, `logits_topk_mask`, `logits_top_p_mask`, `logits_min_p_mask` (all generic `T`). In-place decode-form sampler stages composed before `softmax_categorical_sample`. FFAI-only. |
| sdpa_decode_d512 (head_dim=512 SDPA decode — Gemma 4 global) | ✗ | ✗ | ✓ | `ffai/sdpa_decode_d512.rs` → `ffai_sdpa_decode_d512<T>`. head_dim=512 specialization for Gemma 4's global-attention layers; dispatches at 512 threads/TG (the 16-wide per-lane footprint caps the pipeline below 1024). FFAI-only; verified by `sdpa_decode_d512_gpu_correctness`. Consolidation pass (2026-05-21). |
| rms_norm_wide (RMSNorm for rows past the 4096-element cap) | ✗ | ✗ | ✓ | `mlx/rms_norm.rs` → `mt_rms_norm_wide<T>`. Strided wide-row variant for large-hidden models (Gemma 4 31B, hidden 5376) that exceed the standard `mt_rms_norm` 1024-thread × 4-element single-row cap. Verified by `rms_norm_wide_gpu_correctness`. Consolidation pass (2026-05-21). |
| sdpa_decode + learned attention sink (GPT-OSS-20B) | ✗ | ~ | ~ | **Partial — host-side fallback.** GPT-OSS-20B's per-head learned `sinks` logit is folded into the softmax denominator as a host-side post-hoc rescale of the raw-KV `sdpa_decode` (d64) output — the `sdpa_decode` kernel has a sink-*token* path (`sink_end`) but no learned per-head sink-*logit* term. `aura_flash_sdpa` / `flash_quantized_sdpa` carry sinks for the *quantized* cache only. A `sink_logit` constexpr on `sdpa_decode` would remove the GPT-OSS per-layer CPU sync. |
| gated_rmsnorm (fp32-in gated RMSNorm → activation dtype) | ✗ | ✗ | ✓ | `ffai/gated_rmsnorm.rs` → `ffai_gated_rmsnorm<T>`. Fused Qwen3.5 / 3.6 GDN post-step `out = w·rmsNorm(y)·silu(z)`: `y` arrives fp32 (the `gated_delta` recurrence output), the gate `z` / weight `w` / output are activation-dtype `T`. Reduction-mode, `N = TPG*4`, mirrors `mt_rms_norm` with the fp32-in / `T`-out dtype split and the `silu(z)` gate. Closes the per-GDN-layer host-side CPU sync (≈75 % of Qwen3.5/3.6 layers). Verified by `gated_rmsnorm_gpu_correctness`. |
| ssm_step (2D `A_log` / per-(head,state) decay — Jamba) | ✗ | ~ | ✗ | **NOT PORTED — host-side fallback.** The shipped `ssm_step` (row above) bakes in a per-channel scalar `A`; Jamba's `A_log` is 2D (per-(head,state)), so Jamba runs its entire Mamba 2 selective scan + dt/B/C RMSNorms host-side. A 2D-`A` `ssm_step` variant moves the Jamba scan to the GPU. The other Mamba 2 families (Mamba2, FalconH1, NemotronH, GraniteMoeHybrid) use the scalar-`A` kernel and are unaffected. |
| conv2d (vision patch conv — im2col + tiled GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` (fixed-patch variants, kernel + stride baked in) + `conv2d_generic` (runtime kh/kw/stride/pad). NCHW input, OIHW weight; direct conv (implicit im2col, one thread per output). Generic `T`; verified by `conv2d_gpu_correctness`. Phase 6.5 VLM. |
| patch_embed (fused image unfold + linear projection) | ✗ | ✗ | ✓ | `ffai/patch_embed.rs` → `patch_embed<T>`. Fused image-unfold + linear projection — gathers each patch's pixels and dots them with one weight row, no intermediate unfolded buffer. NCHW image, flat `[hidden, patch_dim]` weight, `[num_patches, hidden]` output. FFAI-specific; verified by `patch_embed_gpu_correctness`. Phase 6.5 VLM. |
| rope_2d (2D positional RoPE for vision tokens) | ✓ | ✓ | ✓ | `ffai/rope_2d.rs` → `ffai_rope_2d<T>`. 2D RoPE over a (row, col) token grid — head_dim split into a row half and a column half, each running rotate-half RoPE. Consumes a per-token `(row, col)` pair. Generic `T`; verified by `rope_2d_gpu_correctness`. Phase 6.5 VLM. |
| mel_spectrogram (STFT + log-Mel filterbank) | ✓ | ✓ | ✓ | `ffai/mel_spectrogram.rs` → `mel_spectrogram<T>`. Fused STFT + Mel filterbank + log; one thread per (frame, mel_bin), direct DFT (fp32/fp16). A radix-FFT path is a perf follow-up (needs complex-type codegen). Verified by `mel_spectrogram_gpu_correctness`. Phase 7. |
| audio_conv1d (wide-stride 1D conv — STT patch embed) | ✓ | ✓ | ✓ | `ffai/audio_conv1d.rs` → `audio_conv1d<T>`. Dense wide-stride multi-channel 1D conv (NCL); distinct from the depthwise `conv1d_causal_step` SSM-stream conv. Generic `T`; verified by `audio_conv1d_gpu_correctness`. Phase 7. |
| vocoder / iSTFT (TTS waveform synthesis) | ✓ | ✓ | ✓ | `ffai/vocoder.rs` → `vocoder_istft<T>`. Inverse-STFT overlap-add — one thread per output sample gathers every covering frame, inverse-DFTs with Hermitian symmetry, COLA-normalises (no atomics). Generic `T`; verified by `vocoder_gpu_correctness`. Phase 7. |

## Kernels with open PRs

These are tracked above with an inline link in the Notes column; collected here
for quick scanning. Status reflects the open PRs as of 2026-05-21.

| PR | Kernel(s) | Affects row | State |
|---|---|---|---|
| [#115](https://github.com/0xClandestine/metaltile/pull/115) | `mt_gated_delta_wy_chunk` — chunked-WY GDN prefill (scalar foundation) | `gated_delta` | Draft / WIP; CI green, needs rebase onto current `dev`. |
| [#125](https://github.com/0xClandestine/metaltile/pull/125) | `mt_moe_gather_qmm_int4` — grouped MoE quantized matmul (m1 scalar) | `quantized (gather_*)`, `moe` | Draft; fmt/clippy/commit-hygiene red. Overlaps #136. |
| [#136](https://github.com/0xClandestine/metaltile/pull/136) | MoE gather BGEMM stack (m8 / MMA / MPP-NAX bm16 + bm64) | `quantized (gather_*)`, `moe` | Draft / WIP — surfaced for visibility; currently regresses vs MLX. |
| [#137](https://github.com/0xClandestine/metaltile/pull/137) | `mt_qmm_mma_mpp` + `mt_mpp_matmul_smoke` — int4 qmm via Apple `mpp::tensor_ops::matmul2d` | `quantized (qmm)` | Draft; MLX-parity, needs rebase + CI. |

## Notes on counting decisions

A few rows mix multiple `.metal` files into one op or split one file into multiple ops:

- **`sdpa_vector*` rows.** Upstream `sdpa_vector.h` defines `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`. Counted as two ops: `sdpa_vector` (single pass) + `sdpa_vector_2pass` (two-pass pair).
- **AURA stack.** Each codec stage (`encode`, `dequant_rotated`, `score`, `value`, `flash_p1`, `flash_pass2`) is a separate row — they're separately compiled kernels with their own dispatch shapes. The `turbo_flash_sdpa` (sinks-fused single-pass) is also its own row.
- **`steel/` family.** Each kernel file in `steel/{attn,conv,gemm}/kernels/` becomes one op row; per-block-shape instantiations are not counted separately. `steel_attention` (scalar-flash) and `steel_attention_mma` (simdgroup-MMA) are counted as two rows because they are separately compiled kernels with different lowering strategies; the bf16-tuned `mt_sdpa_prefill_mma_bf16` is folded into the MMA row as a perf specialization.
- **`quantized.metal`.** Split into three rows by semantic operation (quant/dequant, qmv/qvm/qmm matmul, gather-qmv/qmm) rather than by template instantiation. Quantized-NAX and FP-quantized-NAX are separate rows because the metaltile modules exist (empty) and have separate feature gates.
- **`indexing/`** is one row covering scatter / scatter_axis / gather_axis / gather_front / masked_scatter. Bare `gather` is its own row because metaltile has a dedicated FFAI port.
- **`moe`** is one row for the routing/permute/unpermute orchestration kernels in `ffai/moe.rs`. The grouped quantized BGEMM that the open PRs add is counted under the `quantized (gather_*)` row.
- **`logits processors`** is one row for the FFAI sampler-stage kernels (`temperature`, `repetition_penalty`, `topk` / `top_p` / `min_p` masks). FFAI-only, no MLX counterpart.
- **Cells marked `~`** indicate metaltile has a partial port — typically one bit-width, one dtype, or one block shape where upstream has many. Read the notes column for the specific gap.

## Highest-value un-ported ops (next-up recommendations)

Roughly ordered by FFAI-impact × tractability. The fused-norm/-act family is
largely landed now (`rms_norm_residual` / `_rope` / `_qgemv`,
`batched_qkv_qgemv`, `aura_flash_sdpa`, `flash_quantized_sdpa`, `gated_delta`,
`ssm_replay` all ✓). The DSL has a working simdgroup-matrix MMA path
(`steel_attention_mma`, the `probe/mma_layout_probe.rs` layout probe), so the
remaining `steel_gemm_*` rows are no longer blocked on the primitive itself —
only on the gather / masked / split-K logic layered on top. The `steel_conv`
family (2D, general, 3D) is now fully ported as direct convs (`ffai/conv2d.rs`,
`ffai/conv3d.rs`).

1. **`quantized` gather_qmm / gather_qmv** — the affine grouped-gather matmul.
   In flight in PRs #125 / #136; landing it closes the MoE FFN dispatch-count
   win (one kernel for the whole expert projection).
2. **`steel_gemm_fused` shape coverage** — only `64×64×16` is wired today;
   prefill perf needs more block shapes.
3. **`steel_gemm_splitk` + accum** — two-kernel split-K dispatch + accumulator
   pass. Infra-gated (split-K scheduling primitive).
4. **`steel_gemm_masked`** — block-level predication. Infra-gated.
5. **NAX feature family** — `steel_attention_nax`, `steel_gemm_*_nax`,
   `quantized_nax`, `fp_quantized_nax`. PR #137 demonstrates the Apple
   `mpp::tensor_ops::matmul2d` path; the `nax`-gated rows can follow once the
   feature scaffolding lands.
6. **`fft`** — radix + readwrite. Needs an FFT codegen path (complex types,
   bit-reversal indexing). Lowest FFAI priority.
7. **`fence`** — synchronization primitive. Needs atomics / device-memory
   fence primitives in the DSL; infrastructure, not a compute op.
8. **Winograd fast-conv** — the 3×3 stride-1 perf specialization on the
   `conv` row; the direct-conv `naive_unfold` / depthwise paths are
   landed (`ffai/conv2d.rs`, `ffai/conv3d.rs`), Winograd is the remaining
   perf follow-up.

### Model-enablement kernels (separate track from generic-op completeness)

These don't move the coverage % much but each one unblocks a model family or
removes a measured per-layer CPU sync:

- **Vision (Phase 6.5)** — `conv2d`, `patch_embed`, `rope_2d`: **landed**
  (`ffai/conv2d.rs`, `ffai/patch_embed.rs`, `ffai/rope_2d.rs`). Unblocks the
  VLM vision encoders.
- **STT / TTS (Phase 7)** — `mel_spectrogram`, `audio_conv1d`,
  `vocoder/iSTFT`: **landed** (`ffai/mel_spectrogram.rs`,
  `ffai/audio_conv1d.rs`, `ffai/vocoder.rs`). Unblocks Whisper, Kokoro, and
  Qwen-Omni audio. A radix-FFT path for the STFT / iSTFT is a perf follow-up.
- **Host-fallback closers** — `gated_rmsnorm` (Qwen3.5/3.6 GDN post-step):
  **landed** (`ffai/gated_rmsnorm.rs`). Still open: the `sdpa_decode`
  learned-sink term (GPT-OSS-20B) and a 2D-`A_log` `ssm_step` variant
  (Jamba). Each is correctness-neutral today but trades a per-layer
  CPU↔GPU sync; landing them is a decode-throughput win, not a
  correctness fix.

## Open uncertainties / counting caveats

- The four rows added in the 2026-05-21 refresh (`swiglu`,
  `sdpa_decode_batched`, `moe`, `logits processors`) had their metaltile column
  verified against source; their MLX-upstream / MLX-alpha columns are a
  best-effort read (those repos were not checked out) — treat them as
  provisional.
- `quantized_nax.rs` and `fp_quantized_nax.rs` were re-checked: both are still
  empty (TODO comment only, zero `#[kernel]`) and both are
  `#[cfg(feature = "nax")]`-gated in `mlx/mod.rs`. Counted as `✗` for metaltile.
- `mlx/strided.rs` (`mt_strided_copy`) covers strided copy but the stride
  dimensionalities were not audited — marked `~` defensively. Upstream
  `copy.metal` has multiple `copy_g_nd*` shapes.
- `ffai/sdpa_decode.rs` and `ffai/sdpa_decode_batched.rs` are FFAI-specific
  (`✗ / ✗ / ✓`) — not ports of upstream MLX kernels; they are derivatives of
  `mt_sdpa_vector` with a decoupled `kv_stride` and a batched-Q walk.
- `ffai/aura_flash_p1.rs` is marked `~` because only the `(kb=4, vb=2, dim=128)`
  instantiation is registered; the causal variant from `turbo_flash.metal` and
  other `(kb, vb, dim)` combos aren't ported yet.
- Coverage % treats the alpha-only kernels as in-scope (we maintain the fork,
  so they count toward the union).
- The Gemma / Nemotron-H / GPT-OSS-20B kernel work is now consolidated onto
  `ek/aura-port` and folded into this audit (the `sdpa_decode_d512` and
  `rms_norm_wide` rows). Three host-side fallbacks surfaced by the model
  review (`gated_rmsnorm`, `sdpa_decode` learned-sink, 2D-`A` `ssm_step`) are
  recorded as gap rows — they are correctness-neutral (the host path works)
  but cost a CPU sync per layer on the affected models.
- The Vision / STT / TTS rows (`conv2d`, `patch_embed`, `rope_2d`,
  `mel_spectrogram`, `audio_conv1d`, `vocoder/iSTFT`) are scoped from the
  Phase 6.5 / 7 plan, not yet from checked-out reference source — treat their
  MLX columns as provisional.

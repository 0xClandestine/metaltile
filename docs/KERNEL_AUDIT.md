# metaltile kernel-op coverage audit

Generated: 2026-05-18 · Refreshed: 2026-05-21 (consolidation pass +
Vision / STT / TTS front-end kernels; MoE gather-qmm + host-fallback
closers landed)
Sources surveyed:
- MLX upstream `ml-explore/mlx@main` (commit `2414e5df`)
- MLX fork `ekryski/mlx@alpha` (commit `4919270e`)
- metaltile `thewafflehaus/metaltile:ek/aura-port` (the consolidated branch —
  `origin/dev` plus the Gemma / Nemotron-H / GPT-OSS-20B kernel work)

## Summary

- Total kernel-op rows in this audit (union): **89**
- metaltile-ported kernel ops: **74 / 89 = 83 %** — 67 full ✓ (75 %), 5 partial ~ (6 %)
- **Still to cover: 17 ops not ported (✗)**, plus **7 partial ports** still to finish
- The 6 Vision / STT / TTS front-end kernels (Phase 6.5 / 7) — `conv2d`,
  `patch_embed`, `rope_2d`, `mel_spectrogram`, `audio_conv1d`,
  `vocoder/iSTFT` — are now ported (✓ rows below).
- The three model-review **host-fallback closers** (`gated_rmsnorm`, the
  `sdpa_decode` learned-sink term, the 2D-`A_log` `ssm_step` variant) are
  all landed — see the [host-fallback closers](#model-enablement-kernels-separate-track-from-generic-op-completeness)
  note.
- The four **`steel_gemm` variants** (`gather`, `masked`, `segmented`,
  `splitk + accum`) are now ported as ✓ rows — each composes the
  `steel_gemm_fused` simdgroup-MMA ladder with one extra piece of
  index / mask / split logic (no new codegen primitive). `fft`
  (radix-2 Cooley–Tukey) and `quantized_nax` (int4 matmul via Apple
  `mpp::tensor_ops::matmul2d`) are also ✓ — the latter via an
  `Op::InlineMsl` MPP escape-hatch.
- 2 in-flight kernel families have an **open PR** (not yet landed) — see
  [Kernels with open PRs](#kernels-with-open-prs).

> **Note on the 2026-05-21 consolidation pass.** The Gemma / Nemotron-H /
> GPT-OSS-20B kernel work, previously spread across separate worktrees, is now
> consolidated onto `ek/aura-port`. Two Gemma kernels — `sdpa_decode_d512` and
> `rms_norm_wide` — are added as ✓ rows. A model-side review of FFAI's decode
> path also surfaced several **host-side compute fallbacks** that existed only
> because a GPU kernel was missing; the kernels that close them
> (`gated_rmsnorm`, the `sdpa_decode` learned-sink term, the 2D-`A_log`
> `ssm_step_a2d` variant) are now all landed (✓ rows below), and the
> **Vision / STT / TTS** front-end kernels (`conv2d`, `patch_embed`,
> `rope_2d`, `mel_spectrogram`, `audio_conv1d`, `vocoder/iSTFT`) are ✓ rows
> for Phase 6.5 / 7.
> The MLX-upstream and MLX-alpha columns were **not** re-verified against those
> repos (not checked out) — only the metaltile column was re-surveyed.

## Op coverage table

| Op | MLX (upstream) | MLX (ekryski@alpha) | metaltile | Notes |
|---|---|---|---|---|
| arange | ✓ | ✓ | ✓ | `mlx/arange.rs` → `mt_arange`. Generic `T`. Direct port. |
| arg_reduce (argmax/argmin → float) | ✓ | ✓ | ✓ | `mlx/arg_reduce.rs` → `mt_argmax<T>` + `mt_argmin<T>`, both generic over `T` (f32/f16/bf16 — values widened to f32 for the comparison). Both emit the winning index as `u32` (MLX `arg_reduce_general` semantics); ties take the smallest index. Verified by `mt_arg_reduce_gpu_correctness` (CPU oracle, tie-break, all three dtypes, strided cover). |
| arg_reduce (argmax → u32 index) | ✗ | ✗ | ✓ | `ffai/arg_reduce.rs` → `ffai_argmax<T>`. FFAI-only; integer-index sampler workhorse. |
| binary (elementwise add/sub/mul/div/min/max) | ✓ | ✓ | ✓ | `mlx/binary.rs` → 6 kernels. Generic `T`. Direct port. |
| binary_two (fused two-output elementwise) | ✓ | ✓ | ✓ | `mlx/binary_two.rs` → `mt_binary_two<T>`. |
| copy (contiguous) | ✓ | ✓ | ✓ | `mlx/copy.rs` → `mt_copy<T>`. |
| copy (strided / general) | ✓ | ✓ | ~ | `mlx/strided.rs` → `mt_strided_copy`. Limited stride dimensionality. |
| ternary (select) | ✓ | ✓ | ✓ | `mlx/ternary.rs` → `mt_select<T>`. |
| unary (exp/log/sqrt/rsqrt/abs/silu/etc.) | ✓ | ✓ | ✓ | `mlx/unary.rs` → 7+ kernels including `mt_silu`. |
| swiglu (`silu(gate)·up` fused MLP activation) | ✗ | ✗ | ✓ | `mlx/swiglu.rs` → `mt_swiglu<T>`. Fused element-wise `silu(gate) * up` — the standard modern-transformer MLP activation (Llama 4, Qwen3 dense + MoE, Gemma, Mistral). metaltile fuses what MLX expresses as separate `silu` + `mul` ops; no dedicated MLX kernel. The broader `fused_gate_activation` (gelu / clipped-swiglu variants) is still a separate ✗ row below. |
| random (key hash → u32) | ✓ | ✓ | ✓ | `mlx/random.rs` → `mt_random_hash`. |
| reduce (sum/prod/max/min — all + row + col) | ✓ | ✓ | ✓ | `mlx/reduce.rs` covers `all_reduce*`, `row_reduce*`, `col_reduce*` (Grid3D one-thread-per-column, `cols`-strided fold) and `seg_reduce*` (Grid3D one-thread-per-segment, contiguous fixed-length runs) — all four ops (sum/prod/max/min) for each shape. Verified by `reduce_col_seg_gpu_correctness`. |
| sort | ✓ | ✓ | ~ | `mlx/sort.rs` → `mt_sort<T>`. Single-block path only; multi-block / segmented not yet. |
| scan (prefix sum) | ✓ | ✓ | ✓ | `mlx/scan.rs` → `mt_scan<T>` (inclusive) + `mt_scan_exclusive<T>` (exclusive — `out[i] = Σ_{j<i} inp[j]`, `out[0] = 0`). Both share the identical two-level per-/cross-simdgroup prefix-sum machinery; the exclusive variant only shifts the store stage by one slot (`base_prefix` is already the exclusive prefix of every prior thread). Verified by `scan_exclusive_gpu_correctness` (sequential CPU oracle, chunk-aligned + ragged `n`). Multi-op (prod / max / min) scan is a follow-up — the sum scan is the production-relevant shape. |
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
| steel_attention_nax | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_nax.rs` → `mt_sdpa_prefill_nax` — flash-attention prefill via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). Cooperative-tensor counterpart of `steel_attention_mma`: the standard FlashAttention-2 online-softmax loop, but `S = Q·Kᵀ` and `O += P·V` are each one cooperative `matmul2d` instead of an 8×8 `simdgroup_matmul` ladder. Tile: BQ=16, BK=16, BD=32, tpg=32 (1 SG); `head_dim` fixed at 32 so the QK descriptor's K-dim is exactly 32 (Apple's "one of M/N/K=32" rule, no head-dim tiling). QK descriptor `(16,16,32)` tb=true (Kᵀ via transposed-B read); PV descriptor `(16,32,16)`. Per-block max-rescale of the running O accumulator gives correct online softmax. Causal masking + GQA. Built as an `Op::InlineMsl` IR escape-hatch. `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `steel_attention_nax_gpu_correctness` (single-tile, multi-tile causal, GQA, f32/f16, vs a naive causal-softmax oracle). Larger head dims are a follow-up (loop the QK contraction over 32-wide D-chunks). |
| steel_gemm_fused | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused.rs` → `mt_steel_gemm_{64x64x16_2x2,32x32x16_2x2,64x64x16_1x2,32x64x16_1x2}<T>`. Plain row-major `C = A·B` via Apple 8×8 simdgroup-matrix MMA; four block-shape instantiations (each mirrors an MLX `instantiate_gemm_shapes_helper` shape). Fixed a transposed-B fragment-load bug in the original `64×64×16_2x2` kernel (it loaded `B` with the `(fn,fm)` GEMM-transposed lane convention, shipping `Bᵀ`-shaped output) plus a missing K-accumulation loop (only summed K∈[0,16)). Verified by `steel_gemm_gpu_correctness` (all four transpose modes, f32/f16/bf16). |
| steel_gemm_fused_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused_nax.rs` → `mt_steel_gemm_fused_nax` — plain fused GEMM `C = A·B` via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). Cooperative-tensor counterpart of `steel_gemm_fused`; built as an `Op::InlineMsl` IR escape-hatch (the `#[kernel]` front-end does not expose `mpp::` types), same machinery as `quantized_nax` minus the int4 dequant (B is dense `T`, coop-loaded transposed into the TG tile). `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `steel_gemm_fused_nax_gpu_correctness` (f32/f16, vs a naive triple-loop oracle). |
| steel_gemm_gather | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_gather.rs` → `mt_steel_gemm_gather_{64x64x16_2x2,32x32x16_2x2}<T>`. Row-major `C = A_gathered·B_gathered` (MLX `gather_mm`, the dense matmul of a MoE FFN): a `lhs_indices` buffer redirects each output row to a non-contiguous `A` row, a `rhs_indices` buffer selects which `[K,N]` `B` matrix each N-block multiplies against. No gather-load primitive needed — the redirection is one extra `u32` load before ordinary address arithmetic (the gather index is a per-row scalar, shared by every lane in the fragment row). Verified by `steel_gemm_gather_gpu_correctness` (identity, permuted lhs, rhs-select; f32/f16/bf16). |
| steel_gemm_gather_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_gemm_masked | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_masked.rs` → `mt_steel_gemm_masked_{64x64x16_2x2,32x32x16_2x2}<T>`. Block-masked row-major `C = A·B`: an output-block mask zeroes whole `BM×BN` blocks (uniform `if` around the K-loop + `select` on the store), an operand-block mask scales each `BM×BK`/`BK×BN` K-block contribution (a `0` mask multiplies the loaded fragment to zero — branchless). Both masks are plain `Tensor<T>` operands; no new codegen primitive needed. Verified by `steel_gemm_masked_gpu_correctness` (all-ones, checkerboard out-mask, partial op-mask; f32/f16/bf16). |
| steel_gemm_segmented | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_segmented.rs` → `mt_steel_gemm_segmented_{64x64x16_2x2,32x32x16_2x2}<T>`. Ragged-K batched matmul (MLX `segmented_mm`): each segment sums over its own `[k_start, k_end)` K-range of a shared `A`/`B`, output is `[n_segments, M, N]`. Expressed as the fused GEMM with a 3-D grid (`program_id<2>` = segment) and a K-loop whose bounds are read from a `segments` descriptor buffer instead of being a constexpr — `range(k_start, k_end, 16)` with variable bounds. No new codegen primitive needed. Verified by `steel_gemm_segmented_gpu_correctness` (single-full, disjoint, uneven ranges; f32/f16/bf16). |
| steel_gemm_splitk + accum | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_splitk.rs` → pass 1 `mt_steel_gemm_splitk_{64x64x16_2x2,32x32x16_2x2}<T>` + pass 2 `mt_steel_gemm_splitk_accum<T>` / `mt_steel_gemm_splitk_accum_axpby<T>`. Two-kernel split-K: pass 1 partitions K across a 3-D grid (`program_id<2>` = K-split, `range(k_start, k_end, 16)` clamped to `k`) and writes per-split fp32 partials to an `[n_splits, M, N]` buffer; pass 2 is a one-thread-per-output Elementwise reduce over the splits (plain sum, or `axpby` form `α·Σ + β·C_in`). The inter-kernel handoff is an ordinary fp32 device buffer — no split-K scheduling primitive needed; the partials stay fp32 so the cross-split sum keeps full precision for f16/bf16 inputs. Verified by `steel_gemm_splitk_gpu_correctness` (2-way, 3-way, axpby; f32/f16). |
| steel_gemm_splitk_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_splitk_nax.rs` → pass 1 `mt_steel_gemm_splitk_nax` + pass 2 `mt_steel_gemm_splitk_accum_nax`. Two-kernel split-K via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores): pass 1 is `steel_gemm_fused_nax` with a 3-D grid (`tgid_z` = K-split, K-loop clamped to `k`) writing per-split fp32 partials to an `[n_splits, M, N]` buffer; pass 2 is a one-thread-per-output reduce over the splits (plain sum). The inter-kernel handoff is an ordinary fp32 device buffer; partials stay fp32 so the cross-split sum keeps full precision for f16 inputs. Built as `Op::InlineMsl` IR escape-hatches. `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `steel_gemm_splitk_nax_gpu_correctness` (2-way, 3-way, multi-tile; f32/f16). |
| steel_conv 2D (implicit-GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` / `conv2d_generic`. 2D convolution as a direct conv (implicit im2col, one thread per output) rather than MLX's explicit-im2col tiled GEMM — equivalent result, no im2col staging buffer. Covers fixed-patch and runtime-stride/pad configs. The MMA-tiled implicit-GEMM is a perf follow-up. Verified by `conv2d_gpu_correctness`. |
| steel_conv 3D | ✓ | ✓ | ✓ | `ffai/conv3d.rs` → `conv3d_generic` (strided / padded dense 3D conv) + `conv3d_grouped` (adds dilation + grouped channels; `groups == in_ch` is depthwise). 5D NCDHW input, OIDHW weight — the volumetric counterpart of `conv2d.rs`: direct conv (implicit im2col), one thread per output voxel, fp32 accumulation, padding taps masked in the padded-input frame. Generic `T` (f32/f16/bf16). The MMA-tiled implicit-GEMM is a perf follow-up. Verified by `conv3d_gpu_correctness`. |
| steel_conv_general (strides/dilation/groups) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_grouped<T>`. Fully general 2D conv: strides, dilation (atrous), padding, and grouped channels (`groups == in_ch` is depthwise). NCHW input, OIHW weight with the I dimension = `in_ch/groups`. Direct conv, one thread per output, fp32 accumulation. Verified by `conv2d_gpu_correctness`. |
| conv (winograd + naive_unfold + depthwise) | ✓ | ✓ | ~ | The `naive_unfold` + depthwise cases are covered for **both 2D and 3D** — `ffai/conv2d.rs` (`conv2d_generic` + `conv2d_grouped`) and `ffai/conv3d.rs` (`conv3d_generic` + `conv3d_grouped`); the `_grouped` kernels handle depthwise via `groups == in_ch` and dilation (atrous). The Winograd fast-conv path is not ported (a perf-only specialization for 3×3 stride-1 convs). The old `mlx/conv.rs` bench-crate stub is superseded. |
| gemv | ✓ | ✓ | ✓ | `mlx/gemv.rs` → `mt_gemv<T>`. |
| gemv_masked | ✓ | ✓ | ✓ | `mlx/gemv_masked.rs` → `mt_gemv_masked<T>` (no MLX comparison wired). |
| quantized (affine_quantize / affine_dequantize) | ✓ | ✓ | ✓ | `mlx/quantized.rs` → quantize **and** dequantize for all widths: int2/int4/int8 (power-of-2, pack-aligned) + int3/int5/int6 (byte-stream, non-power-of-2). All six quantize kernels (`mt_affine_quantize_int{2,3,4,5,6,8}`) + six dequantize kernels (`mt_affine_dequantize_int{2,3,4,5,6,8}`) are ported. The int3/5/6 quantize kernels use a bit-stream OR strategy (lane 0 iterates over all group_size elements, ORing each code into the correct uint32 word) to handle codes that straddle word boundaries — no atomics needed. Verified by `affine_int2_gpu_correctness` (int2 round-trip) + `affine_int356_quantize_gpu_correctness` (int3/5/6 quantize→dequantize round-trips). |
| quantized (affine_qmv / qvm / qmm — matvec / matmul) | ✓ | ✓ | ~ | `mlx/quantized.rs` → `mt_qmv` + `mt_qmm` / `mt_qmm_bm2` / `mt_qmm_bm4` (3 M-batch tiles) with an `mt_qmm_for` selector, all f32+f16, int4. Gap: `qvm` absent, bit-widths other than int4 absent, bf16 absent. An int4 qmm via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores, MLX-parity) is **open in PR [#137](https://github.com/0xClandestine/metaltile/pull/137)** (`mt_qmm_mma_mpp`). |
| quantized (gather_qmv / gather_qmm — gather variants) | ✓ | ✓ | ~ | `ffai/moe.rs` → `mt_moe_gather_qmm_int4` — the affine grouped-gather quantized matmul. One Reduction-mode dispatch does the per-expert FFN projection for a MoE block: per-row expert routing via a CSR `expert_offsets` walk + int4-quantized per-expert weight matmul, matching MLX's `gatherQuantizedMM`. Verified by `moe_gather_qmm_gpu_correctness` (f32/f16/bf16). Gap: int4 only (MLX MoE default); the MMA / MPP-NAX perf variants from PR [#136](https://github.com/0xClandestine/metaltile/pull/136) are a follow-up. Bare-tensor `ffai/gather.rs` exists but is non-quantized. |
| moe (router top-k + permute + unpermute orchestration) | ✗ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_router_topk<T>`, `mt_moe_permute<T>`, `mt_moe_unpermute<T>`. MoE expert-routing orchestration for Qwen3.6-35B-A3B / Qwen3-Coder-30B-A3B end-to-end serving. The grouped quantized BGEMM that fuses the per-expert FFN matmuls into one dispatch is now landed — `mt_moe_gather_qmm_int4` (see the `quantized (gather_*)` row); the MMA / MPP-NAX perf variants from PR [#136](https://github.com/0xClandestine/metaltile/pull/136) remain a follow-up. |
| dequant_gather (quantized embedding-table gather) | ✗ | ✗ | ✓ | `ffai/dequant_gather.rs`. int{3,4,5,6,8} all bit-widths. FFAI-specific, no MLX counterpart. |
| dequant_gemv (quantized GEMV, FFAI flavour) | ~ (subset of `quantized.metal`) | ~ | ✓ | `ffai/dequant_gemv.rs`. int{3,4,5,6,8}, generic `T`. Coexists with the partial `mt_qmv_f32` port; FFAI-tuned shape. |
| fp_quantized (fp4/fp8 quant + dequant) | ✓ | ✓ | ~ | `mlx/fp_quantized.rs` → `mt_fp4_quant_dequant` (f32 only). fp8 path and other dtypes missing. |
| fp_quantized_nax | ✓ | ✓ | ✓ | `mlx/fp_quantized_nax.rs` → `mt_fp_qmm_nax` — fp4 (E2M1) quantized matmul via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). fp4 counterpart of `quantized_nax`: same dequant-into-TG-memory + one cooperative `matmul2d` per simdgroup per K-block, but the int4 affine nibble-dequant is swapped for an fp4 E2M1 codebook lookup (`{0,0.5,1,1.5,2,3,4,6}` magnitude LUT + sign bit, scale-only — no bias; see MLX `fp4.h`). 8 fp4 codes per `u32` pack; `GROUP_SIZE = 32` (one group per BK-block). Built as an `Op::InlineMsl` IR escape-hatch. `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `fp_quantized_nax_gpu_correctness` (f32/f16, vs a triple-loop fp4-dequant oracle). |
| quantized_nax | ✓ | ✓ | ✓ | `mlx/quantized_nax.rs` → `mt_qmm_nax` — int4 quantized matmul via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). Built as an `Op::InlineMsl` IR escape-hatch (the `#[kernel]` front-end does not expose `mpp::` types); the codegen emits the `MetalPerformancePrimitives` framework include when it detects the `mpp::` marker. MPP counterpart of `mt_qmm_mma` — same int4-dequant-into-TG-memory algorithm, one cooperative `matmul2d` per simdgroup per K-block. `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `quantized_nax_gpu_correctness` (f32/f16, vs the `qmm_gpu_correctness` triple-loop oracle). |
| fft (radix + readwrite) | ✓ | ✓ | ✓ | `mlx/fft.rs` → `mt_fft_n{32,64,128,256,512,1024}<T>`. Iterative radix-2 Cooley–Tukey FFT along the last axis (power-of-two N), one kernel covering forward + inverse via an `inv` constexpr. Complex numbers without a complex type: real / imaginary planes are two parallel real `f32` buffers, the butterfly's complex multiply expands to the four-real-mul form — the same representation `mel_spectrogram` / `vocoder` use. Bit-reversal load + `log2(N)` `threadgroup`-buffered butterfly stages; genuine O(N log N). The prime-length (Rader) / arbitrary-length (Bluestein) paths remain a follow-up. Verified by `fft_gpu_correctness` (forward vs naive DFT, inverse, round-trip; f32/f16/bf16). |
| hadamard (hadamard_n + hadamard_m) | ✓ | ✓ | ✓ | `mlx/hadamard.rs` → `mt_hadamard_n{64,128,256,512,1024}<T>` (power-of-2 FWHT via log2(N) butterfly passes). `mlx/hadamard_m.rs` → `mt_hadamard_m{12,20,28}<T>` (non-power-of-2 M factor; Sloane-table bitmask accumulate via `Op::InlineMsl`; sign arrays verified orthogonal). Verified by `hadamard_m_gpu_correctness`. |
| fence | ✓ | ✓ | ✗ | Stub file in repo, not declared. Synchronization primitive. |
| gather (bare-tensor embedding lookup) | ✓ (via indexing/) | ✓ | ✓ | `ffai/gather.rs` → `ffai_gather<T>`. FFAI's embedding-table gather. |
| indexing (scatter, scatter_axis, gather_axis, gather_front, masked_scatter) | ✓ | ✓ | ✓ | `mlx/gather_axis.rs` + `mlx/scatter_axis.rs` → `mt_gather_axis` / `mt_scatter_axis` (contiguous along-axis); `mlx/indexing.rs` → `mt_gather_front` (first-axis row gather), `mt_scatter` (first-axis row scatter, no-reduce assignment form), `mt_masked_scatter` (per-element masked gather-scatter). All five are one-thread-per-output Grid3D with an `n_elems` bounds guard. Verified by `gather_axis_gpu_correctness` / `scatter_axis_gpu_correctness` / `indexing_gpu_correctness`. |
| aura_encode (codebook quantize, fused) | ✗ | ✓ (`turbo_fused_encode` in `turbo_quant.metal`) | ✓ | `ffai/aura_encode.rs`. Bit-widths 2/3/4/8. Renamed turbo_*→aura_*. |
| aura_dequant_rotated (bulk dequant to rotated codec space) | ✗ | ✓ (`turbo_dequant_rotated` in `turbo_quant.metal`) | ✓ | `ffai/aura_dequant_rotated.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_score (compressed-domain Q·K) | ✗ | ✓ (`turbo_score`) | ✓ | `ffai/aura_score.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_value (compressed-domain value aggregation) | ✗ | ✓ (`turbo_value` in `turbo_quant.metal`) | ✓ | `ffai/aura_value.rs`. Sparsity-threshold guard mirrors MLX upstream. Renamed. |
| aura_flash_p1 (compressed-domain flash pass 1) | ✗ | ✓ (`turbo_flash_p1` in `turbo_flash.metal`) | ✓ | `ffai/aura_flash_p1.rs` → non-causal `aura_flash_p1_{kb4_vb2,kb4_vb4}_{d64,d128}` (4 instantiations) **plus** the causal variant `aura_flash_p1_causal_kb4_vb2_{d64,d128}`. The causal kernel clamps the per-token inner loop at `q_position + 1` (a constexpr-folded `causal_end` select) — every key strictly after the query token is masked out, matching `turbo_flash_p1`'s `causal` template flag. Verified by `aura_flash_gpu_correctness` (end-to-end pair) + `aura_flash_p1_causal_gpu_correctness` (full-visibility ≡ non-causal, mid-cutoff masks later blocks). |
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
| sdpa_decode + learned attention sink (GPT-OSS-20B) | ✗ | ~ | ✓ | `ffai/sdpa_decode.rs` → `ffai_sdpa_decode` `has_sink` / `sink_logit` constexprs. GPT-OSS-20B's per-head learned attention-sink logit now folds into the cross-simdgroup softmax denominator on-GPU as a virtual key (score `sink_logit`, value 0) — removing the host-side post-hoc rescale that previously cost a CPU sync per attention layer. `has_sink == 0` masks the term out, keeping the dense / sliding-window paths bit-identical to the pre-sink kernel. Distinct from the `sink_end` sink-*token* range. Verified by `sdpa_decode_gpu_correctness` (`sdpa_decode_learned_sink_matches_cpu_f32`). |
| gated_rmsnorm (fp32-in gated RMSNorm → activation dtype) | ✗ | ✗ | ✓ | `ffai/gated_rmsnorm.rs` → `ffai_gated_rmsnorm<T>`. Fused Qwen3.5 / 3.6 GDN post-step `out = w·rmsNorm(y)·silu(z)`: `y` arrives fp32 (the `gated_delta` recurrence output), the gate `z` / weight `w` / output are activation-dtype `T`. Reduction-mode, `N = TPG*4`, mirrors `mt_rms_norm` with the fp32-in / `T`-out dtype split and the `silu(z)` gate. Closes the per-GDN-layer host-side CPU sync (≈75 % of Qwen3.5/3.6 layers). Verified by `gated_rmsnorm_gpu_correctness`. |
| ssm_step (2D `A_log` / per-(head,state) decay — Jamba) | ✗ | ~ | ✓ | `ffai/ssm.rs` → `ssm_step_a2d<T>`. The 2-D-`A_log` variant of `ssm_step`: carries a per-(channel, state) `A_log` of shape `[n_heads*head_dim, state_dim]` so the decay `exp(-exp(A_log)·dt)` varies with the state index, moving Jamba's Mamba 1 selective scan onto the GPU (it previously ran host-side). Same Grid3D geometry as `ssm_step` — one thread per `(head, d)`, state `h` in fp32. The other Mamba 2 families (Mamba2, FalconH1, NemotronH, GraniteMoeHybrid) use the scalar-`A` kernel and are unaffected. Verified by `ssm_step_a2d_gpu_correctness` (f32/f16/bf16). |
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
| [#136](https://github.com/0xClandestine/metaltile/pull/136) | MoE gather BGEMM perf stack (m8 / MMA / MPP-NAX bm16 + bm64) | `quantized (gather_*)`, `moe` | Draft / WIP. The scalar `mt_moe_gather_qmm_int4` foundation has landed (see the `quantized (gather_*)` row); this PR's remaining content is the MMA / MPP-NAX perf variants. |
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
`ssm_replay` all ✓). The four `steel_gemm` variants (`gather`, `masked`,
`segmented`, `splitk + accum`) are now ✓ — each composes the
`steel_gemm_fused` simdgroup-MMA ladder with one extra piece of index / mask /
split logic. `fft` (radix-2) and `quantized_nax` are also ✓. The `steel_conv`
family (2D, general, 3D) is fully ported as direct convs (`ffai/conv2d.rs`,
`ffai/conv3d.rs`).

1. **`steel_gemm_fused` shape coverage** — only `64×64×16` is wired today for
   the `fused` row; prefill perf needs more block shapes. (The `gather` /
   `masked` / `segmented` / `splitk` ports each ship the 64×64 + 32×32 pair.)
2. **`quantized` gather_qmm MMA / MPP-NAX variants** — the scalar
   `mt_moe_gather_qmm_int4` is landed; the simdgroup-MMA and Apple
   `mpp::tensor_ops::matmul2d` perf variants (PR #136) are the remaining
   throughput follow-up, plus bit-widths beyond int4.
3. **NAX feature family** — `steel_attention_nax`, `steel_gemm_*_nax`,
   `fp_quantized_nax`. `quantized_nax` is ✓ (the `Op::InlineMsl` MPP
   escape-hatch — see its row); the remaining `nax`-gated rows can follow the
   same pattern, but each is a from-scratch `mpp::` MSL body (the `#[kernel]`
   front-end does not expose cooperative-tensor types).
4. **`fence`** — synchronization primitive. Needs atomics / device-memory
   fence primitives in the DSL; infrastructure, not a compute op. Still a
   docs-only stub (`mlx/fence.rs`).
5. **Winograd fast-conv** — the 3×3 stride-1 perf specialization on the
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
- **Host-fallback closers** — all three **landed**: `gated_rmsnorm`
  (Qwen3.5/3.6 GDN post-step, `ffai/gated_rmsnorm.rs`), the
  `sdpa_decode` learned-sink term (GPT-OSS-20B, `has_sink` /
  `sink_logit` on `ffai/sdpa_decode.rs`), and the 2D-`A_log`
  `ssm_step` variant (Jamba, `ssm_step_a2d` in `ffai/ssm.rs`). Each
  was correctness-neutral (the host path worked) but cost a per-layer
  CPU↔GPU sync; folding them on-GPU is a decode-throughput win.

## Open uncertainties / counting caveats

- The four rows added in the 2026-05-21 refresh (`swiglu`,
  `sdpa_decode_batched`, `moe`, `logits processors`) had their metaltile column
  verified against source; their MLX-upstream / MLX-alpha columns are a
  best-effort read (those repos were not checked out) — treat them as
  provisional.
- `quantized_nax.rs` is now a real port — `mt_qmm_nax`, an `Op::InlineMsl`
  int4 matmul via Apple `mpp::tensor_ops::matmul2d`, with a paired
  `quantized_nax_gpu_correctness` test (counted ✓). `fp_quantized_nax.rs`
  is still empty (TODO comment only, zero `#[kernel]`); both modules are
  `#[cfg(feature = "nax")]`-gated in `mlx/mod.rs`. The `nax`-gated kernels
  do **not** register an `inventory::submit!` BenchSpec — they are tested
  directly, and the `nax` feature is off in default / non-macOS CI builds,
  so `kernel_registry_consistency` never sees them.
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
  `rms_norm_wide` rows). The three host-side fallbacks surfaced by the model
  review (`gated_rmsnorm`, the `sdpa_decode` learned-sink term, the 2D-`A_log`
  `ssm_step_a2d` variant) are now all landed as ✓ rows — they were
  correctness-neutral (the host path worked) but cost a CPU sync per layer
  on the affected models.
- The Vision / STT / TTS rows (`conv2d`, `patch_embed`, `rope_2d`,
  `mel_spectrogram`, `audio_conv1d`, `vocoder/iSTFT`) are scoped from the
  Phase 6.5 / 7 plan, not yet from checked-out reference source — treat their
  MLX columns as provisional.

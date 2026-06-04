# metaltile kernel-op coverage audit

Snapshot of the kernels shipped by `metaltile-std` as of `dev` `c017c94`. Comparison columns track parity with `ml-explore/mlx@main` (commit `2414e5df`) and `ekryski/mlx@alpha` (commit `4919270e`).

## Summary

- **Total kernels (`tile build`): 651** — all compiled unconditionally; the 7 NAX kernels are runtime-gated to Apple10+ (M4 family and newer). See [§ NAX kernels](#nax-kernels) for what NAX is, which M-series chips activate it, and how it interacts with CI. (The jump from 374 is the PR #2 block-scaled precision matrix — see [§ Quantization precision coverage](#quantization-precision-coverage).)
- **89 / 90 kernel-op rows ported** — 89 ✓, 0 partial, 1 intentionally out of scope (`fence`; see [§ Fence ops](#fence-ops--intentionally-out-of-scope)).
- **Every floating-point kernel exposes f32 / f16 / bf16.** bf16 coverage was completed in PR #152, which also migrated every cooperative-tensor (NAX) kernel from hand-built `Op::InlineMsl` IR to the `#[kernel]` DSL via the `coop_tile_*` intrinsics + `coop_stage(T)` (bf16 → `half` staging because Apple's `matmul2d` mishandles `bfloat` cooperative tensors).
- **int4 and int8 quantized perf paths are at parity.** PR #154 built out int8 dense GEMM (`qmv`/`qmm`/`qmm_mma`/`qmm_mpp`/`qmm_nax`) and int8 MoE BGEMM (`mma`/`bm{8,16,64}_mpp`) plus int4 polish (`rms_norm_qgemv_fast`, `batched_qkv_qgemv_fast`, `dequant_gemv_int4_fast`, `qvm_int4_fast`).
- **Full block-scaled precision matrix across every weight-bearing kernel.** PR #2 (`ek/precision-support`) added spec-conformant **nvfp4 / mxfp4 / mxfp8 (e4m3+e5m2) / nvfp8**, legacy float-scale **fp4 / fp8 (e4m3+e5m2)**, the complete **symmetric integer matrix** — `int2/3/4/5/6/8` (FP32 group) + `mxint2/3/4/5/6/8` (E8M0 block) — and **FP16-scale twins** of every FP32-scaled format (`*_f16`): **30 formats** to *every* quantized family (matmul/MoE/attention/embedding/conv, on the reduction, simdgroup-MMA, MPP, and NAX paths). Each `(family × format × dtype)` is 1:1 `#[test_kernel]`-verified against the `quant::format` CPU oracle and benched. **The full integer family is present in all of them** (incl. the fast tensor-engine paths). See [§ Quantization precision coverage](#quantization-precision-coverage).
- **Attention coverage spans every production head_dim.** PR #157 added `steel_attention_nax_d{64,128,256}`, `mt_sdpa_vector_d{64,96,192,256}`, `sdpa_vector_2pass_d{64,96,256}`, and `flash_quantized_sdpa_d{96,512}` (GPT-NeoX d=96, Gemma 4 global d=512).
- **Vision / STT / TTS front-end has MMA-tiled perf paths.** PR #157 shipped `conv2d_mma` / `conv3d_mma` / `patch_embed_mma` (implicit-im2col + 4-SG 2×2 simdgroup-matrix MMA) and Bluestein non-pow2 FFT (`mt_fft_bluestein_*`, covers Whisper n_fft=400/480).
- **Tail items**: PR #157 added `mt_sort_segmented`, `mt_scan_{prod,max,min}` (+ exclusive), `sdpa_decode_batched_q8`, and 12 `flash_quantized_sdpa_{bool,float}_mask_*` variants.
- **VLM vision-tower attention** added in PR #163: `ffai_sdpa_bidirectional_d{32,64,72}` — multi-query bidirectional SDPA (no causal mask) for SigLIP / CLIP / FastViT / PaliGemma encoders. d=72 uses a ragged 3-elements-per-lane layout (24 active lanes × 3 = 72, 8 lanes idle) for PaliGemma SigLIP-So400m.

## NAX kernels

NAX ("Neural Acceleration") is the cooperative-tensor matmul path exposed by Apple's `MetalPerformancePrimitives.framework` — the `mpp::tensor_ops::matmul2d<desc, execution_simdgroup>` intrinsic. NAX kernels invoke it directly to get tensor-core-class throughput on the Apple GPU's MMA units; the non-NAX equivalents fall back to `simdgroup_matmul` (8×8 frag MMA) or scalar code.

### Hardware support

NAX requires **macOS 26+** (Metal 4) and **Apple GPU family ≥ 10**:

- **M4 family** (M4, M4 Pro, M4 Max, M4 Ultra, iPad M4) — Apple10 ✓
- **M5 family** — Apple11 ✓
- **M1 / M2 / M3** — Apple7/8/9, **no NAX**; correctness tests use a `skip_unless_apple10` runtime gate so the suite still passes on pre-M4 hardware

The runtime gate lives in `crates/metaltile-runtime/src/context.rs::Context::chip_family()` — it reports the highest supported `MTLGPUFamily` value the device claims (returning `None` when no Metal device is available or on the virtualised GPU GitHub macOS runners expose).

### Build-time gating

None. NAX kernels compile unconditionally and register their `inventory::submit!` BenchSpecs alongside every other kernel. `tile build` reports the full 374-kernel count on every host that can compile `metaltile-std`. The decision to dispatch them is made at runtime through `Context::chip_family()` (see [§ CI coverage](#ci-coverage) for the macOS Paravirtual-GPU caveat) and `skip_unless_apple10` guards in the GPU-correctness tests.

The previous `metaltile-std/nax` Cargo feature was removed — there's no longer a way to opt out at build time. Dispatching a NAX kernel on pre-M4 hardware will fail at pipeline-creation time when the device rejects the `mpp::tensor_ops::matmul2d` symbol; callers should consult `chip_family()` before selecting the NAX path.

### NAX kernels

The 7 NAX kernels:

| Kernel | File | Role |
|---|---|---|
| `mt_qmm_nax` | `mlx/quantized_nax.rs` | int4 quantized matmul prefill |
| `mt_qmm_nax_int8` | `mlx/quantized_nax_int8.rs` | int8 quantized matmul prefill |
| `mt_fp_qmm_nax` | `mlx/fp_quantized_nax.rs` | fp4 (E2M1) quantized matmul prefill |
| `mt_steel_gemm_fused_nax` | `mlx/steel/gemm/steel_gemm_fused_nax.rs` | plain fused GEMM |
| `mt_steel_gemm_gather_nax` | `mlx/steel/gemm/steel_gemm_gather_nax.rs` | MoE gather GEMM |
| `mt_steel_gemm_splitk_nax` + `_accum_nax` | `mlx/steel/gemm/steel_gemm_splitk_nax.rs` | split-K GEMM (pass1 + pass2) |
| `mt_sdpa_prefill_nax` | `mlx/steel/attn/steel_attention_nax.rs` | FlashAttention-2 prefill |

The `quantized_mpp` family (`mt_qmm_mma_mpp`, `mt_qmm_mma_mpp_int8`, the four MoE `*_mpp` variants) uses the same MPP cooperative-tensor primitive and is similarly runtime-gated via `skip_unless_apple10`. The distinction between `*_mpp` and `*_nax`: `quantized_mpp` and its MoE siblings have working MXU-fallback paths on M1–M3 via Apple's `matmul2d` itself (slower than NAX hardware but functionally correct), whereas the `*_nax` kernels were authored specifically to exercise the M4+ tensor-core descriptor and have no fallback.

### CI coverage

GitHub's macOS runners expose an Apple Paravirtual GPU that doesn't claim Apple10+, so the NAX kernels and their tests are skipped at runtime via `skip_unless_apple10`. The Tile workflow's `tile build` step still compiles them — if `MetalPerformancePrimitives` headers are unavailable on the runner's Xcode, the build will surface that breakage immediately rather than silently dropping coverage.

Local verification of NAX kernels is the developer's responsibility on M4+ hardware. The `make test` target runs the full suite; tests behind `skip_unless_apple10` execute on real Apple10+ chips and auto-skip elsewhere.

## Op coverage table

| Op | MLX (upstream) | MLX (ekryski@alpha) | metaltile | Notes |
|---|---|---|---|---|
| arange | ✓ | ✓ | ✓ | `mlx/arange.rs` → `mt_arange`. Generic `T`. |
| arg_reduce (argmax/argmin → float) | ✓ | ✓ | ✓ | `mlx/arg_reduce.rs` → `mt_argmax<T>` + `mt_argmin<T>`. Both generic over `T` (values widened to f32 for comparison); winning index emitted as `u32`; ties take the smallest index. |
| arg_reduce (argmax → u32 index) | ✗ | ✗ | ✓ | `ffai/arg_reduce.rs` → `ffai_argmax<T>`. FFAI-only; integer-index sampler workhorse. |
| binary (elementwise add/sub/mul/div/min/max) | ✓ | ✓ | ✓ | `mlx/binary.rs` → 6 kernels. Generic `T`. |
| binary_two (fused two-output elementwise) | ✓ | ✓ | ✓ | `mlx/binary_two.rs` → `mt_binary_two<T>`. |
| copy (contiguous) | ✓ | ✓ | ✓ | `mlx/copy.rs` → `mt_copy<T>`. |
| copy (strided / general) | ✓ | ✓ | ✓ | `mlx/strided.rs` → `mt_strided_copy` (2-D padded) + `mt_strided_copy_nd` (arbitrary-rank). Each output element unravels its flat index against a runtime `shape` array and gathers `src[Σ coord_d · strides[d]]`. Covers padded copies, transposes, broadcasts (stride 0), and dilated slices in one kernel. |
| ternary (select) | ✓ | ✓ | ✓ | `mlx/ternary.rs` → `mt_select<T>`. |
| unary (exp/log/sqrt/rsqrt/abs/silu/etc.) | ✓ | ✓ | ✓ | `mlx/unary.rs` → 7+ kernels including `mt_silu`. Plus `mt_scalar_fma_chain8` (fused 8-way scalar FMA) and `mt_add_rms_norm` (residual-add + RMSNorm fusion, Reduction mode). |
| swiglu (`silu(gate)·up` fused MLP activation) | ✗ | ✗ | ✓ | `mlx/swiglu.rs` → `mt_swiglu<T>`. Standard modern-transformer MLP activation (Llama 4, Qwen3 dense + MoE, Gemma, Mistral). |
| random (key hash → u32) | ✓ | ✓ | ✓ | `mlx/random.rs` → `mt_random_hash`. |
| reduce (sum/prod/max/min — all + row + col) | ✓ | ✓ | ✓ | `mlx/reduce.rs` covers `all_reduce*`, `row_reduce*`, `col_reduce*`, and `seg_reduce*` (Grid3D one-thread-per-segment, contiguous fixed-length runs) — all four ops for each shape. |
| sort | ✓ | ✓ | ✓ | `mlx/sort.rs` → `mt_sort<T>` (single-block bitonic) + `mt_merge<T>` (multi-block merge) + `mt_sort_segmented<T>` (per-row bitonic for `[batch, n]` matrices, `n ≤ 1024`, one TG per row). |
| scan (prefix sum/prod/max/min) | ✓ | ✓ | ✓ | `mlx/scan.rs` → `mt_scan<T>` + `mt_scan_exclusive<T>` (sum), `mt_scan_prod<T>` / `mt_scan_max<T>` / `mt_scan_min<T>` + exclusive variants. Sum pair uses hardware `simd_scan_exclusive`; the prod/max/min pairs use a `tgs[lsize]` threadgroup buffer for sequential cross-thread prefix reads. |
| softmax | ✓ | ✓ | ✓ | `mlx/softmax.rs` → `mt_softmax<T>` (looped + single-row collapsed). |
| logsumexp | ✓ | ✓ | ✓ | `mlx/logsumexp.rs` → `mt_logsumexp<T>`. |
| layer_norm | ✓ | ✓ | ✓ | `mlx/layer_norm.rs` → `mt_layer_norm<T>`. |
| rms_norm | ✓ | ✓ | ✓ | `mlx/rms_norm.rs` → `mt_rms_norm<T>` + `mt_rms_norm_small<T>` (2-elem/thread, small-head_dim per-head q_norm/k_norm) + `mt_rms_norm_wide<T>` (strided wide-row variant for `head_dim > 4096`, e.g. Gemma 4 31B hidden=5376). |
| rope (standard) | ✓ | ✓ | ✓ | `mlx/rope.rs` → `mt_rope`. |
| rope (Llama-3 banded) | ✗ | ✗ | ✓ | `ffai/rope_llama.rs` → `ffai_rope_llama<T>`. Decode-form, generic dtype, optional Llama-3 frequency-band scaling. |
| sdpa_vector (prefill / generic) | ✓ | ✓ | ✓ | `mlx/scaled_dot_product_attention.rs` → `mt_sdpa<T>`. Scalar SDPA for short sequences. |
| sdpa_vector (GQA decode, single pass) | ✓ | ✓ | ✓ | `mlx/sdpa_vector.rs` → `mt_sdpa_vector<T>` (d=128) + `mt_sdpa_vector_d{64,96,192,256}` (every production head_dim). Each scales the per-lane element count (2/3/6/8 elements). TPG=1024 throughout. |
| sdpa_vector_2pass | ✓ | ✓ | ✓ | `ffai/sdpa_decode_2pass.rs` → pass1/pass2 pairs for d ∈ {64, 96, 128, 256}. d=256 uses 4-buffer TG reuse to stay within the 32 KB cap. |
| sdpa_decode (FFAI production decode, decoupled `kv_stride`) | ✗ | ✗ | ✓ | `ffai/sdpa_decode.rs` → `ffai_sdpa_decode<T>` + `sdpa_decode_d{64,256,512}.rs`. FFAI-only; `kv_stride` ≠ `n_kv` (pre-allocated max-seq cache). Covers head_dim ∈ {64, 128, 256, 512}, sliding-window + sink-token (`sink_end` / `window_start`), and the GPT-OSS-20B learned-sink path (`has_sink` / `sink_logit` constexprs fold the per-head learned-sink logit into the cross-simdgroup softmax denominator on-GPU). |
| sdpa_decode_batched (speculative-decode batched-Q) | ✗ | ✗ | ✓ | `ffai/sdpa_decode_batched.rs` → `sdpa_decode_batched_q{2,4,8}<T>` + `sdpa_decode_batched_prefill.rs`. K query positions share one KV walk per dispatch, amortising KV memory bandwidth K×. `q8` dispatches at TPG=256 due to register pressure. FFAI-only. |
| sdpa_bidirectional (VLM vision-tower SDPA, no causal mask) | ✗ | ✗ | ✓ | `ffai/sdpa_bidirectional.rs` → `ffai_sdpa_bidirectional_d{32,64,72}<T>` (PR #163). Multi-query bidirectional SDPA — each query attends `[0, base_kv + n_query)` with no causal gating. Covers SigLIP-base/large + CLIP-L (d=64), FastViT-HD (d=32), and PaliGemma SigLIP-So400m (d=72). d=72 uses a ragged 3-elements-per-lane layout: lanes 0..23 own the 72 valid indices (24 × 3), lanes 24..31 bounds-mask their q·k contribution to 0 and skip their output stores (25 % lane-occupancy loss, acceptable vs the cost of a wholly different parallel decomposition). TPG=1024, one threadgroup per `(query, q_head)`. Online softmax in fp32. FFAI-only. |
| steel_attention (Flash, prefill) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention.rs` → `mt_sdpa_prefill<T>`. Scalar-flash prefill (BQ=4, online softmax, causal), generic `T`, head_dim=128. |
| steel_attention_mma (Flash prefill, simdgroup-MMA) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_mma.rs` → `mt_sdpa_prefill_mma<T>`. Real simdgroup-matrix MMA path; head_dim=128. A pre-M3 bf16-tuned sibling (`steel_attention_mma_bf16.rs`) is selected by `sdpa_prefill_mma_for()`. |
| steel_attention_nax | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_nax.rs` → `mt_sdpa_prefill_nax<T>` (d=32 base) + `mt_sdpa_prefill_nax_d{64,128,256}`. Flash-attention prefill via Apple `mpp::tensor_ops::matmul2d`. The wide variants loop the QK contraction over `head_dim/32` consecutive 32-wide D-chunks inside the outer K-block loop (first chunk uses `overwrite` descriptor, subsequent chunks `accumulate`); PV stores each chunk to a scratch `Opv` tile then accumulates into the full-width O buffer. Causal masking + GQA. Runtime-gated to Apple10+. |
| steel_gemm_fused | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused.rs` → `mt_steel_gemm_{32x32x16_1x2,32x64x16_1x2,32x32x16_2x2,64x64x16_2x2}<T>` (4 block shapes via `instantiate_gemm_shapes_helper`). |
| steel_gemm_fused_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused_nax.rs` → `mt_steel_gemm_fused_nax<T>`. Plain fused GEMM `C = A·B` via NAX cooperative-tensor `matmul2d`. Runtime-gated to Apple10+. |
| steel_gemm_gather | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_gather.rs` → `mt_steel_gemm_gather_{64x64x16_2x2,32x32x16_2x2}<T>`. Row-major `C = A_gathered·B_gathered` (MLX `gather_mm`, the dense matmul of a MoE FFN). |
| steel_gemm_gather_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_gather_nax.rs` → `mt_steel_gemm_gather_nax<T>`. Gather GEMM via NAX `matmul2d`. Runtime-gated to Apple10+. |
| steel_gemm_masked | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_masked.rs` → `mt_steel_gemm_masked_{64x64x16_2x2,32x32x16_2x2}<T>`. Block-masked `C = A·B` (output-block mask zeros whole `BM×BN` blocks; operand-block mask scales each K-block contribution). |
| steel_gemm_segmented | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_segmented.rs` → `mt_steel_gemm_segmented_{64x64x16_2x2,32x32x16_2x2}<T>`. Ragged-K batched matmul (MLX `segmented_mm`); each segment sums over its own `[k_start, k_end)` range. |
| steel_gemm_splitk + accum | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_splitk.rs` → pass 1 `mt_steel_gemm_splitk_{64x64x16_2x2,32x32x16_2x2}<T>` + pass 2 `mt_steel_gemm_splitk_accum<T>` / `mt_steel_gemm_splitk_accum_axpby<T>`. Partials stay fp32 for cross-split precision on f16/bf16 inputs. |
| steel_gemm_splitk_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_splitk_nax.rs` → pass 1 `mt_steel_gemm_splitk_nax<T>` + pass 2 `mt_steel_gemm_splitk_accum_nax<T>`. Split-K via NAX `matmul2d`; partials fp32. Runtime-gated to Apple10+. |
| steel_conv 2D (implicit-GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` / `conv2d_generic`. Direct conv (implicit im2col, one thread per output). **MMA-tiled perf path** (PR #157): `ffai/conv2d_mma.rs` → `conv2d_mma<T>` — implicit-im2col + 4-SG 2×2 simdgroup-matrix MMA, 32×32 output tile (stride=1/dilation=1/pad=0, out_ch and n_pixels divisible by 32). |
| steel_conv 3D | ✓ | ✓ | ✓ | `ffai/conv3d.rs` → `conv3d_generic` + `conv3d_grouped` (depthwise + dilation). 5D NCDHW / OIDHW. **MMA-tiled perf path** (PR #157): `ffai/conv3d_mma.rs` → `conv3d_mma<T>` — same MMA scaffold as 2D, decomposed over `(kd, kh, kw, ic)`. |
| steel_conv_general (strides/dilation/groups) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_grouped<T>`. Fully general 2D conv: strides, dilation (atrous), padding, grouped channels. |
| conv (winograd + naive_unfold + depthwise) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` / `ffai/conv3d.rs` cover `naive_unfold` + depthwise (via `_generic` / `_grouped` for both 2D and 3D). Winograd fast-conv: `ffai/winograd_conv.rs` → `winograd_conv2d_3x3<T>` (F(2×2, 3×3) minimal-filtering, one thread per 2×2 output tile, requires even output dims) + `winograd_filter_transform_3x3` + `winograd_conv2d_3x3_split` (pre-transformed filters, removes O(tiles) redundant transform). |
| gemv | ✓ | ✓ | ✓ | `mlx/gemv.rs` → `mt_gemv<T>`. |
| gemv_masked | ✓ | ✓ | ✓ | `mlx/gemv_masked.rs` → `mt_gemv_masked<T>`. |
| quantized (affine_quantize / affine_dequantize) | ✓ | ✓ | ✓ | `mlx/quantized.rs` — quantize + dequantize for all widths: int2/int4/int8 (pack-aligned) + int3/int5/int6 (byte-stream). 12 kernels (`mt_affine_{quantize,dequantize}_int{2,3,4,5,6,8}`). int3/5/6 quantize uses bit-stream OR (lane 0 ORs codes into u32 words) to handle straddling — no atomics. |
| quantized (affine_qmv / qvm / qmm — matvec / matmul) | ✓ | ✓ | ✓ | `mlx/quantized.rs` — **int4 perf**: `mt_qmv` (8-row-per-TG decode, mirrors MLX `qmv_fast`) + `mt_qmm` / `_bm2` / `_bm4` (M-batched prefill) + `mt_qmm_mma` / `_m16` (simdgroup-matrix MMA prefill) + `mt_qmm_mma_mpp` (MPP) + `mt_qmm_nax` (NAX). **int8 perf** (PR #154): `mt_qmv_int8_fast`, `mt_qmm_int8_fast` / `_bm2` / `_bm4`, `mt_qmm_mma_int8` / `_m16_int8`, `mt_qmm_mma_mpp_int8`, `mt_qmm_nax_int8` — pack-aligned (4 bytes/u32, byte-shift extract), closes the ~6–8× int8-vs-int4 perf gap. **Odd-bitwidth MMA** (PR #157): `mt_qmm_mma_b{3,5,6}` — straddle-aware two-word bit-stream dequant in the 4-SG MMA body. **All bit-widths × all dtypes**: `mt_{qmv,qvm,qmm}_b{3,4,5,6,8}` (correctness-first scalar family). **qvm perf**: `mt_qvm_int4_fast` (PR #154) — 8-col-per-TG, MLX `qvm_fast` shape. |
| quantized (gather_qmv / gather_qmm — gather variants) | ✓ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_gather_qmm_int4` (int4 affine grouped-gather) + `mt_moe_gather_qmm_b{3,5,6,8}` (all bit-widths, scalar). **int4 perf**: `mt_moe_gather_qmm_mma_int4{,_bm16}` + `_m8` (decode) + `_m{16,32}` (PR #157 short-prefill, hand-unrolled `acc0..accN` cells — the DSL doesn't lower runtime-indexed mutable arrays), MPP scale-ups `bm{8,16,64}_mpp` (`ffai/moe_mpp{,_bm8,_bm64}.rs`). **int8 perf** (PR #154): pack-aligned `mt_moe_gather_qmm_mma_int8` (1-SG MMA decode) + `_bm16_mpp` + `_bm8_mpp` (direct-input cooperative tensors, M=8 forbids coop-tensor) + `_bm64_mpp` (4-SG 2×2 long-context prefill). All MPP kernels stage bf16 through `half` cooperative tensors via `coop_stage(T)`. Bare-tensor `ffai/gather.rs` exists but is non-quantized. **Expert-indexed dequant GEMV** (PR #160): `dequant_gemv_int4_expert_indexed` — per-output-row expert selection for the gate/up FFN dispatch shape. |
| moe (router top-k + permute + unpermute orchestration) | ✗ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_router_topk<T>`, `mt_moe_permute<T>`, `mt_moe_unpermute<T>`. MoE expert-routing orchestration. The grouped quantized BGEMM that fuses per-expert FFN matmuls is counted under the `quantized (gather_*)` row. |
| dequant_gather (quantized embedding-table gather) | ✗ | ✗ | ✓ | `ffai/dequant_gather.rs`. int{3,4,5,6,8} all bit-widths. FFAI-only. |
| dequant_gemv (quantized GEMV, FFAI flavour) | ~ | ~ | ✓ | `ffai/dequant_gemv.rs` → `dequant_gemv_int{3,4,5,6,8}<T>` (one-row-per-TG) + `dequant_gemv_int4_fast<T>` (PR #154, 8-row-per-TG, mirrors MLX `qmv_fast`). The non-fast int4 kernel stays because FFAI's GPU-router opts into its indirect Swift wrapper. |
| fp_quantized (fp4/fp8 quant + dequant) | ✓ | ✓ | ✓ | `mlx/fp_quantized.rs` → `mt_fp4_quant_dequant` (fp4 E2M1) + `mt_fp8_e4m3_quant_dequant` / `mt_fp8_e5m2_quant_dequant` (fp8). Pure arithmetic transform (per-group max-scale + mantissa rounding via `floor(log2)`/`exp2`/`round`); exact for fp8 normals/subnormals, saturating (no NaN/Inf). |
| fp_quantized_mma | ✗ | ✗ | ✓ | `mlx/fp_quantized_mma.rs` (PR #157) → `mt_fp4_qmm_mma<T>` + `mt_fp8_e4m3_qmm_mma<T>`. Simdgroup-matrix BM=BN=BK=32 MMA — same 4-SG 2×2 scaffold as `mt_qmm_mma_b{3,5,6}` but with fp4 codebook lookup / fp8 E4M3 biased-exp decode. **Not** NAX-gated — runs on any M1+. Fills the M>1 perf slot between the scalar round-trip kernels and the NAX-gated `fp_quantized_nax`. fp4 decode goes through the `e2m1_decode` intrinsic; fp8 through `e4m3_decode`. The spec-conformant block-scaled MMA family (all 30 formats) lives in `mlx/block_scaled_mma.rs` (its float-scale fp4 kernel is `mt_fp4_float_qmm_mma`). |
| fp_quantized_nax | ✓ | ✓ | ✓ | `mlx/fp_quantized_nax.rs` → `mt_fp_qmm_nax<T>`. fp4 (E2M1) quantized matmul via NAX `matmul2d`. Same dequant-into-TG-memory + one cooperative `matmul2d` per simdgroup per K-block, with fp4 codebook lookup (`{0,0.5,1,1.5,2,3,4,6}` + sign bit, scale-only). 8 fp4 codes per `u32` pack; `GROUP_SIZE = 32`. Runtime-gated to Apple10+. |
| quantized_nax | ✓ | ✓ | ✓ | `mlx/quantized_nax.rs` → `mt_qmm_nax<T>` (int4) + `mt_qmm_nax_int8` (int8, PR #154 in `mlx/quantized_nax_int8.rs`). MPP counterpart of `mt_qmm_mma`: same int4-dequant-into-TG-memory algorithm, one cooperative `matmul2d` per simdgroup per K-block; int8 variant uses byte-shift extract (2 packs/lane). Runtime-gated to Apple10+. |
| fft (radix + readwrite + non-pow2) | ✓ | ✓ | ✓ | `mlx/fft.rs` → `mt_fft_n{32,64,128,256,512,1024}<T>` (iterative radix-2 Cooley–Tukey, forward + inverse via `inv` constexpr; complex via parallel real/imag planes). **Non-pow2 Bluestein** (PR #157): `mt_fft_bluestein_preprocess<T>` + `mt_fft_bluestein_chirp_filter` + `mt_fft_bluestein_cmul<T>` + `mt_fft_bluestein_postprocess<T>` — chirp-Z transform wrapping the existing pow2 FFT for arbitrary N in O(N log N); covers Whisper n_fft=400 / 480 with M=1024 padding. Prime-length (Rader) remains a follow-up. |
| hadamard (hadamard_n + hadamard_m) | ✓ | ✓ | ✓ | `mlx/hadamard.rs` → `mt_hadamard_n{64,128,256,512,1024}<T>` (FWHT, log2(N) butterfly passes). `mlx/hadamard_m.rs` → `mt_hadamard_m{12,20,28}<T>` (non-pow2 M factor, Sloane-table bitmask accumulate). Generic over `T`. |
| fence | ✓ | ✓ | — | **Intentionally out of scope** — a GPU-side sync primitive, not a compute kernel. See [§ Fence ops](#fence-ops--intentionally-out-of-scope). |
| gather (bare-tensor embedding lookup) | ✓ | ✓ | ✓ | `ffai/gather.rs` → `ffai_gather<T>`. FFAI's embedding-table gather. |
| indexing (scatter, scatter_axis, gather_axis, gather_front, masked_scatter) | ✓ | ✓ | ✓ | `mlx/gather_axis.rs` + `mlx/scatter_axis.rs` → `mt_gather_axis` / `mt_scatter_axis`; `mlx/indexing.rs` → `mt_gather_front`, `mt_scatter`, `mt_masked_scatter`. All one-thread-per-output Grid3D with bounds guards. |
| aura_encode (codebook quantize, fused) | ✗ | ✓ | ✓ | `ffai/aura_encode.rs`. Bit-widths 2/3/4/8. |
| aura_dequant_rotated (bulk dequant to rotated codec space) | ✗ | ✓ | ✓ | `ffai/aura_dequant_rotated.rs`. bits ∈ {2,3,4,8}. |
| aura_score (compressed-domain Q·K) | ✗ | ✓ | ✓ | `ffai/aura_score.rs`. bits ∈ {2,3,4,8}. Generic over `T`. |
| aura_value (compressed-domain value aggregation) | ✗ | ✓ | ✓ | `ffai/aura_value.rs`. Sparsity-threshold guard mirrors MLX upstream. Generic over `T`. |
| aura_flash_p1 (compressed-domain flash pass 1) | ✗ | ✓ | ✓ | `ffai/aura_flash_p1.rs` → non-causal `aura_flash_p1_{kb4_vb2,kb4_vb4}_{d64,d128}` (4 instantiations) + causal `aura_flash_p1_causal_kb4_vb2_{d64,d128}`. Generic over `T` (per PR #152). |
| aura_flash_pass2 (cross-block online-softmax merge) | ✗ | ✓ | ✓ | `ffai/aura_flash_pass2.rs`. fp32 accumulators → `T` final. Generic over `T`. |
| aura_flash_sdpa (fused single-pass SDPA, sinks variant) | ✗ | ✓ | ✓ | `ffai/aura_flash_sdpa.rs` → `aura_flash_sdpa_kb*_vb*_d*<T>`. Single-pass online-softmax over compressed K/V with attention sinks + sliding-window causal mask. |
| flash_quantized_sdpa (single-pass quantized SDPA, affine cache) | ✗ | ✓ | ✓ | `ffai/flash_quantized_sdpa.rs` → base `flash_quantized_sdpa_b{4,8}_d{64,96,128,256,512}<T>` (10 kernels) + `flash_quantized_sdpa_{bool,float}_mask_b{4,8}_d{64,128,256}<T>` (12 mask-variant kernels, PR #157). d=96 = GPT-NeoX (group_size=32 since 96 isn't a multiple of 64); d=512 = Gemma 4 global attention (dispatches at 256 threads/TG because 16 elems/lane pushes `maxTotalThreadsPerThreadgroup` below 1024). Bool mask = `Tensor<u32>` segment-skip, combined with the causal gate; float mask = `Tensor<T>` per-token logit bias (ALiBi / T5-relative). Bool/float at d={96,512} are follow-ups. |
| gated_delta (GatedDeltaNet recurrence) | ✗ | ✓ | ✓ | `ffai/gated_delta.rs` → `mt_gated_delta_step<T>` (decode) + `mt_gated_delta_chunk<T>` (chunked-prefill). GDN linear-attention for Qwen3.5 / 3.6 hybrid models. MMA-tiled `mt_gated_delta_wy_chunk` and fused prep+recurrence `mt_gated_delta_prep_step` (`ffai/gated_delta_prep.rs`) are landed — the latter cuts 3 host commit+wait pairs per GDN layer down to 1. |
| gated_delta_replay (tape capture + state replay) | ✗ | ✓ | ✓ | `ffai/gated_delta_replay.rs` → `gated_delta_step_record<T>` + `state_replay<T>`. Speculative-decode rollback on GDN. |
| ssm_step (Mamba 2 SSD single-token decode) | ✗ | ✓ | ✓ | `ffai/ssm.rs` → `ssm_step<T>`, `mt_ssm_step<T>` (scalar `A`). 2D-`A_log` variant `ssm_step_a2d<T>` (Jamba): per-(channel, state) `A_log`, moves Mamba 1 selective scan onto the GPU (previously host-side). |
| conv1d_causal_step (depthwise SSM conv stream) | ✗ | partial | ✓ | `ffai/ssm.rs` → `conv1d_causal_step<T>`. fp32 state recurrence. |
| ssm_replay (sequential tape capture + replay) | ✗ | ✓ | ✓ | `ffai/ssm_replay.rs` → `ssm_step_record<T>` (SSD forward + dA/dBx tape) + `ssm_replay<T>` (re-fold first k entries). |
| fused_gate_activation (silu/gelu × up gate) | ✗ | ✓ | ✓ | `mlx/fused_gate_activation.rs` → `mt_fused_gate_gelu` (gelu-tanh) + `mt_fused_gate_clipped_swiglu` (GPT-OSS: `[-7,7]` clamp, `sigmoid(1.702·g)` gate, `+1` up bias). The `silu` variant ships separately as `mlx/swiglu.rs`. |
| rms_norm_residual (RMSNorm + residual add fused) | ✗ | ✓ | ✓ | `ffai/rms_norm_residual.rs` → `ffai_rms_norm_residual<T>`. Reduction-mode, `N = TPG*4`. ~90 saved dispatches/token on Gemma4-30. |
| rms_norm_rope (RMSNorm + RoPE fused) | ✗ | ✓ | ✓ | `ffai/rms_norm_rope.rs` → `ffai_rms_norm_rope<T>`. Paired-layout RoPE; Q/K post-projection norm+rope in one dispatch. |
| rms_norm_qgemv (RMSNorm + quantized GEMV fused) | ✗ | ✓ | ✓ | `ffai/rms_norm_qgemv.rs` → `ffai_rms_norm_qgemv<T>` (int4, one-row-per-TG correctness shape) + `ffai_rms_norm_qgemv_fast<T>` (int4, 8-row-per-TG perf path, PR #154) + `ffai_rms_norm_qgemv_int8_fast<T>` (int8, 8-row-per-TG, PR #157). |
| batched_qkv_qgemv (Q/K/V 4-bit qGEMV → 1 dispatch) | ✗ | ✓ | ✓ | `ffai/batched_qkv_qgemv.rs` → `ffai_batched_qkv_qgemv<T>` (one-row-per-TG) + `ffai_batched_qkv_qgemv_fast<T>` (8-row-per-TG, GQA-guarded, PR #154). `program_id::<2>()` selects Q/K/V, output concatenated `[Q\|K\|V]`. |
| kv_cache_update (raw bf16/fp16 single-token append) | ✗ | ✗ | ✓ | `ffai/kv_cache.rs` → `kv_cache_update<T>`. FFAI-only; raw cache append. |
| kv_cache (affine-quant int4/int8/fp8 quantize + bulk dequant) | ~ | ~ | ✓ | `ffai/kv_cache.rs` — `quantize_kv` + `bulk_dequant_kv` for int4/int8. **fp8** (PR #157): `quantize_kv_fp8_{e4m3,e5m2}` + `bulk_dequant_kv_fp8_{e4m3,e5m2}`. Per-group amax → scale quantize, byte-shift extract + biased-exp decode. E4M3: mantissa_bits=3, e_bias=-6, max=448; E5M2: mantissa_bits=2, e_bias=-14, max=57344. Closes the host-side fp8 KV round-trip. |
| sampling (softmax + categorical inverse-CDF) | ✗ | ✗ | ✓ | `ffai/sampling.rs` → `softmax_categorical_sample`. Companion to `ffai_argmax` for `T > 0` decode. |
| logits processors (temperature, repetition penalty, top-k / top-p / min-p masks) | ✗ | ✗ | ✓ | `ffai/logits_{processors,topk,top_p,min_p}.rs` — in-place decode-form sampler stages composed before `softmax_categorical_sample`. FFAI-only. |
| sdpa_decode + learned attention sink (GPT-OSS-20B) | ✗ | ~ | ✓ | `ffai/sdpa_decode.rs` `has_sink` / `sink_logit` constexprs. GPT-OSS-20B's per-head learned attention-sink logit folds into the cross-simdgroup softmax denominator on-GPU as a virtual key — removing the host-side post-hoc rescale that previously cost a CPU sync per attention layer. |
| gated_rmsnorm (fp32-in gated RMSNorm → activation dtype) | ✗ | ✗ | ✓ | `ffai/gated_rmsnorm.rs` → `ffai_gated_rmsnorm<T>`. Fused Qwen3.5 / 3.6 GDN post-step `out = w·rmsNorm(y)·silu(z)`; `y` arrives fp32 (the `gated_delta` recurrence output). Closes the per-GDN-layer host-side CPU sync (~75 % of Qwen3.5/3.6 layers). |
| conv2d (vision patch conv — im2col + tiled GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` + `conv2d_generic`. NCHW input, OIHW weight; direct conv (implicit im2col, one thread per output). VLM front-end. |
| patch_embed (fused image unfold + linear projection) | ✗ | ✗ | ✓ | `ffai/patch_embed.rs` → `patch_embed<T>`. Fused image-unfold + linear projection — gathers each patch's pixels and dots them with one weight row, no intermediate unfolded buffer. **MMA-tiled perf path** (PR #157): `ffai/patch_embed_mma.rs` → `patch_embed_mma<T>` — implicit-patch-unfold + 4-SG 2×2 simdgroup-matrix MMA (`hidden` and `num_patches` divisible by 32); targets ViT-L/H shapes. |
| rope_2d (2D positional RoPE for vision tokens) | ✓ | ✓ | ✓ | `ffai/rope_2d.rs` → `ffai_rope_2d<T>`. 2D RoPE over a (row, col) token grid; head_dim split into row half + column half, each running rotate-half RoPE. VLM front-end. |
| mel_spectrogram (STFT + log-Mel filterbank) | ✓ | ✓ | ✓ | `ffai/mel_spectrogram.rs` → `mel_spectrogram<T>` (single-dispatch direct-DFT) + radix-FFT path `mel_stft_window<T>` → `mt_fft_n{n_fft}<T>` → `mel_filterbank<T>` (three kernels, O(N log N)). Generic over `T` per PR #152. STT front-end. All four kernels are bounds-guarded (`idx < n_out`) for threadgroup-rounded dispatch. **Correctness of the two direct-DFT kernels (`mel_spectrogram`, `mel_spectrogram_magnitude`) is gated at f32**: their in-thread DFT hits spectrum cancellation nulls where the GPU's approximate `sin`/`cos` diverge from libm by orders of magnitude relative to the (near-zero) true power, which low-precision *input* rounding moves onto the null — flaky O(6–16) log error on a correct kernel. The kernels stay generic over `T`; f16/bf16 are covered by `mel_filterbank` (post-FFT, no in-thread cancellation). |
| audio_conv1d (wide-stride 1D conv — STT patch embed) | ✓ | ✓ | ✓ | `ffai/audio_conv1d.rs` → `audio_conv1d<T>`. Dense wide-stride multi-channel 1D conv (NCL); distinct from depthwise `conv1d_causal_step`. STT front-end. |
| vocoder / iSTFT (TTS waveform synthesis) | ✓ | ✓ | ✓ | `ffai/vocoder.rs` → `vocoder_istft<T>`. Inverse-STFT overlap-add — one thread per output sample gathers every covering frame, inverse-DFTs with Hermitian symmetry, COLA-normalises. TTS waveform synthesis. |

## Quantization precision coverage

The op-coverage table above records *which ops exist*; this section records
*which precisions each weight-bearing op supports*.

A quantized weight stores a small **code** per element plus a **scale** that
restores magnitude. Formats vary along **three orthogonal axes** — bit-width is
*not* one of them:

1. **Element** — a signed integer (`int2…int8`) *or* a micro-float codebook
   (E2M1 / E4M3 / E5M2). The element is what the code decodes to.
2. **Scale** — how the per-block scale is stored: a raw FP32 per large group, a
   compact `E8M0` power-of-two per small block (OCP MX), an `E4M3` micro-scale ×
   a tensor-wide FP32 global (NVFP4), or FP16 (the memory-halving `*_f16` twins).
3. **Zero-point** — **symmetric** (no zero-point) or **asymmetric** (scale +
   bias). Every Track-1 format below is symmetric; the asymmetric integer track
   (Track 2) carries a zero-point for MLX-checkpoint interop.

The two "tracks" are just the practical split of that space: Track 1 = symmetric
block-/group-scaled (the spec float formats + the integer family); Track 2 =
asymmetric affine integers.

### Track 1 — symmetric block-scaled / float-scale (`quant::format::QFormat`, PR #2)

A weight `[N, K]` is quantized in contiguous **K-blocks**: per-element codes +
one block scale, no zero-point. All formats share the `quant::codec` bit
primitives — one source of truth for the host packer, the CPU correctness
oracle, and the in-kernel decode (the `e2m1_decode` / `e4m3_decode` /
`e5m2_decode` / `int8_decode` intrinsics + the straddle-aware sub-byte
bit-stream extraction for `int2/3/5/6`):

| format | element | block | block scale | global |
|---|---|---|---|---|
| `nvfp4` | E2M1 | 16 | E4M3 (1 B) | FP32 |
| `mxfp4` | E2M1 | 32 | E8M0 pow-2 (1 B) | — |
| `mxfp8_e4m3` | E4M3 | 32 | E8M0 (1 B) | — |
| `mxfp8_e5m2` | E5M2 | 32 | E8M0 (1 B) | — |
| `nvfp8` | E4M3 | 16 | FP32 (4 B) | — |
| `fp4` (legacy) | E2M1 | 32 | FP32 per-group | — |
| `fp8_e4m3` (legacy) | E4M3 | 32 | FP32 per-group | — |
| `fp8_e5m2` (legacy) | E5M2 | 32 | FP32 per-group | — |
| `int2 / int3 / int4 / int5 / int6 / int8` (symmetric) | int N | 64 | FP32 per-group | — |
| `mxint2 / mxint3 / mxint4 / mxint5 / mxint6 / mxint8` | int N | 32 | E8M0 pow-2 (1 B) | — |
| `*_f16` (twins of the FP32-scaled formats above) | as twin | as twin | FP16 (2 B) | — |

**30 formats.** The integer members are symmetric int-N: a plain FP32 group scale
(`int*`, group 64) or an OCP-MX-style `E8M0` power-of-two block scale (`mxint*`,
block 32 — `mxint8` is OCP-ratified MXINT8, the rest follow the same
construction; these map to tensor-core block-scaling units for future NVIDIA /
AMD targets). Sub-byte codes (2/3/4/5/6-bit) tight-bit-pack LSB-first into u32
words (a 4-bit stream is byte-identical to the classic nibble layout); 8-bit is
one byte/code. Every FP32-scaled format additionally has an **FP16-scale twin**
(`nvfp8_f16`, `fp4_f16`, `fp8_e4m3_f16`, `fp8_e5m2_f16`, `int2-6_f16`, `int8_f16`)
— same element + block, scale stored as a 2-byte IEEE half (the layout real
checkpoints use; the host encoder does correct round-to-nearest-even **with
subnormal support**, since wide-range elements like E5M2 push scales into f16's
subnormal range).

### Track 2 — asymmetric affine int (scale + bias)

The **asymmetric** integer track: **int2 / int3 / int4 / int5 / int6 / int8**,
per-group (64) scale **+ bias** (zero-point), in `mlx/quantized.rs`,
`ffai/dequant_gemv.rs`, `ffai/dequant_gather.rs`, `ffai/kv_cache.rs`, and the
int4+int8 MoE / MMA / MPP / NAX perf kernels. The defining difference from the
Track-1 integers (`int*` / `mxint*`) is the **zero-point** — Track 2 is the only
track that can represent a lopsided range.

This track is **current, not legacy** — it predates Track 1 (it's where 4/8-bit
support started, for KV-cache + MLX model quant) and the zero-point keeps it
irreplaceable:
- it is the **on-disk interop format for MLX-quantized checkpoints** (`mlx_lm.convert -q`
  emits asymmetric affine codes + `scales` *and* `biases`; `w = scale·q + bias`). The
  Track-1 integers are **symmetric** (no bias), so even though block-scaled `int4`/`mxint4`
  now exist, they cannot represent an asymmetric MLX checkpoint — **affine int4 (with its
  zero-point) is what's required to load every MLX 4-bit model**, by design, not for lack
  of a 4-bit block-scaled format;
- it is the right scheme for **per-decode-step KV-cache quant** (cheap min/max → scale+bias;
  block-scaled is a static-weight format whose per-step encode would need GPU encode intrinsics).

Track 1 is a parallel symmetric family (the spec float formats + the full symmetric integer
matrix); it does not replace Track 2. (The float-scale `fp4`/`fp8` *within* Track 1 — raw f32
group scale — **are** legacy, superseded by spec mxfp4/nvfp4/mxfp8/nvfp8, kept as labeled
comparison variants.)

### Block-scaled coverage — every family supports all 30 Track-1 formats

Each `(family × format)` ships a 1:1 `#[test_kernel]` (GPU-verified vs
`quant::format::dequant`) + a `#[bench]` with `.flops()` so the latency / GFLOP/s /
roofline columns rank precisions side by side. `fp8_e4m3` reuses each family's
`nvfp8` kernel (identical 8-bit-E4M3 + FP32-scale shape); the rest decode in their
own kernel. The integer formats (`int2-6`, `mxint2-8`) are generated by a
parameterized `(bit-width × scale-kind)` decode macro per family — the same
straddle-aware bit-stream extract + float sign-extend everywhere — so they reuse
each family's proven dispatch geometry verbatim (no new freeze surface).

| family | path | file(s) | also on affine int track |
|---|---|---|---|
| dequant (standalone) | elementwise | `mlx/block_scaled_dequant.rs` | int2–8 |
| qgemv (GEMV decode) | reduction | `mlx/block_scaled_matmul.rs` | int2–8, int4/int8-fast |
| qmm (GEMM prefill) | reduction | `mlx/block_scaled_qmm.rs` | int2–8 |
| qmm — simdgroup-MMA | simdgroup-matrix | `mlx/block_scaled_mma.rs` | int4, int8 |
| qmm — MPP (tensor engine) | MPP `matmul2d` | `mlx/block_scaled_qmm_mpp.rs` | int4, int8 |
| qmm — NAX | NAX `matmul2d` | `mlx/block_scaled_qmm_nax.rs` | int4, int8 |
| MoE gather-qmm | reduction | `mlx/block_scaled_moe.rs` | int3–8 |
| MoE gather — MPP (bm8/16/64) | MPP | `ffai/moe_mpp{,_bm8,_bm64}_block_scaled.rs` | int4, int8 |
| expert-indexed GEMV | reduction | `ffai/dequant_gemv_expert_indexed_block_scaled.rs` | int4 |
| fused RMSNorm + GEMV | reduction | `ffai/rms_norm_block_scaled_qgemv.rs` | int4, int8-fast |
| fused gated-RMSNorm + GEMV | reduction | `ffai/gated_rms_norm_block_scaled_qgemv.rs` | int4 |
| batched-Q/K/V qgemv + qmm | reduction | `ffai/batched_qkv_block_scaled_{qgemv,qmm}.rs` | int4, int8-fast |
| batched-4 qgemv + qmm | reduction | `ffai/batched_4_block_scaled_{qgemv,qmm}.rs` | int4 |
| embedding gather | elementwise | `ffai/dequant_gather_block_scaled.rs` | int3–8 |
| flash SDPA (block-scaled KV) | flash | `ffai/flash_block_scaled_sdpa.rs` (d64/96/128/256/512, all 30 formats¹) | affine int4/int8 KV, same dims |
| patch embed (linear projection) | reduction | `ffai/patch_embed_block_scaled.rs` | — |
| patch embed (simdgroup-MMA) | simdgroup-matrix | `ffai/patch_embed_mma_block_scaled.rs` | — |
| conv2d / conv3d (direct) | reduction | `ffai/{conv2d,conv3d}_block_scaled.rs` | — |
| conv2d / conv3d (im2col-MMA) | simdgroup-matrix | `ffai/{conv2d,conv3d}_mma_block_scaled.rs` | — |
| depthwise conv2d | reduction | `ffai/depthwise_conv2d_block_scaled.rs` | — |
| audio conv1d (STT front-end) | reduction | `ffai/audio_conv1d_block_scaled.rs` | — |
| fishspeech conv1d (TTS front-end) | reduction | `ffai/fishspeech_conv1d_block_scaled.rs` | — |

¹ Flash KV covers every production head dim (d64/96/128/256/512), each × all 30 formats —
**no holes**. int8's group size (64) doesn't divide d96, so that case tiles with a ragged
trailing block: `n_blocks = ceil(dim/block_size)` (a 64-block + a 32-block), with the host
packer and kernel rounding up identically so codes + scales stay self-consistent. The
geometry is one simdgroup per query (grid `[32, n_query, 1]`), identical across dims (only
the per-lane dim count changes).

### The full integer matrix everywhere

The **complete symmetric integer family** — `int2/3/4/5/6/8` (FP32 group scale) and
`mxint2/3/4/5/6/8` (E8M0 block scale) — is present in **every** family above, including
the fast tensor-engine paths (simdgroup-MMA, MPP, NAX, MoE-MPP) where integer throughput
is highest on Apple GPUs / the ANE and where the `mxint*` E8M0 layout maps to tensor-core
block-scaling on future NVIDIA / AMD targets. The core matmul / MoE / RMSNorm-GEMV /
batched-QKV / KV-cache / attention families *additionally* carry the pre-existing
asymmetric affine integers (scale + bias) for MLX-checkpoint interop. No weight-bearing
family lacks an integer path.

### Model-format decode — one codec, no per-oracle drift

The Track-1 `QFormat` matrix above is metaltile's own block-scaled layout. Alongside it,
several kernels consume **external model formats** with their own on-disk byte layouts but
the *same* element/scale arithmetic. These all now decode through the shared
[`quant::codec`](../crates/metaltile-std/src/quant/codec.rs) primitives (`e2m1_decode`,
`e4m3_decode`, `e8m0_decode`, `int8_decode`, `f16_scale_decode`) and the
[`quant::gguf`](../crates/metaltile-std/src/quant/gguf.rs) host packer/oracle — the single
source of truth the kernel, the host quantizer, and the CPU correctness oracle all read,
so an oracle can no longer drift from its kernel (the bug class fixed twice, independently,
in PRs #264 and #265, once per duplicated copy of the layout map):

| Format | Provenance | Element × scale | Decode source |
|---|---|---|---|
| `q8_0` | GGUF / llama.cpp | int8 × f16 block scale | `gguf::{pack,dequant}_q8_0` → `codec` |
| `q2_k` | GGUF k-quant | 2-bit × two-level super-block (d·scale − dmin·min) | `gguf::{pack,dequant}_q2_k`, `gguf::q2_k_qpos` → `codec` |
| DSv4 fp8-block | DeepSeek-V3 safetensors | e4m3 × per-(128×128) f32 | `codec::e4m3_decode` (NaN sentinel kept explicit) |
| DSv4 mxfp4 | DeepSeek-V3 safetensors | = OCP `mxfp4` (e2m1 × e8m0) | `codec::{e2m1,e8m0}_decode` |

The MoE Q2_K correctness oracles (`moe_gather_down_q2k`, `moe_bgemm_q2k_mpp`) import the one
`gguf::q2_k_qpos` index map rather than each carrying a private copy.

- **`iq2_xxs`** (GGUF i-quant) is a **codebook** format — the "element" is a 256×8 signed-octet
  grid lookup plus a 7-bit sign-parity expansion, not an `element × scale` decode — so it sits
  *outside* the codec matrix. The kernel is a **WIP scaffold** (grid table unlanded; ABI/shape
  smoke-test only) and its MoE siblings (`moe_*_iq2xxs`) share that status.
- **`gemm_q8` / `gemv_q8`** consume the `q8_0` format but are **bench-only** (no correctness
  oracle yet) — a test-coverage gap, not a decode-drift risk.

### Gaps / deliberate exclusions

- **Winograd conv** — the filter pre-transform (`GgGᵀ`) amplifies quantization error;
  quantized Winograd is non-standard and counterproductive, so it stays f16/bf16/f32.
- **Activation-only ops** (RoPE, SSM / GatedDeltaNet recurrence, standalone norms, dense
  SDPA / GEMM, elementwise / reduction / softmax / sort / scan / FFT / gather-axis) carry
  no persistent weight tensor → activation-precision (`<T>`) only.
- **KV-cache *write*** quantizes per decode step → affine int4/int8/fp8 (cheap min/max
  encode) is the right scheme; block-scaled is a static-weight format whose per-step
  *encode* would need GPU encode intrinsics.

## Notes on counting decisions

A few rows mix multiple `.metal` files into one op or split one file into multiple ops:

- **`sdpa_vector*`** is counted as two ops: `sdpa_vector` (single pass) + `sdpa_vector_2pass` (two-pass pair). Upstream `sdpa_vector.h` defines `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`.
- **AURA stack** — each codec stage (`encode`, `dequant_rotated`, `score`, `value`, `flash_p1`, `flash_pass2`) is a separate row; `aura_flash_sdpa` (sinks-fused single-pass) is also its own row.
- **`steel/`** — each kernel file becomes one op row; per-block-shape instantiations are not counted separately. `steel_attention` (scalar) and `steel_attention_mma` (simdgroup-MMA) are two rows because they are separately compiled kernels with different lowering strategies.
- **`quantized.metal`** — split into four rows by semantic operation (quant/dequant, qmv/qvm/qmm matmul, gather-qmv/qmm, fp4/fp8). The Apple10+ variants (`quantized_nax`, `fp_quantized_nax`) are separate rows because they live in separate modules with runtime-only dispatch gating. `fp_quantized_mma` is its own row (runs on M1+, no Apple10 gating).
- **`indexing/`** is one row covering scatter / scatter_axis / gather_axis / gather_front / masked_scatter. Bare `gather` is its own row (FFAI-specific).
- **`moe`** is the routing/permute/unpermute orchestration in `ffai/moe.rs`. The grouped quantized BGEMM lives under the `quantized (gather_*)` row.
- **`logits processors`** is one row for the FFAI sampler-stage kernels (`temperature`, `repetition_penalty`, `topk` / `top_p` / `min_p` masks).
- Cells marked **`~`** indicate a partial port (typically one bit-width, one dtype, or one block shape where upstream has many) — see the notes column for the specific gap.

## Out-of-tree micro-optimization proposals

Some hot-path patterns require codegen-layer support to land cleanly and are documented as proposals rather than landed kernels. See [`specs/PROPOSED_OPTIMIZATIONS.md`](PROPOSED_OPTIMIZATIONS.md) for full rationale and implementation sketches:

- **`simd_broadcast` for scale/bias** — int4/int8 GEMV kernels where 4 (int4) / 16 (int8) consecutive lanes share a group scale/bias. Hardware already coalesces same-address loads from one simdgroup, so the optimization is opportunistic (no measured profile signal yet).
- **`fast::` math intrinsics** — `mel_spectrogram`, `mt_softmax`, `mt_logsumexp`, `vocoder_istft` use IEEE-precise built-ins. Switching to `fast::exp`/`fast::log`/`fast::sin`/`fast::cos` would give ~1.5–2× speedup at 1–3 ULP. Needs new `UnaryOpKind` IR variants + precision validation against existing test tolerances.
- **K-loop software pipelining** — overlap next K-block load with current MMA in MMA-tiled K-loop kernels. ~15–25 % throughput win on M3+. Needs a new `Op::PrefetchAsync` IR op + a `prefetch.rs` codegen pass.

Already in place: **`float4` / `half4` vectorized X loads** via the existing `VectorizePass` (`crates/metaltile-codegen/src/passes/vectorize.rs`). **fp32 accumulators** are correctness-required across all production shapes; the f16/bf16-accumulator proposal was rejected.

## Fence ops — intentionally out of scope

MLX's `fence.metal` (`mlx/backend/metal/kernels/fence.metal`, ~52 lines) is **not a compute kernel** — it's a GPU-side synchronisation primitive. Deliberately not ported to metaltile; the `fence` audit row is marked `—` rather than `✗`.

### What the fence ops are

Three kernels: `input_coherent` (force input-buffer visibility), `fence_update` (bump a counter in a shared buffer), and `fence_wait` — a compute kernel that **spin-loops** reading that counter until it changes. Together they order work *across command buffers / streams* without a CPU round-trip.

### How MLX uses them

`mlx/backend/metal/fence.cpp`'s `FenceImpl` has two paths:

- **Default:** `device->newSharedEvent()` — a standard `MTLSharedEvent`. The wait executes in the GPU *command processor*, not a shader core.
- **`use_fast` path** (the `fence.metal` spin-wait kernels): gated behind `GPUFamilyMetal3` + macOS 15 + an opt-in env var (`metal_fast_synch`). **Off by default.**

So MLX itself treats the GPU spin-wait fence as an opt-in latency micro-optimization for its multi-stream `async_eval` workloads — not a primitive every pipeline needs.

### Why FFAI doesn't need it

- FFAI's current pipeline is single-stream autoregressive decode. Within a forward pass, Metal's automatic hazard tracking orders kernels in a command buffer for free; across command buffers on one queue, submission order suffices.
- CPU/GPU pipelining (build command buffer N+1 while the GPU runs N) is `commit` + completion handlers, not a fence.
- For genuine cross-queue / cross-stream GPU sync, `MTLEvent` / `MTLSharedEvent` (encoder-level — `encodeWaitForEvent` / `encodeSignalEvent`) are the correct, power-efficient primitive, and they belong in `metaltile-runtime`'s dispatch layer, not as a `#[kernel]`.
- A `fence_wait` spin-wait is a deliberate near-infinite GPU loop: it burns a shader core + power, and a counter that never updates (a bug, a wrong dispatch) is a permanent GPU pin → hard reboot.

### When this could change

If FFAI later runs **multiple concurrent GPU streams** — e.g. speculative decoding (draft/target overlap), prefill/decode overlap, or ANE+GPU concurrency — it will need cross-stream ordering. The right implementation is `MTLEvent`-based encoder-level sync added to `metaltile-runtime` (MLX's own default), **not** a spin-wait `#[kernel]`. Only if profiling later shows that `MTLEvent`'s command-processor latency is a measured bottleneck for an ultra-fine-grained sync pattern would the opt-in spin-wait become worth revisiting — and even then it's a runtime concern, not a metaltile kernel.

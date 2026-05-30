# DeepSeek V4-Flash forward-path kernel arc

Status snapshot for the multi-session DSv4 decode build-out. Captures
the architecture, the verified design decisions, and the gotchas
that surfaced from four research agents reading the official
`deepseek-ai/DeepSeek-V4-Flash/inference/model.py`, the
`antirez/llama.cpp-deepseek-v4-flash` fork (the engine that produced
Tom's local GGUF), the vLLM #40760 PR series, and the published
issue trackers.

## Confirmed checkpoint shape (DSv4-Flash)

```
hidden=4096   q_lora_rank=1024   o_lora_rank=1024
n_heads=64    num_key_value_heads=1   head_dim=512 (= 448 nope + 64 rope)
qk_rope_head_dim=64
n_routed_experts=256   num_experts_per_tok=6   n_shared_experts=1
moe_intermediate_size=2048
num_hidden_layers=43   num_nextn_predict_layers=1 (MTP, separate file)
sliding_window=128     topk_method=noaux_tc   scoring_func=sqrtsoftplus
max_position_embeddings=1048576   expert_dtype=fp4
hc_mult=4   hc_eps=1e-6   hc_sinkhorn_iters=20
```

Per-layer schedule lives in `compress_ratios` (44 entries):
`[0, 0, 4, 128, 4, 128, ..., 4, 0]` — full MLA on layers 0-1, then
strict CSA(4)/HCA(128) 1:1 interleave on 2-41, plus one boundary
layer at 42.

## Tensor names (official short form, not HF long form)

```
embed.weight
layers.{i}.attn_norm.weight, ffn_norm.weight
layers.{i}.attn.wq_a.{weight,scale}, wq_b.{weight,scale}, q_norm.weight
layers.{i}.attn.wkv.{weight,scale}, kv_norm.weight        # SINGLE projection
layers.{i}.attn.wo_a.{weight,scale}, wo_b.{weight,scale}  # grouped O-LoRA (V4)
layers.{i}.attn.attn_sink                                  # per-head softmax sink
layers.{i}.hc_attn_{fn,scale,base},  hc_ffn_{fn,scale,base}
layers.{i}.ffn.gate.weight,           gate.e_score_correction_bias
layers.{i}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}
layers.{i}.ffn.experts.{e}.{w1,w2,w3}.{weight,scale}      # 256 × 43 = 11k tensors
layers.{i}.attn.compressor.{kv,gate,ape,norm}             # CSA/HCA only
layers.{i}.attn.indexer.{compressor_kv, attn_q_b, proj}   # CSA only
norm.weight, head.weight
```

Weights ship as **block-FP8 (e4m3, 128×128 scales)** for attention +
router + shared experts + lm_head; **MXFP4 (e2m1, block 32, E8M0
scales)** for routed experts only. Tom's IQ2_XXS GGUF is a separate
recompression path (see this PR's existing kernel).

## Per-layer forward (cribbed from antirez/llama.cpp + official Python)

```
residual = x
x, post_a, comb_a = hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)
x = rms_norm(x, attn_norm)

# Q path — MLA LoRA-down/up
q = wq_a @ x; q = rms_norm(q, q_norm); q = wq_b @ q
q = q.view(64, 512)
q = rms_norm(q, eps)
q[..., -64:] = rope(q[..., -64:], freqs_cis)                      # tail-only rope

# KV path — single MQA projection (NOT V3's two-step)
kv = wkv @ x; kv = rms_norm(kv, kv_norm)
kv[..., -64:] = rope(kv[..., -64:], freqs_cis)
kv_cache.append(fp8_kv_quantize(kv))                              # cache is fp8

# Per-layer attention (branch on compress_ratios[i])
if ratio == 0:    attn_out = mla_dense(q, kv_cache, attn_sink)
elif ratio == 4:  # CSA
    kv_comp  = csa_compressor(kv_cache)                            # m=4 overlap pool
    scores   = lightning_indexer(q, kv_comp)
    topk     = argsort_top_k(scores, 512)
    mask     = compressed_mask_from_topk(topk) | window_mask_128
    attn_out = sparse_sdpa(q, kv_comp, mask, attn_sink)
else: # ratio == 128 — HCA
    kv_heavy = hca_compressor(kv_cache)                            # m=128 non-overlap
    attn_out = hca_dense_sdpa(q, kv_heavy, attn_sink)

# V4 QUIRK — inverse RoPE on attention OUTPUT before O-proj
attn_out[..., -64:] = rope(attn_out[..., -64:], freqs_cis, inverse=True)
attn_out = wo_b @ grouped(wo_a, attn_out)                          # 8 head-groups
x = hc_post(attn_out, residual, post_a, comb_a)                    # mHC, NOT add

# FFN
residual = x
x, post_f, comb_f = hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x = rms_norm(x, ffn_norm)
if i < 3:   # layers 0..2 dense
    ffn_out = swiglu(gate, up, down)
else:       # MoE
    s = router @ x; s = sqrt(softplus(s))                          # ★ shipped this PR
    s_biased = s + e_score_correction_bias                         # selection only
    topk = top6(s_biased)
    w    = gather(s, topk) / sum * 1.5                             # routing weights use UNBIASED s
    ffn_out = sum_k w[k] * expert_k(x) + shared_expert(x)
x = hc_post(ffn_out, residual, post_f, comb_f)
```

## Shipped this PR (metaltile #243)

| Piece | Kernel file | Status |
|-------|-------------|--------|
| GGUF Q8_0 dequant | `gguf_dequant_q8_0.rs` | full + e2e tested on 86 GB DSv4 |
| GGUF Q2_K dequant | `gguf_dequant_q2_k.rs` | full + e2e tested on 86 GB DSv4 |
| GGUF IQ2_XXS dequant | `gguf_dequant_iq2_xxs.rs` | full + e2e tested (canonical iq2xxs_grid + ksigns_iq2xs) |
| MXFP4 dequant | `dsv4_mxfp4_dequant.rs` | full + round-trip tested |
| Block-FP8 e4m3 dequant | `dsv4_fp8_block_dequant.rs` | full + round-trip tested |
| sqrt(softplus) router | `moe_router_sqrtsoftplus.rs` | full + 288/512-expert tested |

All compile clean against current `clandestine/dev`, 111/111 tests pass,
workspace clippy + fmt clean. Implicit Store coercion applied per playbook.

## Remaining kernel arc (each its own design pass)

### Medium-lift (next session)

1. **HCA dense SDPA d512 + attn_sink** — clone of `sdpa_decode_d512.rs`
   with the `has_sink` / `sink_logit` constexprs added to the inner
   online-softmax loop. d512 has TPG=512 + 4-phase output reduction; the
   sink fits cleanly into the global-max/global-sum reduction phase.
   Pitfall: attn_sink lives in fp16 in the checkpoint, scale to fp32
   before adding to the online-softmax running sum.

2. **mHC hc_pre / hc_post (4-channel residual mix)** — 4-channel
   stream projection + Sinkhorn-Knopp 4×4 normalize. The 4×4 SK loop
   itself is too small to launch a GPU kernel (~320 ops total per
   forward); compute `comb` on CPU once per layer per forward step
   from the 4×4 `comb_logits` tensor. The hc_pre / hc_post matmuls
   are `[hidden, 4]` × `[4, 4]` — standard small gemm.
   Pitfall: mHC REPLACES `x = x + sublayer_out` — implementing it as
   plain residual breaks the trained gradient flow.

3. **CSA / HCA compressor (overlap m=4 / non-overlap m=128 pool)** —
   softmax-gated weighted sum + APE absolute-position-embed adds.
   CSA overlap-pool aggregates `2*ratio=8` raw tokens per
   compressed entry (NeMo docs). The compressor's `wkv` is shared
   between K and V (`num_kv_heads=1` MLA-style); per-layer per
   forward-step we append one compressed entry on the CSA path,
   one per 128 tokens on HCA.

### Heavy-lift (multi-session)

4. **MLA dense decode with absorbed W_UK** (full-attn layers 0, 1, 42).
   Per-token: `q_absorbed[h] = einsum(q_nope, W_UK[h])`, then dual-dot
   `score = q_absorbed · c_KV + q_pe · k_pe`. fp32 accumulators
   critical (ik_llama #305 "DDDDDD gibberish" without). softmax_scale
   uses `mscale_all_dim²` not `mscale` (Megatron-LM #1429). Output
   absorption: `o ← einsum(o, W_UV)`.

5. **Lightning Indexer** (CSA layers only). Per-layer sub-network:
   `I_{t,s} = Σ_j w_{t,j} · ReLU(q^I_{t,j} · k^I_s)` — 64 heads × 128 dim,
   top-512 selection via bitonic-top-k on 32-thread simdgroup. Has its
   OWN compressor stream + own APE separate from attention's. Indexer
   reuses main `wq_a` LoRA-down; `indexer.wq_b` is the indexer-specific
   up-projection.

6. **CSA sparse-gather SDPA** (CSA layers only). Index-gather inside
   the FA inner loop over top-512 compressed + 128 sliding window
   (≤ 640 positions total). Closest cousin = the shelved
   `tom/feat/block-sparse-sdpa` branch (block skip is wrong shape —
   V4 needs arbitrary-index gather, NOT block-aligned skip).
   Pitfall: indices come from top-k score-sorted; **sort ascending
   before gather** so causal mask + position-dep RoPE apply correctly
   AND the K compressed-cache walk is near-coalesced.

7. **FP8 + MXFP4 fused-with-routing GEMV** — performance follow-up.
   The dequant kernels in this PR materialise full fp16 weight tiles;
   MLX's published MXFP4 path on DSv3.2 ran at **0.27 tok/s vs 16
   tok/s expected** (mlx#3402) because dequant→fp16→GEMM never fused.
   The Apple-side fused variant interleaves the LUT lookup with the
   gemv accumulator directly.

## Known sharp edges

- **mscale**: softmax scale must include the YaRN mscale factor squared
  on long context, not raw `head_dim^(-0.5)` (Megatron-LM #1429).
- **wkv_b quant floor**: in MLA-absorbed mode, `wkv_b` (the K/V
  up-projection) participates in every score gemv — keep it ≥ Q8_0
  even in aggressive MoE-IQ2 mixes or quality drifts severely
  (ik_llama #477).
- **HCA RoPE base**: `compress_rope_theta=160000` applies AT
  compressor-emit time to the last 64 dims; mixing with the main
  `rope_theta=10000` silently degrades long-context recall.
- **MoE matmul shape**: V4's MoE is the per-token bottleneck. Don't
  port `dequant_gemv_int4` directly; the matmul needs to fuse with
  the noaux_tc top-k gather (288 experts × 6 active × small batch
  with sparse index pattern).
- **No native FP8 on Apple**: every FP8 path goes through a 256-byte
  → fp32 LUT (host-precomputed). Arithmetic conversion is 6-8 ALU
  per element with a denormal branch; the LUT collapses to one load.
- **MTP head** ships in a separate `*-MTP-*.gguf` file as ONE decoder
  block; it is NOT weight-shared with the last main layer. Treat as
  an EAGLE-style speculative-decode draft attachment.

## Sources cited across the research

DeepSeek tech reports (V3 arxiv 2412.19437, V4 paper mirror), official
`deepseek-ai/DeepSeek-V4-Flash` HF repo, `antirez/llama.cpp-deepseek-v4-flash`,
vLLM PR #40760 + #40860, transformers `modeling_deepseek_v4.py`, NeMo
dsv4-flash guide, Lior Sinai MLA derivation, FlashMLA, SGLang V3 docs,
turboquant_plus M5 Max analysis, mlx#3402 + mlx#2962, ik_llama #305 +
#477, Megatron-LM #1429.

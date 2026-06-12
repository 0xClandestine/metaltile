<!--
Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
SPDX-License-Identifier: Apache-2.0
-->
# Kernel Consolidation Plan

The single roadmap for restructuring `metaltile-std`'s ~150k lines of kernels
into a smaller, family-organized library. This doc owns **what the target
structure is and the order we get there**; [`STYLE_GUIDE.md`](../STYLE_GUIDE.md)
owns **how an individual kernel/bench/test is written** in the target style.
Where they overlap (naming, per-file shape), defer to the style guide.

> **Status:** in progress. `convolution/` is the proven exemplar (done — see §4)
> and `rope/` has landed. The remaining families migrate one-per-PR per §6.

## 1. Why

The legacy split is two top-level folders — `mlx/` (≈51 files, a kernel with an
upstream metal source) and `ffai/` (≈145 files, everything else). Two problems:

1. **The organizing axis is wrong.** "Does an upstream metal reference exist?"
   is a property of one *bench* — an optional comparator — not of the kernel. It
   says nothing about what the kernel *does*.
2. **Massive duplication.** The same operation appears in many near-identical
   files: one per dtype, per bit-width, per quant format, per dispatch path
   (`<op>` / `<op>_block_scaled` / `<op>_mma` / `<op>_int8` / `<op>_f16`). The
   convolution module alone was 20 files / 24k lines for ~5 operations.

Goal: group by **operation family**, fold the dtype/format/bit-width/head-dim
variation onto compile-time axes, and pull shared sub-expressions into
primitives — reducing LOC dramatically while making each family easy to find and
maintain. Generated MSL is unchanged; kernel inventory names are unchanged (so
the FFAI emit path is unaffected).

## 2. Target directory layout

```
crates/metaltile-std/src/kernels/
  ops/        elementwise/core primitives: binary · unary · ternary · copy · arange ·
              random · reduce · arg_reduce · scan · indexing · gather/scatter · hadamard ·
              fence · clamp · logsumexp · vector_add · axpy · strided
  gemm/       gemm · gemv(_masked) · batched-projection (qkv / 4) · patch_embed · steel/gemm
  sdpa/       ALL attention: bidirectional(+relpos/windowed/conformer) · decode(+d64..d512/
              2pass/batched/sink) · multi(+d256/tree-mask) · prefill_mma · flash_quantized ·
              aura_flash · steel/attn
  moe/        moe orchestration · mpp(bm8/bm64 × int8) · bgemm/gemv(q2k/iq2xxs) · block_scaled_moe
  norm/       rms_norm(+residual/rope/qgemv/gated) · layer_norm · adain1d
  rope/       ✅ DONE — rope · rope_2d · rope_banded · rope_yarn · partial_rope
  convolution/ ✅ DONE — conv1d/2d/3d · depthwise · winograd · steel_conv (see §4)
  ssm/        ssm(_replay) · gated_delta(+wy/prep/chunk) · mamba pregate-rmsnorm
  quant/      INFRA + the op×format matrix (§7): codec · format · gguf · block_scaled_* ·
              quantized_* · fp_quantized_* · affine · aura codec stack · dequant_*
  audio/      mel_spectrogram(+magnitude/stft/filterbank) · lstm · vocoder · snake1d · upsample
  vision/     resize_normalize(+bicubic) · im2col · patch_unfold · pos_emb_2d · avg_pool2d ·
              transpose_th · frame_diff · broadcast_affine
  sampling/   logits_topk/top_p/min_p/processors · sampling · softmax · sort
  kv_cache/   kv_cache(_update_many) · fft
  primitives.rs   cross-family decode/reduce ops (mt_decode_e2m1/e4m3/e5m2/e8m0, mt_unpack_nbit, …)
  mod.rs          pub mod ops; pub mod gemm; pub mod sdpa; …
```

Notes:
- The `kernels/` umbrella keeps the crate root (`lib.rs`, `build.rs`, `utils.rs`)
  uncluttered.
- `quant/` holds **format/codec/lowering infrastructure**, not per-op kernels —
  a quantized matmul lives in `gemm/`, not `quant/` (its format is an *axis* of
  the matmul, §5/§7).
- No `mlx` / `ffai` / `mlx_ref` naming anywhere. A metal reference is an optional
  `.with_reference(...)` on a bench, nothing more.
- **No model names in kernels.** Name a kernel for the operation / layout it
  implements, never for a model (`rope_llama` → `rope_banded`, `kokoro` →
  `adain1d`/`lstm`). Many models share an op in different permutations; the
  differentiator is the *layout*, which the name should describe. Model-specific
  usage notes go in a comment above the kernel definition.
- Folder names spell out abbreviated single words (`convolution`, not `conv`)
  and keep standard acronyms (`gemm`, `sdpa`, `moe`, `rope`, `ssm`, `kv_cache`).

## 3. The three LOC-reduction tools

Apply in this order per family. (Authoring detail for each lives in
[`STYLE_GUIDE.md`](../STYLE_GUIDE.md) §5–6.)

| Tool | What it collapses | Mechanism |
|---|---|---|
| **1. Shared primitives** | A decode/reduce sub-expression repeated across kernels | factor into a `#[kernel]` and **call** it; `KernelInlinePass` inlines at codegen (zero overhead) |
| **2. `#[kernel(variants(...))]`** | dtype / bit-width / head-dim / format families written as `macro_rules!` or copy-paste | one kernel stamped per compile-time tuple, with constant-folded `if FMT == …` decode branches |
| **3. Merge by op, format as an axis** | `<op>.rs` + `<op>_block_scaled.rs` + `<op>_mma.rs` + `<op>_int8.rs` + `<op>_f16.rs` | one `<op>.rs`; the quantized form is the *quantized form of an op*, not a separate family |

**The key insight (from the conv work):** for most kernels the outer loop and
accumulation are byte-for-byte identical across formats — the only line that
differs is the weight decode. Tools 1–2 isolate that line; tool 3 then merges the
files. The macro `*_bench_fmt!` / `*_test_fmt!` pattern (one macro + N
invocations) is the interim DRY step for benches/tests until they move to
`variants(...)`.

## 4. Worked exemplar — `convolution/` (done)

The convolution module is the proof of the recipe:

| | Before | After |
|---|---|---|
| Files | 20 (`conv2d`, `conv2d_block_scaled`, `conv2d_mma`, `conv2d_mma_block_scaled`, depthwise×3, conv3d×4, conv1d×5, winograd, …) | `convolution/{conv1d,conv2d,conv3d}.rs` + `consolidated/{primitives,conv1d}.rs` + `fused/` + `steel_conv/` |
| LOC | ~24,200 | ~1,600 target (~93% reduction) |

What landed: stale `ffai/` duplicates deleted; per-format **benches** and
**tests** collapsed to `*_bench_fmt!` / `*_test_fmt!` macros (one macro + 30
invocations, vs ~30 explicit fns each); the GGUF/DSv4 dequant oracles routed
through the shared `quant::codec`. Phases still open for conv: replace the
remaining `macro_rules!` with `variants(...)`, and merge the `*_block_scaled` /
`*_mma` files into the dimensionality file. The same phases apply to every other
family.

## 5. File-granularity rules

- **One file per operation family member**, not per dtype/format/bit-width. All
  of an op's quantized variants live in the op's file as a format axis.
- A genuinely different *algorithm* for the same math gets its own file under a
  `fused/` (or sibling) subdir — e.g. Winograd vs direct conv, streaming-causal
  conv vs dense.
- A 1-kernel file (kernel + its `kernel_tests` + `kernel_benches`) is fine; merge
  trivially-small siblings only when they share a setup helper.

## 6. Migration plan (one family per PR)

Mechanics, per family:

1. `git mv` the family's files into `kernels/<family>/`; update `lib.rs`
   (`pub mod kernels;`) and `kernels/mod.rs`.
2. Merge fragmented 1-kernel files; extract shared primitives (tool 1).
3. Collapse format/bit-width/scale families onto `variants(...)` (tool 2); merge
   by op (tool 3).
4. Rename to `mt_<op>`, dropping the legacy `ffai_` prefix **and any model name**
   — name the operation / layout, not the model (§9.1); regenerate the FFAI emit
   consumer from the new inventory.
5. Gate: `cargo build` + `tile test -f <family>` green + `make fmt`. The
   *generated MSL per kernel* is unchanged — diffs are whitespace/comments and
   the inventory-name rename.

Order — by independence first (build the pattern on low-risk families), big
payoff last:

| Wave | Families | Rationale | Payoff |
|---|---|---|---|
| ✅ done | `convolution/`, `rope/` | exemplar + first wave-1 family | 24k → ~1.6k |
| 1 | `norm/`, `sampling/`, `ops/` | self-contained, mostly elementwise / few formats | small, sets the pattern |
| 2 | `gemm/`, `ssm/`, `audio/`, `vision/`, `kv_cache/` | moderate size, few cross-deps | medium |
| 3 | `sdpa/`, `moe/`, **`quant/`** | hardest axes (head-dim d64..d512; bm8/bm64×int8; the 30-format matrix) — most of the ~150k LOC | the bulk |

## 7. The `quant/` umbrella — collapsing the op × format matrix

The largest single LOC sink: the same matmul/dequant written once per format
across `block_scaled_*`, `quantized_*`, `fp_quantized_*`, `dequant_*`, the GGUF
k-quant paths, and the AURA codec stack. The target:

- **`quant::format` + `quant::codec`** are the single source of truth for the
  ~30-format `QFormat` matrix (element × scale × layout) and the host
  encode/decode/oracle. Kernels and oracles both decode through `codec` — already
  done for q8_0/q2_k/dsv4-fp8/dsv4-mxfp4 — so they can never drift.
- Each weight-bearing op (gemm, gemv, moe, conv, sdpa) carries the format as a
  `variants(FMT = […])` axis with compile-time-`if` decode branches calling the
  `codec` primitives — *not* a separate `<op>_<format>.rs` file.
- Codebook formats (`iq2_xxs`) and asymmetric super-block formats (q2_k) that
  don't fit the symmetric `element × scale` model keep their layout-specific
  decode in `codec`, still shared between kernel and oracle.
- The **`f16`-scale variants are not separate kernels** — `ScaleKind` (F32 /
  E8M0 / F16) is another axis of the format, so an op's `variants(...)` block
  carries it alongside `FMT` and the `mt_*_f16` twins fold into the base op
  (§9.3).

## 8. What this does NOT change

- Generated MSL per kernel — identical post-consolidation (same IR, same
  passes); `variants(...)` suffix templates reproduce each body exactly.
- `metaltile-core` / `-codegen` / `-runtime` / `-cli` — zero changes; this is a
  `metaltile-std` source reorg only.

(Inventory *names* do change where the `ffai_`/model prefix is dropped, §9.1 —
the FFAI emit consumer is regenerated to match, so this is not a breaking change.)

## 9. Decisions

1. **Name the operation / layout, never the model → `mt_<op>`.** Drop the legacy
   `ffai_` prefix *and* any model name; the differentiator between similar kernels
   is the layout, which the name should describe (`ffai_rope_llama` →
   `mt_rope_banded`, `dsv4_partial_rope` → `mt_partial_rope`). Model-specific usage
   notes go in a comment above the kernel. Names are **not** pinned — the FFAI emit
   consumer is regenerated from the new inventory, so renames are safe. Applied per
   family as part of its migration pass (§6).
2. **Keep `vision/` and `audio/`** as their own folders for now — do not
   distribute their ops into `convolution/` / `norm/` / `ops/`.
3. **`f16`-scale twins become a scale axis.** `ScaleKind` (F32 / E8M0 / F16) is an
   axis of the *format*, carried by `variants(...)`; `mt_*_f16` kernels collapse
   into their base op rather than existing as separate files (§7).
4. **Metal references stay on the kernels that mirror them, indefinitely.** Any
   kernel that is the same op / functionality as an upstream (MLX) reference keeps
   its optional `.with_reference(...)` comparator; kernels with no upstream
   counterpart carry none.

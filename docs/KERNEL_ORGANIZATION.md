# Kernel / Ops Organization Spec

> Status: **proposal** (2026-06-01). Target end-state for `metaltile-std`'s
> kernel source layout, file granularity, and the canonical per-kernel file
> shape. Intentionally NOT executed in one pass — migrate family-by-family to
> avoid conflicts with in-flight work (fp4/fp8/int8 coverage, etc.).

## 1. Why

The current layout splits kernels into two top-level folders:

- `crates/metaltile-std/src/ffai/` — 88 files, kernels with **no** MLX
  counterpart.
- `crates/metaltile-std/src/mlx/` — 43 files, kernels that **mirror an MLX
  metal kernel** (and carry an `mlx="…"` bench comparator).

Two structural problems:

1. **The organizing axis is wrong.** "Does MLX have this kernel?" is not a
   property of the *kernel* — it's a property of one *bench*. And it's already
   expressed per-kernel: the `#[kernel(bench(… mlx="rms{tn}", class=RowNorm))]`
   attribute + the `MetalRef` bench mechanism. The folder split duplicates that
   attribute as directory structure, and it ages badly: as we **diverge from
   and supersede MLX** (custom SDPA, GDN/SSM, AURA, turbo, fp4/fp8), the MLX
   reference is losing value, so a folder defined by it is increasingly
   meaningless. New kernels are landing under `ffai/` regardless of whether an
   MLX analog exists.

2. **Fragmentation + model-name leakage.** Within `ffai/` the same family is
   scattered across many 1-kernel files (`sdpa_bidirectional.rs`,
   `sdpa_bidirectional_d128_relpos.rs`, `sdpa_bidirectional_windowed.rs`,
   `sdpa_rel_pos_conformer.rs`; the `sdpa_decode_d{64,96,256,512}.rs` set; the
   `moe_mpp_*` set; `rope_*`; `logits_*`). Multiple kernels per file is fully
   supported (`sdpa_bidirectional.rs` holds 5; `moe.rs` holds 10) — the
   fragmentation is incremental authoring, not a DSL limit. Separately, files
   were named after *models* (`kokoro.rs`, `fishspeech_conv1d.rs`) — kernels
   are shared infrastructure and must not be filed under a consumer.

**Safe to reorganize:** FFAI's `make regenerate-kernels` emits by **kernel
function name** (file-independent — confirmed in `manifest.json`), so moving a
kernel between files never breaks the FFAI consumer as long as the `pub fn`
name is stable. Inline `#[test_kernel]`/`kernel_benches` move with the kernel;
only the ~13 `insta` MSL snapshots are path-sensitive (and none of the
fragmented families have them).

## 2. Goals / non-goals

**Goals**
- Organize by **kernel family**, not by MLX-presence.
- One **cohesive file per kernel** = the op + all its shape / precision / dtype
  / bit-width variants, with its kernel spec, bench spec, and test spec
  together.
- Make the MLX reference a purely **optional per-bench attribute** that any
  kernel may carry, and dissolve the `mlx/` folder.
- Make the **quant explosion** (affine int2–8, fp4/fp8 mx/nv, int8, aura,
  turbo) drop in cleanly so the in-flight fp4/fp8/int8 work has an obvious home.
- No model names anywhere in file names, `op=`, `subop=`, or bench `name=`.

**Non-goals**
- A big-bang move. Migrate incrementally; keep diffs family-scoped.
- Renaming `pub fn` kernels (would churn the FFAI emit + every caller).
- Changing kernel bodies/IR (this is a layout + convention change only).

## 3. Target directory layout

Replace `ffai/` + `mlx/` with **family directories** under
`crates/metaltile-std/src/kernels/` (the `kernels/` umbrella keeps the crate
root clean; `mod.rs` re-exports families). Proposed families:

```
crates/metaltile-std/src/kernels/
├── core/            # truly-elementwise primitives: binary, unary, copy,
│                    #   arange, gather/scatter, indexing, reduce, cumsum,
│                    #   fence, random, hadamard, logsumexp, arg_reduce
├── gemm/            # dense matmul: gemm, gemv, gemv_masked, patch_embed (+mma)
├── sdpa/            # ALL attention: bidirectional (+relpos/windowed/conformer),
│                    #   decode (+d64/d96/d256/d512/2pass/batched/sink), multi
│                    #   (+d256/tree-mask), prefill_mma, flash_quantized, aura_flash
├── rope/            # rope_2d, rope_llama (+many), rope_yarn
├── norm/            # rms_norm (+residual/rope/qgemv/gated), layer_norm, adain1d
├── moe/             # moe, moe_mpp (+bm8/bm64 × int8), moe_down_swiglu_accum
├── conv/            # conv2d (+mma/grouped/patch), conv3d (+mma), depthwise (+nhwc),
│                    #   conv1d (dense/dilated/transpose/causal-step), winograd
├── ssm/             # ssm, ssm_replay, gated_delta (+wy/prep/prep_chunk)
├── quant/           # see §6 — the quantization umbrella
│   ├── affine.rs            # int2/3/4/5/6/8 dequant_gemv / dequant_gather / mma
│   ├── fp_scaled.rs         # mxfp4 / nvfp4 / mxfp8 / nvfp8 (block-scaled float)
│   ├── int8.rs              # int8-specific gemm/mpp paths
│   ├── aura.rs              # AURA: encode, flash_p1/pass2, score, value, dequant_rotated
│   └── turbo.rs             # (future) turbo quant kernels
├── audio/           # mel_spectrogram (+magnitude/stft/filterbank), lstm, vocoder,
│                    #   fishspeech codec convs → folded into conv/ if generic
├── vision/          # resize_normalize (+bicubic), im2col_patch, pos_emb_2d_add,
│                    #   avg_pool2d_nhwc, transpose_th, clamp_scalar, frame_diff_luma
├── sampling/        # logits_topk / top_p / min_p / processors, sampling, fp32 reduce
├── kv_cache/        # kv_cache, kv_cache_update_many, fft
└── mod.rs           # `pub mod sdpa; pub mod rope; …`
```

Notes:
- **`core` vs `primitives`/`ops`:** `core/` holds the elementwise/data-movement
  primitives. (Pick one name; `core` reads better than `ops` since everything
  here is an op.)
- **`vision`/`audio`** are *capability* groupings for ops that are genuinely
  domain-specific (a mel DFT, a bicubic image resize). A conv that's generic
  lives in `conv/`; only truly domain-shaped kernels live here. When in doubt,
  prefer the *operation* family (`conv/`, `norm/`) over the *domain* folder.
- **`turbo` / `aura`** are quant schemes → under `quant/` (siblings of
  `affine`/`fp_scaled`), not top-level, so all quantization lives in one place.
- The `probe/` folder (debug utilities) stays as-is.

## 4. File-granularity rules — when does a kernel get its own file?

A **file = one kernel family**, where "kernel" means *the operation*, and the
file holds **every variant of that operation**: all dtypes, bit-widths,
group sizes, head dims, and shape specializations.

**One file (group together):**
- dtype permutations (`f32/f16/bf16`) — these are already a macro axis.
- bit-width / group-size permutations (int2…8; fp4/fp8) — macro axis (§6).
- head-dim / tile-size specializations of the *same* algorithm
  (`sdpa_bidirectional_d{32,64,72,80,96}` → one `sdpa/bidirectional.rs`).
- "mode" variants that share the core loop (dense / windowed / relpos /
  conformer bidirectional SDPA → still `bidirectional.rs`; the windowing and
  the rel-pos bias are small deltas on one online-softmax body).
- the naive-vs-FFT routes of one front-end (mel direct-DFT + stft+filterbank).

**Separate file (genuinely distinct op):**
- a different IR / algorithm with little shared body (decode SDPA vs prefill
  SDPA vs bidirectional SDPA → three files under `sdpa/`).
- a variant only forced apart by **hand-written per-lane layout** that the macro
  can't yet generate (today's d64-vs-d80 split exists because the 4-elem-per-
  lane packing is hand-written; see §5 — the *target* is to fold these via a
  codegen macro, at which point they collapse into one file).

**Heuristic:** if two kernels would share ≥ ~60% of their body or their entire
test/bench scaffolding, they belong in one file. The cap is readability — a
file past ~800 lines of *kernel* code (excluding tests) should split along the
algorithm boundary, not the dtype/shape boundary.

## 5. The canonical kernel file

Every kernel file is self-contained: **kernel spec + bench spec + test spec**,
in this order:

```rust
//! <op> — one-paragraph what/why, the layouts, and the ## DISPATCH INVARIANTS.
//! No model names; name the *operation* and list representative consumers
//! generically ("the bidirectional vision-tower SDPA", not "Qwen2.5-VL").

use metaltile::kernel;

#[kernel(bench(op="sdpa", subop="bidirectional", class=…, tol=…, mode=…,
               mlx="…"   /* OPTIONAL — omit when no live MLX comparison */))]
pub fn <name><T>( … ) { … }

// (further variants of the SAME op live here too)

pub mod kernel_tests   { /* naive oracle(s) + #[test_kernel(dtypes=…, tol=…)] */ }
pub mod kernel_benches { /* #[bench(name="ffai/<family>/<op>")] per shape    */ }
```

**Macro requirements (the "all permutations" ask).** The target is that a
single file expresses every permutation declaratively, rather than copy-pasted
`pub fn …_d64` / `…_d80` / `…_int4` / `…_int8`:

- **dtype axis** — already done: `#[test_kernel(dtypes=[f32,f16,bf16], tol=[…])]`.
- **bit-width / group-size axis** — for quant kernels, generate the
  `{2,3,4,5,6,8}` × `{group sizes}` cells from one body (today some are wrapped
  with an outer `macro_rules!` around the whole `#[kernel] fn` — per
  `developing.md`, never inside the body). Make this a first-class macro
  parameter (`bits=[…]`, `group_size=[…]`) so a new scheme adds one line.
- **head-dim / lane-packing axis** — the biggest gap. `sdpa_bidirectional_d80`
  vs `_d64` differ only in elements-per-lane and the ragged tail mask. A
  `head_dim=[32,64,72,80,96,128]` macro that emits the per-lane packing would
  collapse ~5 files + the windowed/relpos/conformer/sink variants into one
  `bidirectional.rs` + one `decode.rs`. **This macro is the prerequisite that
  unlocks the cleanest end-state for `sdpa/`** — until it lands, keep the
  hand-written dim variants in ONE file rather than one-per-dim.
- **optional MLX comparator** — keep `mlx="…"` as a per-`#[bench]` attribute
  (already supported via `MetalRef`). It is the *only* thing that should encode
  "an MLX version exists," and it is optional. Default (`class=GenericEmpty`,
  no `mlx=`) = a metaltile-native kernel benched against itself / a CPU oracle.

## 6. Quantization umbrella (`quant/`) — fp4 / fp8 / int8 plan

The in-flight mxfp4 / nvfp4 / mxfp8 / nvfp8 + int8 work lands here. Organize by
**quant *scheme*, one file each**, every bit-width/dtype as a macro cell:

- `quant/affine.rs` — MLX-style affine `(weight, scales, biases)` int2–8:
  `dequant_gemv`, `dequant_gather`, `qmm_mma`, `dequantize_affine`. One file,
  `bits=[2,3,4,5,6,8]` macro axis.
- `quant/fp_scaled.rs` — block-scaled float: **mxfp4, nvfp4, mxfp8, nvfp8**.
  These share a "dequant a block by its (shared exponent | fp8 scale) then
  gemv/gemm" shape; parameterize on `(mantissa_bits, exp_bits, block, scale
  kind)`. New formats = new macro cells, not new files.
- `quant/int8.rs` — int8 gemm / mpp / per-row-scale paths that don't fit the
  affine triplet.
- `quant/aura.rs` — AURA (rotation + Lloyd-Max codebook): encode, flash_p1,
  flash_pass2, score, value, dequant_rotated.
- `quant/turbo.rs` — (future) turbo quant kernels.

Rule of thumb: a **new quant *format*** (nvfp8, etc.) is a **macro cell** in the
matching scheme file; a **new quant *algorithm*** (a different packing/codebook)
is a **new file** under `quant/`.

## 7. MLX-reference policy (deprioritize)

- The `mlx/` folder **dissolves** — its kernels move into the family folders by
  operation (`mlx/gemv.rs` → `gemm/`, `mlx/rms_norm.rs` → `norm/`,
  `mlx/quantized*.rs` → `quant/`, `mlx/binary.rs` → `core/`, …).
- The MLX comparison survives as the optional `mlx="…"` bench attribute on the
  individual kernels where a side-by-side number still teaches us something
  (the few perf-sensitive primitives). Everywhere else, drop it; bench against a
  CPU oracle / our own baseline.
- **New kernels never require an MLX analog.** `class=GenericEmpty` is the
  default. We are past parity; coverage + speed of *our* kernels is the metric,
  not MLX-delta.

## 8. Naming rules

- **No model names** in file names, `op=`, `subop=`, or bench `name=`. Name the
  operation. (Precedent: `kokoro.rs`→`adain1d.rs`/`lstm.rs`,
  `fishspeech_conv1d.rs`→`conv1d_dilated_transpose.rs`,
  `op="fishspeech_conv1d"`→`op="conv1d"`.)
- `op` = family (`sdpa`, `conv`, `norm`); `subop` = the specific kernel
  (`bidirectional`, `conv1d_transpose`); bench `name` = `ffai/<family>/<subop>`.
- Consumers named generically in docs ("the Conformer acoustic encoders"), as
  *examples*, never as the kernel's identity.

## 9. Migration plan (incremental, conflict-safe)

Because other sessions are actively touching kernels, migrate in **small
family-scoped PRs**, never a big-bang move:

1. **Done already** (precedent): de-model-name `kokoro`/`fishspeech`; group
   `resize_normalize_bicubic`→`resize_normalize.rs`,
   `mel_spectrogram_magnitude`→`mel_spectrogram.rs`.
2. **Per family, one PR:** `git mv` the family's files into `kernels/<family>/`,
   merge fragmented 1-kernel files (combine `kernel_tests`/`kernel_benches`,
   dedupe shared helpers like `ramp`/`naive`/`setup`), update `mod.rs`, then
   `cargo build` + `tile test -f <family>` + `make fmt`. Kernel `pub fn` names
   unchanged → FFAI emit unaffected; coordinate with the FFAI side to run
   `make regenerate-kernels` once after each landed family.
3. **Order by independence:** start with self-contained families with no active
   work (`rope/`, `logits/`→`sampling/`, `norm/`), then `sdpa/`. **Hold `quant/`
   until the fp4/fp8/int8 session lands** — co-design the `quant/` layout with
   that work rather than reorganizing under it.
4. **The lane-packing macro (§5)** is a separate, prerequisite PR; until it
   lands, group the hand-written dim variants into one file but don't try to
   macro-collapse them.

## 10. Open questions

- `core` vs `primitives` vs `ops` as the elementwise-folder name? (Spec assumes
  `core`.)
- `kernels/` umbrella dir vs flattening families to `src/<family>/`? (Spec
  assumes `kernels/` to keep the crate root clean.)
- Do `vision/` + `audio/` earn their own folders, or do their ops distribute
  into `conv/`, `norm/`, `core/` (with only the irreducibly-domain ops —
  mel DFT, bicubic resize, frame-diff — left)? (Spec keeps thin `vision/`/
  `audio/` folders; revisit once populated.)
- Timeline to retire the last `mlx=` comparators entirely?

# Kernel / Ops Organization Spec

> Status: **proposal** (2026-06-01). Target end-state for `metaltile-std`'s
> kernel source layout, file granularity, and the canonical per-kernel file
> shape. Intentionally NOT executed in one pass — migrate family-by-family to
> avoid conflicts with in-flight work.
>
> **Coordinates with the "MetalTile CLI Subprocess Rewrite (v4)"** — that spec
> restructures the *crate/CLI architecture* (subprocess runner, `tile.toml`,
> harness, dep-graph reduction); this one restructures the *kernel files inside
> `metaltile-std`*. They are orthogonal in intent but **touch the same files**
> (`metaltile-std/src/lib.rs`, every kernel file's test/bench imports). The v4
> rewrite is the **governing** change and lands first; the kernel-family reorg
> here applies on top of the v4 end-state. See §11.

## 1. Why

The current layout splits kernels into two top-level folders:

- `crates/metaltile-std/src/ffai/` — 88 files, kernels with **no** upstream
  metal counterpart.
- `crates/metaltile-std/src/mlx/` — 43 files, kernels that historically
  **mirrored an upstream metal kernel** it could be benched against.

Two structural problems:

1. **The organizing axis is wrong.** "Does an upstream metal reference exist?"
   is not a property of the *kernel* — it's a property of one *bench*, and an
   optional one. It's already expressed per-bench: the optional `ref = MetalRef
   { … }` attribute. The folder split duplicates that optional attribute as
   directory structure, and it ages badly: as we **diverge from and supersede
   the references** (custom SDPA, GDN/SSM, AURA, turbo, fp4/fp8), they lose
   value, so a folder defined by them is increasingly meaningless. New kernels
   land under `ffai/` regardless of whether a reference exists.

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
- Organize by **kernel family**, not by whether a kernel happens to have a
  reference implementation.
- One **cohesive file per kernel** = the op + all its shape / precision / dtype
  / bit-width variants, with its kernel spec, bench spec, and test spec
  together.
- Lean into the **metal reference** already being an optional per-bench
  attribute. There is no hardcoded `mlx` / `mlx_ref` naming any more — a ref is
  just an optional metal kernel a bench compares against ("metal ref"), so files
  group by family, not by ref-presence. Dissolve the `mlx/` folder.
- Tame the **quant explosion** (affine int2–8, fp4/fp8 mx/nv, int8, aura, turbo)
  — the fp4/fp8/int8 matrix has now landed (#250) as ~240 per-op `#[kernel]`
  fns / ~39 kLOC, so the job here is **consolidating** it (§6), not finding it a
  home.
- No model names anywhere in file names, `op=`, `subop=`, or bench `name=`.

**Non-goals**
- A big-bang move. Migrate incrementally; keep diffs family-scoped.
- Renaming `pub fn` kernels (would churn the FFAI emit + every caller).
- Changing kernel bodies/IR (this is a layout + convention change only).

## 3. Target directory layout

Replace `ffai/` + `mlx/` with **family directories** under
`crates/metaltile-std/src/kernels/` (the `kernels/` umbrella keeps the crate
root clean; `mod.rs` re-exports families). This is compatible with v4's
"`metaltile-std/src/` top-level **files** = only `lib.rs` + `utils.rs`" rule:
`kernels/` is a *directory* module, and `lib.rs` becomes `pub mod kernels; pub
mod utils;` (replacing v4's interim `pub mod ffai; pub mod mlx;`). v4 also
deletes `probe/`, `bench_types.rs`, `error.rs`, `stats.rs`, `run_kernel.rs`,
`runner.rs` — so those are gone before this reorg, not relocated by it.
Proposed families:

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
├── quant/           # see §6 — format/codec/lowering infra, NOT per-op kernels.
│   ├── format.rs            # QFormat enum (~30 block-scaled formats) + params
│   ├── codec.rs             # host encode/decode + dequant oracle
│   ├── affine.rs            # standard affine (weight, scales, biases) int2–8
│   ├── aura.rs              # AURA: encode, flash_p1/pass2, score, value, dequant_rotated
│   └── turbo.rs             # (future) turbo quant kernels
│   # block-scaled op variants do NOT live here — they fold into the op's own
│   # family file as a format axis (conv2d_block_scaled → conv/conv2d.rs). §6.
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
  `affine` / `format` / `codec`), not top-level, so all quantization infra lives
  in one place.
- `probe/` is **deleted by v4**, not carried into `kernels/`.

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

When a family does split along genuine algorithm boundaries, it becomes a
**sub-folder, one file per distinct op** — keeping each file focused rather than
one oversized family file. SDPA is the canonical example:

```
sdpa/
    bidirectional.rs   # dense + windowed + relpos + conformer (one softmax body)
    decode.rs          # decode-time SDPA (+ d64/d96/d256/d512, 2-pass, sink)
    prefill.rs         # prefill MMA path
    mod.rs             # pub mod bidirectional; pub mod decode; pub mod prefill;
```

The dtype / head-dim / shape variants of *one* op still collapse into that op's
single file (per the "one file" rules above); the sub-folder split is reserved
for genuinely distinct algorithms within the family.

## 5. The canonical kernel file

Every kernel file is self-contained: **kernel spec + bench spec + test spec**,
in this order:

```rust
//! <op> — one-paragraph what/why, the layouts, and the ## DISPATCH INVARIANTS.
//! No model names; name the *operation* and list representative consumers
//! generically ("the bidirectional vision-tower SDPA", not "Qwen2.5-VL").

use metaltile::{bench, kernel, test_kernel};

#[kernel]
pub fn <name><T>( … ) { … }

// (further variants of the SAME op live here too)

pub mod kernel_tests {
    // naive CPU oracle(s) + correctness setups:
    //   #[test_kernel(dtypes = [f32, f16, bf16], tol = […])]
    //   fn test_<op>(dt: DType) -> TestSetup { … }
}

pub mod kernel_benches {
    // one #[bench] per shape; builder-style BenchSetup:
    //   #[bench(name = "ffai/<family>/<op>", dtypes = [f32, f16, bf16])]
    //   fn bench_<op>(dt: DType) -> BenchSetup {
    //       BenchSetup::new(<name>::kernel_ir_for(dt))
    //           .mode(…).buffer(…).constexpr(…).bytes_moved(…)
    //           .flops(…)                        // OPTIONAL — enables GFLOP/s
    //           .with_reference(MetalRef { … })  // OPTIONAL metal ref — omit when none
    //   }
}
```

> Post-v4 (§10), the test/bench modules import from `metaltile::harness::test`
> / `::harness::bench` (not `metaltile::test` / `::bench`), and any
> `crate::bench_types::dtype_label` becomes `crate::utils::dtype_label`.

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
- **optional metal reference** — a bench may carry an optional `ref = MetalRef
  { … }` attribute (or `BenchSetup::with_reference(…)`) naming a metal kernel to
  benchmark against side-by-side. It is just an optional comparator — no
  hardcoded `mlx` / `mlx_ref` naming. Omit it (the default) and the kernel is
  benched on its own / against a CPU oracle.

## 6. Quantization umbrella (`quant/`) — and collapsing the op × format matrix

**Current reality (post-#250).** Comprehensive precision support has landed:
`quant/format.rs` defines a `QFormat` enum of ~30 block-scaled formats (nvfp4,
mxfp4, mxfp8_e4/e5, nvfp8(+f16), the legacy float-scale fp4/fp8, mxint2–8, …)
and `quant/codec.rs` holds the host-side encode/decode + dequant oracle. That
shared infra is the **right** single source of truth and stays.

The problem is everything *downstream* of it. The format matrix is materialized
as **per-op `*_block_scaled.rs` files** — `conv2d_block_scaled.rs`,
`conv3d_mma_block_scaled.rs`, `depthwise_conv2d_block_scaled.rs`,
`dequant_gather_block_scaled.rs`, … — **15 files, ~240 `#[kernel]` fns, ~39 000
LOC**. Within one file the op body is copy-pasted once per format: e.g.
`mt_mxfp4_conv2d` and `mt_nvfp4_conv2d` are the *same* ~120-line im2col
receptive-field walk; the only difference is ~4 lines (element unpack + element
decode + block-scale read) and one extra `global` constexpr. That body then
repeats across every weight-bearing op. **Op (`conv2d`) and format (`mxfp4`) are
being multiplied into files instead of treated as orthogonal axes.**

### Target organization

Block-scaled variants are not their own family — they are the **quantized form
of an existing op**. Fold each `<op>_block_scaled.rs` back into that op's family
file (`conv2d_block_scaled.rs` → a format axis inside `conv/conv2d.rs`), and let
`quant/` hold only the **format/codec/lowering** infrastructure, not per-op
kernels:

- `quant/format.rs` — the `QFormat` enum + per-format params (element type,
  block size, scale kind, packing). (Exists.)
- `quant/codec.rs` — host encode/decode + the dequant oracle. (Exists.)
- `quant/affine.rs` — the standard affine `(weight, scales, biases)` int2–8
  triplet where it doesn't share the block-scaled path.
- `quant/aura.rs` — AURA (rotation + Lloyd-Max codebook).
- `quant/turbo.rs` — (future) turbo quant kernels.

### DSL changes to kill the duplication

The 39 kLOC is a symptom: the DSL has **no way to express "dequantize a
block-scaled element,"** so every kernel inlines the unpack-decode-scale by
hand, per format. Two changes make op and format orthogonal and collapse the
matrix to ~one body per op:

1. **A `dequant` DSL op (the big win).** Add an intrinsic
   ```
   dequant_block_scaled(weight, scales, global, col, FORMAT) -> f32
   ```
   that codegen lowers per `QFormat` — element unpack (straddle-aware), element
   decode (E2M1/E4M3/E5M2/int), and block-scale (E8M0 `exp2(s-127)` / E4M3
   micro-scale × global / FP32 / FP16). Every block-scaled op body then becomes
   **format-agnostic** — one line:
   ```rust
   acc += pix_m * dequant_block_scaled(weight, scales, global, col, FMT);
   ```
   instead of the inlined nibble-shift + `e2m1_decode` + `exp2(scale-127)`
   block. This is the single largest reduction: the per-format decode lives once
   in codegen (mirroring how `quant/codec.rs` already centralizes the *host*
   decoders), not copy-pasted into ~240 kernel bodies. `scales`/`global` are
   simply absent for formats that don't use them.

2. **Format as a first-class macro axis** (already half-proven). The integer
   formats in `conv2d_block_scaled.rs` are *already* generated from one template
   via `int_conv2d_f32!` / `int_conv2d_e8m0!` / `int_conv2d_f16!` parameterized
   on `$bits` — the float formats just never got the same treatment. Promote
   this to a declared axis on the kernel, `formats = [mxfp4, nvfp4, mxfp8_e4,
   …]`, so one `#[kernel]` body emits every format cell. Combined with (1) the
   body is written **once per op**, format-agnostic, and the macro stamps the
   variants — `conv2d_block_scaled.rs` goes from 16 hand-written fns to one body
   + a format list. A new format becomes one `QFormat` arm in the `dequant`
   lowering, automatically available to *every* op — not a new fn × every file.

3. **Fold the three scale-decode macros into the `dequant` op.** The
   `int_conv2d_{f32,e8m0,f16}!` triplication exists only because the *scale read*
   differs by format; once (1) owns scale decode, the three collapse to one.

**Rule of thumb (unchanged in spirit):** a new quant **format** is a `QFormat`
arm + a `dequant`-lowering cell, never a new kernel fn; a new quant **algorithm**
(a different packing/codebook, e.g. AURA) is a new file under `quant/`. Combined
effect: roughly **~240 fns / 39 kLOC → ~15 op bodies** + the shared `quant/`
infra. See §11 for sequencing (this rides on top of the v4 + lane-pack macro
work).

## 7. Reference-kernel policy (deprioritize)

- The `mlx/` folder **dissolves** — its kernels move into the family folders by
  operation (`mlx/gemv.rs` → `gemm/`, `mlx/rms_norm.rs` → `norm/`,
  `mlx/quantized*.rs` → `quant/`, `mlx/binary.rs` → `core/`, …).
- A side-by-side metal comparison survives as the optional `ref = MetalRef { … }`
  bench attribute on the individual kernels where it still teaches us something
  (the few perf-sensitive primitives). Everywhere else, drop it; bench against a
  CPU oracle / our own baseline. The attribute is a generic *metal reference* —
  there is no `mlx` / `mlx_ref` naming.
- **New kernels never require a reference analog.** A bench with no `ref` is the
  default. We are past parity; coverage + speed of *our* kernels is the metric.

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

Because other sessions are actively touching kernels — including the v4 CLI
rewrite (§10), which must land first — migrate in **small family-scoped PRs**,
never a big-bang move:

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
   work (`rope/`, `logits/`→`sampling/`, `norm/`), then `sdpa/`. **`quant/` is
   the largest payoff but rides on the `dequant` DSL op + format-axis macro
   (§6)** — land those first and let the matrix collapse, rather than
   reorganizing the 39 kLOC of per-op block-scaled files by hand.
4. **The lane-packing macro (§5)** is a separate, prerequisite PR; until it
   lands, group the hand-written dim variants into one file but don't try to
   macro-collapse them.

## 10. Coordination with the CLI Subprocess Rewrite (v4)

The v4 rewrite restructures the **crate/CLI architecture**; this spec
restructures the **kernel files**. They're orthogonal, but both edit
`metaltile-std` and the test/bench surface, so they must be sequenced:

- **v4 lands first; this reorg applies on top of its end-state.** Both rewrite
  `metaltile-std/src/lib.rs`'s `pub mod` list and touch every kernel file's
  test/bench imports — doing them concurrently guarantees conflicts.
- **Import-path changes this reorg must adopt** (set by v4):
  - Harness types move `metaltile::{bench,test}::*` → **`metaltile::harness::{bench,test}::*`**.
    Every kernel file's `kernel_tests` / `kernel_benches` `use` updates to the
    `harness::` paths. (The `#[bench]` / `#[test_kernel]` macros emit these
    paths, so the per-file `use` is the only manual change.)
  - **`crate::bench_types::dtype_label(dt)`** (used in some benches, e.g.
    `resize_normalize`) goes away — v4 deletes `bench_types.rs`. Move
    `dtype_label` (and any other still-needed bench helper) into `utils.rs`
    and update callers to `crate::utils::dtype_label`.
  - `probe/`, `error.rs`, `stats.rs`, `run_kernel.rs`, `runner.rs` are deleted
    by v4 — they are not kernel families and don't appear in `kernels/`.
- **`metaltile-std` is facade-only after v4** (`metaltile` + `inventory` +
  `half` + `bytemuck`). Kernel files already import only via `metaltile::…`, so
  no kernel-body change is needed; just don't reach for `-core`/`-codegen`/
  `-runtime` directly.
- **Bench protocol:** v4 routes results through `ProtocolMessage` (incl. the
  optional `ref_gbps` / `mt_pct` fields for a metal comparator). This *reinforces*
  §7 — the metal reference is a per-bench data field, not a folder; a kernel with
  no `ref` simply reports `ref_gbps: None`. Nothing in `kernels/` needs to encode
  reference-presence.
- **`lib.rs` convergence:** v4's interim `pub mod ffai; pub mod mlx; pub mod
  utils;` becomes `pub mod kernels; pub mod utils;` once this reorg lands (the
  `pub use metaltile::harness::registry::{all_benches, all_kernels, all_tests};`
  re-export from v4's `lib.rs` is unaffected — registry population is by
  `inventory`, independent of module layout).

**Net sequencing:** (1) v4 crate/CLI rewrite → (2) the lane-packing macro (§5,
prerequisite for the cleanest `sdpa/`) + the `dequant` DSL op / format-axis
macro (§6, prerequisite for collapsing the block-scaled matrix) → (3)
family-by-family kernel migration (§9), with `quant/` consolidation as the
largest single LOC reduction.

## 11. Open questions

- `core` vs `primitives` vs `ops` as the elementwise-folder name? (Spec assumes
  `core`.)
- `kernels/` umbrella dir vs flattening families to `src/<family>/`? (Spec
  assumes `kernels/` to keep the crate root clean.)
- Do `vision/` + `audio/` earn their own folders, or do their ops distribute
  into `conv/`, `norm/`, `core/` (with only the irreducibly-domain ops —
  mel DFT, bicubic resize, frame-diff — left)? (Spec keeps thin `vision/`/
  `audio/` folders; revisit once populated.)
- Which kernels keep an optional metal `ref` comparator, and for how long?

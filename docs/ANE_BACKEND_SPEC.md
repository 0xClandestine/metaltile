# Apple Neural Engine (ANE) Backend Spec

**Status:** 📋 Proposed / exploratory (design + feasibility; no implementation)
**Scope:** Evaluate and design a path for MetalTile to target the **Apple Neural
Engine**, via two routes — **(A) Core ML / MIL** (supported) and **(B) the direct
private ANE APIs** (reverse-engineered). Focus stays on **compute lowering**, not
model loading/execution (engine concern, separate project).
**References:** *The ANE Book* — https://alvaro-videla.com/ane-book (and the
companion repo https://github.com/videlalvaro/ane-book) · hollance,
*"Everything we actually know about the Apple Neural Engine"* —
https://github.com/hollance/neural-engine

---

## 1. The crux (read this first)

**The ANE is not a programmable kernel target.** Unlike the GPU (Metal/CUDA), Apple
exposes **no public ISA, no kernel-dispatch API, and no shader language** for the
ANE. hollance's FAQ has a section titled *"Can I program the ANE directly?"* — the
practical answer is *no, not in any supported way*. The ANE only runs **whole
graphs/programs compiled by Apple's own ANE compiler** from a **fixed, supported
op set**.

This is a **category mismatch with MetalTile's model.** MetalTile lowers a Rust
`#[kernel]` DSL to *one custom kernel*. The ANE cannot accept a custom kernel —
e.g. our block-scaled `qgemv` with bit-stream unpacking has **no ANE/Core ML
primitive**; you cannot hand the ANE a hand-written decode loop. So an "ANE
backend" for MetalTile can only mean:

> Lower the subset of MetalTile ops that have **ANE-supported equivalents**
> (matmul, conv, attention, norm, elementwise, in fp16 / int8 with Core ML's own
> weight compression) into a **graph** that Apple's compiler places on the ANE —
> and leave the custom fused / bit-packed kernels on the GPU.

It is a **graph emitter constrained to Apple's op set**, not a code generator.
That reframing should drive expectations: the win is offloading the *standard*
matmul/attention/conv layers of a model to the ANE's very high int8/fp16 TOPS at
low power; the win is **not** running MetalTile's bespoke kernels there.

(The roofline `ane_tops` field already in `device_specs.rs` is only a *ceiling for
comparison*, not evidence of a target path.)

## 2. Why bother, given the crux

- The ANE delivers the **highest perf/Watt** on Apple silicon for the standard
  layers and runs **concurrently** with the GPU — a model could place matmul/attn
  heavy layers on ANE while custom/unsupported ops run on the MetalTile GPU path.
- Apple's recent **blockwise / grouped weight quantization** in Core ML
  (coremltools 8, iOS 18 / macOS 15) is conceptually the *same* idea as our
  block-scaled formats — so the **format design transfers** even though the
  *kernels* do not.
- It future-proofs MetalTile as a multi-target toolchain (GPU now; ANE for the
  supported subgraph).

**Field evidence (the §8 references).** This is corroborated by working projects:
`ane-infer` reports a **fused FFN at ~3.6 TFLOPS @ ~3 W** on the ANE (≈3× the
single-op throughput, vs a Metal-GPU Q8 decode at ~32 tok/s @ ~15 W) — the
perf/Watt gap is real, **but only for fused subgraphs**. The same project measures
ANE dispatch overhead at roughly **`119 µs + bytes / 78 GB/s`** per submission, so
tiny per-op dispatches are dominated by fixed cost. **Implication:** the ANE wins
when you hand it a *large fused graph* (whole FFN / attention block), not a stream
of small ops — which reinforces the "graph, not kernel" framing in §1, and means
the lowering pass should **fuse aggressively** before emitting. `CoreML-LLM` shows
the supported-layer ceiling in production: decode at **34–52 tok/s** for 0.3–2 B
models with **92–99.9 % ANE residency**, entirely through Core ML (Route A).

## 3. Route A — Core ML / MIL (supported, recommended)

Lower the supported subgraph to **MIL** (Model Intermediate Language, Core ML's IR)
and emit an `.mlpackage`; Core ML's compiler + runtime decide ANE/GPU/CPU
placement. This is the route *The ANE Book* documents for production LLM inference.

### 3.1 Pipeline
1. **IR → MIL.** Map MetalTile ops with MIL equivalents (`linear`/`matmul`,
   `conv`, `layer_norm`/`rms_norm`-as-ops, `softmax`, `gelu`/`silu`, elementwise)
   to MIL builder ops (the `coremltools` MIL schema). Unsupported ops are refused
   (stay on GPU) — never silently lowered to a wrong-but-close op.
2. **Quantization via Core ML, not our kernels.** Express weights through Core ML
   weight compression — palettization (LUT), linear int8/int4, and **blockwise /
   grouped** quant (coremltools 8). Our `quant::format` informs *which scheme +
   block size* to request; we do **not** ship our decode kernels here (Core ML
   owns the dequant on-device).
3. **Compile + run.** `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`
   (or `.all`); Core ML compiles to `.mlmodelc` and schedules ANE-eligible ops.

### 3.2 Constraints to honor (from the references)
- **fp16-centric.** Activations are effectively fp16 on the ANE; int8 / 4-bit are
  **weight** compressions, not general int compute. (hollance: *"Is the ANE
  16-bit?"*.)
- **4D tensor convention.** The ANE/Core ML favor a `1 × channels × 1 × sequence`
  (the book's `1 × C × T × 1`) layout; non-conforming shapes force GPU/CPU
  fallback. Lowering must reshape to the ANE-friendly layout.
- **Shard ceiling.** Large models must be **split into ~250 MB shards** (the
  book) to stay ANE-resident.
- **Stateful decode / KV cache** via **`MLState`** (the book's stateful decode
  pattern) — relevant to an engine, noted here for completeness.
- **Unsupported layers force fallback.** hollance documents which Core ML layers
  the ANE rejects; the lowering pass must detect these and keep them on GPU.
- **Residency verification.** There is no API that *guarantees* ANE placement;
  confirm via `os_log`/Instruments / the book's **ANE-only residency checks**, and
  treat ANE placement as best-effort. Production projects report **92–99.9 % ANE
  residency** per chunk when conforming (CoreML-LLM).

**Field-tested constraints (skyfallsin field-guide, M3 Max).** Concrete rules the
lowering pass must encode, found empirically:
- **IOSurface min width `W ≥ 32`** for runtime buffers (the ANE's `SP=32`); pad
  narrow tensors. **Width-1 outputs may *compile* but fail at *eval*** — validate
  output shapes, don't trust compile success.
- **Proven-safe MIL ops:** `add`, `mul`, `reduce_sum`, `reshape`, `slice_by_size`,
  `softmax`, `exp`, `concat`, unary. Treat anything outside a vetted allow-list as
  GPU-fallback until proven on-device.
- **MIL `tile` is poison** — it *"poisons ANE state for the rest of the process"*
  in their tests. The emitter must **never** emit `tile`; express broadcasts
  another way.

**Production lowering patterns (CoreML-LLM, ane-infer).**
- **Fuse before emitting.** ANE perf comes from large fused graphs (whole FFN /
  attention block), not op-streams — `ane-infer`'s `mil-gen` emits **fused** FFN /
  QKV / projection MIL (`mega.rs` / `attention.rs` / `ffn.rs`). This is a working
  precedent for *exactly the Route-A emitter this spec proposes* (Rust → MIL text).
- **Multifunction prefill + decode** (separate enumerated shapes, e.g. prefill
  `T=32` / decode `T=1`) sharing weights; **`MLState` `slice_update` KV cache**;
  weight blobs spill SRAM→DRAM above ~32 MB.
- **INT8-per-chunk dominates in practice.** CoreML-LLM ships 4×INT8 chunks +
  fp16-embed sidecars; notably their **blockwise *palettization* probe was a
  negative result**. So our block-scaled formats inform the *scheme/granularity*
  request, but don't assume our exact layout maps to a Core ML win — validate, and
  expect plain Core ML linear-INT8 to be the strong baseline.
- **fp16 reduction bias.** A18/A19 ANE fp16 reductions carry a bias that bit a
  full-vocab repetition-penalty path (CoreML-LLM) — relevant if reductions are
  lowered to ANE; keep accuracy-sensitive reductions verifiable.

### 3.3 What MetalTile contributes vs delegates
- **Contributes:** the IR→MIL lowering pass, the op-eligibility analysis
  (what can go to ANE vs stays on the MetalTile GPU backend), and the
  format→Core-ML-quant mapping.
- **Delegates to Core ML:** the actual ANE compilation (`.mlmodelc`), placement,
  dequant, and execution.

## 4. Route B — direct private ANE APIs (reverse-engineered, research-grade)

Bypass the Core ML *runtime* and submit a precompiled **ANE program** to the
hardware directly, for lower per-inference latency / tighter control. **Even here
you do not write ANE kernels** — you still feed a graph through Apple's **ANE
compiler** to get a hardware executable; you only take over *submission*.

### 4.1 The private stack (names per the references / community RE)
- **`Espresso`** — Core ML's internal inference engine; produces the
  `model.espresso.*` graph inside `.mlmodelc`.
- **ANE compiler** (`ANECompiler.framework` / `aneCompilerEngine` /
  `espressocompiler`) — compiles the graph to a **`.hwx`** (hardware executable /
  "ANE program").
- **`ANEServices.framework`** + the **`aned`** user-space daemon — the runtime
  that loads a `.hwx` and submits inference requests.
- **`AppleNeuralEngine.framework` runtime client** (the symbols `ane-infer`
  drives): `_ANEClient`, `_ANEInMemoryModel` / `_ANEInMemoryModelDescriptor`,
  `_ANEIOSurfaceOutputSets` (output-buffer sets), and the eval entry points
  `doEvaluateDirectWithModel:` (which **bypasses the `aned` daemon for ~10 % faster
  eval**) plus `prepareChainingWithModel:` / `doPrepareChainingWithModel:` for
  multi-procedure chaining. The daemon XPC surface is `_ANEDaemonConnection`
  (~19 methods).
- **IOKit user client `H11ANE` / `AppleH11ANEInterface`** (`H11ANEServicesClient`,
  user-client types 1 & 4) — the kernel interface to the ANE hardware; submission
  goes through IOKit `IOConnectCall*` with ANE-specific request structs.
- **Entitlements.** Direct access is gated by private entitlements
  (`com.apple.ane.*`); third-party apps generally **cannot** obtain them, which is
  why hollance's answer to "program it directly" is effectively *no*.

**Working precedent:** `ane-infer` (a Rust project, like MetalTile) drives exactly
this stack — `mil-gen` emits fused MIL, the ANE private compiler produces the
program, and `doEvaluateDirectWithModel:` submits with inputs/outputs as
**`IOSurface`s** — reporting the ~3 W fused-FFN throughput in §2. It also documents
the headline risk verbatim: *"Private APIs will break on macOS updates."*

### 4.2 What a Route-B backend would do
1. Produce the supported subgraph (as in Route A, or as an espresso net).
2. Invoke the ANE compiler to emit a `.hwx`.
3. Load + submit via `ANEServices` / `H11ANE` IOKit, manage input/output buffers,
   poll/await completion.

### 4.3 Why this is research-only
- **Undocumented + version-fragile.** Every symbol above is private and changes
  between OS releases; there is no stability contract.
- **Entitlement-gated.** Not shippable in App Store / notarized contexts without
  Apple-private entitlements.
- **Still not custom kernels.** You inherit the ANE compiler's op set anyway, so
  the expressiveness ceiling is the same as Route A — you only save Core ML
  runtime overhead.
- **High RE cost.** The references (the ANE Book's converters/validators repo, and
  hollance's "Reverse engineering the ANE" / "How does the ANE work internally")
  are the starting points; expect significant reverse-engineering to get a stable
  submission path.

**Recommendation:** treat Route B as an *experiments/* spike to measure the
latency headroom over Core ML, **not** a production backend, unless Apple exposes
a public low-level ANE API.

## 5. Implementation phases (if pursued)

1. **Eligibility analysis (no codegen).** A pass over the IR/graph that labels each
   op ANE-eligible / GPU-only, given the Route-A op set + shape/dtype constraints.
   Output: a report of "what fraction of family X could run on ANE." Cheap, and it
   tells us whether a backend is even worth it per workload.
2. **Route A MVP.** IR→MIL emitter for a small supported subgraph (e.g. an FFN:
   `linear → silu → linear` with Core ML blockwise int8 weights); compile to
   `.mlpackage`; verify ANE residency + numerical match vs the MetalTile GPU path.
3. **Quant mapping.** Map `quant::format` choices → Core ML weight-compression
   config (palettization / linear / blockwise); validate accuracy parity.
4. **Coverage expansion.** Attention (Core ML SDPA / the book's KV-cache + MLState
   patterns), norms, conv front-ends — the op subset with ANE equivalents.
5. **(Optional, research) Route B spike.** Compile a `.hwx` and submit via the
   private stack on a dev machine; measure latency vs Core ML; document fragility.
6. **Hybrid placement policy.** Decide per-op ANE-vs-GPU at graph-build time (an
   engine concern, but the backend must expose the eligibility metadata).

## 6. Risks / open questions

- **Category mismatch (the big one).** The ANE can't run MetalTile's defining
  artifact — custom fused/quant kernels. The backend's value is narrowly the
  *standard* layers; set expectations accordingly.
- **Quant divergence.** Core ML's blockwise quant ≠ our exact `quant::codec`
  layout. We map to Core ML's scheme; bit-exact parity with our GPU kernels is
  **not** expected — only accuracy parity.
- **Placement is best-effort + opaque.** No guarantee an op lands on the ANE;
  residency must be measured, and OS updates can change placement.
- **fp16 ceiling** limits which precisions are meaningful on-ANE; the int2-6 /
  E8M0 formats have no ANE consumer (they're GPU / future-NVIDIA-oriented).
- **Route B legality/stability** as in §4.3 — private, entitlement-gated,
  unstable.
- **Maintenance:** an MIL emitter tracks coremltools/MIL schema changes; the
  private path tracks OS internals.

## 7. Recommendation

Pursue **Route A (Core ML / MIL)** if/when offloading standard layers to the ANE
becomes a priority, starting with the **eligibility-analysis pass (Phase 1)** —
it's low-cost and tells us whether the payoff justifies an MIL emitter for a given
model. Keep **Route B as a measurement spike only**. Either way, the ANE is a
*graph-offload* target for the supported subset, complementary to — not a
replacement for — MetalTile's GPU kernel generation.

## 8. References
- *The ANE Book* (Alvaro Videla) — https://alvaro-videla.com/ane-book ·
  companion code: https://github.com/videlalvaro/ane-book — production Core ML
  inference: GGUF→Core ML, INT8/INT4 quant, ~250 MB shards, the `1×C×T×1` shape
  convention, `MLState` KV cache + stateful decode, ANE-residency validation.
- hollance, *neural-engine* — https://github.com/hollance/neural-engine —
  community reverse-engineering: device/generation support, `computeUnits`,
  ANE precision, unsupported layers + fallback, *"Can I program the ANE
  directly?"*, "How does the ANE work internally", "Reverse engineering the ANE".
- skyfallsin, *apple-neural-engine-field-guide* —
  https://github.com/skyfallsin/apple-neural-engine-field-guide — empirical
  constraints (M3 Max): `MIL → espresso net → .hwx`; IOSurface `W ≥ 32`; width-1
  outputs fail at eval; the proven-safe MIL op set; `MIL tile` poisons ANE state;
  the `~119 µs + bytes/78 GB/s` dispatch model.
- thebasedcapital, *ane-infer* — https://github.com/thebasedcapital/ane-infer —
  a Rust ANE/GPU LLM runtime via the **direct private path**: `mil-gen` fused-MIL
  emitter, `_ANEClient` / `doEvaluateDirectWithModel:` / chaining,
  `_ANEDaemonConnection`, `H11ANE` IOKit; fused-FFN ~3.6 TFLOPS @ ~3 W; the
  closest existing analog to a MetalTile ANE backend.
- john-rocky, *CoreML-LLM* — https://github.com/john-rocky/CoreML-LLM —
  production **Route A** (Core ML) LLM inference: coremltools 8+, INT8 chunking
  (~250 MB) + fp16-embed sidecars, multifunction prefill/decode, `MLState`
  `slice_update` KV cache, 92–99.9 % ANE residency, 34–52 tok/s on A19 Pro;
  blockwise-palettization noted as a negative result.

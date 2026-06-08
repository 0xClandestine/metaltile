# CUDA Backend — Implementation Scope (working doc)

> **STATUS: ✅ COMPLETE — 100%.** The CUDA backend runs the **full** registered
> `#[test_kernel]` corpus on the GX10 (sm_121): **PASS=4164 / 4164 (100%),
> MISMATCH=0, UNSUPPORTED=0, ERROR=0** — every kernel generates, compiles, and
> runs **bit-accurately** vs the same CPU oracle the Metal harness uses (the
> known-hard list is empty; no masking). Both cooperative-matmul paths
> (simdgroup_matrix + CoopTile/`mpp::matmul2d`, software-emulated) are bit-exact —
> quantized qmm/moe/gather + NAX + flash-attention; plus reductions, Grid3D,
> control flow, locals, activations, atomics, gemv/qgemv, Strided + multi-dim
> indexing, quant decode (incl. NVFP4 for Nemotron). Precision matched to the IEEE
> oracle via NVRTC `--fmad=false` + precise device math. The MPP/NAX cooperative
> layout was cracked from the Apple MPP headers (view stride = inner extent `ei`),
> and the flash-attention accumulator-sharing (`*_acc` setups share C) closed the
> last cases. Metal path untouched; macOS build + 173 codegen + 4 hardware-smoke
> tests green. Software-emulated MMA is correctness-first; the production perf
> path on this hardware is inline-PTX `mma.sync` (a later retune).

Companion to `CUDA_BACKEND_SPEC.md` (Eric, PR #262). The spec is the *design*;
this is the *engineering plan* — concrete seams, file targets, dev env, sequencing.

**Worktree:** `/Users/tom/dev/metaltile-cuda` on branch `feature/cuda-backend`
(off `5b60388`). Metal path stays the zero-config macOS default; CUDA gated.

**Dev/test hardware:** local-network **DGX Spark (GB10, Grace-Blackwell, sm_121)**.
Gives a real Blackwell device → Phase 4 (`tcgen05` scaled-MMA) is testable here,
not just theoretical. A remote build loop runs cargo on-box (see §5).

---

## 0. Progress

- [x] **GX10 wired** (§5) — ssh alias, CUDA 13.0/sm_121 confirmed, remote build loop, full workspace green on aarch64.
- [x] **Cooperative-kernel inventory** — **~52 files** in `metaltile-std` use `mpp::`/`coop_tile_`/`simdgroup`/`nax` (mlx/steel/ffai). Sizes Phase 5; most are quant/tile variants of a few primitives (simdgroup-MMA, MPP, NAX, coop_tile).
- [x] **Op surface mapped** — `Op` has ~65 emit arms (`emit_block.rs`). Categorized: ~45 pure arith/mem/control (port mechanically), ~13 simd/reduce (warp-shuffle, lucky 32-match), ~10 cooperative MMA (Phase 3 re-tile), 1 `InlineMsl` (Phase 5). Decode intrinsics live in `UnaryOpKind` → pure arithmetic, port verbatim.
- [x] **Phase 0 seam landed** — `codegen/src/backend.rs` (`Target`, `CodegenBackend` trait, `TargetProfile` encoding §4.2 as data + the metal/cuda intrinsic maps), `cuda/mod.rs` (`CudaGenerator` stub), `MslGenerator` impls the trait. Non-breaking; Metal unchanged.
- [x] **Phase 1 — smoke kernel GREEN end-to-end on GX10 (sm_121).** `vector_add` (f32): IR → `CudaGenerator` (CUDA C++ op-walker for the elementwise subset) → NVRTC → module → `cuLaunchKernel` → readback → CPU oracle, **max|Δ| = 0.0 (bit-exact)**. Codegen: `cuda/mod.rs` op-walker (ProgramId/Const/Load/Store/BinOp/UnaryOp/Cast/Fma, dtype→CUDA types, binop/intrinsic maps). Runtime: `runtime/src/device/cuda/{mod,ffi}.rs` — `CudaDevice` (hand-rolled `libcuda`+`libnvrtc` FFI: ctx, NVRTC compile w/ `--gpu-architecture=compute_121` + `--include-path`, alloc/upload/launch/download), `cuda` cargo feature + `build.rs` link config, integration test `tests/cuda_smoke.rs`. Plus `scale_add_exp` (constexpr scalar + `__expf`) max|Δ|=1.2e-7. **macOS Metal build unaffected (cuda off by default, no warnings).**
- [x] **Phase 2 (in progress) — reduction family + first real LLM kernel GREEN on GX10.**
  - `row_reduce_sum` (Reduction mode): `StrideReduce` (per-thread grid-stride accum) + `Reduce` (warp `__shfl_down_sync` + `__shared__` tree + `__syncthreads`) — CUDA analog of `msl::reduce`'s two-level `simd_sum`. **max|Δ| = 4.1e-8**. The 32-lane warp ≙ simdgroup match ported cleanly.
  - **`rms_norm`** (out = x·rsqrt(mean(x²)+eps)·w), 1-elem/thread: square=`Mul(x,x)`, threadgroup `Reduce`, `Cast` u32→f32, `Div`, `rsqrtf`, weighted `Store`. **max|Δ| = 3.6e-7** over 16384 elems — **needed ZERO new codegen**: the Phase-2 reduction emitter generalized straight to a fused LLM kernel. Validates the op-walker design.
  - Mode-aware signature (no `_n_elems` for Reduction) + reduction preamble. **173 codegen + 4 hardware kernels green.**

**Strategy decided (was §6 open q):** `emit.rs` is file/manifest/Swift-wrapper/metallib glue — the real op-walker is `msl/mod.rs::emit_kernel` + `emit_block.rs` (61 KB). Do **not** fork it up front. Stand up a minimal CUDA op-walker for the Phase-1 subset, grow per phase, extract a shared walker only once both emitters exist and the common shape is empirical. Premature extraction = wrong abstraction over a hot path.

---

## 0c. Hardware reality + research findings (deep-research, cited)

**GB10 / sm_121 is CONSUMER Blackwell = `mma.sync`-class, NOT `tcgen05`.** TMEM,
`tcgen05.mma`, UMMA, 2-SM cooperative MMA are **sm_100 (datacenter) only**. GB10
gets FP4/FP6 + block-scaling but **via warp-level `mma.sync`**. → **Spec §4.3/§4.5
Phase-4 `tcgen05` plan does NOT apply to our box.** The real Blackwell MMA path
here is **inline-PTX `mma.sync`** (documented per-lane fragment layout), not
`wmma` (fragment interior is *unspecified* → unusable for bit-accuracy) and not
`tcgen05`. cuda-oxide's tcgen05 intrinsics are likewise sm_100-oriented.

Top silent-correctness traps to guard (apply as we extend):
1. **`__shfl_*_sync` mask** — blanket `0xffffffff` on a partial/divergent warp is
   UB (reads undefined, not 0). Likely root cause of the hadamard / col-seg-reduce
   KNOWN_HARD mismatches. Fix = derive mask from the active set / pad to full warps.
2. **`cuLaunchKernel` overhang** — Metal `dispatchThreads` is exact; CUDA is
   blocks×threads → needs bounds guards (we already emit them for Elementwise;
   Grid3D/Reduction rely on Metal-parity launch geometry).
3. **fast-math / `--fmad`** — `__expf` etc. + default FMA fusion diverge from an
   IEEE CPU oracle. We pass within per-dtype tol; for strict bit-accuracy add
   NVRTC `--fmad=false` and avoid `__`-intrinsics.
4. **fp8/fp4** — CUDA 13 has native `__nv_fp8_*`/`__nv_fp4_*` + `__nv_cvt_*`
   (faster/safer than our manual decode) — swap only after matching oracle rounding.
5. **Shared mem** — GB10 static cap **48 KB** (per-block max 99 KB via *dynamic*
   smem + `cuFuncSetAttribute` opt-in). Our per-warp simdgroup tiles hit this on
   `sdpa_prefill_mma` (many tiles) → dynamic-smem follow-up.

Full report: `tasks/` deep-research (mma.sync shapes for sm_121, NVRTC bundled
headers, barrier-scope, atomics ordering).

---

## 0b. Status snapshot (latest)

**CUDA backend runs the real registered `#[test_kernel]` corpus on GX10 (sm_121):
`PASS=3699, MISMATCH=0, ERROR=0`** (~**89%** of the full corpus) — bit-accurate
against the same CPU oracle the Metal harness uses (`tests/cuda_kernel_corpus.rs`).
Now includes **BOTH cooperative-matmul paths**: `simdgroup_matrix` (software, Apple's
exact 8×8 lane layout) AND `CoopTile` (`mpp::matmul2d` software GEMM + dynamic
shared memory) — most quantized qmm/gather/moe/conv MMA kernels bit-accurate. Covered:
- Modes: Elementwise, Reduction, Grid3D
- Ops: const/binop/unary/cast/fma/select; load/store/gather/scatter; program-id;
  reduce + stride-reduce (incl. **transform chain + secondary_src = gemv/qgemv**);
  control flow (**Loop/If** with nested-block recursion); mutable **locals**;
  **activations** (silu/gelu/relu/sigmoid/tanh); **quant decode** (e2m1/e4m3/e5m2/
  int8 — all block-scaled incl. NVFP4); threadgroup/stack **shared memory** +
  warp shuffles (broadcast/xor) + barriers; **atomics**; cross-kernel **inlining**.
- Runtime: `CudaDevice` (NVRTC + Driver API), generic `run_kernel` dispatch,
  `cuda` feature, `build.rs`.

**Remaining ~465 not-passing:** `KNOWN_HARD=456` (generate+run but need bespoke
semantics) — dominated by the **`_nax` Metal-4 neural-accelerator** cooperative
ops (a distinct tensor path, not `mpp::matmul2d`) + specific **MPP block-tiled**
qmm variants (`qmm_mma_mpp`, `*_bm8/bm16_mpp` — while plain/`bm64` qmm pass);
plus the earlier small set (mhc-sinkhorn SSA-shadow, col/seg-reduce axis, hadamard
warp-xor-mask). Plus `UNSUPPORTED=9` (SimdScan 6, Strided params 3). Both
cooperative-matmul subsystems (`Simdgroup*`-matrix AND `CoopTile` mpp::matmul2d)
are **implemented + bit-accurate** for the standard paths; NAX is the last
genuinely-distinct hardware op needing its own lowering.

---

## 1. The two seams (grounded in current code)

Today there is **no backend abstraction** — both layers are concrete Metal:

- Codegen entry: `metaltile-codegen/src/lib.rs` re-exports `MslGenerator`,
  `generator_for_mode`, `kernel_uses_n_simd`. Emission lives in `emit.rs` (28KB) +
  `msl/` (`mod.rs` 74KB, `emit_block.rs` 61KB, `preamble.rs`, `features.rs`,
  `matmul.rs`, `reduce.rs`, `fused.rs`, `helpers.rs`, `config.rs`).
- Runtime entry: `metaltile-runtime/src/lib.rs` → `Context`, concrete
  `device/metal_device.rs` (`MetalDevice`), `device/mod.rs` is `#[cfg(macos)]`.

The passes (`passes/`) and the IR (`metaltile-core`) and quant (`metaltile-std::quant`)
are **already backend-neutral** — do not touch them.

### Seam A — Codegen (`CodegenBackend` trait)

`MslGenerator` becomes `Msl` impl; add `Cuda` impl. Parameterize IR→text by a
`TargetProfile`:

```rust
// metaltile-codegen/src/backend.rs (new)
pub trait CodegenBackend {
    fn emit_kernel(&self, k: &Kernel, sched: &TileSchedule) -> Result<String>;
    fn preamble(&self, features: &FeatureSet) -> String;   // decode helpers, math intrinsics
    fn profile(&self) -> &TargetProfile;
}

pub struct TargetProfile {
    pub lane_width: u32,          // 32 both (the lucky match)
    pub shared_mem_kw: &'static str,      // "threadgroup" | "__shared__"
    pub block_idx: fn(u8)->String,        // tg-pos-in-grid | blockIdx.{x,y,z}
    pub intrinsics: IntrinsicMap,         // exp2->{metal::exp2|exp2f}, rsqrt, etc.
    pub mma: MmaStrategy,                 // Simdgroup8x8 | Wmma16x16x16 | Cutlass | Tcgen05
}
```

The `msl/` emitter is large because it inlines a lot of Metal-specific string
building. Strategy: **extract the backend-neutral structure** (op walking,
SSA-value naming, control flow) into a shared emitter that calls `TargetProfile`
hooks, rather than fork 74KB of `mod.rs`. First pass can be cruder — a parallel
`cuda/` emitter that shares only `TargetProfile` — and converge later. Decide
extraction-vs-fork after reading `emit.rs` + `msl/mod.rs` op dispatch (§4 task 0).

### Seam B — Runtime (`Device` trait)

`MetalDevice` public surface today (`metal_device.rs`): `create`, `device`,
`queue`, `command_buffer`, `get_pso`, `get_msl`, `acquire_shared`,
`acquire_private`. Abstract the dispatch-relevant subset:

```rust
// metaltile-runtime/src/device/mod.rs (promote to trait)
pub trait Device {
    type Buffer;
    fn create() -> Result<Option<Self>> where Self: Sized;
    fn compile_kernel(&self, src: &str, name: &str) -> Result<CompiledKernel>; // PSO | CUmodule+fn
    fn alloc(&self, len: usize) -> Result<Self::Buffer>;
    fn upload(&self, b: &Self::Buffer, data: &[u8]);
    fn dispatch(&self, k: &CompiledKernel, grid: [u32;3], block: [u32;3], args: &[Arg]) -> Result<()>;
    fn readback(&self, b: &Self::Buffer, out: &mut [u8]);
}
```

`MetalDevice` → impl. `CudaDevice` → new (`device/cuda_device.rs`,
`#[cfg(feature = "cuda")]`). `Context` (context.rs, 37KB) currently bakes in
MTL types — needs to be made generic over `Device` or split. **This is the
heaviest refactor; budget for it.**

`cuda-oxide` `cuda-core` (`CudaContext`/`CudaStream`/`DeviceBuffer<T>`) is the
recommended host-runtime dep — vet license (`cuda-bindings` is NVIDIA Software
License, not Apache) before adding. Fallback: hand-rolled `cuModule*`/`cuLaunch*`
FFI via NVRTC.

---

## 2. Phases → file targets → exit criteria

Mirrors spec §5; each independently shippable, verified by existing `#[test_kernel]`
CPU-oracle harness (backend-agnostic).

| # | Phase | New/changed files | Exit criteria |
|---|---|---|---|
| 0 | **Seam refactor** ✅ | `codegen/src/backend.rs`, trait + `TargetProfile` | DONE — Metal green via trait; non-breaking. |
| 1 | **Smoke kernel** ✅ | `codegen/src/cuda/mod.rs` walker, `runtime/src/device/cuda/{mod,ffi}.rs`, `build.rs`, `tests/cuda_smoke.rs` | DONE — `vector_add` f32 bit-exact on GX10 sm_121 via NVRTC compile+launch. (Standalone test, not yet `--target` in the `#[test_kernel]` harness — that's Phase 6.) |
| 2 | **Elementwise + reduction** 🟡 | cuda emitter: `Grid3D`, `Reduction` (warp-shuffle `__shfl_down_sync` + shared-mem tree) | IN PROGRESS — `row_reduce_sum` GREEN (max\|Δ\|=4.1e-8). TODO: Grid3D mode, run backend-neutral passes, multi-dim indexing, StrideReduce transform/secondary (rms-norm, qgemv), dequant/gather/flash-scalar |
| 3 | **MMA (software-dequant block-scaled)** | cuda `matmul.rs` analog: `wmma`/`mma.sync` 16×16×16 re-tiling | qmm-MMA, patch-embed-MMA, conv-MMA green; correctness before perf |
| 4 | **Blackwell scaled-MMA** | `tcgen05` path, `MmaStrategy::Tcgen05`, gated `cc>=sm_100` | mx*/mxint* hardware path matches software oracle **bit-for-bit on Spark** |
| 5 | **Cooperative reimpl** | CUTLASS shims for `Op::InlineMsl` mpp::/NAX kernels | MPP/NAX families green (don't auto-port) |
| 6 | **CLI + CI** | `--target cuda` in build/test/bench; Linux+CUDA CI lane; `device_specs.rs` for Spark | roofline columns populated; CI green |

**Pure-DSL kernels** (Phases 1–4) are the bulk and port through the emitter.
**Cooperative kernels** (`Op::InlineMsl` + `mpp::`/`coop_tile_*`) are Metal-only
raw-MSL escape hatches → Phase 5 manual CUTLASS reimpl. Inventory these first
(grep `InlineMsl`/`mpp::`/`coop_tile_` in `metaltile-std`) to size Phase 5.

---

## 3. Quant payoff (why CUDA is the tractable 2nd backend)

`quant::{codec,format}` is pure host Rust, **reused unchanged**. Decode intrinsics
(`e2m1_decode`/`e4m3_decode`/`e5m2_decode`/`int8_decode`/E8M0 `exp2`/sub-byte
extract) are pure arithmetic → port to CUDA `__device__` preamble helpers verbatim
(mirror `msl/features.rs`). Two consumption paths, selected by `TargetProfile` +
detected compute capability:

- **Software-decode** (all NVIDIA, Ampere/Hopper/Ada): dequant-into-`__shared__`
  + `wmma`/CUTLASS MMA. Phase 2/3.
- **Hardware block-scaling** (Blackwell sm_100+): packed codes + E8M0 scale
  buffers feed `tcgen05.mma` scaled tensor cores with little/no repacking — the
  PR-#2 E8M0/block-32 layout *is* the native Blackwell microscaling layout.
  Phase 4. **Spark validates this for real.**

Formats already in tree (`metaltile-std/src/quant/{codec,format,mod}.rs` +
`ffai/*_block_scaled.rs`): mxfp4, mxfp8, nvfp4, mxint8, mxint4, E8M0 scales.

---

## 4. Immediate next actions

0. **Read the op-dispatch core** — `codegen/src/emit.rs` + `msl/mod.rs` op walker.
   Decide extract-shared-emitter vs parallel-cuda-emitter. (blocks Phase 0 design)
1. **Inventory cooperative kernels** — grep `InlineMsl`/`mpp::`/`coop_tile_` in
   `metaltile-std` → Phase 5 size + list.
2. **Wire up Spark** (§5) — SSH, CUDA toolkit version, compute cap, NVRTC version.
3. **Vet `cuda-oxide`** — license (`cuda-bindings`), alpha/Linux-only constraints,
   whether `cuda-core` alone (host runtime) is worth adopting vs hand-rolled FFI.
4. **Draft `TargetProfile` + `CodegenBackend`/`Device` traits** as a non-breaking
   Phase-0 PR (Metal-only, proves the seam).

---

## 5. Dev/validation hardware (GB10)

**Hardware:** an NVIDIA GB10 (Grace-Blackwell) DGX-Spark-class box on the dev network. Confirmed specs:

| | |
|---|---|
| GPU | **NVIDIA GB10 (Grace-Blackwell), compute cap 12.1 → sm_121 (Blackwell)** |
| Driver | 580.159.03 |
| CUDA Toolkit | **13.0** (`/usr/local/cuda`), `nvcc` V13.0.88 |
| NVRTC | `libnvrtc.so.13.0.88` |
| OS / arch | Ubuntu 24.04.4 LTS, **aarch64** |
| CPU / mem | 20 cores, 121 GiB unified |
| Rust | 1.95.0 stable + cargo (builds metaltile on-box) |
| cmake / clang | cmake 3.28.3; **no clang** (NVRTC path fine; cuda-oxide path would need it) |

**sm_121 = real Blackwell → Phase 4 (`tcgen05` scaled-MMA) is testable here.**

**Dev loop (proven):** a `rsync` + remote `cargo` loop syncs the worktree
(excl. target/.git/.cache) → runs `cargo` on GX10 with CUDA in PATH. Full
workspace builds clean on aarch64 Linux in ~41s: **codegen + runtime compile
fine** because Metal deps are `cfg(target_os="macos")`-gated and the MSL emitter
is pure String-building (no FFI). So the entire seam refactor + CUDA emitter can
be developed and built on-box; Mac stays Metal-only for the Metal path.

Done:
- [x] CUDA 13.0 / NVRTC 13.0.88 / driver 580 / sm_121 confirmed.
- [x] `scripts/gx10-sync.sh` remote build loop; full workspace green on aarch64.

TODO:
- [ ] CI for Phase 6: self-hosted GX10 runner vs cloud Linux+CUDA.
- [ ] Note CUDA **13.0** (newer than spec assumed) — confirm `tcgen05` + NVRTC API
      surface for Phase 4; minor version-skew gating (spec §6).
- [ ] `aarch64` host — verify any x86 assumptions in build scripts (none hit so far).

---

## 6. Open questions for Eric

- **Emitter strategy:** extract a shared backend-neutral emitter from `msl/mod.rs`
  (74KB), or stand up a parallel `cuda/` emitter sharing only `TargetProfile`?
  Affects Phase 0 size a lot.
- **`Context` genericization:** make `Context` generic over `Device`, or split into
  `MetalContext`/`CudaContext` with a thin shared trait? (context.rs is 37KB, MTL-coupled)
- **`cuda-oxide` as host dep:** acceptable given `cuda-bindings` NVIDIA license +
  alpha status? Or hand-roll Driver-API FFI for stability?
- **MMA re-tiling:** start `wmma` (portable 16×16×16) or jump to CUTLASS for the
  MMA families? CUTLASS also needed for Phase 5 cooperative reimpl — adopt early?
- **Scope of Phase 1 target list** — which single kernel is the canonical smoke test?

# AMD / ROCm Backend Spec

**Status:** 📋 Proposed (design only; no implementation yet)
**Scope:** Add an AMD GPU backend so MetalTile's `#[kernel]` DSL / IR lowers to
**HIP / ROCm** (AMD GPUs), reusing as much of the CUDA backend as possible.
**Out of scope:** model loading, graph execution, checkpoint readers — MetalTile
is an optimized-kernel generator, not an inference engine.

> **Read `CUDA_BACKEND_SPEC.md` first.** This spec is deliberately a *delta* on
> it: the structure (a codegen backend + a runtime backend behind the same seam,
> reusing the IR / `#[kernel]` macro / `quant::{codec,format}` layer) is identical.
> This document covers only what's **different for AMD**.

---

## 1. Motivation & the key insight

AMD GPUs are custom-kernel-programmable (HIP C++ → GCN/RDNA/CDNA ISA via the LLVM
AMDGPU backend, or SPIR-V), so the same per-kernel model as CUDA applies. The
*insight that makes this cheap*: **HIP is a CUDA-portable C++ dialect** — AMD ships
`HIPIFY` precisely to mechanically convert CUDA → HIP, and the constructs MetalTile
emits (`__global__`, `blockIdx`/`threadIdx`, `__shared__`, warp shuffles, math
intrinsics) are near-identical between the two. So:

> Once `CUDA_BACKEND_SPEC.md §4.1`'s backend seam exists, the AMD backend is
> largely **a `TargetProfile` (HIP dialect + wavefront width + matrix-core
> strategy) on the shared C++ emitter, plus a HIP runtime `Device` impl.**

And the PR-#2 precision payoff repeats: **AMD CDNA4 (Instinct MI350/MI355X, 2025)
has hardware OCP-microscaling tensor cores (MXFP4 / MXFP6 / MXFP8)** — so the
`mx*` / `mxint*` E8M0-block-32 formats map onto AMD matrix cores just as they do
onto NVIDIA Blackwell. The format work transfers to a *third* hardware target.

## 2. Goals / Non-goals

**Goals**
- A `hip` codegen target that reuses the CUDA C++ emitter via a `TargetProfile`,
  compiled with **hipRTC** (runtime, analogous to NVRTC) or offline `hipcc`.
- A `hip` runtime `Device` impl (HIP runtime API — `hipModuleLoad`,
  `hipLaunchKernel`, `hipMalloc`; 1:1 with the CUDA Driver API).
- Reuse the IR, the `#[kernel]` macro, and the **entire `quant::{codec,format}`
  layer unchanged**.
- Correct handling of the **wavefront-size split (32 vs 64)** — the one structural
  difference from NVIDIA/Apple's fixed 32-lane group.
- Map block-scaled formats onto AMD matrix cores (MFMA / WMMA), with the software
  dequant path as the universal fallback.

**Non-goals**
- Model execution / weight loading (engine concern).
- Day-one parity for the `mpp::`/`InlineMsl` cooperative kernels (reimplement via
  rocWMMA / Composable Kernel — same situation as CUDA's CUTLASS reimpl).
- Bit-exact match to the Metal/CUDA outputs — accuracy-parity vs the CPU oracle.

## 3. What's shared with the CUDA backend (most of it)

Everything in `CUDA_BACKEND_SPEC.md §3–§4.2` applies. The DSL→C++ op mapping is the
same; HIP renames a handful of host calls and intrinsics:

| Concept | CUDA | HIP / ROCm |
|---|---|---|
| host runtime | CUDA Driver API | **HIP runtime** (`hipModuleLoad`, `hipLaunchKernel`, `hipMalloc`, `hipMemcpy`, streams) — direct rename |
| runtime compile | NVRTC | **hipRTC** |
| offline compile | `nvcc` | `hipcc` (clang/LLVM AMDGPU) |
| kernel qualifier / ids / shared mem | `__global__` / `blockIdx`/`threadIdx` / `__shared__` | **identical** |
| math intrinsics | `__expf`, `exp2f`, `rsqrtf` | identical / `__ocml_*` |
| MMA library (CUTLASS) | CUTLASS | **Composable Kernel (CK)** / **rocWMMA** |
| dtypes | `__half`, `__nv_bfloat16` | `__half` (`_Float16`), `__hip_bfloat16` |

So the emitter is the CUDA C++ emitter parameterized by a `TargetProfile { dialect:
Hip, … }`; only the small dialect deltas above differ.

## 4. What's genuinely different for AMD

### 4.1 Wavefront size 32 **or** 64 — the main hazard

MetalTile kernels assume a **32-lane simdgroup** (Metal) ≙ 32-lane warp (NVIDIA).
On AMD this is **not fixed**:
- **RDNA (gfx10/11/12, consumer + some pro)** runs **wave32** for compute — maps
  cleanly to the existing 32-lane reductions/shuffles.
- **CDNA / GCN (Instinct MI-series, gfx9/9xa/94x/95x)** is **wave64** — twice the
  lane count. Any kernel that hard-codes 32 (lane masks, `simd_sum`,
  reduce-tree widths, MMA fragment mapping) must be re-parameterized.

**Mitigation:** make wavefront width a `TargetProfile` constant the emitter and the
reduction lowering read (`WARP = profile.wave_size`); emit `__shfl`-based
reductions sized to it; where the kernel logic truly needs 32, target RDNA wave32
or split a wave64 into two 32-lane halves. The geometry-audit discipline (no
silent geometry change) from the CUDA spec carries over and is *more* important
here because of this split.

### 4.2 Matrix cores: MFMA (CDNA) vs WMMA (RDNA3+)

AMD's tensor-core analog differs by arch:
- **CDNA matrix cores → MFMA** (`__builtin_amdgcn_mfma_*` / rocWMMA / CK). MI300
  (gfx942/CDNA3) adds **FP8 (OCP E4M3/E5M2) MFMA**; **MI350 (gfx950/CDNA4) adds
  MXFP4/6/8 microscaling MFMA**.
- **RDNA3/RDNA4 → WMMA** (`__builtin_amdgcn_wmma_*`); RDNA4 adds FP8.

These have **different fragment shapes** from both Metal `simdgroup_matrix` (8×8)
and NVIDIA `wmma` (16×16×16), so the MMA kernels need AMD-specific tiling — same
"re-tile per backend" caveat as CUDA, with one more shape family. Use **rocWMMA**
or **Composable Kernel** as the high-level path (the CUTLASS analog).

### 4.3 Block-scaled formats on AMD

- **Software-decode path (all AMD GPUs):** the `quant::codec` decode ports to HIP
  device functions verbatim (it's arithmetic), feeding dequant-into-LDS + MFMA/
  WMMA — mirroring the CUDA software path.
- **Hardware microscaling (CDNA4 / MI350+):** `mxfp4`/`mxfp8` and `mxint*` map onto
  the OCP-microscaling MFMA path (E8M0 block-32 scale operands) — the AMD analog of
  Blackwell `tcgen05`. The host packer is reused unchanged; only kernel-side
  consumption (software dequant vs hardware scaled-MFMA) is selected by
  `TargetProfile` + the detected `gfx` arch.
- **FP8 (MI300 / RDNA4):** `nvfp8` / `fp8_*` map to native FP8 MFMA/WMMA even
  pre-CDNA4.

### 4.4 Toolchain & ISA targets

- **Compile:** hipRTC (runtime) → code object, or `hipcc`/clang offline. The LLVM
  AMDGPU backend emits **GCN/RDNA/CDNA ISA**; target the right `gfx` (e.g. gfx90a
  MI200, gfx942 MI300, gfx950 MI350, gfx1100 RDNA3, gfx120x RDNA4).
- **Libraries:** rocBLAS / rocWMMA / Composable Kernel (CUTLASS-class), rocPRIM
  (CUB-class).
- **Alt portable path:** SPIR-V / Vulkan compute is a more portable but
  lower-control option that also covers AMD; out of scope here (HIP/ROCm gives the
  perf + matrix-core access this spec targets).

## 5. Implementation phases

Mirror `CUDA_BACKEND_SPEC.md §5`, sequenced to retire the AMD-specific risk early:

1. **Seam reuse + HIP smoke kernel.** Add the `Hip` `TargetProfile` to the shared
   C++ emitter; `HipDevice` over the HIP runtime + hipRTC; one elementwise
   `#[test_kernel]` green on an AMD GPU (RDNA wave32 first — simplest).
2. **Wavefront-64 correctness.** Bring reductions/shuffles up on **CDNA wave64**;
   parameterize `WARP`. This is the gating risk — do it before breadth.
3. **Elementwise + reduction families.** dequant, qgemv, rms-norm, gather, conv,
   flash (scalar) — the bulk of pure-DSL kernels, both wave32 and wave64.
4. **Matrix cores.** MFMA (CDNA) + WMMA (RDNA3+) re-tiling via rocWMMA/CK for the
   MMA kernels; software-dequant block-scaled.
5. **CDNA4 microscaling MFMA.** Hardware `mx*`/`mxint*` path on gfx950+, feature-gated.
6. **Cooperative reimpl + CLI/CI.** rocWMMA/CK equivalents of the MPP/NAX kernels;
   `--target hip` across build/test/bench; an AMD CI lane; device-spec rows
   (peak BW / TFLOPs / matrix-core TOPS) for the roofline columns.

## 6. Risks / open questions

- **Wavefront 32/64 (the big one).** §4.1 — pervasive 32-lane assumptions; budget
  the wave64 adaptation as a first-class phase, not a footnote.
- **Rust/ROCm ecosystem maturity.** HIP Rust bindings are thinner than CUDA's
  (no `cuda-oxide` equivalent). Expect raw FFI over hipRTC + the HIP runtime, or
  the experimental rustc AMDGPU LLVM target. More glue than the CUDA backend.
- **ROCm platform support is narrower.** Linux-centric; official support skews to
  Instinct (CDNA) + select RDNA pro cards; consumer-RDNA ROCm support is uneven by
  version/OS. CI needs real AMD hardware (CDNA *and* RDNA to cover both wavefronts).
- **MFMA/WMMA fragment shapes** differ from Metal/CUDA — another MMA retile + tuning
  pass; the cooperative kernels don't auto-port.
- **`gfx` fragmentation:** intrinsics/dtypes vary by arch (FP8 on gfx942+, MXFP on
  gfx950+, WMMA on gfx1100+) — gate by detected `gfx` like CUDA gates by compute
  capability.
- **Numerics:** validate the hardware microscaling-MFMA path bit-for-bit vs the
  software oracle on real CDNA4 before trusting it.

## 7. Why this is a low-marginal-cost third backend

Because HIP is CUDA-portable, the AMD backend reuses the CUDA backend's emitter and
runtime *structure* almost wholesale — the genuinely new work is the **wavefront
32/64 parameterization**, the **MFMA/WMMA tiling**, and the **HIP runtime glue**.
And the same `mx*`/`mxint*` formats that target Blackwell also target CDNA4
microscaling, so MetalTile's quant matrix already spans Apple GPU (today) +
NVIDIA + AMD hardware.

## 8. References

- **ROCm / HIP** — HIP runtime + hipRTC (runtime compile), `HIPIFY` (CUDA→HIP
  porting), `hipcc`/LLVM AMDGPU. The §3 host-call + dialect mapping.
- **rocWMMA / Composable Kernel (CK)** — the CUTLASS-class libraries for the
  matrix-core (§4.2) and cooperative-kernel (§6) paths; **rocBLAS**, **rocPRIM**.
- **AMD matrix cores** — MFMA (CDNA, `__builtin_amdgcn_mfma_*`), WMMA (RDNA3+,
  `__builtin_amdgcn_wmma_*`); MI300/gfx942 FP8, **MI350/gfx950 (CDNA4) OCP MXFP
  microscaling** — the hardware target the PR-#2 block-scaled formats map onto
  (§4.3), the AMD analog of NVIDIA Blackwell.
- `CUDA_BACKEND_SPEC.md` — the shared backend-seam design this spec is a delta on.

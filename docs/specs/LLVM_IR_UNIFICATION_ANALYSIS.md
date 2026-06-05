# LLVM IR Unification Analysis

**Status:** Research Note — not a spec revision yet
**Author:** Feynman (prompted by team discussion)
**Date:** 2026-06-05

## The Core Claim

> AMD, NVIDIA, and Vulkan/SPIR-V all consume LLVM IR under the hood. We can skip
> the C++/HIP/GLSL intermediate and emit LLVM IR directly, sharing one emitter
> across all three backends. Apple doesn't provide this API, so Metal keeps its
> MSL path.

The existing specs (`CUDA_BACKEND_SPEC.md`, `AMD_BACKEND_SPEC.md`, `VULKAN_BACKEND_SPEC.md`)
proposed emitting CUDA C++ → NVRTC → PTX, HIP C++ → hipRTC → ISA, and GLSL/rspirv →
SPIR-V respectively. This note evaluates whether emitting LLVM IR directly is
viable and simpler — and finds that **LLVM IR unifies all three non-Apple backends**.

---

## 1. Evidence: All Three Backends ARE LLVM Backends

### NVIDIA / NVPTX

- The NVPTX LLVM backend consumes LLVM IR and emits PTX. Documented in the
  [LLVM NVPTX Usage Guide](https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html):
  *"To support GPU programming, the NVPTX back-end supports a subset of LLVM IR
  along with a defined set of conventions."*
- NVIDIA ships `nvvm` (their LLVM-based compiler) which takes LLVM IR bitcode and
  produces PTX. Their **NVVM IR** specification states: *"NVVM IR is a compiler IR
  based on the LLVM IR. The NVVM IR is designed to represent GPU compute kernels."*
  And: *"Technically speaking, NVVM IR is LLVM IR with a set of rules, restrictions,
  and conventions."*
- A simple kernel in LLVM IR: use `ptx_kernel` calling convention, read thread IDs
  via `@llvm.nvvm.read.ptx.sreg.tid.x()`, use address spaces (1=global, 3=shared,
  4=constant, 5=local), annotate with `!nvvm.annotations` metadata.
- The compile pipeline: `llc -mcpu=sm_XX kernel.ll -o kernel.ptx` → `cuModuleLoadData`.

**Verdict:** You can feed LLVM IR to the NVPTX backend and **skip CUDA C++ / NVRTC entirely**.

### AMD / AMDGPU

- The AMDGPU LLVM backend consumes LLVM IR and emits GCN/RDNA/CDNA ISA. Documented
  in the [LLVM AMDGPU Usage Guide](https://llvm.org/docs/AMDGPUUsage.html).
- HIP C++ is compiled by `clang` (LLVM frontend) → LLVM IR → AMDGPU backend → ISA.
  The LLVM IR path is the same, just with the frontend step removed.
- The ROCm runtime (`rocm-amdhsa`) loads code objects produced by the AMDGPU backend.
- AMD GPU intrinsics: `@llvm.amdgcn.workitem.id.x`, `@llvm.amdgcn.s.barrier`, etc.

**Verdict:** You can feed LLVM IR to the AMDGPU backend and **skip HIP C++ / hipRTC entirely**.

### Vulkan / SPIR-V

- LLVM ships a **SPIR-V backend** (promoted to official target status as of LLVM 19.x
  — see [RFC](https://discourse.llvm.org/t/rfc-promoting-spir-v-to-an-official-target/83614)).
- `llc -mtriple=spirv64-unknown-vulkan input.ll -o output.spvt` produces SPIR-V
  binary consumable by Vulkan `vkCreateShaderModule`.
- The [LLVM SPIR-V Usage Guide](https://releases.llvm.org/23.0.0/docs/SPIRVUsage.html)
  documents the full pipeline, including SPIR-V extensions (cooperative matrix,
  subgroup ops, atomics, fp16/int8/bfloat16) and multiple OS targets (`vulkan`,
  `vulkan1.2`, `vulkan1.3`).
- Active development (2026): PR #196101 adds `vulkan` as an OS for the `spirv` target
  triple; PR #174910 adds SPIR-V support in `gpuintrin.h`.
- Prior art: **khal** ([dimforge/khal](https://github.com/dimforge/khal)) already
  does this — same Rust shader compiles to SPIR-V (WebGPU/Vulkan), PTX (CUDA), and
  CPU. **Google clspv** compiles OpenCL C → LLVM IR → SPIR-V for Vulkan compute
  in production. **Khronos SPIRV-LLVM-Translator** provides bi-directional LLVM IR ↔
  SPIR-V translation.

**Verdict:** You can feed LLVM IR to the SPIR-V backend and **skip GLSL/rspirv entirely**.

### Apple / Metal — AIR IS LLVM bitcode (but the backend is locked)

**tl;dr:** Apple's GPU compiler stack uses LLVM IR *internally*, and there are now
open-source tools that generate it, but Apple does not expose a public API to feed
LLVM IR directly. The MSL emitter is still the safe path today.

---

**Key finding: Apple's AIR (Apple Intermediate Representation) IS LLVM bitcode.**

The `.air` files produced by `xcrun metal -c` have the LLVM bitcode magic number
(`DE C0 17 0B`). You can run `llvm-dis` on them to get human-readable LLVM IR, and
`llc` to compile them to x86-64 or ARM64 assembly. This was proven in 2018 by
reverse-engineering the `.metallib` container format
([worthdoingbadly.com/metalbitcode](https://worthdoingbadly.com/metalbitcode/)).

**The `.metallib` container** wraps AIR bitcode with a metadata header (`MTLB`
format: NAME/TYPE/HASH/MDSZ/OFFT/VERS/ENDT tags). The format has been
reverse-engineered and multiple independent tools can produce it.

**Two open-source projects now generate AIR/metallib from LLVM IR:**

1. **llvm-to-air** ([sueszli/llvm-to-air](https://github.com/sueszli/llvm-to-air))
   — Python-based. Takes LLVM IR, lowers it to AIR (LLVM bitcode with
   `llvm.air.*` intrinsics), packages it into a `.metallib` via `xcrun metallib`.
   30+ kernel tests (matmul, softmax, conv2d, reductions, activations).
   1150× speedup on mandelbrot vs Python. Experimental but functional.

2. **CuMetal** ([Lulzx/cuda-metal](https://github.com/Lulzx/cuda-metal))
   — C++-based. Has `cumetal-air-emitter` for low-level AIR/metallib writing,
   `air_inspect` for inspecting metallibs, `air_validate` for validation.
   Takes `.ll` → `.metallib` directly. Also translates CUDA/PTX → LLVM IR → AIR.

3. **xDSL MPS backend** ([docs.xdsl.dev](https://docs.xdsl.dev/reference/backend/mps/))
   — Complete system-level backend for Apple GPUs. Uses an MPS Dialect → AIR →
   metallib pipeline. Documents that skipping MSL gives: (1) direct instruction
   selection for simd ops, (2) predictable codegen, (3) reduced JIT latency.

**The pipeline would be:**

```
LLVM IR → llvm-to-air (or CuMetal) → AIR (.air) → xcrun metallib → .metallib
                                                                        ↓
                                                          Metal runtime loads
                                                          (proprietary AIR→ISA)
```

**The catch:** The final step (AIR → GPU ISA) is Apple's proprietary backend,
invoked by the Metal runtime when loading a `.metallib` or by `xcrun metal`.
There is no public API to feed LLVM IR or AIR directly to the driver — you must
produce a `.metallib` that the Metal runtime accepts.

**Practical status:**

| Approach | Status | Risk |
|---|---|---|
| **MSL emitter** (current) | Production, Apple-supported | None |
| **AIR via llvm-to-air** | Experimental, Python, 30+ kernels working | Fragile, undocumented |
| **AIR via CuMetal** | Experimental, C++, has emitter/validator | Less mature, fewer kernels |
| **AIR via xDSL MPS** | Python/xDSL, in development | Python dep, not for Rust project |

**Verdict:** The MSL emitter stays as the production Apple path. The AIR path is a
promising research direction that could *eventually* let the shared LLVM IR emitter
cover all four backends, but it is not production-ready today. Document and monitor.

---

## 2. The Unified Pipeline

### Production paths (today)

```
MetalTile IR
    │
    ▼
Shared LLVM IR emitter  ─── produces .ll text
    │
    ├─→ llc NVPTX    → .ptx  → ptxas → .cubin   (NVIDIA)
    ├─→ llc AMDGPU   → .o                         (AMD)
    └─→ llc SPIR-V   → .spvt                      (Vulkan, any GPU)

[Separate:] IR → MSL emitter → .metal → xcrun metal → .metallib   (Apple)
```

### Future possibility — Apple via AIR (experimental)

```
MetalTile IR → Shared LLVM IR emitter
    │
    ├─→ llvm-to-air / CuMetal → AIR (.air) → xcrun metallib → .metallib   (Apple)
    └─→ (same NVPTX / AMDGPU / SPIR-V paths as above)
```

The AIR path is not production-ready (see §1 Apple / Metal), but it proves the
shared LLVM IR emitter *could* cover all four backends if the tooling matures.

---

## 3. What the Shared LLVM IR Looks Like

```llvm
; Shared data layout (same string for NVPTX, AMDGPU, SPIR-V)
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-..."

; Backend-specific: filled in by TargetProfile
target triple = "<backend-specific>"   ; e.g. nvptx64-nvidia-cuda / amdgcn-amd-amdhsa / spirv64-unknown-vulkan

define <kernel-calling-convention> void @my_kernel(
    ptr addrspace(1) %input,     ; global memory (addrspace 1 on all three)
    ptr addrspace(1) %output
) {
  ; Backend-specific: intrinsic name from TargetProfile
  %tid = call i32 @<thread-id-intrinsic>(...)

  %ptr = getelementptr float, ptr addrspace(1) %input, i32 %tid
  %val = load float, ptr addrspace(1) %ptr
  ; ... compute ...
  store float %result, ptr addrspace(1) %output_ptr
  ret void
}

; Backend-specific: kernel annotation
!<kernel-annotation-metadata> = ...
```

### Shared constructs (same LLVM IR for all three)

| Concept | LLVM IR construct |
|---------|------------------|
| Arithmetic | `add`, `fadd`, `mul`, `fmul`, `sub`, `fsub`, `div`, etc. |
| Math intrinsics | `llvm.sqrt.*`, `llvm.fma.*`, `llvm.copysign.*` |
| Bit ops | `llvm.ctpop.*`, `llvm.ctlz.*`, `llvm.cttz.*`, `llvm.bswap.*` |
| Memory | `load`, `store`, `getelementptr` |
| Control flow | `br`, `switch`, `phi`, `select`, `ret` |
| Shared memory | `addrspace(3)` — **same number on all three!** |
| Global memory | `addrspace(1)` — **same number on all three!** |
| Constants | `addrspace(2)` — maps to constant/uniform on all three |

### Backend-specific (parameterized by TargetProfile)

| Feature | NVIDIA (NVVM) | AMD (AMDGPU) | Vulkan (SPIR-V) |
|---|---|---|---|
| Target triple | `nvptx64-nvidia-cuda` | `amdgcn-amd-amdhsa` | `spirv64-unknown-vulkan` |
| Kernel calling conv | `ptx_kernel` | `amdgcn_kernel` | `spir_kernel` |
| Thread ID | `@llvm.nvvm.read.ptx.sreg.tid.{x,y,z}` | `@llvm.amdgcn.workitem.id.{x,y,z}` | `__spirv_BuiltInLocalInvocationId` |
| Block ID | `@llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` | `@llvm.amdgcn.workgroup.id.{x,y,z}` | `__spirv_BuiltInWorkgroupId` |
| Block dim | `@llvm.nvvm.read.ptx.sreg.ntid.{x,y,z}` | `@llvm.amdgcn.dispatch.ids` | `__spirv_BuiltInWorkgroupSize` |
| Subgroup/warp size | `@llvm.nvvm.read.ptx.sreg.warpsize()` → 32 | `@llvm.amdgcn.wavefrontsize()` → 32/64 | SPIR-V `SubgroupSize` |
| Barrier | `@llvm.nvvm.barrier0()` | `@llvm.amdgcn.s.barrier()` | `__spirv_ControlBarrier` |
| Shuffle | `@llvm.nvvm.shfl.sync.i32(...)` | `@llvm.amdgcn.permlane16_32(...)` | `__spirv_GroupNonUniformShuffle` |
| Tensor/coop matrix | `@llvm.nvvm.hmma.*` / `tcgen05.*` | `@llvm.amdgcn.mfma.*` / `wmma.*` | `SPV_KHR_cooperative_matrix` (optional) |
| Hardware microscaling | `tcgen05` (Blackwell) | CDNA4 MX MFMA | **None** (software dequant only) |
| Kernel annotation | `!nvvm.annotations` | `"kernel"` fn attr | `!spirv.ExecutionMode` |

---

## 4. Comparison: C++/GLSL Emitters vs LLVM IR Emitter

### Current spec plan (three separate emitters + MSL)

```
IR → CUDA C++ emitter  → NVRTC    → PTX     → Cubin      (NVIDIA)
IR → HIP C++ emitter   → hipRTC   → ISA                 (AMD)
IR → GLSL/rspirv emitter → shaderc → .spvt              (Vulkan)
IR → MSL emitter       → xcrun metal → .metallib         (Apple)
```

**Dependencies:** CUDA Toolkit (NVRTC), ROCm (hipRTC), shaderc/glslang, Xcode.

### Proposed LLVM IR plan (one shared emitter → three production backends + Apple research path)

```
                   ┌→ llc  NVPTX   → PTX  → ptxas → .cubin   (NVIDIA, production)
                   ├→ llc  AMDGPU  → ISA                     (AMD, production)
IR → LLVM IR ──→   ├→ llc  SPIR-V  → .spvt                    (Vulkan, production)
  emitter           └→ llvm-to-air / CuMetal → AIR → metallib (Apple, experimental)
```

**Dependencies:** `llc` (one LLVM build with NVPTX + AMDGPU + SPIR-V targets), Xcode.
Apple AIR path additionally needs `llvm-to-air` or `CuMetal` (experimental).

### Tradeoffs

| Dimension | Separate emitters (current spec) | Shared LLVM IR emitter (proposed) |
|---|---|---|
| **Number of emitters** | 3 (CUDA C++ + HIP C++ + GLSL/rspirv) | **1** (shared LLVM) |
| **Total emitter code** | ~6000 lines (3 × ~2000) | ~2500 shared + ~600 per-backend intrinsic tables |
| **Target backends** | 2 native (NVIDIA, AMD) | **3** (NVIDIA + AMD + Vulkan) |
| **External toolchain deps** | NVRTC + hipRTC + shaderc/glslang | `llc` (one binary) + `ptxas` (NVIDIA only) |
| **Optimization passes** | Each toolchain owns its own | Shared LLVM pass pipeline — we control it |
| **Tensor cores** | Vendor intrinsics per language | LLVM intrinsics (same mechanism, different names) |
| **Vulkan portability** | Separate GLSL/rspirv emitter | Free — SPIR-V backend uses same LLVM IR |
| **Debuggability** | Easy (readable C++/GLSL) | Harder (LLVM IR is verbose) |
| **Test effort** | One codegen test suite per emitter | One shared IR test suite; backend-specific tests for intrinsics only |
| **Apple path** | MSL emitter (separate, unavoidable) | MSL (production) + AIR path (research, would unify) |

**The simplification is real and compounding.** Going from 3 separate text emitters
to 1 shared LLVM IR emitter eliminates the C++/HIP/GLSL toolchain dependencies,
gives direct control over the optimization pipeline, and adds Vulkan portability as
a near-zero-cost add-on.

---

## 5. Key Risks and Mitigations

### Risk 1: LLVM is a heavy dependency

Building LLVM via `llvm-sys` is slow. Mitigations:
- Use pre-built LLVM binaries via `llvm-sys` `no-llvm-linking` feature.
- Use system LLVM that ships with Xcode (macOS) or the CUDA Toolkit (Linux).
- **Emit LLVM IR text (`.ll`), invoke `llc` as a subprocess** — no Rust LLVM binding needed at all. Same pipeline shape as the current NVRTC/hipRTC calls.

**Recommendation:** Start with LLVM IR text emission + `llc` subprocess.

### Risk 2: Tensor-core / cooperative-matrix intrinsics differ per backend

Each backend has different intrinsic names and fragment shapes for matrix multiply.

**Mitigation:** Same situation as separate emitters — you'd write different code anyway. In the shared LLVM IR approach, the differences are parameterized in `TargetProfile` intrinsic tables. No regression.

### Risk 3: Wavefront/subgroup size variability

- NVIDIA: fixed 32.
- AMD: 32 (RDNA) or 64 (CDNA).
- Vulkan: variable (8/16/32/64), **runtime-queried**.

**Mitigation:** The shared emitter takes `lane_width` from `TargetProfile`. For Vulkan, use subgroup-agnostic workgroup reductions as the portable baseline, with subgroup ops as a queried fast path (same strategy as existing Vulkan spec).

### Risk 4: Each LLVM backend accepts a subset of LLVM IR

NVVM IR bans `fence`, `invoke`, `landingpad`, fp128. The SPIR-V backend has its own restrictions.

**Mitigation:** MetalTile's IR doesn't generate most of these. A `validate()` pass in the shared emitter checks against the target's allowed subset.

### Risk 5: SPIR-V has no hardware microscaling

Block-scaled formats (`mx*`/`mxint*`) run via software dequant on Vulkan.

**Mitigation:** Same as the existing Vulkan spec — software dequant is the universal fallback. The E8M0 hardware payoff stays with NVPTX and AMDGPU native backends.

---

## 6. Concrete Simplify Assessment

| Backend | Separate emitter path | LLVM IR path | Net change |
|---------|----------------------|--------------|------------|
| NVIDIA | CUDA C++ → NVRTC → PTX | LLVM IR → `llc` NVPTX → PTX | **Eliminates NVRTC dep, shares emitter** |
| AMD | HIP C++ → hipRTC → ISA | LLVM IR → `llc` AMDGPU → ISA | **Eliminates hipRTC dep, shares emitter** |
| Vulkan | GLSL/rspirv → .spvt | LLVM IR → `llc` SPIR-V → .spvt | **LLVM IR path is simpler than rspirv builder** |
| Apple (MSL) | MSL emitter → xcrun metal | MSL emitter → xcrun metal | **Production path — unchanged** |
| Apple (AIR) | N/A (no public API) | LLVM IR → llvm-to-air/CuMetal → AIR → metallib | **Research path — would unify all four** |

**Total emitters:** 3 → 1 (plus AIR research path).
**Total external toolchains:** 4 (NVRTC + hipRTC + shaderc + Xcode) → 2 (`llc` + Xcode).
**Apple AIR path:** experimental, adds `llvm-to-air` or `CuMetal` dependency if adopted.

---

## 7. Recommended Path

### Phase 0 — Prove LLVM IR viability (this sprint)

Write three hand-crafted `.ll` files for a simple elementwise kernel:

1. **NVIDIA:** `kernel.ll` → `llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90` → `.ptx` → `ptxas` → `.cubin` → launch via CUDA Driver API
2. **AMD:** `kernel.ll` → `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942` → `.o` → load via ROCm runtime
3. **Vulkan:** `kernel.ll` → `llc -mtriple=spirv64-unknown-vulkan` → `.spvt` → `vkCreateShaderModule` → dispatch
4. All three: `tile test` passes correctness against CPU oracle

### Phase 1 — LLVM IR text emitter + codegen refactor

- Add shared LLVM IR emitter producing `.ll` text
- Define `TargetProfile` with backend-specific intrinsic tables, triples, calling conventions
- Implement `NvidiaBackend`, `AmdBackend`, `SpirvBackend` as `CodegenBackend` impls
- Replace NVRTC/hipRTC/shaderc calls with `llc` subprocess invocations
- Keep MSL emitter unchanged

### Phase 2 — Subgroup/wavefront parameterization

- Add `lane_width` handling (32 NVIDIA, 32/64 AMD, runtime-queried Vulkan)
- Portable workgroup reductions for Vulkan with subgroup fast path

### Phase 3 — Optimization pipeline

- Run LLVM optimization passes (`-O3`, `-nvvm-reflect` for NVIDIA) via `opt`
- Add custom passes for MetalTile-specific patterns

### Phase 4 (optional) — In-process LLVM via inkwell

- Replace `llc` subprocess with in-process LLVM compilation for lower latency

### Phase 5 (research) — Apple AIR direct codegen

- Evaluate **llvm-to-air** and **CuMetal** for producing `.metallib` from LLVM IR
- If viable: add an `AppleAirBackend` `CodegenBackend` impl that wraps the external tool
- If the tooling matures to production quality: retire the MSL emitter path
- **Gating question:** Is the experimental AIR tooling reliable enough for CI?

---

## 8. Sources

- **NVVM IR Specification 12.9** — https://docs.nvidia.com/cuda/archive/12.9.1/nvvm-ir-spec/index.html
- **LLVM NVPTX Backend Usage Guide** — https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html
- **LLVM AMDGPU Backend Usage Guide** — https://llvm.org/docs/AMDGPUUsage.html
- **LLVM SPIR-V Backend Usage Guide** — https://releases.llvm.org/23.0.0/docs/SPIRVUsage.html
- **SPIR-V Promotion to Official LLVM Target (RFC)** — https://discourse.llvm.org/t/rfc-promoting-spir-v-to-an-official-target/83614
- **SPIRV-LLVM-Translator (Khronos)** — https://github.com/KhronosGroup/SPIRV-LLVM-Translator
- **clspv (Google, OpenCL C → LLVM → SPIR-V → Vulkan)** — https://github.com/google/clspv
- **khal (dimforge, write once → SPIR-V + PTX + CPU)** — https://github.com/dimforge/khal
- **LLVM PR #196101 — `vulkan` as SPIR-V OS target** — https://github.com/llvm/llvm-project/pull/196101
- **LLVM PR #174910 — SPIR-V `gpuintrin.h` support** — https://github.com/llvm/llvm-project/pull/174910
- **LLVM Compile CUDA with clang** — https://prereleases.llvm.org/15.0.0/rc2/docs/CompileCudaWithLLVM.html
- **cuda-oxide Architecture** — https://nvlabs.github.io/cuda-oxide/compiler/architecture-overview.html
- **llvm-to-air (sueszli, reverse-engineered LLVM IR → AIR → metallib)** — https://github.com/sueszli/llvm-to-air
- **xDSL MPS backend (Apple GPU AIR backend)** — https://docs.xdsl.dev/reference/backend/mps/
- **CuMetal (Lulzx, CUDA/LLVM IR → AIR → metallib)** — https://github.com/Lulzx/cuda-metal
- **Metal .air/.metallib reverse engineering** — https://worthdoingbadly.com/metalbitcode/
- **MetalLibraryArchive (metallib parser)** — https://github.com/YuAo/MetalLibraryArchive
- **Apple LLVM GPU Compiler talk (2017)** — https://llvm.org/devmtg/2017-10/slides/Chandrasekaran-Maggioni-Apple%20LLVM%20GPU%20Compiler.pdf
- **Apple Metal shader converter (DXIL → Metal IR)** — https://developer.apple.com/metal/shader-converter/
- **`CUDA_BACKEND_SPEC.md`** — NVIDIA backend spec (revised, LLVM IR codegen)
- **`AMD_BACKEND_SPEC.md`** — AMD backend spec (revised, LLVM IR codegen)
- **`VULKAN_BACKEND_SPEC.md`** — Vulkan backend spec (revised, LLVM IR codegen)
- **`AIR_BACKEND_SPEC.md`** — Apple AIR backend spec (experimental, LLVM IR codegen)

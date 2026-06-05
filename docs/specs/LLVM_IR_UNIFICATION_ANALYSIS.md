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

### Apple / Metal

- Metal Shading Language (MSL) is the **only** input format. Apple's internal stack
  (MSL → Air → GPU ISA) is proprietary and does not accept LLVM IR.
- No documented way to feed LLVM IR to an Apple GPU.

**Verdict:** Metal requires its own MSL emitter — unavoidable.

---

## 2. The Unified Pipeline

```
MetalTile IR
    │
    ▼
Shared LLVM IR emitter  ─── produces .ll text ─── one emitter, all backends
    │
    ├─→ llc -mtriple=nvptx64-nvidia-cuda  -mcpu=sm_XX  → .ptx  → ptxas → .cubin  (NVIDIA)
    ├─→ llc -mtriple=amdgcn-amd-amdhsa    -mcpu=gfxXXXX → .o                              (AMD)
    └─→ llc -mtriple=spirv64-unknown-vulkan              → .spvt                          (Vulkan)
                                                                                          (any GPU)

[Separate path for Apple:]
MetalTile IR → MSL emitter → .metal → xcrun metal → .metallib   (Apple, unavoidable)
```

**One emitter, three backends.** Only `TargetProfile` differs: intrinsic names,
address-space numbers, calling convention, and LLVM target triple.

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

### Current spec plan (three separate emitters)

```
IR → CUDA C++ emitter  → NVRTC    → PTX     → Cubin      (NVIDIA)
IR → HIP C++ emitter   → hipRTC   → ISA                 (AMD)
IR → GLSL/rspirv emitter → shaderc → .spvt              (Vulkan)
IR → MSL emitter       → xcrun metal → .metallib         (Apple)
```

**Dependencies:** CUDA Toolkit (NVRTC), ROCm (hipRTC), shaderc/glslang, Xcode.

### Proposed LLVM IR plan (one shared LLVM emitter → three backends)

```
                   ┌→ llc  NVPTX   → PTX  → ptxas → .cubin   (NVIDIA)
IR → LLVM IR ──→   ├→ llc  AMDGPU  → ISA                     (AMD)
  emitter          └→ llc  SPIR-V  → .spvt                    (Vulkan, any GPU)

[Separate:] IR → MSL emitter → xcrun metal → .metallib        (Apple)
```

**Dependencies:** `llc` (one LLVM build with all three targets), Xcode.

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
| Apple | MSL emitter → xcrun metal | MSL emitter → xcrun metal | **Unchanged** |

**Total emitters:** 3 → 1. **Total external toolchains:** 4 (NVRTC + hipRTC + shaderc + Xcode) → 2 (`llc` + Xcode).

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
- **inkwell crate** — https://crates.io/crates/inkwell
- **`CUDA_BACKEND_SPEC.md`** — NVIDIA backend spec (revised, LLVM IR codegen)
- **`AMD_BACKEND_SPEC.md`** — AMD backend spec (revised, LLVM IR codegen)
- **`VULKAN_BACKEND_SPEC.md`** — Vulkan backend spec (revised, LLVM IR codegen)

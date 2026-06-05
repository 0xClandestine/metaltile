# LLVM IR Unification Analysis

**Status:** Research Note — not a spec revision yet
**Author:** Feynman (prompted by team discussion)
**Date:** 2026-06-05

## The Core Claim

> AMD and CUDA both use LLVM under the hood. We can skip the C++/HIP codegen step entirely and emit LLVM IR directly, sharing one emitter across both backends. Apple doesn't provide this API, so Metal keeps its MSL path.

The existing specs (`CUDA_BACKEND_SPEC.md`, `AMD_BACKEND_SPEC.md`) propose emitting CUDA C++ → NVRTC → PTX and HIP C++ → hipRTC → ISA respectively. This note evaluates whether emitting LLVM IR directly is viable and simpler.

---

## 1. Evidence: Both Backends ARE LLVM Backends

### NVIDIA / NVPTX

- The NVPTX LLVM backend consumes LLVM IR and emits PTX. This is documented in the [LLVM NVPTX Usage Guide](https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html): *"To support GPU programming, the NVPTX back-end supports a subset of LLVM IR along with a defined set of conventions used to represent GPU programming concepts."*
- NVIDIA ships `nvvm` (their LLVM-based compiler) which takes LLVM IR bitcode and produces PTX. Their **NVVM IR** specification ([NVVM IR Spec 12.9](https://docs.nvidia.com/cuda/archive/12.9.1/nvvm-ir-spec/index.html)) is literally: *"NVVM IR is a compiler IR based on the LLVM IR. The NVVM IR is designed to represent GPU compute kernels. The NVVM compiler generates PTX code from NVVM IR."* And: *"Technically speaking, NVVM IR is LLVM IR with a set of rules, restrictions, and conventions."*
- A simple kernel in LLVM IR: use `ptx_kernel` calling convention, read thread IDs via `@llvm.nvvm.read.ptx.sreg.tid.x()`, use address spaces (1=global, 3=shared, 4=constant, 5=local), annotate with `!nvvm.annotations = !{!0}` metadata.
- The compile pipeline is: `llc -mcpu=sm_XX kernel.ll -o kernel.ptx` → `cuModuleLoadData`.
- **CICC (the CUDA C++ compiler) is just a frontend that produces NVVM IR.** The `tileiras` tool (CUDA Toolkit 13.x) goes MLIR → NVPTX, skipping C++ entirely. The common trunk is the NVPTX backend.

**Verdict:** You can feed LLVM IR to the NVPTX backend and **skip CUDA C++ / NVRTC entirely**.

### AMD / AMDGPU

- The AMDGPU LLVM backend consumes LLVM IR and emits GCN/RDNA/CDNA ISA. Documented in [LLVM AMDGPU Usage Guide](https://llvm.org/docs/AMDGPUUsage.html): *"The AMDGPU backend provides ISA code generation for AMD GPUs."*
- HIP C++ is compiled by `clang` (LLVM frontend) → LLVM IR → AMDGPU backend → ISA. The LLVM IR path is the same as the C++ path, just with the frontend step removed.
- The ROCm runtime (`rocm-amdhsa`) loads code objects produced by the AMDGPU backend. The LLVM mailing list discussion confirms feeding LLVM IR directly to the AMDGPU backend for ROCm works: *"Compile an LLVM IR module with AMDGPU backend to a .o file using amdgcn triple."*
- AMD GPU address spaces and intrinsics are accessed via LLVM IR constructs: `amdgcn` target triple, intrinsic math functions, barrier intrinsics.

**Verdict:** You can feed LLVM IR to the AMDGPU backend and **skip HIP C++ / hipRTC entirely**.

### Apple / Metal

- Metal Shading Language (MSL) is the **only** input format. Apple's internal compiler stack (MSL → Air → GPU ISA) is proprietary and does not accept LLVM IR.
- There is no documented way to feed LLVM IR to an Apple GPU. The `metaltile-codegen` MSL emitter is structurally unavoidable for Apple GPUs.

**Verdict:** Your statement that "Apple simply doesn't provide the needed API" is correct. Metal requires its own emitter.

---

## 2. What an LLVM IR Emitter Would Look Like

Instead of emitting C++ text, the codegen would emit LLVM IR (either as text `.ll` or via `inkwell`/`llvm-sys` programmatic IR construction).

### Shared across both backends

```
target datalayout = "e-p:64:64:64-..."
target triple = "<backend-specific>"

; Kernel function
define ptx_kernel      ; NVIDIA uses ptx_kernel calling convention
       define amdgcn_kernel  ; AMD uses amdgcn_kernel calling convention
void @my_kernel(
    ptr addrspace(1) %input,   ; global memory
    ptr addrspace(1) %output
) {
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()   ; NVIDIA
  ; --or--
  %tid = call i32 @llvm.amdgcn.workitem.id.x()        ; AMD

  %ptr = getelementptr float, ptr addrspace(1) %input, i32 %tid
  %val = load float, ptr addrspace(1) %ptr
  ; ... compute ...
  store float %result, ptr addrspace(1) %output_ptr
  ret void
}

!nvvm.annotations = !{!0}    ; NVIDIA kernel annotation
!0 = !{ptr @my_kernel, !"kernel", i32 1}
```

### What's shared (same LLVM IR for both)

| Concept | LLVM IR construct |
|---------|------------------|
| Arithmetic | `add`, `fadd`, `mul`, `fmul`, `sub`, `fsub`, `div`, etc. |
| Math intrinsics | `llvm.sqrt.*`, `llvm.fma.*`, `llvm.copysign.*` |
| Bit ops | `llvm.ctpop.*`, `llvm.ctlz.*`, `llvm.cttz.*`, `llvm.bswap.*` |
| Memory | `load`, `store`, `getelementptr` |
| Control flow | `br`, `switch`, `phi`, `select`, `ret` |
| Barriers | `@llvm.nvvm.barrier0()` / `@llvm.amdgcn.s.barrier()` (different names, same concept) |
| Shared memory | `addrspace(3)` (same number on both!) |

**Critical match:** Both NVPTX and AMDGPU use address space **3** for shared/LDS memory. Many other address spaces also align (1=global, 5=local).

### What's backend-specific (parameterized by TargetProfile)

| Feature | NVIDIA (NVVM) | AMD (AMDGPU) |
|---------|--------------|--------------|
| Target triple | `nvptx64-nvidia-cuda` | `amdgcn-amd-amdhsa` |
| Kernel calling conv | `ptx_kernel` | `amdgcn_kernel` |
| Thread ID | `@llvm.nvvm.read.ptx.sreg.tid.{x,y,z}` | `@llvm.amdgcn.workitem.id.{x,y,z}` |
| Block ID | `@llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` | `@llvm.amdgcn.workgroup.id.{x,y,z}` |
| Block dim | `@llvm.nvvm.read.ptx.sreg.ntid.{x,y,z}` | `@llvm.amdgcn.dispatch.ids` (different mechanism) |
| Warp size | `@llvm.nvvm.read.ptx.sreg.warpsize()` → 32 | `@llvm.amdgcn.wavefrontsize()` → 32 or 64 |
| Barrier | `@llvm.nvvm.barrier0()` | `@llvm.amdgcn.s.barrier()` |
| Shuffle | `@llvm.nvvm.shfl.sync.i32(...)` | `__builtin_amdgcn_permlane16_32(...)` |
| Tensor cores (pre-Blackwell) | `@llvm.nvvm.hmma.*` (wmma-style) | `__builtin_amdgcn_mfma_*` / `__builtin_amdgcn_wmma_*` |
| Tensor cores (Blackwell) | `@llvm.nvvm.tcgen05.*` | AMD CDNA4 MX MFMA intrinsics |
| Kernel annotation | `!nvvm.annotations` metadata | `!amdgpu.annotations` or `"kernel"` attribute |
| Data layout | Same string works for both | Same string works for both |

---

## 3. Comparison: C++ Emitter vs LLVM IR Emitter

### Current spec plan (two C++ emitters)

```
IR → CUDA C++ emitter → NVRTC (CUDA Toolkit) → PTX → Cubin
IR → HIP C++ emitter  → hipRTC (ROCm)         → ISA
IR → MSL emitter      → xcrun metal            → metallib   (Apple, unavoidable)
```

**Dependencies:** CUDA Toolkit (NVRTC), ROCm (hipRTC), xcrun (Xcode).

### Proposed LLVM IR plan (one shared LLVM emitter)

```
IR → LLVM IR emitter → NVPTX backend (llvm-sys/inkwell) → PTX → Cubin
                     → AMDGPU backend (llvm-sys/inkwell) → ISA
IR → MSL emitter     → xcrun metal                      → metallib   (Apple, unchanged)
```

**Dependencies:** `llvm-sys` or `inkwell` crate (one LLVM build), Xcode.

### Tradeoffs

| Dimension | C++ emitter (current spec) | LLVM IR emitter (proposed) |
|-----------|---------------------------|---------------------------|
| **Emitter count** | 2 (CUDA + HIP) | 1 (shared LLVM) |
| **Skipped toolchain deps** | NVRTC, hipRTC | None — but need LLVM libs |
| **LLVM dependency** | Implicit (via NVRTC/hipRTC) | Explicit (llvm-sys/inkwell + LLVM .dylib) |
| **LLVM version** | Whatever NVRTC/hipRTC ships | Pinned by Cargo.toml |
| **Optimization passes** | NVRTC/hipRTC own them | We control the pass pipeline |
| **Tensor core intrinsics** | Via PTX inline asm or CUTLASS | Via LLVM intrinsics (`llvm.nvvm.tcgen05.*` etc.) |
| **Complexity of emitter** | High (C++ syntax, type system) | Medium (LLVM IR is simpler, more regular) |
| **Debuggability** | Easy (readable C++) | Harder (LLVM IR is verbose) |
| **Portability to new backends** | New emitter per backend | Single emitter, just new intrinsic set |
| **Rust ecosystem** | Mature (string templating) | Medium (inkwell, llvm-sys) |
| **Build time impact** | None (external tools) | Significant (LLVM compilation via llvm-sys) |

---

## 4. Key Risks and Mitigations

### Risk 1: LLVM is a heavy dependency

Building LLVM via `llvm-sys` is slow and painful. Alternatives:
- Use pre-built LLVM binaries (the `llvm-sys` `no-llvm-linking` feature + system LLVM).
- Use the system LLVM that ships with Xcode (macOS) or the CUDA Toolkit (Linux).
- Emit LLVM IR **text** (`.ll`), invoke `llc` as a subprocess (no Rust binding needed). This is the simplest integration: same shape as the current NVRTC/hipRTC calls, just replacing them with `llc` invocation.

**Recommendation:** Start with LLVM IR text emission + `llc` subprocess. This is the lowest-risk path: no new Rust dependencies, the same pipeline shape, and you can see the IR you're emitting.

### Risk 2: Tensor core intrinsics differ

NVIDIA's `llvm.nvvm.tcgen05.*` and AMD's MFMA/WMMA intrinsics have no common subset. You'll still need backend-specific IR snippets for MMA kernels.

**Mitigation:** Same situation as the C++ emitter approach — you'd emit different PTX inline asm vs HIP intrinsics. In LLVM IR, you emit different intrinsic calls, parameterized by `TargetProfile`. No regression.

### Risk 3: Wavefront 32 vs 64 (AMD)

This is identical to the risk in `AMD_BACKEND_SPEC.md §4.1`. Parameterize `WARP_SIZE` in the emitter.

### Risk 4: NVVM is a *subset* of LLVM IR

You must avoid unsupported constructs (fp128, `fence` instruction, `invoke`, `landingpad`, `cmpxchg` on types other than i32/i64/i128, etc.). A validation pass is needed.

**Mitigation:** Mostly straightforward — MetalTile's IR doesn't generate most of these. The subset restriction is not onerous for compute kernels.

### Risk 5: `inkwell`/`llvm-sys` version coupling

You must match LLVM versions. Using text IR + `llc` subprocess avoids this entirely.

---

## 5. Concrete Simplify Assessment

| Concern | C++ emitter path | LLVM IR path | Net |
|---------|----------------|--------------|-----|
| Lines of emitter code | ~2000 each (CUDA + HIP) | ~2500 shared + ~500 backend-specific | **Roughly same** |
| External toolchain deps | NVRTC + hipRTC | `llc` (x86\_64) + `ptxas` | **Fewer** (no hipRTC dep) |
| Codegen passes | Built into NVRTC/hipRTC | Shared LLVM pass pipeline | **More control** |
| Apple path | Unavoidable MSL | Unavoidable MSL | Same |
| Test/debug | Readable C++ | Verbose LLVM IR | **Worse for debugging** |
| Future backend (Intel?) | New emitter | New intrinsic set | **Easier** |

**The simplification is real but incremental, not transformative.** You go from 2 C++ text emitters to 1 LLVM IR text emitter + per-backend intrinsic selection. The bigger win is eliminating the NVRTC/hipRTC toolchain dependencies and getting direct control over the LLVM optimization pipeline.

---

## 6. Recommended Path

### Phase 0 — Prove LLVM IR viability (this sprint)

Write two hand-crafted `.ll` files (one for NVPTX, one for AMDGPU) for a simple elementwise kernel:

1. **NVIDIA:** `kernel.ll` → `llc -mcpu=sm_90` → `kernel.ptx` → `ptxas` → `kernel.cubin` → `cuModuleLoadData` → launch
2. **AMD:** `kernel.ll` → `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942` → `kernel.o` → load via ROCm runtime
3. Both: `tile test` passes correctness against CPU oracle

If this works, the LLVM IR approach is validated end-to-end.

### Phase 1 — LLVM IR text emitter + codegen refactor

- Add an `LlmIr` codegen backend that emits `.ll` text instead of `.cu`/`.hip` text
- Add a `TargetProfile` with backend-specific intrinsic tables and calling conventions
- Keep the MSL emitter unchanged
- Replace NVRTC/hipRTC calls with `llc` subprocess invocations

### Phase 2 — Optimization pipeline

- Run LLVM optimization passes (`-O3`, `-nvvm-reflect` for NVIDIA) via `opt`
- Add custom passes for MetalTile-specific patterns (e.g., block-scaled dequant fusion)

### Phase 3 (optional) — In-process LLVM via inkwell

- Replace `llc` subprocess with in-process LLVM compilation for lower latency
- This is an optimization, not a correctness requirement

---

## 7. Sources

- **NVVM IR Specification 12.9** — https://docs.nvidia.com/cuda/archive/12.9.1/nvvm-ir-spec/index.html
- **LLVM NVPTX Backend Usage Guide** — https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html
- **LLVM AMDGPU Backend Usage Guide** — https://llvm.org/docs/AMDGPUUsage.html
- **LLVM Compile CUDA with clang** — https://prereleases.llvm.org/15.0.0/rc2/docs/CompileCudaWithLLVM.html
- **cuda-oxide Architecture** — https://nvlabs.github.io/cuda-oxide/compiler/architecture-overview.html
- **inkwell crate** — https://crates.io/crates/inkwell — Rust LLVM bindings
- `CUDA_BACKEND_SPEC.md` — current spec proposing CUDA C++ + NVRTC path
- `AMD_BACKEND_SPEC.md` — current spec proposing HIP C++ + hipRTC path

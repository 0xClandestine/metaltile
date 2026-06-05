# Apple AIR Backend Spec

**Status:** 📋 Research / Experimental (not production-ready)
**Scope:** Add an **Apple GPU AIR** backend that lowers MetalTile's `#[kernel]` DSL / IR to LLVM IR compiled via reverse-engineered tooling to **AIR (Apple Intermediate Representation)** bitcode, packaged as a `.metallib` and dispatched through the Metal runtime — potentially unifying Apple under the shared LLVM IR emitter.
**Out of scope:** model loading, graph execution, checkpoint readers — MetalTile is an optimized-kernel generator, not an inference engine.

> **Read [`LLVM_IR_UNIFICATION_ANALYSIS.md`](LLVM_IR_UNIFICATION_ANALYSIS.md) first.** This spec documents the experimental Apple AIR path, which is **Phase 5 (research)** in the unification plan. The production Apple path remains the MSL emitter.

---

## 1. Executive Summary

**AIR (Apple Intermediate Representation) IS LLVM bitcode.** The `.air` files produced by `xcrun metal -c` have the standard LLVM bitcode magic number (`DE C0 17 0B`). This means Apple's GPU compiler stack uses LLVM IR *internally* — the same format the shared LLVM IR emitter produces.

Three independent open-source projects now generate AIR from LLVM IR:
- **llvm-to-air** (Python, 30+ kernel tests, 1150× speedup demonstration)
- **CuMetal** (C++, with `cumetal-air-emitter` for direct metallib writing)
- **xDSL MPS backend** (Python/xDSL, complete system-level MLIR → AIR pipeline)

This opens the possibility of **unifying all four backends** (NVPTX + AMDGPU + SPIR-V + Apple AIR) under a single shared LLVM IR emitter, with only the packaging step differing per backend.

**However:** The AIR tooling is reverse-engineered and experimental. Apple does not expose a public API for feeding LLVM IR or AIR directly to the GPU driver. The production Apple path remains the MSL emitter. This spec documents the AIR path as a research track.

---

## 2. Current Status — what exists vs what is missing

| Concern | Status |
|---|---|
| **LLVM IR → AIR bitcode** | ✅ Reverse-engineered by `llvm-to-air` (Python) and `CuMetal` (C++) |
| **AIR → `.metallib` packaging** | ✅ Container format reverse-engineered (`MTLB` format). `CuMetal` has `cumetal-air-emitter`. Also can use `xcrun metallib` |
| **AIR intrinsics (`llvm.air.*`)** | ✅ Documented by `llvm-to-air` — ~30 intrinsics identified |
| **AIR metadata conventions** | ✅ Reverse-engineered — kernel signatures, argument bindings, threadgroup dimensions |
| **Apple GPU ISA backend** | ❌ Proprietary, inside Metal runtime and `xcrun metal`. Not replaceable. |
| **Production reliability** | ❌ Experimental — no CI, no version stability guarantees |
| **Apple API to feed LLVM IR** | ❌ No public API. Must produce `.metallib` accepted by `MTLDevice.newLibraryWithData:` |
| **Simdgroup/tensor-core access** | ❓ Partially reverse-engineered, not well tested |

---

## 3. The AIR Pipeline

### 3.1 Production path (current, safe)

```
IR → MSL emitter → .metal → xcrun metal → .air → xcrun metallib → .metallib → Metal runtime
```

### 3.2 AIR research path (experimental)

```
IR → LLVM IR emitter                                 (shared with NVPTX/AMDGPU/SPIR-V)
     │
     ├─→ llvm-to-air (Python subprocess)              LLVM IR (.ll) → AIR (.air)
     │       │
     │       └─→ xcrun metallib → .metallib → Metal runtime
     │
     └─→ CuMetal `cumetalc` (subprocess or lib)       LLVM IR (.ll) → .metallib directly
              │
              └─→ Metal runtime
```

Both paths are subprocess calls (same pattern as `llc` for the other backends), not in-process Rust dependencies.

### 3.3 Shared components

| Component | Shared with | Notes |
|---|---|---|
| LLVM IR emitter | NVPTX, AMDGPU, SPIR-V | Same `CodegenBackend::emit_llvm_ir()` with `TargetProfile::air(...)` |
| `TargetProfile` | All backends | Parameterizes intrinsics, address spaces, calling convention |
| Grid/block dispatch | Metal runtime | Same `MTLDevice` + `MTLComputePipelineState` as MSL path |
| Buffer management | Metal runtime | Same `MTLBuffer` allocation |
| `quant::{codec,format}` | All backends | Pure Rust, backend-agnostic |

---

## 4. Rust API Design

Follows the same `CodegenBackend` trait defined in [`CUDA_BACKEND_SPEC.md §4`](CUDA_BACKEND_SPEC.md#4-unified-rust-api-design).

### 4.1 `TargetProfile::air(…)` construction

```rust
/// Apple AIR GPU profile targeting a specific Metal/OS version.
impl TargetProfile {
    /// Create an Apple AIR profile for the given Metal feature set.
    ///
    /// `metal_version` is the Metal Shading Language version to target
    /// (e.g. `MetalVersion::V3_0`), which affects available intrinsics.
    pub fn air(metal_version: MetalVersion) -> Self;
}

/// Metal Shading Language version, affects AIR intrinsic availability.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MetalVersion(u32);

impl MetalVersion {
    pub const V2_0: Self = MetalVersion(0x020000);
    pub const V2_3: Self = MetalVersion(0x020300);  // simdgroup_matrix, bfloat
    pub const V3_0: Self = MetalVersion(0x030000);  // M3+ GPU family
    pub const V3_1: Self = MetalVersion(0x030100);  // M4+ GPU family
}
```

### 4.2 `AirBackend` — the Apple AIR `CodegenBackend` impl

```rust
/// Apple GPU AIR codegen backend (experimental).
///
/// Emits LLVM IR with `llvm.air.*` intrinsics, then invokes an external
/// tool (llvm-to-air or CuMetal) to produce a `.metallib` loadable by the
/// Metal runtime.
///
/// # Status
///
/// **Experimental.** The AIR tooling is reverse-engineered and may break
/// across Xcode/macOS updates. The production Apple path remains the MSL
/// emitter. Enable this backend with `--target air` for testing only.
pub struct AirBackend { /* private fields */ }

impl AirBackend {
    /// Create a new Apple AIR backend.
    ///
    /// `tool` selects the external tool for AIR generation.
    pub fn new(version: MetalVersion, tool: AirTool) -> Self;
}

/// The external tool used to produce AIR from LLVM IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AirTool {
    /// Use `llvm-to-air` (Python subprocess). Requires `uv run llvm_to_air.py`.
    LlvmToAir,
    /// Use `CuMetal`'s `cumetalc` (binary subprocess). Requires `cumetalc` installed.
    CuMetal,
}

impl CodegenBackend for AirBackend {
    fn profile(&self) -> &TargetProfile;
    fn emit_llvm_ir(&self, kernel: &Kernel) -> LmResult<String>;
    fn compile(&self, llvm_ir: &str) -> Result<CompiledKernel, CompileError>;
    fn name(&self) -> &'static str { "air" }
}

impl AirBackend {
    /// The Metal version string embedded in AIR metadata, e.g. `"3.0"`.
    fn metal_version_string(&self) -> &str;
}
```

### 4.3 `AirDevice` — reusing Metal runtime

```rust
/// Apple GPU device backed by the Metal runtime, dispatching AIR-compiled kernels.
///
/// Reuses the existing `MetalDevice` under the hood — the `.metallib` is loaded
/// via `MTLDevice.newLibraryWithData:` just like MSL-compiled metallibs.
pub struct AirDevice { /* private: wraps MetalDevice */ }

impl Device for AirDevice { /* see CUDA_BACKEND_SPEC.md §4.6 */ }
```

The `AirDevice` compiles via `AirBackend` (producing a `.metallib`), then loads and dispatches it through the same Metal runtime API as the MSL path. The only difference is the compilation frontend.

### 4.4 Error types

Additional compile errors specific to the AIR path:

```rust
#[derive(Debug)]
#[non_exhaustive]
pub enum AirCompileError {
    /// `llvm-to-air` or `cumetalc` subprocess failed.
    ToolFailed { stderr: String },
    /// `xcrun metallib` packaging failed.
    MetallibFailed { stderr: String },
    /// Metal runtime rejected the `.metallib` (e.g. invalid AIR version).
    MetalRejection(String),
    /// AIR tool not installed or not found in PATH.
    ToolNotFound(String),
}

impl From<AirCompileError> for CompileError { ... }
```

---

## 5. DSL → LLVM IR → AIR mapping

The shared LLVM IR emitter produces standard LLVM IR. The AIR-specific pass (handled by `llvm-to-air` or `CuMetal`) then transforms the LLVM IR into AIR by:

### 5.1 Intrinsic renaming

Standard LLVM GPU intrinsics are replaced with `llvm.air.*` intrinsics:

| Concept | Standard LLVM IR | AIR (llvm.air.*) |
|---|---|---|
| Barrier | `@llvm.nvvm.barrier0()` (or generic) | `@air.wg.barrier(i32, i32)` |
| Thread position in grid | `@llvm.nvvm.read.ptx.sreg.ctaid.*` | `@air.grid_origin(...)` or MSL-style builtins |
| Thread index in group | `@llvm.nvvm.read.ptx.sreg.tid.*` | `@air.thread_position_in_grid(...)` |
| Group memory barrier | `@llvm.nvvm.barrier0()` | `@air.simdgroup_barrier(...)` |
| Simd shuffle | `@llvm.nvvm.shfl.sync.i32(...)` | `@air.simd_shuffle(...)` |
| Simd reduce | Warp shuffle reduction | `@air.simd_sum(...)`, `@air.simd_min(...)` |
| Math intrinsics | `llvm.sqrt.*`, `llvm.fma.*` | Same (`llvm.*`) — AIR accepts standard LLVM math intrinsics |
| Fence / membar | `llvm.nvvm.membar.*` | `@air.fence(...)` |

### 5.2 Address spaces

AIR uses the same address space numbering as NVPTX and AMDGPU:

| Memory | Address space | MSL equivalent |
|---|---|---|
| Device (global) | 1 | `device` |
| Constant | 2 | `constant` |
| Shared (threadgroup) | 3 | `threadgroup` |
| Local (private) | 5 | N/A (automatic) |

**Critical match:** `addrspace(3)` is shared/threadgroup memory across NVPTX, AMDGPU, SPIR-V, **and** AIR. The shared LLVM IR emitter's address space handling works unchanged.

### 5.3 AIR metadata conventions

AIR uses named metadata nodes to describe kernel parameters — the `llvm-to-air` `MetadataGenerator` class handles this. Key metadata:

```llvm
; Kernel argument metadata
!air.kernel = !{!0}
!0 = !{!"kernel_name", !"kernel_func", !{!1, !2}, !"3.0"}
!1 = !{i32 0, !"arg_name", !"float*", i32 0, i32 1}  ; buffer arg (addrspace, binding)
!2 = !{i32 1, !"arg_value", !"float", i32 0, i32 0}    ; value arg

; Threadgroup size metadata
!air.threadgroup_size = !{!3}
!3 = !{i32 256, i32 1, i32 1}

; Metal version
!air.metal_version = !{!"3.0"}
```

### 5.4 Threadgroup memory allocation

AIR represents threadgroup memory as `addrspace(3)` global variables (identical to NVPTX/AMDGPU/SPIR-V):

```llvm
@_threadgroup_memory = internal addrspace(3) global [1024 x float] undef, align 4
```

The shared LLVM IR emitter already handles this identically for all backends.

---

## 6. What's different for AIR

### 6.1 Subgroup size: always 32 (same as NVIDIA)

Apple GPUs use a fixed **32-lane simdgroup**, identical to NVIDIA warps. This is the same 32-lane assumption the shared emitter already handles. No wavefront-64 variability (AMD) or runtime query (Vulkan) needed.

**This is the lucky structural match:** the 32-lane reductions, shuffles, and simd operations generated for NVIDIA work unchanged for Apple AIR.

### 6.2 Simdgroup matrix (tensor cores)

Apple GPUs have `simdgroup_matrix` operations accessible via `llvm.air.simdgroup.*` intrinsics:

| Operation | AIR intrinsic | Fragment shape |
|---|---|---|
| Matrix multiply-accumulate | `@air.simdgroup_async_matmul(...)` | 8×8 (matches Metal's `simdgroup_matrix`) |
| Matrix load | `@air.simdgroup_async_copy(...)` | 8×8 tiles |
| Matrix store | `@air.simdgroup_store(...)` | 8×8 tiles |

**Fragment shape match:** Apple's 8×8 simdgroup matrix matches MetalTile's IR default. No re-tiling needed (unlike NVIDIA's 16×16×16 `wmma` or AMD's MFMA/WMMA).

### 6.3 No hardware microscaling

Apple GPUs (as of M4) do not have hardware E8M0 microscaling. Block-scaled formats (`mx*`/`mxint*`) run via software dequant, same as pre-Blackwell NVIDIA and Vulkan.

### 6.4 Tooling dependencies

| Tool | Required by | Availability |
|---|---|---|
| `llvm-to-air` (`llvm_to_air.py`) | `AirTool::LlvmToAir` | `pip install` or `uv run` from GitHub |
| `CuMetal` (`cumetalc`) | `AirTool::CuMetal` | Build from source, or use pre-built binary |
| `xcrun metallib` | Both (for packaging) | Ships with Xcode Command Line Tools |

---

## 7. Implementation phases

### Phase 5a — Tool evaluation (this sprint)

1. Install `llvm-to-air` and `CuMetal` on a macOS machine with Apple Silicon.
2. Produce a hand-crafted `.ll` file (elementwise kernel) and run through both tools.
3. Load the resulting `.metallib` via `MTLDevice.newLibraryWithData:` — does the runtime accept it?
4. Dispatch and verify correctness against CPU oracle.
5. Document which tool works, which AIR/metallib features are reliable, and known failure modes.

**Gating question:** Can we get a `#[test_kernel]` green through the AIR path?

### Phase 5b — `AirBackend` + `TargetProfile::air(...)`

If Phase 5a passes:
1. Add `TargetProfile::air(...)` with AIR-specific intrinsic names and metadata conventions.
2. Implement `AirBackend` wrapping `llvm-to-air` or `CuMetal` as a subprocess.
3. Define `AirDevice` reusing the Metal runtime for dispatch.
4. Gate behind `--target air` and an experimental feature flag.

### Phase 5c — Coverage parity

- Elementwise + reduction kernels (same as other backends).
- Simdgroup matrix kernels (8×8 tiles — no re-tiling needed).
- Block-scaled format support via software dequant.
- Cooperative kernel reimplementation (may need AIR-specific intrinsics).

### Phase 5d — Production readiness assessment

- Run full `tile test` suite through AIR path.
- Benchmark against MSL path — is there a performance difference?
- Evaluate stability across Xcode/macOS updates.
- **Decision gate:** Is the AIR path reliable enough to:
  - Replace the MSL emitter? (Unlikely in near term)
  - Serve as a secondary Apple path for CI/test?
  - Be documented as experimental with known caveats?

---

## 8. Risks / open questions

- **Tooling fragility (the big one).** `llvm-to-air` and `CuMetal` are reverse-engineered projects with no stability guarantees. An Xcode update could break the `.metallib` format or AIR intrinsic set.
- **No Apple API for AIR.** There is no documented way to feed LLVM IR or AIR directly to the Metal driver. The `.metallib` must be accepted by `MTLDevice.newLibraryWithData:`, which is a black box.
- **Limited intrinsic coverage.** Not all `llvm.air.*` intrinsics are reverse-engineered. Simdgroup matrix operations may have gaps.
- **Performance unknown.** The AIR path may produce slower code than MSL (less optimization from Apple's MSL frontend).
- **No tensor-core microscaling.** Apple GPUs don't have hardware E8M0 support — same limitation as Vulkan.
- **Maintenance burden.** Keeping up with Xcode/macOS changes for an undocumented format is costly.

---

## 9. Why this backend is worth investigating

**The payoff is unification:** if the AIR path matures, the shared LLVM IR emitter covers **all four backends** (NVPTX + AMDGPU + SPIR-V + Apple AIR), and the MSL emitter can be retired. Given that Apple's internal GPU compiler stack already uses LLVM IR (AIR is LLVM bitcode), this is a natural convergence.

Even if it never replaces MSL in production, the AIR path provides:
- A reference implementation for testing the shared emitter's backend-independence.
- A faster iteration loop for kernel development (skip MSL syntax, go straight to LLVM IR).
- Insurance against any future Apple MSL deprecation or restriction.

---

## 10. References

- **`LLVM_IR_UNIFICATION_ANALYSIS.md`** — evidence that all backends consume LLVM IR; AIR research as Phase 5.
- **`CUDA_BACKEND_SPEC.md`** — shared `CodegenBackend` trait and Rust API design.
- **llvm-to-air** — https://github.com/sueszli/llvm-to-air — Python-based LLVM IR → AIR → metallib.
- **xDSL MPS backend** — https://docs.xdsl.dev/reference/backend/mps/ — MLIR → MPS dialect → AIR → metallib.
- **CuMetal** — https://github.com/Lulzx/cuda-metal — CUDA/LLVM IR → AIR → metallib with `cumetal-air-emitter`.
- **AIR is LLVM bitcode (reverse engineering)** — https://worthdoingbadly.com/metalbitcode/
- **MetalLibraryArchive** — https://github.com/YuAo/MetalLibraryArchive — `.metallib` parser/extractor.
- **Apple LLVM GPU Compiler (2017 talk)** — https://llvm.org/devmtg/2017-10/slides/Chandrasekaran-Maggioni-Apple%20LLVM%20GPU%20Compiler.pdf
- **Apple Metal shader converter (official, DXIL → Metal IR)** — https://developer.apple.com/metal/shader-converter/

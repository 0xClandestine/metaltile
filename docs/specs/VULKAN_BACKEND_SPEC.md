# Vulkan / SPIR-V Backend Spec

**Status:** 📋 Revised (2026-06-05) — unified LLVM IR codegen, third `CodegenBackend` impl alongside NVIDIA and AMD
**Previous:** original proposed GLSL/rspirv path
**See also:** [`LLVM_IR_UNIFICATION_ANALYSIS.md`](LLVM_IR_UNIFICATION_ANALYSIS.md) — evidence that NVIDIA, AMD, and Vulkan all consume LLVM IR via their respective LLVM backends, enabling a single shared emitter.
**See also:** [`CUDA_BACKEND_SPEC.md`](CUDA_BACKEND_SPEC.md) — the shared `CodegenBackend` trait and Rust API design; this spec is the Vulkan `CodegenBackend` impl, the third peer.

**Scope:** Add a **portable** GPU backend that lowers MetalTile's `#[kernel]` DSL / IR to LLVM IR compiled via `llc` targeting SPIR-V, dispatched through Vulkan compute — one backend that runs across AMD, NVIDIA, Intel, Qualcomm Adreno, ARM Mali, and Apple (via MoltenVK).
**Out of scope:** model loading, graph execution, checkpoint readers — MetalTile is an optimized-kernel generator, not an inference engine.

---

## 1. Positioning — the portability target

CUDA (`CUDA_BACKEND_SPEC.md`) and AMD/ROCm (`AMD_BACKEND_SPEC.md`) are **per-vendor, peak-perf** backends — they reach tensor/matrix cores and hardware microscaling directly. Vulkan is the opposite trade:

> **One backend, (almost) every GPU.** A single SPIR-V/Vulkan backend covers AMD, NVIDIA, **Intel (Arc/Xe), Qualcomm Adreno (Android), ARM Mali**, and Apple GPUs **via MoltenVK** — at the cost of vendor-specific peak features.

The key insight that makes this cheap: **LLVM ships a SPIR-V backend** (`llc -mtriple=spirv64-unknown-vulkan`). The same shared LLVM IR emitter that serves NVPTX (NVIDIA) and AMDGPU (AMD) also serves SPIR-V. The Vulkan backend is:

- A `TargetProfile::spirv(...)` construction (SPIR-V intrinsics, subgroup handling, Vulkan OS target).
- A `SpirvBackend` `CodegenBackend` impl over the shared emit-and-compile pipeline.
- A `VulkanDevice` `Device` impl over the Vulkan API.

Prior art validates this approach:
- **khal** ([dimforge/khal](https://github.com/dimforge/khal)) — same Rust shader compiles to SPIR-V (WebGPU/Vulkan), PTX (CUDA), and CPU.
- **clspv** (Google) — compiles OpenCL C → LLVM IR → SPIR-V → Vulkan compute in production.
- **Khronos SPIRV-LLVM-Translator** — bi-directional LLVM IR ↔ SPIR-V translation.

---

## 2. Goals / Non-goals

### Goals

- A `CodegenBackend` impl for Vulkan that emits LLVM IR text, compiled via `llc -mtriple=spirv64-unknown-vulkan` to SPIR-V binary, loaded into a `VkShaderModule`.
- A `Device` impl (`VulkanDevice`) over the Vulkan compute API (`ash` or `vulkano`): instance/device, descriptor sets, storage buffers, compute pipeline, `vkCmdDispatch`, fences.
- Reuse the IR, the `#[kernel]` macro, the **entire `quant::{codec,format}` layer**, and the **shared LLVM IR emitter** (parameterized by `TargetProfile::spirv(...)`).
- **Runtime feature detection + graceful fallback** for non-guaranteed bits: subgroup size, cooperative matrix, fp16/int8 — the heart of portability.
- `spirv-val` validation on all emitted SPIR-V modules in CI.

### Non-goals

- Model execution / weight loading (engine concern).
- Matching a native backend's peak perf — Vulkan trades peak for reach.
- Hardware E8M0 microscaling (no portable Vulkan path yet — software dequant only).

---

## 3. Unified Rust API — shared with NVIDIA and AMD backends

All types defined in [`CUDA_BACKEND_SPEC.md §4`](CUDA_BACKEND_SPEC.md#4-unified-rust-api-design) apply. The Vulkan-specific pieces are:

### 3.1 `TargetProfile::spirv(…)` construction

```rust
/// Vulkan/SPIR-V GPU profile targeting a specific Vulkan version.
///
/// # Examples
///
/// ```rust
/// // Vulkan 1.3 (SPIR-V 1.6) — most portable modern target
/// let profile = TargetProfile::spirv(SpirvVersion::V1_6);
/// ```
impl TargetProfile {
    pub fn spirv(version: SpirvVersion) -> Self;
}

/// SPIR-V logical version for the `-mtriple` suffix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpirvVersion(u32);

impl SpirvVersion {
    pub const V1_3: Self = SpirvVersion(0x010300);
    pub const V1_6: Self = SpirvVersion(0x010600);

    /// Target triple suffix, e.g. `"vulkan1.3"`, `"vulkan"`.
    pub fn vulkan_triple_suffix(self) -> &'static str;
}
```

### 3.2 `SpirvBackend` — the Vulkan `CodegenBackend` impl

```rust
/// Vulkan SPIR-V codegen backend.
///
/// Emits LLVM IR targeting the SPIR-V backend, compiled by `llc` with
/// `-mtriple=spirv64-unknown-vulkan`. Matrix operations use SPIR-V
/// `SPV_KHR_cooperative_matrix` where available, falling back to
/// workgroup-level reductions.
pub struct SpirvBackend { /* private fields */ }

impl SpirvBackend {
    /// Create a new Vulkan backend targeting the given SPIR-V version.
    pub fn new(version: SpirvVersion) -> Self;
}

impl CodegenBackend for SpirvBackend {
    fn profile(&self) -> &TargetProfile;
    fn emit_llvm_ir(&self, kernel: &Kernel) -> LmResult<String>;
    fn compile(&self, llvm_ir: &str) -> Result<CompiledKernel, CompileError>;
    fn name(&self) -> &'static str { "vulkan" }
}
```

### 3.3 `VulkanDevice` — the runtime `Device` impl

```rust
/// Vulkan GPU device.
///
/// Wraps the Vulkan compute API: `vkCreateShaderModule`, compute pipeline,
/// descriptor sets, `vkCmdDispatch`, fences. Built on `ash` (thin FFI) or
/// `vulkano` (safe).
pub struct VulkanDevice { /* private: instance, device, queues, descriptor pools */ }

impl Device for VulkanDevice { /* see CUDA_BACKEND_SPEC.md §4.6 */ }

impl VulkanDevice {
    /// Open the first suitable compute-capable GPU.
    pub fn new() -> Result<Self, DeviceError>;

    /// Query the subgroup size for the selected device.
    pub fn subgroup_size(&self) -> u32;

    /// Whether `VK_KHR_cooperative_matrix` is supported.
    pub fn has_cooperative_matrix(&self) -> bool;

    /// Whether `VK_KHR_shader_float16_int8` is supported.
    pub fn has_fp16(&self) -> bool;
}
```

### 3.4 Error types

Vulkan-specific errors fold into the shared `CompileError`, `AllocError`, `TransferError`, and `DispatchError` types defined in [`CUDA_BACKEND_SPEC.md §4`](CUDA_BACKEND_SPEC.md#4-unified-rust-api-design). Additional Vulkan-specific variants may be added under `#[non_exhaustive]`.

---

## 4. DSL → LLVM IR op mapping (Vulkan specifics)

The shared LLVM IR emitter handles all common compute constructs. These are the SPIR-V-specific intrinsics and conventions from [`TargetProfile::spirv(...)`](CUDA_BACKEND_SPEC.md#41-targetprofile--opaque-backend-descriptor):

| DSL / IR construct | NVIDIA (NVVM) | AMD (AMDGPU) | Vulkan (SPIR-V via LLVM) |
|---|---|---|---|
| Target triple | `nvptx64-nvidia-cuda` | `amdgcn-amd-amdhsa` | `spirv64-unknown-vulkan` |
| Kernel calling conv | `ptx_kernel` | `amdgcn_kernel` | `spir_kernel` |
| Thread ID (x/y/z) | `@llvm.nvvm.read.ptx.sreg.tid.*` | `@llvm.amdgcn.workitem.id.*` | `__spirv_BuiltInLocalInvocationId` |
| Block ID (x/y/z) | `@llvm.nvvm.read.ptx.sreg.ctaid.*` | `@llvm.amdgcn.workgroup.id.*` | `__spirv_BuiltInWorkgroupId` |
| Subgroup size | `@llvm.nvvm.read.ptx.sreg.warpsize()` → 32 | `@llvm.amdgcn.wavefrontsize()` → 32/64 | `__spirv_BuiltInSubgroupSize` (runtime, 8–64) |
| Barrier | `@llvm.nvvm.barrier0()` | `@llvm.amdgcn.s.barrier()` | `__spirv_ControlBarrier` |
| Shuffle | `@llvm.nvvm.shfl.sync.i32(...)` | `@llvm.amdgcn.permlane16_32(...)` | `__spirv_GroupNonUniformShuffle` |
| Shared memory | `addrspace(3)` | `addrspace(3)` | `addrspace(3)` — matches! |
| FMA | `llvm.fma.*` | `llvm.fma.*` | `llvm.fma.*` — same |
| Math lib | `__nv_expf` (libdevice) | `__ocml_exp_f32` (OCML) | GLSL.std.450 `Exp` (via `__spirv_*`) |
| Tensor/coop matrix | `@llvm.nvvm.hmma.*` / `tcgen05.*` | `@llvm.amdgcn.mfma.*` / `wmma.*` | `SPV_KHR_cooperative_matrix` (optional, runtime-query) |
| Hardware microscaling | Blackwell `tcgen05` | CDNA4 MX MFMA | **None** — software dequant only |
| Kernel annotation | `!nvvm.annotations` | `"kernel"` fn attr | `!spirv.ExecutionMode` metadata |
| SPIR-V extensions | — | — | `-spirv-ext=+SPV_KHR_cooperative_matrix,...` via `llc` flag |

---

## 5. What's genuinely different for Vulkan

### 5.1 Subgroup size is variable AND not guaranteed — the central hazard

This is the AMD wave32/64 problem taken to its limit: Vulkan's **subgroup** has **no fixed, portable size**:
- NVIDIA 32, AMD 32 or 64, **Intel 8/16/32, Adreno/Mali vary**, and a driver may pick per-dispatch.
- Subgroup ops (`subgroupAdd`, `subgroupShuffle`, …) require Vulkan 1.1 subgroup support and the relevant `VkSubgroupFeatureFlagBits` — themselves **optional**.

Two coping strategies, used together:
- **`VK_EXT_subgroup_size_control`**: query the supported range and *require* a specific subgroup size at pipeline creation where the device allows it.
- **Subgroup-agnostic reductions**: lower `reduce_sum`/`simd_sum` to a **workgroup-level** reduction over `shared` memory (a barrier tree) that does **not** depend on subgroup width — portable, slightly slower. Use subgroup ops only as a fast path when size + features are confirmed.

The lowering must make subgroup width a **queried runtime value**, not a compile-time `32`. The shared emitter takes `lane_width` from `TargetProfile`; for Vulkan this becomes a Vulkan device query.

### 5.2 Matrix multiply: `VK_KHR_cooperative_matrix` (optional, queried)

Portable tensor/matrix-core access is **`VK_KHR_cooperative_matrix`** (and `VK_KHR_cooperative_matrix2`). It is:
- **optional** — not all devices expose it (notably weaker on mobile);
- **runtime-shape-queried** — `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` returns supported `{M,N,K}` combinations that differ per vendor and from Metal 8×8 / CUDA 16×16×16 / AMD MFMA.

**Fallback ladder:** `VK_KHR_cooperative_matrix` (preferred) → subgroup-op tiled MMA → plain shared-memory `Reduction` GEMM. The cooperative kernels are reimplemented against this ladder.

### 5.3 Data types are extension-gated

- **fp16**: `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` — widely but not universally available.
- **int8**: same extensions.
- **bf16**: `VK_KHR_shader_bfloat16` is very new/sparse — treat as fall-back-to-fp32 on Vulkan.
- Required SPIR-V capabilities must be declared and device features queried/enabled, else pipeline creation fails. Feature-gating mirrors how CUDA gates by compute capability.

### 5.4 Block-scaled formats on Vulkan

- **Software-decode path (everywhere):** the `quant::codec` decode ports to LLVM IR device functions verbatim (it's arithmetic), feeding dequant-into-`shared` + cooperative-matrix (fp16/int8) or the shared-mem GEMM fallback. This is the universal path.
- **No portable hardware microscaling.** No portable Vulkan equivalent of Blackwell `tcgen05` / CDNA4 MXFP exists today. `mx*`/`mxint*` run via software dequant. (Re-evaluate if/when a `VK_*_cooperative_matrix` scaling extension ships.)
- `nvfp8`/`fp8_*` likewise run software-dequant unless a future FP8 cooperative-matrix type is exposed.

### 5.5 SPIR-V extensions via `llc`

The LLVM SPIR-V backend accepts `-spirv-ext=` flags to enable capabilities:

```bash
llc -mtriple=spirv64-unknown-vulkan \
    -spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_shader_float16_int8 \
    kernel.ll -o kernel.spvt
```

The `SpirvBackend` selects required extensions based on `TargetProfile` and the kernel's needs. This is cleaner than manually declaring SPIR-V capabilities in IR metadata — the LLVM backend handles the mapping.

---

## 6. Compilation & dispatch pipeline

```
Kernel IR
  │
  ▼
emit_llvm_ir()  ─── produces .ll text ─── shared with NVIDIA + AMD
  │
  ▼
SpirvBackend::compile()
  │
  ├── llc -mtriple=spirv64-unknown-vulkan
  │       -spirv-ext=<kernel-required extensions>
  │       kernel.ll → kernel.spvt
  │
  └── VulkanDevice:
        vkCreateShaderModule → VkShaderModule
        VkPipelineShaderStageCreateInfo (stage = VK_SHADER_STAGE_COMPUTE_BIT)
        VkComputePipelineCreateInfo → vkCreateComputePipelines
        VkDescriptorSet + VkWriteDescriptorSet (storage buffers)
        vkCmdBindPipeline + vkCmdBindDescriptorSets
        vkCmdDispatch(grid_x, grid_y, grid_z)
        vkQueueSubmit + vkWaitForFences
        vkMapMemory / vkGetBufferMemoryAddress (readback)
```

- **Compilation:** `llc` subprocess (same as NVIDIA/AMD; shared code path).
- **Validation:** `spirv-val` on all emitted SPIR-V modules in CI.
- **Runtime dispatch:** via `ash` (thin FFI) or `vulkano` (safe Rust bindings).

---

## 7. Implementation phases

1. **Seam + SPIR-V smoke kernel.** `SpirvBackend` impl emitting LLVM IR for an elementwise kernel; `llc -mtriple=spirv64-unknown-vulkan` compilation; `VulkanDevice` over `ash`; one `#[test_kernel]` green on a desktop GPU. Run `spirv-val` on emitted modules.
2. **Portable reductions.** Workgroup/shared-memory reductions independent of subgroup width; bring up `Reduction`-mode families (dequant, qgemv, rms-norm, gather, conv, flash-scalar). Retires the §5.1 hazard early.
3. **Subgroup fast path.** Add `VK_EXT_subgroup_size_control` + subgroup-op reductions as a queried fast path over the portable baseline.
4. **Cooperative-matrix MMA.** `VK_KHR_cooperative_matrix` with runtime shape-query + the §5.2 fallback ladder; software-dequant block-scaled.
5. **Feature/dtype gating + breadth.** fp16/int8 extension gating; validate across ≥3 vendors (NVIDIA, AMD, Intel) + MoltenVK on Apple.
6. **CLI + CI.** `--target {metal,cuda,hip,vulkan}` across build/test/bench; a multi-vendor (or software-rasterizer, e.g. SwiftShader/lavapipe) CI lane; roofline device-spec rows where queryable.

---

## 8. Risks / open questions

- **Subgroup-size variability (§5.1).** No portable fixed width; the portable reduction path is mandatory, subgroup ops are an optional fast path.
- **Optional everything.** Cooperative matrix, fp16/int8, subgroup arithmetic are all extensions/features — every kernel needs feature-detect + fallback, making the runtime more conditional than CUDA/AMD.
- **No hardware microscaling (§5.4).** Block-scaled = software dequant on Vulkan; the E8M0 hardware payoff stays with the native backends.
- **Perf ceiling.** Vulkan generally trails a tuned native backend on the same GPU (less vendor-specific tensor-core reach, more portable-but-generic codegen).
- **Driver fragmentation.** Behavior/perf vary widely across vendors + driver versions, especially mobile (Adreno/Mali) — broad testing required.
- **MoltenVK is a translation layer**, not native — useful for reach/CI on Apple but slower than MetalTile's own Metal backend; it's a fallback, not the Apple path.
- **`llc` SPIR-V backend maturity.** The SPIR-V backend was promoted to official target in LLVM 19.x. The Vulkan OS target was added in May 2026. Feature coverage is active but may have gaps compared to the mature NVPTX/AMDGPU backends.

---

## 9. Why this backend earns its place

It's the **breadth** backend: one SPIR-V/Vulkan target reaches Intel, Qualcomm, ARM, and any GPU without a native MetalTile backend (plus Apple via MoltenVK), reusing the IR + `#[kernel]` macro + the full `quant` codec + the **shared LLVM IR emitter**. It complements — not replaces — the peak-perf native backends: Metal/CUDA/HIP for the vendor you're on, Vulkan for everywhere else and as the vendor-neutral baseline.

Because the LLVM IR emitter is already shared with NVIDIA and AMD, adding Vulkan costs approximately:
- One `TargetProfile::spirv(...)` constructor (intrinsic names, SPIR-V extensions).
- One `SpirvBackend` struct (thin wrapper over the shared pipeline, ~100 lines).
- One `VulkanDevice` over `ash` (the main implementation effort).
- Subgroup-variability handling (the genuinely new Vulkan work, already called out in §5.1).

---

## 10. References

- **`LLVM_IR_UNIFICATION_ANALYSIS.md`** — evidence that NVIDIA, AMD, and Vulkan all consume LLVM IR via their respective LLVM backends.
- **`CUDA_BACKEND_SPEC.md`** — the shared `CodegenBackend` trait and Rust API design.
- **`AMD_BACKEND_SPEC.md`** — the AMD `CodegenBackend` impl, sharing the same LLVM IR emitter.
- **LLVM SPIR-V Backend Usage Guide** — https://releases.llvm.org/23.0.0/docs/SPIRVUsage.html
- **LLVM PR #196101 — `vulkan` as SPIR-V OS target** — https://github.com/llvm/llvm-project/pull/196101
- **LLVM PR #174910 — SPIR-V `gpuintrin.h` support** — https://github.com/llvm/llvm-project/pull/174910
- **Vulkan compute** — `VkPipeline` (compute), descriptor sets, `vkCmdDispatch`. Rust runtimes: **`ash`** (thin FFI), **`vulkano`** (safe).
- **SPIR-V** — **`spirv-val`** (validation), extension registry.
- **Subgroups** — Vulkan 1.1 subgroup ops + `VkSubgroupFeatureFlagBits`; **`VK_EXT_subgroup_size_control`** (§5.1).
- **Cooperative matrix** — **`VK_KHR_cooperative_matrix`**, `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (§5.2).
- **Dtypes** — `VK_KHR_shader_float16_int8`, `VK_KHR_16bit_storage` / `8bit_storage`, `VK_KHR_shader_bfloat16` (§5.3).
- **MoltenVK** — Vulkan-on-Metal (Apple reach/CI).
- **clspv (Google, OpenCL C → LLVM → SPIR-V → Vulkan)** — https://github.com/google/clspv
- **khal (dimforge, write once → SPIR-V + PTX + CPU)** — https://github.com/dimforge/khal
- **Kompute** — GPU compute framework for Vulkan.
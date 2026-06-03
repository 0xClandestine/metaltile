# Vulkan / SPIR-V Backend Spec

**Status:** 📋 Proposed (design only; no implementation yet)
**Scope:** Add a **portable** GPU backend that lowers MetalTile's `#[kernel]` DSL /
IR to **SPIR-V** and dispatches it through **Vulkan compute** — one backend that
runs across AMD, NVIDIA, Intel, Qualcomm Adreno, ARM Mali, and Apple (via
MoltenVK).
**Out of scope:** model loading, graph execution, checkpoint readers — MetalTile
is an optimized-kernel generator, not an inference engine.

> Read `CUDA_BACKEND_SPEC.md` (the backend-seam design) and `AMD_BACKEND_SPEC.md`
> (the wavefront-width hazard) first. This spec reuses the same seam — IR + macro +
> `quant::{codec,format}` unchanged — and covers what's **Vulkan-specific**.

---

## 1. Positioning — the portability target

CUDA (`CUDA_BACKEND_SPEC.md`) and HIP/ROCm (`AMD_BACKEND_SPEC.md`) are
**per-vendor, peak-perf** backends — they reach tensor/matrix cores and hardware
microscaling directly. Vulkan is the opposite trade:

> **One backend, (almost) every GPU.** A single SPIR-V/Vulkan backend covers AMD,
> NVIDIA, **Intel (Arc/Xe), Qualcomm Adreno (Android), ARM Mali**, and Apple GPUs
> **via MoltenVK** — at the cost of vendor-specific peak features.

So Vulkan is the **reach / fallback** backend, not a replacement for the
vendor-specific ones:
- Use **Metal** on Apple, **CUDA** on NVIDIA, **HIP** on AMD for maximum perf.
- Use **Vulkan** to reach everything *else* (Intel, mobile/embedded, ARM), and as
  a vendor-neutral baseline where a native backend doesn't exist yet.

A second reason it's attractive: **SPIR-V is an IR**, so MetalTile's IR → SPIR-V is
an **IR-to-IR lowering** (via `rspirv`), not a text-emit-then-reparse step — a more
natural structural fit than MSL/CUDA-C++ text, and a good candidate for the
*reference* portable backend.

## 2. Goals / Non-goals

**Goals**
- A `spirv` codegen target lowering the IR to SPIR-V (direct via `rspirv`; or GLSL/
  HLSL → SPIR-V via `shaderc`/`glslang` as an alternative).
- A `vulkan` runtime `Device` impl (instance/device, `VkShaderModule`, compute
  `VkPipeline`, descriptor sets over storage `VkBuffer`s, `vkCmdDispatch`, fences)
  — via `ash` (thin) or `vulkano` (safe).
- Reuse the IR, the `#[kernel]` macro, and the **entire `quant::{codec,format}`
  layer unchanged**.
- **Runtime feature detection + graceful fallback** for the non-guaranteed bits
  (subgroup size, cooperative matrix, fp16/int8) — the heart of portability.

**Non-goals**
- Model execution / weight loading (engine concern).
- Matching a native backend's peak perf — Vulkan trades peak for reach.
- Hardware E8M0 microscaling (no portable Vulkan path yet — software dequant).

## 3. What's shared with the CUDA/AMD backends

The backend seam (`CUDA_BACKEND_SPEC.md §4.1`) and the bulk of the DSL→target
mapping apply. The structural analogs:

| Concept | CUDA / HIP | Vulkan / SPIR-V |
|---|---|---|
| dispatch | `<<<grid,block>>>` / `hipLaunchKernel` | `vkCmdDispatch(gx,gy,gz)` |
| workgroup id | `blockIdx.{x,y,z}` | `gl_WorkGroupID` |
| local thread id | `threadIdx.{x,y,z}` | `gl_LocalInvocationID` |
| workgroup size (`lsize`) | `blockDim` | `local_size_*` (SPIR-V exec mode / spec-constant) |
| shared memory | `__shared__` | `shared` (Workgroup storage class) |
| buffers | device pointers | storage `VkBuffer` + descriptor sets |
| math | `__expf`/`exp2f`/… | SPIR-V `ExtInst` (GLSL.std.450: `Exp`, `Exp2`, `InverseSqrt`) |
| runtime compile | NVRTC / hipRTC | offline SPIR-V (or `shaderc` at runtime) + `vkCreateShaderModule` |
| decode intrinsics | device fns | SPIR-V arithmetic (ports verbatim) |

`program_id` / `tid` / `lsize`, `Grid3D`/`Reduction` modes, `threadgroup_alloc`,
the decode intrinsics, and the whole `quant` layer carry over.

## 4. What's genuinely different for Vulkan

### 4.1 Subgroup size is variable AND not guaranteed — the central hazard

This is the AMD wave32/64 problem *taken to its limit*: Vulkan's **subgroup**
(the warp/simdgroup/wavefront analog) has **no fixed, portable size**:
- NVIDIA 32, AMD 32 or 64, **Intel 8/16/32, Adreno/Mali vary**, and a driver may
  even pick per-dispatch.
- Subgroup *ops* (`subgroupAdd`, `subgroupShuffle`, …) require **Vulkan 1.1
  subgroup support** and the relevant `VkSubgroupFeatureFlagBits` (arithmetic,
  shuffle, …) — themselves **optional**.

MetalTile's pervasive **32-lane** assumption cannot hold. Two coping strategies,
used together:
- **`VK_EXT_subgroup_size_control`**: query the supported range and *require* a
  specific subgroup size at pipeline creation (`VkPipelineShaderStageRequired­
  SubgroupSizeCreateInfo`) where the device allows it — recover a known width.
- **Subgroup-agnostic reductions**: lower `reduce_sum`/`simd_sum` to a
  **workgroup-level** reduction over `shared` memory (a barrier tree) that does
  **not** depend on subgroup width — portable, slightly slower. Use subgroup ops
  only as a fast path when size + features are confirmed.

The lowering must therefore make subgroup width a **queried runtime value /
spec-constant**, not a compile-time `32`. Geometry-audit discipline from the other
specs carries over and matters most here.

### 4.2 Matrix multiply: `VK_KHR_cooperative_matrix` (optional, queried)

The portable tensor/matrix-core access is **`VK_KHR_cooperative_matrix`** (with
the newer `VK_KHR_cooperative_matrix2` and the vendor `VK_NV_*` variants). It is:
- **optional** — not all devices/drivers expose it (notably weaker on mobile);
- **runtime-shape-queried** — `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`
  returns the supported `{M,N,K} × {A,B,C,result} × type` combinations, which
  **differ per vendor** (and differ again from Metal 8×8 / CUDA 16×16×16 / AMD
  MFMA). The MMA kernels must pick a supported tile at runtime or fall back.

**Fallback ladder for the MMA / cooperative kernels:**
`VK_KHR_cooperative_matrix` (preferred) → subgroup-op tiled MMA → plain
shared-memory `Reduction` GEMM. The `mpp::`/`InlineMsl` Metal coop kernels are
reimplemented against this ladder (same "doesn't auto-port" caveat as CUDA/AMD).

### 4.3 Data types are extension-gated

- **fp16**: `VK_KHR_shader_float16_int8` (arithmetic) + `VK_KHR_16bit_storage`
  (buffers) — widely but **not universally** available.
- **int8** storage/arithmetic: same `shader_float16_int8` + `8bit_storage`.
- **bf16**: `VK_KHR_shader_bfloat16` is **very new / sparse** — treat bf16 as a
  fall-back-to-fp32 path on Vulkan for now.
- Required SPIR-V capabilities (`Float16`, `Int8`, `StorageBuffer16BitAccess`, …)
  must be declared and **the matching device features queried/enabled**, else
  pipeline creation fails. Element-type availability becomes a per-device feature
  gate, mirroring how CUDA gates by compute capability.

### 4.4 Block-scaled formats on Vulkan

- **Software-decode path (everywhere):** the `quant::codec` decode is arithmetic →
  ports to SPIR-V verbatim, feeding dequant-into-`shared` + cooperative-matrix
  (fp16/int8) or the shared-mem GEMM fallback. This is the universal path.
- **No portable hardware microscaling.** There is **no portable Vulkan equivalent
  of Blackwell `tcgen05` / CDNA4 MXFP** today — the E8M0-microscaling hardware
  path is reachable only through the native CUDA/HIP backends. On Vulkan, `mx*`/
  `mxint*` run via software dequant. (Re-evaluate if/when a `VK_*_cooperative_
  matrix` scaling extension ships.)
- `nvfp8`/`fp8_*` likewise run software-dequant unless a future FP8
  cooperative-matrix type is exposed and queried.

## 5. Codegen strategy options

Three ways to produce SPIR-V, mirroring the CUDA spec's C++/NVRTC-vs-`cuda-oxide`
choice:

| Strategy | How | Notes |
|---|---|---|
| **Direct SPIR-V (`rspirv`)** — recommended | IR → SPIR-V module via the `rspirv` builder | IR-to-IR; no text round-trip; full control over capabilities/exec-modes. Best fit for MetalTile's IR. |
| **GLSL/HLSL text → `shaderc`** | emit GLSL compute, compile with `shaderc`/`glslang` | reuses a text-emitter shape (like MSL/CUDA-C++); easy to read/debug; extra compile dep. |
| **Rust → SPIR-V (`rust-gpu`)** | emit Rust device code, compile via Embark's `rust-gpu` | the Vulkan analog of `cuda-oxide`; idiomatic Rust kernels, but heavier toolchain + maturity caveats. On the radar, not the default. |

Recommend **direct `rspirv`** as the primary path (natural IR→IR), with the
GLSL/`shaderc` path as a debugging/bring-up convenience.

## 6. Implementation phases

1. **Seam + SPIR-V smoke kernel.** `Spirv` codegen target (`rspirv`) for an
   elementwise kernel; `VulkanDevice` over `ash` (module → compute pipeline →
   descriptor set → dispatch → readback); one `#[test_kernel]` green on a desktop
   GPU. Run `spirv-val` on emitted modules in CI.
2. **Portable reductions.** Workgroup/`shared`-memory reductions independent of
   subgroup width; bring up `Reduction`-mode families (dequant, qgemv, rms-norm,
   gather, conv, flash-scalar). This retires the §4.1 hazard early.
3. **Subgroup fast path.** Add `VK_EXT_subgroup_size_control` + subgroup-op
   reductions as a queried fast path over the portable baseline.
4. **Cooperative-matrix MMA.** `VK_KHR_cooperative_matrix` with runtime
   shape-query + the §4.2 fallback ladder; software-dequant block-scaled.
5. **Feature/dtype gating + breadth.** fp16/int8 extension gating; validate across
   ≥3 vendors (e.g. NVIDIA, AMD, Intel) + MoltenVK on Apple.
6. **CLI + CI.** `--target vulkan` across build/test/bench; a multi-vendor (or
   software-rasterizer, e.g. SwiftShader/lavapipe) CI lane; roofline device-spec
   rows where queryable.

## 7. Risks / open questions

- **Subgroup-size variability (the big one).** §4.1 — no portable fixed width;
  the portable reduction path is mandatory, subgroup ops are an optional fast path.
- **Optional everything.** cooperative-matrix, fp16/int8, subgroup arithmetic are
  all extensions/features — every kernel needs a feature-detect + fallback, so the
  runtime is more conditional than CUDA/HIP.
- **No hardware microscaling.** block-scaled = software dequant on Vulkan (§4.4);
  the E8M0 hardware payoff stays with the native backends.
- **Perf ceiling.** Vulkan generally trails a tuned native backend on the same GPU
  (less vendor-specific tensor-core reach, more portable-but-generic codegen).
- **Driver fragmentation.** behavior/perf vary widely across vendors + driver
  versions, especially mobile (Adreno/Mali) — broad testing required.
- **MoltenVK is a translation layer**, not native — useful for reach/CI on Apple
  but slower than MetalTile's own Metal backend; it's a fallback, not the Apple
  path.
- **Tooling maturity:** `ash`/`vulkano`/`rspirv` are mature; `rust-gpu` is less so
  (if that codegen strategy is chosen).

## 8. Why this backend earns its place

It's the **breadth** backend: one SPIR-V/Vulkan target reaches Intel, Qualcomm,
ARM, and any GPU without a native MetalTile backend (plus Apple via MoltenVK),
reusing the IR + `#[kernel]` macro + the full `quant` codec. It complements — not
replaces — the peak-perf native backends: Metal/CUDA/HIP for the vendor you're on,
Vulkan for everywhere else and as the vendor-neutral baseline.

## 9. References

- **Vulkan compute** — `VkPipeline` (compute), descriptor sets, `vkCmdDispatch`;
  the §3 host-mapping. Rust runtimes: **`ash`** (thin FFI), **`vulkano`** (safe).
- **SPIR-V** — **`rspirv`** (Rust SPIR-V builder/assembler; the recommended direct
  emitter), `shaderc`/`glslang` (GLSL/HLSL→SPIR-V), `spirv-val` (validation),
  **`rust-gpu`** (Rust→SPIR-V, the `cuda-oxide` analog — §5).
- **Subgroups** — Vulkan 1.1 subgroup ops + `VkSubgroupFeatureFlagBits`;
  **`VK_EXT_subgroup_size_control`** (§4.1).
- **Cooperative matrix** — **`VK_KHR_cooperative_matrix`** (+ `_cooperative_matrix2`,
  `VK_NV_cooperative_matrix*`), `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`
  (§4.2).
- **Dtypes** — `VK_KHR_shader_float16_int8`, `VK_KHR_16bit_storage` /
  `8bit_storage`, `VK_KHR_shader_bfloat16` (§4.3).
- **MoltenVK** — Vulkan-on-Metal (Apple reach/CI).
- Prior art for GPU LLM compute on Vulkan: llama.cpp's Vulkan backend, Kompute,
  `wgpu` (WebGPU over Vulkan/Metal/DX12).
- `CUDA_BACKEND_SPEC.md` / `AMD_BACKEND_SPEC.md` — the shared seam + the
  wavefront/subgroup-width lineage this spec extends.

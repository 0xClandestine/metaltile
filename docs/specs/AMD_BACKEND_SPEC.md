# AMD / ROCm Backend Spec

**Status:** 📋 Revised (2026-06-05) — unified LLVM IR codegen, peer `CodegenBackend` impl alongside NVIDIA
**Previous:** original proposed HIP C++ + hipRTC delta on CUDA spec
**See also:** [`LLVM_IR_UNIFICATION_ANALYSIS.md`](LLVM_IR_UNIFICATION_ANALYSIS.md) — evidence that both NVIDIA and AMD consume LLVM IR directly, enabling a single shared emitter.
**See also:** [`CUDA_BACKEND_SPEC.md`](CUDA_BACKEND_SPEC.md) — the NVIDIA `CodegenBackend` impl; this spec is its AMD peer, sharing the same trait and LLVM IR emitter architecture.

**Scope:** Add an AMD GPU backend so MetalTile's `#[kernel]` DSL / IR lowers to **ROCm** (AMD GPUs) through the same `CodegenBackend` trait as the NVIDIA backend.
**Out of scope:** model loading, graph execution, checkpoint readers — MetalTile is an optimized-kernel generator, not an inference engine.

---

## 1. Motivation

AMD GPUs are custom-kernel-programmable, so the same per-kernel DSL model as CUDA applies. The key insight that makes this cheap: **AMD's GPU compiler stack (the AMDGPU LLVM backend) consumes LLVM IR directly**, just like NVIDIA's NVPTX backend (see [`LLVM_IR_UNIFICATION_ANALYSIS.md`](LLVM_IR_UNIFICATION_ANALYSIS.md)). The HIP C++ → hipRTC pipeline is a detour through a C++ frontend that we can skip entirely.

By emitting LLVM IR instead of HIP C++, the AMD backend:
- **Shares the same emitter** as the NVIDIA backend — one `CodegenBackend` trait, two impls.
- **Eliminates the hipRTC dependency** — no ROCm installation required at compile time; just `llc` (from any LLVM build) targeting the AMDGPU backend.
- **Gets LLVM optimization passes for free** — the same pipeline that serves NVIDIA also serves AMD.

The PR-#2 precision payoff repeats: **AMD CDNA4 (Instinct MI350/MI355X, 2025) has hardware OCP-microscaling tensor cores (MXFP4 / MXFP6 / MXFP8)** — so the `mx*` / `mxint*` E8M0-block-32 formats map onto AMD matrix cores through the same LLVM intrinsic mechanism used for Blackwell.

---

## 2. Goals / Non-goals

### Goals

- A `CodegenBackend` impl for AMD that emits LLVM IR text, compiled via `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfxXXXX` to a code object, loaded via the ROCm runtime.
- A `Device` impl (`AmdDevice`) over the ROCm runtime API (`hipModuleLoad`, `hipLaunchKernel`, `hipMalloc`).
- Reuse the IR, the `#[kernel]` macro, the **entire `quant::{codec,format}` layer**, and the **shared LLVM IR emitter** (parameterized by `TargetProfile::amd(...)`).
- Correct handling of the **wavefront-size split (32 vs 64)** — the one structural difference from NVIDIA/Apple's fixed 32-lane group.
- Map block-scaled formats onto AMD matrix cores (MFMA / WMMA) via LLVM intrinsics, with the software dequant path as the universal fallback.

### Non-goals

- Model execution / weight loading (engine concern).
- Day-one parity for the `mpp::`/`InlineMsl` cooperative kernels (reimplement via rocWMMA / Composable Kernel — same situation as CUDA's CUTLASS reimpl).
- Bit-exact match to Metal/CUDA outputs — accuracy-parity vs the CPU oracle.

---

## 3. Unified Rust API — shared with the NVIDIA backend

All types defined in [`CUDA_BACKEND_SPEC.md §4`](CUDA_BACKEND_SPEC.md#4-unified-rust-api-design) apply to the AMD backend. The AMD-specific pieces are:

### 3.1 `TargetProfile::amd(…)` construction

```rust
/// AMD GPU profile targeting a specific gfx architecture.
///
/// # Examples
///
/// ```rust
/// let profile = TargetProfile::amd(GfxArch::new(94, 2)); // MI300
/// let wave64    = profile.requires_wave64();
/// ```
impl TargetProfile {
    pub fn amd(gfx_arch: GfxArch) -> Self;
}

impl TargetProfile {
    /// Whether this AMD profile requires wavefront-64 mode.
    ///
    /// Returns `true` for CDNA/GCN (gfx9xx), `false` for RDNA (gfx10xx+).
    pub fn requires_wave64(&self) -> bool;
}
```

### 3.2 `AmdBackend` — the AMD `CodegenBackend` impl

```rust
/// AMD GPU codegen backend.
///
/// Emits LLVM IR targeting the AMDGPU backend, compiled by `llc` with the
/// `amdgcn-amd-amdhsa` triple. Matrix-core operations use AMD-specific
/// LLVM intrinsics (`@llvm.amdgcn.mfma.*`, `@llvm.amdgcn.wmma.*`).
pub struct AmdBackend { /* private fields */ }

impl AmdBackend {
    /// Create a new AMD backend targeting the given gfx architecture.
    pub fn new(gfx_arch: GfxArch) -> Self;
}

impl CodegenBackend for AmdBackend {
    fn profile(&self) -> &TargetProfile;
    fn emit_llvm_ir(&self, kernel: &Kernel) -> LmResult<String>;
    fn compile(&self, llvm_ir: &str) -> Result<CompiledKernel, CompileError>;
    fn name(&self) -> &'static str { "hip" }
}
```

### 3.3 `AmdDevice` — the runtime `Device` impl

```rust
/// AMD GPU device, backed by the ROCm runtime API.
///
/// Wraps `hipModuleLoad`, `hipLaunchKernel`, `hipMalloc`, etc. through
/// raw FFI (no mature Rust ROCm bindings analogous to cuda-oxide exist).
pub struct AmdDevice { /* private: hipModule, hipCtx, device properties */ }

impl Device for AmdDevice { /* see CUDA_BACKEND_SPEC.md §4.6 */ }
```

### 3.4 Error types

AMD-specific errors are folded into the shared `CompileError`, `AllocError`, `TransferError`, and `DispatchError` types defined in [`CUDA_BACKEND_SPEC.md §4`](CUDA_BACKEND_SPEC.md#4-unified-rust-api-design). No additional error variants are needed — `llc` failures and ROCm runtime errors already fit the existing structure.

---

## 4. DSL → LLVM IR op mapping (AMD specifics)

The shared LLVM IR emitter (see [`CUDA_BACKEND_SPEC.md §5`](CUDA_BACKEND_SPEC.md#5-dsl--llvm-ir-op-mapping)) handles mapping for all compute constructs. These are the AMD-specific intrinsics and conventions that differ from NVIDIA:

| DSL / IR construct | NVIDIA (NVVM) | AMD (AMDGPU) |
|---|---|---|
| Target triple | `nvptx64-nvidia-cuda` | `amdgcn-amd-amdhsa` |
| Kernel calling convention | `ptx_kernel` | `amdgcn_kernel` |
| Thread ID (x/y/z) | `@llvm.nvvm.read.ptx.sreg.tid.{x,y,z}` | `@llvm.amdgcn.workitem.id.{x,y,z}` |
| Block ID (x/y/z) | `@llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` | `@llvm.amdgcn.workgroup.id.{x,y,z}` |
| Block dim (x/y/z) | `@llvm.nvvm.read.ptx.sreg.ntid.{x,y,z}` | `@llvm.amdgcn.dispatch.ids` (different mechanism) |
| Warp / wavefront size | `@llvm.nvvm.read.ptx.sreg.warpsize()` → 32 | `@llvm.amdgcn.wavefrontsize()` → 32 or 64 |
| Barrier | `@llvm.nvvm.barrier0()` | `@llvm.amdgcn.s.barrier()` |
| Lane shuffle | `@llvm.nvvm.shfl.sync.i32(...)` | `@llvm.amdgcn.permlane16_32(...)` or `ds_bpermute` |
| Shared memory allocation | `addrspace(3)` global variable | `addrspace(3)` global variable (same!) |
| FMA | `llvm.fma.*` (direct) | `llvm.fma.*` (same) |
| Math lib | `__nv_expf`, `__nv_powf` (libdevice) | `__ocml_exp_f32`, `__ocml_pow_f32` (OCML) |
| Tensor cores (pre-CDNA4) | `@llvm.nvvm.hmma.*` (wmma-style, 16×16×16) | `@llvm.amdgcn.mfma.*` (CDNA) / `@llvm.amdgcn.wmma.*` (RDNA3+) — different shapes |
| Tensor cores (CDNA4) | `@llvm.nvvm.tcgen05.*` (Blackwell) | CDNA4 MX MFMA intrinsics (`@llvm.amdgcn.mfma.*` with MX formats) |
| Kernel annotation | `!nvvm.annotations` metadata | `"kernel"` function attribute + `!amdgpu.annotations` |

---

## 5. What's genuinely different for AMD

### 5.1 Wavefront size 32 **or** 64 — the main hazard

MetalTile kernels assume a **32-lane simdgroup** (Metal) ≙ 32-lane warp (NVIDIA). On AMD this is **not fixed**:

- **RDNA (gfx10/11/12, consumer + some pro)** runs **wave32** for compute — maps cleanly to the existing 32-lane reductions/shuffles.
- **CDNA / GCN (Instinct MI-series, gfx9xx)** is **wave64** — twice the lane count. Any kernel that hard-codes 32 (lane masks, `simd_sum`, reduce-tree widths, MMA fragment mapping) must be re-parameterized.

**Mitigation:** `TargetProfile::requires_wave64()` drives the emitter and reduction lowering. Where kernel logic truly needs 32, target RDNA wave32 or split a wave64 into two 32-lane halves. The geometry-audit discipline (no silent geometry change) carries over from the NVIDIA spec and is *more* important here.

### 5.2 Matrix cores: MFMA (CDNA) vs WMMA (RDNA3+)

- **CDNA matrix cores → MFMA** (`@llvm.amdgcn.mfma.*`). MI300 (gfx942/CDNA3) adds FP8 MFMA; MI350 (gfx950/CDNA4) adds MXFP4/6/8 microscaling MFMA.
- **RDNA3/RDNA4 → WMMA** (`@llvm.amdgcn.wmma.*`); RDNA4 adds FP8.

These have **different fragment shapes** from both Metal `simdgroup_matrix` (8×8) and NVIDIA `wmma` (16×16×16), so the MMA kernels need AMD-specific tiling — the same "re-tile per backend" caveat applies, with one more shape family.

### 5.3 Block-scaled formats on AMD

- **Software-decode path (all AMD GPUs):** the `quant::codec` decode ports to LLVM IR device functions verbatim (it's arithmetic), feeding dequant-into-LDS + MFMA/WMMA.
- **Hardware microscaling (CDNA4 / MI350+):** `mxfp4`/`mxfp8` and `mxint*` map onto the OCP-microscaling MFMA path (E8M0 block-32 scale operands) — the AMD analog of Blackwell `tcgen05`. The host packer is reused unchanged; only kernel-side consumption differs.
- **FP8 (MI300 / RDNA4):** `nvfp8` / `fp8_*` map to native FP8 MFMA/WMMA even pre-CDNA4.

### 5.4 Toolchain targets

| AMD arch | gfx target | Wavefront | Tensor cores | MX support |
|---|---|---|---|---|
| MI200 (CDNA2) | gfx90a | 64 | MFMA (FP16/BF16) | software |
| MI300 (CDNA3) | gfx942 | 64 | MFMA (FP8/FP16/BF16) | software |
| MI350 (CDNA4) | gfx950 | 64 | MFMA (MXFP4/6/8, FP8) | **hardware** |
| RDNA3 (RX 7000) | gfx1100 | 32 | WMMA (FP16/BF16) | software |
| RDNA4 (RX 9000) | gfx120x | 32 | WMMA (FP8) | software |

---

## 6. Implementation phases

1. **Seam + smoke kernel.** `AmdBackend` impl emitting LLVM IR for a trivial elementwise kernel; `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100` compilation; `AmdDevice` via raw HIP runtime FFI; one `#[test_kernel]` green on RDNA wave32.
2. **Wavefront-64 correctness.** Bring reductions/shuffles up on CDNA wave64; parameterize `WARP` via `TargetProfile::requires_wave64()`. This is the gating risk.
3. **Elementwise + reduction families.** dequant, qgemv, rms-norm, gather, conv, flash (scalar) — the bulk of pure-DSL kernels, both wave32 and wave64.
4. **Matrix cores.** MFMA (CDNA) + WMMA (RDNA3+) via `@llvm.amdgcn.mfma.*` / `@llvm.amdgcn.wmma.*` intrinsics; software-dequant block-scaled.
5. **CDNA4 microscaling MFMA.** Hardware `mx*`/`mxint*` path on gfx950+, feature-gated.
6. **Cooperative reimpl + CLI/CI.** rocWMMA/CK equivalents of MPP/NAX kernels; `--target {metal,cuda,hip}`; an AMD CI lane on real CDNA and RDNA hardware; device-spec rows for roofline columns.

---

## 7. Risks / open questions

- **Wavefront 32/64 (§5.1).** The single biggest AMD-specific risk. Budget the wave64 adaptation as a first-class phase.
- **Rust/ROCm ecosystem maturity.** No mature Rust ROCm bindings analogous to `cuda-oxide`. Expect raw FFI over the HIP runtime (`hipModuleLoad`, `hipLaunchKernel`, `hipMalloc`).
- **ROCm platform support.** Linux-centric; official support skews to Instinct (CDNA) + select RDNA pro cards. CI needs real AMD hardware (CDNA *and* RDNA).
- **MFMA/WMMA fragment shapes** differ from Metal/CUDA — another MMA retile + tuning pass.
- **`gfx` fragmentation.** Intrinsics/dtypes vary by arch (FP8 on gfx942+, MXFP on gfx950+, WMMA on gfx1100+) — gate by detected `gfx` like NVIDIA gates by compute capability.
- **Numerics.** Validate the hardware microscaling-MFMA path bit-for-bit vs the software oracle on real CDNA4 before trusting it.

---

## 8. Why this is a low-marginal-cost third backend

Because the LLVM IR emitter is **shared with the NVIDIA backend**, the AMD backend is largely:

- A `TargetProfile::amd(...)` construction (intrinsic names, wavefront width, triple).
- An `AmdBackend` that is a thin wrapper over the shared emit-and-compile pipeline.
- An `AmdDevice` over the HIP runtime FFI.
- Wave64 adaptation and MFMA/WMMA tiling — the genuinely new AMD work.

And the same `mx*`/`mxint*` formats that target Blackwell also target CDNA4 microscaling, so MetalTile's quant matrix already spans Apple GPU (today) + NVIDIA + AMD hardware — all through a single `CodegenBackend` trait.

---

## 9. References

- **`LLVM_IR_UNIFICATION_ANALYSIS.md`** — evidence that NVIDIA and AMD both consume LLVM IR, enabling a shared emitter.
- **`CUDA_BACKEND_SPEC.md`** — the NVIDIA `CodegenBackend` impl and shared Rust API design.
- **LLVM AMDGPU Backend Usage Guide** — https://llvm.org/docs/AMDGPUUsage.html
- **ROCm / HIP** — HIP runtime + hipRTC (runtime compile), `hipcc`/LLVM AMDGPU.
- **rocWMMA / Composable Kernel (CK)** — CUTLASS-class libraries for matrix-core and cooperative-kernel paths.
- **AMD matrix cores** — MFMA (CDNA, `@llvm.amdgcn.mfma.*`), WMMA (RDNA3+, `@llvm.amdgcn.wmma.*`); MI300/gfx942 FP8; MI350/gfx950 (CDNA4) OCP MXFP microscaling.
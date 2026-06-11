# CUDA / NVIDIA Backend Spec

**Status:** 📋 Revised (2026-06-05) — unified LLVM IR codegen, single `CodegenBackend` trait
**Previous:** original proposed CUDA C++ + NVRTC path
**See also:** [`LLVM_IR_UNIFICATION_ANALYSIS.md`](LLVM_IR_UNIFICATION_ANALYSIS.md) — evidence that both NVIDIA and AMD consume LLVM IR directly, enabling a single shared emitter.
**Scope:** Add a second code-generation + runtime backend so MetalTile's existing `#[kernel]` DSL / IR lowers to **CUDA** (NVIDIA GPUs) in addition to Metal/MSL.
**Out of scope:** model loading, graph execution, tokenization, checkpoint readers — MetalTile is an optimized-kernel generator, not an inference engine.

---

## 1. Motivation

MetalTile today is a single-target toolchain: the IR in `metaltile-core` lowers through `metaltile-codegen` to **Metal Shading Language** only, and `metaltile-runtime` dispatches exclusively through Metal. The algorithm IR and the `#[kernel]` DSL are, by contrast, **backend-neutral** — they describe parallel compute (program ids, threadgroup memory, simd reductions, MMA tiles, elementwise math), not Metal specifics.

NVIDIA GPUs are the natural second target because, unlike the ANE (see `ANE_BACKEND_SPEC.md`), they are **directly programmable with custom kernels**. The same per-kernel DSL model applies 1:1. And the precision work in PR #2 lands us in a strong position: the **`mx*` / `mxint*` formats use E8M0 microscaling with block 32 — exactly what NVIDIA Blackwell's 5th-gen tensor cores consume in hardware** (`tcgen05` scaled-MMA for MXFP4/6/8, NVFP4, MXINT8). The quant codec/format layer is pure host Rust and already backend-independent.

**Key insight (see [`LLVM_IR_UNIFICATION_ANALYSIS.md`](LLVM_IR_UNIFICATION_ANALYSIS.md) for the full evidence):** NVIDIA's NVPTX backend and AMD's AMDGPU backend **both consume LLVM IR directly**. The CUDA C++ → NVRTC → PTX pipeline is a detour through a C++ frontend that could be skipped. By emitting LLVM IR instead of C++, we get:

- **One emitter** for both NVIDIA and AMD, not two.
- **Direct LLVM optimization pipeline control**, not whatever NVRTC/hipRTC ships.
- **Fewer toolchain dependencies** — no NVRTC or hipRTC, just `llc` (from any LLVM installation) and `ptxas` (from the CUDA Toolkit).

Apple GPUs remain the exception: Metal does **not** accept LLVM IR, so the MSL emitter stays.

**Goal:** one DSL, N backends — author a kernel once, emit correct code for Apple GPUs (MSL), NVIDIA GPUs (LLVM IR → NVPTX), and AMD GPUs (LLVM IR → AMDGPU) through a shared codegen abstraction.

---

## 2. Goals / Non-goals

### Goals

- A `CodegenBackend` trait with a single `Nvidia` impl that emits LLVM IR text, compiled via `llc` (NVPTX backend) to PTX, then optionally via `ptxas` to cubin.
- A `Device` trait with a `CudaDevice` impl over the CUDA Driver API (`cuModuleLoadData`, `cuLaunchKernel`, `cuMemAlloc`).
- Reuse the IR, the `#[kernel]` macro, and the **entire `quant::{codec,format}` layer unchanged**.
- Map the block-scaled formats onto Blackwell hardware block-scaling where supported (scaled tensor-core MMA), with a portable software-decode path for pre-Blackwell.
- Rust-idiomatic API: traits, opaque types, newtype wrappers, `Result`-based error handling, builder construction.

### Non-goals

- Model execution / weight loading (engine concern, separate project).
- 100% kernel parity on day one — the cooperative `mpp::`/`InlineMsl` kernels (MMA/MPP/NAX) need per-backend reimplementation (see §7); the pure-DSL kernels port through the shared LLVM IR emitter.
- ROCm/AMD is a separate `CodegenBackend` impl — see `AMD_BACKEND_SPEC.md`.
- SPIR-V / Vulkan compute — possible future backend, not this spec.

---

## 3. Current state — what already generalizes vs what is Metal-coupled

| Layer | Crate | Backend-neutral? | Notes |
|---|---|---|---|
| Algorithm IR (`Op`, `Kernel`, `DType`, `Shape`, `ConstExpr`) | `metaltile-core` | **Yes** | `op.rs` is abstract math/parallelism; already references "backends" (plural). |
| `#[kernel]` DSL macro | `metaltile-macros` | **Yes** | Produces IR, not MSL. |
| Quant codec / format / packer | `metaltile-std::quant` | **Yes** | Pure host Rust; the 30-format matrix is layout + arithmetic, no Metal. |
| Codegen | `metaltile-codegen` | **No** | `emit.rs` + `msl/` emit MSL strings directly; no backend seam yet. |
| Runtime | `metaltile-runtime` | **No** | `metal_device.rs`, Metal dispatch/buffers, `gpu_family.rs`. |
| Cooperative kernels (`Op::InlineMsl` with `mpp::`, `coop_tile_*`) | `metaltile-std` | **Partly** | The raw-MSL escape hatch is Metal-only; needs NVIDIA/AMD analogs via inline PTX / LLVM intrinsics. |

The work is **one shared backend seam + two backend impls**, not a rewrite.

---

## 4. Unified Rust API Design

This section defines the shared abstraction that both the NVIDIA and AMD backends implement. All types follow Rust best practices: private fields, builder construction, newtype wrappers, `Result` propagation, documented public APIs.

### 4.1 `TargetProfile` — opaque backend descriptor

A `TargetProfile` encodes everything the shared LLVM IR emitter needs to know about the target. It is constructed through named factory methods, never by struct literals.

```rust
/// Backend-specific parameters that specialize the shared LLVM IR emitter.
///
/// Construct via the named factory methods — do not use struct literals.
/// Fields are private to insulate callers from internal changes.
pub struct TargetProfile { /* private fields */ }

impl TargetProfile {
    /// NVIDIA GPU profile targeting a specific compute capability.
    pub fn nvidia(sm_version: SmVersion) -> Self;

    /// AMD GPU profile targeting a specific gfx architecture.
    pub fn amd(gfx_arch: GfxArch) -> Self;
}
```

### 4.2 `SmVersion` and `GfxArch` — newtype wrappers

```rust
/// An NVIDIA compute capability version, e.g. `SmVersion::new(10, 0)` for Blackwell.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SmVersion(u32);

impl SmVersion {
    pub fn new(major: u8, minor: u8) -> Self;
    pub fn major(self) -> u8;
    pub fn minor(self) -> u8;
}

/// An AMD gfx architecture version, e.g. `GfxArch::new(94, 2)` for MI300.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GfxArch(u32);

impl GfxArch {
    pub fn new(major: u8, minor: u8) -> Self;
    pub fn major(self) -> u8;
    pub fn minor(self) -> u8;
}
```

### 4.3 `CodegenBackend` trait — the single codegen abstraction

```rust
/// A backend that lowers MetalTile IR to a device-executable binary.
///
/// Every backend (Nvidia, Amd) is a `CodegenBackend` impl. The shared logic
/// lives in methods on `TargetProfile`; backends are thin wrappers that
/// provide the profile, emit LLVM IR text, and invoke `llc`.
///
/// # Errors
///
/// All fallible methods return `Result`. Callers handle errors via `?`.
pub trait CodegenBackend: Send + Sync {
    /// Return the target description this backend was configured with.
    fn profile(&self) -> &TargetProfile;

    /// Emit LLVM IR text for `kernel`, parameterized by `self.profile()`.
    ///
    /// The output is valid LLVM IR targeting this backend's triple and CPU.
    fn emit_llvm_ir(&self, kernel: &Kernel) -> LmResult<String>;

    /// Compile LLVM IR text to a device binary.
    ///
    /// On NVIDIA: invokes `llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_XX`
    /// followed by `ptxas` to produce a cubin.
    /// On AMD: invokes `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfxXXXX`.
    fn compile(&self, llvm_ir: &str) -> Result<CompiledKernel, CompileError>;

    /// Human-readable backend identifier, e.g. `"cuda"` or `"hip"`.
    fn name(&self) -> &'static str;
}
```

### 4.4 `CompiledKernel` — opaque compiled artifact

```rust
/// A kernel compiled to a device binary.
///
/// Constructed by `CodegenBackend::compile`. Fields are private; dispatch
/// happens through the `Device` trait.
pub struct CompiledKernel { /* private: device-specific handle */ }
```

### 4.5 `CompileError` — non-exhaustive error enum

```rust
/// Errors from kernel compilation.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompileError {
    /// LLVM IR emission failed (e.g. unsupported IR construct).
    Emission(String),
    /// `llc` subprocess returned a non-zero exit code.
    Llc { stderr: String },
    /// `ptxas` (NVIDIA only) returned a non-zero exit code.
    Ptxas { stderr: String },
    /// I/O error reading/writing temp files.
    Io(std::io::Error),
}
```

### 4.6 `Device` trait — runtime abstraction

```rust
/// A physical or virtual GPU device.
///
/// Implementations are responsible for memory management, kernel dispatch,
/// and synchronization. All fallible methods return `Result`.
pub trait Device: Send + Sync {
    /// Compile a kernel on this device.
    fn compile(&self, kernel: &Kernel) -> Result<CompiledKernel, CompileError>;

    /// Allocate a device buffer of `size` bytes.
    fn alloc(&self, size: u64) -> Result<Buffer, AllocError>;

    /// Upload `data` to `buf`.
    fn upload(&self, buf: &mut Buffer, data: &[u8]) -> Result<(), TransferError>;

    /// Download `buf` contents into `dst`.
    fn readback(&self, buf: &Buffer, dst: &mut [u8]) -> Result<(), TransferError>;

    /// Dispatch `kernel` with the given grid, block, and arguments.
    fn dispatch(
        &self,
        kernel: &CompiledKernel,
        grid: GridSize,
        block: BlockSize,
        args: &[Arg],
    ) -> Result<(), DispatchError>;

    /// Human-readable device identifier, e.g. `"NVIDIA GeForce RTX 5090"`.
    fn name(&self) -> &str;
}
```

### 4.7 Value types

```rust
/// A device buffer handle.
#[derive(Clone, Debug)]
pub struct Buffer { /* private: device-specific pointer + size */ }

/// 3D grid dimensions.
#[derive(Copy, Clone, Debug)]
pub struct GridSize { /* private fields */ }

impl GridSize {
    pub fn new(x: u32, y: u32, z: u32) -> Self;
}

/// 3D thread-block dimensions.
#[derive(Copy, Clone, Debug)]
pub struct BlockSize { /* private fields */ }

impl BlockSize {
    pub fn new(x: u32, y: u32, z: u32) -> Self;
}

/// A kernel argument.
#[derive(Clone, Debug)]
pub struct Arg(/* private: tagged union of device types */);

impl From<&Buffer> for Arg { ... }
impl From<u32> for Arg { ... }
impl From<f32> for Arg { ... }

/// Errors.
#[derive(Debug)]
#[non_exhaustive]
pub enum AllocError { OutOfMemory(u64), DeviceError(String) }

#[derive(Debug)]
#[non_exhaustive]
pub enum TransferError { InvalidSize, DeviceError(String) }

#[derive(Debug)]
#[non_exhaustive]
pub enum DispatchError { InvalidConfig(String), DeviceError(String) }
```

### 4.8 `LmResult` — convenience alias

```rust
/// Convenience result alias for the codegen crate.
pub type LmResult<T> = Result<T, LmError>;

/// Error type for the codegen crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum LmError {
    /// An IR construct was encountered the emitter cannot lower.
    UnsupportedOp(&'static str),
    /// An internal invariant was violated (bug, not user error).
    Internal(String),
    /// I/O error writing IR text.
    Io(std::io::Error),
}

impl From<std::io::Error> for LmError { ... }
```

---

## 5. DSL → LLVM IR op mapping

The shared emitter produces LLVM IR text. Backend-specific details (intrinsic names, address spaces, calling convention) come from `TargetProfile`.

| DSL / IR construct | MSL (Apple, today) | LLVM IR (NVIDIA + AMD, proposed) |
|---|---|---|
| `program_id::<0/1/2>()` | `threadgroup_position_in_grid` | `@llvm.nvvm.read.ptx.sreg.ctaid.*` (NVIDIA) / `@llvm.amdgcn.workgroup.id.*` (AMD) |
| `tid` | `thread_index_in_threadgroup` | `@llvm.nvvm.read.ptx.sreg.tid.*` (NVIDIA) / `@llvm.amdgcn.workitem.id.*` (AMD) |
| `lsize` | `threads_per_threadgroup` | `@llvm.nvvm.read.ptx.sreg.ntid.*` (NVIDIA) / `@llvm.amdgcn.dispatch.*` (AMD) |
| `KernelMode::Grid3D` | `[threads] [grid]` dispatch | `<<<grid, block>>>` via CUDA Driver API |
| `KernelMode::Reduction` | simdgroup reductions | block reduction via warp shuffles + shared memory tree |
| `threadgroup_alloc / _store / _load` | `threadgroup` memory | `addrspace(3)` (shared) on both NVIDIA and AMD |
| `simd_sum` / lane ops | 32-lane simdgroup | 32/64-lane warp shuffle (`llvm.nvvm.shfl.sync.*` / `llvm.amdgcn.permlane.*`) |
| `reduce_sum` | simd + threadgroup | warp-reduce + shared-mem tree |
| `simdgroup_matmul` (8×8) | `simdgroup_matrix` | `@llvm.nvvm.hmma.*` (NVIDIA) / `@llvm.amdgcn.mfma.*` or `wmma.*` (AMD) — re-tiling required |
| `exp` / `exp2` / `rsqrt` / `log` / `sqrt` | `metal::precise::exp` | `llvm.sqrt.*`, `llvm.fma.*`, or libdevice (`__nv_expf`, `__ocml_exp_f32`) |
| `select`, `cast`, bit ops | MSL | LLVM `select`, `bitcast`, `trunc`/`zext`/`sext`, `and`/`or`/`xor` |
| decode intrinsics | MSL preamble helpers | LLVM IR device functions (pure arithmetic, ports verbatim) |

**Key structural match:** Both NVPTX and AMDGPU use address space **3** for shared/LDS memory and address space **1** for global memory. The shared memory model is identical at the LLVM IR level.

---

## 6. The quant formats on NVIDIA

- **Software-decode path (all NVIDIA GPUs):** the `quant::codec` decode is pure arithmetic and becomes LLVM IR device functions. Every block-scaled kernel works on Ampere/Hopper/Ada via dequant-into-shared + tensor-core MMA.
- **Hardware block-scaling (Blackwell, sm_100+):** `mxfp4`/`mxfp8`/`nvfp4` map onto the `llvm.nvvm.tcgen05.*` intrinsics for scaled tensor-core MMA. The E8M0/block-32 layout from PR #2 is the native Blackwell microscaling layout.
- The host packer is reused unchanged; only the kernel-side consumption differs, selected by `TargetProfile` + compute capability.

---

## 7. Compilation & dispatch pipeline

```
Kernel IR
  │
  ▼
emit_llvm_ir()  ─── produces .ll text ─── shared for both NVIDIA + AMD
  │                                         (TargetProfile specializes intrinsics)
  ▼
CodegenBackend::compile()
  │
  ├── NVIDIA: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_XX → .ptx
  │            ptxas → .cubin
  │            cuModuleLoadData / cuLaunchKernel
  │
  └── AMD:    llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfxXXXX → .o
               (loaded via ROCm runtime)
```

- **Runtime compile:** `llc` subprocess (or in-process via `inkwell`/`llvm-sys` in a future optimization). `ptxas` is called for NVIDIA cubin generation.
- **Offline option:** emit `.ll`, cache the compiled binary for AOT use.
- **Correctness harness:** the `#[test_kernel]` CPU-oracle model is backend-agnostic — run the same setups against any `Device` impl and assert the same tolerances.

---

## 8. NVlabs `cuda-oxide`

`cuda-oxide` (https://github.com/NVlabs/cuda-oxide) is a Rust CUDA stack. Its `cuda-core`/`cuda-async` crates provide safe wrappers over the CUDA Driver API that could serve as the `CudaDevice` runtime impl. Its compiler path (Rust → MIR → Pliron IR → LLVM IR → PTX) is an alternative to our LLVM IR text emission, but alpha maturity and heavy toolchain dependencies make it higher-risk for now. **Recommendation:** adopt `cuda-core` for the host runtime if it saves FFI work; keep our own LLVM IR text emission for codegen.

---

## 9. Implementation phases

1. **Seam + smoke kernel.** Define `CodegenBackend` trait; `Nvidia` backend emitting LLVM IR for a trivial elementwise kernel; `llc` + `ptxas` compilation; `CudaDevice` via CUDA Driver API; one `#[test_kernel]` green.
2. **Elementwise + reduction families.** Map `Grid3D` + `Reduction` modes. Brings dequant, qgemv, rms-norm, gather, conv-direct, flash (scalar) online.
3. **MMA path.** `llvm.nvvm.hmma.*` intrinsics — re-tiling required (Metal 8×8 → CUDA 16×16×16). Software-dequant block-scaled.
4. **Blackwell scaled-MMA.** `llvm.nvvm.tcgen05.*` intrinsics for `mx*`/`mxint*`; feature-gated on compute capability ≥ sm_100.
5. **Cooperative reimpl.** Replace `mpp::`/`InlineMsl` kernels with CUTLASS or inline-PTX equivalents.
6. **CLI + CI.** `--target {metal,cuda}` across `build`/`test`/`bench`; Linux+CUDA CI lane; device-spec table.

---

## 10. Risks / open questions

- **Cooperative kernels.** Anything using `Op::InlineMsl` with `mpp::` or `coop_tile_*` is Metal-specific; budget a CUTLASS-based reimplementation.
- **MMA tile-size mismatch.** Metal simdgroup-matrix is 8×8; CUDA `wmma` is 16×16×16. Tiling constants are Metal-tuned and need CUDA-specific retuning.
- **Freeze hazard is Metal-specific.** The bad-geometry hard-freeze is Apple-GPU only; CUDA returns errors.
- **`llc` version availability.** `llc` must be discoverable — bundle with the CUDA Toolkit or via system LLVM. Fall back to error on missing tool.
- **Build ergonomics.** CUDA Toolkit + NVIDIA GPU (or CI runner) required. Metal path stays zero-config on macOS.

---

## 11. Why this is the tractable second backend

The IR + DSL + the entire 30-format quant codec are reused as-is. The LLVM IR emitter is shared with AMD (see [`AMD_BACKEND_SPEC.md`](AMD_BACKEND_SPEC.md)), so the NVIDIA backend is one `CodegenBackend` impl with its own intrinsic set and target triple. Lane widths match (32), format choices align with Blackwell hardware. Contrast with the ANE, which is **not** custom-kernel-programmable.

---

## 12. References

- **`LLVM_IR_UNIFICATION_ANALYSIS.md`** — evidence that NVIDIA and AMD both consume LLVM IR, enabling a shared emitter.
- **`AMD_BACKEND_SPEC.md`** — the AMD `CodegenBackend` impl, sharing the same LLVM IR emitter.
- **`VULKAN_BACKEND_SPEC.md`** — the Vulkan/SPIR-V `CodegenBackend` impl, the third peer backend.
- **`AIR_BACKEND_SPEC.md`** — the Apple AIR `CodegenBackend` impl (experimental, LLVM IR codegen).
- **NVVM IR Specification 12.9** — https://docs.nvidia.com/cuda/archive/12.9.1/nvvm-ir-spec/index.html
- **LLVM NVPTX Backend Usage Guide** — https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html
- **NVlabs `cuda-oxide`** — https://github.com/NVlabs/cuda-oxide — Rust CUDA stack.
- **CUTLASS** — candidate for MMA / cooperative-matmul reimplementation.
- **NVIDIA Blackwell microscaling** — MXFP4/6/8, NVFP4, MXINT8 via `tcgen05` scaled-MMA.

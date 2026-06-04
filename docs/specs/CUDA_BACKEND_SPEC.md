# CUDA / NVIDIA Backend Spec

**Status:** 📋 Proposed (design only; no implementation yet)
**Scope:** Add a second code-generation + runtime backend so MetalTile's existing
`#[kernel]` DSL / IR lowers to **CUDA** (NVIDIA GPUs) in addition to Metal/MSL.
**Out of scope:** model loading, graph execution, tokenization, checkpoint
readers — MetalTile is an optimized-kernel generator, not an inference engine.

---

## 1. Motivation

MetalTile today is a single-target toolchain: the IR in `metaltile-core` lowers
through `metaltile-codegen` to **Metal Shading Language** only (`codegen/lib.rs`:
*"lowers the algorithm IR to Metal Shading Language"*), and `metaltile-runtime`
dispatches exclusively through Metal (`metal_device.rs`). The algorithm IR and
the `#[kernel]` DSL are, by contrast, **backend-neutral** — they describe parallel
compute (program ids, threadgroup memory, simd reductions, MMA tiles, elementwise
math), not Metal specifics.

NVIDIA GPUs are the natural second target because, unlike the ANE (see
`ANE_BACKEND_SPEC.md`), they are **directly programmable with custom kernels**
(CUDA C++ / PTX). The same per-kernel DSL model applies 1:1. And the precision
work in PR #2 lands us in a strong position: the **`mx*` / `mxint*` formats use
E8M0 microscaling with block 32 — exactly what NVIDIA Blackwell's 5th-gen tensor
cores consume in hardware** (`tcgen05` scaled-MMA for MXFP4/6/8, NVFP4, MXINT8).
The quant codec/format layer is pure host Rust and already backend-independent.

**Goal:** one DSL, two backends — author a kernel once, emit correct (and
eventually tuned) code for both Apple GPUs and NVIDIA GPUs.

## 2. Goals / Non-goals

**Goals**
- A `cuda` codegen backend that lowers the existing IR to CUDA C++ (compiled via
  NVRTC at runtime, or offline `nvcc` → PTX/cubin) for every kernel expressible
  in the pure `#[kernel]` DSL.
- A `cuda` runtime backend (device, buffers, dispatch) behind a shared trait, so
  `metaltile-std` kernels and the `tile` CLI (`build`/`test`/`bench`) work against
  either backend.
- Reuse the IR, the `#[kernel]` macro, and the **entire `quant::{codec,format}`
  layer unchanged**.
- Map the block-scaled formats onto Blackwell hardware block-scaling where the
  GPU supports it (scaled tensor-core MMA), with a portable software-decode path
  for pre-Blackwell (Ampere/Hopper/Ada).

**Non-goals**
- Model execution / weight loading (engine concern, separate project).
- 100% kernel parity on day one — the cooperative `mpp::`/`InlineMsl` kernels
  (MMA/MPP/NAX) need per-backend reimplementation (see §6); the pure-DSL kernels
  port through the new emitter.
- ROCm/AMD or SPIR-V (a later backend could reuse the same seam).

## 3. Current state — what already generalizes vs what is Metal-coupled

| Layer | Crate | Backend-neutral? | Notes |
|---|---|---|---|
| Algorithm IR (`Op`, `Kernel`, `DType`, `Shape`, `ConstExpr`) | `metaltile-core` | **Yes** | `op.rs` is abstract math/parallelism; comment already references "backends" (plural). |
| `#[kernel]` DSL macro | `metaltile-macros` | **Yes** | Produces IR, not MSL. |
| Quant codec / format / packer | `metaltile-std::quant` | **Yes** | Pure host Rust; the 30-format matrix is layout + arithmetic, no Metal. |
| Codegen | `metaltile-codegen` | **No** | `emit.rs` + `msl/` emit MSL strings directly; no backend seam yet. |
| Runtime | `metaltile-runtime` | **No** | `metal_device.rs`, Metal dispatch/buffers, `gpu_family.rs`. |
| Cooperative kernels (`Op::InlineMsl` with `mpp::`, `coop_tile_*`) | `metaltile-std` | **Partly** | The raw-MSL escape hatch is Metal-only; needs a CUDA analog. |

So the work is **two new backend modules + one abstraction seam**, not a rewrite.

## 4. Design

### 4.1 The backend seam

Introduce a backend abstraction at two layers:

- **Codegen:** a `CodegenBackend` trait (or an enum dispatch) selecting the
  emitter. Today's `MslGenerator` becomes the `Msl` impl; add a `Cuda` impl. The
  IR → text lowering is parameterized by a small `TargetProfile` (lane width,
  shared-mem syntax, intrinsic names, MMA strategy).
- **Runtime:** a `Device` trait abstracting `compile_kernel`, `alloc`, `upload`,
  `dispatch(grid, block, args)`, `readback`. `MetalDevice` is one impl; add
  `CudaDevice` (CUDA Driver API + NVRTC).

`tile build --target {metal,cuda}` and `tile bench --target cuda` select the
backend; default stays `metal`.

### 4.2 DSL → CUDA op mapping

| DSL / IR construct | Metal (today) | CUDA (proposed) |
|---|---|---|
| `program_id::<0/1/2>()` | threadgroup position in grid | `blockIdx.{x,y,z}` |
| `tid` | thread index in threadgroup | `threadIdx.x` (+ y/z) |
| `lsize` | threads per threadgroup | `blockDim.x` |
| `KernelMode::Grid3D` | grid/tpg dispatch | `<<<grid, block>>>` |
| `KernelMode::Reduction` (TG-per-output) | simdgroup reductions | block reduction (shared mem + warp shuffles) |
| `threadgroup_alloc / _store / _load` | `threadgroup` memory | `__shared__` |
| `simd_sum` / lane ops | 32-lane simdgroup | 32-lane **warp** (`__shfl_down_sync`) — lane widths match |
| `reduce_sum` (across TG) | simd + threadgroup | warp-reduce + shared-mem tree |
| `simdgroup_matmul` (8×8) | `simdgroup_matrix` | `wmma` (16×16×16) / `mma.sync`, or CUTLASS — **re-tiling required** |
| `exp/exp2/rsqrt/log/sqrt` | metal:: | `__expf`/`exp2f`/`rsqrtf`/… |
| `select`, `cast`, bit ops | MSL | C++/CUDA equivalents (identical semantics) |
| decode intrinsics (`e2m1_decode`, `e4m3_decode`, `e5m2_decode`, `int8_decode`, sub-byte bit-stream extract) | MSL preamble helpers (`features.rs`) | CUDA `__device__` preamble helpers — **pure arithmetic, ports directly** |
| `f16` / `bf16` element types | `half` / `bfloat`-via-`half` | `__half` / `__nv_bfloat16` |

The 32-lane simdgroup ≙ 32-lane warp correspondence is the lucky structural match
that makes the reduction and lane-shuffle kernels port cleanly.

### 4.3 The quant formats on NVIDIA — the payoff

- **Software-decode path (all NVIDIA GPUs):** the `quant::codec` decode (E2M1/
  E4M3/E5M2 codebooks, E8M0 `exp2`, FP32/FP16 scales, sub-byte sign-extend) is
  arithmetic and ports to CUDA `__device__` helpers verbatim. Every block-scaled
  kernel works on Ampere/Hopper/Ada via dequant-into-shared + `wmma`/CUTLASS MMA,
  mirroring the Metal reduction/MMA kernels.
- **Hardware block-scaling (Blackwell, sm_100+):** `mxfp4`/`mxfp8`/`nvfp4` and
  `mxint8`/`mxint4` map onto the **scaled tensor-core MMA** (`tcgen05.mma` with
  E8M0/E4M3 scale-factor operands). The E8M0/block-32 layout chosen in PR #2 is
  the native Blackwell microscaling layout — the packed codes + scale buffers can
  feed the hardware path with little/no repacking. This is the single biggest
  reason the precision work transfers.
- The host packer is reused unchanged; only the *kernel-side* consumption differs
  (software dequant vs hardware scaled-MMA), selected by `TargetProfile` + the
  detected compute capability.

### 4.4 Compilation & dispatch

- **Runtime compile:** NVRTC (`nvrtcCompileProgram`) → PTX → `cuModuleLoadData` →
  `cuLaunchKernel`. Mirrors Metal's `newLibraryWithSource` flow, so `tile build`'s
  emit-and-compile loop and the codegen-consistency tests carry over.
- **Offline option:** emit `.cu`, compile with `nvcc` to cubin, for AOT use.
- **Correctness harness:** the `#[test_kernel]` CPU-oracle model is
  backend-agnostic — run the same setups against `CudaDevice` and assert the same
  tolerances. The `quant::format::dequant` oracle is identical.

### 4.5 Tooling option — NVlabs `cuda-oxide`

`cuda-oxide` (https://github.com/NVlabs/cuda-oxide, Apache-2.0 except the
`cuda-bindings` crate which is under the NVIDIA Software License) is a Rust CUDA
stack worth evaluating for two distinct parts of this backend:

- **As the host runtime (low-risk, recommended):** its `cuda-core` / `cuda-async`
  crates already provide safe `CudaContext`, `CudaStream`, `DeviceBuffer<T>` over
  the Driver API, plus raw `cuda-bindings` FFI to `cuda.h`. The `CudaDevice` impl
  (§4.1) could sit on these instead of hand-rolling `cuModule*`/`cuLaunch*` FFI.
- **As an alternative codegen strategy (higher-leverage, higher-risk):**
  `cuda-oxide` is fundamentally a **rustc codegen backend** —
  `Rust → MIR → Pliron IR → LLVM IR → PTX`, single-source via `cargo oxide build`.
  That offers a *second* path to a CUDA backend distinct from §4.2:

  | | **§4.2 path: IR → CUDA C++ → NVRTC → PTX** | **cuda-oxide path: IR → Rust device code → PTX** |
  |---|---|---|
  | Fits MetalTile's model | Yes — same "emit target-language text" shape as the MSL emitter | Partly — emit *Rust* instead of CUDA C++ |
  | Blackwell tensor cores | We must emit PTX / inline-asm or use CUTLASS for `tcgen05`/WGMMA | **Built-in intrinsics** (`tcgen05`, WGMMA, MMA, TMEM, TMA, `cta_group::2`, sm_100a) |
  | Toolchain weight | CUDA Toolkit + NVRTC | CUDA 12.x **+ Clang/libclang + nightly `rust-src`/`rustc-dev`/`llvm-tools` + LLVM 21+** for the advanced intrinsics |
  | Maturity | NVRTC is stable, shipping | **Alpha / experimental, active dev, Linux-only** (Ubuntu 24.04 tested) |
  | License cleanliness | toolkit-only | `cuda-bindings` under NVIDIA Software License (not Apache) — vet before depending |

  **Recommendation:** default to the §4.2 **C++/NVRTC** path (lighter, stable, and
  the closest fit to the existing MSL text-emitter); adopt `cuda-core` for the
  host runtime if it saves FFI work. Keep the `cuda-oxide` *Rust→PTX* path on the
  radar specifically for **Phase 4 (Blackwell scaled-MMA)** — even if we don't
  depend on it, its `tcgen05`/WGMMA intrinsics are a concrete reference for the PTX
  our emitter (or a CUTLASS shim) must produce for the `mx*`/`mxint*` hardware
  block-scaling path. Re-evaluate as a primary dependency once it leaves alpha.

## 5. Implementation phases

1. **Seam + smoke kernel.** `CodegenBackend`/`Device` traits; `Cuda` emitter for a
   trivial elementwise kernel (`copy`, `binary`); NVRTC compile + launch; one
   `#[test_kernel]` green on CUDA. Proves the pipeline end-to-end.
2. **Elementwise + reduction families.** Map `Grid3D` + `Reduction` modes
   (block reductions via warp shuffles). Brings dequant, qgemv, rms-norm,
   gather, conv-direct, flash (scalar) online — the bulk of the pure-DSL kernels.
3. **MMA path.** `wmma`/`mma.sync` (or CUTLASS) re-tiling for the simdgroup-MMA
   kernels (qmm-MMA, patch-embed-MMA, conv-MMA). Software-dequant block-scaled.
4. **Blackwell scaled-MMA.** `tcgen05` scaled tensor-core path for `mx*`/`mxint*`;
   feature-gated on compute capability ≥ sm_100.
5. **Cooperative reimpl.** Replace the `mpp::`/`InlineMsl` MPP/NAX kernels with
   CUTLASS collective-MMA CUDA equivalents (these don't auto-port).
6. **CLI + CI.** `--target cuda` across `build`/`test`/`bench`; a Linux+CUDA CI
   lane; device-spec table (peak BW / TFLOPs / tensor-core TOPS) for the roofline
   columns, mirroring `device_specs.rs`.

Each phase is independently shippable and verified by the existing harness.

## 6. Risks / open questions

- **Cooperative kernels don't port automatically.** Anything using
  `Op::InlineMsl` with `mpp::` or the `coop_tile_*` intrinsics is Metal-specific;
  budget a CUTLASS-based reimplementation for the MMA/MPP/NAX families.
- **MMA tile-size mismatch.** Metal simdgroup-matrix is 8×8; CUDA `wmma` is
  16×16×16. Tiling/skew constants (the `stride=36`, 1152-element staging) are
  Metal-tuned and need CUDA-specific retuning — correctness first, then profile.
- **Freeze hazard is Metal-specific.** The `n_simd==0` / bad-geometry hard-freeze
  is an Apple-GPU failure mode; CUDA rejects illegal launch configs with an error
  instead. The geometry-audit discipline still applies but the failure is safer.
- **NVRTC vs driver-API version skew**, `bf16` availability per arch, and
  `__nv_bfloat16` cooperative-matmul support are arch-dependent — gate by compute
  capability.
- **Build/dev ergonomics:** CUDA toolkit + a NVIDIA GPU (or CI runner) required;
  the Metal path must stay the zero-config default on macOS.
- **Validation:** confirm the Blackwell scaled-MMA path matches the software
  oracle bit-for-bit on a real sm_100 device before claiming the hardware path.

## 7. Why this is the tractable second backend

The IR + DSL + the entire 30-format quant codec are reused as-is; only a codegen
emitter and a runtime device are new, and the lane-width + format choices already
line up with NVIDIA hardware. Contrast with the ANE (`ANE_BACKEND_SPEC.md`), which
is **not** custom-kernel-programmable and forces a graph/compiler model.

## 8. References

- NVlabs **`cuda-oxide`** — https://github.com/NVlabs/cuda-oxide — Rust CUDA
  stack: a rustc `Rust → MIR → Pliron → LLVM → PTX` codegen backend, `cuda-core`/
  `cuda-async` host runtime (`CudaContext`/`CudaStream`/`DeviceBuffer<T>`), and
  Blackwell tensor-core intrinsics (`tcgen05`, WGMMA, MMA, TMEM, sm_100a). See §4.5
  for how it could serve the host runtime and/or the Blackwell path. Alpha,
  Linux-only, heavy toolchain (LLVM 21+); `cuda-bindings` crate under the NVIDIA
  Software License.
- **NVRTC** (runtime CUDA→PTX compilation) + the **CUDA Driver API**
  (`cuModuleLoadData` / `cuLaunchKernel`) — the default §4.2/§4.4 compile+dispatch
  path.
- **CUTLASS** — a candidate for the MMA / cooperative-matmul reimplementation
  (§6) and the Blackwell scaled-MMA path for the `mx*`/`mxint*` formats.
- **NVIDIA Blackwell microscaling** (MXFP4/6/8, NVFP4, MXINT8 via `tcgen05`
  scaled-MMA, E8M0/E4M3 scale operands) — the hardware target the PR-#2 block-scaled
  formats map onto (§4.3).

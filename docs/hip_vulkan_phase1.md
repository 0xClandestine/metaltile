# HIP + Vulkan backends — Phase 1 (RX 9070 XT / gfx1201)

> Companion to `CUDA_BACKEND_SCOPE.md`. Same backend seam, same `#[kernel]`
> macro, same IR — additive Target variants + per-backend runtime.

## Status

| Backend | Codegen | Runtime | Smoke green | Notes |
|---|---|---|---|---|
| HIP / ROCm | `metaltile-codegen::hip::HipGenerator` (CudaGenerator + textual transform) | `metaltile-runtime::HipDevice` (hand-rolled `amdhip64` + `hiprtc` FFI) | **4 / 4** on RX 9070 XT (gfx1201) | wave32 RDNA 4; wave64 CDNA gated by `TargetProfile::hip_wave64`, not yet exercised |
| Vulkan / SPIR-V | `metaltile-codegen::spirv::GlslGenerator` (fresh elementwise walker → GLSL 460 compute) | `metaltile-runtime::VulkanDevice` (hand-rolled `vulkan-1` + `shaderc_combined` FFI) | **4 / 4** on RX 9070 XT | Phase-1 Elementwise only; Reduction / Grid3D / Coop are Phase 2+ |

Build: `cargo test -p metaltile-runtime --features hip,vulkan`.
Bit-accuracy: `vector_add` is **bit-exact** on both backends
(max |Δ| = 0); `scale_add_exp` (exp + constexpr) passes at
**max_rel ≈ 1.2e-7** on both — the same precision band as the CUDA
`--fmad=false` path. The HIP path additionally runs `row_reduce_sum`
green (max_rel ≈ 3.4e-7), exercising the warp-shuffle + shared-memory tree.

## What was added (no Metal / CUDA behavior change)

1. `Target::Hip`, `Target::Spirv` enum variants + matching `TargetProfile`
   constructors (`hip()`, `hip_wave64()`, `vulkan()`).
2. Extended `MmaStrategy` with `AmdWmma`, `AmdMfma`, `Software`,
   `VkCooperativeMatrix` (all data-only — no behavior wired yet).
3. `metaltile-codegen/src/hip/mod.rs` — `HipGenerator` composes
   `CudaGenerator` and post-processes the emitted source with three
   surgical rewrites (header includes, bf16 type name, shuffle-mask width).
4. `metaltile-codegen/src/spirv/mod.rs` — `GlslGenerator` (fresh walker for
   the elementwise op subset) + `GlslBindingPlan` side-table used by the
   runtime to bind storage buffers without reparsing the shader.
5. `metaltile-runtime/src/device/hip/{mod,ffi}.rs` — `HipDevice`.
6. `metaltile-runtime/src/device/vulkan/{mod,ffi}.rs` — `VulkanDevice`.
7. Cargo features `hip` and `vulkan`; `build.rs` extended with
   Windows-aware `HIP_PATH` / `VULKAN_SDK` linker discovery.
8. Smoke tests `tests/hip_smoke.rs` (4) and `tests/vulkan_smoke.rs` (4).

The CUDA emitter, the CUDA runtime, and the Metal path are **unchanged**.

## Gotchas — first-bring-up notes

### HIP

1. **hipRTC does not bundle `hip/hip_fp16.h` or `hip/hip_bf16.h`.**
   The CUDA preamble's `#include <cuda_fp16.h>` becomes
   `#include <hip/hip_fp16.h>` after the text transform, but hipRTC then
   fails to find the header. **Fix:** pass `-I<HIP_PATH>/include` as a
   compile option (`device/hip/mod.rs::compile`). Without this the build
   fails for any kernel, not just bf16/fp16 ones, because both headers are
   always emitted in the preamble.
2. **HIP 7.1 `__shfl_*_sync` enforces a 64-bit mask type.** A
   `static_assert(sizeof(MaskT) == 8)` in the HIP runtime header rejects
   `0xffffffffu` (the CUDA emitter's wave32 mask). The text transform
   rewrites it to `0xffffffffull`. Wave64 (CDNA) needs
   `0xffffffffffffffffull` — a wave64-specific transform pass for Phase 2.
3. **`hipDeviceGetAttribute(HIP_DEVICE_ATTRIBUTE_WARP_SIZE)` returns
   nonsense on ROCm 7.1 / Windows.** The first run returned `65536`. The
   attribute enum index is unstable across ROCm releases. Phase 1 derives
   wave size from the gfx family instead: gfx9* → 64, otherwise 32.
4. **gfx target detection.** Windows `hipDeviceGetName` returns the
   marketing name ("AMD Radeon RX 9070 XT"), not the gfx code. Phase 1
   reads `METALTILE_HIP_GFX` if set; default `gfx1201` matches the user's
   RX 9070 XT. Override for RDNA 3 (`gfx1100`), MI300 (`gfx942`), MI350
   (`gfx950`).
5. **Wave32 + the 32-lane shuffle table = free port.** Beyond the mask
   rewrite, *zero* further HIP-specific work was needed to run
   `row_reduce_sum`. The CUDA emitter's `__shfl_down_sync` + per-warp
   shared-mem tree ported verbatim because RDNA 4 wave32 ≙ NVIDIA warp32.

### Vulkan / SPIR-V

1. **`shaderc_compute_shader` is enum value `2`, not 4.** Mismatched
   `shader_kind` doesn't error — shaderc silently treats the source as a
   different stage, then GLSL fails with `'local_size_x': no such layout
   identifier for this stage` and `gl_GlobalInvocationID undeclared`.
   Always verify the enum mapping against `shaderc/env.h`.
2. **shaderc on Windows: link `shaderc_combined.lib`, not
   `shaderc_shared.lib`.** The combined static lib bundles libshaderc +
   glslang + SPIRV-Tools — one link, no DLL dep chain on `PATH`. Linux
   keeps `shaderc_shared`.
3. **Zero-byte Vulkan allocations are rejected.** `vkCreateBuffer(size=0)`
   returns `VK_ERROR_INVALID_*`. Phase-1 alloc clamps small buffers to 4 B.
4. **GLSL.std.450 is missing `erf` / `erfinv` / `expm1` / `log10`.**
   The preamble synthesises them via `mt_erf`/`mt_erfinv`/`mt_expm1`/
   `mt_log10`. `mt_erfinv` is a coarse Winitzki approximation —
   sufficient for Phase-1 smoke but should be tuned to the CPU oracle's
   tolerance in Phase 2.
5. **SPIR-V is u32 words, but shaderc returns bytes.** Verify
   `spv.len() % 4 == 0` and check the magic number `0x07230203` (LE bytes:
   `03 02 23 07`) before feeding it to `vkCreateShaderModule`.
6. **No `gfx`-equivalent identifier yet.** `VulkanDevice::name()` returns
   the placeholder `"vulkan-device"` — Phase 2 wires
   `vkGetPhysicalDeviceProperties` for the actual device name + the
   subgroup-size-control query (`VK_EXT_subgroup_size_control`) needed for
   the subgroup fast-path reductions per spec §4.1.

## Reproduce

```pwsh
# HIP (RDNA 4 / gfx1201). Add ROCm to PATH so amdhip64_7.dll loads:
$env:Path += ";C:\Program Files\AMD\ROCm\7.1\bin"
cargo test -p metaltile-runtime --features hip --test hip_smoke -- --nocapture

# Vulkan. shaderc_combined is statically linked, so no extra PATH munging:
cargo test -p metaltile-runtime --features vulkan --test vulkan_smoke -- --nocapture
```

Override `gfx` (e.g. for RDNA 3 / MI300): `$env:METALTILE_HIP_GFX = "gfx1100"`.

## What's next (Phase 2 outline)

- **HIP:** the corpus tests that worked for CUDA should largely work for
  HIP since the kernel surface is identical post-transform. Wire the same
  test harness used in `cuda_kernel_corpus` against `HipDevice` to measure
  what fraction passes. Wave64 / CDNA is a separate axis (mask width +
  reductions sized to 64-lane warps).
- **Vulkan:** add Reduction-mode lowering using the portable workgroup-
  shared barrier-tree (subgroup-width agnostic; the §4.1 hazard mitigation).
  This unlocks `row_reduce_sum`, `rms_norm`, `qgemv`. The subgroup-op fast
  path lives behind a feature query.
- **Both:** wire `--target hip` / `--target vulkan` into `tile build` and
  `tile bench`. Add device-spec rows for the roofline view.

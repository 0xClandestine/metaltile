//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Backend abstraction seam (CUDA_BACKEND_SPEC §4.1 / SCOPE Phase 0).
//!
//! Today MetalTile lowers IR → MSL only. This module introduces the
//! *codegen* half of the backend seam so the same IR can target a second
//! backend (CUDA) without forking the IR, the `#[kernel]` macro, or the
//! `quant::{codec,format}` layer.
//!
//! - [`Target`] selects the backend (`tile build --target {metal,cuda}`).
//! - [`CodegenBackend`] is the per-backend "IR → target-language text"
//!   trait. `MslGenerator` is the `Metal` impl (see `msl/mod.rs`); the
//!   CUDA impl lives in `cuda/mod.rs`.
//! - [`TargetProfile`] captures the *leaf* differences between backends —
//!   lane width, shared-memory keyword, math-intrinsic names, MMA strategy
//!   — so the op-walker structure can eventually be shared rather than
//!   duplicated. This is the encoding of spec §4.2's DSL→op mapping table.
//!
//! Phase 0 is intentionally non-breaking: Metal keeps working exactly as
//! before; the CUDA impl is a stub that carries a real [`TargetProfile`]
//! but does not yet emit (that is Phase 1).

use metaltile_core::ir::{Kernel, UnaryOpKind};

use crate::Result;

/// Which GPU backend to lower the IR to. Default stays `Metal` — the
/// zero-config macOS path. `cuda` is opt-in via `--target cuda`.
///
/// **Hip** (AMD ROCm) — see `specs/AMD_BACKEND_SPEC.md`. Reuses the CUDA
/// emitter via a HIP-flavored `TargetProfile`; structurally identical to
/// CUDA at the kernel level (same `__global__` / `__shared__` / shuffle
/// API), with the dialect deltas (headers, bf16 type) handled in `hip/`.
///
/// **Spirv** (Vulkan compute) — see `specs/VULKAN_BACKEND_SPEC.md`. The
/// portable/breadth backend; lowers IR to GLSL compute (shaderc → SPIR-V)
/// or direct SPIR-V. Reaches AMD, NVIDIA, Intel, Adreno, Mali, MoltenVK.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Target {
    #[default]
    Metal,
    Cuda,
    Hip,
    Spirv,
}

impl Target {
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Cuda => "cuda",
            Target::Hip => "hip",
            Target::Spirv => "spirv",
        }
    }

    /// File extension for emitted source (`emit.rs` writes `<name>.<ext>`).
    /// HIP source compiles to AMDGPU code-objects via hipRTC; the `.hip`
    /// suffix is a hipcc convention and is fine for hipRTC too.
    /// SPIR-V uses GLSL compute as the textual surface (`.comp`), compiled
    /// to SPIR-V binary by shaderc at runtime.
    pub fn source_ext(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Cuda => "cu",
            Target::Hip => "hip",
            Target::Spirv => "comp",
        }
    }
}

/// Per-backend "IR → target-language text" generator. The MSL generator
/// (`msl::MslGenerator`) and the CUDA generator (`cuda::CudaGenerator`)
/// both implement this so `emit.rs`, the CLI, and the codegen-consistency
/// tests can be parameterized over the target.
pub trait CodegenBackend {
    /// Which target this backend emits for.
    fn target(&self) -> Target;

    /// The leaf-difference profile (intrinsics, lane width, MMA strategy).
    fn profile(&self) -> &TargetProfile;

    /// Lower `kernel`'s IR to a complete source string for this target.
    fn generate(&self, kernel: &Kernel) -> Result<String>;
}

/// MMA lowering strategy for cooperative matmul ops
/// (`SimdgroupMatMul`, `CoopTile*`). Metal uses fixed 8×8 simdgroup
/// matrices; CUDA has several escalating options (spec §4.2 / §6);
/// AMD has MFMA (CDNA) and WMMA (RDNA3+); Vulkan has the optional
/// `VK_KHR_cooperative_matrix` runtime-queried path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmaStrategy {
    /// Metal `simdgroup_matrix` 8×8 (today).
    Simdgroup8x8,
    /// CUDA `wmma` fragments, 16×16×16 — portable Ampere+. Phase 3.
    Wmma16x16x16,
    /// CUTLASS collective-MMA shim — cooperative/MPP/NAX reimpl. Phase 5.
    Cutlass,
    /// Blackwell `tcgen05` scaled tensor-core MMA, sm_100+. Phase 4.
    Tcgen05,
    /// AMD RDNA3+ WMMA (`__builtin_amdgcn_wmma_*`). RDNA4 adds FP8.
    /// `AMD_BACKEND_SPEC.md §4.2`.
    AmdWmma,
    /// AMD CDNA MFMA (`__builtin_amdgcn_mfma_*`). MI300 FP8, MI350 MXFP.
    AmdMfma,
    /// Software-emulated MMA over shared memory — universal baseline.
    /// Phase 1 for both HIP (wave32 RDNA) and Vulkan; matches the CUDA
    /// software path. Bit-accurate but slow.
    Software,
    /// Software-emulated MMA with the per-warp **C accumulator moved to
    /// lane-local VGPRs** (each lane holds `m*n/32` floats). Saves
    /// `nw * m*n * 4` bytes of shared LDS — the difference that makes
    /// MPP `bm64` kernels fit on RDNA 4 (80KB → 64KB at the cap). A is
    /// still shared per-warp; B is still shared per-warp; only C moves.
    /// Used by HIP wave32 by default; equivalent to `Software` for
    /// kernels without SimdGroup-scope CoopTile (no behavior change).
    SoftwareLocalC,
    /// `VK_KHR_cooperative_matrix` runtime-queried fragment shapes.
    /// `VULKAN_BACKEND_SPEC.md §4.2`. Fragment shape decided at pipeline
    /// creation from `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`.
    VkCooperativeMatrix,
}

/// The leaf textual/structural differences between backends. Everything
/// the op-walker needs to differ on lives here, so the walker itself can
/// stay backend-agnostic. Encodes the spec §4.2 mapping table as data.
#[derive(Debug, Clone)]
pub struct TargetProfile {
    pub target: Target,
    /// SIMD/warp lane width. **32 on Metal simdgroup, CUDA warp, RDNA
    /// wave32**; **64 on CDNA wave64**; **variable on Vulkan subgroup**
    /// (queried at runtime via `VK_EXT_subgroup_size_control`). The
    /// reduction/shuffle lowering reads this; 32 is the lucky structural
    /// match that lets MetalTile's existing kernels port.
    pub lane_width: u32,
    /// Threadgroup/shared-memory address-space keyword.
    /// Metal: `threadgroup`; CUDA/HIP: `__shared__`; GLSL: `shared`.
    pub shared_mem_kw: &'static str,
    /// Barrier intrinsic across the block/threadgroup.
    /// Metal: `threadgroup_barrier(mem_flags::mem_threadgroup)`;
    /// CUDA/HIP: `__syncthreads()`; GLSL: `barrier()`.
    pub block_barrier: &'static str,
    /// Warp/simd barrier. Metal: `simdgroup_barrier(...)`;
    /// CUDA/HIP: `__syncwarp()`; GLSL: `subgroupBarrier()`.
    pub warp_barrier: &'static str,
    pub mma: MmaStrategy,
    /// Lower f32 subgroup-sum (`SimdReduce::Sum`/`Mean`) to a linear-order
    /// `__shfl_sync(_, v, i)` loop over `i = 0..lane_width` so the result
    /// matches a left-to-right CPU `iter().sum()` rounding bit-exactly.
    /// Default `false` (fast butterfly path, ~5 ops on wave32). HIP sets
    /// `true` to match the GPU output to the CPU oracle on recurrence-
    /// sensitive kernels (gated DeltaNet over 4 tokens with state-magnitude
    /// amplification). CUDA stays `false` — its bitwise-identical baseline
    /// is the butterfly. `AMD_BACKEND_SPEC.md §4.3`.
    pub precise_simd_sum: bool,
}

impl TargetProfile {
    pub fn metal() -> Self {
        TargetProfile {
            target: Target::Metal,
            lane_width: 32,
            shared_mem_kw: "threadgroup",
            block_barrier: "threadgroup_barrier(mem_flags::mem_threadgroup)",
            warp_barrier: "simdgroup_barrier(mem_flags::mem_threadgroup)",
            mma: MmaStrategy::Simdgroup8x8,
            precise_simd_sum: false,
        }
    }

    /// Default CUDA profile (software-decode path; portable Ampere+).
    /// Phase 4 swaps `mma` to `Tcgen05` when compute capability ≥ sm_100.
    pub fn cuda() -> Self {
        TargetProfile {
            target: Target::Cuda,
            lane_width: 32,
            shared_mem_kw: "__shared__",
            block_barrier: "__syncthreads()",
            warp_barrier: "__syncwarp()",
            mma: MmaStrategy::Wmma16x16x16,
            precise_simd_sum: false,
        }
    }

    /// Default HIP profile — assumes RDNA wave32 (the most common AMD
    /// consumer / Windows configuration: RX 7000/9000 series). Wave64
    /// (CDNA / Instinct) uses [`TargetProfile::hip_wave64`]; the emitter
    /// reads `lane_width` so reductions/shuffles size correctly.
    ///
    /// The kernel-source dialect is HIP C++ (hipRTC-compatible), which
    /// shares the entire CUDA kernel surface (`__global__`, `__shared__`,
    /// `__shfl_*_sync`, atomics) — only headers + bf16 type name differ.
    /// `AMD_BACKEND_SPEC.md §3`.
    pub fn hip() -> Self {
        TargetProfile {
            target: Target::Hip,
            lane_width: 32,
            shared_mem_kw: "__shared__",
            block_barrier: "__syncthreads()",
            warp_barrier: "__syncwarp()",
            // Phase-2 default: SoftwareLocalC keeps per-warp C in VGPRs
            // so the MPP `bm64` family fits under RDNA 4's 64KB LDS cap.
            // Equivalent to plain `Software` for kernels without
            // SimdGroup-scope CoopTile, so no other kernel changes shape.
            mma: MmaStrategy::SoftwareLocalC,
            // Match the CPU oracle's left-to-right `iter().sum()` rounding
            // on f32 subgroup sums. Closes the gated-DeltaNet KNOWN_HARD
            // (same fix Vulkan uses) at the cost of a 32-step shuffle loop
            // instead of the 5-step butterfly. Pure correctness trade.
            precise_simd_sum: true,
        }
    }

    /// HIP profile for wave64 architectures (CDNA: gfx9xx / MI200 / MI300 /
    /// MI350). Doubles the shuffle/reduction lane mask; the existing emitter's
    /// `0xffffffffu` 32-lane masks must be replaced with `0xffffffffffffffffull`
    /// (this becomes a `Phase 2` hazard per `AMD_BACKEND_SPEC.md §4.1`).
    pub fn hip_wave64() -> Self {
        TargetProfile {
            target: Target::Hip,
            lane_width: 64,
            shared_mem_kw: "__shared__",
            block_barrier: "__syncthreads()",
            warp_barrier: "__syncwarp()",
            // Wave64 (CDNA): same LocalC strategy. The math works out
            // identically — per-warp C tile of `m*n` elements is split
            // across `lane_width` lanes, so each lane holds `m*n/64`
            // floats instead of `m*n/32`.
            mma: MmaStrategy::SoftwareLocalC,
            // Linear-order subgroup sum across 64 lanes — same rationale
            // as wave32 HIP.
            precise_simd_sum: true,
        }
    }

    /// Default Vulkan / GLSL profile. Subgroup size is **runtime-queried**
    /// (`VK_EXT_subgroup_size_control`); 32 is the most common modern desktop
    /// width (NVIDIA, AMD RDNA, Intel Xe-HPG). Reductions lower to the
    /// portable workgroup-shared baseline regardless of subgroup width
    /// (`VULKAN_BACKEND_SPEC.md §4.1`), so the Phase-1 kernel set runs
    /// correctly on any subgroup size. The fast subgroup-op path queries
    /// `VkSubgroupFeatureFlagBits` at pipeline creation.
    pub fn vulkan() -> Self {
        TargetProfile {
            target: Target::Spirv,
            lane_width: 32,
            shared_mem_kw: "shared",
            block_barrier: "barrier()",
            warp_barrier: "subgroupBarrier()",
            // SoftwareLocalC mirrors the HIP path: per-warp CoopTile C
            // accumulator moves to lane-local private arrays so the
            // bm64 / MPP family fits under desktop Vulkan's shared-mem
            // cap (32-48 KB typical, vs 80 KB the per-warp shared layout
            // needs).
            // Gated coopmat path: `MT_VK_COOPMAT=1` flips the MMA
            // strategy to real `VK_KHR_cooperative_matrix` fragment ops
            // (16×16×16 fp16→fp32) for SimdGroup-scope CoopTile kernels
            // on RDNA4. Default (unset) stays on the bit-exact
            // SoftwareLocalC scalar path, so the corpus / Paris e2e are
            // unchanged unless the lever is explicitly pulled. The
            // SPIR-V emitter branches on `mma`; the runtime requests the
            // cooperativeMatrix feature + Vulkan memory model (see
            // `device/vulkan/mod.rs`). subgroupSize stays pinned at 32 —
            // on RDNA4 wave32 the 16×16 coopmat fragment spans 32 lanes,
            // and the CoopTile staging math assumes 32-lane subgroups.
            mma: if std::env::var("MT_VK_COOPMAT").map(|v| v == "1").unwrap_or(false) {
                MmaStrategy::VkCooperativeMatrix
            } else {
                MmaStrategy::SoftwareLocalC
            },
            // The Vulkan path already uses linear-order `mt_subgroup_add`
            // unconditionally for f32 subgroup sums (see `spirv/mod.rs`).
            // This flag is here for completeness; the SPIR-V emitter
            // doesn't currently branch on it.
            precise_simd_sum: true,
        }
    }

    /// `program_id(axis)` source for this target.
    /// Metal: threadgroup-position-in-grid;
    /// CUDA / HIP: `blockIdx.{x,y,z}` (HIP shares the CUDA built-in);
    /// Vulkan/SPIR-V: `gl_WorkGroupID.{x,y,z}` (GLSL compute).
    pub fn block_idx(&self, axis: u32) -> String {
        let comp = ["x", "y", "z"].get(axis as usize).copied().unwrap_or("x");
        match self.target {
            // Metal injects tgid via attributes; the codegen references `tgid_<c>`.
            Target::Metal => format!("tgid_{comp}"),
            // HIP exposes the same `blockIdx`/`threadIdx`/`blockDim` built-ins
            // as CUDA (verbatim from `hip_runtime.h`), so the source share.
            Target::Cuda | Target::Hip => format!("blockIdx.{comp}"),
            Target::Spirv => format!("gl_WorkGroupID.{comp}"),
        }
    }

    /// Math-intrinsic name for a unary op. The decode intrinsics
    /// (`DecodeE2m1`/`E4m3`/`E5m2`/`Int8`) are NOT here — they lower to
    /// preamble `__device__`/inline helpers that port verbatim from
    /// `msl/features.rs` (pure arithmetic; spec §4.2 / §4.3).
    pub fn unary_intrinsic(&self, op: UnaryOpKind) -> &'static str {
        use UnaryOpKind::*;
        match self.target {
            Target::Metal => match op {
                Exp => "exp", Exp2 => "exp2", Expm1 => "expm1",
                Log => "log", Log2 => "log2", Log10 => "log10",
                Sqrt => "sqrt", Rsqrt => "rsqrt", Recip => "1.0/",
                Abs => "abs", Neg => "-", Ceil => "ceil", Floor => "floor",
                Round => "round", Trunc => "trunc", Sign => "sign",
                Sin => "sin", Cos => "cos", Tan => "tan",
                Asin => "asin", Acos => "acos", Atan => "atan",
                Sinh => "sinh", Cosh => "cosh",
                Asinh => "asinh", Acosh => "acosh", Atanh => "atanh",
                Erf => "erf", ErfInv => "erfinv",
            },
            // CUDA *precise* device math (single-precision `f` suffix) — NOT
            // the `__`-prefixed fast-math intrinsics: the CPU oracle is IEEE,
            // and fast-math's multi-ULP error compounds in accumulation-heavy
            // / gain-sensitive kernels (recurrences, softplus). Precision over
            // speed for bit-accuracy (perf retune is a later pass).
            //
            // HIP shares the **identical** single-precision device-math name
            // table — `expf`/`exp2f`/`rsqrtf`/etc. are exposed via HIP's
            // CUDA-compat headers; `__frcp_rn` is provided by hipRTC built-ins.
            // Reusing the CUDA mapping verbatim is the whole point of HIP being
            // a CUDA-portable C++ dialect (`AMD_BACKEND_SPEC.md §3`).
            // `__expf` (CUDA fast exp intrinsic) is ~2x faster than `expf`
            // with ~2 ULP error vs ~1 ULP. Safe for attention softmax.
            // HIP: keep `expf` (hipRTC may not have `__expf`).
            Target::Cuda => match op {
                Exp => "__expf", Exp2 => "exp2f", Expm1 => "expm1f",
                Log => "logf", Log2 => "log2f", Log10 => "log10f",
                Sqrt => "sqrtf", Rsqrt => "rsqrtf", Recip => "__frcp_rn",
                Abs => "fabsf", Neg => "-", Ceil => "ceilf", Floor => "floorf",
                Round => "roundf", Trunc => "truncf", Sign => "copysignf",
                Sin => "sinf", Cos => "cosf", Tan => "tanf",
                Asin => "asinf", Acos => "acosf", Atan => "atanf",
                Sinh => "sinhf", Cosh => "coshf",
                Asinh => "asinhf", Acosh => "acoshf", Atanh => "atanhf",
                Erf => "erff", ErfInv => "erfinvf",
            },
            Target::Hip => match op {
                Exp => "expf", Exp2 => "exp2f", Expm1 => "expm1f",
                Log => "logf", Log2 => "log2f", Log10 => "log10f",
                Sqrt => "sqrtf", Rsqrt => "rsqrtf", Recip => "__frcp_rn",
                Abs => "fabsf", Neg => "-", Ceil => "ceilf", Floor => "floorf",
                Round => "roundf", Trunc => "truncf", Sign => "copysignf",
                Sin => "sinf", Cos => "cosf", Tan => "tanf",
                Asin => "asinf", Acos => "acosf", Atan => "atanf",
                Sinh => "sinhf", Cosh => "coshf",
                Asinh => "asinhf", Acosh => "acoshf", Atanh => "atanhf",
                Erf => "erff", ErfInv => "erfinvf",
            },
            // GLSL.std.450 names (the SPIR-V extended-instruction set used
            // by Vulkan compute). `shaderc` lowers these via the standard
            // GLSL compute shader; `inversesqrt` is the GLSL spelling of
            // rsqrt. `erf`/`erfinv` aren't in GLSL.std.450 — kernels using
            // them fall back to a software approximation in the preamble.
            Target::Spirv => match op {
                Exp => "exp", Exp2 => "exp2", Expm1 => "mt_expm1",
                Log => "log", Log2 => "log2", Log10 => "mt_log10",
                Sqrt => "sqrt", Rsqrt => "inversesqrt", Recip => "1.0/",
                Abs => "abs", Neg => "-", Ceil => "ceil", Floor => "floor",
                Round => "roundEven", Trunc => "trunc", Sign => "sign",
                Sin => "sin", Cos => "cos", Tan => "tan",
                Asin => "asin", Acos => "acos", Atan => "atan",
                Sinh => "sinh", Cosh => "cosh",
                Asinh => "asinh", Acosh => "acosh", Atanh => "atanh",
                // Software approximations live in the GLSL preamble.
                Erf => "mt_erf", ErfInv => "mt_erfinv",
            },
        }
    }
}

// ─── Metal impl ──────────────────────────────────────────────────────
// `MslGenerator` is the `Metal` backend. The profile is a process-wide
// constant, so it lives in a `OnceLock` rather than on the struct (keeps
// `MslGenerator::new(config)` non-breaking).

impl CodegenBackend for crate::msl::MslGenerator {
    fn target(&self) -> Target { Target::Metal }

    fn profile(&self) -> &TargetProfile {
        static METAL: std::sync::OnceLock<TargetProfile> = std::sync::OnceLock::new();
        METAL.get_or_init(TargetProfile::metal)
    }

    fn generate(&self, kernel: &Kernel) -> Result<String> {
        // Delegate to the existing inherent method (unchanged behavior).
        crate::msl::MslGenerator::generate(self, kernel)
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::KernelMode;

    use super::*;

    #[test]
    fn msl_generator_impls_codegen_backend_as_metal() {
        // The Metal generator is the `Metal` CodegenBackend impl.
        let g = crate::generator_for_mode(KernelMode::Elementwise, None);
        let backend: &dyn CodegenBackend = &g;
        assert_eq!(backend.target(), Target::Metal);
        assert_eq!(backend.profile().shared_mem_kw, "threadgroup");
    }

    #[test]
    fn lane_widths_match_the_lucky_32() {
        // The whole reduction/shuffle port hinges on this equality.
        assert_eq!(TargetProfile::metal().lane_width, TargetProfile::cuda().lane_width);
        assert_eq!(TargetProfile::cuda().lane_width, 32);
    }

    #[test]
    fn cuda_profile_uses_cuda_idioms() {
        let p = TargetProfile::cuda();
        assert_eq!(p.shared_mem_kw, "__shared__");
        assert_eq!(p.block_barrier, "__syncthreads()");
        assert_eq!(p.block_idx(0), "blockIdx.x");
        assert_eq!(p.block_idx(2), "blockIdx.z");
        assert_eq!(p.unary_intrinsic(UnaryOpKind::Rsqrt), "rsqrtf");
        assert_eq!(p.unary_intrinsic(UnaryOpKind::Exp2), "exp2f");
        assert_eq!(p.mma, MmaStrategy::Wmma16x16x16);
    }

    #[test]
    fn metal_profile_unchanged() {
        let p = TargetProfile::metal();
        assert_eq!(p.shared_mem_kw, "threadgroup");
        assert_eq!(p.unary_intrinsic(UnaryOpKind::Rsqrt), "rsqrt");
        assert_eq!(p.mma, MmaStrategy::Simdgroup8x8);
    }

    #[test]
    fn source_ext_per_target() {
        assert_eq!(Target::Metal.source_ext(), "metal");
        assert_eq!(Target::Cuda.source_ext(), "cu");
    }
}

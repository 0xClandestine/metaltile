//! Device layer: Metal adapter, CUDA adapter, HIP adapter, Vulkan adapter,
//! buffer pool, GPU family.

#[cfg(target_os = "macos")]
pub(crate) mod buffer_pool;
pub(crate) mod gpu_family;
#[cfg(target_os = "macos")]
pub(crate) mod metal_device;

/// CUDA/NVIDIA backend (NVRTC + Driver API). Opt-in via the `cuda` feature;
/// builds on Linux without the Metal toolchain (CUDA_BACKEND_SPEC §4.1).
#[cfg(feature = "cuda")]
pub mod cuda;

/// AMD ROCm / HIP backend (hipRTC + HIP runtime API). Opt-in via the `hip`
/// feature; builds on Linux and Windows where ROCm is installed
/// (`AMD_BACKEND_SPEC.md §4-§5`).
#[cfg(feature = "hip")]
pub mod hip;

/// Vulkan / SPIR-V compute backend (shaderc + Vulkan compute pipeline).
/// Opt-in via the `vulkan` feature; the portable / reach backend
/// (`VULKAN_BACKEND_SPEC.md`).
#[cfg(feature = "vulkan")]
pub mod vulkan;

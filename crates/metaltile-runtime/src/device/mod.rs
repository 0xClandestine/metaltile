//! Device layer: Metal adapter, CUDA adapter, buffer pool, GPU family.

#[cfg(target_os = "macos")]
pub(crate) mod buffer_pool;
pub(crate) mod gpu_family;
#[cfg(target_os = "macos")]
pub(crate) mod metal_device;

/// CUDA/NVIDIA backend (NVRTC + Driver API). Opt-in via the `cuda` feature;
/// builds on Linux without the Metal toolchain (CUDA_BACKEND_SPEC §4.1).
#[cfg(feature = "cuda")]
pub mod cuda;

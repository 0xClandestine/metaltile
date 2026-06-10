//! Device layer: Metal adapter, buffer pool, GPU family.

#[cfg(target_os = "macos")]
pub(crate) mod buffer_pool;
pub(crate) mod gpu_family;
#[cfg(target_os = "macos")]
pub(crate) mod metal_device;

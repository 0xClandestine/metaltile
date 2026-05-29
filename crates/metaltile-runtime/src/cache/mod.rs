//! Caching layer: PSO compilation and MSL source generation.

#[cfg(target_os = "macos")]
pub(crate) mod msl_cache;
#[cfg(any(target_os = "macos", test))]
pub(crate) mod pso_cache;

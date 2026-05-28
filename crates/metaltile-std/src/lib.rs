pub mod ffai;
pub mod mlx;
pub mod probe;

/// Called by the `__tile_runner` binary to pull this crate into the link,
/// ensuring all `inventory::submit!` kernel registrations are included.
#[doc(hidden)]
pub fn __link_kernels() {}

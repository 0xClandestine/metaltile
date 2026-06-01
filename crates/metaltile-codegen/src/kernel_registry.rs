//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! In-process kernel IR registry consumed by [`KernelInlinePass`].
//!
//! [`KernelEntry`] lives here — alongside the pass that uses it — rather than
//! in `metaltile-core`, because kernel discovery is a runner/codegen concern.
//! The `tile` CLI never calls `all_kernels()` or instantiates `KernelEntry`;
//! those operations only happen inside the `__tile_runner` subprocess.
//!
//! The `metaltile` facade re-exports [`KernelEntry`] and [`all_kernels`] from
//! `metaltile::harness::registry` so user code and the runner module can access
//! them without importing codegen directly.

use metaltile_core::{DType, ir::Kernel};

// ---------------------------------------------------------------------------
// KernelEntry
// ---------------------------------------------------------------------------

/// Registry entry for a MetalTile kernel available for cross-kernel inlining.
///
/// Each `#[kernel]` macro auto-submits one of these via `inventory::submit!`.
/// [`KernelInlinePass`](crate::passes::KernelInlinePass) calls [`all_kernels`]
/// to resolve `Op::KernelCall` nodes during MSL generation.
pub struct KernelEntry {
    name: &'static str,
    builder: fn(&[DType]) -> Kernel,
}

impl KernelEntry {
    /// Create a new registry entry. Called by the `#[kernel]` macro.
    pub const fn new(name: &'static str, builder: fn(&[DType]) -> Kernel) -> Self {
        KernelEntry { name, builder }
    }

    /// The kernel's DSL function name (e.g. `"mt_silu"`, `"mt_rms_norm"`).
    pub fn name(&self) -> &str { self.name }

    /// Build the kernel IR for the given dtype(s).
    pub fn build(&self, dtypes: &[DType]) -> Kernel { (self.builder)(dtypes) }
}

// `collect!` must be in the same crate as the type definition.
inventory::collect!(KernelEntry);

// ---------------------------------------------------------------------------
// Accessor
// ---------------------------------------------------------------------------

/// Iterate all registered kernel IR builders.
///
/// Called by [`KernelInlinePass`](crate::passes::KernelInlinePass) at codegen
/// time inside the runner subprocess. No CLI code should call this.
pub fn all_kernels() -> impl Iterator<Item = &'static KernelEntry> {
    inventory::iter::<KernelEntry>.into_iter()
}

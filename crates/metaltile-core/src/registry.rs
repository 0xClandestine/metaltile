//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! In-process kernel IR registry consumed by [`KernelInlinePass`].
//!
//! [`KernelEntry`] is the only type here — bench and test registries live in
//! `metaltile::harness::registry` to avoid pulling bench infrastructure into
//! the codegen crate.
//!
//! `metaltile-codegen` depends on `metaltile-core` (not the facade), so the
//! kernel entry type and its `all_kernels()` accessor must live here.

use crate::{dsl::dtype::DType, ir::Kernel};

// ---------------------------------------------------------------------------
// KernelEntry
// ---------------------------------------------------------------------------

/// Registry entry for a MetalTile kernel available for cross-kernel inlining.
///
/// Each `#[kernel]` macro auto-submits one of these via `inventory::submit!`.
/// [`KernelInlinePass`] calls [`all_kernels`] to resolve `Op::KernelCall` nodes.
pub struct KernelEntry {
    name:    &'static str,
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
// Accessor — re-exported at the crate root for metaltile-codegen
// ---------------------------------------------------------------------------

/// Iterate all registered kernel IR builders.
///
/// Called by [`KernelInlinePass`] to resolve `Op::KernelCall` nodes at
/// codegen time. This function is the only caller of `inventory::iter` for
/// `KernelEntry` — no other module should call it directly.
pub fn all_kernels() -> impl Iterator<Item = &'static KernelEntry> {
    inventory::iter::<KernelEntry>.into_iter()
}

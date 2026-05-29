//! Inventory registry: the single place in the codebase that calls
//! `inventory::collect!` and `inventory::iter`.
//!
//! All three in-process registries live here:
//! - [`KernelEntry`]      — kernel IR builders, consumed by [`KernelInlinePass`]
//! - [`KernelBenchEntry`] — bench definitions, consumed by the bench runner
//! - [`KernelTestEntry`]  — test definitions, consumed by the test runner
//!
//! Typed accessors (`all_kernels`, `all_benches`, `all_tests`) are the only
//! public API for iterating the registries. No other module calls
//! `inventory::iter` directly.
//!
//! # Subprocess migration note
//!
//! These registries are in-process: `inventory::submit!` is called at link time
//! inside whichever binary has the kernel crates compiled in. In the subprocess
//! model, that binary is the runner spawned by `tile`; the CLI side communicates
//! via the NDJSON protocol in `toolchain/protocol.rs` and never touches these
//! iterators. The registries themselves don't need to change for that migration.

use crate::{
    dsl::dtype::DType,
    ir::Kernel,
    toolchain::{bench::KernelBenchEntry, test::KernelTestEntry},
};

// ---------------------------------------------------------------------------
// KernelEntry
// ---------------------------------------------------------------------------

/// Registry entry for a MetalTile kernel available for cross-kernel inlining.
///
/// Each `#[kernel]` macro auto-submits one of these via `inventory::submit!`.
/// [`KernelInlinePass`] calls [`all_kernels`] to resolve `Op::KernelCall` nodes.
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

// ---------------------------------------------------------------------------
// inventory::collect! — one place, all three types
// ---------------------------------------------------------------------------

inventory::collect!(KernelEntry);
inventory::collect!(KernelBenchEntry);
inventory::collect!(KernelTestEntry);

// ---------------------------------------------------------------------------
// Typed accessors — the only public API for iterating the registries
// ---------------------------------------------------------------------------

/// Iterate all registered kernel IR builders.
pub fn all_kernels() -> impl Iterator<Item = &'static KernelEntry> {
    inventory::iter::<KernelEntry>.into_iter()
}

/// Iterate all registered bench definitions.
pub fn all_benches() -> impl Iterator<Item = &'static KernelBenchEntry> {
    inventory::iter::<KernelBenchEntry>.into_iter()
}

/// Iterate all registered test definitions.
pub fn all_tests() -> impl Iterator<Item = &'static KernelTestEntry> {
    inventory::iter::<KernelTestEntry>.into_iter()
}

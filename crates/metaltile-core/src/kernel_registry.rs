//! Kernel registry for cross-kernel inline calling.
//!
//! Each `#[kernel]` macro auto-submits a [`KernelEntry`] via
//! `inventory::submit!`. The [`KernelInlinePass`] iterates over all
//! registered entries to resolve `Op::KernelCall` nodes.

use crate::{dtype::DType, ir::Kernel};

/// Registry entry for a MetalTile kernel available for cross-kernel calling.
pub struct KernelEntry {
    name: &'static str,
    builder: fn(&[DType]) -> Kernel,
}

impl KernelEntry {
    /// Create a new registry entry. Called by the `#[kernel]` macro.
    pub const fn new(name: &'static str, builder: fn(&[DType]) -> Kernel) -> Self {
        KernelEntry { name, builder }
    }

    /// The kernel's DSL function name (e.g., `"mt_silu"`, `"mt_rms_norm"`).
    pub fn name(&self) -> &str { self.name }

    /// Build the kernel IR for the given dtype(s).
    /// For non-generic kernels the slice is ignored; for generic kernels
    /// `dtypes[0]` is the primary type param.
    pub fn build(&self, dtypes: &[DType]) -> Kernel { (self.builder)(dtypes) }
}

inventory::collect!(KernelEntry);

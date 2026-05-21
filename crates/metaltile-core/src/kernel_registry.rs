//! Kernel registry for cross-kernel inline calling.
//!
//! Each `#[kernel]` macro auto-submits a [`KernelEntry`] via
//! `inventory::submit!`. The [`KernelInlinePass`] iterates over all
//! registered entries to resolve `Op::KernelCall` nodes.

use crate::{dtype::DType, ir::Kernel};

/// Registry entry for a MetalTile kernel available for cross-kernel calling.
pub struct KernelEntry {
    /// The kernel's DSL function name (e.g., `"mt_silu_f32"`, `"mt_rms_norm"`).
    pub name: &'static str,
    /// Build the kernel IR for the given dtype(s).
    /// For non-generic kernels, the slice is ignored.
    /// For kernels with N type params, `dtypes[0..N]` are used.
    pub build: fn(&[DType]) -> Kernel,
}

inventory::collect!(KernelEntry);

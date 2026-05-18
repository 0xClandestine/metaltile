//! Shared kernel utilities used by `build`, `inspect`, `profile`, and `bench` subcommands.

use crate::spec::BenchSpec;

/// The mode to actually use for codegen / display: prefer the spec's
/// explicit `kernel_mode` override, otherwise fall back to
/// [`BenchDispatch::default_mode`].
///
/// Codegen-only kernels (e.g. the FFAI ports in `ffai/`) set
/// `kernel_mode: Some(Reduction|Grid3D)` so the MSL header declares
/// the `tid`/`lsize`/`tgid_*` aliases their bodies depend on even
/// though dispatch is `Generic` with empty `shapes`.
pub fn effective_mode(spec: &BenchSpec) -> metaltile_core::ir::KernelMode {
    spec.kernel_mode.unwrap_or_else(|| spec.dispatch.default_mode(spec.shapes))
}

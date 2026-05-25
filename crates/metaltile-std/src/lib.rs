//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile kernel standard library — kernel definitions only.
//!
//! Contains `mlx/` and `ffai/` kernel submodules. No runner, no stats,
//! no bench types — those live in `metaltile-core`, `metaltile-runtime`,
//! and `metaltile`.

pub mod ffai;
pub mod mlx;

// Re-exports for `#[bench_kernel]` macro-generated code (the macro emits
// `crate::spec::*` and `crate::bench_types::*` paths).
#[doc(hidden)]
pub use metaltile_core::bench::spec;
#[doc(hidden)]
pub use metaltile_core::bench::types as bench_types;

/// Convenience wrapper over `inventory::iter::<BenchSpec>` so internal tests
/// and downstream consumers can discover all registered kernel specs.
pub fn bench_specs() -> Vec<&'static metaltile_core::bench::spec::BenchSpec> {
    let mut v: Vec<&metaltile_core::bench::spec::BenchSpec> =
        inventory::iter::<metaltile_core::bench::spec::BenchSpec>.into_iter().collect();
    v.sort_unstable_by_key(|s| (s.op, s.subop, s.kernel_name));
    v
}

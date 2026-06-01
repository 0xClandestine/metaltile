//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile kernel standard library: benchmark metadata and type definitions.
//!
//! `metaltile-std` provides the kernel definitions (`#[kernel]` / `#[bench]` /
//! `#[test_kernel]`) and the metadata/runner types shared with the CLI.

pub mod bench_types;
pub mod ffai;
pub mod mlx;
pub mod probe;
pub mod quant;

// Re-export the kernel inventories from the harness registry. The `#[kernel]` /
// `#[bench]` / `#[test_kernel]` registrations live in this crate's `ffai` /
// `mlx` modules; importing these accessors via `metaltile_std` (rather than
// `metaltile`) pulls the std rlib into a downstream link, which is what
// retains those inventory statics.
pub use metaltile::harness::registry::{all_benches, all_kernels, all_tests};
pub mod utils;

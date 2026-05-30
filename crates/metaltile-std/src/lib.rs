//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile kernel standard library: benchmark metadata and type definitions.
//!
//! `metaltile-std` provides the data types shared between kernel definitions
//! (`#[kernel(bench(...))]`) and the CLI runner. It contains no GPU runtime code.

pub mod bench_types;
pub mod error;
pub mod ffai;
pub mod mlx;
pub mod probe;

// Re-export the kernel inventories from `metaltile-core`. The `#[kernel]` /
// `#[bench]` / `#[test_kernel]` registrations live in this crate's `ffai` /
// `mlx` modules; importing these accessors via `metaltile_std` (rather than
// `metaltile_core`) pulls the std rlib into a downstream link, which is what
// retains those inventory statics. Integration tests + tools that enumerate
// the registries should import them from here — importing from `metaltile_core`
// directly yields empty registries because nothing force-links the std crate.
pub use metaltile_core::{all_benches, all_kernels, all_tests};
pub mod run_kernel;
pub mod run_spec;
pub mod runner;
pub mod spec;
pub mod stats;
pub mod utils;

//! MetalTile kernel standard library: benchmark metadata and type definitions.
//!
//! `metaltile-std` provides the data types shared between kernel definitions
//! (`#[bench_kernel]`) and the CLI runner. It contains no GPU runtime code.

pub mod bench_types;
pub mod error;
pub mod ffai;
pub mod mlx;
pub mod probe;
pub mod run_spec;
pub mod runner;
pub mod spec;
pub mod stats;

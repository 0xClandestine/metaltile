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
pub mod run_kernel;
pub mod run_spec;
pub mod runner;
pub mod spec;
pub mod stats;
pub mod utils;

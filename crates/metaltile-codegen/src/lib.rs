//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile codegen: lowers the algorithm IR to Metal Shading Language (MSL).
//!
//! This crate performs:
//! - Schedule application (thread-to-tile mapping, vectorization)
//! - MSL text generation
//! - Optimization passes (fusion, working-set analysis, pipelining)
//!
//! The output is a valid MSL source string that can be compiled by the
//! Metal runtime.

pub mod emit;
pub mod error;
pub mod kernel_registry;
pub mod msl;
pub mod passes;

pub use error::{Error, Result};
pub use kernel_registry::{KernelEntry, all_kernels};
pub use msl::{MslGenerator, config::TileSchedule, generator_for_mode, kernel_uses_n_simd};

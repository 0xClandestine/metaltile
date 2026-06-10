//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile codegen: lowers the algorithm IR to GPU kernel source.
//!
//! This crate performs:
//! - Schedule application (thread-to-tile mapping, vectorization)
//! - Target source generation (Metal Shading Language, and — behind the
//!   [`backend`] seam — CUDA C++)
//! - Optimization passes (fusion, working-set analysis, pipelining)
//!
//! The output is a valid source string for the selected [`backend::Target`]
//! that can be compiled by the corresponding runtime.

pub mod backend;
pub mod cuda;
pub mod emit;
pub mod error;
pub mod hip;
pub mod kernel_registry;
pub mod msl;
pub mod passes;
pub mod spirv;

pub use backend::{CodegenBackend, MmaStrategy, Target, TargetProfile};
pub use cuda::CudaGenerator;
pub use error::{Error, Result};
pub use hip::HipGenerator;
pub use kernel_registry::{KernelEntry, all_kernels};
pub use msl::{MslGenerator, config::TileSchedule, generator_for_mode, kernel_uses_n_simd};
pub use spirv::GlslGenerator;

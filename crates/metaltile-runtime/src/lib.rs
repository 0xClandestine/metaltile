//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile runtime: GPU dispatch, buffer management, and autotuning.
//!
//! This crate handles the runtime execution of compiled MetalTile kernels:
//! - Metal device and command queue management
//! - Pipeline state compilation and caching
//! - Buffer allocation and transfer
//! - Autotuner with persistent disk cache

pub mod autotune;
pub mod buffer;
pub mod capture;
pub mod context;
pub mod error;
pub mod gpu_family;

pub use capture::{start_gpu_trace, stop_gpu_trace};
pub use context::{Context, DispatchResult, DispatchSpec, ResidentBuffer};
pub use error::MetalTileError;
pub use gpu_family::GpuFamily;

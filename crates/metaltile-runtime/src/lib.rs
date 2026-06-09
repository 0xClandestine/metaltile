//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! MetalTile runtime: GPU dispatch and buffer management.
//!
//! This crate handles the runtime execution of compiled MetalTile kernels:
//! - Metal device and command queue management
//! - Pipeline state compilation and caching
//! - Buffer allocation and transfer

pub mod buffer;
mod cache;
pub mod context;
mod device;
mod dispatch;
pub mod error;

pub use context::{Context, DispatchResult, DispatchSpec, ResidentBuffer};
pub use device::gpu_family::GpuFamily;
pub use error::MetalTileError;

#[cfg(feature = "cuda")]
pub use device::cuda::{CudaDevice, CudaFunction, CudaModule, DeviceBuffer};

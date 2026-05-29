//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile core: IR types, shape algebra, and DType system.
//!
//! This crate defines the foundational types that all other crates share:
//! - [`DType`]: numeric types (f16, f32, i32, etc.)
//! - [`Shape`]: compile-time dimension tracking via type-level markers
//! - [`ConstExpr`]: constexpr values resolved at kernel compile time
//! - Kernel IR nodes: the SSA-form intermediate representation

pub mod dsl;
pub mod error;
pub mod ir;
pub mod toolchain;

/// Backward-compat re-export: `metaltile_core::bench` still works.
pub mod bench {
    pub use crate::toolchain::{bench::*, test::*};
}

// Backward-compat flat re-exports for the DSL types.
pub use dsl::{ConstExpr, DType, Dim, DimExpr, Shape, constexpr, dtype, shape, tile};
pub use error::{Error, Result};
/// Re-export of `inventory` so generated `inventory::submit!` code in
/// `#[kernel]`-expanded modules can use `metaltile_core::inventory::submit!`.
#[doc(hidden)]
pub use inventory;
pub use ir::{
    ActKind,
    Block,
    BlockId,
    CoopTileAccMode,
    CoopTileScope,
    Kernel,
    KernelCallArg,
    KernelMode,
    Op,
    Param,
    TypedSlot,
    UnaryOpKind,
    ValueId,
    VarId,
};
pub use toolchain::registry::{KernelEntry, all_benches, all_kernels, all_tests};

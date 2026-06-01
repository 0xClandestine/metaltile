//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile core: IR types, shape algebra, and DType system.
//!
//! This crate defines the foundational types shared across all other crates:
//! - [`DType`]: numeric element types (f16, f32, i32, etc.)
//! - [`Shape`]: compile-time dimension tracking via type-level markers
//! - [`ConstExpr`]: constexpr values resolved at kernel compile time
//! - Kernel IR: the SSA-form intermediate representation ([`Op`], [`Block`], [`Kernel`])
//! - [`protocol`]: JSON Lines wire types for the runner ↔ CLI protocol
//! - [`registry`]: [`KernelEntry`] and the `all_kernels()` accessor used by codegen

pub mod dsl;
pub mod error;
pub mod ir;
pub mod protocol;

// Flat re-exports for the DSL types used throughout the codebase.
pub use dsl::{ConstExpr, DType, Dim, DimExpr, Shape, constexpr, dtype, shape, tile};
pub use error::{Error, Result};
/// Re-export of `inventory` so generated `inventory::submit!` code in
/// `#[kernel]`-expanded modules can use `metaltile_core::inventory::submit!`.
///
/// `KernelEntry` and `all_kernels()` live in `metaltile-codegen` (runner
/// concern). This re-export exists solely to provide the submit! path used by
/// the `#[kernel]` macro without forcing user code to depend on codegen.
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

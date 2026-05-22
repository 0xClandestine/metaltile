//! MetalTile core: IR types, shape algebra, and DType system.
//!
//! This crate defines the foundational types that all other crates share:
//! - [`DType`]: numeric types (f16, f32, i32, etc.)
//! - [`Shape`]: compile-time dimension tracking via type-level markers
//! - [`ConstExpr`]: constexpr values resolved at kernel compile time
//! - Kernel IR nodes: the SSA-form intermediate representation

pub mod constexpr;
pub mod dtype;
pub mod error;
pub mod gpu_family;
pub mod ir;
pub mod kernel_registry;
pub mod shape;
pub mod utils;

pub use constexpr::ConstExpr;
pub use dtype::DType;
pub use error::{Error, Result};
pub use gpu_family::GpuFamily;
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
pub use kernel_registry::KernelEntry;
pub use shape::{Dim, DimExpr, Shape, tile};

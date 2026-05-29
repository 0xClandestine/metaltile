//! MetalTile IR: SSA-form intermediate representation for tile-level kernels.
//!
//! The IR is the central data structure of the compiler. It is:
//! - **SSA-form**: every value is produced once, by one operation.
//! - **Explicit**: no implicit broadcasts, no hidden state.
//! - **Small**: designed to be traversed and transformed efficiently.
//!
//! ## Structure
//!
//! A [`Kernel`] contains:
//! - Parameters (tensor inputs/outputs with shapes)
//! - Constexpr declarations
//! - A body [`Block`] with a sequence of [`Op`]s
//!
//! ## Algorithm vs Schedule IR
//!
//! The algorithm IR (defined here) describes *what* to compute.
//! The schedule IR (in `metaltile-codegen`) annotates ops with *how* to compute it:
//! thread mapping, tile sizes, unroll factors, pipelining.

pub mod ids;
pub mod kernel;
pub mod op;
pub mod param;

// Flat re-exports so existing `use metaltile_core::ir::*` paths remain valid.
pub use ids::{BlockId, ValueId, VarId};
pub use kernel::{Block, Kernel, KernelMode};
pub use op::{
    ActKind,
    AtomicKind,
    AtomicScope,
    BinOpKind,
    CoopTileAccMode,
    CoopTileScope,
    IndexExpr,
    KernelCallArg,
    Op,
    ReduceKind,
    UnaryOpKind,
};
pub use param::{ConstExprDecl, Param, ParamKind, TypedSlot};

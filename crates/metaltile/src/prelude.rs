//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Re-exports and placeholder DSL items for `#[kernel]` functions.
//!
//! Import this module with `use metaltile::prelude::*;` in the same Rust module as your kernels.
//! It provides the items you need to **write** and **launch** kernels — types, macros, runtime
//! bindings, and DSL stubs — without the IR/codegen plumbing that is only needed when building
//! compiler passes or inspecting generated code.
//!
//! For raw IR types (`Op`, `Block`, `ValueId`, etc.) use [`metaltile::core::ir`] directly.
//! For codegen types (`MslGenerator`, `TileSchedule`, etc.) use [`metaltile::codegen`] directly.
//!
//! # What's here
//!
//! - **Macros:** [`#[kernel]`], [`#[constexpr]`], [`#[scalar]`],
//!   [`#[strided]`], [`shape!`], [`tile!`]
//! - **IR types (user-facing):** [`ConstExpr`], [`ConstExprValues`], [`DType`], [`Dim`],
//!   [`DimExpr`], [`Shape`], [`Kernel`], [`KernelMode`], [`UnaryOpKind`], [`BinOpKind`],
//!   [`ActKind`], [`ReduceKind`], [`AtomicKind`], [`AtomicScope`], [`CoopTileScope`],
//!   [`CoopTileAccMode`]
//! - **Runtime:** [`Context`], [`DispatchResult`], [`DispatchSpec`], [`ResidentBuffer`],
//!   [`MetalTileError`], [`start_gpu_trace`], [`stop_gpu_trace`]
//! - **Other:** [`GpuFamily`], [`KernelEntry`], [`make_tile`]
//! - **DSL stubs:** [`Tensor`], [`program_id`], [`load`], [`store`], [`dot`],
//!   `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `silu`, `gelu`, `relu`, `tanh`,
//!   `sigmoid`, `sin`, `cos`, `ceil`, `floor`, `recip`
//!
//! The exported functions exist so Rust can parse kernel bodies before the proc macro runs. The
//! `#[kernel]` macro rewrites the function body, so calling these helpers outside a kernel will
//! panic.
//!
//! Output tensors are identified by parameter name today. Use `c`, `out`, or `output` when you
//! want the generated launch path to treat a tensor parameter as writable output.

use std::{marker::PhantomData, ops::Index};

/// Registry entry for a MetalTile kernel available for cross-kernel calling.
///
/// You only need this when registering a kernel for use as an inlined callee via the
/// `inventory::collect!` mechanism. For ordinary `#[kernel]` definitions this is handled
/// automatically by the macro.
pub use metaltile_core::KernelEntry;
/// Compile-time symbolic values used in shape annotations and generated IR.
pub use metaltile_core::constexpr::ConstExpr;
/// A collection of resolved constexpr values for a specific kernel launch.
pub use metaltile_core::constexpr::ConstExprValues;
/// Scalar and tensor element types supported by the IR and MSL codegen.
pub use metaltile_core::dtype::DType;
// IR types — user-facing subset (op-kind enums and kernel-level containers).
// Raw IR plumbing (Op, Block, ValueId, Param, etc.) lives in `metaltile::core::ir`.
/// Neural activation function kind.
pub use metaltile_core::ir::ActKind;
/// Atomic operation kind.
pub use metaltile_core::ir::AtomicKind;
/// Memory scope for an atomic op (device vs threadgroup).
pub use metaltile_core::ir::AtomicScope;
/// Binary operation kind.
pub use metaltile_core::ir::BinOpKind;
/// Accumulation mode for cooperative tile matmul.
pub use metaltile_core::ir::CoopTileAccMode;
/// Execution scope for cooperative tile operations (simdgroup vs threadgroup).
pub use metaltile_core::ir::CoopTileScope;
/// A complete kernel in the IR.
pub use metaltile_core::ir::Kernel;
/// Kernel execution mode metadata for IR/codegen inspection.
pub use metaltile_core::ir::KernelMode;
/// Reduction kind.
pub use metaltile_core::ir::ReduceKind;
/// Unary math operation kind.
pub use metaltile_core::ir::UnaryOpKind;
/// Shape-building helpers.
pub use metaltile_core::shape::Dim;
/// Build a 2D tile shape at runtime: `make_tile(rows, cols) -> Shape`.
///
/// For a compile-time equivalent use the [`tile!`] macro instead.
pub use metaltile_core::shape::tile as make_tile;
/// A single dimension expression used in shape algebra.
// (grouped by rustfmt)
pub use metaltile_core::shape::{DimExpr, Shape};
/// Marks a kernel parameter as a compile-time constant.
pub use metaltile_macros::constexpr;
/// Marks a function as a MetalTile kernel.
pub use metaltile_macros::kernel;
/// Marks a `Tensor` parameter for `constant T&` lowering in MSL.
pub use metaltile_macros::scalar;
/// Constructs a `Shape` from dimension expressions.
pub use metaltile_macros::shape;
/// Marks a `Tensor` parameter for strided lowering (shape + stride arrays emitted).
pub use metaltile_macros::strided;
/// Constructs a 2D tile shape at macro-expansion time.
pub use metaltile_macros::tile;
/// Metal GPU device and command queue context.
pub use metaltile_runtime::Context;
/// Output buffers returned after a kernel dispatch.
pub use metaltile_runtime::DispatchResult;
/// Input buffer spec for the launched dispatch pipeline.
pub use metaltile_runtime::DispatchSpec;
/// Apple GPU family inference from Metal device name strings.
pub use metaltile_runtime::GpuFamily;
/// Top-level runtime error.
pub use metaltile_runtime::MetalTileError;
/// A resident Metal buffer managed by the context.
pub use metaltile_runtime::ResidentBuffer;
/// Start GPU trace capture (Xcode GPU frame capture).
pub use metaltile_runtime::start_gpu_trace;
/// Stop GPU trace capture.
pub use metaltile_runtime::stop_gpu_trace;

/// Placeholder tensor type used in `#[kernel]` signatures.
///
/// `Tensor<T, S>` is a zero-sized marker that carries element type `T` and optional shape metadata
/// `S` for proc-macro parsing. The generated launch surface still binds raw byte buffers by
/// parameter name; this type does not own storage or runtime shape information yet.
pub struct Tensor<T, S = ()> {
    _p: PhantomData<(T, S)>,
}

/// `a[idx]` syntax inside a kernel body.
///
/// The body parser recognizes tensor indexing syntactically and lowers it into IR load/store index
/// expressions. This implementation only exists so the Rust parser accepts the syntax.
impl<T, S> Index<u32> for Tensor<T, S> {
    type Output = u32;
    fn index(&self, _idx: u32) -> &u32 { panic!("Tensor indexing only valid inside #[kernel]") }
}

// ---- DSL function stubs (panic if called outside #[kernel]) ----

/// Return the current program/thread id for the given axis.
pub fn program_id<const AXIS: u32>() -> u32 { panic!("program_id only valid inside #[kernel]") }

/// Load a value from a tensor index expression.
pub fn load<T>(_src: u32) -> T { panic!("load only valid inside #[kernel]") }

/// Store a value into a tensor index expression.
pub fn store<T>(_dst: u32, _value: T) { panic!("store only valid inside #[kernel]") }

/// Dot product placeholder used by tiled kernels.
pub fn dot<T>(_a: T, _b: T) -> T { panic!("dot only valid inside #[kernel]") }

// Elementwise math — the body parser recognizes these by name
macro_rules! unary {
    ($name:ident) => {
        pub fn $name<T>(_x: T) -> T {
            panic!(concat!(stringify!($name), " only valid inside #[kernel]"))
        }
    };
}
unary!(exp);
unary!(log);
unary!(sqrt);
unary!(rsqrt);
unary!(abs);
unary!(silu);
unary!(gelu);
unary!(relu);
unary!(tanh);
unary!(sigmoid);
unary!(sin);
unary!(cos);
unary!(ceil);
unary!(floor);
unary!(recip);

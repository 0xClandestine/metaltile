//! Re-exports and placeholder DSL items for `#[kernel]` functions.
//!
//! Import this module with `use metaltile::prelude::*;` in the same Rust module as your kernels.
//! It provides **everything** from the `metaltile-core`, `metaltile-macros`,
//! `metaltile-runtime`, and `metaltile-codegen` crates — types, macros, runtime
//! bindings, and codegen entry points — plus the placeholder DSL stubs
//! ([`Tensor`], [`program_id`], [`load`], [`store`], [`dot`], and unary math)
//! that the `#[kernel]` proc macro rewrites at compile time.
//!
//! # What's here
//!
//! - **Macros:** [`#[kernel]`], [`#[bench_kernel]`], [`#[constexpr]`], [`#[scalar]`],
//!   [`#[strided]`], [`shape!`], [`tile!`]
//! - **IR types:** [`ConstExpr`], [`ConstExprValues`], [`DType`], [`Dim`], [`DimExpr`],
//!   [`Shape`], [`Kernel`], [`KernelMode`], [`Op`], [`Block`], [`ValueId`], [`BlockId`],
//!   [`VarId`], [`Param`], [`TypedSlot`], [`UnaryOpKind`], [`BinOpKind`], [`ActKind`],
//!   [`ReduceKind`], [`AtomicKind`], [`AtomicScope`], [`CoopTileScope`],
//!   [`CoopTileAccMode`], [`IndexExpr`], [`KernelCallArg`], [`KernelEntry`]
//! - **Runtime:** [`Context`], [`DispatchResult`], [`DispatchSpec`], [`ResidentBuffer`],
//!   [`MetalTileError`], [`start_gpu_trace`], [`stop_gpu_trace`]
//! - **Codegen:** [`MslGenerator`], [`TileSchedule`], [`generator_for_mode`],
//!   [`CodegenError`]
//! - **Other:** [`GpuFamily`], [`IdCounter`]
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

// ═══════════════════════════════════════════════════════════════════════════
// metaltile-core — IR types, shape algebra, DType system
// ═══════════════════════════════════════════════════════════════════════════

/// Compile-time symbolic values used in shape annotations and generated IR.
pub use metaltile_core::constexpr::ConstExpr;
/// A collection of resolved constexpr values for a specific kernel launch.
pub use metaltile_core::constexpr::ConstExprValues;
/// Scalar and tensor element types supported by the IR and MSL codegen.
pub use metaltile_core::dtype::DType;
/// Core error type.
pub use metaltile_core::error::Error;
/// Apple GPU family inference from Metal device name strings.
pub use metaltile_core::gpu_family::GpuFamily;

// IR types
/// Neural activation function kind.
pub use metaltile_core::ir::ActKind;
/// Atomic operation kind.
pub use metaltile_core::ir::AtomicKind;
/// Memory scope for an atomic op (device vs threadgroup).
pub use metaltile_core::ir::AtomicScope;
/// Binary operation kind.
pub use metaltile_core::ir::BinOpKind;
/// A basic block: a sequence of operations.
pub use metaltile_core::ir::Block;
/// Unique identifier for a block.
pub use metaltile_core::ir::BlockId;
/// Accumulation mode for cooperative tile matmul.
pub use metaltile_core::ir::CoopTileAccMode;
/// Execution scope for cooperative tile operations (simdgroup vs threadgroup).
pub use metaltile_core::ir::CoopTileScope;
/// Index expression for loads/stores.
pub use metaltile_core::ir::IndexExpr;
/// A complete kernel in the IR.
pub use metaltile_core::ir::Kernel;
/// An argument to a cross-kernel call.
pub use metaltile_core::ir::KernelCallArg;
/// Kernel execution mode metadata for IR/codegen inspection.
pub use metaltile_core::ir::KernelMode;
/// A single operation in the IR.
pub use metaltile_core::ir::Op;
/// A kernel parameter: a tensor input or output.
pub use metaltile_core::ir::Param;
/// Reduction kind.
pub use metaltile_core::ir::ReduceKind;
/// A typed slot for inline MSL outputs and other typed holes.
pub use metaltile_core::ir::TypedSlot;
/// Unary math operation kind.
pub use metaltile_core::ir::UnaryOpKind;
/// Unique identifier for a value in the IR.
pub use metaltile_core::ir::ValueId;
/// Unique identifier for a loop/block-level variable.
pub use metaltile_core::ir::VarId;

/// Registry entry for a MetalTile kernel available for cross-kernel calling.
pub use metaltile_core::kernel_registry::KernelEntry;

/// Shape-building helpers.
pub use metaltile_core::shape::Dim;
pub use metaltile_core::shape::DimExpr;
pub use metaltile_core::shape::Shape;
pub use metaltile_core::shape::tile;

/// A counter for generating unique IDs.
pub use metaltile_core::utils::IdCounter;

// ═══════════════════════════════════════════════════════════════════════════
// metaltile-macros — proc-macro attributes and function-like macros
// ═══════════════════════════════════════════════════════════════════════════

/// Marks a function as a MetalTile kernel.
pub use metaltile_macros::kernel;
/// Marks a kernel parameter as a compile-time constant.
pub use metaltile_macros::constexpr;
/// Marks a `Tensor` parameter for `constant T&` lowering in MSL.
pub use metaltile_macros::scalar;
/// Marks a `Tensor` parameter for strided lowering (shape + stride arrays emitted).
pub use metaltile_macros::strided;
/// Constructs a `Shape` from dimension expressions.
pub use metaltile_macros::shape;
/// Constructs a 2D tile shape.
pub use metaltile_macros::tile;
/// Registers a kernel for automatic benchmarking (place before `#[kernel]`).
pub use metaltile_macros::bench_kernel;

// ═══════════════════════════════════════════════════════════════════════════
// metaltile-runtime — GPU dispatch, buffering, tracing
// ═══════════════════════════════════════════════════════════════════════════

/// Metal GPU device and command queue context.
pub use metaltile_runtime::Context;
/// Output buffers returned after a kernel dispatch.
pub use metaltile_runtime::DispatchResult;
/// Input buffer spec for the launched dispatch pipeline.
pub use metaltile_runtime::DispatchSpec;
/// A resident Metal buffer managed by the context.
pub use metaltile_runtime::ResidentBuffer;
/// Top-level runtime error.
pub use metaltile_runtime::MetalTileError;
/// Start GPU trace capture (Xcode GPU frame capture).
pub use metaltile_runtime::start_gpu_trace;
/// Stop GPU trace capture.
pub use metaltile_runtime::stop_gpu_trace;

// ═══════════════════════════════════════════════════════════════════════════
// metaltile-codegen — MSL lowering, schedule config
// ═══════════════════════════════════════════════════════════════════════════

/// MSL generator for converting kernel IR to Metal Shading Language.
pub use metaltile_codegen::MslGenerator;
/// Tile schedule configuration for codegen.
pub use metaltile_codegen::TileSchedule;
/// Select the right MSL generator for a given kernel mode.
pub use metaltile_codegen::generator_for_mode;
/// Codegen error (aliased to avoid conflict with core `Error`).
pub use metaltile_codegen::error::Error as CodegenError;

// ═══════════════════════════════════════════════════════════════════════════
// Prelude-local DSL stubs — panic when called outside #[kernel]
// ═══════════════════════════════════════════════════════════════════════════

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
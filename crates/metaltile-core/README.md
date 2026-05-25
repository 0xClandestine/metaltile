# metaltile-core

Core IR types, shape algebra, DType system, and GPU-family detection for the
MetalTile GPU kernel compiler. This is the foundation crate — every other
crate in the workspace depends on it.

Defines the SSA-form intermediate representation that `#[kernel]`
functions parse into, that `metaltile-codegen` optimizes and lowers,
and that `metaltile-runtime` reads for dispatch metadata. Also provides
the `inventory`-based kernel registry and Apple GPU family probes.

## Position in the pipeline

```
metaltile-macros ──► metaltile-core (this crate) ──► metaltile-codegen
  (parses Rust      (IR · DType · Shape ·              (lowers IR → MSL)
   → IR)             KernelRegistry)
                              │
                    metaltile-runtime
                    (reads IR for dispatch metadata)
```

`metaltile-core` defines the shared types that flow through the entire
compiler pipeline. It depends on `metaltile-macros` for the derive macros
(`OpFlags`, `ValueRefs`, `VariantName`) that annotate the IR types, but
has no dependency on the codegen or runtime layers.

## Quick start

Build IR programmatically (no proc-macro needed):

```rust,ignore
use metaltile_core::ir::{Kernel, Block, Op, Param, ParamKind, BinOpKind};
use metaltile_core::dtype::DType;
use metaltile_core::shape::Shape;

let mut kernel = Kernel::new("add_two");
kernel.params.push(Param {
    name: "x".into(),
    dtype: DType::F32,
    shape: Shape::scalar(),
    is_output: false,
    kind: ParamKind::Tensor,
});

let mut block = Block::new(BlockId::new(0));
let a_id = ValueId::new(0);
let two_id = ValueId::new(1);
let sum_id = ValueId::new(2);
block.push_op(Op::Load { src: "x".into(), indices: vec![], mask: None, other: None }, a_id);
block.push_op(Op::Const { value: 2 }, two_id);
block.push_op(Op::BinOp { op: BinOpKind::Add, lhs: a_id, rhs: two_id }, sum_id);
block.push_op(Op::Store { dst: "out".into(), indices: vec![], value: sum_id, mask: None }, ValueId::new(3));
kernel.body = block;
```

> **Note:** `ParamKind` and `BinOpKind` are defined in `src/ir.rs` but not
> re-exported from the crate root. Use the full path
> `metaltile_core::ir::ParamKind` or import from the `ir` module directly.

## Crate contents

| Module | Purpose |
|---|---|
| `ir` | SSA-form kernel IR: `Kernel`, `Block`, `Op`, `ValueId`, `Param` |
| `dtype` | `DType` enum: F32, F16, BF16, I32, U32, I8, U8, I4, U64, I64, Bool |
| `shape` | `Shape`, `Dim`, `DimExpr`, `tile()` constructor |
| `constexpr` | `ConstExpr` — symbolic constants resolved at kernel compile time |
| `error` | `Error` enum and `Result<T>` alias |
| `gpu_family` | `GpuFamily` — Apple GPU generation detection (M1/M2/M3+/M5) |
| `kernel_registry` | `KernelEntry` — registry for kernel discovery via `inventory` |
| `utils` | Internal helpers (bit manipulation, alignment) |

## API reference

### Core types

| Type | Purpose | Defined in | Re-exported |
|---|---|---|---|
| `Kernel` | Top-level IR container: params, constexprs, blocks | `src/ir.rs` | ✅ |
| `Block` | Sequence of `Op`s; owned by a `Kernel` | `src/ir.rs` | ✅ |
| `Op` | A single IR operation (load, store, binary, reduce, loop, etc.) | `src/ir.rs` | ✅ |
| `ValueId` | SSA value handle | `src/ir.rs` | ✅ |
| `BlockId` | Block handle | `src/ir.rs` | ✅ |
| `VarId` | Loop / block-level variable handle | `src/ir.rs` | ✅ |
| `Param` | Kernel tensor/scalar parameter descriptor | `src/ir.rs` | ✅ |
| `ParamKind` | How a param is bound: `Tensor`, `Strided`, or `Scalar` | `src/ir.rs` | ❌ (use `ir::ParamKind`) |
| `KernelMode` | Dispatch shape hint: `Elementwise`, `Reduction`, `Grid3D`, `Tile2D` | `src/ir.rs` | ✅ |
| `KernelCallArg` | Typed argument for inline kernel calls | `src/ir.rs` | ✅ |
| `TypedSlot` | Typed hole for inline MSL outputs | `src/ir.rs` | ✅ |
| `ActKind` | Activation function kind (Silu, Gelu, Relu, Tanh, Sigmoid) | `src/ir.rs` | ✅ |
| `UnaryOpKind` | Unary operation variant enum | `src/ir.rs` | ✅ |
| `CoopTileAccMode` | Cooperative tile accumulation mode | `src/ir.rs` | ✅ |
| `CoopTileScope` | Cooperative tile scope | `src/ir.rs` | ✅ |
| `DType` | Numeric type: F32, F16, BF16, I32, U32, I8, U8, I4, U64, I64, Bool | `src/dtype.rs` | ✅ |
| `Shape` | Compile-time dimension tracking (array of `Dim`) | `src/shape.rs` | ✅ |
| `Dim` | A single dimension: `Known(usize)`, `ConstExpr(name)`, or `Any` | `src/shape.rs` | ✅ |
| `DimExpr` | Symbolic dimension expression (Scale, Const, Var, Add, Range) | `src/shape.rs` | ✅ |
| `ConstExpr` | Named compile-time constant used in shapes and kernel configs | `src/constexpr.rs` | ✅ |
| `GpuFamily` | Apple GPU family level (7=M1, 8=M2, 9=M3/M4, 10=M5) | `src/gpu_family.rs` | ✅ |
| `KernelEntry` | Inventory-registered kernel metadata | `src/kernel_registry.rs` | ✅ |
| `Error` / `Result<T>` | Error enum and result alias | `src/error.rs` | ✅ |

### Op variants

The `Op` enum supports these operation categories:

| Category | Op variants |
|---|---|
| Memory | `Load`, `Store`, `VectorLoad`, `VectorStore` |
| Arithmetic | `BinOp` (Add, Sub, Mul, Div, Max, Min, Pow, And, Or, Xor, CmpLt, CmpGt, CmpLe, CmpGe, CmpEq, CmpNe, Shl, Shr), `UnaryOp` (Neg, Recip, Exp, Log, Sqrt, Rsqrt, Abs, Ceil, Floor, Sin, Cos, Erf, Exp2, Log2, Sign, Round, Trunc) |
| Activations | `Activation` (Silu, Gelu, Relu, Tanh, Sigmoid) — separate from UnaryOp |
| Reductions | `Reduce` (Sum, Max, Min, Mean), `Dot`, `StrideReduce` |
| Control flow | `Loop`, `If` |
| Shape ops | `Transpose`, `ExpandDims`, `Reshape`, `Cat`, `Slice`, `Broadcast` |
| Tile ops | `Zeros`, `Splat`, `Arange`, `Cast`, `Select` |
| High-level ML | `FlashAttention`, `SlidingWindowAttention`, `RmsNorm`, `GatedMlp` |
| Misc | `ProgramId`, `Const`, `FusedElementwise`, `InlineMsl` |

### Error types

`Error` — forwarded from all crates that produce or transform IR.
`Result<T>` — `std::result::Result<T, Error>`.

## Dependencies

### Internal

| Crate | Role |
|---|---|
| `metaltile-macros` | Derive macros (`OpFlags`, `ValueRefs`, `VariantName`) for IR types |
| `inventory` | Re-exported for `#[kernel]`-expanded `inventory::submit!` calls |

### External

| Crate | Role |
|---|---|
| `thiserror` | Derive `Error` |
| `smallvec` | Compact small-vector storage in IR structures |
| `serde` | Serialization for IR dump and manifest |
| `rustc-hash` | `FxHashMap` for IR value tracking |

## MSRV / platform

No platform gating — pure data structures, no GPU calls.
Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New DType variant:** `src/dtype.rs` — add to the `DType` enum. Update
  `size_bytes()`, `msl_name()`, `is_float()`, and `is_int()`. Run workspace
  tests — most passes and the MSL emitter match on `DType`.

- **New IR op variant:** `src/ir.rs` — add to the `Op` enum. Add a
  `Display` arm. Update `metaltile-codegen` passes that exhaustively match
  `Op` (start with `type_check` and `msl::emit_block`).

- **New shape constructors:** `src/shape.rs` — add free functions or methods
  on `Shape`. If you need a macro, add it to `metaltile-macros`.

- **New error variant:** `src/error.rs` — add to `Error` enum.

- **Tests to update:** `src/ir.rs` tests, pass tests in `metaltile-codegen`,
  `metaltile-macros` tests.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-codegen` README](../metaltile-codegen/README.md) — the optimization passes that operate on this IR
- [`metaltile-macros` README](../metaltile-macros/README.md) — how `#[kernel]` produces this IR
- [Crate docs on docs.rs](https://docs.rs/metaltile-core)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
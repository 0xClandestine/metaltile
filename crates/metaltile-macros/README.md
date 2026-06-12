# metaltile-macros

Proc-macro crate providing the `#[kernel]` DSL for MetalTile GPU kernels.
Parses Rust function signatures and bodies at compile time, translates
DSL intrinsics into `metaltile-core` IR, and generates host-side launch
code.

This crate is the front door of the MetalTile compiler: user-written
`#[kernel]` functions enter here, and IR + dispatch surfaces exit. It
also provides `shape!`/`tile!` constructors for shape annotations,
`#[kernel(variants(...))]` for compile-time specialisation, and the `#[bench]` /
`#[test_kernel]` attributes for declarative bench / correctness-test registration.

## Position in the pipeline

```
User Rust code (fn with #[kernel])
            │
      metaltile-macros (this crate)
            │
      metaltile-core IR (Kernel, Block, Op, …)
            │
      metaltile-codegen (opt passes → MSL)
            │
      metaltile-runtime (GPU dispatch)
```

The proc macro runs at the call site's compile time. It consumes the
user's token stream and produces a module containing `kernel_ir()`,
`kernel_ir_for(DType)`, `LaunchBuilder`, and a `launch()` entry point.

## Quick start

Define and expand a kernel:

```rust,ignore
use metaltile_macros::kernel;

#[kernel]
pub fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
    let idx = program_id::<0>();
    store(c[idx], load(a[idx]) + load(b[idx]));
}

// The macro generates:
//   pub mod vector_add {
//       pub fn kernel_ir() -> Kernel { … }
//       pub fn kernel_ir_for(_t: DType) -> Kernel { … }
//       pub struct LaunchBuilder<'a> { … }
//       pub fn launch(ctx: &Context) -> LaunchBuilder<'_> { … }
//   }
```

For generic kernels:

```rust,ignore
#[kernel]
pub fn scale<T>(a: Tensor<T>, factor: f32, out: Tensor<T>) {
    let idx = program_id::<0>();
    store(out[idx], load(a[idx]) * factor);
}
// Now call: scale::kernel_ir_for(DType::F16)
```

## Crate contents

| Module | Purpose |
|---|---|
| `lib.rs` | All proc-macro entry points: `#[kernel]`, `#[constexpr]`, `#[scalar]`, `#[strided]`, `shape!`, `tile!`, `ValueRefs` / `OpFlags` derive macros |
| `kernel/body.rs` | `DslBodyParser` — walks `syn::Expr` trees and translates DSL calls into IR-building token streams |
| `kernel/sig.rs` | Signature parsing: `parse_kernel_params_generic`, `extract_constexprs_typed`, `extract_param_names` |
| `kernel/variants.rs` | `#[kernel(variants(...))]` compile-time specialisation — stamps one kernel per parameter tuple (`VariantsSpec`, `substitute_fn`) |
| `bench.rs` / `test.rs` | The `#[bench]` and `#[test_kernel]` attributes that register a bench/test setup into the `inventory` |
| `derive/mod.rs` | Derive macros: `ValueRefs` (value-id traversal) and `OpFlags` (elementwise/side-effect/etc. predicates) |

## API reference

### Macros

| Macro | Kind | What it does |
|---|---|---|
| `#[kernel]` | attribute | Parses a Rust function into IR + generates a module with `kernel_ir`, `kernel_ir_for`, `LaunchBuilder`, and `launch()` |
| `#[autotune]` | attribute | Placed before `#[kernel]` to enable autotuning: `#[autotune(configs = [...], key = [M, N, K])]`. **Not yet implemented** — `AutotuneArgs` struct exists but parsing is a TODO in `expand_kernel`. |
| `#[kernel(variants(...))]` | attribute | Compile-time specialisation — stamps one kernel per parameter tuple (`variants(BITS = [2,4,8], suffix = "int{BITS}")`), constant-folding the values into the body. See the [Kernel Style Guide](../../docs/STYLE_GUIDE.md). |
| `#[bench]` / `#[test_kernel]` | attribute | Register a kernel's throughput bench / GPU correctness test (and its setup callback) into the `inventory`. `#[bench(dtypes = [...])]`, `#[test_kernel(dtypes = [...], tol = [...])]`; both accept the same `variants(...)` syntax. |
| `#[constexpr]` | attribute | Pass-through: marks a function parameter as a compile-time constant detected by `#[kernel]` |
| `#[scalar]` | attribute | Pass-through: marks a `Tensor` parameter for `constant T&` lowering in MSL |
| `#[strided]` | attribute | Pass-through: marks a `Tensor` parameter for strided lowering (shape + stride arrays emitted) |
| `shape!` | function-like | Constructs a `Shape` from dimension expressions: `shape!(M, K)`, `shape!(32, 64)`, `shape!()` |
| `tile!` | function-like | Constructs a 2D tile shape: `tile!(TILE_M, TILE_N)`, `tile!(32, 64)` |
| `#[derive(ValueRefs)]` | derive | Derives `Op::value_refs()` and `Op::for_each_value_id_mut()` for IR traversal |
| `#[derive(OpFlags)]` | derive | Derives op-flag predicates (`is_elementwise`, `has_side_effects`, etc.) from variant attributes |

### What `#[kernel]` expands to

For a kernel `pub fn my_kernel(a: Tensor<f32>, out: Tensor<f32>) { … }`, the expansion produces:

```
pub mod my_kernel {
    // Build IR for specific DType(s). For non-generic kernels this takes ().
    pub fn kernel_ir_for(_t: DType) -> Kernel { … }

    // Default to f32.
    pub fn kernel_ir() -> Kernel { kernel_ir_for(DType::F32) }

    // Host-side builder.
    pub struct LaunchBuilder<'a> { … }
    impl<'a> LaunchBuilder<'a> {
        pub fn input(self, name: &str, data: Vec<u8>) -> Self { … }
        pub fn dispatch(self) -> Result<DispatchResult, MetalTileError> { … }
    }

    // Entry point.
    pub fn launch(ctx: &Context) -> LaunchBuilder<'_> { … }
}
```

For generic kernels (`fn foo<T>(a: Tensor<T>, …)`), `kernel_ir_for` takes
one `DType` argument per type parameter (`kernel_ir_for(_t: DType)`).
Passing `bench(...)` to `#[kernel]` detects generics and calls `kernel_ir_for`
directly instead of wrapping in a closure.

Output tensors are detected by one of:
- `mut` binding on a `Tensor` parameter (e.g. `mut result: Tensor<f32>`)
- Legacy heuristic: parameter named `out`, `c`, or `output`

### Kernel-level attributes

Attributes placed on the function itself (before or alongside `#[kernel]`):

| Attribute | Effect |
|---|---|
| `#[autotune(configs = [...], key = [M, N, K])]` | **Not yet implemented.** Enables autotuning for this kernel. `configs` is a comma-separated list of config names (defined in the autotuner). `key` lists the shape dimensions used for cache bucketing. |

### Kernel parameter attributes

| Attribute | Effect |
|---|---|
| `#[constexpr]` | Extracts the parameter as a `ConstExprDecl` in the kernel IR. Used for shape dimensions and compile-time constants. Automatically deduplicated — the same name appearing in multiple tensor shapes only generates one constexpr. |
| `#[scalar]` | Emits the parameter as `constant T& name` in MSL rather than `device T*`. Used for scalar values like `eps` or `scale`. |
| `#[strided]` | Emits the parameter as `device T*` plus `constant uint* name_shape` and `constant uint* name_strides` in MSL. Used for non-contiguous tensor views. |

### `#[bench]` / `#[test_kernel]`

Benches and correctness tests are declared **next to the kernel** (in
`kernel_benches` / `kernel_tests` modules) rather than via attribute arguments.
Each attribute wraps a setup function that returns a builder:

```rust,ignore
#[bench(dtypes = [f32, f16, bf16])]
fn bench_mt_scale(dt: DType) -> BenchSetup { /* BenchSetup::new(...).buffer(...).bytes_moved(...) */ }

#[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e-3])]
fn test_mt_scale(dt: DType) -> TestSetup { /* TestSetup::new(...).input(...).expect(...) */ }
```

`dtypes` selects the float types to run; `tol` is the per-dtype absolute
tolerance; both accept the same `variants(...)` syntax as `#[kernel]`. The bench
name is the function name (there is no `name` key). A bench may attach an
optional metal reference via `.with_reference(...)` on the `BenchSetup`. See the
[Kernel Style Guide](../../docs/STYLE_GUIDE.md) for the full builder surface.

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile-core` | Emits IR type constructors (`Kernel`, `Block`, `Op`, `DType`, `Shape`, `ConstExpr`) in the generated token stream |

### External

| Crate | Role |
|---|---|
| `syn` | Parses user-written Rust functions and DSL bodies |
| `quote` | Token-stream construction for generated code |
| `proc-macro2` | Proc-macro token stream API |

## MSRV / platform

No platform gating — pure compile-time code, no GPU calls.
Rust: nightly (workspace-wide, edition 2024).
Requires `[lib] proc-macro = true` in `Cargo.toml`.

## Extending

- **New DSL intrinsic:** `src/kernel/body.rs` — add a recognized function name to the
  expression walker. Update the `Recognized call:` list in the module doc comment.

- **New kernel parameter attribute:** `src/lib.rs` — add a new `#[proc_macro_attribute]`
  pass-through function, update `has_attr` checks, and wire it into
  `parse_kernel_params_generic`.

- **New kernel-level attribute (like `#[autotune]`):** `src/lib.rs` — add the
  `#[proc_macro_attribute]` pass-through, parse its args in `expand_kernel`,
  and emit the corresponding token stream into the generated module.

- **New bench class:** `src/bench.rs` — add variant to `ClassKind` enum,
  add a match arm in `generate_submit` with its `ShapeSpec` and `BenchDispatch` variant.

- **New bench argument:** `src/bench.rs` — add field to `BenchArgs`, add parse
  arm in `BenchArgs::parse()`, consume in `generate_submit`.

- **New shape/tile constructor syntax:** `src/lib.rs` — add a new `#[proc_macro]`
  function following the `shape!` / `tile!` pattern.

- **Tests to update:** Unit tests in `src/lib.rs` (at bottom of file). The tests
  cover param output detection, constexpr deduplication, and legacy output naming.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-core` README](../metaltile-core/README.md) — the IR types emitted by these macros
- [`metaltile-codegen` README](../metaltile-codegen/README.md) — the passes that consume the generated IR
- [`metaltile-std` README](../metaltile-std/README.md) — the `BenchSpec` type that `#[kernel(bench(...))]` submits to
- [Crate docs on docs.rs](https://docs.rs/metaltile-macros)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
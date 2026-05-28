//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile proc macros: `#[kernel]`, `#[bench]`, `#[test_kernel]`, `shape!`, `tile!`.
//!
//! Each macro lives in its own submodule; this file is a thin routing layer.
//! Kernel authors never need to look here — see `docs/TOOLCHAIN_DESIGN.md`.

mod bench;
mod derive;
mod kernel;
mod shape;
mod test;

use proc_macro::TokenStream;
use syn::{ItemFn, parse_macro_input};

// ---------------------------------------------------------------------------
// Derive macros — delegated to `derive/`
// ---------------------------------------------------------------------------

/// Derive `Op::value_refs()` and `Op::for_each_value_id_mut()`.
///
/// Annotate `ValueId` fields with `#[vid]`, `#[vid_opt]`, `#[vid_vec]`,
/// `#[vid_exprs]`, or `#[vid_recursive]`.
#[proc_macro_derive(ValueRefs, attributes(vid, vid_opt, vid_vec, vid_exprs, vid_recursive))]
pub fn derive_value_refs(input: TokenStream) -> TokenStream { derive::value_refs(input) }

/// Derive op-flag predicates (`is_elementwise`, `has_side_effects`, etc.).
///
/// Annotate variants with `#[elementwise]`, `#[side_effect]`, `#[unpredictable]`,
/// `#[cheap_alu]`, `#[op_load]`, etc.
#[proc_macro_derive(
    OpFlags,
    attributes(
        elementwise,
        side_effect,
        unpredictable,
        cheap_alu,
        op_load,
        op_store,
        barrier,
        op_loop,
        op_if,
        op_fused,
        op_const,
        shape_op,
        needs_simd_lane,
        needs_simd_group,
        needs_simdgroup_matrix,
        needs_simd_product,
        no_result,
        result_u32,
        result_i32,
        result_f32_scalar,
        result_f16_scalar,
        result_same_type,
        result_custom
    )
)]
pub fn derive_op_flags(input: TokenStream) -> TokenStream { derive::op_flags(input) }

/// Derive `Op::variant_name()` — returns the variant identifier as a `&'static str`.
///
/// Use `#[variant_name("CustomName")]` on variants that need a display name
/// different from their Rust identifier.
#[proc_macro_derive(VariantName, attributes(variant_name))]
pub fn derive_variant_name(input: TokenStream) -> TokenStream { derive::variant_name(input) }

// ---------------------------------------------------------------------------
// Pass-through attributes — recognised by #[kernel] signature parsing
// ---------------------------------------------------------------------------

/// Marks a parameter as a compile-time constant expression.
///
/// The `#[kernel]` macro detects this attribute during signature parsing and
/// emits the corresponding `ConstExprDecl` in the kernel IR.
#[proc_macro_attribute]
pub fn constexpr(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

/// Marks a `Tensor` parameter as scalar (`constant T&` in MSL).
///
/// The `#[kernel]` macro detects this attribute during signature parsing.
#[proc_macro_attribute]
pub fn scalar(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

/// Marks a `Tensor` parameter as strided (emits shape/strides buffer slots).
///
/// The `#[kernel]` macro detects this attribute during signature parsing.
#[proc_macro_attribute]
pub fn strided(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

// ---------------------------------------------------------------------------
// #[kernel] — kernel IR generation
// ---------------------------------------------------------------------------

/// Marks a function as a MetalTile kernel.
///
/// The function body uses the MetalTile DSL (`load`, `store`, `dot`, etc.) and
/// is compiled into an IR module at macro-expansion time. A host-side `launch`
/// builder and a `KernelEntry` inventory submission are also generated.
///
/// # Example
///
/// ```ignore
/// #[kernel]
/// pub fn vector_add(a: Tensor<f16>, b: Tensor<f16>, mut out: Tensor<f16>) {
///     let idx = program_id::<0>();
///     let x = load(a[idx]);
///     let y = load(b[idx]);
///     store(out[idx], x + y);
/// }
/// ```
///
/// Benchmark and test registrations live in separate `#[bench]` and
/// `#[test_kernel]` attributes — see `docs/TOOLCHAIN_DESIGN.md`.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let _kernel_attr = parse_macro_input!(attr as kernel::KernelAttr);
    let input_fn = parse_macro_input!(item as ItemFn);
    TokenStream::from(kernel::KernelMacroBuilder::new(input_fn).expand())
}

// ---------------------------------------------------------------------------
// #[bench] — benchmark registration
// ---------------------------------------------------------------------------

/// Registers a setup function as a `KernelBench` in the `tile bench` inventory.
///
/// # Required keys
///
/// | Key      | Type                | Description                            |
/// |----------|---------------------|----------------------------------------|
/// | `name`   | `"op/subop"`        | Slash-separated benchmark name         |
/// | `dtypes` | `[f32, f16, bf16]`  | Data types to exercise                 |
///
/// # Optional keys
///
/// | Key     | Type                         | Description                            |
/// |---------|------------------------------|----------------------------------------|
/// | `bytes` | `\|s: &BenchSetup\| -> u64`  | Override bytes-moved calculation       |
/// | `ref`   | `MetalRef { ... }`           | Metal reference kernel for comparison  |
///
/// # Example
///
/// ```ignore
/// #[bench(name = "unary/exp", dtypes = [f32, f16, bf16])]
/// fn exp_bench(dt: DType) -> BenchSetup {
///     let n = 1 << 20;
///     BenchSetup::new(Grid::linear(n))
///         .input("a", BenchBuffer::random(n, dt))
///         .output("out", BenchBuffer::zeros(n, dt))
/// }
/// ```
#[proc_macro_attribute]
pub fn bench(attr: TokenStream, item: TokenStream) -> TokenStream { bench::expand(attr, item) }

// ---------------------------------------------------------------------------
// #[test_kernel] — correctness test registration
// ---------------------------------------------------------------------------

/// Registers a setup function as a `KernelTest` in the `tile test` inventory.
///
/// The test name is taken from the annotated function's name — no `name` key needed.
///
/// # Required keys
///
/// | Key      | Type               | Description        |
/// |----------|--------------------|--------------------|
/// | `dtypes` | `[f32, f16, bf16]` | Data types to test |
///
/// # Optional keys
///
/// | Key   | Type                                          | Description                                     |
/// |-------|-----------------------------------------------|-------------------------------------------------|
/// | `tol` | `f64`, `[f64, ...]`, or `{ dtype: f64, ... }` | Element-wise tolerance override (default: 1e-4) |
///
/// `tol` accepts three forms:
/// - **Scalar** `tol = 1e-4` — same threshold for every dtype.
/// - **Array** `tol = [1e-6, 1e-3, 1e-2]` — one value per dtype, in the same order as `dtypes`.
/// - **Table** `tol = { f32: 1e-6, f16: 1e-3, bf16: 1e-2 }` — keyed by dtype name; every dtype
///   in `dtypes` must appear exactly once.
///
/// # Example
///
/// ```ignore
/// #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e-2])]
/// fn exp_test(dt: DType) -> TestSetup {
///     let n = 256;
///     let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
///     let expected: Vec<f32> = input.iter().map(|x| x.exp()).collect();
///     TestSetup::new(Grid::linear(n))
///         .input("a", TestBuffer::from_f32(&input, dt))
///         .expected("out", TestBuffer::from_f32(&expected, dt))
/// }
/// ```
#[proc_macro_attribute]
pub fn test_kernel(attr: TokenStream, item: TokenStream) -> TokenStream { test::expand(attr, item) }

// ---------------------------------------------------------------------------
// shape! / tile! — shape constructor macros
// ---------------------------------------------------------------------------

/// Construct a [`Shape`] from dimension expressions.
///
/// ```ignore
/// shape!(M, K)    // 2-D constexpr shape
/// shape!(32, 64)  // 2-D known shape
/// shape!()        // scalar
/// ```
#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream { shape::expand_shape(input) }

/// Construct a 2-D tile shape.
///
/// ```ignore
/// tile!(TILE_M, TILE_N)  // constexpr tile
/// tile!(32, 64)           // known tile
/// ```
#[proc_macro]
pub fn tile(input: TokenStream) -> TokenStream { shape::expand_tile(input) }

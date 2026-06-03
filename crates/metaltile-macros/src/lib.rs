//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile proc macros: `#[kernel]`, `#[bench]`, `#[test_kernel]`, `shape!`, `tile!`.
//!
//! Each macro lives in its own submodule; this file is a thin routing layer.
//! Kernel authors never need to look here — see `specs/TOOLCHAIN_DESIGN.md`.

mod bench;
mod derive;
mod kernel;
mod shape;
mod test;

use proc_macro::TokenStream;

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
/// `#[cheap_alu]`, or `#[op_load]`.
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
#[proc_macro_derive(VariantName, attributes(variant_name))]
pub fn derive_variant_name(input: TokenStream) -> TokenStream { derive::variant_name(input) }

// ---------------------------------------------------------------------------
// Pass-through marker attributes
// ---------------------------------------------------------------------------

/// Marks a parameter as a compile-time constant. Detected by `#[kernel]`.
#[proc_macro_attribute]
pub fn constexpr(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

/// Marks a `Tensor` param as scalar (`constant T&` in MSL). Detected by `#[kernel]`.
#[proc_macro_attribute]
pub fn scalar(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

/// Marks a `Tensor` param as strided (emits shape/strides arrays). Detected by `#[kernel]`.
#[proc_macro_attribute]
pub fn strided(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

// ---------------------------------------------------------------------------
// #[kernel]
// ---------------------------------------------------------------------------

/// Marks a function as a MetalTile kernel.
///
/// Use the separate `#[bench]` and `#[test_kernel]` attributes for
/// benchmark and correctness-test registration.
///
/// ```ignore
/// #[kernel]
/// pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) { … }
/// ```
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream { kernel::expand(attr, item) }

// ---------------------------------------------------------------------------
// #[bench] — new OO bench registration
// ---------------------------------------------------------------------------

/// Register a kernel benchmark with the `tile bench` runner.
///
/// ```ignore
/// #[bench(name = "unary/exp", dtypes = [f32, f16, bf16])]
/// fn bench_mt_exp(s: &BenchSetup) -> BenchBuffer { … }
/// ```
#[proc_macro_attribute]
pub fn bench(attr: TokenStream, item: TokenStream) -> TokenStream { bench::expand(attr, item) }

// ---------------------------------------------------------------------------
// #[test_kernel] — new OO test registration
// ---------------------------------------------------------------------------

/// Register a kernel correctness test with the `tile test` runner.
///
/// ```ignore
/// #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-4)]
/// fn test_mt_exp(s: &TestSetup) -> TestBuffer { … }
/// ```
#[proc_macro_attribute]
pub fn test_kernel(attr: TokenStream, item: TokenStream) -> TokenStream { test::expand(attr, item) }

// ---------------------------------------------------------------------------
// shape! / tile! macros
// ---------------------------------------------------------------------------

/// Construct a [`Shape`] from dimension expressions.
///
/// ```ignore
/// shape!(M, K)   // 2D shape with constexpr dims
/// shape!(32, 64) // 2D shape with known dims
/// shape!()       // scalar
/// ```
#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream { shape::expand_shape(input) }

/// Construct a 2D tile shape.
///
/// ```ignore
/// tile!(TILE_M, TILE_N)
/// tile!(32, 64)
/// ```
#[proc_macro]
pub fn tile(input: TokenStream) -> TokenStream { shape::expand_tile(input) }

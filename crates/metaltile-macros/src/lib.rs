//! MetalTile proc macros: `#[kernel]`, `shape!`, `tile!`, `#[autotune]`.
//!
//! These macros parse user-written Rust functions and transform them
//! into MetalTile IR and host-side launch code.

mod bench_impl;
mod body_parser;
mod derive_op;
mod sig_parser;

use body_parser::DslBodyParser;
use proc_macro::TokenStream;
use quote::quote;
use sig_parser::{extract_constexprs_typed, extract_param_names, parse_kernel_params_generic};
use syn::{ItemFn, parse_macro_input};

// ---------------------------------------------------------------------------
// #[kernel] — the main macro
// ---------------------------------------------------------------------------

/// Marks a function as a MetalTile kernel.
///
/// The function body uses the MetalTile DSL (load, store, dot, etc.) and
/// is parsed into IR at compile time. A host-side `launch` method is
/// also generated.
///
/// # Attributes
///
/// - `#[autotune(configs = [...], key = [M, N, K])]` — enable autotuning
///   with the given configs and bucketing keys.
///
/// # Example
///
/// ```ignore
/// #[kernel]
/// pub fn vector_add(
///     a: Tensor<f16>,
///     b: Tensor<f16>,
///     c: Tensor<f16>,
/// ) {
///     let idx = program_id::<0>();
///     let x = load(a[idx]);
///     let y = load(b[idx]);
///     store(c[idx], x + y);
/// }
/// ```
/// Derive `Op::value_refs()` and `Op::for_each_value_id_mut()`.
///
/// Annotate `ValueId` fields with `#[vid]`, `#[vid_opt]`, `#[vid_vec]`,
/// `#[vid_exprs]`, or `#[vid_recursive]`. See `derive_op` module docs for details.
#[proc_macro_derive(ValueRefs, attributes(vid, vid_opt, vid_vec, vid_exprs, vid_recursive))]
pub fn derive_value_refs(input: TokenStream) -> TokenStream { derive_op::derive_value_refs(input) }

/// Derive op-flag predicates (`is_elementwise`, `has_side_effects`, etc.).
///
/// Annotate variants with `#[elementwise]`, `#[side_effect]`, `#[unpredictable]`,
/// `#[cheap_alu]`, or `#[op_load]`. See `derive_op` module docs for details.
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
pub fn derive_op_flags(input: TokenStream) -> TokenStream { derive_op::derive_op_flags(input) }

/// Derive `Op::variant_name()` — returns the variant identifier as a &'static str.
///
/// Supports `#[variant_name("CustomName")]` on variants that need a display name different
/// from their Rust identifier.
#[proc_macro_derive(VariantName, attributes(variant_name))]
pub fn derive_variant_name(input: TokenStream) -> TokenStream {
    derive_op::derive_variant_name(input)
}

#[proc_macro_attribute]
pub fn constexpr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: just marks a parameter as a constexpr for #[kernel] to detect
    item
}

#[proc_macro_attribute]
pub fn scalar(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: marks a Tensor param as a scalar (constant T& in MSL)
    item
}

#[proc_macro_attribute]
pub fn strided(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: marks a Tensor param as strided (emits shape/strides arrays)
    item
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    expand_kernel(input_fn)
}

fn expand_kernel(input_fn: ItemFn) -> TokenStream {
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let vis = &input_fn.vis;

    // Extract type parameters from generics: <T>, <T, U>, etc.
    // Each type param gets a corresponding DType arg variable: T→_t, U→_u, V→_v, W→_w.
    let type_param_names: Vec<String> = input_fn
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(tp) = p { Some(tp.ident.to_string()) } else { None }
        })
        .collect();
    let arg_var_names: Vec<String> = type_param_names
        .iter()
        .enumerate()
        .map(|(i, _)| format!("_{}", ['t', 'u', 'v', 'w'].get(i).copied().unwrap_or('x')))
        .collect();
    // Map from type-param name ("T") to the DType arg ident token (_t).
    let type_var_map: std::collections::HashMap<String, proc_macro2::TokenStream> =
        type_param_names
            .iter()
            .zip(arg_var_names.iter())
            .map(|(tp, av)| {
                let ident = syn::Ident::new(av, proc_macro2::Span::call_site());
                (tp.clone(), quote! { #ident })
            })
            .collect();

    // Parse function signature for tensor parameters and constexprs
    let param_decls = parse_kernel_params_generic(&input_fn.sig, &type_var_map);
    let constexpr_info = extract_constexprs_typed(&input_fn.sig);
    let constexpr_names: Vec<String> = constexpr_info.iter().map(|(n, _)| n.clone()).collect();
    let param_names = extract_param_names(&input_fn.sig);

    // Parse the DSL body into IR-building token stream
    let body_ir = DslBodyParser::parse_with_type_vars(
        &input_fn.block,
        &param_names,
        &constexpr_names,
        &type_var_map,
    );

    let constexpr_idents: Vec<_> = constexpr_names
        .iter()
        .map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()))
        .collect();
    let constexpr_dtypes: Vec<proc_macro2::TokenStream> =
        constexpr_info.iter().map(|(_, d)| d.clone()).collect();

    // Build kernel_ir_for signature and kernel_ir() default call.
    // For non-generic kernels, kernel_ir_for takes no args (same as today).
    let arg_var_idents: Vec<_> =
        arg_var_names.iter().map(|n| syn::Ident::new(n, proc_macro2::Span::call_site())).collect();
    let kernel_ir_for_sig = quote! { pub fn kernel_ir_for(#(#arg_var_idents: DType),*) -> Kernel };
    // kernel_ir() calls kernel_ir_for with F32 defaults for each type param.
    let f32_defaults = arg_var_idents.iter().map(|_| quote! { DType::F32 });

    // Generate the expanded output: both the IR constant and the launch builder.
    let expanded = quote! {
        #vis mod #fn_name {
            use super::*;
            use metaltile_core::ir::{Kernel, Block, Op, ValueId, BlockId, VarId, Param, ParamKind, TypedSlot, ConstExprDecl, BinOpKind, ReduceKind, AtomicKind, AtomicScope, IndexExpr, UnaryOpKind, ActKind, KernelCallArg, CoopTileAccMode, CoopTileScope};
            use metaltile_core::shape::{Shape, Dim};
            use metaltile_core::dtype::DType;
            use metaltile_core::constexpr::ConstExpr;

            /// Build the kernel IR for specific dtype(s).
            /// For non-generic kernels this takes no arguments.
            /// For generic kernels (e.g. `fn foo<T>`) call `kernel_ir_for(DType::F16)`.
            #kernel_ir_for_sig {
                let mut kernel = Kernel::new(#fn_name_str);

                // Constexpr declarations
                #(
                    kernel.constexprs.push(ConstExprDecl {
                        name: ConstExpr::new(stringify!(#constexpr_idents)),
                        dtype: #constexpr_dtypes,
                        value: None,
                    });
                )*

                // Tensor parameters parsed from the signature
                #param_decls

                // DSL body translated to IR ops
                #body_ir

                kernel
            }

            /// The kernel IR, defaulting all type params to f32.
            pub fn kernel_ir() -> Kernel {
                kernel_ir_for(#(#f32_defaults),*)
            }

            /// Host-side launch builder.
            /// Accepts a context and named input buffers.
            pub struct LaunchBuilder<'a> {
                ctx: &'a metaltile_runtime::Context,
                /// Named input buffers.
                buffers: std::collections::BTreeMap<String, Vec<u8>>,
            }

            impl<'a> LaunchBuilder<'a> {
                pub fn new(ctx: &'a metaltile_runtime::Context) -> Self {
                    LaunchBuilder {
                        ctx,
                        buffers: std::collections::BTreeMap::new(),
                    }
                }

                /// Bind a named input buffer.
                pub fn input(mut self, name: &str, data: Vec<u8>) -> Self {
                    self.buffers.insert(name.to_string(), data);
                    self
                }

                /// Dispatch the kernel.
                pub fn dispatch(self) -> std::result::Result<metaltile_runtime::DispatchResult, metaltile_runtime::MetalTileError> {
                    let kernel = kernel_ir();
                    self.ctx.dispatch_with_buffers(&kernel, &self.buffers)
                }
            }

            /// Launch method on the module.
            pub fn launch(ctx: &metaltile_runtime::Context) -> LaunchBuilder<'_> {
                LaunchBuilder::new(ctx)
            }

            // Use `const _: ()` hygiene scope so `__build_for_inline` does not
            // leak into the enclosing module's namespace.
            const _: () = {
                fn __build_for_inline(dtypes: &[metaltile_core::dtype::DType]) -> metaltile_core::ir::Kernel {
                    #[allow(unused_variables)]
                    let _t = dtypes.first().copied().unwrap_or(metaltile_core::dtype::DType::F32);
                    kernel_ir_for(#(#arg_var_idents),*)
                }

                metaltile_core::inventory::submit! {
                    metaltile_core::KernelEntry::new(#fn_name_str, __build_for_inline)
                }
            };
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// shape! macro
// ---------------------------------------------------------------------------

/// Construct a [`Shape`] from dimension expressions.
///
/// ```ignore
/// shape!(M, K)       // 2D shape with constexpr dims M and K
/// shape!(32, 64)     // 2D shape with known dims
/// shape!()           // scalar
/// ```
#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    let dims: Vec<_> = input
        .to_string()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let dim_exprs: Vec<_> = dims
        .iter()
        .map(|d| {
            if let Ok(n) = d.parse::<usize>() {
                quote! { Dim::Known(#n) }
            } else {
                let ident = syn::Ident::new(d, proc_macro2::Span::call_site());
                quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
            }
        })
        .collect();

    let expanded = quote! {
        {
            use metaltile_core::shape::Shape;
            use metaltile_core::shape::Dim;
            use metaltile_core::constexpr::ConstExpr;
            Shape::new([#(#dim_exprs),*])
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// tile! macro
// ---------------------------------------------------------------------------

/// Construct a 2D tile shape.
///
/// ```ignore
/// tile!(TILE_M, TILE_N)  // 2D tile of constexpr dimensions
/// tile!(32, 64)           // 2D tile of known dimensions
/// ```
#[proc_macro]
pub fn tile(input: TokenStream) -> TokenStream {
    let parts: Vec<_> = input.to_string().split(',').map(|s| s.trim().to_string()).collect();

    let rows = parse_dim_expr(&parts[0]);
    let cols = parse_dim_expr(parts.get(1).map_or("1", |s| s.as_str()));

    let expanded = quote! {
        {
            use metaltile_core::shape::tile;
            use metaltile_core::shape::Dim;
            use metaltile_core::constexpr::ConstExpr;
            tile(#rows, #cols)
        }
    };

    TokenStream::from(expanded)
}

fn parse_dim_expr(s: &str) -> proc_macro2::TokenStream {
    if let Ok(n) = s.parse::<usize>() {
        quote! { Dim::Known(#n) }
    } else {
        let ident = syn::Ident::new(s, proc_macro2::Span::call_site());
        quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
    }
}

// ---------------------------------------------------------------------------
// #[bench_kernel] — declarative benchmark registration
// ---------------------------------------------------------------------------

/// Registers a `#[kernel]` function for automatic benchmarking.
///
/// Must be placed **before** `#[kernel]` so it sees the original function
/// signature. Generates an `inventory::submit! { BenchSpec { ... } }` alongside
/// the kernel, which the bench suite collects via `inventory::iter::<BenchSpec>`.
///
/// # Required args
/// - `op    = "group"` — bench table group (e.g. `"unary"`)
/// - `subop = "name"`  — sub-operation label (e.g. `"exp"`)
/// - `class = Unary | Binary | AllReduce | RowReduce`
/// - `cpu   = fn_ptr`  — CPU reference (named fn, not closure)
/// - `tol   = 1e-4`    — maximum absolute correctness error
///
/// # Optional args
/// - `input = Signed|Positive|Half|Unit` (Unary default: `Half`)
/// - `input_a / input_b` (Binary, default: `Half`)
/// - `metal_file = "foo.metal"` — MLX reference source file (loaded via `include_str!` at compile time)
/// - `mlx = "pattern"` — kernel name pattern; `{tn}` → MLX type name
/// - `dtypes = IDENT`  — `&'static [DType]` (default: `FLOAT_DTYPES`)
///
/// # Example
/// ```ignore
/// fn cpu_exp(x: f32) -> f32 { x.exp() }
///
/// #[bench_kernel(op="unary", subop="exp", class=Unary, cpu=cpu_exp,
///                input=Signed, tol=1e-4, metal_file="unary.metal", mlx="v_Exp{tn}{tn}")]
/// #[kernel]
/// pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) { … }
/// ```
#[proc_macro_attribute]
pub fn bench_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    use bench_impl::{BenchArgs, generate_submit};

    let args = match syn::parse::<BenchArgs>(attr) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    let (fn_name, is_generic) = {
        let f = match syn::parse::<syn::ItemFn>(item.clone()) {
            Ok(f) => f,
            Err(e) => return e.to_compile_error().into(),
        };
        let generic = !f.sig.generics.params.is_empty();
        (f.sig.ident.clone(), generic)
    };

    let submit = generate_submit(&fn_name, &args, is_generic);
    let item_ts: proc_macro2::TokenStream = item.into();
    quote! { #item_ts  #submit }.into()
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;
    #[cfg(test)]
    use crate::sig_parser::extract_constexprs;

    #[test]
    fn mutable_tensor_outputs_override_legacy_name_heuristics() {
        let item: ItemFn = parse_quote! {
            fn kernel(a: Tensor<f32>, mut result: Tensor<f32>, c: Tensor<f32>) {}
        };

        let tokens =
            parse_kernel_params_generic(&item.sig, &std::collections::HashMap::new()).to_string();

        assert_param_output(&tokens, "a", false);
        assert_param_output(&tokens, "result", true);
        assert_param_output(&tokens, "c", false);
    }

    #[test]
    fn legacy_output_names_still_work_without_mutable_tensor_params() {
        let item: ItemFn = parse_quote! {
            fn kernel(a: Tensor<f32>, c: Tensor<f32>, output: Tensor<f32>) {}
        };

        let tokens =
            parse_kernel_params_generic(&item.sig, &std::collections::HashMap::new()).to_string();

        assert_param_output(&tokens, "a", false);
        assert_param_output(&tokens, "c", true);
        assert_param_output(&tokens, "output", true);
    }

    #[test]
    fn extract_constexprs_deduplicates_shape_dims() {
        let item: ItemFn = parse_quote! {
            fn kernel(
                a: Tensor<f32, shape!(M, N)>,
                b: Tensor<f32, shape!(M, N)>,
                #[constexpr] K: u32,
                out: Tensor<f32, shape!(K, N)>,
            ) {}
        };

        assert_eq!(extract_constexprs(&item.sig), vec!["M", "N", "K"]);
    }

    fn assert_param_output(tokens: &str, name: &str, expected: bool) {
        let needle = format!(
            "name : \"{name}\" . to_string () , dtype : DType :: F32 , shape : Shape :: scalar () , is_output : {expected}"
        );
        assert!(tokens.contains(&needle), "missing `{needle}` in `{tokens}`");
    }
}

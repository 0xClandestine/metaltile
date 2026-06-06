#![allow(dead_code)]
//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[kernel]` proc-macro implementation.
//!
//! [`KernelAttr`] parses the `#[kernel]` attribute, which accepts an optional
//! `variants(...)` block for compile-time kernel specialisation.
//! [`KernelMacroBuilder`] orchestrates DSL-function → IR-module expansion for
//! a single concrete kernel (i.e. after any variant substitution has occurred).

mod body;
mod sig;
pub(crate) mod variants;

use std::collections::HashMap;

pub(crate) use body::DslBodyParser;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use sig::{extract_constexprs_typed, extract_param_names, parse_kernel_params_generic};
use syn::ItemFn;
use variants::VariantsSpec;

/// Parsed arguments for `#[kernel]`.
///
/// An argument-free `#[kernel]` expands the annotated function as a single
/// kernel, identical to the previous behaviour.  With `#[kernel(variants(...))]`
/// the function body is cloned once per variant, substituting the listed
/// compile-time integer constants and renaming each clone.
pub(crate) struct KernelAttr {
    /// Optional variant-specialisation spec.  `None` for plain `#[kernel]`.
    pub variants: Option<VariantsSpec>,
}

impl syn::parse::Parse for KernelAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(KernelAttr { variants: None });
        }

        // The only supported argument is `variants(...)`.
        let kw: syn::Ident = input.parse()?;
        if kw != "variants" {
            return Err(syn::Error::new(
                kw.span(),
                format!(
                    "expected `variants(...)` or no arguments, found `{kw}`; \
                     use #[bench] and #[test_kernel] for registration"
                ),
            ));
        }

        let inner;
        syn::parenthesized!(inner in input);
        let spec = inner.parse::<VariantsSpec>()?;

        if !input.is_empty() {
            return Err(input.error("unexpected tokens after `variants(...)`"));
        }
        Ok(KernelAttr { variants: Some(spec) })
    }
}

/// Orchestrates the full `#[kernel]` macro expansion.
///
/// Owns the parsed function and generates the kernel IR module, launch builder,
/// and inventory entry.
pub(crate) struct KernelMacroBuilder {
    input_fn: ItemFn,
}

impl KernelMacroBuilder {
    /// Create a new builder from the parsed kernel function.
    pub(crate) fn new(input_fn: ItemFn) -> Self { KernelMacroBuilder { input_fn } }

    /// Run the full expansion pipeline for a single kernel and return its token stream.
    ///
    /// For `#[kernel(variants(...))]` callers, this is invoked once per variant
    /// after [`variants::substitute_fn`] has already rewritten the function.
    /// For plain `#[kernel]`, it is called exactly once with the original function.
    pub(crate) fn expand_one(self) -> TokenStream2 {
        let fn_name = &self.input_fn.sig.ident;
        let fn_name_str = fn_name.to_string();
        let vis = &self.input_fn.vis;

        let type_param_names = self.extract_type_param_names();
        let arg_var_names = self.build_arg_var_names(&type_param_names);
        let type_var_map = self.build_type_var_map(&type_param_names, &arg_var_names);

        let param_decls = parse_kernel_params_generic(&self.input_fn.sig, &type_var_map);
        let constexpr_info = extract_constexprs_typed(&self.input_fn.sig);
        let constexpr_names: Vec<String> = constexpr_info.iter().map(|(n, _)| n.clone()).collect();
        let param_names = extract_param_names(&self.input_fn.sig);

        let body_ir = DslBodyParser::parse_with_type_vars(
            &self.input_fn.block,
            &param_names,
            &constexpr_names,
            &type_var_map,
        );

        let constexpr_idents: Vec<_> = constexpr_names
            .iter()
            .map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()))
            .collect();
        let constexpr_dtypes: Vec<TokenStream2> =
            constexpr_info.iter().map(|(_, d)| d.clone()).collect();

        let arg_var_idents: Vec<_> = arg_var_names
            .iter()
            .map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()))
            .collect();
        let kernel_ir_for_sig = quote! {
            pub fn kernel_ir_for(#(#arg_var_idents: DType),*) -> Kernel
        };
        let f32_defaults: Vec<_> = arg_var_idents.iter().map(|_| quote! { DType::F32 }).collect();

        let kernel_module = self.generate_kernel_module(
            fn_name,
            &fn_name_str,
            vis,
            &arg_var_idents,
            &kernel_ir_for_sig,
            &f32_defaults,
            &constexpr_idents,
            &constexpr_dtypes,
            &param_decls,
            &body_ir,
        );

        quote! { #kernel_module }
    }

    fn extract_type_param_names(&self) -> Vec<String> {
        self.input_fn
            .sig
            .generics
            .params
            .iter()
            .filter_map(|p| {
                if let syn::GenericParam::Type(tp) = p { Some(tp.ident.to_string()) } else { None }
            })
            .collect()
    }

    fn build_arg_var_names(&self, type_param_names: &[String]) -> Vec<String> {
        type_param_names
            .iter()
            .enumerate()
            .map(|(i, _)| format!("_{}", ['t', 'u', 'v', 'w'].get(i).copied().unwrap_or('x')))
            .collect()
    }

    fn build_type_var_map(
        &self,
        type_param_names: &[String],
        arg_var_names: &[String],
    ) -> HashMap<String, TokenStream2> {
        type_param_names
            .iter()
            .zip(arg_var_names.iter())
            .map(|(tp, av)| {
                let ident = syn::Ident::new(av, proc_macro2::Span::call_site());
                (tp.clone(), quote! { #ident })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_kernel_module(
        &self,
        fn_name: &syn::Ident,
        fn_name_str: &str,
        vis: &syn::Visibility,
        arg_var_idents: &[syn::Ident],
        kernel_ir_for_sig: &TokenStream2,
        f32_defaults: &[TokenStream2],
        constexpr_idents: &[syn::Ident],
        constexpr_dtypes: &[TokenStream2],
        param_decls: &TokenStream2,
        body_ir: &TokenStream2,
    ) -> TokenStream2 {
        quote! {
            #vis mod #fn_name {
                use super::*;
                use metaltile::core::ir::{
                    Kernel, Block, Op, ValueId, BlockId, VarId, Param, ParamKind,
                    TypedSlot, ConstExprDecl, BinOpKind, ReduceKind, AtomicKind,
                    AtomicScope, IndexExpr, UnaryOpKind, ActKind, KernelCallArg,
                    CoopTileAccMode, CoopTileScope,
                };
                use metaltile::core::shape::{Shape, Dim};
                use metaltile::core::dtype::DType;
                use metaltile::core::constexpr::ConstExpr;

                /// Build the kernel IR for specific dtype(s).
                ///
                /// For non-generic kernels this takes no arguments.
                /// For generic kernels (e.g. `fn foo<T>`) call `kernel_ir_for(DType::F16)`.
                #kernel_ir_for_sig {
                    let mut kernel = Kernel::new(#fn_name_str);

                    #(
                        kernel.constexprs.push(ConstExprDecl {
                            name: ConstExpr::new(stringify!(#constexpr_idents)),
                            dtype: #constexpr_dtypes,
                            value: None,
                        });
                    )*

                    #param_decls

                    #body_ir

                    kernel
                }

                /// Build the kernel IR with all type parameters defaulting to `f32`.
                pub fn kernel_ir() -> Kernel {
                    kernel_ir_for(#(#f32_defaults),*)
                }

                /// Host-side launch builder.
                pub struct LaunchBuilder<'a> {
                    ctx: &'a metaltile::Context,
                    buffers: std::collections::BTreeMap<String, Vec<u8>>,
                }

                impl<'a> LaunchBuilder<'a> {
                    /// Create a new builder for the given runtime context.
                    pub fn new(ctx: &'a metaltile::Context) -> Self {
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

                    /// Dispatch the kernel and return the result.
                    pub fn dispatch(
                        self,
                    ) -> std::result::Result<
                        metaltile::DispatchResult,
                        metaltile::MetalTileError,
                    > {
                        let kernel = kernel_ir();
                        self.ctx.dispatch_with_buffers(&kernel, &self.buffers)
                    }
                }

                /// Create a launch builder for the given runtime context.
                pub fn launch(ctx: &metaltile::Context) -> LaunchBuilder<'_> {
                    LaunchBuilder::new(ctx)
                }

                const _: () = {
                    fn __build_for_inline(
                        dtypes: &[metaltile::core::dtype::DType],
                    ) -> metaltile::core::ir::Kernel {
                        #[allow(unused_variables)]
                        let _t = dtypes
                            .first()
                            .copied()
                            .unwrap_or(metaltile::core::dtype::DType::F32);
                        kernel_ir_for(#(#arg_var_idents),*)
                    }

                    metaltile::core::inventory::submit! {
                        metaltile::harness::registry::KernelEntry::new(#fn_name_str, __build_for_inline)
                    }
                };
            }
        }
    }
}

/// Expand the `#[kernel]` attribute.
///
/// For a plain `#[kernel]` this calls `KernelMacroBuilder::expand_one` once.
/// For `#[kernel(variants(...))]` this calls `expand_one` once per variant,
/// after substituting the variant's compile-time constants into the function
/// body and renaming it.  The source function itself is **not** emitted; only
/// the N variant modules are registered in the kernel inventory.
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: TokenStream2 = attr.into();
    let item2: TokenStream2 = item.into();

    let attr_parsed = match syn::parse2::<KernelAttr>(attr2) {
        Ok(a) => a,
        Err(e) => return TokenStream::from(e.into_compile_error()),
    };
    let input_fn = match syn::parse2::<ItemFn>(item2) {
        Ok(f) => f,
        Err(e) => return TokenStream::from(e.into_compile_error()),
    };

    match attr_parsed.variants {
        None => KernelMacroBuilder::new(input_fn).expand_one().into(),
        Some(spec) => {
            let base_name = input_fn.sig.ident.to_string();
            let mut out = TokenStream2::new();
            for i in 0..spec.variant_count {
                let params_ordered: Vec<(String, variants::VariantValue)> = spec
                    .params
                    .iter()
                    .map(|(name, vals)| (name.clone(), vals[i].clone()))
                    .collect();
                let variant_fn = match variants::substitute_fn(
                    input_fn.clone(),
                    &params_ordered,
                    &base_name,
                    &spec.suffix,
                ) {
                    Ok(f) => f,
                    Err(e) => return TokenStream::from(e.into_compile_error()),
                };
                out.extend(KernelMacroBuilder::new(variant_fn).expand_one());
            }
            out.into()
        },
    }
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::sig::{extract_constexprs, parse_kernel_params_generic};

    #[test]
    fn mutable_tensor_outputs_override_legacy_name_heuristics() {
        let item: syn::ItemFn = parse_quote! {
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
        let item: syn::ItemFn = parse_quote! {
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
        let item: syn::ItemFn = parse_quote! {
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

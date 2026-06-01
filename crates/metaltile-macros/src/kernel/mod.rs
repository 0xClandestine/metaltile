#![allow(dead_code)]
//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[kernel]` proc-macro implementation.
//!
//! `KernelAttr` parses the (currently argument-free) `#[kernel]` attribute.
//! `KernelMacroBuilder` orchestrates DSL-function → IR-module expansion.

mod body;
mod sig;

use std::collections::HashMap;

pub(crate) use body::DslBodyParser;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use sig::{extract_constexprs_typed, extract_param_names, parse_kernel_params_generic};
use syn::{ItemFn, parse_macro_input};

/// Parsed arguments for `#[kernel]`.
///
/// Currently accepts no arguments — bench and test registrations live in the
/// separate `#[bench]` and `#[test_kernel]` attributes.
pub(crate) struct KernelAttr;

impl syn::parse::Parse for KernelAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if !input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "#[kernel] takes no arguments — use #[bench] and #[test_kernel] for registration",
            ));
        }
        Ok(KernelAttr)
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

    /// Run the full expansion pipeline and return the generated token stream.
    pub(crate) fn expand(self) -> TokenStream2 {
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
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let _attr = parse_macro_input!(attr as KernelAttr);
    let input_fn = parse_macro_input!(item as ItemFn);
    let builder = KernelMacroBuilder::new(input_fn);
    TokenStream::from(builder.expand())
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

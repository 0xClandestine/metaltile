//! Legacy `#[kernel]` macro implementation — preserved for backward compatibility
//! with the `#[kernel(bench(...))]` attribute syntax used in metaltile-std kernels.
//!
//! This module will be removed once all kernel files are migrated to the new
//! `#[bench]` / `#[test_kernel]` attributes (PR 3 / PR 4).

mod bench_impl;
mod body_parser;
mod sig_parser;

use std::collections::HashMap;

use bench_impl::{BenchArgs, generate_submit};
use body_parser::DslBodyParser;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use sig_parser::{extract_constexprs_typed, extract_param_names, parse_kernel_params_generic};
use syn::{ItemFn, Token, parse_macro_input};

// ---------------------------------------------------------------------------
// Attribute parsing — unified #[kernel(bench(...))]
// ---------------------------------------------------------------------------

pub(crate) struct KernelAttr {
    benches: Vec<BenchArgs>,
}

impl syn::parse::Parse for KernelAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(KernelAttr { benches: Vec::new() });
        }

        let mut benches = Vec::new();
        loop {
            let ident: syn::Ident = input.parse()?;
            if ident != "bench" {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected `bench(...)` as #[kernel] argument",
                ));
            }

            let content;
            syn::parenthesized!(content in input);
            benches.push(content.parse::<BenchArgs>()?);

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                if input.is_empty() {
                    break;
                }
            } else {
                break;
            }
        }

        if !input.is_empty() {
            return Err(syn::Error::new(input.span(), "unexpected tokens after bench(...)"));
        }

        Ok(KernelAttr { benches })
    }
}

// ---------------------------------------------------------------------------
// KernelMacroBuilder
// ---------------------------------------------------------------------------

struct KernelMacroBuilder {
    input_fn: ItemFn,
    bench_args: Vec<BenchArgs>,
}

impl KernelMacroBuilder {
    fn new(input_fn: ItemFn, bench_args: Vec<BenchArgs>) -> Self {
        KernelMacroBuilder { input_fn, bench_args }
    }

    fn expand(self) -> TokenStream2 {
        let fn_name = &self.input_fn.sig.ident;
        let fn_name_str = fn_name.to_string();
        let vis = &self.input_fn.vis;
        let is_generic = !self.input_fn.sig.generics.params.is_empty();

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

        let bench_submit: TokenStream2 =
            self.bench_args.iter().map(|args| generate_submit(fn_name, args, is_generic)).collect();

        quote! {
            #kernel_module
            #bench_submit
        }
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
                use metaltile_core::ir::{
                    Kernel, Block, Op, ValueId, BlockId, VarId, Param, ParamKind,
                    TypedSlot, ConstExprDecl, BinOpKind, ReduceKind, AtomicKind,
                    AtomicScope, IndexExpr, UnaryOpKind, ActKind, KernelCallArg,
                    CoopTileAccMode, CoopTileScope,
                };
                use metaltile_core::shape::{Shape, Dim};
                use metaltile_core::dtype::DType;
                use metaltile_core::constexpr::ConstExpr;

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

                pub fn kernel_ir() -> Kernel {
                    kernel_ir_for(#(#f32_defaults),*)
                }

                pub struct LaunchBuilder<'a> {
                    ctx: &'a metaltile_runtime::Context,
                    buffers: std::collections::BTreeMap<String, Vec<u8>>,
                }

                impl<'a> LaunchBuilder<'a> {
                    pub fn new(ctx: &'a metaltile_runtime::Context) -> Self {
                        LaunchBuilder {
                            ctx,
                            buffers: std::collections::BTreeMap::new(),
                        }
                    }

                    pub fn input(mut self, name: &str, data: Vec<u8>) -> Self {
                        self.buffers.insert(name.to_string(), data);
                        self
                    }

                    pub fn dispatch(
                        self,
                    ) -> std::result::Result<
                        metaltile_runtime::DispatchResult,
                        metaltile_runtime::MetalTileError,
                    > {
                        let kernel = kernel_ir();
                        self.ctx.dispatch_with_buffers(&kernel, &self.buffers)
                    }
                }

                pub fn launch(ctx: &metaltile_runtime::Context) -> LaunchBuilder<'_> {
                    LaunchBuilder::new(ctx)
                }

                const _: () = {
                    fn __build_for_inline(
                        dtypes: &[metaltile_core::dtype::DType],
                    ) -> metaltile_core::ir::Kernel {
                        #[allow(unused_variables)]
                        let _t = dtypes
                            .first()
                            .copied()
                            .unwrap_or(metaltile_core::dtype::DType::F32);
                        kernel_ir_for(#(#arg_var_idents),*)
                    }

                    metaltile_core::inventory::submit! {
                        metaltile_core::KernelEntry::new(#fn_name_str, __build_for_inline)
                    }
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point called from lib.rs
// ---------------------------------------------------------------------------

/// Expand the `#[kernel]` attribute, supporting optional `bench(...)` args.
pub(crate) fn expand_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let kernel_attr = parse_macro_input!(attr as KernelAttr);
    let input_fn = parse_macro_input!(item as ItemFn);
    let builder = KernelMacroBuilder::new(input_fn, kernel_attr.benches);
    TokenStream::from(builder.expand())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::{sig_parser::extract_constexprs, *};

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

    #[test]
    fn kernel_attr_parses_empty() {
        let attr: KernelAttr = syn::parse_quote! {};
        assert!(attr.benches.is_empty());
    }

    #[test]
    fn kernel_attr_parses_bench() {
        let attr: KernelAttr = syn::parse_quote! {
            bench(op="unary", subop="exp", class=Unary, tol=1e-4)
        };
        assert_eq!(attr.benches.len(), 1);
        let args = &attr.benches[0];
        assert_eq!(args.op.value(), "unary");
        assert_eq!(args.subop.value(), "exp");
    }

    #[test]
    fn kernel_attr_parses_multiple_bench() {
        let attr: KernelAttr = syn::parse_quote! {
            bench(op="sdpa", subop="prefill", class=SdpaPrefill, h=128, n_heads=32, gqa_factor=4, batch=1, q_len=512, k_len=512, bq=32, bk=16, wm=4, wn=1, tpg=128, tol=2e-2),
            bench(op="sdpa", subop="batched_q8", class=SdpaBatchedDecode, h=128, n_kv=4096, n_heads=32, gqa_factor=4, batch_q=8, variant=PrefillTile, bq=32, bk=16, wm=4, wn=1, tpg=128, tol=2e-2, kernel_mode=SimdGroup2D)
        };
        assert_eq!(attr.benches.len(), 2);
        assert_eq!(attr.benches[0].subop.value(), "prefill");
        assert_eq!(attr.benches[1].subop.value(), "batched_q8");
    }

    fn assert_param_output(tokens: &str, name: &str, expected: bool) {
        let needle = format!(
            "name : \"{name}\" . to_string () , dtype : DType :: F32 , shape : Shape :: scalar () , is_output : {expected}"
        );
        assert!(tokens.contains(&needle), "missing `{needle}` in `{tokens}`");
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[bench]` proc-macro attribute implementation.
//!
//! Generates a `KernelBench` impl and inventory submission from a plain
//! setup function annotated with `#[bench(name = "...", dtypes = [...])]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, Ident, ItemFn, LitStr, Token, parse::ParseStream};

/// Parsed arguments for the `#[bench]` attribute.
struct BenchAttr {
    /// Benchmark name, e.g. `"unary/exp"`.
    name: LitStr,
    /// Data types to exercise, e.g. `[f32, f16, bf16]`.
    dtypes: Vec<Ident>,
    /// Optional `|s: &BenchSetup| -> u64` closure overriding bytes-moved.
    bytes: Option<Expr>,
    /// Optional `|s: &BenchSetup| -> u64` closure overriding the FLOP count.
    flops: Option<Expr>,
    /// Optional `MetalRef { ... }` expression for reference comparison.
    metal_ref: Option<Expr>,
}

impl syn::parse::Parse for BenchAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut dtypes = None;
        let mut bytes = None;
        let mut flops = None;
        let mut metal_ref = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            if key == "name" {
                name = Some(input.parse::<LitStr>()?);
            } else if key == "dtypes" {
                let content;
                syn::bracketed!(content in input);
                let list = content.parse_terminated(Ident::parse, Token![,])?;
                dtypes = Some(list.into_iter().collect::<Vec<_>>());
            } else if key == "bytes" {
                bytes = Some(input.parse::<Expr>()?);
            } else if key == "flops" {
                flops = Some(input.parse::<Expr>()?);
            } else if key == "ref" {
                metal_ref = Some(input.parse::<Expr>()?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    format!(
                        "unknown #[bench] key `{key}` — valid keys: name, dtypes, bytes, flops, ref"
                    ),
                ));
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(BenchAttr {
            name: name.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[bench] requires `name = \"...\"`",
                )
            })?,
            dtypes: dtypes.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[bench] requires `dtypes = [f32, ...]`",
                )
            })?,
            bytes,
            flops,
            metal_ref,
        })
    }
}

/// Map a dtype identifier string to its `DType::*` token stream.
pub(crate) fn dtype_token(ident: &str) -> TokenStream2 {
    match ident {
        "f32" => quote! { ::metaltile::core::DType::F32 },
        "f16" => quote! { ::metaltile::core::DType::F16 },
        "bf16" => quote! { ::metaltile::core::DType::BF16 },
        "i32" => quote! { ::metaltile::core::DType::I32 },
        "u32" => quote! { ::metaltile::core::DType::U32 },
        "i8" => quote! { ::metaltile::core::DType::I8 },
        "u8" => quote! { ::metaltile::core::DType::U8 },
        other => {
            let msg = format!("unknown dtype `{other}` in dtypes = [...]");
            quote! { compile_error!(#msg) }
        },
    }
}

/// Expand `#[bench(...)]` on a setup function into a `KernelBench` impl.
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let bench_attr = syn::parse_macro_input!(attr as BenchAttr);
    let input_fn = syn::parse_macro_input!(item as ItemFn);

    if input_fn.sig.inputs.len() != 1 {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[bench] setup function must take exactly one argument: `dt: DType`",
        )
        .into_compile_error()
        .into();
    }

    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    // Private impl struct — unique per function name within the module.
    let impl_name = syn::Ident::new(&format!("__BenchImpl_{fn_name_str}"), fn_name.span());

    let name_lit = &bench_attr.name;
    let dtype_tokens: Vec<TokenStream2> =
        bench_attr.dtypes.iter().map(|id| dtype_token(&id.to_string())).collect();

    let bytes_impl: TokenStream2 = match &bench_attr.bytes {
        Some(expr) => quote! {
            fn bytes_moved(
                &self,
                setup: &::metaltile::harness::bench::BenchSetup,
            ) -> u64 {
                (#expr)(setup)
            }
        },
        None => quote! {},
    };

    let flops_impl: TokenStream2 = match &bench_attr.flops {
        Some(expr) => quote! {
            fn flops(
                &self,
                setup: &::metaltile::core::bench::BenchSetup,
            ) -> ::std::option::Option<u64> {
                ::std::option::Option::Some((#expr)(setup))
            }
        },
        None => quote! {},
    };

    let metal_ref_impl: TokenStream2 = match &bench_attr.metal_ref {
        Some(expr) => quote! {
            fn reference_kernel(
                &self,
            ) -> ::std::option::Option<::metaltile::harness::bench::RefKernel> {
                ::std::option::Option::Some(#expr)
            }
        },
        None => quote! {},
    };

    let static_name = syn::Ident::new(&format!("__STATIC_{fn_name_str}"), fn_name.span());

    TokenStream::from(quote! {
        #input_fn

        #[allow(non_camel_case_types)]
        struct #impl_name;

        impl ::metaltile::harness::bench::KernelBench for #impl_name {
            fn name(&self) -> &str { #name_lit }

            fn dtypes(&self) -> &[::metaltile::core::DType] {
                &[#(#dtype_tokens),*]
            }

            fn setup(
                &self,
                dt: ::metaltile::core::DType,
            ) -> ::metaltile::harness::bench::BenchSetup {
                #fn_name(dt)
            }

            #bytes_impl
            #flops_impl
            #metal_ref_impl
        }

        #[allow(non_upper_case_globals)]
        static #static_name: #impl_name = #impl_name;
        ::metaltile::core::inventory::submit! {
            ::metaltile::harness::bench::KernelBenchEntry::new(&#static_name, file!())
        }
    })
}

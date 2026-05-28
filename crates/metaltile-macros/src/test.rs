//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[test_kernel]` proc-macro attribute implementation.
//!
//! Generates a `KernelTest` impl and inventory submission from a plain
//! setup function annotated with `#[test_kernel(name = "...", dtypes = [...])]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Ident, ItemFn, LitFloat, LitStr, Token, parse::ParseStream};

use crate::bench::dtype_token;

/// Parsed arguments for the `#[test_kernel]` attribute.
struct TestAttr {
    /// Test name, e.g. `"unary/exp"`.
    name: LitStr,
    /// Data types to test, e.g. `[f32, f16, bf16]`.
    dtypes: Vec<Ident>,
    /// Element-wise tolerance override (default: `1e-4`).
    tol: Option<f64>,
}

impl syn::parse::Parse for TestAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut dtypes = None;
        let mut tol = None;

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
            } else if key == "tol" {
                let lit = input.parse::<LitFloat>()?;
                tol = Some(lit.base10_parse::<f64>().map_err(|_| {
                    syn::Error::new(lit.span(), "tol must be a float literal, e.g. 1e-4")
                })?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown #[test_kernel] key `{key}` — valid keys: name, dtypes, tol"),
                ));
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(TestAttr {
            name: name.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[test_kernel] requires `name = \"...\"`",
                )
            })?,
            dtypes: dtypes.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[test_kernel] requires `dtypes = [f32, ...]`",
                )
            })?,
            tol,
        })
    }
}

/// Expand `#[test_kernel(...)]` on a setup function into a `KernelTest` impl.
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let test_attr = syn::parse_macro_input!(attr as TestAttr);
    let input_fn = syn::parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    // Private impl struct — unique per function name within the module.
    let impl_name = syn::Ident::new(&format!("__TestImpl_{fn_name_str}"), fn_name.span());

    let name_lit = &test_attr.name;
    let dtype_tokens: Vec<TokenStream2> =
        test_attr.dtypes.iter().map(|id| dtype_token(&id.to_string())).collect();

    let tol_impl: TokenStream2 = match test_attr.tol {
        Some(tol) => quote! {
            fn tolerance(&self, _dt: ::metaltile::core::DType) -> f64 { #tol }
        },
        None => quote! {},
    };

    let static_name = syn::Ident::new(&format!("__STATIC_{fn_name_str}"), fn_name.span());

    TokenStream::from(quote! {
        #input_fn

        #[allow(non_camel_case_types)]
        struct #impl_name;

        impl ::metaltile::core::bench::KernelTest for #impl_name {
            fn name(&self) -> &str { #name_lit }

            fn dtypes(&self) -> &[::metaltile::core::DType] {
                &[#(#dtype_tokens),*]
            }

            fn setup(
                &self,
                dt: ::metaltile::core::DType,
            ) -> ::metaltile::core::bench::TestSetup {
                #fn_name(dt)
            }

            #tol_impl
        }

        #[allow(non_upper_case_globals)]
        static #static_name: #impl_name = #impl_name;
        ::metaltile::core::inventory::submit! {
            ::metaltile::core::KernelTestEntry::new(&#static_name)
        }
    })
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[test_kernel]` proc-macro attribute implementation.
//!
//! Generates a `KernelTest` impl and inventory submission from a plain
//! setup function annotated with `#[test_kernel(dtypes = [...])]`.
//! The test name is taken from the annotated function's identifier.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Ident, ItemFn, LitFloat, Token, parse::ParseStream};

use crate::{
    bench::dtype_token,
    kernel::variants::{self, VariantsSpec},
};

#[derive(Clone)]
enum Tolerance {
    Scalar(f64),
    Table(Vec<(Ident, f64)>),
    Array(Vec<f64>),
}

/// Parsed arguments for the `#[test_kernel]` attribute.
#[derive(Clone)]
struct TestAttr {
    /// Optional explicit test name; if absent the function name is used.
    name: Option<syn::LitStr>,
    /// Data types to test, e.g. `[f32, f16, bf16]`.
    dtypes: Vec<Ident>,
    /// Element-wise tolerance override (default: `1e-4`).
    tol: Option<Tolerance>,
    /// Optional compile-time variant specialisation.  When present, the macro
    /// expands into N test registrations — one per variant — substituting
    /// integer constants into the function body and renaming each clone.
    variants: Option<VariantsSpec>,
}

fn parse_tol_value(input: ParseStream) -> syn::Result<f64> {
    let lit = input.parse::<LitFloat>()?;
    lit.base10_parse::<f64>()
        .map_err(|_| syn::Error::new(lit.span(), "tol must be a float literal, e.g. 1e-4"))
}

fn parse_tolerance(input: ParseStream) -> syn::Result<Tolerance> {
    if input.peek(syn::token::Bracket) {
        let content;
        syn::bracketed!(content in input);
        let mut vals = Vec::new();
        while !content.is_empty() {
            vals.push(parse_tol_value(&content)?);
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        return Ok(Tolerance::Array(vals));
    }
    if input.peek(syn::token::Brace) {
        let content;
        syn::braced!(content in input);
        let mut table = Vec::new();
        while !content.is_empty() {
            let dtype: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            table.push((dtype, parse_tol_value(&content)?));
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        return Ok(Tolerance::Table(table));
    }
    parse_tol_value(input).map(Tolerance::Scalar)
}

fn validate_tolerance(dtypes: &[Ident], tol: &Tolerance) -> syn::Result<()> {
    match tol {
        Tolerance::Scalar(_) => Ok(()),
        Tolerance::Array(vals) => {
            if vals.len() != dtypes.len() {
                return Err(syn::Error::new(
                    dtypes.last().map_or(proc_macro2::Span::call_site(), Ident::span),
                    format!(
                        "tol array has {} values but dtypes has {} entries",
                        vals.len(),
                        dtypes.len()
                    ),
                ));
            }
            Ok(())
        },
        Tolerance::Table(table) => {
            let dtypes_set =
                dtypes.iter().map(ToString::to_string).collect::<std::collections::BTreeSet<_>>();
            let mut seen = std::collections::BTreeSet::new();
            for (dtype, _) in table {
                let name = dtype.to_string();
                if !dtypes_set.contains(&name) {
                    return Err(syn::Error::new(
                        dtype.span(),
                        format!("tol table includes dtype `{name}` not listed in `dtypes = [...]`"),
                    ));
                }
                if !seen.insert(name.clone()) {
                    return Err(syn::Error::new(
                        dtype.span(),
                        format!("duplicate tol entry for dtype `{name}`"),
                    ));
                }
            }

            for dtype in dtypes {
                let name = dtype.to_string();
                if !seen.contains(&name) {
                    return Err(syn::Error::new(
                        dtype.span(),
                        format!("tol table missing dtype `{name}` listed in `dtypes = [...]`"),
                    ));
                }
            }

            Ok(())
        },
    }
}

fn dtype_match_token(dtype: &Ident) -> syn::Result<TokenStream2> {
    let s = dtype.to_string();
    match s.as_str() {
        "f32" | "f16" | "bf16" | "i32" | "u32" | "i8" | "u8" => Ok(dtype_token(&s)),
        other =>
            Err(syn::Error::new(dtype.span(), format!("unknown dtype `{other}` in tol table"))),
    }
}

impl syn::parse::Parse for TestAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut dtypes = None;
        let mut tol = None;
        let mut variants_spec = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;

            if key == "variants" {
                // `variants(...)` — no `=`, parenthesised spec follows.
                let inner;
                syn::parenthesized!(inner in input);
                variants_spec = Some(inner.parse::<VariantsSpec>()?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }

            input.parse::<Token![=]>()?;

            if key == "name" {
                name = Some(input.parse::<syn::LitStr>()?);
            } else if key == "dtypes" {
                let content;
                syn::bracketed!(content in input);
                let list = content.parse_terminated(Ident::parse, Token![,])?;
                dtypes = Some(list.into_iter().collect::<Vec<_>>());
            } else if key == "tol" {
                tol = Some(parse_tolerance(input)?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    format!(
                        "unknown #[test_kernel] key `{key}` —                          valid keys: name, dtypes, tol, variants"
                    ),
                ));
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let attr = TestAttr {
            name,
            dtypes: dtypes.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[test_kernel] requires `dtypes = [f32, ...]`",
                )
            })?,
            tol,
            variants: variants_spec,
        };
        if let Some(tol) = &attr.tol {
            validate_tolerance(&attr.dtypes, tol)?;
        }
        Ok(attr)
    }
}

/// Expand `#[test_kernel(...)]` on a setup function into a `KernelTest` impl.
/// Expand a single concrete `#[test_kernel]` function into its `KernelTest` impl.
///
/// Called once per variant when `variants(...)` is present, or once directly
/// for a plain `#[test_kernel]`.
fn expand_one(
    test_attr: &TestAttr,
    explicit_name: Option<&syn::LitStr>,
    input_fn: ItemFn,
) -> TokenStream2 {
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let impl_name = syn::Ident::new(&format!("__TestImpl_{fn_name_str}"), fn_name.span());

    let name_lit = explicit_name
        .cloned()
        .or_else(|| test_attr.name.clone())
        .unwrap_or_else(|| syn::LitStr::new(&fn_name_str, fn_name.span()));
    let dtype_tokens: Vec<TokenStream2> =
        test_attr.dtypes.iter().map(|id| dtype_token(&id.to_string())).collect();

    let tol_impl: TokenStream2 = match test_attr.tol.clone() {
        Some(Tolerance::Scalar(tol)) => quote! {
            fn tolerance(&self, _dt: ::metaltile::core::DType) -> f64 { #tol }
        },
        Some(Tolerance::Array(vals)) => {
            let dtypes = test_attr.dtypes.clone();
            let arms = dtypes.iter().zip(vals.iter()).map(|(dtype, tol)| {
                let dt = dtype_token(&dtype.to_string());
                quote! { #dt => #tol }
            });
            quote! {
                fn tolerance(&self, dt: ::metaltile::core::DType) -> f64 {
                    match dt {
                        #(#arms,)*
                        _ => unreachable!("dtype {:?} missing from tol array for {}", dt, #name_lit),
                    }
                }
            }
        },
        Some(Tolerance::Table(table)) => {
            let arms = match table
                .iter()
                .map(|(dtype, tol)| Ok((dtype_match_token(dtype)?, *tol)))
                .collect::<syn::Result<Vec<_>>>()
            {
                Ok(arms) => arms,
                Err(err) => return err.into_compile_error(),
            };
            let arms = arms.iter().map(|(dtype, tol)| quote! { #dtype => #tol });
            quote! {
                fn tolerance(&self, dt: ::metaltile::core::DType) -> f64 {
                    match dt {
                        #(#arms,)*
                        _ => unreachable!("dtype {:?} missing from tol table for {}", dt, #name_lit),
                    }
                }
            }
        },
        None => quote! {},
    };

    let static_name = syn::Ident::new(&format!("__STATIC_{fn_name_str}"), fn_name.span());

    quote! {
        #input_fn

        #[allow(non_camel_case_types)]
        struct #impl_name;

        impl ::metaltile::harness::test::KernelTest for #impl_name {
            fn name(&self) -> &str { #name_lit }

            fn dtypes(&self) -> &[::metaltile::core::DType] {
                &[#(#dtype_tokens),*]
            }

            fn setup(
                &self,
                dt: ::metaltile::core::DType,
            ) -> ::metaltile::harness::test::TestSetup {
                #fn_name(dt)
            }

            #tol_impl
        }

        #[allow(non_upper_case_globals)]
        static #static_name: #impl_name = #impl_name;
        ::metaltile::core::inventory::submit! {
            ::metaltile::harness::test::KernelTestEntry::new(&#static_name, file!())
        }
    }
}

/// Expand `#[test_kernel(...)]` on a setup function.
///
/// Without `variants(...)` this emits a single `KernelTest` registration.
/// With `variants(...)` it emits one registration per variant, substituting
/// integer constants into the function body and renaming each clone.
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: proc_macro2::TokenStream = attr.into();
    let item2: proc_macro2::TokenStream = item.into();

    let test_attr = match syn::parse2::<TestAttr>(attr2) {
        Ok(a) => a,
        Err(e) => return e.into_compile_error().into(),
    };
    let input_fn = match syn::parse2::<ItemFn>(item2) {
        Ok(f) => f,
        Err(e) => return e.into_compile_error().into(),
    };

    if input_fn.sig.inputs.len() != 1 {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[test_kernel] setup function must take exactly one argument: `dt: DType`",
        )
        .into_compile_error()
        .into();
    }

    let Some(spec) = test_attr.variants.as_ref() else {
        // Plain #[test_kernel] — single expansion.
        return expand_one(&test_attr, None, input_fn).into();
    };

    // Variants path: expand once per variant.
    let base_name = input_fn.sig.ident.to_string();
    let mut out = proc_macro2::TokenStream::new();

    for i in 0..spec.variant_count {
        let params_ordered: Vec<(String, i64)> =
            spec.params.iter().map(|(n, vals)| (n.clone(), vals[i])).collect();

        let variant_fn = match variants::substitute_fn(
            input_fn.clone(),
            &params_ordered,
            &base_name,
            &spec.suffix,
        ) {
            Ok(f) => f,
            Err(e) => return e.into_compile_error().into(),
        };

        out.extend(expand_one(&test_attr, None, variant_fn));
    }

    out.into()
}

#[cfg(test)]
mod tests {
    use super::{TestAttr, Tolerance};

    #[test]
    fn parses_scalar_tolerance() {
        let attr: TestAttr = syn::parse_str(r#"dtypes = [f32, f16], tol = 1e-4"#).unwrap();
        assert!(matches!(attr.tol, Some(Tolerance::Scalar(t)) if (t - 1e-4).abs() < f64::EPSILON));
    }

    #[test]
    fn parses_dtype_tolerance_table() {
        let attr: TestAttr =
            syn::parse_str(r#"dtypes = [f32, f16], tol = { f32: 1e-6, f16: 1e-3 }"#).unwrap();
        assert!(matches!(attr.tol, Some(Tolerance::Table(table)) if table.len() == 2));
    }

    #[test]
    fn rejects_incomplete_tolerance_table() {
        let err = syn::parse_str::<TestAttr>(r#"dtypes = [f32, f16], tol = { f32: 1e-6 }"#)
            .err()
            .unwrap();
        assert!(err.to_string().contains("missing dtype `f16`"));
    }

    #[test]
    fn parses_array_tolerance() {
        let attr: TestAttr =
            syn::parse_str(r#"dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e0]"#).unwrap();
        assert!(matches!(&attr.tol, Some(Tolerance::Array(vals)) if vals.len() == 3));
        if let Some(Tolerance::Array(vals)) = &attr.tol {
            assert!((vals[0] - 1e-6).abs() < f64::EPSILON);
            assert!((vals[1] - 1e-3).abs() < f64::EPSILON);
            assert!((vals[2] - 1e0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn rejects_array_tolerance_wrong_length() {
        let err = syn::parse_str::<TestAttr>(r#"dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3]"#)
            .err()
            .unwrap();
        assert!(err.to_string().contains("tol array has 2 values but dtypes has 3"));
    }
}

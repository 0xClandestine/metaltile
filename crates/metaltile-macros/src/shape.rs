//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `shape!` and `tile!` proc-macro implementations.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

/// Expand `shape!(dim, dim, ...)` into a `Shape` constructor expression.
pub(crate) fn expand_shape(input: TokenStream) -> TokenStream {
    let dims: Vec<_> = input
        .to_string()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let dim_exprs: Vec<_> = dims.iter().map(|d| parse_dim_expr(d)).collect();

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

/// Expand `tile!(rows, cols)` into a 2D `Shape` via `metaltile_core::shape::tile`.
pub(crate) fn expand_tile(input: TokenStream) -> TokenStream {
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

/// Parse a single dimension string into a `Dim::Known` or `Dim::ConstExpr` token.
pub(crate) fn parse_dim_expr(s: &str) -> TokenStream2 {
    if let Ok(n) = s.parse::<usize>() {
        quote! { Dim::Known(#n) }
    } else {
        let ident = syn::Ident::new(s, proc_macro2::Span::call_site());
        quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
    }
}

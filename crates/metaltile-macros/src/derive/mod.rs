//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[derive(ValueRefs)]`, `#[derive(OpFlags)]`, and `#[derive(VariantName)]` for the `Op` enum.
//!
//! ## Generated code references core types by name
//!
//! The generated `impl Op` methods reference these `metaltile_core` symbols
//! through `quote!` tokens — if any are renamed in `metaltile-core`, the
//! consumer's rustc error will point at generated code.  Keep this list in
//! sync:
//!
//! - `::smallvec::SmallVec`
//! - `ValueId` (bare name, imported from `metaltile_core::ir`)
//! - `IndexExpr::value_id()` / `IndexExpr::value_id_mut()`
//! - `Op` (bare name, the enum itself — self-referential in `#[vid_recursive]` arms)
//!
//! ## ValueRefs
//!
//! Generates two methods on the annotated enum:
//!
//! ```ignore
//! pub fn value_refs(&self) -> ::smallvec::SmallVec<[&ValueId; 4]>
//! pub fn for_each_value_id_mut<F: FnMut(&mut ValueId)>(&mut self, f: &mut F)
//! ```
//!
//! Field annotations control which fields participate:
//!
//! | Annotation      | Field type          | Behaviour                                      |
//! |-----------------|---------------------|------------------------------------------------|
//! | `#[vid]`        | `ValueId`           | Single value                                   |
//! | `#[vid_opt]`    | `Option<ValueId>`   | Included when `Some`                           |
//! | `#[vid_vec]`    | `Vec<ValueId>`      | All elements                                   |
//! | `#[vid_exprs]`  | `Vec<IndexExpr>`    | Calls `ix.value_id()` / `ix.value_id_mut()`   |
//! | `#[vid_recursive]` | `Vec<Op>`        | Recurses into each sub-op                      |
//!
//! Unannotated fields are ignored.
//!
//! ## OpFlags
//!
//! Generates predicate methods from variant-level annotations:
//!
//! | Annotation        | Generated method         |
//! |-------------------|--------------------------|
//! | `#[elementwise]`  | `is_elementwise() -> bool` |
//! | `#[side_effect]`  | `has_side_effects() -> bool` |
//! | `#[unpredictable]`| `is_unpredictable() -> bool` |
//! | `#[cheap_alu]`    | `is_cheap_alu() -> bool` |
//! | `#[op_load]`      | `is_load() -> bool`      |

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

// ---------------------------------------------------------------------------
// ValueRefs derive
// ---------------------------------------------------------------------------

pub fn derive_value_refs(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input, "ValueRefs can only be derived on enums")
            .to_compile_error()
            .into();
    };

    let mut refs_arms: Vec<TokenStream2> = Vec::new();
    let mut visit_arms: Vec<TokenStream2> = Vec::new();

    for variant in &data.variants {
        let vname = &variant.ident;

        let named = match &variant.fields {
            Fields::Named(n) => &n.named,
            Fields::Unit => {
                refs_arms.push(quote! { #name::#vname => {} });
                visit_arms.push(quote! { #name::#vname => {} });
                continue;
            },
            Fields::Unnamed(_) => {
                refs_arms.push(quote! { #name::#vname(..) => {} });
                visit_arms.push(quote! { #name::#vname(..) => {} });
                continue;
            },
        };

        // Collect fields that have a vid annotation.
        struct AnnotatedField<'a> {
            ident: &'a syn::Ident,
            kind: VidKind,
        }
        enum VidKind {
            Plain,
            Opt,
            Vec,
            Exprs,
            Recursive,
        }

        let mut annotated: Vec<AnnotatedField<'_>> = Vec::new();
        for field in named {
            let fname = field.ident.as_ref().unwrap();
            let kind = if has_attr(field, "vid") {
                VidKind::Plain
            } else if has_attr(field, "vid_opt") {
                VidKind::Opt
            } else if has_attr(field, "vid_vec") {
                VidKind::Vec
            } else if has_attr(field, "vid_exprs") {
                VidKind::Exprs
            } else if has_attr(field, "vid_recursive") {
                VidKind::Recursive
            } else {
                continue;
            };
            annotated.push(AnnotatedField { ident: fname, kind });
        }

        if annotated.is_empty() {
            refs_arms.push(quote! { #name::#vname { .. } => {} });
            visit_arms.push(quote! { #name::#vname { .. } => {} });
            continue;
        }

        let field_names: Vec<_> = annotated.iter().map(|f| f.ident).collect();

        let refs_stmts: Vec<TokenStream2> = annotated
            .iter()
            .map(|f| {
                let fname = f.ident;
                match f.kind {
                    VidKind::Plain => quote! { refs.push(#fname); },
                    VidKind::Opt => quote! { if let Some(v) = #fname { refs.push(v); } },
                    VidKind::Vec => quote! { refs.extend(#fname.iter()); },
                    VidKind::Exprs => quote! {
                        for _ix in #fname.iter() {
                            if let Some(_v) = _ix.value_id() { refs.push(_v); }
                        }
                    },
                    VidKind::Recursive => quote! {
                        for _op in #fname.iter() { refs.extend(_op.value_refs()); }
                    },
                }
            })
            .collect();

        let visit_stmts: Vec<TokenStream2> = annotated
            .iter()
            .map(|f| {
                let fname = f.ident;
                match f.kind {
                    VidKind::Plain => quote! { f(#fname); },
                    VidKind::Opt => quote! { if let Some(v) = #fname { f(v); } },
                    VidKind::Vec => quote! { for v in #fname.iter_mut() { f(v); } },
                    VidKind::Exprs => quote! {
                        for _ix in #fname.iter_mut() {
                            if let Some(_v) = _ix.value_id_mut() { f(_v); }
                        }
                    },
                    VidKind::Recursive => quote! {
                        for _op in #fname.iter_mut() { _op.for_each_value_id_mut(f); }
                    },
                }
            })
            .collect();

        refs_arms.push(quote! {
            #name::#vname { #(#field_names,)* .. } => { #(#refs_stmts)* }
        });
        visit_arms.push(quote! {
            #name::#vname { #(#field_names,)* .. } => { #(#visit_stmts)* }
        });
    }

    quote! {
        impl #name {
            /// Collect read-only references to every `ValueId` in this op.
            ///
            /// The `SmallVec<[&ValueId; 4]>` is stack-allocated for ops with
            /// ≤4 value references (covers ~95 % of all ops). Variadic ops
            /// (`InlineMsl.inputs`, `Cat.values`, `FusedElementwise`) spill to
            /// the heap.
            pub fn value_refs(&self) -> ::smallvec::SmallVec<[&ValueId; 4]> {
                let mut refs = ::smallvec::SmallVec::new();
                match self { #(#refs_arms,)* }
                refs
            }

            /// Visit every `ValueId` in this op mutably via a callback.
            ///
            /// Prefer this over `value_refs_mut()` for substitution passes:
            /// the callback pattern avoids lifetime conflicts when `ValueId`s
            /// are nested inside `Vec<IndexExpr>` or `Vec<Op>`.
            pub fn for_each_value_id_mut<F: FnMut(&mut ValueId)>(&mut self, f: &mut F) {
                match self { #(#visit_arms,)* }
            }
        }
    }
    .into()
}

// ---------------------------------------------------------------------------
// VariantName derive
// ---------------------------------------------------------------------------

/// Generates `fn variant_name(&self) -> &'static str` from variant idents.
///
/// Supports `#[variant_name("CustomName")]` on variants that need a name different
/// from their Rust identifier.
pub fn derive_variant_name(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input, "VariantName can only be derived on enums")
            .to_compile_error()
            .into();
    };

    let arms: Vec<TokenStream2> = data
        .variants
        .iter()
        .map(|v| {
            let vname = &v.ident;
            // Check for explicit #[name("...")] override.
            let override_name = variant_name_override(v);
            let label = match override_name {
                Some(s) => quote! { #s },
                None => {
                    let s = vname.to_string();
                    quote! { #s }
                },
            };
            match &v.fields {
                Fields::Unit => quote! { #name::#vname => #label },
                _ => quote! { #name::#vname { .. } => #label },
            }
        })
        .collect();

    let expanded = quote! {
        impl #name {
            /// A stable display name for this variant, used in error messages and diagnostics.
            pub fn variant_name(&self) -> &'static str {
                match self { #(#arms,)* }
            }
        }
    };
    expanded.into()
}

/// Extract a `#[variant_name("...")]` override from a variant's attributes.
fn variant_name_override(variant: &syn::Variant) -> Option<String> {
    for attr in &variant.attrs {
        if !attr.path().is_ident("variant_name") {
            continue;
        }
        // Parse the attribute: #[name("Foo")]
        if let syn::Meta::NameValue(mnv) = &attr.meta
            && let syn::Expr::Lit(expr_lit) = &mnv.value
            && let syn::Lit::Str(s) = &expr_lit.lit
        {
            return Some(s.value());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// OpFlags derive
// ---------------------------------------------------------------------------

pub fn derive_op_flags(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input, "OpFlags can only be derived on enums")
            .to_compile_error()
            .into();
    };

    // For each flag, collect the variant patterns that set it.
    let flags: &[(&str, &str)] = &[
        ("elementwise", "is_elementwise"),
        ("side_effect", "has_side_effects"),
        ("unpredictable", "is_unpredictable"),
        ("cheap_alu", "is_cheap_alu"),
        ("op_load", "is_load"),
        ("op_store", "is_store"),
        ("barrier", "is_barrier"),
        ("op_loop", "is_loop"),
        ("op_if", "is_if"),
        ("op_fused", "is_fused_elementwise"),
        ("op_const", "is_const"),
        ("shape_op", "is_shape_op"),
        // Feature-requirement flags
        ("needs_simd_lane", "needs_simd_lane"),
        ("needs_simd_group", "needs_simd_group"),
        ("needs_simdgroup_matrix", "needs_simdgroup_matrix"),
        ("needs_simd_product", "needs_simd_product"),
        // Result classification
        ("no_result", "is_no_result"),
        // Result-type hints for type inference
        ("result_u32", "is_result_u32_scalar"),
        ("result_i32", "is_result_i32_scalar"),
        ("result_f32_scalar", "is_result_f32_scalar"),
        ("result_f16_scalar", "is_result_f16_scalar"),
        ("result_same_type", "is_result_same_type"),
        ("result_custom", "is_result_custom"),
    ];

    let methods: Vec<TokenStream2> = flags
        .iter()
        .map(|(attr, method_name)| {
            let method = syn::Ident::new(method_name, proc_macro2::Span::call_site());
            let matching: Vec<TokenStream2> = data
                .variants
                .iter()
                .filter(|v| has_variant_attr(v, attr))
                .map(|v| {
                    let vname = &v.ident;
                    match &v.fields {
                        Fields::Unit => quote! { #name::#vname },
                        _ => quote! { #name::#vname { .. } },
                    }
                })
                .collect();

            if matching.is_empty() {
                quote! {
                    pub fn #method(&self) -> bool { false }
                }
            } else {
                quote! {
                    pub fn #method(&self) -> bool {
                        matches!(self, #(#matching)|*)
                    }
                }
            }
        })
        .collect();

    quote! {
        impl #name {
            #(#methods)*
        }
    }
    .into()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn has_attr(field: &syn::Field, name: &str) -> bool {
    field.attrs.iter().any(|a| a.path().is_ident(name))
}

fn has_variant_attr(variant: &syn::Variant, name: &str) -> bool {
    variant.attrs.iter().any(|a| a.path().is_ident(name))
}

// ---------------------------------------------------------------------------
// Module-level re-exports used by lib.rs
// ---------------------------------------------------------------------------

/// Expand `#[derive(ValueRefs)]`.
pub(crate) fn value_refs(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_value_refs(input)
}

/// Expand `#[derive(OpFlags)]`.
pub(crate) fn op_flags(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_op_flags(input)
}

/// Expand `#[derive(VariantName)]`.
pub(crate) fn variant_name(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_variant_name(input)
}

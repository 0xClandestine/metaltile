//! `#[derive(ValueRefs)]` and `#[derive(OpFlags)]` for the `Op` enum.
//!
//! ## ValueRefs
//!
//! Generates two methods on the annotated enum:
//!
//! ```ignore
//! pub fn value_refs(&self) -> ::smallvec::SmallVec<[&ValueId; 4]>
//! pub fn for_each_value_id_mut(&mut self, f: &mut dyn FnMut(&mut ValueId))
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
            pub fn for_each_value_id_mut(&mut self, f: &mut dyn FnMut(&mut ValueId)) {
                match self { #(#visit_arms,)* }
            }
        }
    }
    .into()
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

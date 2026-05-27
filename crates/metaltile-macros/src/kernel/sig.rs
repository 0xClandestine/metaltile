//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeSet;

use quote::quote;

/// Parse tensor parameters from function signature into IR param declarations.
/// `type_vars` maps type-param names (e.g. "T") to their DType arg-variable tokens (e.g. `_t`).
///
/// ## Persistent state-buffer convention
///
/// A `mut Tensor<T>` parameter is marked `is_output: true` and emits MSL
/// `device T*` (non-const). Within a single kernel dispatch, both `Op::Load`
/// and `Op::Store` may target the same buffer — Metal's memory model
/// supports read+write through a single pointer without undefined behavior.
/// This is the supported pattern for **persistent state buffers** that the
/// host re-binds to the same `MTLBuffer` across dispatches (e.g. AURA's
/// rotating index buffer): declare a single `mut state: Tensor<u32>` param
/// and intermix `load(state[..])` and `store(state[..], ..)` in the body.
///
/// Do *not* declare a separate `state_in: Tensor<T>` + `mut state_out: Tensor<T>`
/// pair and bind the same buffer to both positions on the host side — Metal's
/// `const device T*` qualifier on the read param would let the compiler
/// assume no aliasing, producing undefined behavior.
pub(crate) fn parse_kernel_params_generic(
    sig: &syn::Signature,
    type_vars: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    let mut param_builders = Vec::new();
    let use_explicit_outputs = sig.inputs.iter().any(has_mutable_tensor_param);

    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input
            && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
        {
            let param_name = pat_ident.ident.to_string();
            let ty = &pat_type.ty;

            if !is_tensor_type(ty) {
                continue;
            }

            let is_output = if use_explicit_outputs {
                pat_ident.mutability.is_some()
            } else {
                is_legacy_output_name(&param_name)
            };
            let (dtype, shape, _shape_ces) = parse_tensor_type_generic(ty, type_vars);

            let kind = if has_attr(pat_type, "scalar") {
                quote! { ParamKind::Scalar }
            } else if has_attr(pat_type, "strided") {
                quote! { ParamKind::Strided }
            } else {
                quote! { Default::default() }
            };

            param_builders.push(quote! {
                kernel.params.push(Param {
                    name: #param_name.to_string(),
                    dtype: #dtype,
                    shape: #shape,
                    is_output: #is_output,
                    kind: #kind,
                });
            });
        }
    }

    if param_builders.is_empty() {
        quote! {}
    } else {
        quote! { #(#param_builders)* }
    }
}

fn has_mutable_tensor_param(input: &syn::FnArg) -> bool {
    if let syn::FnArg::Typed(pat_type) = input {
        return matches!(&*pat_type.pat, syn::Pat::Ident(pat_ident) if pat_ident.mutability.is_some())
            && is_tensor_type(&pat_type.ty);
    }
    false
}

fn is_legacy_output_name(name: &str) -> bool { matches!(name, "out" | "c" | "output") }

/// Check if a typed parameter has a given attribute by name.
fn has_attr(pat_type: &syn::PatType, attr_name: &str) -> bool {
    pat_type.attrs.iter().any(|a| a.path().is_ident(attr_name))
}

/// Check if a type looks like a Tensor (contains "Tensor" in its path).
fn is_tensor_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        type_path.path.segments.iter().any(|seg| seg.ident == "Tensor")
    } else {
        false
    }
}

/// Returns (dtype_tokens, shape_tokens, constexpr_names_from_shape).
/// `type_vars` maps type-param names to their runtime DType arg tokens.
fn parse_tensor_type_generic(
    ty: &syn::Type,
    type_vars: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream, Vec<String>) {
    let mut dtype_tokens = quote! { DType::F32 };
    let mut shape_tokens = quote! { Shape::scalar() };
    let mut shape_ces = Vec::new();

    if let syn::Type::Path(type_path) = ty {
        for seg in &type_path.path.segments {
            if seg.ident == "Tensor"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            {
                let mut iter = args.args.iter();
                if let Some(syn::GenericArgument::Type(dtype_ty)) = iter.next() {
                    dtype_tokens = parse_dtype_generic(dtype_ty, type_vars);
                }
                if let Some(arg) = iter.next() {
                    let (tokens, ces) = parse_shape_arg(arg);
                    shape_tokens = tokens;
                    shape_ces = ces;
                }
            }
        }
    }
    (dtype_tokens, shape_tokens, shape_ces)
}

/// Backwards-compat wrapper with empty type_vars map.
fn parse_tensor_type(
    ty: &syn::Type,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream, Vec<String>) {
    parse_tensor_type_generic(ty, &std::collections::HashMap::new())
}

/// Returns (shape_tokens, constexpr_names_from_dims).
fn parse_shape_arg(arg: &syn::GenericArgument) -> (proc_macro2::TokenStream, Vec<String>) {
    let str = quote! { #arg }.to_string();
    let inner = str.trim();
    let mut ces = Vec::new();

    if let Some(start) = inner.find('(')
        && let Some(end) = inner.rfind(')')
    {
        let content = &inner[start + 1..end];
        let dims: Vec<_> =
            content.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();

        if !dims.is_empty() {
            let dim_tokens: Vec<_> = dims
                .iter()
                .map(|d| {
                    if let Ok(n) = d.parse::<usize>() {
                        quote! { Dim::Known(#n) }
                    } else {
                        ces.push(d.clone());
                        let ident = syn::Ident::new(d, proc_macro2::Span::call_site());
                        quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
                    }
                })
                .collect();

            return (
                quote! {
                    { use metaltile_core::shape::Shape; use metaltile_core::shape::Dim; use metaltile_core::constexpr::ConstExpr;
                    Shape::new([#(#dim_tokens),*]) }
                },
                ces,
            );
        }
    }
    (quote! { Shape::scalar() }, ces)
}

fn parse_dtype_generic(
    ty: &syn::Type,
    type_vars: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    if let syn::Type::Path(type_path) = ty {
        let ident = &type_path.path.segments.last().unwrap().ident;
        let name = ident.to_string();
        // If this ident is a known type parameter (T, U, ...), emit the runtime arg variable.
        if let Some(arg_tok) = type_vars.get(&name) {
            return arg_tok.clone();
        }
        return match name.as_str() {
            "f32" => quote! { DType::F32 },
            "f16" => quote! { DType::F16 },
            "bf16" => quote! { DType::BF16 },
            "i32" => quote! { DType::I32 },
            "u32" => quote! { DType::U32 },
            "i8" => quote! { DType::I8 },
            "u8" => quote! { DType::U8 },
            "bool" => quote! { DType::Bool },
            _ => quote! { DType::F32 },
        };
    }
    quote! { DType::F32 }
}

/// Extract constexpr names from `#[constexpr]` params and tensor shape dims.
#[cfg(test)]
pub(crate) fn extract_constexprs(sig: &syn::Signature) -> Vec<String> {
    extract_constexprs_typed(sig).into_iter().map(|(n, _)| n).collect()
}

/// Extract constexpr names with their DType tokens from `#[constexpr]` params and tensor shape dims.
pub(crate) fn extract_constexprs_typed(
    sig: &syn::Signature,
) -> Vec<(String, proc_macro2::TokenStream)> {
    let mut entries: Vec<(String, proc_macro2::TokenStream)> = Vec::new();
    let mut seen = BTreeSet::new();

    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if pat_type.attrs.iter().any(|a| a.path().is_ident("constexpr"))
                && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
            {
                let name = pat_ident.ident.to_string();
                let dtype = rust_type_to_dtype_tokens(&pat_type.ty);
                push_unique_typed(&mut entries, &mut seen, name, dtype);
            }

            if is_tensor_type(&pat_type.ty) {
                let (_, _, shape_ces) = parse_tensor_type(&pat_type.ty);
                for ce_name in shape_ces {
                    push_unique_typed(&mut entries, &mut seen, ce_name, quote! { DType::U32 });
                }
            }
        }
    }

    entries
}

/// Map a Rust scalar type path to a `DType::*` token stream.
fn rust_type_to_dtype_tokens(ty: &syn::Type) -> proc_macro2::TokenStream {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        return match seg.ident.to_string().as_str() {
            "f32" => quote! { DType::F32 },
            "f16" | "half" => quote! { DType::F16 },
            "f64" => quote! { DType::F64 },
            "i32" => quote! { DType::I32 },
            "i64" => quote! { DType::I64 },
            "u64" => quote! { DType::U64 },
            _ => quote! { DType::U32 },
        };
    }
    quote! { DType::U32 }
}

fn push_unique_typed(
    entries: &mut Vec<(String, proc_macro2::TokenStream)>,
    seen: &mut BTreeSet<String>,
    name: String,
    dtype: proc_macro2::TokenStream,
) {
    if seen.insert(name.clone()) {
        entries.push((name, dtype));
    }
}

/// Extract tensor parameter names from a kernel function signature.
pub(crate) fn extract_param_names(sig: &syn::Signature) -> Vec<String> {
    let mut names = Vec::new();
    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input
            && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
            && is_tensor_type(&pat_type.ty)
        {
            names.push(pat_ident.ident.to_string());
        }
    }
    names
}

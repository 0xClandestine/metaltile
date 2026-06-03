//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Compile-time kernel variant generation for `#[kernel(variants(...))]`.
//!
//! This module implements the `variants(...)` argument, which produces N
//! structurally-identical kernels that differ only in compile-time integer
//! constants.  Each listed variant gets its own kernel module and inventory
//! entry, exactly as if the user had written N separate `#[kernel]` functions.
//!
//! ## Mechanism
//!
//! 1. **Parse**: [`VariantsSpec`] reads `variants(PARAM = [...], suffix = "...")`.
//! 2. **Substitute**: [`substitute_fn`] rewrites the function body via a
//!    [`proc_macro2::TokenTree`]-level pass that replaces bare identifiers
//!    matching a parameter name with the corresponding integer literal.
//!    String literal contents are never modified.
//! 3. **Rename**: the assembled function name `base_name + "_" + suffix_value`
//!    is validated as a legal Rust identifier and set on the cloned function.
//! 4. **Expand**: `mod.rs` feeds each renamed, substituted function into the
//!    standard [`super::KernelMacroBuilder::expand_one`] pipeline unchanged.

use std::collections::HashMap;

use proc_macro2::{Literal, Span, TokenStream, TokenTree};
use syn::{
    ExprBinary,
    ExprLit,
    ExprParen,
    ExprPath,
    ItemFn,
    Token,
    parse::{Parse, ParseStream},
};

// ── Public types ─────────────────────────────────────────────────────────────

/// Parsed `variants(...)` argument block from `#[kernel(variants(...))]`.
///
/// All parameter value lists have equal length ([`variant_count`]).
#[derive(Debug)]
pub(crate) struct VariantsSpec {
    /// Named compile-time parameters in declaration order.
    ///
    /// Each tuple is `(param_name, values_per_variant)`.  All inner [`Vec`]s
    /// have length [`variant_count`].
    pub params: Vec<(String, Vec<i64>)>,

    /// Optional suffix template string, e.g. `"m{M}"` or `"b{BITS}"`.
    ///
    /// When `None`, an auto-suffix is derived by appending
    /// `_{lowercase_param}{value}` for each parameter in declaration order.
    pub suffix: Option<String>,

    /// Total number of variants (= length of each parameter's value list).
    pub variant_count: usize,
}

// ── Parsing ───────────────────────────────────────────────────────────────────

impl Parse for VariantsSpec {
    /// Parse the token stream inside `variants(...)`.
    ///
    /// Grammar (comma-separated, trailing comma allowed):
    /// ```text
    /// IDENT = [ INT_LIT , ... ]
    /// suffix = "TEMPLATE_STRING"
    /// ```
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut params: Vec<(String, Vec<i64>)> = Vec::new();
        let mut suffix: Option<String> = None;
        let mut first = true;

        while !input.is_empty() {
            if !first {
                let _comma: Token![,] = input.parse()?;
                // Allow trailing comma.
                if input.is_empty() {
                    break;
                }
            }
            first = false;

            let ident: syn::Ident = input.parse()?;
            let _eq: Token![=] = input.parse()?;
            let name = ident.to_string();

            if name == "suffix" {
                let lit: syn::LitStr = input.parse()?;
                suffix = Some(lit.value());
            } else {
                let bracket_content;
                syn::bracketed!(bracket_content in input);
                let values = parse_integer_list(&bracket_content, &ident)?;
                params.push((name, values));
            }
        }

        if params.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "variants: at least one parameter list is required",
            ));
        }

        // All parameter lists must have the same length.
        let variant_count = params[0].1.len();
        for (pname, vals) in &params {
            if vals.len() != variant_count {
                let first_name = &params[0].0;
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "variants: param lists must have equal length: \
                         {first_name}={variant_count}, {pname}={}",
                        vals.len()
                    ),
                ));
            }
        }

        Ok(VariantsSpec { params, suffix, variant_count })
    }
}

/// Parse a `[ INT_LIT , ... ]` body that has already been delimited.
fn parse_integer_list(
    content: &syn::parse::ParseBuffer<'_>,
    name_ident: &syn::Ident,
) -> syn::Result<Vec<i64>> {
    let mut values: Vec<i64> = Vec::new();
    let mut first = true;

    while !content.is_empty() {
        if !first {
            let _comma: Token![,] = content.parse()?;
            if content.is_empty() {
                break;
            }
        }
        first = false;

        let lit: syn::LitInt = content.parse().map_err(|_| {
            syn::Error::new(content.span(), "variants: list values must be integer literals")
        })?;
        values.push(lit.base10_parse::<i64>()?);
    }

    if values.is_empty() {
        return Err(syn::Error::new(name_ident.span(), "variants: param list must not be empty"));
    }
    Ok(values)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Clone and specialize a function for one set of variant parameter values.
///
/// This performs three transformations:
///
/// 1. **Body substitution**: every bare [`syn::Ident`] in the function body
///    that matches a key in `params_ordered` is replaced by the corresponding
///    integer literal via a [`TokenTree`]-level rewrite.  String literal
///    contents are **not** affected.
///
/// 2. **Name construction**: the variant suffix is evaluated from
///    `suffix_template` (or auto-derived), and the new function name is set to
///    `"{base_name}_{suffix}"`.
///
/// 3. **Validation**: the assembled name must be a valid Rust identifier.
///
/// `params_ordered` must be in declaration order so that auto-suffix
/// derivation produces a deterministic, stable name.
pub(crate) fn substitute_fn(
    mut input: ItemFn,
    params_ordered: &[(String, i64)],
    base_name: &str,
    suffix_template: &Option<String>,
) -> syn::Result<ItemFn> {
    let params: HashMap<String, i64> = params_ordered.iter().cloned().collect();

    // Evaluate (or auto-derive) the suffix string.
    let suffix_str = match suffix_template {
        Some(tmpl) => eval_suffix(tmpl, &params)?,
        None => auto_suffix(params_ordered),
    };

    // Assemble and validate the new function name.
    let new_name = format!("{base_name}_{suffix_str}");
    if syn::parse_str::<syn::Ident>(&new_name).is_err() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("variants: assembled name {new_name:?} is not a valid identifier"),
        ));
    }

    // Rewrite the function body: replace variant param idents with literals.
    let block = &input.block;
    let block_tokens: TokenStream = quote::quote! { #block };
    let substituted = substitute_tokens(block_tokens, &params);
    input.block = Box::new(syn::parse2::<syn::Block>(substituted)?);

    // Set the variant's function name.
    input.sig.ident = syn::Ident::new(&new_name, Span::call_site());

    Ok(input)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Recursively replace identifiers matching `params` keys with integer literals.
///
/// Only [`TokenTree::Ident`] tokens are considered for replacement.
/// [`TokenTree::Literal`] tokens (including string literals) are never touched.
/// [`TokenTree::Group`] tokens are recursed into, preserving their delimiter.
fn substitute_tokens(stream: TokenStream, params: &HashMap<String, i64>) -> TokenStream {
    stream
        .into_iter()
        .map(|tt| match tt {
            TokenTree::Ident(ref ident) =>
                if let Some(&val) = params.get(&ident.to_string()) {
                    let mut lit = Literal::i64_unsuffixed(val);
                    lit.set_span(ident.span());
                    TokenTree::Literal(lit)
                } else {
                    tt
                },
            TokenTree::Group(group) => {
                let inner = substitute_tokens(group.stream(), params);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), inner);
                new_group.set_span(group.span());
                TokenTree::Group(new_group)
            },
            // Punctuation and literals pass through unchanged.
            other => other,
        })
        .collect()
}

/// Evaluate a suffix template by replacing `{expr}` segments with computed
/// integer values and concatenating literal fragments between them.
///
/// ## Template syntax
///
/// ```text
/// "m{M}"           →  "m8" when M=8
/// "d{ELEMS * 32}"  →  "d256" when ELEMS=8
/// "{A}x{B}"        →  "2x1" when A=2, B=1
/// ```
///
/// Only `+`, `-`, `*`, `/`, and parenthesised sub-expressions are supported
/// inside `{...}`.  Any other operator produces a compile error.
fn eval_suffix(template: &str, params: &HashMap<String, i64>) -> syn::Result<String> {
    let mut result = String::new();
    let mut remaining = template;

    while let Some(open) = remaining.find('{') {
        // Append the literal fragment that precedes the `{`.
        result.push_str(&remaining[..open]);
        remaining = &remaining[open + 1..];

        let close = remaining.find('}').ok_or_else(|| {
            syn::Error::new(Span::call_site(), "variants: unclosed `{` in suffix template")
        })?;
        let expr_str = &remaining[..close];
        remaining = &remaining[close + 1..];

        let expr: syn::Expr = syn::parse_str(expr_str).map_err(|_| {
            syn::Error::new(
                Span::call_site(),
                format!("variants: failed to parse suffix expression `{expr_str}`"),
            )
        })?;
        let val = eval_expr(&expr, params)?;
        result.push_str(&val.to_string());
    }

    // Append any trailing literal fragment after the last `}`.
    result.push_str(remaining);
    Ok(result)
}

/// Recursively evaluate an arithmetic expression over compile-time parameters.
///
/// Supported node types: integer literal, parameter path, binary `+/-*/÷`,
/// and parenthesised expressions.  All other node types are rejected with a
/// descriptive compile error.
fn eval_expr(expr: &syn::Expr, params: &HashMap<String, i64>) -> syn::Result<i64> {
    match expr {
        // Integer literal — parse its decimal value.
        syn::Expr::Lit(ExprLit { lit: syn::Lit::Int(int), .. }) =>
            int.base10_parse::<i64>().map_err(|e| syn::Error::new(int.span(), e.to_string())),

        // Identifier — look up in params map.
        syn::Expr::Path(ExprPath { path, .. }) => {
            let name = path.get_ident().map(|i| i.to_string()).unwrap_or_default();
            params.get(&name).copied().ok_or_else(|| {
                syn::Error::new(
                    Span::call_site(),
                    format!("variants: suffix references unknown param `{name}`"),
                )
            })
        },

        // Binary arithmetic — only `+ - * /` are supported.
        syn::Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let lv = eval_expr(left, params)?;
            let rv = eval_expr(right, params)?;
            match op {
                syn::BinOp::Add(_) => Ok(lv + rv),
                syn::BinOp::Sub(_) => Ok(lv - rv),
                syn::BinOp::Mul(_) => Ok(lv * rv),
                syn::BinOp::Div(_) =>
                    if rv == 0 {
                        Err(syn::Error::new(
                            Span::call_site(),
                            "variants: division by zero in suffix expression",
                        ))
                    } else {
                        Ok(lv / rv)
                    },
                other => Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "variants: unsupported operator `{}` in suffix expression",
                        quote::quote! { #other }
                    ),
                )),
            }
        },

        // Parenthesised — recurse.
        syn::Expr::Paren(ExprParen { expr, .. }) => eval_expr(expr, params),

        _ => Err(syn::Error::new(
            Span::call_site(),
            "variants: unsupported expression type in suffix template",
        )),
    }
}

/// Auto-derive a suffix from ordered parameters when no template is provided.
///
/// Each parameter contributes `{lowercase_name}{value}`, joined with `_`.
/// For example, `M = 8` → `"m8"` so the assembled name becomes `base_m8`.
/// For multi-parameter cases an explicit `suffix = "..."` is recommended to
/// avoid unwieldy names like `elems8_phase_count2`.
fn auto_suffix(params: &[(String, i64)]) -> String {
    params
        .iter()
        .map(|(name, val)| format!("{}{val}", name.to_lowercase()))
        .collect::<Vec<_>>()
        .join("_")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use syn::parse_quote;

    use super::*;

    // ── VariantsSpec parsing ──────────────────────────────────────────────────

    #[test]
    fn single_param_correct_variant_count_and_names() {
        let spec: VariantsSpec = syn::parse_str("M = [8, 16, 32], suffix = \"m{M}\"").unwrap();
        assert_eq!(spec.variant_count, 3);
        assert_eq!(spec.params.len(), 1);
        assert_eq!(spec.params[0].0, "M");
        assert_eq!(spec.params[0].1, vec![8, 16, 32]);
        assert_eq!(spec.suffix.as_deref(), Some("m{M}"));
    }

    #[test]
    fn multi_param_zipped_not_cartesian() {
        let spec: VariantsSpec =
            syn::parse_str("ELEMS = [2, 3, 4], PHASE_COUNT = [1, 1, 2], suffix = \"d{ELEMS}\"")
                .unwrap();
        assert_eq!(spec.variant_count, 3);
        assert_eq!(spec.params[0].1, vec![2, 3, 4]);
        assert_eq!(spec.params[1].1, vec![1, 1, 2]);
    }

    #[test]
    fn error_mismatched_list_lengths() {
        let err = syn::parse_str::<VariantsSpec>("A = [1, 2], B = [1]").unwrap_err();
        assert!(err.to_string().contains("equal length"), "{err}");
    }

    #[test]
    fn error_empty_list() {
        let err = syn::parse_str::<VariantsSpec>("M = []").unwrap_err();
        assert!(err.to_string().contains("must not be empty"), "{err}");
    }

    #[test]
    fn trailing_comma_is_accepted() {
        let spec: VariantsSpec = syn::parse_str("M = [8, 16,], suffix = \"m{M}\",").unwrap();
        assert_eq!(spec.variant_count, 2);
    }

    // ── eval_suffix / eval_expr ───────────────────────────────────────────────

    #[test]
    fn suffix_literal_passthrough() {
        let params = HashMap::from([("M".to_string(), 16i64)]);
        assert_eq!(eval_suffix("prefix", &params).unwrap(), "prefix");
    }

    #[test]
    fn suffix_simple_param_substitution() {
        let params = HashMap::from([("M".to_string(), 16i64)]);
        assert_eq!(eval_suffix("m{M}", &params).unwrap(), "m16");
    }

    #[test]
    fn suffix_arithmetic_mul() {
        let params = HashMap::from([("ELEMS".to_string(), 8i64)]);
        assert_eq!(eval_suffix("d{ELEMS * 32}", &params).unwrap(), "d256");
    }

    #[test]
    fn suffix_multi_param() {
        let params = HashMap::from([("A".to_string(), 2i64), ("B".to_string(), 4i64)]);
        assert_eq!(eval_suffix("{A}x{B}", &params).unwrap(), "2x4");
    }

    #[test]
    fn suffix_paren_grouping() {
        let params = HashMap::from([("N".to_string(), 3i64)]);
        assert_eq!(eval_suffix("s{(N + 1) * 8}", &params).unwrap(), "s32");
    }

    #[test]
    fn suffix_error_unknown_param() {
        let params = HashMap::from([("M".to_string(), 8i64)]);
        let err = eval_suffix("{FOO}", &params).unwrap_err();
        assert!(err.to_string().contains("unknown param"), "{err}");
    }

    #[test]
    fn suffix_error_unsupported_operator() {
        let params = HashMap::from([("M".to_string(), 8i64)]);
        let err = eval_suffix("{M % 3}", &params).unwrap_err();
        assert!(err.to_string().contains("unsupported operator"), "{err}");
    }

    // ── auto_suffix ───────────────────────────────────────────────────────────

    #[test]
    fn auto_suffix_single_param() {
        assert_eq!(auto_suffix(&[("M".to_string(), 16)]), "m16");
    }

    #[test]
    fn auto_suffix_multi_param() {
        let s = auto_suffix(&[("ELEMS".to_string(), 4), ("PHASE_COUNT".to_string(), 2)]);
        assert_eq!(s, "elems4_phase_count2");
    }

    // ── substitute_tokens ─────────────────────────────────────────────────────

    #[test]
    fn substitution_replaces_bare_ident() {
        let params = HashMap::from([("M".to_string(), 8i64)]);
        let input: TokenStream = quote::quote! { range(0u32, M, 1u32) };
        let output = substitute_tokens(input, &params).to_string();
        // M → 8; u32-suffixed literals should remain unchanged.
        assert!(output.contains(" 8 "), "expected 8 in: {output}");
        assert!(!output.contains(" M "), "M should be gone: {output}");
    }

    #[test]
    fn substitution_does_not_touch_string_literals() {
        let params = HashMap::from([("M".to_string(), 8i64)]);
        // "M" inside a string literal must not be replaced.
        let input: TokenStream = quote::quote! { stack_alloc("M_sized", M, "f32") };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains('"'), "string literal lost");
        assert!(output.contains("M_sized"), "string literal was modified");
    }

    #[test]
    fn substitution_recurses_into_groups() {
        let params = HashMap::from([("N".to_string(), 4i64)]);
        let input: TokenStream = quote::quote! { (a + N) * b };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("4"), "substitution missed group");
    }

    // ── substitute_fn ─────────────────────────────────────────────────────────

    #[test]
    fn substitute_fn_correct_name_and_body() {
        let input_fn: ItemFn = parse_quote! {
            pub fn mt_moe<T>(x: Tensor<T>) {
                let y = M + 1u32;
            }
        };
        let params = vec![("M".to_string(), 16i64)];
        let result = substitute_fn(input_fn, &params, "mt_moe", &Some("m{M}".to_string())).unwrap();
        assert_eq!(result.sig.ident.to_string(), "mt_moe_m16");
        let body = quote::quote! { #result }.to_string();
        assert!(body.contains("16 + 1u32"), "body substitution failed: {body}");
    }

    #[test]
    fn substitute_fn_auto_suffix() {
        let input_fn: ItemFn = parse_quote! { pub fn f() { let _ = M; } };
        let params = vec![("M".to_string(), 8i64)];
        let result = substitute_fn(input_fn, &params, "f", &None).unwrap();
        assert_eq!(result.sig.ident.to_string(), "f_m8");
    }
}

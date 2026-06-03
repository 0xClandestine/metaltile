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
//!    [`proc_macro2::TokenTree`]-level pass that:
//!    - Replaces bare identifiers matching a parameter name with the
//!      corresponding integer literal (e.g. `BITS` → `4`).
//!    - Performs ident-embedding for compound identifiers that contain a param
//!      name as a substring (e.g. `kernel_intBITS` → `kernel_int4`).
//!    - Evaluates **compile-time `if` blocks** whose condition references only
//!      variant parameters and integer literals, stripping the unselected
//!      branch entirely before the `#[kernel]` body parser runs.
//!      String literal contents are never modified.
//! 3. **Rename**: the assembled function name `base_name + "_" + suffix_value`
//!    is validated as a legal Rust identifier and set on the cloned function.
//! 4. **Expand**: `mod.rs` feeds each renamed, substituted function into the
//!    standard [`super::KernelMacroBuilder::expand_one`] pipeline unchanged.
//!
//! ## Compile-time `if`
//!
//! Inside any variants-expanded body, an `if` expression whose condition
//! contains **only** known parameter names, integer literals, and operators is
//! evaluated at macro-expansion time:
//!
//! ```rust,ignore
//! #[kernel(variants(BITS = [2, 3, 4, 5, 6, 8], suffix = "int{BITS}"))]
//! pub fn dequant_gemv<T>(…) {
//!     if BITS % 2 == 0 {
//!         // compiled into int2 / int4 / int8 only
//!     } else {
//!         // compiled into int3 / int5 / int6 only
//!     }
//! }
//! ```
//!
//! **Auto-detection rule**: an `if` is compile-time when every `Ident` in its
//! condition is either a known parameter name (ALL_CAPS by convention) or the
//! literal keywords `true`/`false`.  Any other ident (lowercase variable,
//! function call, type name, …) makes the condition runtime — the `if` is
//! passed through to the `#[kernel]` body parser as `Op::If`.
//!
//! **Supported operators**: `==`, `!=`, `<`, `<=`, `>`, `>=`, `&&`, `||`,
//! `!`, `+`, `-`, `*`, `/`, `%`, and parenthesised sub-expressions.
//!
//! **`else if` chains**: the outer `else if` sub-expression is processed
//! recursively.  If its own condition is param-only it is evaluated
//! compile-time; otherwise it is emitted as a runtime `if` with substitution
//! applied to all of its parts.

use std::collections::HashMap;

use proc_macro2::{Delimiter, Group, Ident, Literal, Span, TokenStream, TokenTree};
use syn::{
    ExprBinary,
    ExprLit,
    ExprParen,
    ExprPath,
    ExprUnary,
    ItemFn,
    Token,
    UnOp,
    parse::{Parse, ParseStream},
};

// ── Public types ─────────────────────────────────────────────────────────────

/// Parsed `variants(...)` argument block from `#[kernel(variants(...))]`.
///
/// All parameter value lists have equal length ([`variant_count`]).
#[derive(Clone, Debug)]
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

/// Parsed else-branch while processing an `if`-expression in the token stream.
enum ElseBranch {
    /// A plain `else { … }` block (the group is the brace-delimited body).
    Block(Group),
    /// An `else if …` chain that has already been fully processed by
    /// [`substitute_if`].  The [`TokenStream`] is either:
    /// - the selected branch contents (when the inner condition was param-only
    ///   and evaluated compile-time), or
    /// - a full `if COND { … } [else …]` expression (fallthrough path).
    Processed(TokenStream),
}

/// Substitute variant parameters into a token stream.
///
/// Three substitution modes apply, tried in order:
///
/// 1. **Compile-time `if`** — when an `if` keyword is followed by a condition
///    that references only known parameter names and literals, the condition is
///    evaluated and the unselected branch is stripped entirely.  The body
///    parser never sees it.  `else { }` and `else if` chains are supported.
///    Any `if` whose condition references a non-param ident is passed through
///    as a runtime `if` with substitution applied to all sub-tokens.
///
/// 2. **Exact match** — a bare identifier equal to a parameter name is
///    replaced with an unsuffixed integer literal (e.g. `BITS` → `4`).
///
/// 3. **Ident-embedding** — an identifier that *contains* a parameter name
///    as a substring is rewritten with the value injected
///    (e.g. `kernel_intBITS` → `kernel_int4`).
///
/// [`TokenTree::Literal`] tokens are never modified.
/// [`TokenTree::Group`] tokens are recursed into, preserving their delimiter.
fn substitute_tokens(stream: TokenStream, params: &HashMap<String, i64>) -> TokenStream {
    let tts: Vec<TokenTree> = stream.into_iter().collect();
    let mut out = TokenStream::new();
    let mut i = 0;

    while i < tts.len() {
        if let TokenTree::Ident(ref ident) = tts[i] {
            let s = ident.to_string();

            // Mode 1 — compile-time if.
            if s == "if" {
                let (consumed, ts) = substitute_if(&tts, i, params);
                out.extend(ts);
                i += consumed;
                continue;
            }

            // Mode 2 — exact param match → integer literal.
            if let Some(&val) = params.get(&s) {
                let mut lit = Literal::i64_unsuffixed(val);
                lit.set_span(ident.span());
                out.extend(std::iter::once(TokenTree::Literal(lit)));
                i += 1;
                continue;
            }

            // Mode 3 — ident-embedding: substitute all param substrings.
            let mut new_s = s.clone();
            for (name, val) in params {
                new_s = new_s.replace(name.as_str(), &val.to_string());
            }
            if new_s != s {
                out.extend(std::iter::once(TokenTree::Ident(Ident::new(&new_s, ident.span()))));
            } else {
                out.extend(std::iter::once(tts[i].clone()));
            }
            i += 1;
            continue;
        }

        // Recurse into groups; pass everything else through unchanged.
        if let TokenTree::Group(ref group) = tts[i] {
            let inner = substitute_tokens(group.stream(), params);
            let mut new_group = Group::new(group.delimiter(), inner);
            new_group.set_span(group.span());
            out.extend(std::iter::once(TokenTree::Group(new_group)));
            i += 1;
            continue;
        }

        out.extend(std::iter::once(tts[i].clone()));
        i += 1;
    }

    out
}

/// Process an `if`-expression starting at `tts[start]` (the `if` keyword).
///
/// Returns `(tokens_consumed, output_stream)`.
///
/// When the condition is param-only and evaluates to a `bool`:
/// - The selected branch's tokens are recursively substituted and returned.
/// - The unselected branch is dropped entirely.
///
/// When the condition references non-param idents (runtime `if`):
/// - All sub-tokens (condition, then-block, else-block) have substitution
///   applied, and the full `if … { } [else { }]` expression is returned.
///
/// `else if` chains are handled recursively: the inner expression is processed
/// by a recursive call and inserted as the else-branch.
fn substitute_if(
    tts: &[TokenTree],
    start: usize,
    params: &HashMap<String, i64>,
) -> (usize, TokenStream) {
    // `tts[start]` is the `if` keyword — skip it.
    let mut i = start + 1;

    // ── Collect condition tokens (everything before the first brace group) ──
    let cond_start = i;
    while i < tts.len() {
        if let TokenTree::Group(g) = &tts[i]
            && g.delimiter() == Delimiter::Brace
        {
            break;
        }
        i += 1;
    }
    if i >= tts.len() {
        // Malformed (no then-block) — emit as-is, no substitution attempted.
        let mut ts = TokenStream::new();
        ts.extend(tts[start..i].iter().cloned());
        return (i - start, ts);
    }

    let cond_stream: TokenStream = tts[cond_start..i].iter().cloned().collect();

    // ── Consume the then-block ─────────────────────────────────────────────
    let then_group = match &tts[i] {
        TokenTree::Group(g) => g.clone(),
        _ => unreachable!("checked by loop above"),
    };
    i += 1;

    // ── Optionally consume `else { }` or `else if …` ──────────────────────
    let else_branch: Option<ElseBranch> = if i < tts.len() {
        if let TokenTree::Ident(else_kw) = &tts[i] {
            if *else_kw == "else" {
                i += 1; // consume `else`
                if i < tts.len() {
                    match &tts[i] {
                        TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                            let g = g.clone();
                            i += 1;
                            Some(ElseBranch::Block(g))
                        },
                        TokenTree::Ident(kw) if *kw == "if" => {
                            // `else if …` — recurse.
                            let (inner_consumed, inner_ts) = substitute_if(tts, i, params);
                            i += inner_consumed;
                            Some(ElseBranch::Processed(inner_ts))
                        },
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let consumed = i - start;

    // ── Try compile-time evaluation ────────────────────────────────────────
    if condition_is_param_only(&cond_stream, params)
        && let Ok(expr) = syn::parse2::<syn::Expr>(cond_stream.clone())
        && let Some(taken) = eval_bool_expr(&expr, params)
    {
        let selected = if taken {
            substitute_tokens(then_group.stream(), params)
        } else {
            match else_branch {
                Some(ElseBranch::Block(g)) => substitute_tokens(g.stream(), params),
                // Processed inner was already substituted by the recursive call.
                Some(ElseBranch::Processed(ts)) => ts,
                None => TokenStream::new(),
            }
        };
        return (consumed, selected);
    }

    // ── Fallthrough: reassemble as a runtime `if` with substitution ────────
    let mut ts = TokenStream::new();
    ts.extend(std::iter::once(tts[start].clone())); // `if` keyword
    ts.extend(substitute_tokens(cond_stream, params));

    let sub_then = substitute_tokens(then_group.stream(), params);
    let mut new_then = Group::new(Delimiter::Brace, sub_then);
    new_then.set_span(then_group.span());
    ts.extend(std::iter::once(TokenTree::Group(new_then)));

    if let Some(eb) = else_branch {
        ts.extend(std::iter::once(TokenTree::Ident(Ident::new("else", Span::call_site()))));
        match eb {
            ElseBranch::Block(g) => {
                let sub_else = substitute_tokens(g.stream(), params);
                let mut new_else = Group::new(Delimiter::Brace, sub_else);
                new_else.set_span(g.span());
                ts.extend(std::iter::once(TokenTree::Group(new_else)));
            },
            ElseBranch::Processed(inner_ts) => {
                // If the inner was compile-time-reduced it is bare branch
                // contents (no leading `if`); wrap in braces so we emit a
                // valid `else { … }`.  If the inner is a full runtime
                // if-expression its first token will be `if`, in which case
                // we emit it directly as `else if … { } …`.
                let first_is_if = inner_ts
                    .clone()
                    .into_iter()
                    .next()
                    .map(|tt| matches!(tt, TokenTree::Ident(ref id) if *id == "if"))
                    .unwrap_or(false);
                if first_is_if {
                    ts.extend(inner_ts);
                } else {
                    let wrapped = Group::new(Delimiter::Brace, inner_ts);
                    ts.extend(std::iter::once(TokenTree::Group(wrapped)));
                }
            },
        }
    }

    (consumed, ts)
}

/// Returns `true` when every [`Ident`] token in `stream` is either a known
/// variant parameter or the literal keyword `true`/`false`.
///
/// Any other identifier (lowercase variable, function name, type path, …)
/// makes the condition a runtime expression rather than a compile-time one.
fn condition_is_param_only(stream: &TokenStream, params: &HashMap<String, i64>) -> bool {
    for tt in stream.clone().into_iter() {
        match tt {
            TokenTree::Ident(ident) => {
                let s = ident.to_string();
                if s == "true" || s == "false" {
                    continue;
                }
                if !params.contains_key(&s) {
                    return false;
                }
            },
            TokenTree::Group(g) if !condition_is_param_only(&g.stream(), params) => {
                return false;
            },
            _ => {},
        }
    }
    true
}

/// Evaluate a boolean expression over compile-time variant parameters.
///
/// Returns `None` when the expression cannot be evaluated (unsupported node
/// type or arithmetic error).
///
/// Supported forms:
/// - Comparison: `PARAM == LIT`, `PARAM != LIT`, `<`, `<=`, `>`, `>=`
/// - Arithmetic in either operand (delegated to [`eval_expr`])
/// - Boolean combinators: `&&`, `||`
/// - Unary negation: `!EXPR`
/// - Parenthesised sub-expressions
/// - Literal `true` / `false`
fn eval_bool_expr(expr: &syn::Expr, params: &HashMap<String, i64>) -> Option<bool> {
    match expr {
        syn::Expr::Binary(ExprBinary { left, op, right, .. }) => match op {
            syn::BinOp::Eq(_) =>
                Some(eval_expr(left, params).ok()? == eval_expr(right, params).ok()?),
            syn::BinOp::Ne(_) =>
                Some(eval_expr(left, params).ok()? != eval_expr(right, params).ok()?),
            syn::BinOp::Lt(_) =>
                Some(eval_expr(left, params).ok()? < eval_expr(right, params).ok()?),
            syn::BinOp::Le(_) =>
                Some(eval_expr(left, params).ok()? <= eval_expr(right, params).ok()?),
            syn::BinOp::Gt(_) =>
                Some(eval_expr(left, params).ok()? > eval_expr(right, params).ok()?),
            syn::BinOp::Ge(_) =>
                Some(eval_expr(left, params).ok()? >= eval_expr(right, params).ok()?),
            syn::BinOp::And(_) =>
                Some(eval_bool_expr(left, params)? && eval_bool_expr(right, params)?),
            syn::BinOp::Or(_) =>
                Some(eval_bool_expr(left, params)? || eval_bool_expr(right, params)?),
            _ => None,
        },
        syn::Expr::Unary(ExprUnary { op: UnOp::Not(_), expr, .. }) =>
            eval_bool_expr(expr, params).map(|v| !v),
        syn::Expr::Paren(ExprParen { expr, .. }) => eval_bool_expr(expr, params),
        syn::Expr::Lit(ExprLit { lit: syn::Lit::Bool(b), .. }) => Some(b.value),
        _ => None,
    }
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
pub(crate) fn eval_suffix(template: &str, params: &HashMap<String, i64>) -> syn::Result<String> {
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

        // Binary arithmetic — `+ - * / %` are supported.
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
                syn::BinOp::Rem(_) =>
                    if rv == 0 {
                        Err(syn::Error::new(
                            Span::call_site(),
                            "variants: modulo by zero in suffix expression",
                        ))
                    } else {
                        Ok(lv % rv)
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
    fn suffix_modulo_operator() {
        // `%` is now supported; 8 % 3 == 2.
        let params = HashMap::from([("M".to_string(), 8i64)]);
        assert_eq!(eval_suffix("{M % 3}", &params).unwrap(), "2");
    }

    #[test]
    fn suffix_error_unsupported_operator() {
        // Bitwise shift is still unsupported.
        let params = HashMap::from([("M".to_string(), 8i64)]);
        let err = eval_suffix("{M << 1}", &params).unwrap_err();
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
    fn substitution_embeds_in_ident() {
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        // `dequant_gather_intBITS` is one Ident token; BITS is a substring.
        let input: TokenStream = quote::quote! { dequant_gather_intBITS::kernel_ir_for(dt) };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("dequant_gather_int4"), "expected int4 in: {output}");
        assert!(!output.contains("BITS"), "BITS should be gone: {output}");
    }

    #[test]
    fn substitution_ident_embedding_multiple_params() {
        let params = HashMap::from([("M".to_string(), 8i64), ("N".to_string(), 16i64)]);
        let input: TokenStream = quote::quote! { kernel_mMxN };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("kernel_m8x16"), "expected m8x16 in: {output}");
    }

    #[test]
    fn substitution_exact_still_emits_literal_not_ident() {
        let params = HashMap::from([("BITS".to_string(), 3i64)]);
        // Standalone BITS should become a literal 3, not an ident "3".
        let input: TokenStream = quote::quote! { k * BITS / 32u32 };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("k * 3 / 32u32"), "expected literal in: {output}");
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

    // ── eval_bool_expr ────────────────────────────────────────────────────────

    fn bool_params() -> HashMap<String, i64> {
        HashMap::from([("BITS".to_string(), 4i64), ("DIM".to_string(), 128i64)])
    }

    fn parse_and_eval(expr: &str, params: &HashMap<String, i64>) -> Option<bool> {
        let e: syn::Expr = syn::parse_str(expr).unwrap();
        eval_bool_expr(&e, params)
    }

    #[test]
    fn bool_eq_true() {
        assert_eq!(parse_and_eval("BITS == 4", &bool_params()), Some(true));
    }

    #[test]
    fn bool_eq_false() {
        assert_eq!(parse_and_eval("BITS == 8", &bool_params()), Some(false));
    }

    #[test]
    fn bool_ne() {
        assert_eq!(parse_and_eval("BITS != 8", &bool_params()), Some(true));
    }

    #[test]
    fn bool_modulo_parity_even() {
        // BITS=4 is even
        assert_eq!(parse_and_eval("BITS % 2 == 0", &bool_params()), Some(true));
    }

    #[test]
    fn bool_modulo_parity_odd() {
        let odd = HashMap::from([("BITS".to_string(), 3i64)]);
        assert_eq!(parse_and_eval("BITS % 2 == 0", &odd), Some(false));
    }

    #[test]
    fn bool_and_both_true() {
        assert_eq!(parse_and_eval("BITS == 4 && DIM == 128", &bool_params()), Some(true));
    }

    #[test]
    fn bool_and_one_false() {
        assert_eq!(parse_and_eval("BITS == 4 && DIM == 64", &bool_params()), Some(false));
    }

    #[test]
    fn bool_or_one_true() {
        assert_eq!(parse_and_eval("BITS == 8 || BITS == 4", &bool_params()), Some(true));
    }

    #[test]
    fn bool_not() {
        assert_eq!(parse_and_eval("!(BITS == 8)", &bool_params()), Some(true));
    }

    #[test]
    fn bool_literal_true() {
        assert_eq!(parse_and_eval("true", &bool_params()), Some(true));
    }

    #[test]
    fn bool_runtime_ident_returns_none() {
        // `n` is not a known param — must not be evaluated.
        assert_eq!(parse_and_eval("n > 0", &bool_params()), None);
    }

    // ── condition_is_param_only ───────────────────────────────────────────────

    #[test]
    fn param_only_true_for_params_and_literals() {
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let ts: TokenStream = quote::quote! { BITS % 2 == 0 };
        assert!(condition_is_param_only(&ts, &params));
    }

    #[test]
    fn param_only_false_for_lowercase_ident() {
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let ts: TokenStream = quote::quote! { n > 0 };
        assert!(!condition_is_param_only(&ts, &params));
    }

    #[test]
    fn param_only_allows_true_false_keywords() {
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let ts: TokenStream = quote::quote! { true };
        assert!(condition_is_param_only(&ts, &params));
    }

    // ── compile-time if substitution ─────────────────────────────────────────

    #[test]
    fn compile_time_if_selects_then_branch() {
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let x = 1u32; } else { let x = 2u32; }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("1u32"), "then-branch missing: {output}");
        assert!(!output.contains("2u32"), "else-branch leaked: {output}");
        assert!(!output.contains("if"), "if keyword should be gone: {output}");
    }

    #[test]
    fn compile_time_if_selects_else_branch() {
        let params = HashMap::from([("BITS".to_string(), 8i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let x = 1u32; } else { let x = 2u32; }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("2u32"), "else-branch missing: {output}");
        assert!(!output.contains("1u32"), "then-branch leaked: {output}");
    }

    #[test]
    fn compile_time_if_no_else_false_emits_nothing() {
        let params = HashMap::from([("BITS".to_string(), 8i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let x = 1u32; }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(!output.contains("1u32"), "branch should be gone: {output}");
        assert!(output.is_empty() || !output.contains("if"), "if leaked: {output}");
    }

    #[test]
    fn compile_time_if_modulo_parity() {
        // BITS=4 (even) → selects then-branch
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let input: TokenStream = quote::quote! {
            if BITS % 2 == 0 { pack_strided() } else { elem_strided() }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("pack_strided"), "{output}");
        assert!(!output.contains("elem_strided"), "{output}");
    }

    #[test]
    fn compile_time_if_modulo_parity_odd() {
        // BITS=3 (odd) → selects else-branch
        let params = HashMap::from([("BITS".to_string(), 3i64)]);
        let input: TokenStream = quote::quote! {
            if BITS % 2 == 0 { pack_strided() } else { elem_strided() }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("elem_strided"), "{output}");
        assert!(!output.contains("pack_strided"), "{output}");
    }

    #[test]
    fn compile_time_if_or_condition() {
        let params = HashMap::from([("BITS".to_string(), 8i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 || BITS == 8 { pack() } else { odd() }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("pack"), "{output}");
        assert!(!output.contains("odd"), "{output}");
    }

    #[test]
    fn compile_time_if_substitutes_params_in_selected_branch() {
        // BITS inside the selected branch must also be substituted.
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let n = BITS * 8u32; } else { let n = 0u32; }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("4 * 8u32"), "param not substituted in branch: {output}");
    }

    #[test]
    fn runtime_if_with_non_param_ident_passes_through() {
        let params = HashMap::from([("BITS".to_string(), 4i64)]);
        let input: TokenStream = quote::quote! {
            if n > 0 { let x = BITS; } else { let x = 0u32; }
        };
        let output = substitute_tokens(input, &params).to_string();
        // `if` must still be present; `n` stays; BITS → 4
        assert!(output.contains("if"), "if should be present: {output}");
        assert!(output.contains("n >"), "n should be present: {output}");
        assert!(output.contains("4"), "BITS should be substituted: {output}");
    }

    #[test]
    fn else_if_chain_both_param_only() {
        // BITS=3: first branch false, second branch true → picks middle
        let params = HashMap::from([("BITS".to_string(), 3i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 { path_a() } else if BITS == 3 { path_b() } else { path_c() }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("path_b"), "{output}");
        assert!(!output.contains("path_a"), "{output}");
        assert!(!output.contains("path_c"), "{output}");
    }

    #[test]
    fn else_if_chain_falls_to_final_else() {
        // BITS=8: neither inner condition matches
        let params = HashMap::from([("BITS".to_string(), 8i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 { path_a() } else if BITS == 3 { path_b() } else { path_c() }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("path_c"), "{output}");
        assert!(!output.contains("path_a"), "{output}");
        assert!(!output.contains("path_b"), "{output}");
    }

    #[test]
    fn nested_compile_time_if() {
        // Outer: BITS==4 → then; Inner: DIM==128 → then
        let params = HashMap::from([("BITS".to_string(), 4i64), ("DIM".to_string(), 128i64)]);
        let input: TokenStream = quote::quote! {
            if BITS == 4 {
                if DIM == 128 { deep() } else { shallow() }
            } else {
                other()
            }
        };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("deep"), "{output}");
        assert!(!output.contains("shallow"), "{output}");
        assert!(!output.contains("other"), "{output}");
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

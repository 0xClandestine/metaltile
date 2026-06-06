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
    punctuated::Punctuated,
    spanned::Spanned,
};

// ── Public types ─────────────────────────────────────────────────────────────

/// A single value in a `variants(...)` parameter list.
///
/// Parameters can be integer compile-time constants **or** primitive Rust types.
/// Type variants substitute into tensor element-type positions in both the
/// function signature and body (e.g. `weight: Tensor<WT>` with `WT = [u32, u8]`).
/// Integer variants substitute into expression positions and enable compile-time
/// `if` evaluation.
#[derive(Clone, Debug)]
pub(crate) enum VariantValue {
    /// A compile-time integer constant (e.g. `4`, `128`).
    Int(i64),
    /// A float literal (e.g. `0.5f32`, `1.0f32`).
    ///
    /// Float params substitute into expression positions exactly like integers
    /// (mode 2 exact match), but are excluded from compile-time `if` conditions
    /// and ident-embedding — both of which require integer arithmetic.
    Float(Literal),
    /// A primitive type (e.g. `u32`, `u8`, `f16`).
    Type(TokenStream),
}

impl VariantValue {
    /// Return the integer value, or `None` for float/type variants.
    fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(_) | Self::Type(_) => None,
        }
    }

    /// Convert to a string for use in suffix templates or auto-derived suffixes.
    ///
    /// - Integers: decimal value (`"128"`)
    /// - Floats: type suffix stripped, `.` replaced with `_` for ident safety
    ///   (`0.5f32` → `"0_5"`, `1.0` → `"1_0"`)
    /// - Types: whitespace stripped (`Vec < u8 >` → `"Vec<u8>"`)
    fn to_suffix_string(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(lit) => {
                let s = lit.to_string();
                // Strip type suffixes (f32, f64) then replace `.` with `_`.
                let s = s.trim_end_matches("f64").trim_end_matches("f32");
                s.replace('.', "_")
            },
            Self::Type(ts) => ts.to_string().replace(' ', ""),
        }
    }
}

/// Parsed `variants(...)` argument block from `#[kernel(variants(...))]`.
///
/// All parameter value lists have equal length ([`variant_count`]).
#[derive(Clone, Debug)]
pub(crate) struct VariantsSpec {
    /// Named compile-time parameters in declaration order.
    ///
    /// Each tuple is `(param_name, values_per_variant)`.  All inner [`Vec`]s
    /// have length [`variant_count`].  Values may be integers or types.
    pub params: Vec<(String, Vec<VariantValue>)>,

    /// Optional suffix template string, e.g. `"m{M}"` or `"b{BITS}"`.
    ///
    /// When `None`, an auto-suffix is derived by appending
    /// `_{lowercase_param}{value}` for each parameter in declaration order.
    pub suffix: Option<String>,

    /// Total number of variants (= length of each parameter's value list).
    pub variant_count: usize,
}

/// A `#[optional(only_when = "EXPR")]` constexpr declaration on a kernel
/// function parameter. The constexpr is included in the generated kernel's
/// MSL signature only when `EXPR` (a boolean expression over variant params
/// and integer literals) evaluates to `true` for the current variant. When
/// `EXPR` is `false`, the param is stripped from the signature entirely, so
/// the constexpr neither consumes a buffer slot nor requires a host-side
/// binding. The body must gate any reference to the optional constexpr so
/// that FMT-pruning leaves no dangling references in the pruned branches.
#[derive(Clone, Debug)]
pub(crate) struct OptionalConstexpr {
    /// The parameter name (Rust ident).
    pub name: String,
    /// The condition expression, AST form. Evaluated via the same machinery
    /// as compile-time `if` conditions.
    pub condition: syn::Expr,
    /// Span of the `#[optional(...)]` attribute (for error reporting).
    pub span: Span,
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
        let mut params: Vec<(String, Vec<VariantValue>)> = Vec::new();
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
                let values = parse_value_list(&bracket_content, &ident)?;
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

/// Parse a `[ VALUE , ... ]` body that has already been delimited.
///
/// Each element is tried in order: integer literal, float literal, type path.
/// The grammar therefore accepts:
/// - `[2, 4, 8]` — integer params
/// - `[0.5f32, 1.0f32]` — float params (substituted verbatim; excluded from
///   compile-time `if` and ident-embedding)
/// - `[u32, u8]` — type params (substituted in `Tensor<WT>` positions)
fn parse_value_list(
    content: &syn::parse::ParseBuffer<'_>,
    name_ident: &syn::Ident,
) -> syn::Result<Vec<VariantValue>> {
    let mut values: Vec<VariantValue> = Vec::new();
    let mut first = true;

    while !content.is_empty() {
        if !first {
            let _comma: Token![,] = content.parse()?;
            if content.is_empty() {
                break;
            }
        }
        first = false;

        if content.peek(syn::LitInt) {
            let lit: syn::LitInt = content.parse()?;
            values.push(VariantValue::Int(lit.base10_parse::<i64>()?));
        } else if content.peek(syn::LitFloat) {
            let lit: syn::LitFloat = content.parse()?;
            // Re-emit as a proc_macro2::Literal so we can clone and emit it
            // unchanged in substitute_tokens.
            let literal: Literal = lit.token();
            values.push(VariantValue::Float(literal));
        } else {
            let ty: syn::Type = content.parse().map_err(|_| {
                syn::Error::new(
                    content.span(),
                    "variants: list values must be integer literals, float literals, \
                     or type paths (e.g. u32, u8)",
                )
            })?;
            values.push(VariantValue::Type(quote::quote! { #ty }));
        }
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
    params_ordered: &[(String, VariantValue)],
    base_name: &str,
    suffix_template: &Option<String>,
) -> syn::Result<ItemFn> {
    let params: HashMap<String, VariantValue> = params_ordered.iter().cloned().collect();

    // Collect `#[optional(only_when = "EXPR")]` constexprs BEFORE the body
    // is FMT-substituted, so the condition can still reference variant params.
    // Conditions are AST `syn::Expr` nodes; we'll evaluate them per-variant
    // below using the same `eval_bool_expr` machinery as compile-time `if`.
    let optional_constexprs = collect_optional_constexprs(&input.sig)?;

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

    // Rewrite the function signature: substitutes type variants in parameter
    // type positions (e.g. `weight: Tensor<WT>` → `weight: Tensor<u32>`).
    // Integer variants embedded in param type idents are also handled here.
    let sig = &input.sig;
    let sig_tokens: TokenStream = quote::quote! { #sig };
    let substituted_sig = substitute_tokens(sig_tokens, &params);
    input.sig = syn::parse2::<syn::Signature>(substituted_sig)?;

    // Rewrite the function body: replace variant param idents with values.
    let block = &input.block;
    let block_tokens: TokenStream = quote::quote! { #block };
    let substituted = substitute_tokens(block_tokens, &params);
    input.block = Box::new(syn::parse2::<syn::Block>(substituted)?);

    // Strip optional constexpr params whose `only_when` condition fails for
    // this variant. The body is already FMT-pruned, so it cannot reference
    // any param we strip here (the compiler-time gating has removed the
    // branch that uses it).
    let int_params: HashMap<String, i64> =
        params.iter().filter_map(|(k, v)| v.as_int().map(|n| (k.clone(), n))).collect();
    strip_optional_constexprs(&mut input.sig, &optional_constexprs, &int_params)?;

    // Override the function name with the computed variant name.  The sig
    // substitution above may have modified it via ident-embedding; always
    // set it to the suffix-derived name for correctness.
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
fn substitute_tokens(stream: TokenStream, params: &HashMap<String, VariantValue>) -> TokenStream {
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

            // Mode 2 — exact param match → integer literal, float literal, or type tokens.
            if let Some(val) = params.get(&s) {
                match val {
                    VariantValue::Int(v) => {
                        let mut lit = Literal::i64_unsuffixed(*v);
                        lit.set_span(ident.span());
                        out.extend(std::iter::once(TokenTree::Literal(lit)));
                    },
                    VariantValue::Float(lit) => {
                        let mut lit = lit.clone();
                        lit.set_span(ident.span());
                        out.extend(std::iter::once(TokenTree::Literal(lit)));
                    },
                    VariantValue::Type(ts) => {
                        out.extend(ts.clone());
                    },
                }
                i += 1;
                continue;
            }

            // Mode 3 — ident-embedding: substitute integer param substrings into idents.
            // Type params are never embedded into identifier names.
            let mut new_s = s.clone();
            for (name, val) in params {
                if let VariantValue::Int(v) = val {
                    new_s = new_s.replace(name.as_str(), &v.to_string());
                }
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
    params: &HashMap<String, VariantValue>,
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
    // Only integer params are meaningful in boolean conditions; extract the
    // int subset to pass to eval_bool_expr (which operates on i64).
    let int_params: HashMap<String, i64> =
        params.iter().filter_map(|(k, v)| v.as_int().map(|n| (k.clone(), n))).collect();
    if condition_is_param_only(&cond_stream, params)
        && let Ok(expr) = syn::parse2::<syn::Expr>(cond_stream.clone())
        && let Some(taken) = eval_bool_expr(&expr, &int_params)
    {
        // Wrap the selected branch content back in braces so the result is a
        // block expression `{ … }` rather than a bare token sequence.
        //
        // Without braces, a `let elem = if CONST { let x = …; expr }` becomes
        // `let elem = let x = …; expr` after compile-time stripping — which
        // syn parses as `let elem = (let x = …);` (Expr::Let) followed by
        // `expr;` as a separate statement.  `Expr::Let` falls into the body
        // parser's catch-all and `elem` is left with an unproduced VID, so
        // `v_elem` is undeclared in the generated MSL.  The braces make this
        // `let elem = { let x = …; expr }` (Expr::Block), which the body
        // parser's Expr::Block arm handles correctly.
        //
        // Statement-level compile-time ifs (no `let` binding) also benefit:
        // `{ stmts… }` is a valid block statement and parse_expr_stmt already
        // iterates the inner stmts, so behaviour is unchanged for those.
        //
        // `ElseBranch::Processed` comes from a recursive `substitute_if` call
        // which already wraps its output, so we use it as-is to avoid
        // double-wrapping.
        let selected = match (taken, &else_branch) {
            (_, Some(ElseBranch::Processed(ts))) if !taken => ts.clone(),
            _ => {
                let inner = if taken {
                    substitute_tokens(then_group.stream(), params)
                } else {
                    match &else_branch {
                        Some(ElseBranch::Block(g)) => substitute_tokens(g.stream(), params),
                        None => TokenStream::new(),
                        Some(ElseBranch::Processed(_)) => unreachable!(),
                    }
                };
                let mut g = Group::new(Delimiter::Brace, inner);
                g.set_span(then_group.span());
                let mut ts = TokenStream::new();
                ts.extend(Some(TokenTree::Group(g)));
                ts
            },
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
fn condition_is_param_only(stream: &TokenStream, params: &HashMap<String, VariantValue>) -> bool {
    for tt in stream.clone().into_iter() {
        match tt {
            TokenTree::Ident(ident) => {
                let s = ident.to_string();
                if s == "true" || s == "false" {
                    continue;
                }
                // Only integer variant params are evaluable in boolean expressions.
                // Type params (e.g. `WT = u32`) cannot be compared arithmetically,
                // so their presence makes the condition a runtime expression.
                if !params.get(&s).is_some_and(|v| v.as_int().is_some()) {
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
pub(crate) fn eval_suffix(
    template: &str,
    params: &HashMap<String, VariantValue>,
) -> syn::Result<String> {
    // Pre-extract the integer subset for arithmetic evaluation.
    let int_params: HashMap<String, i64> =
        params.iter().filter_map(|(k, v)| v.as_int().map(|n| (k.clone(), n))).collect();

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

        // A bare identifier naming a non-integer param (type or float, e.g.
        // `{WT}` or `{SCALE}`) is stringified directly rather than parsed as
        // an arithmetic expression.
        let trimmed = expr_str.trim();
        if let Some(val @ (VariantValue::Float(_) | VariantValue::Type(_))) = params.get(trimmed) {
            result.push_str(&val.to_suffix_string());
        } else {
            let expr: syn::Expr = syn::parse_str(expr_str).map_err(|_| {
                syn::Error::new(
                    Span::call_site(),
                    format!("variants: failed to parse suffix expression `{expr_str}`"),
                )
            })?;
            let val = eval_expr(&expr, &int_params)?;
            result.push_str(&val.to_string());
        }
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
fn auto_suffix(params: &[(String, VariantValue)]) -> String {
    params
        .iter()
        .map(|(name, val)| format!("{}{}", name.to_lowercase(), val.to_suffix_string()))
        .collect::<Vec<_>>()
        .join("_")
}

/// Collect `#[optional(only_when = "EXPR")]` constexpr declarations from a
/// function signature. Two attribute styles are supported:
///
/// 1. Two separate attributes on the same param:
///    ```text
///    #[optional(only_when = "DILATED == 1u32")]
///    #[constexpr] dilation: u32,
///    ```
///
/// 2. A single combined `#[constexpr(only_when = "EXPR")]` attribute
///    (a constexpr that also carries its own gating condition):
///    ```text
///    #[constexpr(only_when = "DILATED == 1u32")] dilation: u32,
///    ```
///
/// In both cases, the returned entry means: this param is a constexpr
/// that should be stripped from the generated signature when the condition
/// is false. The `only_when = "EXPR"` RHS is a string literal whose
/// contents are parsed as a `syn::Expr`; the same arithmetic-and-comparison
/// subset that `eval_bool_expr` accepts for compile-time `if` is supported
/// (param idents, integer literals, `==`/`!=`/`<`/etc, `&&`/`||`/`!`).
pub(crate) fn collect_optional_constexprs(
    sig: &syn::Signature,
) -> syn::Result<Vec<OptionalConstexpr>> {
    let mut out = Vec::new();
    for input in &sig.inputs {
        let syn::FnArg::Typed(pat_type) = input else { continue };

        // Walk the param's attributes. A constexpr is "optional" if it
        // carries a `#[optional(...)]` attribute alongside, OR if the
        // constexpr attribute itself is in list form with `only_when = ...`.
        let mut opt_attr: Option<&syn::Attribute> = None;
        let mut has_plain_constexpr = false;
        let mut constexpr_attr: Option<&syn::Attribute> = None;
        for attr in &pat_type.attrs {
            if attr.path().is_ident("optional") {
                opt_attr = Some(attr);
            } else if attr.path().is_ident("constexpr") {
                // Distinguish bare `#[constexpr]` from list form
                // `#[constexpr(only_when = ...)]`.
                if let syn::Meta::List(_) = &attr.meta {
                    opt_attr = Some(attr);
                } else {
                    has_plain_constexpr = true;
                }
                constexpr_attr = Some(attr);
            }
        }

        // The param must be a constexpr in some form (plain `#[constexpr]`
        // or list-form `#[constexpr(only_when = ...)]`). A bare
        // `#[optional(...)]` with no `#[constexpr]` is silently ignored —
        // `#[optional]` only makes sense as a constexpr modifier.
        if !has_plain_constexpr && constexpr_attr.is_none() {
            continue;
        }
        let attr = match opt_attr {
            Some(a) => a,
            None => continue,
        };

        // Extract the param ident — only simple idents are supported.
        let syn::Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "variants: `#[optional]` is only valid on simple ident patterns",
            ));
        };
        let name = pat_ident.ident.to_string();
        let span = attr.span();

        // The attribute's Meta is `List { tokens, .. }` since both syntaxes
        // are list-form. Parse `only_when = "EXPR"` from the tokens.
        let syn::Meta::List(_list) = &attr.meta else {
            return Err(syn::Error::new(
                span,
                "variants: `#[optional]` requires list form `#[optional(only_when = \"EXPR\")]`",
            ));
        };
        let condition_expr = parse_optional_only_when(&_list.tokens).map_err(|mut e| {
            e.combine(syn::Error::new(
                span,
                "variants: `#[optional]` must be `#[optional(only_when = \"<expr>\")]`",
            ));
            e
        })?;
        out.push(OptionalConstexpr { name, condition: condition_expr, span });
    }
    Ok(out)
}

/// Parse the contents of `#[optional(only_when = "EXPR")]` — a single
/// `name = string_literal` token pair where the string literal is a
/// Rust-expression source that is then parsed into a `syn::Expr`.
fn parse_optional_only_when(tokens: &TokenStream) -> syn::Result<syn::Expr> {
    // Expect exactly: `only_when = "EXPR"`. Parse token-by-token to keep
    // the implementation minimal and the error messages span-accurate.
    let wrap = quote::quote! { #tokens };
    let mut iter = wrap.into_iter();

    let key = iter.next().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "variants: expected `only_when = \"EXPR\"`")
    })?;
    let key_ident = match &key {
        TokenTree::Ident(id) if id == "only_when" => id.clone(),
        _ =>
            return Err(syn::Error::new(
                key.span(),
                "variants: `#[optional]` must start with `only_when`",
            )),
    };
    let eq = iter
        .next()
        .ok_or_else(|| syn::Error::new(key_ident.span(), "expected `=` after `only_when`"))?;
    match &eq {
        TokenTree::Punct(p) if p.as_char() == '=' => {},
        _ => return Err(syn::Error::new(eq.span(), "variants: expected `=` after `only_when`")),
    }
    let lit_tt = iter
        .next()
        .ok_or_else(|| syn::Error::new(eq.span(), "expected string-literal expression"))?;
    let lit_tokens = match &lit_tt {
        TokenTree::Literal(_) => lit_tt.to_string(),
        _ =>
            return Err(syn::Error::new(
                lit_tt.span(),
                "variants: `only_when` RHS must be a string literal containing a Rust expression",
            )),
    };
    // Parse the literal text as a `syn::LitStr` (it has the form `LitStr`
    // with surrounding quotes — syn knows how to strip them).
    let lit_str: syn::LitStr = syn::parse_str(&lit_tokens).map_err(|e| {
        syn::Error::new(
            lit_tt.span(),
            format!("variants: `only_when` RHS must be a string literal: {e}"),
        )
    })?;
    syn::parse_str::<syn::Expr>(&lit_str.value()).map_err(|e| {
        syn::Error::new(
            lit_tt.span(),
            format!("variants: failed to parse `only_when` expression: {e}"),
        )
    })
}

/// Strip optional constexpr parameters from a function signature when their
/// `only_when` condition evaluates to `false` for the current variant. Returns
/// the same signature in `Ok(sig)` form. The condition is evaluated using the
/// same `eval_bool_expr` machinery that handles compile-time `if` — only
/// integer variant params and integer literals are accepted.
pub(crate) fn strip_optional_constexprs(
    sig: &mut syn::Signature,
    optional: &[OptionalConstexpr],
    int_params: &HashMap<String, i64>,
) -> syn::Result<()> {
    if optional.is_empty() {
        return Ok(());
    }
    // `syn::Punctuated` has no `retain`; rebuild via `into_iter` + collect.
    let kept: Punctuated<syn::FnArg, Token![,]> = sig
        .inputs
        .iter()
        .filter(|input| {
            let syn::FnArg::Typed(pat_type) = input else { return true };
            let syn::Pat::Ident(pat_ident) = &*pat_type.pat else { return true };
            let name = pat_ident.ident.to_string();
            let Some(opt) = optional.iter().find(|o| o.name == name) else {
                return true;
            };
            match eval_bool_expr(&opt.condition, int_params) {
                Some(true) => true,   // Keep the param: condition holds.
                Some(false) => false, // Strip the param: condition fails.
                None => {
                    // Not evaluable as a boolean — leave the param in place
                    // and let the body parser / type-checker report the
                    // issue later. Better than silently dropping.
                    true
                },
            }
        })
        .cloned()
        .collect();
    sig.inputs = kept;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use syn::parse_quote;

    use super::*;

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Build a `HashMap<String, VariantValue>` of integer params for tests.
    fn int_params(pairs: &[(&str, i64)]) -> HashMap<String, VariantValue> {
        pairs.iter().map(|(k, v)| (k.to_string(), VariantValue::Int(*v))).collect()
    }

    /// Build a `Vec<(String, VariantValue)>` of integer params for tests.
    fn int_slice(pairs: &[(&str, i64)]) -> Vec<(String, VariantValue)> {
        pairs.iter().map(|(k, v)| (k.to_string(), VariantValue::Int(*v))).collect()
    }

    /// Extract the integer values from a `Vec<VariantValue>` (panics on type variants).
    fn int_vals(vals: &[VariantValue]) -> Vec<i64> {
        vals.iter().map(|v| v.as_int().expect("expected Int variant")).collect()
    }

    // ── VariantsSpec parsing ──────────────────────────────────────────────────

    #[test]
    fn single_param_correct_variant_count_and_names() {
        let spec: VariantsSpec = syn::parse_str("M = [8, 16, 32], suffix = \"m{M}\"").unwrap();
        assert_eq!(spec.variant_count, 3);
        assert_eq!(spec.params.len(), 1);
        assert_eq!(spec.params[0].0, "M");
        assert_eq!(int_vals(&spec.params[0].1), vec![8, 16, 32]);
        assert_eq!(spec.suffix.as_deref(), Some("m{M}"));
    }

    #[test]
    fn multi_param_zipped_not_cartesian() {
        let spec: VariantsSpec =
            syn::parse_str("ELEMS = [2, 3, 4], PHASE_COUNT = [1, 1, 2], suffix = \"d{ELEMS}\"")
                .unwrap();
        assert_eq!(spec.variant_count, 3);
        assert_eq!(int_vals(&spec.params[0].1), vec![2, 3, 4]);
        assert_eq!(int_vals(&spec.params[1].1), vec![1, 1, 2]);
    }

    #[test]
    fn type_param_parsed() {
        let spec: VariantsSpec = syn::parse_str("WT = [u32, u8], suffix = \"wt{WT}\"").unwrap();
        assert_eq!(spec.variant_count, 2);
        assert!(matches!(spec.params[0].1[0], VariantValue::Type(_)));
        assert!(matches!(spec.params[0].1[1], VariantValue::Type(_)));
        assert_eq!(spec.suffix.as_deref(), Some("wt{WT}"));
    }

    #[test]
    fn float_param_parsed() {
        let spec: VariantsSpec =
            syn::parse_str("SCALE = [0.5f32, 1.0f32, 2.0f32], suffix = \"s{SCALE}\"").unwrap();
        assert_eq!(spec.variant_count, 3);
        assert!(matches!(spec.params[0].1[0], VariantValue::Float(_)));
        assert!(matches!(spec.params[0].1[1], VariantValue::Float(_)));
        assert_eq!(spec.suffix.as_deref(), Some("s{SCALE}"));
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
        assert_eq!(eval_suffix("prefix", &int_params(&[("M", 16)])).unwrap(), "prefix");
    }

    #[test]
    fn suffix_simple_param_substitution() {
        assert_eq!(eval_suffix("m{M}", &int_params(&[("M", 16)])).unwrap(), "m16");
    }

    #[test]
    fn suffix_arithmetic_mul() {
        assert_eq!(eval_suffix("d{ELEMS * 32}", &int_params(&[("ELEMS", 8)])).unwrap(), "d256");
    }

    #[test]
    fn suffix_multi_param() {
        assert_eq!(eval_suffix("{A}x{B}", &int_params(&[("A", 2), ("B", 4)])).unwrap(), "2x4");
    }

    #[test]
    fn suffix_paren_grouping() {
        assert_eq!(eval_suffix("s{(N + 1) * 8}", &int_params(&[("N", 3)])).unwrap(), "s32");
    }

    #[test]
    fn suffix_error_unknown_param() {
        let err = eval_suffix("{FOO}", &int_params(&[("M", 8)])).unwrap_err();
        assert!(err.to_string().contains("unknown param"), "{err}");
    }

    #[test]
    fn suffix_modulo_operator() {
        // `%` is now supported; 8 % 3 == 2.
        assert_eq!(eval_suffix("{M % 3}", &int_params(&[("M", 8)])).unwrap(), "2");
    }

    #[test]
    fn suffix_error_unsupported_operator() {
        // Bitwise shift is still unsupported.
        let err = eval_suffix("{M << 1}", &int_params(&[("M", 8)])).unwrap_err();
        assert!(err.to_string().contains("unsupported operator"), "{err}");
    }

    #[test]
    fn suffix_type_param_stringified() {
        let params: HashMap<String, VariantValue> =
            [("WT".to_string(), VariantValue::Type(quote::quote! { u32 }))].into_iter().collect();
        assert_eq!(eval_suffix("wt{WT}", &params).unwrap(), "wtu32");
    }

    #[test]
    fn suffix_float_param_stringified() {
        let lit: proc_macro2::Literal = syn::parse_str::<syn::LitFloat>("0.5f32").unwrap().token();
        let params: HashMap<String, VariantValue> =
            [("SCALE".to_string(), VariantValue::Float(lit))].into_iter().collect();
        assert_eq!(eval_suffix("s{SCALE}", &params).unwrap(), "s0_5");
    }

    #[test]
    fn suffix_float_no_fractional_part() {
        let lit: proc_macro2::Literal = syn::parse_str::<syn::LitFloat>("1.0f32").unwrap().token();
        let params: HashMap<String, VariantValue> =
            [("SCALE".to_string(), VariantValue::Float(lit))].into_iter().collect();
        assert_eq!(eval_suffix("s{SCALE}", &params).unwrap(), "s1_0");
    }

    // ── auto_suffix ───────────────────────────────────────────────────────────

    #[test]
    fn auto_suffix_single_param() {
        assert_eq!(auto_suffix(&int_slice(&[("M", 16)])), "m16");
    }

    #[test]
    fn auto_suffix_multi_param() {
        let s = auto_suffix(&int_slice(&[("ELEMS", 4), ("PHASE_COUNT", 2)]));
        assert_eq!(s, "elems4_phase_count2");
    }

    #[test]
    fn auto_suffix_type_param() {
        let params = vec![("WT".to_string(), VariantValue::Type(quote::quote! { u8 }))];
        assert_eq!(auto_suffix(&params), "wtu8");
    }

    #[test]
    fn auto_suffix_float_param() {
        let lit: proc_macro2::Literal = syn::parse_str::<syn::LitFloat>("0.5f32").unwrap().token();
        let params = vec![("SCALE".to_string(), VariantValue::Float(lit))];
        assert_eq!(auto_suffix(&params), "scale0_5");
    }

    // ── substitute_tokens ─────────────────────────────────────────────────────

    #[test]
    fn substitution_replaces_bare_ident() {
        let input: TokenStream = quote::quote! { range(0u32, M, 1u32) };
        let output = substitute_tokens(input, &int_params(&[("M", 8)])).to_string();
        // M → 8; u32-suffixed literals should remain unchanged.
        assert!(output.contains(" 8 "), "expected 8 in: {output}");
        assert!(!output.contains(" M "), "M should be gone: {output}");
    }

    #[test]
    fn substitution_replaces_type_ident() {
        let params: HashMap<String, VariantValue> =
            [("WT".to_string(), VariantValue::Type(quote::quote! { u32 }))].into_iter().collect();
        let input: TokenStream = quote::quote! { let w: Tensor<WT> = load(ptr); };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("u32"), "type not substituted: {output}");
        assert!(!output.contains("WT"), "WT should be gone: {output}");
    }

    #[test]
    fn substitution_replaces_float_literal() {
        let lit: proc_macro2::Literal = syn::parse_str::<syn::LitFloat>("0.5f32").unwrap().token();
        let params: HashMap<String, VariantValue> =
            [("SCALE".to_string(), VariantValue::Float(lit))].into_iter().collect();
        let input: TokenStream = quote::quote! { let x = a * SCALE; };
        let output = substitute_tokens(input, &params).to_string();
        assert!(output.contains("0.5f32"), "float not substituted: {output}");
        assert!(!output.contains("SCALE"), "SCALE should be gone: {output}");
    }

    #[test]
    fn float_param_not_embedded_in_ident() {
        // Float params must NOT do ident-embedding (unlike integer params).
        let lit: proc_macro2::Literal = syn::parse_str::<syn::LitFloat>("0.5f32").unwrap().token();
        let params: HashMap<String, VariantValue> =
            [("SCALE".to_string(), VariantValue::Float(lit))].into_iter().collect();
        // `kernel_SCALE` has SCALE as a substring; must NOT be rewritten.
        let input: TokenStream = quote::quote! { kernel_SCALE };
        let output = substitute_tokens(input, &params).to_string();
        // With no exact match (it's `kernel_SCALE` not bare `SCALE`), the ident
        // should pass through unchanged — float params can't embed into idents.
        assert!(output.contains("kernel_SCALE"), "ident should be unchanged: {output}");
    }

    #[test]
    fn substitution_embeds_in_ident() {
        // `dequant_gather_intBITS` is one Ident token; BITS is a substring.
        let input: TokenStream = quote::quote! { dequant_gather_intBITS::kernel_ir_for(dt) };
        let output = substitute_tokens(input, &int_params(&[("BITS", 4)])).to_string();
        assert!(output.contains("dequant_gather_int4"), "expected int4 in: {output}");
        assert!(!output.contains("BITS"), "BITS should be gone: {output}");
    }

    #[test]
    fn substitution_ident_embedding_multiple_params() {
        let input: TokenStream = quote::quote! { kernel_mMxN };
        let output = substitute_tokens(input, &int_params(&[("M", 8), ("N", 16)])).to_string();
        assert!(output.contains("kernel_m8x16"), "expected m8x16 in: {output}");
    }

    #[test]
    fn substitution_exact_still_emits_literal_not_ident() {
        // Standalone BITS should become a literal 3, not an ident "3".
        let input: TokenStream = quote::quote! { k * BITS / 32u32 };
        let output = substitute_tokens(input, &int_params(&[("BITS", 3)])).to_string();
        assert!(output.contains("k * 3 / 32u32"), "expected literal in: {output}");
    }

    #[test]
    fn substitution_does_not_touch_string_literals() {
        // "M" inside a string literal must not be replaced.
        let input: TokenStream = quote::quote! { stack_alloc("M_sized", M, "f32") };
        let output = substitute_tokens(input, &int_params(&[("M", 8)])).to_string();
        assert!(output.contains('"'), "string literal lost");
        assert!(output.contains("M_sized"), "string literal was modified");
    }

    #[test]
    fn substitution_recurses_into_groups() {
        let input: TokenStream = quote::quote! { (a + N) * b };
        let output = substitute_tokens(input, &int_params(&[("N", 4)])).to_string();
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
        let ts: TokenStream = quote::quote! { BITS % 2 == 0 };
        assert!(condition_is_param_only(&ts, &int_params(&[("BITS", 4)])));
    }

    #[test]
    fn param_only_false_for_lowercase_ident() {
        let ts: TokenStream = quote::quote! { n > 0 };
        assert!(!condition_is_param_only(&ts, &int_params(&[("BITS", 4)])));
    }

    #[test]
    fn param_only_allows_true_false_keywords() {
        let ts: TokenStream = quote::quote! { true };
        assert!(condition_is_param_only(&ts, &int_params(&[("BITS", 4)])));
    }

    #[test]
    fn param_only_false_for_type_param_in_condition() {
        // Type params cannot appear in boolean conditions — must be runtime.
        let params: HashMap<String, VariantValue> =
            [("WT".to_string(), VariantValue::Type(quote::quote! { u32 }))].into_iter().collect();
        let ts: TokenStream = quote::quote! { WT == u32 };
        assert!(!condition_is_param_only(&ts, &params));
    }

    // ── compile-time if substitution ─────────────────────────────────────────

    #[test]
    fn compile_time_if_selects_then_branch() {
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let x = 1u32; } else { let x = 2u32; }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 4)])).to_string();
        assert!(output.contains("1u32"), "then-branch missing: {output}");
        assert!(!output.contains("2u32"), "else-branch leaked: {output}");
        assert!(!output.contains("if"), "if keyword should be gone: {output}");
    }

    #[test]
    fn compile_time_if_selects_else_branch() {
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let x = 1u32; } else { let x = 2u32; }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 8)])).to_string();
        assert!(output.contains("2u32"), "else-branch missing: {output}");
        assert!(!output.contains("1u32"), "then-branch leaked: {output}");
    }

    #[test]
    fn compile_time_if_no_else_false_emits_nothing() {
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let x = 1u32; }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 8)])).to_string();
        assert!(!output.contains("1u32"), "branch should be gone: {output}");
        assert!(output.is_empty() || !output.contains("if"), "if leaked: {output}");
    }

    #[test]
    fn compile_time_if_modulo_parity() {
        // BITS=4 (even) → selects then-branch
        let input: TokenStream = quote::quote! {
            if BITS % 2 == 0 { pack_strided() } else { elem_strided() }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 4)])).to_string();
        assert!(output.contains("pack_strided"), "{output}");
        assert!(!output.contains("elem_strided"), "{output}");
    }

    #[test]
    fn compile_time_if_modulo_parity_odd() {
        // BITS=3 (odd) → selects else-branch
        let input: TokenStream = quote::quote! {
            if BITS % 2 == 0 { pack_strided() } else { elem_strided() }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 3)])).to_string();
        assert!(output.contains("elem_strided"), "{output}");
        assert!(!output.contains("pack_strided"), "{output}");
    }

    #[test]
    fn compile_time_if_or_condition() {
        let input: TokenStream = quote::quote! {
            if BITS == 4 || BITS == 8 { pack() } else { odd() }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 8)])).to_string();
        assert!(output.contains("pack"), "{output}");
        assert!(!output.contains("odd"), "{output}");
    }

    #[test]
    fn compile_time_if_substitutes_params_in_selected_branch() {
        // BITS inside the selected branch must also be substituted.
        let input: TokenStream = quote::quote! {
            if BITS == 4 { let n = BITS * 8u32; } else { let n = 0u32; }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 4)])).to_string();
        assert!(output.contains("4 * 8u32"), "param not substituted in branch: {output}");
    }

    #[test]
    fn runtime_if_with_non_param_ident_passes_through() {
        let input: TokenStream = quote::quote! {
            if n > 0 { let x = BITS; } else { let x = 0u32; }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 4)])).to_string();
        // `if` must still be present; `n` stays; BITS → 4
        assert!(output.contains("if"), "if should be present: {output}");
        assert!(output.contains("n >"), "n should be present: {output}");
        assert!(output.contains("4"), "BITS should be substituted: {output}");
    }

    #[test]
    fn else_if_chain_both_param_only() {
        // BITS=3: first branch false, second branch true → picks middle
        let input: TokenStream = quote::quote! {
            if BITS == 4 { path_a() } else if BITS == 3 { path_b() } else { path_c() }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 3)])).to_string();
        assert!(output.contains("path_b"), "{output}");
        assert!(!output.contains("path_a"), "{output}");
        assert!(!output.contains("path_c"), "{output}");
    }

    #[test]
    fn else_if_chain_falls_to_final_else() {
        // BITS=8: neither inner condition matches
        let input: TokenStream = quote::quote! {
            if BITS == 4 { path_a() } else if BITS == 3 { path_b() } else { path_c() }
        };
        let output = substitute_tokens(input, &int_params(&[("BITS", 8)])).to_string();
        assert!(output.contains("path_c"), "{output}");
        assert!(!output.contains("path_a"), "{output}");
        assert!(!output.contains("path_b"), "{output}");
    }

    #[test]
    fn nested_compile_time_if() {
        // Outer: BITS==4 → then; Inner: DIM==128 → then
        let input: TokenStream = quote::quote! {
            if BITS == 4 {
                if DIM == 128 { deep() } else { shallow() }
            } else {
                other()
            }
        };
        let output =
            substitute_tokens(input, &int_params(&[("BITS", 4), ("DIM", 128)])).to_string();
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
        let result =
            substitute_fn(input_fn, &int_slice(&[("M", 16)]), "mt_moe", &Some("m{M}".to_string()))
                .unwrap();
        assert_eq!(result.sig.ident.to_string(), "mt_moe_m16");
        let body = quote::quote! { #result }.to_string();
        assert!(body.contains("16 + 1u32"), "body substitution failed: {body}");
    }

    #[test]
    fn substitute_fn_auto_suffix() {
        let input_fn: ItemFn = parse_quote! { pub fn f() { let _ = M; } };
        let result = substitute_fn(input_fn, &int_slice(&[("M", 8)]), "f", &None).unwrap();
        assert_eq!(result.sig.ident.to_string(), "f_m8");
    }

    #[test]
    fn substitute_fn_type_param_in_signature() {
        let input_fn: ItemFn = parse_quote! {
            pub fn conv_w(weight: Tensor<WT>, x: f32) {
                let _w: WT = load(weight);
            }
        };
        let params = vec![("WT".to_string(), VariantValue::Type(quote::quote! { u32 }))];
        let result =
            substitute_fn(input_fn, &params, "conv_w", &Some("wt{WT}".to_string())).unwrap();
        assert_eq!(result.sig.ident.to_string(), "conv_w_wtu32");
        let sig = quote::quote! { #result }.to_string();
        // u32 must appear where WT was in both sig and body.
        assert!(sig.contains("u32"), "type param not substituted: {sig}");
        assert!(!sig.contains("WT"), "WT leaked: {sig}");
    }

    #[test]
    fn substitute_fn_float_param_in_body() {
        let lit: proc_macro2::Literal = syn::parse_str::<syn::LitFloat>("0.5f32").unwrap().token();
        let input_fn: ItemFn = parse_quote! {
            pub fn rescale(x: f32) -> f32 {
                x * SCALE
            }
        };
        let params = vec![("SCALE".to_string(), VariantValue::Float(lit))];
        let result =
            substitute_fn(input_fn, &params, "rescale", &Some("s{SCALE}".to_string())).unwrap();
        assert_eq!(result.sig.ident.to_string(), "rescale_s0_5");
        let body = quote::quote! { #result }.to_string();
        assert!(body.contains("0.5f32"), "float literal not substituted: {body}");
        assert!(!body.contains("SCALE"), "SCALE leaked: {body}");
    }

    // ── optional constexprs ───────────────────────────────────────────────────

    #[test]
    fn optional_constexpr_collected_from_signature() {
        let input: ItemFn = parse_quote! {
            pub fn mt<T>(
                input: Tensor<T>,
                #[constexpr] a: u32,
                #[constexpr(only_when = "DILATED == 1u32")] dilation: u32,
                #[constexpr] b: u32,
            ) {}
        };
        let opts = collect_optional_constexprs(&input.sig).unwrap();
        assert_eq!(opts.len(), 1);
        assert_eq!(opts[0].name, "dilation");
    }

    #[test]
    fn optional_constexpr_requires_constexpr_attr() {
        // `#[optional]` on a non-`#[constexpr]` param is silently ignored.
        let input: ItemFn = parse_quote! {
            pub fn mt(
                #[optional(only_when = "DILATED == 1u32")] dilation: u32,
            ) {}
        };
        let opts = collect_optional_constexprs(&input.sig).unwrap();
        assert_eq!(opts.len(), 0);
    }

    #[test]
    fn optional_constexpr_stripped_when_condition_false() {
        let input: ItemFn = parse_quote! {
            pub fn mt<T>(
                input: Tensor<T>,
                #[constexpr] batch: u32,
                #[constexpr(only_when = "DILATED == 1u32")] dilation: u32,
                #[constexpr] block_size: u32,
            ) {}
        };
        let opts = collect_optional_constexprs(&input.sig).unwrap();
        // DILATED=0: dilation should be stripped.
        let mut sig = input.sig.clone();
        let int_p: HashMap<String, i64> = int_params(&[("DILATED", 0)])
            .into_iter()
            .filter_map(|(k, v)| v.as_int().map(|n| (k, n)))
            .collect();
        strip_optional_constexprs(&mut sig, &opts, &int_p).unwrap();
        let s = quote::quote! { #sig }.to_string();
        assert!(!s.contains("dilation"), "dilation should be stripped at DILATED=0: {s}");
        assert!(s.contains("batch"));
        assert!(s.contains("block_size"));
    }

    #[test]
    fn optional_constexpr_kept_when_condition_true() {
        let input: ItemFn = parse_quote! {
            pub fn mt<T>(
                #[constexpr] batch: u32,
                #[constexpr(only_when = "DILATED == 1u32")] dilation: u32,
            ) {}
        };
        let opts = collect_optional_constexprs(&input.sig).unwrap();
        // DILATED=1: dilation should be kept.
        let mut sig = input.sig.clone();
        let int_p: HashMap<String, i64> = int_params(&[("DILATED", 1)])
            .into_iter()
            .filter_map(|(k, v)| v.as_int().map(|n| (k, n)))
            .collect();
        strip_optional_constexprs(&mut sig, &opts, &int_p).unwrap();
        let s = quote::quote! { #sig }.to_string();
        assert!(s.contains("dilation"), "dilation should be kept at DILATED=1: {s}");
    }

    #[test]
    fn optional_constexpr_end_to_end_in_substitute_fn() {
        // Full pipeline: define a function with an `#[optional]` constexpr,
        // substitute a DILATED=0 variant, and verify dilation is gone.
        let input: ItemFn = parse_quote! {
            pub fn mt<T>(
                #[constexpr] batch: u32,
                #[constexpr(only_when = "DILATED == 1u32")] dilation: u32,
            ) {
                let _ = batch;
                if DILATED == 0u32 {
                    let _ = batch + 1u32;
                } else {
                    let _ = dilation + 1u32;
                }
            }
        };
        let result = substitute_fn(
            input,
            &int_slice(&[("DILATED", 0)]),
            "mt",
            &Some("d{DILATED}".to_string()),
        )
        .unwrap();
        let s = quote::quote! { #result }.to_string();
        assert!(!s.contains("dilation"), "dilation leaked at DILATED=0: {s}");
        assert_eq!(result.sig.ident.to_string(), "mt_d0");
    }
}

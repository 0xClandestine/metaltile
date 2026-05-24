//! Expression evaluator for `$var` references and arithmetic.
//!
//! Handles:
//! - Literal integers and floats: `"128"`, `"1.0"`
//! - Variable substitution: `"$hidden_dim"` → param lookup
//! - Integer arithmetic: `"$n_heads * $head_dim"`, `"$n_heads / $n_kv_heads"`
//! - Float arithmetic: `"1.0 / sqrt($head_dim)"`
//! - Layer-index resolution: `"$layers.$idx.attn_norm"` → `"layers.3.attn_norm"`
//!
//! Uses a shunting-yard parser + stack evaluator — O(n), no recursion.

use std::collections::HashMap;

use crate::error::ModelError;

// ── Tokenizer ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Tok {
    Num(f64),
    Var(String),       // $name
    Path(Vec<String>), // $layers.$idx.attn_norm
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Sqrt,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, ModelError> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < c.len() {
        if c[i].is_whitespace() {
            i += 1;
            continue;
        }
        match c[i] {
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            },
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            },
            '*' => {
                out.push(Tok::Star);
                i += 1;
            },
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            },
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            },
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            },
            '$' => {
                i += 1;
                let mut parts = Vec::new();
                loop {
                    let start = i;
                    while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                        i += 1;
                    }
                    if i == start {
                        return Err(ModelError::InvalidExpr {
                            expr: s.into(),
                            detail: "expected identifier after $".into(),
                        });
                    }
                    parts.push(s[start..i].to_string());
                    if i < c.len() && c[i] == '.' {
                        i += 1;
                        if i < c.len() && c[i] == '$' {
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                out.push(if parts.len() == 1 {
                    Tok::Var(parts.into_iter().next().unwrap())
                } else {
                    Tok::Path(parts)
                });
            },
            '0'..='9' => {
                let start = i;
                while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.') {
                    i += 1;
                }
                let val = s[start..i].parse::<f64>().map_err(|e| ModelError::InvalidExpr {
                    expr: s.into(),
                    detail: e.to_string(),
                })?;
                out.push(Tok::Num(val));
            },
            ch if ch.is_alphabetic() || ch == '_' => {
                let start = i;
                while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                    i += 1;
                }
                let word = &s[start..i];
                if word == "sqrt" {
                    out.push(Tok::Sqrt);
                } else {
                    return Err(ModelError::InvalidExpr {
                        expr: s.into(),
                        detail: format!("unknown keyword '{word}'"),
                    });
                }
            },
            ch =>
                return Err(ModelError::InvalidExpr {
                    expr: s.into(),
                    detail: format!("unexpected character '{ch}'"),
                }),
        }
    }
    Ok(out)
}

// ── Shunting-yard → RPN ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    SqrtFn,
}

fn to_rpn(tokens: &[Tok]) -> Result<Vec<RpnToken>, ModelError> {
    let mut out = Vec::new();
    let mut ops: Vec<(Op, u8)> = Vec::new(); // (op, precedence)

    fn prec(op: &Tok) -> u8 {
        match op {
            Tok::Plus | Tok::Minus => 1,
            Tok::Star | Tok::Slash => 2,
            _ => 0,
        }
    }

    for tok in tokens {
        match tok {
            Tok::Num(_) | Tok::Var(_) | Tok::Path(_) => out.push(RpnToken::Value(tok.clone())),
            Tok::Sqrt => ops.push((Op::SqrtFn, 3)),
            Tok::Plus | Tok::Minus | Tok::Star | Tok::Slash => {
                let p = prec(tok);
                while let Some((_, tp)) = ops.last() {
                    if *tp >= p {
                        let (o, _) = ops.pop().unwrap();
                        out.push(RpnToken::Op(o));
                    } else {
                        break;
                    }
                }
                ops.push((
                    match tok {
                        Tok::Plus => Op::Add,
                        Tok::Minus => Op::Sub,
                        Tok::Star => Op::Mul,
                        Tok::Slash => Op::Div,
                        _ => unreachable!(),
                    },
                    p,
                ));
            },
            Tok::LParen => ops.push((Op::Neg, 0)), // reuse Neg as sentinel for '('
            Tok::RParen => {
                loop {
                    match ops.pop() {
                        Some((Op::Neg, 0)) => break, // matching '('
                        Some((o, _)) => out.push(RpnToken::Op(o)),
                        None =>
                            return Err(ModelError::InvalidExpr {
                                expr: "mismatched parentheses".into(),
                                detail: "extra ')'".into(),
                            }),
                    }
                }
                // Check for function left on stack before the '('
                if let Some((Op::SqrtFn, _)) = ops.last() {
                    out.push(RpnToken::Op(Op::SqrtFn));
                    ops.pop();
                }
            },
        }
    }
    while let Some((o, _)) = ops.pop() {
        if o == Op::Neg && ops.is_empty() {
            return Err(ModelError::InvalidExpr {
                expr: "mismatched parentheses".into(),
                detail: "extra '('".into(),
            });
        }
        out.push(RpnToken::Op(o));
    }
    Ok(out)
}

#[derive(Debug, Clone)]
enum RpnToken {
    Value(Tok),
    Op(Op),
}

// ── RPN evaluator ──────────────────────────────────────────────────────

fn eval_rpn(
    rpn: &[RpnToken],
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
    layer_idx: Option<usize>,
) -> Result<f64, ModelError> {
    let mut stack: Vec<f64> = Vec::new();
    for tok in rpn {
        match tok {
            RpnToken::Value(v) => {
                let f = match v {
                    Tok::Num(n) => *n,
                    Tok::Var(name) =>
                        if name == "idx" {
                            layer_idx.ok_or_else(|| ModelError::InvalidExpr {
                                expr: "$idx".into(),
                                detail: "no layer index available".into(),
                            })? as f64
                        } else if let Some(v) = float_params.get(name) {
                            *v
                        } else if let Some(v) = params.get(name) {
                            *v as f64
                        } else {
                            return Err(ModelError::UnknownParam { name: name.clone() });
                        },
                    Tok::Path(_) =>
                        return Err(ModelError::InvalidExpr {
                            expr: "dotted path".into(),
                            detail: "cannot evaluate path as number".into(),
                        }),
                    _ => unreachable!(),
                };
                stack.push(f);
            },
            RpnToken::Op(Op::Neg) => {
                let a = stack.pop().ok_or_else(|| err("negate"))?;
                stack.push(-a);
            },
            RpnToken::Op(Op::SqrtFn) => {
                let a = stack.pop().ok_or_else(|| err("sqrt"))?;
                stack.push(a.sqrt());
            },
            RpnToken::Op(Op::Add) => {
                let (b, a) =
                    (stack.pop().ok_or_else(|| err("+"))?, stack.pop().ok_or_else(|| err("+"))?);
                stack.push(a + b);
            },
            RpnToken::Op(Op::Sub) => {
                let (b, a) =
                    (stack.pop().ok_or_else(|| err("-"))?, stack.pop().ok_or_else(|| err("-"))?);
                stack.push(a - b);
            },
            RpnToken::Op(Op::Mul) => {
                let (b, a) =
                    (stack.pop().ok_or_else(|| err("*"))?, stack.pop().ok_or_else(|| err("*"))?);
                stack.push(a * b);
            },
            RpnToken::Op(Op::Div) => {
                let (b, a) =
                    (stack.pop().ok_or_else(|| err("/"))?, stack.pop().ok_or_else(|| err("/"))?);
                if b == 0.0 {
                    return Err(ModelError::InvalidExpr {
                        expr: "division by zero".into(),
                        detail: "".into(),
                    });
                }
                stack.push(a / b);
            },
        }
    }
    stack.pop().ok_or_else(|| ModelError::InvalidExpr {
        expr: "empty expression".into(),
        detail: "".into(),
    })
}

fn err(op: &str) -> ModelError {
    ModelError::InvalidExpr { expr: op.into(), detail: "stack underflow".into() }
}

// ── Public API ─────────────────────────────────────────────────────────

fn parse_and_eval(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
    layer_idx: Option<usize>,
) -> Result<f64, ModelError> {
    let tokens = tokenize(expr)?;
    let rpn = to_rpn(&tokens)?;
    eval_rpn(&rpn, params, float_params, layer_idx)
}

/// Evaluate a constexpr that may reference runtime state variables.
/// Returns `Ok(Some(value))` if statically resolvable, `Ok(None)` if it
/// references unknown variables (likely runtime state).
pub fn eval_constexpr_fallible(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> Result<Option<u32>, ModelError> {
    match parse_and_eval(expr, params, float_params, None) {
        Ok(val) => {
            if val < 0.0 || val > u32::MAX as f64 {
                return Err(ModelError::InvalidConstExpr {
                    expr: expr.into(),
                    detail: format!("value {val} out of u32 range"),
                });
            }
            let rounded = val.round();
            if (rounded - val).abs() > 0.001 {
                return Err(ModelError::InvalidConstExpr {
                    expr: expr.into(),
                    detail: format!("expected integer, got {val}"),
                });
            }
            Ok(Some(rounded as u32))
        },
        Err(ModelError::UnknownParam { .. }) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Evaluate as u32, requiring all variables resolvable at compile time.
pub fn eval_constexpr(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> Result<u32, ModelError> {
    eval_constexpr_fallible(expr, params, float_params)?.ok_or_else(|| {
        ModelError::InvalidConstExpr {
            expr: expr.into(),
            detail: "unresolved runtime variable".into(),
        }
    })
}

/// Evaluate as f32.
pub fn eval_float_expr(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> Result<f32, ModelError> {
    Ok(parse_and_eval(expr, params, float_params, None)? as f32)
}

/// Resolve a dotted tensor reference like `"$layers.$idx.attn_norm"` → `"layers.3.attn_norm"`.
pub fn resolve_tensor_ref(expr: &str, layer_idx: usize, _params: &HashMap<String, u32>) -> String {
    if expr.starts_with('_') {
        return expr.to_string();
    }
    let tokens = match tokenize(expr) {
        Ok(t) => t,
        Err(_) => return expr.to_string(),
    };
    let parts = match &tokens[..] {
        [Tok::Path(p)] => p,
        [Tok::Var(name)] if name.starts_with('_') => return name.clone(),
        [Tok::Var(name)] => return name.clone(),
        _ => return expr.to_string(),
    };
    parts
        .iter()
        .map(|p| if p == "idx" { format!("{layer_idx}") } else { p.clone() })
        .collect::<Vec<_>>()
        .join(".")
}

/// Resolve a dotted reference from pre-split path parts.
pub fn resolve_dotted_ref(parts: &[String], layer_idx: Option<usize>) -> String {
    parts
        .iter()
        .map(|p| {
            if p == "idx" { layer_idx.map_or("$idx".into(), |i| i.to_string()) } else { p.clone() }
        })
        .collect::<Vec<_>>()
        .join(".")
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> HashMap<String, u32> {
        let mut m = HashMap::new();
        m.insert("n_heads".into(), 32);
        m.insert("head_dim".into(), 128);
        m.insert("hidden_dim".into(), 4096);
        m.insert("n_kv_heads".into(), 8);
        m
    }

    fn fp() -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("scale".into(), 1.0 / 128.0_f64.sqrt());
        m
    }

    #[test]
    fn lit_int() {
        assert_eq!(eval_constexpr("128", &p(), &HashMap::new()).unwrap(), 128);
    }
    #[test]
    fn lit_float() {
        assert!((eval_float_expr("1.5", &p(), &HashMap::new()).unwrap() - 1.5).abs() < 1e-6);
    }
    #[test]
    fn var() {
        assert_eq!(eval_constexpr("$hidden_dim", &p(), &HashMap::new()).unwrap(), 4096);
    }
    #[test]
    fn mul() {
        assert_eq!(
            eval_constexpr("$n_heads * $head_dim", &p(), &HashMap::new()).unwrap(),
            32 * 128
        );
    }
    #[test]
    fn div() {
        assert_eq!(eval_constexpr("$n_heads / $n_kv_heads", &p(), &HashMap::new()).unwrap(), 4);
    }
    #[test]
    fn parens() {
        assert_eq!(
            eval_constexpr("($n_heads + 1) * 2", &p(), &HashMap::new()).unwrap(),
            (32 + 1) * 2
        );
    }
    #[test]
    fn sqrt_float() {
        let v = eval_float_expr("1.0 / sqrt($head_dim)", &p(), &fp()).unwrap();
        assert!((v - 1.0 / 128.0_f32.sqrt()).abs() < 1e-6);
    }
    #[test]
    fn unknown_var() {
        assert!(eval_constexpr("$nonexistent", &p(), &HashMap::new()).is_err());
    }
    #[test]
    fn tensor_ref() {
        assert_eq!(
            resolve_tensor_ref("$layers.$idx.attn_norm", 3, &HashMap::new()),
            "layers.3.attn_norm"
        );
    }
    #[test]
    fn tensor_ref_intermediate() {
        assert_eq!(resolve_tensor_ref("_normed", 0, &HashMap::new()), "_normed");
    }
    #[test]
    fn tensor_ref_simple_var() {
        assert_eq!(resolve_tensor_ref("$residual", 0, &HashMap::new()), "residual");
    }
    #[test]
    fn fallible_unknown() {
        assert_eq!(eval_constexpr_fallible("$runtime_var", &p(), &HashMap::new()).unwrap(), None);
    }
}

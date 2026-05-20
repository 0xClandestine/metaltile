//! Minimal expression evaluator for `$var` references and arithmetic.
//!
//! Handles:
//! - Literal integers and floats: `"128"`, `"1.0"`
//! - Variable substitution: `"$hidden_dim"` → look up in params
//! - Integer arithmetic: `"$n_heads * $head_dim"`, `"$n_heads / $n_kv_heads"`
//! - Float arithmetic: `"1.0 / sqrt($head_dim)"`
//! - Layer-index resolution: `"$layers.$idx.attn_norm"` → `"layers.3.attn_norm"`
//!
//! Grammar (recursive descent, no external parser crate):
//!
//! ```text
//! expr    → term (('+' | '-') term)*
//! term    → factor (('*' | '/') factor)*
//! factor  → '-' factor | '(' expr ')' | 'sqrt' '(' expr ')' | atom
//! atom    → integer | float | '$' ident ('.' ident)*
//! ```
//!
//! All arithmetic is 64-bit (f64 for float, u64 for int, cast at boundary).
//! This avoids the need for a type system in the TOML compiler.

use std::collections::HashMap;

use crate::error::ModelError;

// ── Tokenizer ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Int(u64),
    Float(f64),
    Dollar,
    Ident(String),
    Dot,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Sqrt,
}

fn tokenize(expr: &str) -> Result<Vec<Token>, ModelError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '$' => {
                tokens.push(Token::Dollar);
                i += 1;
            },
            '.' => {
                tokens.push(Token::Dot);
                i += 1;
            },
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            },
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            },
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            },
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            },
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            },
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            },
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                // Check for decimal point and fractional digits.
                let is_float = i < chars.len() && chars[i] == '.';
                if is_float {
                    i += 1; // skip '.'
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    let val: f64 =
                        expr[start..i].parse::<f64>().map_err(|e: std::num::ParseFloatError| {
                            ModelError::InvalidExpr {
                                expr: expr.to_string(),
                                detail: e.to_string(),
                            }
                        })?;
                    tokens.push(Token::Float(val));
                } else {
                    let val: u64 =
                        expr[start..i].parse::<u64>().map_err(|e: std::num::ParseIntError| {
                            ModelError::InvalidExpr {
                                expr: expr.to_string(),
                                detail: e.to_string(),
                            }
                        })?;
                    tokens.push(Token::Int(val));
                }
            },
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word = &expr[start..i];
                if word == "sqrt" {
                    tokens.push(Token::Sqrt);
                } else {
                    tokens.push(Token::Ident(word.to_string()));
                }
            },
            _ => {
                return Err(ModelError::InvalidExpr {
                    expr: expr.to_string(),
                    detail: format!("unexpected character: {c}"),
                });
            },
        }
    }
    Ok(tokens)
}

// ── Recursive descent parser ───────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self { Self { tokens, pos: 0 } }

    fn peek(&self) -> Option<&Token> { self.tokens.get(self.pos) }

    fn advance(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }

    fn expect(&mut self, expected: Token) -> Result<(), ModelError> {
        let Some(tok) = self.advance() else {
            return Err(ModelError::InvalidExpr {
                expr: "unexpected end of expression".into(),
                detail: "expected token".into(),
            });
        };
        if *tok != expected {
            return Err(ModelError::InvalidExpr {
                expr: format!("{:?}", tok),
                detail: format!("expected {:?}", expected),
            });
        }
        Ok(())
    }

    /// Evaluate to an f64, using params to resolve `$var` references.
    ///
    /// For `eval_constexpr` (integer results), the caller rounds/casts.
    fn eval_expr(
        &mut self,
        params: &HashMap<String, u32>,
        float_params: &HashMap<String, f64>,
        layer_idx: Option<usize>,
    ) -> Result<f64, ModelError> {
        self.eval_add_sub(params, float_params, layer_idx)
    }

    fn eval_add_sub(
        &mut self,
        params: &HashMap<String, u32>,
        float_params: &HashMap<String, f64>,
        layer_idx: Option<usize>,
    ) -> Result<f64, ModelError> {
        let mut left = self.eval_mul_div(params, float_params, layer_idx)?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    let right = self.eval_mul_div(params, float_params, layer_idx)?;
                    left += right;
                },
                Some(Token::Minus) => {
                    self.advance();
                    let right = self.eval_mul_div(params, float_params, layer_idx)?;
                    left -= right;
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn eval_mul_div(
        &mut self,
        params: &HashMap<String, u32>,
        float_params: &HashMap<String, f64>,
        layer_idx: Option<usize>,
    ) -> Result<f64, ModelError> {
        let mut left = self.eval_factor(params, float_params, layer_idx)?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    let right = self.eval_factor(params, float_params, layer_idx)?;
                    left *= right;
                },
                Some(Token::Slash) => {
                    self.advance();
                    let right = self.eval_factor(params, float_params, layer_idx)?;
                    if right == 0.0 {
                        return Err(ModelError::InvalidExpr {
                            expr: "division by zero".into(),
                            detail: "divisor is zero".into(),
                        });
                    }
                    left /= right;
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn eval_factor(
        &mut self,
        params: &HashMap<String, u32>,
        float_params: &HashMap<String, f64>,
        layer_idx: Option<usize>,
    ) -> Result<f64, ModelError> {
        match self.peek() {
            Some(Token::Minus) => {
                self.advance();
                Ok(-self.eval_factor(params, float_params, layer_idx)?)
            },
            Some(Token::LParen) => {
                self.advance();
                let val = self.eval_add_sub(params, float_params, layer_idx)?;
                self.expect(Token::RParen)?;
                Ok(val)
            },
            Some(Token::Sqrt) => {
                self.advance();
                self.expect(Token::LParen)?;
                let val = self.eval_add_sub(params, float_params, layer_idx)?;
                self.expect(Token::RParen)?;
                Ok(val.sqrt())
            },
            Some(Token::Int(n)) => {
                let val = *n as f64;
                self.advance();
                Ok(val)
            },
            Some(Token::Float(n)) => {
                let val = *n;
                self.advance();
                Ok(val)
            },
            Some(Token::Dollar) => {
                self.advance();
                // $ident(.ident)* — variable or dotted path
                let mut parts = Vec::new();
                loop {
                    let Some(Token::Ident(name)) = self.advance().cloned() else {
                        return Err(ModelError::InvalidExpr {
                            expr: "expected identifier after $".into(),
                            detail: "dangling $".into(),
                        });
                    };
                    parts.push(name);
                    if matches!(self.peek(), Some(Token::Dot)) {
                        self.advance(); // skip '.'
                    } else {
                        break;
                    }
                }
                if parts.len() == 1 {
                    // Simple variable: look up in params or float_params.
                    let name = &parts[0];
                    // Special: $idx is the layer index.
                    if name == "idx" {
                        let idx_val = layer_idx.ok_or_else(|| ModelError::InvalidExpr {
                            expr: "$idx used outside of layer context".into(),
                            detail: "no layer index available".into(),
                        })? as f64;
                        return Ok(idx_val);
                    }
                    // Try float_params first, then int params.
                    if let Some(v) = float_params.get(name) {
                        return Ok(*v);
                    }
                    if let Some(v) = params.get(name) {
                        return Ok(*v as f64);
                    }
                    Err(ModelError::UnknownParam { name: name.clone() })
                } else {
                    // Dotted path like $layers.$idx.attn_norm.
                    // Resolve $idx in the path and return as a name reference
                    // (not a numeric value — this is for tensor wiring, not constexpr).
                    // The caller handles this differently.
                    let _resolved = resolve_dotted_ref(&parts, layer_idx);
                    // For constexpr evaluation, we can't return a name.
                    // This is a tensor reference — use a sentinel NaN to signal
                    // to the caller that this is a name, not a number.
                    // Actually, we should handle this at a higher level.
                    // For now, return a special error that the caller can catch.
                    Err(ModelError::InvalidExpr {
                        expr: format!("${}", parts.join(".")),
                        detail: "dotted path reference is not a numeric value; use in tensor wiring context only".into(),
                    })
                }
            },
            _ => Err(ModelError::InvalidExpr {
                expr: "unexpected token".into(),
                detail: format!("{:?}", self.peek()),
            }),
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Evaluate a constexpr that may reference runtime state variables.
///
/// Returns `Ok(Some(value))` if the expression resolves statically.
/// Returns `Ok(None)` if the expression references unknown variables
/// (likely runtime state). Returns `Err` on syntax errors or arithmetic
/// errors.
pub fn eval_constexpr_fallible(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> Result<Option<u32>, ModelError> {
    let tokens = tokenize(expr)?;
    let mut parser = Parser::new(tokens);
    match parser.eval_expr(params, float_params, None) {
        Ok(val) => {
            if val < 0.0 || val > u32::MAX as f64 {
                return Err(ModelError::InvalidConstExpr {
                    expr: expr.to_string(),
                    detail: format!("value {val} out of u32 range"),
                });
            }
            let rounded = val.round();
            if (rounded - val).abs() > 0.001 {
                return Err(ModelError::InvalidConstExpr {
                    expr: expr.to_string(),
                    detail: format!("expected integer, got {val}"),
                });
            }
            Ok(Some(rounded as u32))
        },
        Err(ModelError::UnknownParam { .. }) => {
            // Unknown param → likely runtime state, defer to dispatch time.
            Ok(None)
        },
        Err(e) => Err(e),
    }
}

/// Evaluate an expression that should produce a `u32` value.
///
/// Used for constexpr fields and shape dimensions. Unlike
/// `eval_constexpr_fallible`, this requires all variables to be
/// resolvable at compile time.
pub fn eval_constexpr(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> Result<u32, ModelError> {
    eval_constexpr_fallible(expr, params, float_params)?.ok_or_else(|| {
        ModelError::InvalidConstExpr {
            expr: expr.to_string(),
            detail: "unresolved runtime variable".into(),
        }
    })
}

/// Evaluate an expression that should produce an `f32` value.
pub fn eval_float_expr(
    expr: &str,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> Result<f32, ModelError> {
    let tokens = tokenize(expr)?;
    let mut parser = Parser::new(tokens);
    let val = parser.eval_expr(params, float_params, None)?;
    Ok(val as f32)
}

/// Resolve a dotted reference path like `"$layers.$idx.attn_norm"` →
/// `"layers.3.attn_norm"` (substituting `$idx` = layer index and other
/// `$var` references).
///
/// Returns `None` if the expression isn't a dotted `$path`, meaning
/// it should be treated as a simple var reference or intermediate name.
pub fn resolve_tensor_ref(expr: &str, layer_idx: usize, _params: &HashMap<String, u32>) -> String {
    let tokens = match tokenize(expr) {
        Ok(t) => t,
        Err(_) => return expr.to_string(),
    };

    // Must start with $.
    if !matches!(tokens.first(), Some(Token::Dollar)) {
        return expr.to_string();
    }

    let mut parts = Vec::new();
    let mut i = 1; // skip initial $
    while i < tokens.len() {
        // Handle $var within the path (e.g., $layers.$idx.attn_norm).
        if tokens[i] == Token::Dollar {
            i += 1;
            if i >= tokens.len() {
                break;
            }
        }
        if let Token::Ident(ref name) = tokens[i] {
            if name == "idx" {
                parts.push(format!("{layer_idx}"));
            } else {
                parts.push(name.clone());
            }
            i += 1;
            // Skip dots.
            if i < tokens.len() && matches!(tokens[i], Token::Dot) {
                i += 1;
            }
        } else {
            // Non-ident after $ — just return original.
            return expr.to_string();
        }
    }

    // For intermediate names (starting with _), return as-is.
    if parts.len() == 1 && parts[0].starts_with('_') {
        return parts[0].clone();
    }

    parts.join(".")
}

/// Resolve a dotted reference with the path parts already split.
fn resolve_dotted_ref(parts: &[String], layer_idx: Option<usize>) -> String {
    let resolved: Vec<String> = parts
        .iter()
        .map(|part| {
            if part == "idx" {
                layer_idx.map_or("$idx".to_string(), |idx| idx.to_string())
            } else {
                part.clone()
            }
        })
        .collect();
    resolved.join(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> HashMap<String, u32> {
        let mut p = HashMap::new();
        p.insert("n_heads".into(), 32);
        p.insert("head_dim".into(), 128);
        p.insert("hidden_dim".into(), 4096);
        p.insert("n_kv_heads".into(), 8);
        p
    }

    fn test_float_params() -> HashMap<String, f64> {
        let mut p = HashMap::new();
        p.insert("scale".into(), 1.0 / 128.0_f64.sqrt());
        p
    }

    #[test]
    fn eval_literal_integer() {
        let params = test_params();
        let fp = HashMap::new();
        assert_eq!(eval_constexpr("128", &params, &fp).unwrap(), 128);
    }

    #[test]
    fn eval_simple_var() {
        let params = test_params();
        let fp = HashMap::new();
        assert_eq!(eval_constexpr("$hidden_dim", &params, &fp).unwrap(), 4096);
    }

    #[test]
    fn eval_integer_multiplication() {
        let params = test_params();
        let fp = HashMap::new();
        assert_eq!(eval_constexpr("$n_heads * $head_dim", &params, &fp).unwrap(), 32 * 128);
    }

    #[test]
    fn eval_integer_division() {
        let params = test_params();
        let fp = HashMap::new();
        assert_eq!(eval_constexpr("$n_heads / $n_kv_heads", &params, &fp).unwrap(), 32 / 8);
    }

    #[test]
    fn eval_float_arithmetic() {
        let params = test_params();
        let fp = test_float_params();
        let val = eval_float_expr("1.0 / sqrt($head_dim)", &params, &fp).unwrap();
        let expected = 1.0_f32 / 128.0_f32.sqrt();
        assert!((val - expected).abs() < 1e-6);
    }

    #[test]
    fn eval_parens() {
        let params = test_params();
        let fp = HashMap::new();
        assert_eq!(eval_constexpr("($n_heads + 1) * 2", &params, &fp).unwrap(), (32 + 1) * 2);
    }

    #[test]
    fn eval_unknown_var_is_error() {
        let params = test_params();
        let fp = HashMap::new();
        assert!(eval_constexpr("$nonexistent", &params, &fp).is_err());
    }

    #[test]
    fn resolve_tensor_ref_with_idx() {
        let params = HashMap::new();
        let result = resolve_tensor_ref("$layers.$idx.attn_norm", 3, &params);
        assert_eq!(result, "layers.3.attn_norm");
    }

    #[test]
    fn resolve_tensor_ref_intermediate() {
        let params = HashMap::new();
        let result = resolve_tensor_ref("_normed", 0, &params);
        assert_eq!(result, "_normed");
    }

    #[test]
    fn resolve_tensor_ref_simple_var() {
        let params = HashMap::new();
        let result = resolve_tensor_ref("$residual", 0, &params);
        assert_eq!(result, "residual");
    }
}

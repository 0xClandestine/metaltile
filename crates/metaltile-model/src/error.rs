//! Error types for metaltile-model.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("unknown kernel op: {op}")]
    UnknownOp { op: String },

    #[error("unknown parameter: ${name}")]
    UnknownParam { name: String },

    #[error("invalid constexpr expression '{expr}': {detail}")]
    InvalidConstExpr { expr: String, detail: String },

    #[error("invalid expression '{expr}': {detail}")]
    InvalidExpr { expr: String, detail: String },

    #[error("tensor reference not found: {name}")]
    TensorNotFound { name: String },

    #[error("shape mismatch for tensor '{name}': expected {expected}, got {actual}")]
    ShapeMismatch { name: String, expected: String, actual: String },

    #[error("missing required field: {field}")]
    MissingField { field: String },

    /// Dispatch grid would exceed Metal hardware limits or violate kernel
    /// invariants, potentially causing GPU hangs or out-of-bounds writes.
    #[error("unsafe dispatch for '{op}': {detail}")]
    UnsafeDispatch { op: String, detail: String },

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

impl From<String> for ModelError {
    fn from(s: String) -> Self { ModelError::Other(s) }
}

impl From<&str> for ModelError {
    fn from(s: &str) -> Self { ModelError::Other(s.to_string()) }
}

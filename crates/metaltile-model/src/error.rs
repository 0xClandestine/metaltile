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

    #[error("unsafe dispatch for '{op}': {detail}")]
    UnsafeDispatch { op: String, detail: String },

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("non-contiguous fuse group '{tag}' declared at node {first_instance} and reused at node {second_start}; fuse groups must be contiguous")]
    NonContiguousFuseGroup { tag: String, first_instance: usize, second_start: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("runtime error: {0}")]
    Runtime(#[from] metaltile_runtime::MetalTileError),

    #[error("{0}")]
    Other(String),
}
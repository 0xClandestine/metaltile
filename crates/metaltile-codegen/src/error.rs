//! Codegen errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported operation in MSL codegen: {0}")]
    UnsupportedOp(String),

    #[error("MSL generation error: {0}")]
    Generation(String),

    #[error("core error: {0}")]
    Core(#[from] metaltile_core::error::Error),

    #[error("block {0} not found in kernel IR")]
    BlockNotFound(u32),

    #[error("op not found in block: {0}")]
    OpNotFound(String),

    #[error("pass '{pass}' failed: {reason}")]
    PassFailed { pass: &'static str, reason: String },

    #[error("type inference failed: {0}")]
    TypeInference(String),
}

pub type Result<T> = std::result::Result<T, Error>;

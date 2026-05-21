//! Runtime errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalTileError {
    #[error("metal error: {0}")]
    Metal(String),

    #[error("no Metal device found")]
    NoDevice,

    #[error("kernel compilation failed: {0}")]
    Compilation(String),

    #[error("buffer allocation failed: {0}")]
    Buffer(String),

    #[error("dispatch failed: {0}")]
    Dispatch(String),

    #[error("autotune error: {0}")]
    Autotune(String),

    #[error("core error: {0}")]
    Core(#[from] metaltile_core::error::Error),

    #[error("codegen error: {0}")]
    Codegen(#[from] metaltile_codegen::error::Error),

    #[error("not implemented on this platform")]
    UnsupportedPlatform,

    #[error("metal device creation failed")]
    DeviceCreation,

    #[error("metal command queue creation failed")]
    QueueCreation,

    #[error("MSL library compilation failed: {0}")]
    MslCompilation(String),

    #[error("kernel function '{name}' not found in compiled library")]
    FunctionNotFound { name: String },

    #[error("compute pipeline creation failed for '{name}': {reason}")]
    PipelineCreation { name: String, reason: String },

    #[error("buffer allocation failed for '{param}': {reason}")]
    BufferAllocation { param: String, reason: String },

    #[error("buffer size mismatch for '{param}': expected {expected} bytes, got {actual}")]
    BufferSize { param: String, expected: usize, actual: usize },

    #[error("mutex poisoned: {0}")]
    LockPoisoned(String),

    #[error("GPU frame capture not supported: {0}")]
    CaptureNotSupported(String),

    #[error("GPU frame capture failed: {0}")]
    CaptureFailed(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

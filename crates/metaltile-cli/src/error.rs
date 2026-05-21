use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("metal compile failed: {0}")]
    MetalCompile(String),

    #[error("GPU runner initialization failed: {0}")]
    GpuInit(String),

    #[error("subprocess failed: {0}")]
    Subprocess(String),

    #[error("{0}")]
    Other(String),
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("model error: {0}")]
    Model(#[from] metaltile_model::ModelError),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("runtime error: {0}")]
    Runtime(#[from] metaltile_runtime::MetalTileError),

    #[error("missing config field: {0}")]
    MissingField(&'static str),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("HuggingFace hub error: {0}")]
    Hub(String),

    #[error("{0}")]
    Other(String),
}
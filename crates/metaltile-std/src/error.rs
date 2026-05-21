use thiserror::Error;

#[derive(Debug, Error)]
pub enum StdError {
    #[error("runner error: {0}")]
    Runner(String),
    #[error("Metal error: {0}")]
    Metal(String),
}

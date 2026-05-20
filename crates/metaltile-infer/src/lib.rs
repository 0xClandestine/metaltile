//! MetalTile inference engine.
//!
//! Provides checkpoint loading, KV cache management, and a token-by-token
//! inference loop for TOML-defined models.
//!
//! ## Quick start
//!
//! ```no_run
//! use metaltile_infer::{Session, ModelConfig};
//! use metaltile_core::dtype::DType;
//!
//! # async fn run() -> Result<(), metaltile_infer::InferError> {
//! let config = ModelConfig::from_file("/path/to/model/config.json")?;
//! let toml_src = std::fs::read_to_string("models/llama_decode.toml")?;
//! let mut session = Session::new("/path/to/model", &toml_src, config, DType::F16)?;
//! let out = session.generate("Hello, world!", 100, 0.8, |tok| print!("{tok}"))?;
//! # Ok(())
//! # }
//! ```

pub mod checkpoint;
pub mod config;
pub mod error;
pub mod hub;
pub mod session;

pub use checkpoint::{
    load_safetensors, load_safetensors_dir, load_weights, remap_hf_llama_names,
};
pub use config::ModelConfig;
pub use error::InferError;
pub use session::Session;

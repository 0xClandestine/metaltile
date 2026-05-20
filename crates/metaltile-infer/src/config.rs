//! Parse `config.json` from a HuggingFace Llama model repo into a `ModelConfig`.

use serde::Deserialize;
use std::path::Path;

use crate::error::InferError;

/// Architecture hyperparameters extracted from HuggingFace `config.json`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

/// Raw serde target for HuggingFace config.json — only the fields we need.
#[derive(Deserialize)]
struct HfConfig {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
}

impl ModelConfig {
    /// Load from a HuggingFace `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, InferError> {
        let bytes = std::fs::read(path)?;
        let raw: HfConfig = serde_json::from_slice(&bytes)?;
        Self::from_hf(&raw)
    }

    pub fn from_json_str(s: &str) -> Result<Self, InferError> {
        let raw: HfConfig = serde_json::from_str(s)?;
        Self::from_hf(&raw)
    }

    fn from_hf(raw: &HfConfig) -> Result<Self, InferError> {
        let n_kv_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        let max_seq_len = raw.max_position_embeddings.unwrap_or(4096);

        Ok(ModelConfig {
            n_layers: raw.num_hidden_layers,
            n_heads: raw.num_attention_heads,
            n_kv_heads,
            head_dim,
            hidden_dim: raw.hidden_size,
            ffn_dim: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            max_seq_len,
        })
    }
}

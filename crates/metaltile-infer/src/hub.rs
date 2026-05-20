//! HuggingFace Hub helpers — download model files to local cache.

use std::path::PathBuf;

use hf_hub::{Repo, RepoType, api::tokio::Api};

use crate::error::InferError;

/// Download a model repo to the local HF cache and return the local directory.
///
/// Files downloaded: all `.safetensors` shards + `tokenizer.json` +
/// `config.json`. Uses the HF_HOME cache (default `~/.cache/huggingface`).
pub async fn snapshot_download(repo_id: &str, revision: &str) -> Result<PathBuf, InferError> {
    let api = Api::new().map_err(|e| InferError::Hub(e.to_string()))?;
    let repo =
        api.repo(Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string()));

    // Fetch the file listing from the Hub
    let info = repo.info().await.map_err(|e| InferError::Hub(e.to_string()))?;

    let mut local_dir: Option<PathBuf> = None;
    for sibling in &info.siblings {
        let filename = &sibling.rfilename;
        let is_wanted = filename.ends_with(".safetensors")
            || filename == "tokenizer.json"
            || filename == "config.json"
            || filename == "tokenizer_config.json"
            || filename == "special_tokens_map.json";
        if !is_wanted {
            continue;
        }
        let local = repo.get(filename).await.map_err(|e| InferError::Hub(e.to_string()))?;
        if local_dir.is_none() {
            local_dir = local.parent().map(|p| p.to_path_buf());
        }
    }

    local_dir.ok_or_else(|| InferError::Hub(format!("no files found in {repo_id}")))
}

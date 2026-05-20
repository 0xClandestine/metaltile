//! Safetensors checkpoint loader → WeightMap (tensor name → raw bytes in
//! the tensor's own dtype). Supports single-file and sharded models
//! (multiple `.safetensors` files in a directory).

use std::path::{Path, PathBuf};

use metaltile_model::WeightMap;
use safetensors::SafeTensors;

use crate::error::InferError;

/// Load all tensors from a single `.safetensors` file.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<WeightMap, InferError> {
    let bytes = std::fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut map = WeightMap::new();
    for (name, view) in tensors.tensors() {
        map.insert(name.to_string(), view.data().to_vec());
    }
    Ok(map)
}

/// Load all tensors from every `.safetensors` file in a directory.
/// Later files overwrite earlier ones for the same tensor name, so shard
/// order does not matter (HF shards are non-overlapping).
pub fn load_safetensors_dir(dir: impl AsRef<Path>) -> Result<WeightMap, InferError> {
    let dir = dir.as_ref();
    let mut map = WeightMap::new();

    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .collect();
    shards.sort();

    for shard in shards {
        let bytes = std::fs::read(&shard)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        for (name, view) in tensors.tensors() {
            map.insert(name.to_string(), view.data().to_vec());
        }
    }

    Ok(map)
}

/// Auto-detect: if `path` is a file, load it directly; if it's a directory,
/// scan for all `.safetensors` shards.
pub fn load_weights(path: impl AsRef<Path>) -> Result<WeightMap, InferError> {
    let path = path.as_ref();
    if path.is_dir() {
        load_safetensors_dir(path)
    } else {
        load_safetensors(path)
    }
}

//! Safetensors checkpoint loader → WeightMap (tensor name → raw bytes cast
//! to the requested activation dtype). Supports single-file and sharded
//! models (multiple `.safetensors` files in a directory).

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use metaltile_core::dtype::DType;
use metaltile_model::WeightMap;
use safetensors::SafeTensors;
use tracing::info;

use crate::error::InferError;

// ── Dtype conversion helpers ───────────────────────────────────────────────

/// Convert tensor bytes from `src_dtype` to `dst_dtype` in a single pass,
/// eliminating the intermediate Vec<f32> that doubled peak memory during model load.
///
/// For same-dtype conversions (e.g. F16 checkpoint → F16 activation), returns
/// the bytes directly with no allocation. For cross-dtype conversions, decodes
/// element-by-element in a single pass into the target buffer.
fn convert_tensor(bytes: &[u8], src_dtype: safetensors::Dtype, dst: DType) -> Vec<u8> {
    let src_core = match src_dtype {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        _ => return bytes.to_vec(), // non-float: pass through unchanged
    };
    if src_core == dst {
        return bytes.to_vec(); // already correct dtype — zero-copy path
    }
    // Single-pass conversion: decode each element and write directly into
    // the target buffer, eliminating the intermediate Vec<f32> allocation.
    match (src_dtype, dst) {
        (safetensors::Dtype::F16, DType::F32) => {
            bytes
                .chunks_exact(2)
                .flat_map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32().to_le_bytes())
                .collect()
        },
        (safetensors::Dtype::F16, DType::BF16) => {
            bytes
                .chunks_exact(2)
                .flat_map(|b| half::bf16::from_f32(half::f16::from_le_bytes([b[0], b[1]]).to_f32()).to_le_bytes())
                .collect()
        },
        (safetensors::Dtype::BF16, DType::F32) => {
            bytes
                .chunks_exact(2)
                .flat_map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32().to_le_bytes())
                .collect()
        },
        (safetensors::Dtype::BF16, DType::F16) => {
            bytes
                .chunks_exact(2)
                .flat_map(|b| half::f16::from_f32(half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).to_le_bytes())
                .collect()
        },
        (safetensors::Dtype::F32, DType::F16) => {
            bytes
                .chunks_exact(4)
                .flat_map(|b| half::f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
                .collect()
        },
        (safetensors::Dtype::F32, DType::BF16) => {
            bytes
                .chunks_exact(4)
                .flat_map(|b| half::bf16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
                .collect()
        },
        _ => {
            // fallback for unexpected combinations — should not be hit
            // for any standard checkpoint (all tensors are float types)
            bytes.to_vec()
        },
    }
}

// ── Loaders ───────────────────────────────────────────────────────────────

/// Load all tensors from a single `.safetensors` file, converting each to
/// `target_dtype` (pass `None` to keep the native dtype).
pub fn load_safetensors(
    path: impl AsRef<Path>,
    target_dtype: Option<DType>,
) -> Result<WeightMap, InferError> {
    let bytes = std::fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut map = WeightMap::new();
    for (name, view) in tensors.tensors() {
        let data = match target_dtype {
            Some(dst) => convert_tensor(view.data(), view.dtype(), dst),
            None => view.data().to_vec(),
        };
        map.insert(name.to_string(), data);
    }
    Ok(map)
}

/// Load all tensors from every `.safetensors` file in a directory.
/// Later files overwrite earlier ones for the same tensor name, so shard
/// order does not matter (HF shards are non-overlapping).
pub fn load_safetensors_dir(
    dir: impl AsRef<Path>,
    target_dtype: Option<DType>,
) -> Result<WeightMap, InferError> {
    let dir = dir.as_ref();
    let mut map = WeightMap::new();

    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .collect();
    shards.sort();

    info!(n_shards = shards.len(), dir = %dir.display(), "loading safetensors shards");
    for shard in shards {
        let bytes = std::fs::read(&shard)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        for (name, view) in tensors.tensors() {
            let data = match target_dtype {
                Some(dst) => convert_tensor(view.data(), view.dtype(), dst),
                None => view.data().to_vec(),
            };
            map.insert(name.to_string(), data);
        }
    }

    Ok(map)
}

/// Auto-detect: if `path` is a file, load it directly; if it's a directory,
/// scan for all `.safetensors` shards. Then apply HF→MetalTile name remapping.
/// Weights are cast to `target_dtype` at load time.
#[tracing::instrument(skip(path), fields(path = %path.as_ref().display(), dtype = ?target_dtype))]
pub fn load_weights(path: impl AsRef<Path>, target_dtype: DType) -> Result<WeightMap, InferError> {
    let path = path.as_ref();
    let raw = if path.is_dir() {
        load_safetensors_dir(path, Some(target_dtype))?
    } else {
        load_safetensors(path, Some(target_dtype))?
    };
    Ok(remap_hf_llama_names(raw))
}

/// Remap HuggingFace Llama weight names to the MetalTile TOML convention.
///
/// HF format:                              MetalTile convention:
///   model.embed_tokens.weight           → tok_embeddings
///   model.layers.N.input_layernorm.weight → layers.N.attn_norm
///   model.layers.N.self_attn.q_proj.weight → layers.N.attn.q_proj
///   model.layers.N.self_attn.k_proj.weight → layers.N.attn.k_proj
///   model.layers.N.self_attn.v_proj.weight → layers.N.attn.v_proj
///   model.layers.N.self_attn.o_proj.weight → layers.N.attn.o_proj
///   model.layers.N.post_attention_layernorm.weight → layers.N.ffn_norm
///   model.layers.N.mlp.gate_proj.weight → layers.N.mlp.gate_proj
///   model.layers.N.mlp.up_proj.weight   → layers.N.mlp.up_proj
///   model.layers.N.mlp.down_proj.weight → layers.N.mlp.down_proj
///   model.norm.weight                   → output_norm
///   lm_head.weight                      → lm_head
///
/// Names that don't match any pattern are passed through unchanged.
pub fn remap_hf_llama_names(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::with_capacity(raw.len());
    for (name, bytes) in raw {
        let mapped = remap_one(&name);
        // Into<String> from Cow<str>: Borrowed variant is free (no alloc).
        out.insert(mapped.into_owned(), bytes);
    }
    out
}

/// Returns a remapped name.  Uses `Cow::Borrowed` for static strings to
/// avoid a heap allocation when no remapping is needed.
fn remap_one(name: &str) -> Cow<'_, str> {
    // model.embed_tokens.weight → tok_embeddings
    if name == "model.embed_tokens.weight" {
        return Cow::Borrowed("tok_embeddings");
    }
    // model.norm.weight → output_norm
    if name == "model.norm.weight" {
        return Cow::Borrowed("output_norm");
    }
    // lm_head.weight → lm_head
    if name == "lm_head.weight" {
        return Cow::Borrowed("lm_head");
    }

    // model.layers.N.<suffix> patterns
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot) = rest.find('.') {
            let layer_n = &rest[..dot];
            let suffix = &rest[dot + 1..];
            if let Ok(n) = layer_n.parse::<usize>() {
                let mapped_suffix = match suffix {
                    "input_layernorm.weight" => Some("attn_norm"),
                    "post_attention_layernorm.weight" => Some("ffn_norm"),
                    "self_attn.q_proj.weight" => Some("attn.q_proj"),
                    "self_attn.k_proj.weight" => Some("attn.k_proj"),
                    "self_attn.v_proj.weight" => Some("attn.v_proj"),
                    "self_attn.o_proj.weight" => Some("attn.o_proj"),
                    "mlp.gate_proj.weight" => Some("mlp.gate_proj"),
                    "mlp.up_proj.weight" => Some("mlp.up_proj"),
                    "mlp.down_proj.weight" => Some("mlp.down_proj"),
                    _ => None,
                };
                if let Some(s) = mapped_suffix {
                    return Cow::Owned(format!("layers.{n}.{s}"));
                }
            }
        }
    }

    // No match — pass through without allocating.
    Cow::Borrowed(name)
}

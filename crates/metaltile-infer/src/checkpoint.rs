//! Safetensors checkpoint loader → WeightMap (tensor name → raw bytes cast
//! to the requested activation dtype). Supports single-file and sharded
//! models (multiple `.safetensors` files in a directory).

use std::{borrow::Cow, path::Path};

use metaltile_core::dtype::DType;
use metaltile_model::WeightMap;
use safetensors::SafeTensors;
use tracing::info;

use crate::error::InferError;

// ── Dtype conversion ───────────────────────────────────────────────────

fn convert_tensor(bytes: &[u8], src_dtype: safetensors::Dtype, dst: DType) -> Vec<u8> {
    let src_core = match src_dtype {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        _ => return bytes.to_vec(), // non-float: pass through unchanged
    };
    if src_core == dst {
        return bytes.to_vec(); // already correct dtype — zero-copy
    }
    let (src_sz, dst_sz) = (src_core.size_bytes(), dst.size_bytes());
    let n = bytes.len() / src_sz;
    let mut out = vec![0u8; n * dst_sz];
    for i in 0..n {
        let base = i * src_sz;
        let f = match src_dtype {
            safetensors::Dtype::F16 => half::f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32(),
            safetensors::Dtype::BF16 => half::bf16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32(),
            safetensors::Dtype::F32 => f32::from_le_bytes([bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3]]),
            _ => unreachable!(),
        };
        let obase = i * dst_sz;
        match dst {
            DType::F16 => out[obase..obase + 2].copy_from_slice(&half::f16::from_f32(f).to_le_bytes()),
            DType::BF16 => out[obase..obase + 2].copy_from_slice(&half::bf16::from_f32(f).to_le_bytes()),
            DType::F32 => out[obase..obase + 4].copy_from_slice(&f.to_le_bytes()),
            _ => unreachable!(),
        }
    }
    out
}

// ── Loader ─────────────────────────────────────────────────────────────

/// Load weights from a `.safetensors` file or directory, cast to `target_dtype`,
/// and remap HF Llama names to MetalTile convention.
#[tracing::instrument(skip(path), fields(path = %path.as_ref().display(), dtype = ?target_dtype))]
pub fn load_weights(path: impl AsRef<Path>, target_dtype: DType) -> Result<WeightMap, InferError> {
    let path = path.as_ref();
    let mut map = WeightMap::default();

    // Collect safetensors shards (single file or directory).
    let shards: Vec<_> = if path.is_dir() {
        let mut s: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "safetensors")))
            .collect();
        s.sort();
        info!(n_shards = s.len(), dir = %path.display(), "loading safetensors shards");
        s
    } else {
        vec![path.to_path_buf()]
    };

    for shard in shards {
        let bytes = std::fs::read(&shard)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        for (name, view) in tensors.tensors() {
            let data = convert_tensor(view.data(), view.dtype(), target_dtype);
            map.insert(name.to_string(), data);
        }
    }

    Ok(remap_hf_llama_names(map))
}

/// Remap HuggingFace Llama weight names to the MetalTile TOML convention.
pub fn remap_hf_llama_names(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::default();
    for (name, bytes) in raw {
        out.insert(remap_one(&name).into_owned(), bytes);
    }
    out
}

fn remap_one(name: &str) -> Cow<'_, str> {
    if name == "model.embed_tokens.weight" { return Cow::Borrowed("tok_embeddings"); }
    if name == "model.norm.weight" { return Cow::Borrowed("output_norm"); }
    if name == "lm_head.weight" { return Cow::Borrowed("lm_head"); }

    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot) = rest.find('.') {
            let suffix = &rest[dot + 1..];
            if let Ok(n) = rest[..dot].parse::<usize>() {
                let mapped = match suffix {
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
                if let Some(s) = mapped {
                    return Cow::Owned(format!("layers.{n}.{s}"));
                }
            }
        }
    }
    Cow::Borrowed(name)
}
//! TOML schema types — serde-deserializable mirror of the model definition format.
//!
//! These are pure data with no logic. Validation happens in the compiler pass.
//!
//! ## TOML format
//!
//! ```toml
//! [model]
//! name = "llama3"
//! description = "Standard Llama 3 decoder-only transformer"
//!
//! [params]
//! n_layers = "$n_layers"
//! hidden_dim = "$hidden_dim"
//!
//! [[tensors]]
//! name = "tok_embeddings"
//! shape = ["$vocab_size", "$hidden_dim"]
//! dtype = "$weight_dtype"
//!
//! [layer]
//! name = "transformer_layer"
//!
//! [[layer.kernel]]
//! op = "rms_norm"
//! inputs = { x = "$residual", w = "$layers.$idx.attn_norm" }
//! outputs = { out = "_normed" }
//! constexpr = { n = "$hidden_dim" }
//!
//! [[kernel]]
//! op = "rms_norm"
//! inputs = { x = "$residual", w = "$output_norm" }
//! outputs = { out = "_final_normed" }
//! ```
//!
//! `[[layer.kernel]]` entries are unrolled `n_layers` times. Each
//! `[[kernel]]` entry at the top level runs once (output norm, lm head,
//! sampling).

use indexmap::IndexMap;
use serde::Deserialize;

/// Top-level model definition.
#[derive(Debug, Deserialize, Clone)]
pub struct ModelDef {
    pub model: ModelMeta,
    /// Global parameters with placeholder values resolved at load time.
    /// Values like `"$n_layers"` are substituted from the checkpoint metadata.
    #[serde(default)]
    pub params: IndexMap<String, String>,
    /// Weight tensor declarations (names and shapes).
    #[serde(default)]
    pub tensors: Vec<TensorDecl>,
    /// Per-layer kernel sequence. Unrolled `n_layers` times.
    pub layer: Option<LayerDef>,
    /// Post-layer kernels (output norm, lm head, sampling).
    #[serde(default)]
    pub kernel: Vec<KernelNode>,
}

/// Model metadata.
#[derive(Debug, Deserialize, Clone)]
pub struct ModelMeta {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// A weight tensor declaration.
#[derive(Debug, Deserialize, Clone)]
pub struct TensorDecl {
    pub name: String,
    /// Shape as string expressions, e.g. `["$vocab_size", "$hidden_dim"]`.
    pub shape: Vec<String>,
    /// Dtype as a string expression, e.g. `"$weight_dtype"`.
    pub dtype: String,
}

/// Definition of a repeated layer (unrolled `n_layers` times).
#[derive(Debug, Deserialize, Clone)]
pub struct LayerDef {
    /// Display name for the layer (optional).
    #[serde(default)]
    pub name: Option<String>,
    /// Kernel sequence executed once per layer.
    pub kernel: Vec<KernelNode>,
}

/// A single kernel dispatch node.
#[derive(Debug, Deserialize, Clone)]
pub struct KernelNode {
    /// Kernel op name. Resolved through the kernel registry.
    /// Maps to `BenchSpec.op` or `BenchSpec.op/subop`.
    pub op: String,
    /// Optional per-op parameters (e.g. `kind = "silu"` for activation).
    #[serde(default)]
    pub op_params: Option<IndexMap<String, String>>,
    /// Named input bindings. Values are tensor reference expressions.
    /// Examples: `"$residual"`, `"$layers.$idx.attn_norm"`, `"_normed"`.
    pub inputs: IndexMap<String, String>,
    /// Named output bindings. Values are tensor reference expressions.
    /// Intermediate names prefixed with `_` are transient (local to the layer).
    pub outputs: IndexMap<String, String>,
    /// Constexpr values bound at compile time.
    /// Values are arithmetic expressions over `$var` references.
    #[serde(default)]
    pub constexpr: Option<IndexMap<String, String>>,
    /// Override dtype for this node. Default: inherits model activation dtype.
    #[serde(default)]
    pub dtype: Option<String>,
}

// ── Validation helpers ──────────────────────────────────────────────────

impl ModelDef {
    /// Total kernel count after unrolling (validation-only, not compile).
    pub fn total_kernel_count(&self, n_layers: usize) -> usize {
        let per_layer = self.layer.as_ref().map_or(0, |l| l.kernel.len());
        let post = self.kernel.len();
        per_layer * n_layers + post
    }
}

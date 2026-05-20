//! Graph compiler: `ModelDef` + checkpoint params → `ExecutionPlan`.
//!
//! The compiler resolves all `$var` expressions, unrolls the layer loop,
//! validates kernel references against the registry, evaluates dispatch
//! hints, and assigns buffer slots.
//!
//! Compilation is pure CPU-side — no Metal device needed. The resulting
//! `ExecutionPlan` can be dispatched later via the executor.

use std::collections::HashMap;

use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::context::GridSpec;
use metaltile_std::spec::effective_mode;

use crate::{
    ConstexprValue,
    error::ModelError,
    expr::{eval_constexpr, eval_constexpr_fallible, eval_float_expr, resolve_tensor_ref},
    liveness::assign_slots,
    plan::{DispatchNode, ExecutionPlan, SlotRef},
    registry::KernelRegistry,
    schema::ModelDef,
};

/// Parameters resolved from a checkpoint, passed to the compiler.
#[derive(Debug, Clone)]
pub struct CompileParams {
    /// Integer model params (n_heads, hidden_dim, etc.).
    pub params: HashMap<String, u32>,
    /// Float model params (rope_theta, scale, etc.).
    pub float_params: HashMap<String, f64>,
    /// Activation dtype (f16, bf16, f32).
    pub activation_dtype: DType,
    /// Number of transformer layers in the model.
    pub n_layers: usize,
    /// Names treated as runtime state (not weights).
    /// Exact matches or prefix matches (e.g. "kv_cache" matches "kv_cache.0.k").
    pub state_keys: Vec<String>,
}

impl CompileParams {
    fn is_state_key(&self, name: &str) -> bool {
        self.state_keys.iter().any(|k| name == k.as_str() || name.starts_with(&format!("{k}.")))
    }
}

/// Compile a `ModelDef` into an `ExecutionPlan`.
///
/// Steps:
/// 1. Resolve `ModelDef.params` placeholder values against `CompileParams`.
/// 2. Unroll pre_kernel + layer loop + post kernels.
/// 3. For each node: validate op in registry, evaluate constexprs,
///    resolve tensor references to `SlotRef`s, compute grid.
/// 4. Run liveness analysis on intermediate buffers, assign `BufferSlot`s.
/// 5. Return the fully resolved `ExecutionPlan`.
pub fn compile(
    def: &ModelDef,
    p: &CompileParams,
    reg: &KernelRegistry,
) -> Result<ExecutionPlan, ModelError> {
    // ── Step 0: Merge def.params with CompileParams.params ──────────
    let resolved_params: HashMap<String, u32> = def
        .params
        .iter()
        .map(|(name, expr)| {
            let cleaned = expr.strip_prefix('$').unwrap_or(expr);
            let val = p
                .params
                .get(cleaned)
                .copied()
                .ok_or_else(|| ModelError::UnknownParam { name: expr.clone() })?;
            Ok((name.clone(), val))
        })
        .collect::<Result<HashMap<_, _>, ModelError>>()?;

    let resolved_float_params: HashMap<String, f64> = p.float_params.clone();

    // ── Step 1: Build the unrolled node list ────────────────────────
    let n_layers = p.n_layers;

    #[derive(Debug)]
    struct RawNode {
        label: String,
        node: crate::schema::KernelNode,
        layer_idx: Option<usize>,
    }

    let mut raw_nodes: Vec<RawNode> = Vec::new();

    // Pre-layer kernels (run once: token embedding, etc.).
    for kn in &def.pre_kernel {
        raw_nodes.push(RawNode {
            label: format!("pre.{}", kn.op),
            node: kn.clone(),
            layer_idx: None,
        });
    }

    // Unroll per-layer kernels.
    if let Some(layer_def) = &def.layer {
        for layer_idx in 0..n_layers {
            for kn in &layer_def.kernel {
                raw_nodes.push(RawNode {
                    label: format!("layer.{layer_idx}.{}", kn.op),
                    node: kn.clone(),
                    layer_idx: Some(layer_idx),
                });
            }
        }
    }

    // Post-layer kernels.
    for kn in &def.kernel {
        raw_nodes.push(RawNode {
            label: format!("post.{}", kn.op),
            node: kn.clone(),
            layer_idx: None,
        });
    }

    // ── Step 2: Compile each RawNode → DispatchNode ────────────────
    let mut nodes: Vec<DispatchNode> = Vec::with_capacity(raw_nodes.len());
    let mut intermediate_outputs: Vec<Vec<(String, usize)>> = Vec::with_capacity(raw_nodes.len());
    let mut intermediate_inputs: Vec<Vec<String>> = Vec::with_capacity(raw_nodes.len());

    for raw in &raw_nodes {
        // 2a. Look up BenchSpec.
        let spec = reg
            .get(&raw.node.op)
            .ok_or_else(|| ModelError::UnknownOp { op: raw.node.op.clone() })?;
        let mode = effective_mode(spec);

        // 2b. Generate a dummy kernel to extract param metadata.
        let kernel = (spec.kernel_ir)(p.activation_dtype);

        // 2c. Evaluate dispatch hints (for grid sizing and buffer sizing).
        let dispatch_hints = eval_dispatch_hints(
            raw.node.dispatch.as_ref(),
            &resolved_params,
            &resolved_float_params,
        );

        // 2d. Resolve constexpr expressions.
        let mut cexprs: Vec<(String, ConstexprValue)> = Vec::new();
        if let Some(ref ce_map) = raw.node.constexpr {
            for (name, expr) in ce_map {
                let is_float = kernel
                    .constexprs
                    .iter()
                    .any(|decl| decl.name.name() == name && decl.dtype.is_float());

                if is_float {
                    match eval_float_expr(expr, &resolved_params, &resolved_float_params) {
                        Ok(val) =>
                            cexprs.push((name.clone(), ConstexprValue::Static(val.to_bits()))),
                        Err(ModelError::UnknownParam { .. }) => {
                            let var = expr.trim().strip_prefix('$').unwrap_or(expr);
                            cexprs.push((name.clone(), ConstexprValue::State(var.to_string())));
                        },
                        Err(e) => return Err(e),
                    }
                } else {
                    match eval_constexpr_fallible(expr, &resolved_params, &resolved_float_params)? {
                        Some(val) => cexprs.push((name.clone(), ConstexprValue::Static(val))),
                        None => {
                            let var = expr.trim().strip_prefix('$').unwrap_or(expr);
                            cexprs.push((name.clone(), ConstexprValue::State(var.to_string())));
                        },
                    }
                }
            }
        }

        // 2e. Resolve tensor references.
        // Closure: resolve expr → SlotRef (intermediates stay as Weight("_name") until
        // slot assignment replaces them with Slot(idx)).
        let resolve_ref = |expr: &str| -> Result<SlotRef, ModelError> {
            let expr_clean = expr.trim();
            // Intermediates: `_`-prefixed names become Slot placeholders.
            if expr_clean.starts_with('_') {
                return Ok(SlotRef::Weight(expr_clean.to_string()));
            }
            // Resolve $idx in dotted paths.
            let resolved = if let Some(layer_idx) = raw.layer_idx {
                resolve_tensor_ref(expr_clean, layer_idx, &resolved_params)
            } else {
                // For non-layer nodes, still resolve $var but no $idx.
                let cleaned = expr_clean.strip_prefix('$').unwrap_or(expr_clean);
                if cleaned.contains('.') {
                    // Dotted path without $idx — return as-is.
                    cleaned.to_string()
                } else {
                    cleaned.to_string()
                }
            };
            // Check if resolved name is a state key.
            if p.is_state_key(&resolved) {
                return Ok(SlotRef::State(resolved));
            }
            Ok(SlotRef::Weight(resolved))
        };

        // Build input bindings.
        let mut input_bindings: Vec<(String, SlotRef)> = Vec::new();
        let mut node_intermediate_inputs: Vec<String> = Vec::new();

        for (param_name, tensor_expr) in &raw.node.inputs {
            let slot_ref = resolve_ref(tensor_expr)?;
            if tensor_expr.starts_with('_') {
                node_intermediate_inputs.push(tensor_expr.clone());
            }
            input_bindings.push((param_name.clone(), slot_ref));
        }

        // Build output bindings.
        let mut output_bindings: Vec<(String, SlotRef)> = Vec::new();
        let mut node_intermediate_outputs: Vec<(String, usize)> = Vec::new();

        for (param_name, tensor_expr) in &raw.node.outputs {
            let slot_ref = resolve_ref(tensor_expr)?;
            if tensor_expr.starts_with('_') {
                let size_bytes = compute_buffer_size(&dispatch_hints, p.activation_dtype);
                node_intermediate_outputs.push((tensor_expr.clone(), size_bytes));
            }
            output_bindings.push((param_name.clone(), slot_ref));
        }

        intermediate_outputs.push(node_intermediate_outputs);
        intermediate_inputs.push(node_intermediate_inputs);

        // 2f. Compute grid dimensions from dispatch hints.
        let grid = compute_grid(&mode, &dispatch_hints)?;

        let kernel_name: &'static str = spec.kernel_name;
        let kernel_ir = spec.kernel_ir;

        nodes.push(DispatchNode {
            label: raw.label.clone(),
            kernel_name,
            kernel_ir,
            mode,
            input_bindings,
            output_bindings,
            cexprs,
            grid,
            dtype: p.activation_dtype,
        });
    }

    // ── Step 3: Liveness analysis → slot assignment ────────────────
    let slots = assign_slots(nodes.len(), &intermediate_outputs, &intermediate_inputs);

    // Build name → slot index map.
    let name_to_slot: HashMap<String, usize> =
        slots.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();

    // Replace Weight("_name") placeholders with Slot(idx).
    for node in &mut nodes {
        for (_, slot_ref) in node.input_bindings.iter_mut().chain(node.output_bindings.iter_mut()) {
            if let SlotRef::Weight(name) = slot_ref
                && name.starts_with('_')
                && let Some(slot_idx) = name_to_slot.get(name)
            {
                *slot_ref = SlotRef::Slot(*slot_idx);
            }
        }
    }

    // ── Step 4: Determine output slot ──────────────────────────────
    // Use the last node's first output slot.
    let output_slot = nodes
        .last()
        .and_then(|n| n.output_bindings.first())
        .and_then(|(_, sr)| if let SlotRef::Slot(idx) = sr { Some(*idx) } else { None })
        .unwrap_or(0);

    Ok(ExecutionPlan { nodes, slots, output_slot, n_layers })
}

// ── Dispatch hint evaluation ───────────────────────────────────────────

/// Evaluate the `dispatch` map from a KernelNode into a resolved `u32` map.
///
/// Dispatch hints are compile-time expressions used for grid sizing and
/// output buffer sizing. Unknown variables cause the hint to be skipped
/// (not an error, just falls back to defaults).
fn eval_dispatch_hints(
    dispatch: Option<&indexmap::IndexMap<String, String>>,
    params: &HashMap<String, u32>,
    float_params: &HashMap<String, f64>,
) -> HashMap<String, u32> {
    let Some(map) = dispatch else { return HashMap::new() };
    let mut out = HashMap::new();
    for (key, expr) in map {
        if let Ok(val) = eval_constexpr(expr, params, float_params) {
            out.insert(key.clone(), val);
        }
        // else: skip unresolvable hints
    }
    out
}

// ── Grid computation ───────────────────────────────────────────────────

/// Compute the `GridSpec` for a kernel dispatch from its mode and
/// resolved dispatch hints.
fn compute_grid(mode: &KernelMode, hints: &HashMap<String, u32>) -> Result<GridSpec, ModelError> {
    match mode {
        KernelMode::Elementwise => {
            let n = hints.get("n").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Elementwise { n })
        },
        KernelMode::Reduction => {
            let rows = hints.get("rows").copied().unwrap_or(1) as usize;
            let tpg = hints.get("tpg").copied().unwrap_or(256) as usize;
            Ok(GridSpec::Reduction { num_rows: rows, threads_per_group: tpg })
        },
        KernelMode::Grid3D => {
            let x = hints.get("grid_x").copied().unwrap_or(1) as usize;
            let y = hints.get("grid_y").copied().unwrap_or(1) as usize;
            let z = hints.get("grid_z").copied().unwrap_or(1) as usize;
            let tpg = hints.get("tpg").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Grid3D { x, y, z, threads_per_group: tpg })
        },
        KernelMode::Tile2D => {
            let n = hints.get("n").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Elementwise { n })
        },
        KernelMode::SimdGroup2D => {
            let rows = hints.get("rows").copied().unwrap_or(1) as usize;
            let tpg = hints.get("tpg").copied().unwrap_or(1024) as usize;
            Ok(GridSpec::Reduction { num_rows: rows, threads_per_group: tpg })
        },
    }
}

// ── Buffer size estimation ─────────────────────────────────────────────

/// Compute the size in bytes for an intermediate output buffer.
///
/// Uses the `out_elems` dispatch hint (number of output elements).
/// Falls back to `hidden_dim * dtype.size_bytes()` if not specified.
fn compute_buffer_size(hints: &HashMap<String, u32>, dtype: DType) -> usize {
    let elems = hints.get("out_elems").copied().unwrap_or(0) as usize;
    if elems > 0 {
        elems * dtype.size_bytes()
    } else {
        // Conservative fallback — should be overridden via dispatch hints.
        4096 * dtype.size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ModelDef;

    fn make_registry() -> KernelRegistry { KernelRegistry::build() }

    fn base_state_keys() -> Vec<String> {
        vec![
            "token_id".into(),
            "position".into(),
            "rms_eps".into(),
            "temperature".into(),
            "uniform".into(),
            "kv_cache".into(),
        ]
    }

    /// A minimal TOML that just runs rms_norm once (no layer loop).
    fn minimal_toml() -> &'static str {
        r#"
[model]
name = "test"
description = "Minimal test model"

[params]
hidden_dim = "$hidden_dim"

[model.layer]
name = "test_layer"

[[layer.kernel]]
op = "rms_norm"
inputs = { x = "_input", w = "_weight" }
outputs = { out = "_output" }
constexpr = { n = "$hidden_dim" }
dispatch = { rows = "1", tpg = "1024", out_elems = "$hidden_dim" }
"#
    }

    #[test]
    fn compile_minimal_model_succeeds() {
        let def: ModelDef = toml::from_str(minimal_toml()).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: {
                let mut m = HashMap::new();
                m.insert("hidden_dim".into(), 128);
                m.insert("n_layers".into(), 1);
                m
            },
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 1,
            state_keys: vec![],
        };

        let plan = compile(&def, &params, &reg).expect("compile");
        assert_eq!(plan.nodes.len(), 1, "one layer × one kernel = 1 node");
        assert_eq!(plan.n_layers, 1);
    }

    #[test]
    fn layer_unrolling() {
        let def: ModelDef = toml::from_str(minimal_toml()).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: {
                let mut m = HashMap::new();
                m.insert("hidden_dim".into(), 128);
                m.insert("n_layers".into(), 4);
                m
            },
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 4,
            state_keys: vec![],
        };

        let plan = compile(&def, &params, &reg).expect("compile");
        assert_eq!(plan.nodes.len(), 4, "4 layers × 1 kernel = 4 nodes");
        assert_eq!(plan.n_layers, 4);

        for i in 0..4 {
            assert_eq!(plan.nodes[i].label, format!("layer.{i}.rms_norm"));
        }
    }

    #[test]
    fn unknown_op_is_error() {
        let toml_src = r#"
[model]
name = "bad"

[model.layer]
[[layer.kernel]]
op = "nonexistent_kernel"
inputs = {}
outputs = {}
"#;
        let def: ModelDef = toml::from_str(toml_src).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: HashMap::new(),
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 1,
            state_keys: vec![],
        };

        let result = compile(&def, &params, &reg);
        assert!(result.is_err(), "unknown op should fail");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent_kernel"), "error should mention the bad op: {err}");
    }

    #[test]
    fn llama_decode_toml_parses_and_compiles() {
        let src = include_str!("../../../models/llama_decode.toml");
        let def: ModelDef = toml::from_str(src).expect("parse llama_decode.toml");
        assert_eq!(def.model.name, "llama-decode");

        let reg = make_registry();

        // Verify pre_kernel ops exist.
        for kn in &def.pre_kernel {
            assert!(reg.get(&kn.op).is_some(), "pre op '{}' not found in kernel registry", kn.op);
        }

        // Verify all ops in the layer exist in the registry.
        if let Some(ref layer) = def.layer {
            for kn in &layer.kernel {
                assert!(
                    reg.get(&kn.op).is_some(),
                    "layer op '{}' not found in kernel registry",
                    kn.op
                );
            }
        }

        // Verify all post-layer ops exist.
        for kn in &def.kernel {
            assert!(reg.get(&kn.op).is_some(), "post op '{}' not found in kernel registry", kn.op);
        }

        // Try compiling.
        let params = CompileParams {
            params: {
                let mut m = HashMap::new();
                m.insert("n_layers".into(), 4);
                m.insert("n_heads".into(), 32);
                m.insert("n_kv_heads".into(), 8);
                m.insert("head_dim".into(), 128);
                m.insert("hidden_dim".into(), 4096);
                m.insert("ffn_dim".into(), 14336);
                m.insert("vocab_size".into(), 128256);
                m.insert("max_seq_len".into(), 8192);
                m
            },
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 4,
            state_keys: base_state_keys(),
        };

        let plan = compile(&def, &params, &reg).expect("compile llama_decode");
        assert_eq!(plan.n_layers, 4);

        let pre = def.pre_kernel.len();
        let per_layer = def.layer.as_ref().unwrap().kernel.len();
        let post = def.kernel.len();
        let expected_nodes = pre + per_layer * 4 + post;
        assert_eq!(
            plan.nodes.len(),
            expected_nodes,
            "{pre} pre + {per_layer} kernels/layer × 4 layers + {post} post = {expected_nodes} nodes"
        );

        for node in &plan.nodes {
            assert!(!node.label.is_empty(), "node must have a label");
        }
    }
}

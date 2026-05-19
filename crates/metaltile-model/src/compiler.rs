//! Graph compiler: `ModelDef` + checkpoint params → `ExecutionPlan`.
//!
//! The compiler resolves all `$var` expressions, unrolls the layer loop,
//! validates kernel references against the registry, computes grid
//! dimensions, and assigns buffer slots.
//!
//! Compilation is pure CPU-side — no Metal device needed. The resulting
//! `ExecutionPlan` can be dispatched later via the executor.

use std::collections::HashMap;

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::context::GridSpec;
use metaltile_std::spec::{BenchSpec, effective_mode};

use crate::{
    ConstexprValue,
    error::ModelError,
    expr::{eval_constexpr_fallible, eval_float_expr, resolve_tensor_ref},
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
}

/// Compile a `ModelDef` into an `ExecutionPlan`.
///
/// Steps:
/// 1. Resolve `ModelDef.params` placeholder values against `CompileParams`.
/// 2. Unroll the layer loop `n_layers` times, substituting `$idx` in all
///    tensor references and constexpr expressions.
/// 3. For each unrolled node: validate op in registry, evaluate constexprs,
///    resolve tensor references to `SlotRef`s, compute grid.
/// 4. Run liveness analysis on intermediate buffers, assign `BufferSlot`s.
/// 5. Return the fully resolved `ExecutionPlan`.
pub fn compile(
    def: &ModelDef,
    p: &CompileParams,
    reg: &KernelRegistry,
) -> Result<ExecutionPlan, ModelError> {
    // ── Step 0: Merge def.params with CompileParams.params ──────────
    // The TOML `params` section has placeholder values like `"$n_layers"`.
    // We resolve those against the concrete CompileParams.
    let resolved_params: HashMap<String, u32> = def
        .params
        .iter()
        .map(|(name, expr)| {
            // Strip leading $ if present.
            let cleaned = expr.strip_prefix('$').unwrap_or(expr);
            let val = p.params.get(cleaned).copied().ok_or_else(|| {
                ModelError::UnknownParam {
                    name: expr.clone(),
                }
            })?;
            Ok((name.clone(), val))
        })
        .collect::<Result<HashMap<_, _>, ModelError>>()?;

    // Float params — similarly resolved.
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
    let mut intermediate_outputs: Vec<Vec<(String, usize)>> =
        Vec::with_capacity(raw_nodes.len());
    let mut intermediate_inputs: Vec<Vec<String>> =
        Vec::with_capacity(raw_nodes.len());

    for raw in &raw_nodes {
        // 2a. Look up BenchSpec.
        let spec = reg.get(&raw.node.op).ok_or_else(|| ModelError::UnknownOp {
            op: raw.node.op.clone(),
        })?;
        let mode = effective_mode(spec);

        // 2b. Generate a dummy kernel to extract param metadata.
        // We only need the param list — the IR body is discarded.
        let kernel = (spec.kernel_ir)(p.activation_dtype);

        // 2c. Resolve constexpr expressions.
        // Static params are looked up in resolved_params.
        // Unknown params (like $position) become State references
        // resolved per-dispatch from the runtime state map.
        let mut cexprs: Vec<(String, ConstexprValue)> = Vec::new();
        if let Some(ref ce_map) = raw.node.constexpr {
            for (name, expr) in ce_map {
                // Check if it's a float constexpr (check kernel signature).
                let is_float = kernel
                    .constexprs
                    .iter()
                    .any(|decl| decl.name.name() == name && decl.dtype.is_float());

                if is_float {
                    match eval_float_expr(expr, &resolved_params, &resolved_float_params) {
                        Ok(val) => cexprs.push((name.clone(), ConstexprValue::Static(val.to_bits()))),
                        Err(ModelError::UnknownParam { .. }) => {
                            // Float runtime state — stored as State reference.
                            // Extract the var name from the expression.
                            let var = expr.trim().strip_prefix('$').unwrap_or(expr);
                            cexprs.push((name.clone(), ConstexprValue::State(var.to_string())));
                        },
                        Err(e) => return Err(e),
                    }
                } else {
                    match eval_constexpr_fallible(expr, &resolved_params, &resolved_float_params)? {
                        Some(val) => cexprs.push((name.clone(), ConstexprValue::Static(val))),
                        None => {
                            // Unknown variable → runtime state.
                            let var = expr.trim().strip_prefix('$').unwrap_or(expr);
                            cexprs.push((name.clone(), ConstexprValue::State(var.to_string())));
                        },
                    }
                }
            }
        }

        // 2d. Resolve tensor references for inputs and outputs.
        let mut bindings: Vec<(String, SlotRef)> = Vec::new();

        // Helper: resolve a tensor reference to a SlotRef.
        let resolve_ref =
            |expr: &str| -> Result<SlotRef, ModelError> {
                let expr_clean = expr.trim();
                // Check for intermediates: names starting with _
                if expr_clean.starts_with('_') {
                    return Ok(SlotRef::Weight(expr_clean.to_string()));
                }
                // Check for state variables (kv_cache, position).
                // These are identified by context; for now, use a convention:
                // names containing "kv_cache" → State.
                if expr_clean.contains("kv_cache") {
                    return Ok(SlotRef::State(expr_clean.to_string()));
                }
                // Otherwise, it's a weight tensor reference.
                // Resolve $idx in dotted paths.
                let resolved = if let Some(layer_idx) = raw.layer_idx {
                    resolve_tensor_ref(
                        expr_clean,
                        layer_idx,
                        &resolved_params,
                    )
                } else {
                    expr_clean.to_string()
                };
                Ok(SlotRef::Weight(resolved))
            };

        for (param_name, tensor_expr) in &raw.node.inputs {
            let slot_ref = resolve_ref(tensor_expr)?;
            bindings.push((param_name.clone(), slot_ref));
        }

        // Track intermediate outputs for liveness analysis.
        let mut node_intermediate_outputs: Vec<(String, usize)> = Vec::new();
        let mut node_intermediate_inputs: Vec<String> = Vec::new();

        for (param_name, tensor_expr) in &raw.node.outputs {
            let slot_ref = resolve_ref(tensor_expr)?;

            // If the name starts with _, it's an intermediate.
            if tensor_expr.starts_with('_') {
                let size_bytes = compute_buffer_size(
                    &kernel,
                    param_name,
                    &resolved_params,
                    p.activation_dtype,
                );
                node_intermediate_outputs.push((
                    tensor_expr.clone(),
                    size_bytes,
                ));
            }

            bindings.push((param_name.clone(), slot_ref));
        }

        // Collect intermediate inputs (from input bindings that reference
        // intermediates).
        for (_, tensor_expr) in &raw.node.inputs {
            if tensor_expr.starts_with('_') {
                node_intermediate_inputs.push(tensor_expr.clone());
            }
        }

        intermediate_outputs.push(node_intermediate_outputs);
        intermediate_inputs.push(node_intermediate_inputs);

        // 2e. Compute grid dimensions.
        let grid = compute_grid(
            spec,
            &mode,
            &resolved_params,
            &cexprs,
        )?;

        // 2f. Build DispatchNode.
        // Note: 'static str requirement — we need to leak or use &'static.
        // Since BenchSpec is 'static, we can borrow from it.
        let kernel_name: &'static str = spec.kernel_name;
        let kernel_ir = spec.kernel_ir;

        nodes.push(DispatchNode {
            label: raw.label.clone(),
            kernel_name,
            kernel_ir,
            mode,
            bindings,
            cexprs,
            grid,
            dtype: p.activation_dtype,
        });
    }

    // ── Step 3: Liveness analysis → slot assignment ────────────────
    let slots = assign_slots(
        nodes.len(),
        &intermediate_outputs,
        &intermediate_inputs,
    );

    // Update SlotRef::Weight for intermediate names to SlotRef::Slot.
    // We need to build a name → slot index map.
    let name_to_slot: HashMap<String, usize> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.clone(), i))
        .collect();

    for node in &mut nodes {
        for (_, slot_ref) in &mut node.bindings {
            if let SlotRef::Weight(name) = slot_ref
                && name.starts_with('_')
                && let Some(slot_idx) = name_to_slot.get(name)
            {
                *slot_ref = SlotRef::Slot(*slot_idx);
            }
        }
    }

    // ── Step 4: Determine output slot ──────────────────────────────
    // The last node's last output binding is the final output.
    // For now, use slot 0 as a reasonable default (the logits buffer).
    // A more precise approach would trace the output through the plan.
    let output_slot = 0usize;

    Ok(ExecutionPlan {
        nodes,
        slots,
        output_slot,
        n_layers,
    })
}

// ── Grid computation ───────────────────────────────────────────────────

/// Compute the `GridSpec` for a kernel dispatch from its BenchSpec,
/// KernelMode, and resolved constexprs.
fn compute_grid(
    _spec: &BenchSpec,
    mode: &KernelMode,
    _params: &HashMap<String, u32>,
    cexprs: &[(String, ConstexprValue)],
) -> Result<GridSpec, ModelError> {
    // Build a map from constexpr name → value for quick lookup.
    // Only Static values are used for grid computation.
    let ce_map: HashMap<&str, u32> = cexprs
        .iter()
        .filter_map(|(k, v)| match v {
            ConstexprValue::Static(val) => Some((k.as_str(), *val)),
            ConstexprValue::State(_) => None,
        })
        .collect();

    match mode {
        KernelMode::Elementwise => {
            // Elementwise: one thread per element. The total element count
            // is typically `n` or the output buffer size.
            let n = ce_map.get("n").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Elementwise { n })
        },
        KernelMode::Reduction => {
            // Reduction: `b` threadgroups × `tpg` threads per group.
            let b = ce_map.get("b").copied().unwrap_or(1) as usize;
            let tpg = 1024usize; // default; could be overridden
            Ok(GridSpec::Reduction {
                num_rows: b,
                threads_per_group: tpg,
            })
        },
        KernelMode::Grid3D => {
            // Grid3D: explicit x, y, z dimensions.
            // For now, extract from common constexpr patterns.
            let x = ce_map.get("n_heads").copied().unwrap_or(1) as usize;
            let y = ce_map.get("half_dim").copied().unwrap_or(1) as usize;
            let z = 1usize;
            let tpg = 1usize;
            Ok(GridSpec::Grid3D {
                x,
                y,
                z,
                threads_per_group: tpg,
            })
        },
        KernelMode::Tile2D => {
            // Tile2D: treat as elementwise for now (single threadgroup).
            let n = ce_map.get("n").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Elementwise { n })
        },
        KernelMode::SimdGroup2D => {
            // SimdGroup2D: tiled matmul/SDPA.
            // For decode SDPA: one threadgroup per Q head, full TG size.
            let n_q_heads = ce_map.get("n_q_heads").copied().unwrap_or(1) as usize;
            let tpg = 1024usize;
            Ok(GridSpec::Reduction {
                num_rows: n_q_heads,
                threads_per_group: tpg,
            })
        },
    }
}

// ── Buffer size estimation ─────────────────────────────────────────────

/// Compute the size in bytes for an output buffer, given the kernel's
/// param metadata and resolved constexprs.
///
/// This uses heuristics based on kernel type and constexpr parameters.
/// A more precise approach would evaluate the kernel's return_shapes at
/// compile time.
fn compute_buffer_size(
    kernel: &Kernel,
    param_name: &str,
    params: &HashMap<String, u32>,
    dtype: DType,
) -> usize {
    // Try to find the param in the kernel's param list.
    let Some(param) = kernel.params.iter().find(|p| p.name == param_name) else {
        // Fallback: use a conservative estimate based on known dimensions.
        let n = params.get("hidden_dim").copied().unwrap_or(4096) as usize;
        return n * dtype.size_bytes();
    };

    // If the param has a static shape, use it.
    if let Some(elems) = param.shape.num_elements() {
        return elems * dtype.size_bytes();
    }

    // Constexpr-driven shape: use hidden_dim as fallback.
    let n = params.get("hidden_dim").copied().unwrap_or(4096) as usize;
    n * dtype.size_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ModelDef};

    fn make_registry() -> KernelRegistry {
        KernelRegistry::build()
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
inputs = { x = "$input", w = "$weight" }
outputs = { out = "$output" }
constexpr = { n = "$hidden_dim" }
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
        };

        let plan = compile(&def, &params, &reg).expect("compile");
        assert_eq!(plan.nodes.len(), 4, "4 layers × 1 kernel = 4 nodes");
        assert_eq!(plan.n_layers, 4);

        // Labels should reflect layer indices.
        for i in 0..4 {
            assert_eq!(plan.nodes[i].label, format!("layer.{i}.rms_norm"));
        }
    }

    #[test]
    fn unknown_op_is_error() {
        let toml = r#"
[model]
name = "bad"

[model.layer]
[[layer.kernel]]
op = "nonexistent_kernel"
inputs = {}
outputs = {}
"#;
        let def: ModelDef = toml::from_str(toml).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: HashMap::new(),
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 1,
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
            assert!(
                reg.get(&kn.op).is_some(),
                "post op '{}' not found in kernel registry",
                kn.op
            );
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
        };

        let plan = compile(&def, &params, &reg).expect("compile llama_decode");
        assert_eq!(plan.n_layers, 4);

        // Llama decode: ~20 kernels per layer + 3 post-layer.
        let per_layer = def.layer.as_ref().unwrap().kernel.len();
        let post = def.kernel.len();
        let expected_nodes = per_layer * 4 + post;
        assert_eq!(
            plan.nodes.len(),
            expected_nodes,
            "{per_layer} kernels/layer × 4 layers + {post} post = {expected_nodes} nodes"
        );

        // Each node should have a valid kernel_ir and grid.
        for node in &plan.nodes {
            assert!(!node.label.is_empty(), "node must have a label");
            assert!(!node.bindings.is_empty(), "node must have buffer bindings");
        }
    }
}
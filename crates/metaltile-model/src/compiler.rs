//! Graph compiler: `ModelDef` + checkpoint params → `ExecutionPlan`.
//!
//! The compiler resolves all `$var` expressions, unrolls the layer loop,
//! validates kernel references against the registry, evaluates dispatch
//! hints, and assigns buffer slots.
//!
//! Compilation is pure CPU-side — no Metal device needed. The resulting
//! `ExecutionPlan` can be dispatched later via the executor.

use std::collections::{HashMap, HashSet};

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
    schema::{KernelNode, ModelDef},
};

/// An un-compiled node from the TOML, before op resolution and grid
/// computation.
#[derive(Debug)]
struct RawNode {
    label: String,
    node: KernelNode,
    layer_idx: Option<usize>,
}

/// Controls how kernel fusion is applied during compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMode {
    /// No fusion — dispatch every kernel individually.
    None,
    /// Honor TOML `fuse = "..."` tags. Contiguous kernels with the same
    /// tag are dispatched through a single `dispatch_chain` call.
    TomlDriven,
    /// Run the automatic graph-level fusion pass — identifies
    /// producer-consumer chains and fuses them. Ignores TOML fuse tags.
    GraphDriven,
}

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
/// 4. Run graph-level fusion or apply TOML fuse tags (depending on
///    `fusion_mode`).
/// 5. Run liveness analysis on intermediate buffers, assign `BufferSlot`s.
/// 6. Return the fully resolved `ExecutionPlan`.
pub fn compile(
    def: &ModelDef,
    p: &CompileParams,
    reg: &KernelRegistry,
    fusion_mode: FusionMode,
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
        let grid = compute_grid(&raw.node.op, &mode, &dispatch_hints)?;

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
            fuse_group: None,
        });
    }

    // ── Step 2.5: Graph-level fusion or TOML fuse groups ─────────
    match fusion_mode {
        FusionMode::TomlDriven => {
            validate_fuse_groups(&raw_nodes)?;
            assign_toml_fuse_groups(&mut nodes, &raw_nodes);
        },
        FusionMode::GraphDriven => {
            let (n_fused_groups, n_unfused) = fuse_dispatch_nodes(&mut nodes);
            // Log fusion stats if there were any fused groups.
            if n_fused_groups > 0 {
                eprintln!(
                    "[compiler] graph fusion: {n_fused_groups} groups, {n_unfused} standalone nodes"
                );
            }
        },
        FusionMode::None => { /* no fusion */ },
    }

    // ── Step 3: Liveness analysis → slot assignment ────────────────
    // name_to_slot is the canonical map built during assignment — it preserves
    // all tenant names even when slots are reused (slot.name is overwritten on
    // reuse, so rebuilding from the slot vector would lose earlier tenant names).

    // Filter out intra-fuse-group intermediates — they're handled by
    // dispatch_chain's private-memory aliasing and don't need persistent
    // BufferSlots.
    let (filtered_outputs, filtered_inputs) =
        filter_intra_group_intermediates(&nodes, &intermediate_outputs, &intermediate_inputs);

    let (slots, name_to_slot) =
        assign_slots(nodes.len(), &filtered_outputs, &filtered_inputs);

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

// ── Graph-level kernel fusion ──────────────────────────────────────────

/// Maximum number of nodes that can be fused into one `dispatch_chain` call.
const MAX_FUSED_PER_CHAIN: usize = 8;

/// Intermediate name prefix — tensors with this prefix are transient
/// (local to the layer) and can participate in fusion.
const INTERMEDIATE_PREFIX: char = '_';

/// Fuse adjacent `DispatchNode`s whose intermediate outputs have a single
/// consumer within their local scope.  Returns `(n_fused_groups, n_unfused_nodes)`.
///
/// Algorithm (inspired by the IR-level `fuse_block` in `passes/fusion.rs`):
/// 1. Build ordered write/read position lists per intermediate name.
///    Because intermediate names are reused across layers (e.g. `_gate`
///    appears in every layer), we track per-instance scoping: an
///    intermediate is single-use only if, between its write and the next
///    write to the same name, there is exactly one reader — and that
///    reader is the adjacent node.
/// 2. Walk nodes in reverse, finding maximal fusible chains.
/// 3. Assign sequential `fuse_group` IDs to each chain.
fn fuse_dispatch_nodes(nodes: &mut [DispatchNode]) -> (usize, usize) {
    let n = nodes.len();
    if n < 2 {
        return (0, n);
    }

    // ── Phase 1: Build ordered position lists ──────────────────
    // writes[name] = sorted node indices that write `name`.
    // reads[name]  = sorted node indices that read `name`.
    // defs[i]       = intermediate names written by node `i`.
    let mut writes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut reads: HashMap<String, Vec<usize>> = HashMap::new();
    let mut defs: Vec<Vec<String>> = vec![Vec::new(); n];

    for (i, node) in nodes.iter().enumerate() {
        for (_, slot_ref) in &node.input_bindings {
            if let SlotRef::Weight(name) = slot_ref
                && name.starts_with(INTERMEDIATE_PREFIX)
            {
                reads.entry(name.clone()).or_default().push(i);
            }
        }
        for (_, slot_ref) in &node.output_bindings {
            if let SlotRef::Weight(name) = slot_ref
                && name.starts_with(INTERMEDIATE_PREFIX)
            {
                writes.entry(name.clone()).or_default().push(i);
                defs[i].push(name.clone());
            }
        }
    }

    // Sort positions (they're inserted in order already, but be safe).
    for v in writes.values_mut() { v.sort_unstable(); }
    for v in reads.values_mut() { v.sort_unstable(); }

    // ── Phase 2: Find maximal fusible chains (backward scan) ─────
    let mut fused: HashSet<usize> = HashSet::default();
    let mut chains: Vec<Vec<usize>> = Vec::new();

    for i in (0..n).rev() {
        if fused.contains(&i) {
            continue;
        }

        let mut chain: Vec<usize> = vec![i];
        let mut cursor = i;

        loop {
            let Some(pred) =
                find_single_use_producer_local(cursor, nodes, &writes, &reads, &defs)
            else {
                break;
            };
            if fused.contains(&pred)
                || !is_fusible_local(nodes, pred, cursor, &writes, &reads)
            {
                break;
            }
            if chain.len() >= MAX_FUSED_PER_CHAIN {
                break;
            }
            chain.push(pred);
            cursor = pred;
        }

        if chain.len() >= 2 {
            chain.reverse();
            for &idx in &chain {
                fused.insert(idx);
            }
            chains.push(chain);
        }
    }

    // ── Phase 3: Assign sequential fuse_group IDs ────────────────
    let total_fused: usize = chains.iter().map(|c| c.len()).sum();
    let n_unfused = n - total_fused;

    for (group_id, chain) in chains.iter().enumerate() {
        for &node_idx in chain {
            nodes[node_idx].fuse_group = Some(group_id);
        }
    }

    (chains.len(), n_unfused)
}

/// Check whether `name` written at `producer_idx` has a single consumer
/// in its local scope (between this write and the next write to `name`).
/// Returns `true` when exactly one reader exists and it is at
/// `producer_idx + 1`.
fn is_local_single_use(
    name: &str,
    producer_idx: usize,
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
) -> bool {
    let Some(write_list) = writes.get(name) else { return false };
    let Some(read_list) = reads.get(name) else { return false };

    // Find this write's position in the ordered write list.
    let Ok(write_pos) = write_list.binary_search(&producer_idx) else { return false };
    let next_write = write_list.get(write_pos + 1).copied();

    // Find the first reader at or after producer_idx.
    let read_start = match read_list.binary_search(&producer_idx) {
        Ok(pos) => pos + 1, // skip the write itself if it happens to also read
        Err(pos) => pos,
    };
    // Find the first reader at or after the next write (exclusive bound).
    let read_end = match next_write {
        Some(nw) => match read_list.binary_search(&nw) {
            Ok(pos) => pos,
            Err(pos) => pos,
        },
        None => read_list.len(),
    };

    let local_readers = &read_list[read_start..read_end];
    local_readers.len() == 1 && local_readers[0] == producer_idx + 1
}

/// Find the immediate predecessor of `cursor` that writes an intermediate
/// which `cursor` reads and which is single-use in its local scope.
fn find_single_use_producer_local(
    cursor: usize,
    nodes: &[DispatchNode],
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
    defs: &[Vec<String>],
) -> Option<usize> {
    let pred = cursor.checked_sub(1)?;

    // All intermediates written by pred that cursor also reads.
    for name in &defs[pred] {
        let cursor_reads = nodes[cursor]
            .input_bindings
            .iter()
            .any(|(_, sr)| matches!(sr, SlotRef::Weight(n) if n == name));
        if cursor_reads && is_local_single_use(name, pred, writes, reads) {
            return Some(pred);
        }
    }
    None
}

/// Predicate: can node `producer` and `consumer` be fused?
/// Uses local-scope single-use semantics (respects intermediate name reuse
/// across layers).
fn is_fusible_local(
    nodes: &[DispatchNode],
    producer: usize,
    consumer: usize,
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
) -> bool {
    if consumer != producer + 1 {
        return false;
    }

    let mut has_shared = false;
    for (_, slot_ref) in &nodes[producer].output_bindings {
        if let SlotRef::Weight(name) = slot_ref
            && name.starts_with(INTERMEDIATE_PREFIX)
        {
            // Check if consumer reads this name.
            let read_by_consumer = nodes[consumer]
                .input_bindings
                .iter()
                .any(|(_, sr)| matches!(sr, SlotRef::Weight(n) if n == name));

            if !read_by_consumer {
                // Producer writes this name, but consumer doesn't read it.
                // That's fine — but we must NOT penalize multi-use here.
                // Only check local single-use for names the consumer DOES read.
                continue;
            }

            // Consumer reads this name — now verify local single-use.
            if !is_local_single_use(name, producer, writes, reads) {
                return false;
            }
            has_shared = true;
        }
    }
    has_shared
}

// ── TOML fuse tag handling ─────────────────────────────────────────────

/// Validate that TOML `fuse` tags are contiguous.
fn validate_fuse_groups(raw_nodes: &[RawNode]) -> Result<(), ModelError> {
    let mut seen: HashMap<&str, usize> = HashMap::default();
    let mut in_group: Option<&str> = None;

    for (i, raw) in raw_nodes.iter().enumerate() {
        let tag = raw.node.fuse.as_deref();
        match (in_group, tag) {
            (Some(cur), Some(tag)) if cur == tag => {
                // Still in same group.
            },
            (Some(_), Some(tag)) => {
                // Switching to a different group.
                if seen.contains_key(tag) {
                    return Err(ModelError::NonContiguousFuseGroup {
                        tag: tag.to_string(),
                        first_instance: seen[tag],
                        second_start: i,
                    });
                }
                seen.entry(tag).or_insert(i);
                in_group = Some(tag);
            },
            (Some(_), None) => {
                in_group = None;
            },
            (None, Some(tag)) => {
                if seen.contains_key(tag) {
                    return Err(ModelError::NonContiguousFuseGroup {
                        tag: tag.to_string(),
                        first_instance: seen[tag],
                        second_start: i,
                    });
                }
                seen.entry(tag).or_insert(i);
                in_group = Some(tag);
            },
            (None, None) => { /* no group */ },
        }
    }

    Ok(())
}

/// Assign `fuse_group` IDs from TOML `fuse` annotations.
/// Contiguous nodes with matching `fuse` tags get the same group ID.
fn assign_toml_fuse_groups(nodes: &mut [DispatchNode], raw_nodes: &[RawNode]) {
    let mut current_tag: Option<&str> = None;
    let mut next_group_id: usize = 0;

    for (i, node) in nodes.iter_mut().enumerate() {
        let tag = raw_nodes[i].node.fuse.as_deref();
        match (current_tag, tag) {
            (None, Some(tag)) => {
                current_tag = Some(tag);
                node.fuse_group = Some(next_group_id);
            },
            (Some(cur), Some(tag)) if cur == tag => {
                node.fuse_group = Some(next_group_id);
            },
            (Some(_), maybe_tag) => {
                // Tag changed or ended.
                next_group_id += 1;
                current_tag = maybe_tag;
                if maybe_tag.is_some() {
                    node.fuse_group = Some(next_group_id);
                }
            },
            (None, None) => { /* no group */ },
        }
    }
}

/// Filter out intermediates that are both written and read entirely within
/// a single fuse group. These are handled by `dispatch_chain`'s
/// private-memory aliasing and don't need a persistent `BufferSlot`.
///
/// Uses local-scope analysis: an intermediate instance (between a write
/// and the next write to the same name) is intra-group-only if ALL its
/// readers share the same `fuse_group` as the writer.
fn filter_intra_group_intermediates(
    nodes: &[DispatchNode],
    intermediate_outputs: &[Vec<(String, usize)>],
    intermediate_inputs: &[Vec<String>],
) -> (Vec<Vec<(String, usize)>>, Vec<Vec<String>>) {
    let n = nodes.len();

    // Build ordered write/read positions per intermediate name.
    let mut writes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut reads: HashMap<String, Vec<usize>> = HashMap::new();

    for i in 0..n {
        for (name, _) in &intermediate_outputs[i] {
            writes.entry(name.clone()).or_default().push(i);
        }
        for name in &intermediate_inputs[i] {
            reads.entry(name.clone()).or_default().push(i);
        }
    }

    // For each name, for each write instance, check if all local readers
    // share the same fuse_group as the writer. If so, the name AT THAT
    // WRITE INSTANCE is intra-group-only and can be skipped from liveness.
    // We track this as a set of (name, write_position) pairs.
    let mut intra_group_instances: HashSet<(String, usize)> = HashSet::new();

    for (name, write_list) in &writes {
        let Some(read_list) = reads.get(name) else { continue };

        for (wi, &write_pos) in write_list.iter().enumerate() {
            let writer_group = nodes[write_pos].fuse_group;
            let Some(writer_group) = writer_group else { continue };

            let next_write = write_list.get(wi + 1).copied();

            // Find readers between write_pos and next_write.
            let read_start = match read_list.binary_search(&write_pos) {
                Ok(pos) => pos + 1,
                Err(pos) => pos,
            };
            let read_end = match next_write {
                Some(nw) => match read_list.binary_search(&nw) {
                    Ok(pos) => pos,
                    Err(pos) => pos,
                },
                None => read_list.len(),
            };

            let local_readers = &read_list[read_start..read_end];

            // All local readers must be in the same fuse_group as the writer.
            if local_readers.is_empty() {
                continue;
            }
            if local_readers
                .iter()
                .all(|&r| nodes[r].fuse_group == Some(writer_group))
            {
                intra_group_instances.insert((name.clone(), write_pos));
            }
        }
    }

    if intra_group_instances.is_empty() {
        return (intermediate_outputs.to_vec(), intermediate_inputs.to_vec());
    }

    // Build filtered lists: for each node, skip (name, _) entries where
    // (name, node_index) is in intra_group_instances.
    let filtered_outputs: Vec<Vec<(String, usize)>> = intermediate_outputs
        .iter()
        .enumerate()
        .map(|(i, outs)| {
            outs.iter()
                .filter(|(name, _)| !intra_group_instances.contains(&(name.clone(), i)))
                .cloned()
                .collect()
        })
        .collect();

    let filtered_inputs: Vec<Vec<String>> = intermediate_inputs
        .iter()
        .enumerate()
        .map(|(i, ins)| {
            ins.iter()
                .filter(|name| {
                    // Keep this input if its name+producer pair is NOT intra-group.
                    // We need to find the producer of this name for cursor i.
                    // The producer is the most recent write before i.
                    let producer = writes
                        .get(*name)
                        .and_then(|wl| {
                            match wl.binary_search(&i) {
                                Ok(pos) => wl.get(pos),
                                Err(pos) => wl.get(pos.checked_sub(1)?),
                            }
                        });
                    match producer {
                        Some(&p) => !intra_group_instances.contains(&((*name).clone(), p)),
                        None => true,
                    }
                })
                .cloned()
                .collect()
        })
        .collect();

    (filtered_outputs, filtered_inputs)
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

/// Maximum threads-per-group allowed by the Metal hardware on all supported
/// Apple Silicon targets. Dispatching more threads causes a Metal error.
const METAL_MAX_THREADS_PER_GROUP: usize = 1024;

/// Compute the `GridSpec` for a kernel dispatch from its mode and
/// resolved dispatch hints.
///
/// Validates `tpg` against Metal hardware limits before returning:
/// - `tpg == 0`        → error (would dispatch 0 threads, no useful work)
/// - `tpg > 1024`      → error (exceeds Metal hardware limit)
///
/// The op name is included in error messages for easy diagnosis.
fn compute_grid(
    op: &str,
    mode: &KernelMode,
    hints: &HashMap<String, u32>,
) -> Result<GridSpec, ModelError> {
    // Shared tpg validation for modes that use it.
    let validate_tpg = |tpg: usize| -> Result<(), ModelError> {
        if tpg == 0 {
            return Err(ModelError::UnsafeDispatch {
                op: op.to_string(),
                detail: "tpg=0 would dispatch 0 threads per group".into(),
            });
        }
        if tpg > METAL_MAX_THREADS_PER_GROUP {
            return Err(ModelError::UnsafeDispatch {
                op: op.to_string(),
                detail: format!(
                    "tpg={tpg} exceeds Metal limit of {METAL_MAX_THREADS_PER_GROUP}; \
                     reduce tpg or use a larger-grain kernel"
                ),
            });
        }
        Ok(())
    };

    // Minimum TPG for kernels that use simdgroup operations (n_simd = tpg/32).
    // tpg < 32 produces n_simd = 0 and an infinite GPU loop.
    const MIN_SIMD_TPG: usize = 32;

    let validate_simd_tpg = |tpg: usize| -> Result<(), ModelError> {
        if tpg < MIN_SIMD_TPG {
            return Err(ModelError::UnsafeDispatch {
                op: op.to_string(),
                detail: format!(
                    "tpg={tpg} < {MIN_SIMD_TPG} for reduction kernel — \
                     n_simd=tpg/32=0 would cause an infinite GPU loop; \
                     minimum safe TPG is {MIN_SIMD_TPG}"
                ),
            });
        }
        Ok(())
    };

    match mode {
        KernelMode::Elementwise => {
            let n = hints.get("n").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Elementwise { n })
        },
        KernelMode::Reduction => {
            let rows = hints.get("rows").copied().unwrap_or(1) as usize;
            let tpg = hints.get("tpg").copied().unwrap_or(256) as usize;
            validate_tpg(tpg)?;
            validate_simd_tpg(tpg)?;
            Ok(GridSpec::Reduction { num_rows: rows, threads_per_group: tpg })
        },
        KernelMode::Grid3D => {
            let x = hints.get("grid_x").copied().unwrap_or(1) as usize;
            let y = hints.get("grid_y").copied().unwrap_or(1) as usize;
            let z = hints.get("grid_z").copied().unwrap_or(1) as usize;
            let tpg = hints.get("tpg").copied().unwrap_or(1) as usize;
            validate_tpg(tpg)?;
            Ok(GridSpec::Grid3D { x, y, z, threads_per_group: tpg })
        },
        KernelMode::Tile2D => {
            let n = hints.get("n").copied().unwrap_or(1) as usize;
            Ok(GridSpec::Elementwise { n })
        },
        KernelMode::SimdGroup2D => {
            let rows = hints.get("rows").copied().unwrap_or(1) as usize;
            let tpg = hints.get("tpg").copied().unwrap_or(1024) as usize;
            validate_tpg(tpg)?;
            validate_simd_tpg(tpg)?;
            Ok(GridSpec::Reduction { num_rows: rows, threads_per_group: tpg })
        },
    }
}

// ── Buffer size estimation ─────────────────────────────────────────────

/// Compute the size in bytes for an intermediate output buffer.
///
/// Priority:
/// 1. `out_bytes` hint — explicit byte count (use for u32/fixed-dtype outputs).
/// 2. `out_elems` × `dtype.size_bytes()` — element count scaled by activation dtype.
/// 3. Conservative fallback: 4096 × dtype.size_bytes().
fn compute_buffer_size(hints: &HashMap<String, u32>, dtype: DType) -> usize {
    if let Some(bytes) = hints.get("out_bytes").copied() {
        return bytes as usize;
    }
    let elems = hints.get("out_elems").copied().unwrap_or(0) as usize;
    if elems > 0 {
        elems * dtype.size_bytes()
    } else {
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
            "n_kv".into(),
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

        let plan = compile(&def, &params, &reg, FusionMode::None).expect("compile");
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

        let plan = compile(&def, &params, &reg, FusionMode::None).expect("compile");
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

        let result = compile(&def, &params, &reg, FusionMode::None);
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

        let plan = compile(&def, &params, &reg, FusionMode::None).expect("compile llama_decode");
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

    // ── Fusion pass tests ─────────────────────────────────────────

    /// A minimal two-node MLP (gemv→silu) that should fuse.
    fn fusion_toml_gate_silu() -> &'static str {
        r#"
[model]
name = "fusion_test"

[params]
hidden_dim = "$hidden_dim"
ffn_dim = "$ffn_dim"

[model.layer]
name = "test"

[[layer.kernel]]
op = "gemv"
inputs = { mat = "_weight", vec = "_input" }
outputs = { out = "_gate" }
constexpr = { k = "$hidden_dim" }
dispatch = { rows = "$ffn_dim", tpg = "256", out_elems = "$ffn_dim" }

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_gate" }
outputs = { out = "_gated" }
dispatch = { n = "$ffn_dim", out_elems = "$ffn_dim" }
"#
    }

    #[test]
    fn graph_fusion_fuses_gate_silu() {
        let def: ModelDef = toml::from_str(fusion_toml_gate_silu()).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: {
                let mut m = HashMap::new();
                m.insert("hidden_dim".into(), 4096);
                m.insert("ffn_dim".into(), 14336);
                m.insert("n_layers".into(), 1);
                m
            },
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 1,
            state_keys: vec![],
        };

        let plan = compile(&def, &params, &reg, FusionMode::GraphDriven).expect("compile");
        assert_eq!(plan.nodes.len(), 2);

        // Both nodes should share the same fuse_group.
        let g0 = plan.nodes[0].fuse_group;
        let g1 = plan.nodes[1].fuse_group;
        assert!(g0.is_some(), "gemv should be in a fused group");
        assert_eq!(g0, g1, "gemv and silu should share the same fuse group");
    }

    #[test]
    fn no_fusion_mode_skips_fusion() {
        let def: ModelDef = toml::from_str(fusion_toml_gate_silu()).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: {
                let mut m = HashMap::new();
                m.insert("hidden_dim".into(), 4096);
                m.insert("ffn_dim".into(), 14336);
                m.insert("n_layers".into(), 1);
                m
            },
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 1,
            state_keys: vec![],
        };

        let plan = compile(&def, &params, &reg, FusionMode::None).expect("compile");
        assert_eq!(plan.nodes.len(), 2);
        assert!(
            plan.nodes.iter().all(|n| n.fuse_group.is_none()),
            "FusionMode::None should leave fuse_group as None"
        );
    }

    /// A three-node MLP where fan-out prevents full fusion.
    /// gate → silu → mul, but _ffn_normed feeds both gate and up gemv.
    /// The graph fusion pass should detect that gate and silu are fusible
    /// (single-use _gate), but silu and mul are not adjacent.
    fn fusion_toml_fanout() -> &'static str {
        r#"
[model]
name = "fusion_fanout"

[params]
hidden_dim = "$hidden_dim"
ffn_dim = "$ffn_dim"

[model.layer]
name = "test"

[[layer.kernel]]
op = "gemv"
inputs = { mat = "_w_gate", vec = "_normed" }
outputs = { out = "_gate" }
constexpr = { k = "$hidden_dim" }
dispatch = { rows = "$ffn_dim", tpg = "256", out_elems = "$ffn_dim" }

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_gate" }
outputs = { out = "_gated" }
dispatch = { n = "$ffn_dim", out_elems = "$ffn_dim" }

[[layer.kernel]]
op = "gemv"
inputs = { mat = "_w_up", vec = "_normed" }
outputs = { out = "_up" }
constexpr = { k = "$hidden_dim" }
dispatch = { rows = "$ffn_dim", tpg = "256", out_elems = "$ffn_dim" }
"#
    }

    #[test]
    fn graph_fusion_handles_fanout() {
        let def: ModelDef = toml::from_str(fusion_toml_fanout()).expect("parse TOML");
        let reg = make_registry();
        let params = CompileParams {
            params: {
                let mut m = HashMap::new();
                m.insert("hidden_dim".into(), 4096);
                m.insert("ffn_dim".into(), 14336);
                m.insert("n_layers".into(), 1);
                m
            },
            float_params: HashMap::new(),
            activation_dtype: DType::F32,
            n_layers: 1,
            state_keys: vec![],
        };

        let plan = compile(&def, &params, &reg, FusionMode::GraphDriven).expect("compile");
        assert_eq!(plan.nodes.len(), 3);

        // gate (node 0) → silu (node 1) should be fused (single-use _gate).
        assert!(
            plan.nodes[0].fuse_group.is_some(),
            "gate gemv should be fused with silu"
        );
        assert_eq!(
            plan.nodes[0].fuse_group, plan.nodes[1].fuse_group,
            "gate and silu should share same group"
        );

        // up gemv (node 2) reads _normed (fan-out), so it's NOT fused with
        // silu — node 1 does NOT consume _up.
        // The graph fusion should leave node 2 unfused.
        assert!(
            plan.nodes[2].fuse_group.is_none(),
            "up gemv should NOT be fused (no shared intermediate with silu)"
        );
    }

    /// TOML-driven fusion: two nodes with matching `fuse` tags should
    /// get the same fuse_group.
    #[test]
    fn toml_fuse_tag_groups_contiguous_nodes() {
        let toml_src = r#"
[model]
name = "toml_fuse"

[params]
hidden_dim = "$hidden_dim"

[model.layer]
name = "test"

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_in" }
outputs = { out = "_mid" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }
fuse = "my_group"

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_mid" }
outputs = { out = "_out" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }
fuse = "my_group"
"#;

        let def: ModelDef = toml::from_str(toml_src).expect("parse TOML");
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

        let plan =
            compile(&def, &params, &reg, FusionMode::TomlDriven).expect("compile");
        assert_eq!(plan.nodes.len(), 2);

        let g0 = plan.nodes[0].fuse_group;
        let g1 = plan.nodes[1].fuse_group;
        assert!(g0.is_some(), "first node should be in a fused group");
        assert_eq!(g0, g1, "matching fuse tags should share group ID");
    }

    /// TOML-driven: non-contiguous matching tags should error.
    #[test]
    fn toml_fuse_non_contiguous_is_error() {
        let toml_src = r#"
[model]
name = "bad_fuse"

[params]
hidden_dim = "$hidden_dim"

[model.layer]
name = "test"

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_in" }
outputs = { out = "_mid" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }
fuse = "my_group"

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_mid" }
outputs = { out = "_out" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_out" }
outputs = { out = "_final" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }
fuse = "my_group"
"#;

        let def: ModelDef = toml::from_str(toml_src).expect("parse TOML");
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

        let result = compile(&def, &params, &reg, FusionMode::TomlDriven);
        assert!(result.is_err(), "non-contiguous fuse tags should error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("non-contiguous"),
            "error should mention non-contiguous: {err}"
        );
    }

    /// GraphDriven mode ignores TOML fuse tags.
    #[test]
    fn graph_driven_ignores_toml_fuse_tags() {
        let toml_src = r#"
[model]
name = "ignore_tags"

[params]
hidden_dim = "$hidden_dim"

[model.layer]
name = "test"

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_in" }
outputs = { out = "_mid" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }
fuse = "ignored"

[[layer.kernel]]
op = "unary/silu"
inputs = { a = "_mid" }
outputs = { out = "_out" }
dispatch = { n = "$hidden_dim", out_elems = "$hidden_dim" }
"#;

        let def: ModelDef = toml::from_str(toml_src).expect("parse TOML");
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

        // GraphDriven should succeed and apply its own fusion rules,
        // ignoring the `fuse = "ignored"` tags.
        let plan = compile(&def, &params, &reg, FusionMode::GraphDriven).expect("compile");
        assert_eq!(plan.nodes.len(), 2);
        // The graph pass may fuse these (silu→silu with single-use _mid).
        // Either way, it should not error on the TOML tags.
    }
}

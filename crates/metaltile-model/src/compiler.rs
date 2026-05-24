//! Graph compiler: `ModelDef` + checkpoint params → `ExecutionPlan`.
//!
//! The compiler resolves all `$var` expressions, unrolls the layer loop,
//! validates kernel references against the registry, evaluates dispatch
//! hints, and assigns buffer slots.
//!
//! Compilation is pure CPU-side — no Metal device needed. The resulting
//! `ExecutionPlan` can be dispatched later via the executor.

use std::collections::{HashMap, HashSet};

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::context::GridSpec;
use metaltile_std::spec::effective_mode;
use tracing::info;

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
#[tracing::instrument(skip(def, p, reg), fields(n_layers = p.n_layers, n_params = p.params.len(), fusion = ?fusion_mode))]
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

    // Find the index of the first prefill_skip = true node.
    // Nodes from this index onward are skipped during non-final prefill.
    let mut prefill_node_count =
        raw_nodes.iter().position(|rn| rn.node.prefill_skip).unwrap_or(raw_nodes.len());

    // ── Step 2: Compile each RawNode → DispatchNode ────────────────
    let mut nodes: Vec<DispatchNode> = Vec::with_capacity(raw_nodes.len());
    let mut cached_kernels: Vec<Kernel> = Vec::with_capacity(raw_nodes.len());
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

        // Pre-compute grid_dims and set kernel mode at compile time
        // so the executor can use the cached kernel by reference.
        let grid_dims = grid_to_dims(&grid);
        let mut kernel = kernel;
        kernel.mode = mode;

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
            grid_dims,
            fuse_group: None,
        });
        cached_kernels.push(kernel);
    }

    // ── Step 2.5: Graph-level fusion or TOML fuse groups ─────────
    match fusion_mode {
        FusionMode::TomlDriven => {
            validate_fuse_groups(&raw_nodes)?;
            assign_toml_fuse_groups(&mut nodes, &raw_nodes);
            let total = nodes.len();
            let fused_count = nodes.iter().filter(|n| n.fuse_group.is_some()).count();
            let n_groups = nodes.iter().filter_map(|n| n.fuse_group).max().map_or(0, |m| m + 1);
            let dispatches = n_groups + (total - fused_count);
            if n_groups > 0 {
                info!(
                    "TOML fusion: {total} nodes → {dispatches} dispatches \
                     ({n_groups} fused groups, {} standalone)",
                    total - fused_count
                );
            } else {
                info!("no TOML fuse tags: {total} nodes → {total} dispatches");
            }
        },
        FusionMode::GraphDriven => {
            let total = nodes.len();
            let (n_fused_groups, n_unfused) = fuse_dispatch_nodes(&mut nodes);
            let dispatches = n_fused_groups + n_unfused;
            if n_fused_groups > 0 {
                info!(
                    "graph fusion: {total} nodes → {dispatches} dispatches \
                     ({n_fused_groups} fused groups, {n_unfused} standalone)"
                );
            }
        },
        FusionMode::None => {
            info!("no fusion: {} nodes → {} dispatches", nodes.len(), nodes.len());
        },
    }

    // ── Step 2.6: Kernel-body fusion synthesis ───────────────────
    // For compatible fuse groups, replace N DispatchNodes with a single
    // synthesized "host kernel" that chains the individual kernels via
    // Op::KernelCall.  KernelInlinePass (first pass in standard_pipeline)
    // splices the callees inline at MSL generation time, eliminating
    // intermediate buffer round-trips for intra-group tensors.
    //
    // Must run BEFORE slot assignment so liveness analysis doesn't allocate
    // slots for eliminated intra-group intermediates.
    if fusion_mode != FusionMode::None {
        crate::fuse_group::synthesize_fuse_groups(
            &mut nodes,
            &mut cached_kernels,
            &mut intermediate_outputs,
            &mut intermediate_inputs,
            &mut prefill_node_count,
            p.activation_dtype,
        );
    }

    // ── Step 3: Liveness analysis → slot assignment ────────────────
    // name_to_slot is the canonical map built during assignment — it preserves
    // all tenant names even when slots are reused (slot.name is overwritten on
    // reuse, so rebuilding from the slot vector would lose earlier tenant names).
    //
    // All intermediates go through liveness + slot_data, even those within
    // fused groups.  Fusion reduces dispatch_chain calls but doesn't bypass
    // the proven slot_data dataflow — this avoids correctness issues from
    // external buffer aliasing across Metal barriers.
    let (slots, name_to_slot) =
        assign_slots(nodes.len(), &intermediate_outputs, &intermediate_inputs);

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

    let single_dispatch = fusion_mode != FusionMode::None;
    Ok(ExecutionPlan {
        nodes,
        slots,
        output_slot,
        n_layers,
        cached_kernels,
        single_dispatch,
        prefill_node_count,
    })
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
    for v in writes.values_mut() {
        v.sort_unstable();
    }
    for v in reads.values_mut() {
        v.sort_unstable();
    }

    // ── Phase 2: Find maximal fusible chains (backward scan) ─────
    let mut fused: HashSet<usize> = HashSet::default();
    let mut chains: Vec<Vec<usize>> = Vec::new();

    for i in (0..n).rev() {
        if fused.contains(&i) {
            continue;
        }

        // `chain` is built in reverse order: chain[0] = anchor (rightmost),
        // chain.last() = current leftmost head.  Reversed before storing.
        let mut chain: Vec<usize> = vec![i];

        loop {
            let head = *chain.last().unwrap();
            let Some(pred) = head.checked_sub(1) else { break };
            if fused.contains(&pred) {
                break;
            }
            if chain.len() >= MAX_FUSED_PER_CHAIN {
                break;
            }
            if !can_prepend_to_chain(pred, &chain, nodes, &writes, &reads, &defs) {
                break;
            }
            chain.push(pred);
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

    // ── Phase 4: Merge adjacent parallel-Reduction pairs ────────
    // Scan adjacent fuse groups and merge [R_a, E_a] + [R_b, E_b]
    // when they form a diamond dependency: R_a and R_b are parallel
    // identically-spec'd Reductions, and E_b reads from E_a.
    merge_parallel_reduction_pairs(nodes, &writes, &reads);

    (chains.len(), n_unfused)
}

/// Merge adjacent same-spec parallel-Reduction fuse groups when they form a
/// diamond dependency: two parallel [R→E] pairs where the second E reads
/// from the first E.
///
/// Scans pairs of consecutive fuse groups (group_id, group_id+1) and merges
/// them when:
/// - Both groups have exactly 2 nodes: [R, E] in execution order.
/// - Both R nodes have identical Reduction specs (num_rows, threads_per_group).
/// - R_b does NOT read any tensor written by R_a or E_a (parallel, not dependent).
/// - E_b reads at least one tensor written by E_a (diamond dependency).
/// - No unfused nodes exist between the two groups.
///
/// When conditions are met, group_id+1's nodes are reassigned to group_id.
fn merge_parallel_reduction_pairs(
    nodes: &mut [DispatchNode],
    _writes: &HashMap<String, Vec<usize>>,
    _reads: &HashMap<String, Vec<usize>>,
) {
    use metaltile_runtime::context::GridSpec;

    if nodes.len() < 4 {
        return;
    }

    // Collect groups: group_id → Vec<node_index>
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let Some(gid) = node.fuse_group {
            groups.entry(gid).or_default().push(i);
        }
    }

    if groups.len() < 2 {
        return;
    }

    // Sort groups by their first node's execution position.
    let mut sorted_groups: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    sorted_groups.sort_unstable_by_key(|(_, indices)| indices[0]);

    let n_groups = sorted_groups.len();

    // Process adjacent pairs in execution order.
    let mut merged_set: HashSet<usize> = HashSet::new(); // set of first-node indices that are merged
    for gi in 0..n_groups.saturating_sub(1) {
        let (gid_a, group_a) = (&sorted_groups[gi].0, &sorted_groups[gi].1);
        let (_gid_b, group_b) = (&sorted_groups[gi + 1].0, &sorted_groups[gi + 1].1);

        // Both groups must have exactly 2 nodes: [R, E].
        if group_a.len() != 2 || group_b.len() != 2 {
            continue;
        }

        let (ra_idx, ea_idx) = (group_a[0], group_a[1]);
        let (rb_idx, eb_idx) = (group_b[0], group_b[1]);

        // Groups must be adjacent: no unfused nodes between.
        if ea_idx + 1 != rb_idx {
            continue;
        }

        // Skip if either group was already merged by another pair.
        if merged_set.contains(&ra_idx) || merged_set.contains(&rb_idx) {
            continue;
        }

        let ra = &nodes[ra_idx];
        let _ea = &nodes[ea_idx];
        let rb = &nodes[rb_idx];
        let _eb = &nodes[eb_idx];

        // Both first nodes must be Reduction.
        let GridSpec::Reduction { num_rows: rows_a, threads_per_group: tpg_a } = &ra.grid else {
            continue;
        };
        let GridSpec::Reduction { num_rows: rows_b, threads_per_group: tpg_b } = &rb.grid else {
            continue;
        };

        // Identical Reduction specs.
        if rows_a != rows_b || tpg_a != tpg_b {
            continue;
        }

        // R_b must NOT read any tensor written by R_a or E_a (parallel, not dependent).
        let tensors_written_by_a: HashSet<&str> = [ra_idx, ea_idx]
            .iter()
            .flat_map(|&idx| {
                nodes[idx].output_bindings.iter().filter_map(|(_, sr)| match sr {
                    SlotRef::Weight(name) if name.starts_with('_') => Some(name.as_str()),
                    _ => None,
                })
            })
            .collect();

        let rb_reads_from_a = nodes[rb_idx].input_bindings.iter().any(|(_, sr)| match sr {
            SlotRef::Weight(name) => tensors_written_by_a.contains(name.as_str()),
            _ => false,
        });
        if rb_reads_from_a {
            continue;
        }

        // E_b must read at least one tensor written by E_a.
        let ea_writes: HashSet<&str> = nodes[ea_idx]
            .output_bindings
            .iter()
            .filter_map(|(_, sr)| match sr {
                SlotRef::Weight(name) if name.starts_with('_') => Some(name.as_str()),
                _ => None,
            })
            .collect();

        let eb_reads_ea = nodes[eb_idx].input_bindings.iter().any(|(_, sr)| match sr {
            SlotRef::Weight(name) => ea_writes.contains(name.as_str()),
            _ => false,
        });
        if !eb_reads_ea {
            continue;
        }

        // Merge group B's nodes into group A.
        for &idx in group_b {
            nodes[idx].fuse_group = Some(*gid_a);
        }
        merged_set.insert(ra_idx);
        merged_set.insert(rb_idx);
    }
}

/// Returns `true` when every local reader of `name` (in the scope between
/// `writer_idx` and the next write to the same name) is a member of `group`.
///
/// "Local readers" are those at positions strictly after `writer_idx` and
/// before the next write to `name`.  Returns `true` for a written-but-never-
/// read name (dead intermediate — safe to fuse away).
fn is_local_readers_all_in_set(
    name: &str,
    writer_idx: usize,
    group: &HashSet<usize>,
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
) -> bool {
    let Some(write_list) = writes.get(name) else { return false };
    let Ok(write_pos) = write_list.binary_search(&writer_idx) else { return false };
    let next_write = write_list.get(write_pos + 1).copied();

    let Some(read_list) = reads.get(name) else {
        // Written but never read — can be eliminated as a dead intermediate.
        return true;
    };

    let read_start = read_list.partition_point(|&r| r <= writer_idx);
    let read_end = match next_write {
        Some(nw) => read_list.partition_point(|&r| r < nw),
        None => read_list.len(),
    };

    let local_readers = &read_list[read_start..read_end];
    !local_readers.is_empty() && local_readers.iter().all(|r| group.contains(r))
}

/// Returns `true` when `candidate` is grid-compatible with all nodes already
/// in `chain`.
///
/// Supports two synthesis paths:
///
/// **Path B (same-grid)**: all nodes share an identical grid spec:
/// - All `Reduction` nodes share `(num_rows, threads_per_group)`.
/// - All `Elementwise` nodes share `n`.
///
/// **Pattern 1 (epilogue chain)**: one "host" node (Reduction or non-flat
/// Grid3D) followed by N "follower" nodes (Elementwise or flat-1D Grid3D).
/// Each follower's element count must be divisible by the host's output count.
/// The `has_shared` check in `can_prepend_to_chain` filters out orderings
/// that are data-flow invalid (e.g. Elementwise before its Reduction host).
///
/// `Tile2D` and `SimdGroup2D` are always incompatible.
fn chain_grid_compatible(
    candidate: &DispatchNode,
    chain: &[usize],
    nodes: &[DispatchNode],
) -> bool {
    // Classify a node as "host" (Reduction or non-flat Grid3D) or "follower"
    // (Elementwise or flat-1D Grid3D).  Returns `(is_host, output_count)`.
    // Returns `None` for unsupported modes (Tile2D, SimdGroup2D, etc.).
    fn node_class(mode: &KernelMode, grid: &GridSpec) -> Option<(bool, usize)> {
        match (mode, grid) {
            (KernelMode::Reduction, GridSpec::Reduction { num_rows, .. }) =>
                Some((true, *num_rows)),
            (KernelMode::Grid3D, GridSpec::Grid3D { x, y, z, .. }) if !(*y == 1 && *z == 1) =>
                Some((true, x * y * z)),
            (KernelMode::Grid3D, GridSpec::Grid3D { x, y: 1, z: 1, .. }) => Some((false, *x)),
            (KernelMode::Elementwise, GridSpec::Elementwise { n }) => Some((false, *n)),
            _ => None,
        }
    }

    let Some((cand_is_host, cand_n)) = node_class(&candidate.mode, &candidate.grid) else {
        return false;
    };

    // Scan existing chain members to extract:
    //   reduction_spec — for Path B same-Reduction constraint
    //   host_out       — output count of the host node (Reduction or Grid3D)
    //   follower_n     — element count shared by follower nodes (no-host case)
    let mut reduction_spec: Option<(usize, usize)> = None;
    let mut host_out: Option<usize> = None;
    let mut follower_n: Option<usize> = None;

    for &cidx in chain {
        let n = &nodes[cidx];
        let Some((is_host, count)) = node_class(&n.mode, &n.grid) else {
            return false;
        };
        if is_host {
            // Accumulate Reduction spec for Path B.
            if let (KernelMode::Reduction, GridSpec::Reduction { num_rows, threads_per_group }) =
                (&n.mode, &n.grid)
            {
                match reduction_spec {
                    None => reduction_spec = Some((*num_rows, *threads_per_group)),
                    Some((r, t)) if r == *num_rows && t == *threads_per_group => {},
                    _ => return false,
                }
            }
            // All host nodes in a chain must agree on output count.
            match host_out {
                None => host_out = Some(count),
                Some(h) if h == count => {},
                _ => return false,
            }
        } else {
            // Follower: if a host is already present it must divide evenly.
            if let Some(hoc) = host_out {
                if count % hoc != 0 {
                    return false;
                }
            } else {
                // No host yet — Path B same-n followers.
                match follower_n {
                    None => follower_n = Some(count),
                    Some(fn_) if fn_ == count => {},
                    _ => return false,
                }
            }
        }
    }

    // Check whether the candidate fits the chain so far.
    if cand_is_host {
        // Adding a host to the chain.
        if let Some(fn_) = follower_n {
            // Chain currently has only followers; host must divide them evenly.
            if fn_ % cand_n != 0 {
                return false;
            }
        }
        if let Some(hoc) = host_out {
            // Chain already has a host; output counts must match.
            if hoc != cand_n {
                return false;
            }
        }
        // Path B same-Reduction: spec must match.
        if let (KernelMode::Reduction, GridSpec::Reduction { num_rows, threads_per_group }) =
            (&candidate.mode, &candidate.grid)
            && let Some((r, t)) = reduction_spec
            && (r != *num_rows || t != *threads_per_group)
        {
            return false;
        }
    } else {
        // Adding a follower to the chain.
        //
        // Pattern 1 ordering guard: a follower must come AFTER its host in
        // execution.  The chain is built backward (earlier nodes prepended),
        // so `chain.last()` is the current leftmost/earliest node.  If that
        // node is a host (Reduction or non-flat Grid3D) and we prepend a
        // follower further left, the resulting order is [follower, host, …] —
        // Pattern 1 cannot synthesize this.  Reject here so the backward scan
        // can instead find valid producer-consumer pairs (e.g. up→mul).
        if let Some(&leftmost) = chain.last() {
            let lm = &nodes[leftmost];
            if let Some((lm_is_host, _)) = node_class(&lm.mode, &lm.grid)
                && lm_is_host
            {
                return false;
            }
        }

        if let Some(hoc) = host_out {
            // Host present: follower count must be divisible by host output count.
            if cand_n % hoc != 0 {
                return false;
            }
        } else {
            // No host: Path B same-n followers.
            match follower_n {
                None => {},
                Some(fn_) if fn_ == cand_n => {},
                _ => return false,
            }
        }
    }

    true
}

/// Returns `true` when `pred` can be prepended to `chain`, extending it
/// leftward by one position.
///
/// Requirements:
/// 1. `pred == chain.last() - 1` — must be the immediate predecessor
///    (contiguity is required for the splice in `synthesize_fuse_groups`).
/// 2. `pred` is grid-compatible with every node already in `chain`.
/// 3. At least one intermediate written by `pred` is read exclusively within
///    `chain` — it will become an intra-group tensor, eliminating the global-
///    memory round-trip.
///
/// This handles diamond fan-out patterns (e.g. `_ffn_normed` flowing into
/// both `gemv_gate` and `gemv_up`): the candidate's output only needs to be
/// consumed by *some* chain node, not necessarily the immediate chain head.
/// Intermediates written by `pred` that are read both inside and outside
/// `chain` remain as external outputs of the fused node — they don't block
/// fusion.
fn can_prepend_to_chain(
    pred: usize,
    chain: &[usize],
    nodes: &[DispatchNode],
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
    defs: &[Vec<String>],
) -> bool {
    let head = *chain.last().unwrap();
    if pred + 1 != head {
        return false;
    }

    if !chain_grid_compatible(&nodes[pred], chain, nodes) {
        return false;
    }

    let chain_set: HashSet<usize> = chain.iter().copied().collect();

    let mut has_shared = false;
    for name in &defs[pred] {
        let read_by_chain = chain.iter().any(|&cidx| {
            nodes[cidx]
                .input_bindings
                .iter()
                .any(|(_, sr)| matches!(sr, SlotRef::Weight(n) if n == name))
        });
        if !read_by_chain {
            continue;
        }
        // Name is read by at least one chain node.  If ALL its local readers
        // are within the chain it becomes a fully-intra tensor (eliminated).
        // If some readers are outside the chain, it stays as an external
        // output — that's fine, it doesn't prevent fusion.
        if is_local_readers_all_in_set(name, pred, &chain_set, writes, reads) {
            has_shared = true;
        }
    }

    has_shared
}

// ── TOML fuse tag handling ─────────────────────────────────────────────

/// Validate that TOML `fuse` tags are contiguous.
/// Returns the effective fuse tag for a raw node, scoped by layer index.
///
/// Per-layer kernels with `fuse = "q_chain"` in layer 0 get effective tag
/// `"q_chain.0"`, in layer 1 `"q_chain.1"`, etc. This prevents the
/// validator from flagging same-named groups in different layers as
/// non-contiguous while still allowing cross-layer re-use of tag names.
fn effective_fuse_tag(raw: &RawNode) -> Option<String> {
    let tag = raw.node.fuse.as_deref()?;
    if let Some(layer_idx) = raw.layer_idx {
        Some(format!("{tag}.{layer_idx}"))
    } else {
        Some(tag.to_string())
    }
}

fn validate_fuse_groups(raw_nodes: &[RawNode]) -> Result<(), ModelError> {
    let mut seen: HashMap<String, usize> = HashMap::default();
    let mut in_group: Option<String> = None;

    for (i, raw) in raw_nodes.iter().enumerate() {
        let tag = effective_fuse_tag(raw);
        match (in_group.as_deref(), tag.as_deref()) {
            (Some(cur), Some(t)) if cur == t => {
                // Still in same group.
            },
            (Some(_), Some(t)) => {
                if seen.contains_key(t) {
                    return Err(ModelError::NonContiguousFuseGroup {
                        tag: raw.node.fuse.clone().unwrap_or_default(),
                        first_instance: seen[t],
                        second_start: i,
                    });
                }
                seen.entry(t.to_string()).or_insert(i);
                in_group = tag;
            },
            (Some(_), None) => {
                in_group = None;
            },
            (None, Some(t)) => {
                if seen.contains_key(t) {
                    return Err(ModelError::NonContiguousFuseGroup {
                        tag: raw.node.fuse.clone().unwrap_or_default(),
                        first_instance: seen[t],
                        second_start: i,
                    });
                }
                seen.entry(t.to_string()).or_insert(i);
                in_group = tag;
            },
            (None, None) => { /* no group */ },
        }
    }

    Ok(())
}

/// Assign `fuse_group` IDs from TOML `fuse` annotations.
/// Contiguous nodes with matching effective tags get the same group ID.
fn assign_toml_fuse_groups(nodes: &mut [DispatchNode], raw_nodes: &[RawNode]) {
    let mut current_tag: Option<String> = None;
    let mut next_group_id: usize = 0;

    for (i, node) in nodes.iter_mut().enumerate() {
        let tag = effective_fuse_tag(&raw_nodes[i]);
        match (current_tag.as_deref(), tag.as_deref()) {
            (None, Some(_)) => {
                current_tag = tag;
                node.fuse_group = Some(next_group_id);
            },
            (Some(cur), Some(t)) if cur == t => {
                node.fuse_group = Some(next_group_id);
            },
            (Some(_), _) => {
                next_group_id += 1;
                let has_tag = tag.is_some();
                current_tag = tag;
                if has_tag {
                    node.fuse_group = Some(next_group_id);
                }
            },
            (None, None) => { /* no group */ },
        }
    }
}

/// Convert a `GridSpec` to `(grid_groups, threads_per_group)` dimensions.
/// Called once per node at compile time; result is stored in `DispatchNode.grid_dims`.
pub(crate) fn grid_to_dims(grid: &GridSpec) -> ([usize; 3], [usize; 3]) {
    match grid {
        GridSpec::Elementwise { n } => {
            let tpg = 256usize;
            ([n.div_ceil(tpg), 1, 1], [tpg, 1, 1])
        },
        GridSpec::Reduction { num_rows, threads_per_group } =>
            ([*num_rows, 1, 1], [*threads_per_group, 1, 1]),
        GridSpec::Grid3D { x, y, z, threads_per_group } =>
            ([*x, *y, *z], [*threads_per_group, 1, 1]),
    }
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
    if elems > 0 { elems * dtype.size_bytes() } else { 4096 * dtype.size_bytes() }
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
        assert!(plan.nodes[0].fuse_group.is_some(), "gate gemv should be fused with silu");
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

        let plan = compile(&def, &params, &reg, FusionMode::TomlDriven).expect("compile");
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
        assert!(err.contains("non-contiguous"), "error should mention non-contiguous: {err}");
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

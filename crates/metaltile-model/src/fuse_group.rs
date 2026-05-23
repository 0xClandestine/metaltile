//! TOML fuse-group kernel body synthesis.
//!
//! Synthesizes a single "host kernel" for each compatible fuse group.
//! Two synthesis paths are attempted in order:
//!
//! ## Path A — Pattern 1 direct IR synthesis (general N-node linear chain)
//!
//! Handles any N ≥ 2 node chain where:
//! - `node[0]` is `Reduction` or `Grid3D` (drives dispatch)
//! - each `node[i]` (i ≥ 1) is `Elementwise` or flat-1D `Grid3D`, has a flat body,
//!   and reads exactly one intra tensor produced by `node[i-1]`
//! - each epilogue node's element count `n` is a multiple of `node[0]`'s `num_rows`
//!
//! Examples: `gemv/bm4 → binary/add` (`attn_out`/`ffn_out`),
//! `gemv/bm4 → kv_cache_update` (`v_chain`),
//! `rope_llama → kv_cache_update` (`k_chain` sub-pair).
//!
//! The first node's IR body is cloned into the host kernel.  Each subsequent
//! node's flat body is injected in place of the preceding node's intra-output
//! stores: the stored scalar becomes `intra_vid` and the store's index VID
//! becomes the epilogue's `program_id(0)`.  Intermediate intra tensors are
//! eliminated from global memory entirely.
//!
//! ## Path B — `KernelCall`-chain synthesis (same-grid groups)
//!
//! For groups where all nodes share the same grid dimensions (same-grid
//! `Elementwise` or same-spec `Reduction`, or mixed `R+E` with `n == num_rows`),
//! a host kernel with one `Op::KernelCall` per group node is built.
//! `KernelInlinePass` resolves the calls into a flat kernel at MSL generation time.
//!
//! ## Common steps (both paths)
//!
//! 1. **Intra-group tensor detection** — an intermediate tensor is "pure intra"
//!    when it is written by a group node and ALL reads between that write and
//!    the next write to the same name come from other group nodes.  Intra
//!    tensors flow as computed scalars instead of round-tripping through
//!    global memory.
//!
//! 2. **Node replacement** — the N group nodes are replaced by 1 fused node
//!    in `nodes` and `cached_kernels`; intermediate tracking is updated to
//!    exclude eliminated intra tensors.

use std::collections::{BTreeMap, HashMap, HashSet};

use metaltile_core::{
    DType,
    constexpr::ConstExpr,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelCallArg, KernelMode, Op, Param, ParamKind, ValueId},
    shape::Shape,
};
use metaltile_runtime::context::GridSpec;
use tracing::debug;

use crate::{
    ConstexprValue,
    compiler::grid_to_dims,
    plan::{DispatchNode, SlotRef},
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Placeholder `kernel_ir` function for synthesized fused nodes.
///
/// The executor uses `cached_kernels[i]` directly and never calls `kernel_ir`
/// at dispatch time.  This placeholder satisfies the `fn(DType) -> Kernel`
/// field type requirement on `DispatchNode`.
pub fn fused_kernel_placeholder(_dtype: DType) -> Kernel { Kernel::new("fused_placeholder") }

/// Synthesize fused kernels for all compatible fuse groups in place.
///
/// Called in `compiler::compile` between step 2.5 (fuse-group assignment)
/// and step 3 (liveness / slot assignment).  Replaces each compatible N-node
/// fuse group with a single synthesized node whose kernel body chains the
/// individual kernels via `Op::KernelCall`.
///
/// Groups are processed from the last to the first so that the `Vec::splice`
/// calls don't invalidate earlier node indices.
///
/// `prefill_node_count` is adjusted downward by the number of nodes removed
/// before the original prefill boundary.
pub fn synthesize_fuse_groups(
    nodes: &mut Vec<DispatchNode>,
    cached_kernels: &mut Vec<Kernel>,
    intermediate_outputs: &mut Vec<Vec<(String, usize)>>,
    intermediate_inputs: &mut Vec<Vec<String>>,
    prefill_node_count: &mut usize,
    dtype: DType,
) {
    let groups = collect_groups(nodes);
    if groups.is_empty() {
        return;
    }

    let (writes, reads) = build_write_read_maps(nodes);

    // group_node_set: original node index → group_id (for intra-tensor analysis).
    let group_node_set: HashMap<usize, usize> =
        nodes.iter().enumerate().filter_map(|(i, n)| n.fuse_group.map(|g| (i, g))).collect();

    // Sort groups by start index descending so we splice from end to start.
    let mut group_list: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    group_list.sort_unstable_by(|a, b| b.1[0].cmp(&a.1[0]));

    for (group_id, node_indices) in &group_list {
        let group_nodes: Vec<&DispatchNode> = node_indices.iter().map(|&i| &nodes[i]).collect();
        let group_cached: Vec<&Kernel> = node_indices.iter().map(|&i| &cached_kernels[i]).collect();

        // Intra-tensor analysis is needed by all synthesis paths.
        let intra_tensors =
            find_pure_intra_tensors(node_indices, &group_node_set, &writes, &reads);

        let synthesis_result = synthesize_pattern3(
            &group_nodes,
            &group_cached,
            &intra_tensors,
            dtype,
            *group_id,
        )
        .or_else(|| {
            synthesize_pattern1(
                &group_nodes,
                &group_cached,
                &intra_tensors,
                dtype,
                *group_id,
            )
        })
        .or_else(|| {
            let Some((fused_mode, fused_grid)) = check_grid_compatibility(&group_nodes) else {
                debug!("fuse group {group_id}: incompatible grids, skipping synthesis");
                return None;
            };
            build_fused_group(
                &group_nodes,
                &group_cached,
                node_indices,
                &intra_tensors,
                dtype,
                fused_mode,
                fused_grid,
                *group_id,
            )
        });

        let Some((fused_node, fused_kernel)) = synthesis_result else {
            // Full synthesis failed.  Try sub-pair decomposition: find consecutive
            // 2-node (R→E) pairs within the group that satisfy Pattern 1 and
            // synthesize each independently.  This handles the `ffn_act` diamond
            // (gemv_gate→silu, gemv_up→mul) without requiring a custom kernel.
            let subpairs = try_subpair_decomposition(
                &group_nodes,
                &group_cached,
                node_indices,
                &writes,
                &reads,
                dtype,
                *group_id,
            );

            if subpairs.is_empty() {
                debug!("fuse group {group_id}: no synthesis path succeeded, skipping");
                continue;
            }

            // Drop borrows of nodes/cached_kernels before the mutable splice below.
            drop(group_nodes);
            drop(group_cached);

            apply_subpair_decomposition(
                subpairs,
                node_indices,
                nodes,
                cached_kernels,
                intermediate_outputs,
                intermediate_inputs,
                &writes,
                &reads,
                prefill_node_count,
                *group_id,
            );
            continue;
        };

        // Fused intermediate tracking: drop intra tensors, deduplicate.
        let fused_int_out: Vec<(String, usize)> = {
            let mut seen = HashSet::new();
            node_indices
                .iter()
                .flat_map(|&ni| intermediate_outputs[ni].iter().cloned())
                .filter(|(name, _)| !intra_tensors.contains(name.as_str()))
                .filter(|(name, _)| seen.insert(name.clone()))
                .collect()
        };
        let fused_int_in: Vec<String> = {
            let mut seen = HashSet::new();
            node_indices
                .iter()
                .flat_map(|&ni| intermediate_inputs[ni].iter().cloned())
                .filter(|name| !intra_tensors.contains(name.as_str()))
                .filter(|name| seen.insert(name.clone()))
                .collect()
        };

        let start = node_indices[0];
        let end = *node_indices.last().unwrap() + 1;
        let n_removed = end - start - 1; // nodes collapsed: N replaced by 1

        let elim_count = intra_tensors.len();
        debug!(
            "fuse group {group_id}: {} nodes → 1, eliminated {elim_count} intermediates",
            node_indices.len()
        );

        nodes.splice(start..end, std::iter::once(fused_node));
        cached_kernels.splice(start..end, std::iter::once(fused_kernel));
        intermediate_outputs.splice(start..end, std::iter::once(fused_int_out));
        intermediate_inputs.splice(start..end, std::iter::once(fused_int_in));

        // Adjust the prefill boundary: if nodes were removed before it, shift down.
        if start < *prefill_node_count {
            let removed_before = n_removed.min(*prefill_node_count - start);
            *prefill_node_count -= removed_before;
        }
    }
}

// ---------------------------------------------------------------------------
// Group collection
// ---------------------------------------------------------------------------

fn collect_groups(nodes: &[DispatchNode]) -> HashMap<usize, Vec<usize>> {
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let Some(gid) = node.fuse_group {
            groups.entry(gid).or_default().push(i);
        }
    }
    groups
}

// ---------------------------------------------------------------------------
// Write/read position maps for intermediate tensors
// ---------------------------------------------------------------------------

fn build_write_read_maps(
    nodes: &[DispatchNode],
) -> (HashMap<String, Vec<usize>>, HashMap<String, Vec<usize>>) {
    let mut writes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut reads: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        for (_, slot_ref) in &node.input_bindings {
            if let SlotRef::Weight(name) = slot_ref
                && name.starts_with('_')
            {
                reads.entry(name.clone()).or_default().push(i);
            }
        }
        for (_, slot_ref) in &node.output_bindings {
            if let SlotRef::Weight(name) = slot_ref
                && name.starts_with('_')
            {
                writes.entry(name.clone()).or_default().push(i);
            }
        }
    }

    (writes, reads)
}

// ---------------------------------------------------------------------------
// Grid compatibility check
// ---------------------------------------------------------------------------

/// Returns `Some((fused_mode, fused_grid))` when all nodes in the group can be
/// fused, `None` when any node is incompatible.
///
/// Rules:
/// - Only `Elementwise` and `Reduction` modes are supported.  `Grid3D`,
///   `Tile2D`, and `SimdGroup2D` are always incompatible.
/// - All `Reduction` nodes must share the same `(num_rows, threads_per_group)`.
/// - All `Elementwise` nodes must share the same `n`.
/// - In a mixed group, `n` (Elementwise) must equal `num_rows` (Reduction).
fn check_grid_compatibility(nodes: &[&DispatchNode]) -> Option<(KernelMode, GridSpec)> {
    let mut reduction_spec: Option<(usize, usize)> = None; // (num_rows, tpg)
    let mut elementwise_n: Option<usize> = None;

    for node in nodes {
        match (&node.mode, &node.grid) {
            (KernelMode::Reduction, GridSpec::Reduction { num_rows, threads_per_group }) => {
                match reduction_spec {
                    None => reduction_spec = Some((*num_rows, *threads_per_group)),
                    Some((r, t)) if r == *num_rows && t == *threads_per_group => {},
                    _ => return None, // mismatched reduction dims
                }
            },
            (KernelMode::Elementwise, GridSpec::Elementwise { n }) => {
                match elementwise_n {
                    None => elementwise_n = Some(*n),
                    Some(en) if en == *n => {},
                    _ => return None, // mismatched elementwise dims
                }
            },
            _ => return None, // Grid3D, Tile2D, SimdGroup2D or unexpected combo
        }
    }

    match (reduction_spec, elementwise_n) {
        // Mixed: Elementwise n must equal Reduction num_rows.
        (Some((num_rows, _)), Some(n)) if n != num_rows => None,
        // Reduction wins in mixed groups.
        (Some((num_rows, tpg)), _) =>
            Some((KernelMode::Reduction, GridSpec::Reduction { num_rows, threads_per_group: tpg })),
        (None, Some(n)) => Some((KernelMode::Elementwise, GridSpec::Elementwise { n })),
        (None, None) => None, // empty group
    }
}

// ---------------------------------------------------------------------------
// Intra-group tensor detection
// ---------------------------------------------------------------------------

/// Returns the set of intermediate tensor names that are "pure intra-group":
/// written by a group node and read ONLY by group nodes between that write
/// and the next write to the same name.
///
/// These tensors can be eliminated from global memory — they flow as scalar
/// `Value` args between `KernelCall` ops.
fn find_pure_intra_tensors(
    node_indices: &[usize],
    _group_node_set: &HashMap<usize, usize>,
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
) -> HashSet<String> {
    let group_set: HashSet<usize> = node_indices.iter().copied().collect();
    let mut intra = HashSet::new();

    for (tensor_name, write_positions) in writes {
        for (pos, &writer_idx) in write_positions.iter().enumerate() {
            if !group_set.contains(&writer_idx) {
                continue;
            }

            let next_write = write_positions.get(pos + 1).copied();

            let Some(read_positions) = reads.get(tensor_name) else {
                // Nobody reads this tensor at all → intra (eliminates dead slot).
                intra.insert(tensor_name.clone());
                continue;
            };

            // Reads strictly after writer_idx and before next_write.
            let read_start = read_positions.partition_point(|&r| r <= writer_idx);
            let read_end = match next_write {
                Some(nw) => read_positions.partition_point(|&r| r < nw),
                None => read_positions.len(),
            };

            let local_reads = &read_positions[read_start..read_end];

            // Tensor is intra when ALL local readers are within the group.
            if !local_reads.is_empty() && local_reads.iter().all(|r| group_set.contains(r)) {
                intra.insert(tensor_name.clone());
            }
        }
    }

    intra
}

// ---------------------------------------------------------------------------
// Host kernel synthesis
// ---------------------------------------------------------------------------

/// Extract the string name from a `SlotRef` (only valid before slot assignment).
fn slot_ref_name(slot_ref: &SlotRef) -> Option<String> {
    match slot_ref {
        SlotRef::Weight(name) => Some(name.clone()),
        SlotRef::State(name) => Some(name.clone()),
        SlotRef::Slot(_) => None, // should not appear before assign_slots
    }
}

/// Create a valid Metal identifier from a tensor name:
/// - Strip leading `_`
/// - Replace `.`, `/`, `-` and other non-alnum chars with `_`
/// - Prefix with `p_` to avoid Metal keyword collisions
///
/// If the sanitized name collides with an existing `used` name, append `_N`.
fn make_param_name(tensor_name: &str, used: &mut HashSet<String>) -> String {
    let no_leading = tensor_name.trim_start_matches('_');
    let base: String = no_leading
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let candidate = format!("p_{base}");
    if used.insert(candidate.clone()) {
        return candidate;
    }
    let mut i = 2u32;
    loop {
        let c = format!("p_{base}_{i}");
        if used.insert(c.clone()) {
            return c;
        }
        i += 1;
    }
}

/// Build the synthesized fused `DispatchNode` and `Kernel` for a compatible group.
///
/// Returns `None` if the group cannot be synthesized (e.g., a callee contains
/// nested `KernelCall`, a param lookup fails, or a `SlotRef::Slot` is seen
/// before slot assignment).
fn build_fused_group(
    group_nodes: &[&DispatchNode],
    group_cached: &[&Kernel],
    _node_indices: &[usize],
    intra_tensors: &HashSet<String>,
    dtype: DType,
    fused_mode: KernelMode,
    fused_grid: GridSpec,
    group_id: usize,
) -> Option<(DispatchNode, Kernel)> {
    // Safety: reject if any callee has nested KernelCall (would fail the
    // KernelInlinePass assertion and can't be inlined).
    for callee in group_cached {
        if callee.body.ops.iter().any(|op| matches!(op, Op::KernelCall { .. })) {
            return None;
        }
    }

    // Reject any callee that uses Strided params (requires extra shape/strides
    // buffers that the synthesis doesn't yet model).
    for callee in group_cached {
        if callee.params.iter().any(|p| p.kind == ParamKind::Strided) {
            return None;
        }
    }

    let kernel_name: &'static str = Box::leak(format!("fused_group_{group_id}").into_boxed_str());

    let mut host = Kernel::new(kernel_name);
    host.mode = fused_mode;

    // Fused node bindings.
    let mut fused_input_bindings: Vec<(String, SlotRef)> = Vec::new();
    let mut fused_output_bindings: Vec<(String, SlotRef)> = Vec::new();
    let mut fused_cexprs: Vec<(String, ConstexprValue)> = Vec::new();

    // Deduplication: original tensor name → host param name.
    let mut input_host_names: HashMap<String, String> = HashMap::new();
    let mut output_host_names: HashMap<String, String> = HashMap::new();
    // Tracks all declared host param/constexpr names to avoid collisions.
    let mut used_param_names: HashSet<String> = HashSet::new();

    // Host kernel body.
    let mut next_vid: u32 = 0;
    let pid_vid = ValueId::new(next_vid);
    next_vid += 1;
    // Emit ProgramId(0) so KernelInlinePass builds a non-empty caller_pids map.
    host.body.push_op(Op::ProgramId { axis: 0 }, pid_vid);

    // intra_tensor_name → ValueId produced by the KernelCall that wrote it.
    let mut intra_value_map: HashMap<String, ValueId> = HashMap::new();

    for (gi, (node, callee)) in group_nodes.iter().zip(group_cached.iter()).enumerate() {
        // Callee param lists (positional order for KernelCall args).
        let input_params: Vec<&str> =
            callee.params.iter().filter(|p| !p.is_output).map(|p| p.name.as_str()).collect();
        let output_params: Vec<&str> =
            callee.params.iter().filter(|p| p.is_output).map(|p| p.name.as_str()).collect();

        // For kernels with >1 output param we treat all outputs as external to
        // avoid the complexity of mapping multiple call_results.
        let single_output = output_params.len() == 1;

        let mut args: Vec<KernelCallArg> = Vec::new();

        // ── Input params ──────────────────────────────────────────────────
        for &param_name in &input_params {
            let Some((_, slot_ref)) = node.input_bindings.iter().find(|(n, _)| n == param_name)
            else {
                return None; // param not found — IR / TOML mismatch
            };

            let tensor_name = slot_ref_name(slot_ref)?;

            if tensor_name.starts_with('_') && intra_tensors.contains(&tensor_name) {
                // Intra-group tensor: pass the scalar ValueId computed earlier.
                let vid = *intra_value_map.get(&tensor_name)?;
                args.push(KernelCallArg::Value(vid));
            } else {
                // External tensor: get or create a host param.
                let host_name = if let Some(hn) = input_host_names.get(&tensor_name) {
                    hn.clone()
                } else {
                    let hn = make_param_name(&tensor_name, &mut used_param_names);
                    input_host_names.insert(tensor_name.clone(), hn.clone());
                    host.params.push(Param {
                        name: hn.clone(),
                        dtype,
                        shape: Shape::scalar(),
                        is_output: false,
                        kind: ParamKind::Tensor,
                    });
                    fused_input_bindings.push((hn.clone(), slot_ref.clone()));
                    hn
                };
                args.push(KernelCallArg::Tensor(host_name));
            }
        }

        // ── Constexpr params ─────────────────────────────────────────────
        for cexpr_decl in &callee.constexprs {
            let cexpr_name = cexpr_decl.name.name();
            // Unique host name: n{group_pos}_{callee_name}
            let host_cexpr = format!("n{}_{}", gi, cexpr_name);

            let Some((_, cexpr_val)) = node.cexprs.iter().find(|(n, _)| n == cexpr_name) else {
                return None; // constexpr not found in node
            };

            host.constexprs.push(ConstExprDecl {
                name: ConstExpr::new(&host_cexpr),
                dtype: cexpr_decl.dtype,
                value: None,
            });
            fused_cexprs.push((host_cexpr.clone(), cexpr_val.clone()));
            args.push(KernelCallArg::Tensor(host_cexpr));
        }

        // ── Output params ────────────────────────────────────────────────
        // For single-output kernels: if the output is intra, omit the arg so
        // KernelInlinePass skips the Store and maps the stored value to
        // call_result.  For external outputs (or multi-output kernels), pass
        // a Tensor arg so the store is kept.
        let mut call_result: Option<ValueId> = None;

        for &param_name in &output_params {
            let Some((_, slot_ref)) = node.output_bindings.iter().find(|(n, _)| n == param_name)
            else {
                return None;
            };

            let tensor_name = slot_ref_name(slot_ref)?;

            let is_intra = single_output
                && tensor_name.starts_with('_')
                && intra_tensors.contains(&tensor_name);

            if is_intra && call_result.is_none() {
                // Omit the arg → KernelInlinePass maps stored value → call_result.
                let vid = ValueId::new(next_vid);
                next_vid += 1;
                intra_value_map.insert(tensor_name, vid);
                call_result = Some(vid);
            } else {
                // External output: add a Tensor arg, keep the store.
                let host_name = if let Some(hn) = output_host_names.get(&tensor_name) {
                    hn.clone()
                } else {
                    let hn = make_param_name(&tensor_name, &mut used_param_names);
                    output_host_names.insert(tensor_name.clone(), hn.clone());
                    host.params.push(Param {
                        name: hn.clone(),
                        dtype,
                        shape: Shape::scalar(),
                        is_output: true,
                        kind: ParamKind::Tensor,
                    });
                    fused_output_bindings.push((hn.clone(), slot_ref.clone()));
                    hn
                };
                args.push(KernelCallArg::Tensor(host_name));
            }
        }

        let kc = Op::KernelCall { callee: node.kernel_name.to_string(), args, dtype };
        match call_result {
            Some(vid) => host.body.push_op(kc, vid),
            None => host.body.push_op_no_result(kc),
        }
    }

    // A fused node with no outputs is useless.
    if fused_output_bindings.is_empty() {
        return None;
    }

    let grid_dims = grid_to_dims(&fused_grid);

    let fused_node = DispatchNode {
        label: format!("fused.group.{group_id}"),
        kernel_name,
        kernel_ir: fused_kernel_placeholder,
        mode: fused_mode,
        input_bindings: fused_input_bindings,
        output_bindings: fused_output_bindings,
        cexprs: fused_cexprs,
        grid: fused_grid,
        dtype,
        grid_dims,
        fuse_group: None,
    };

    Some((fused_node, host))
}


// ---------------------------------------------------------------------------
// Direct IR synthesis — Pattern 1 (Reduction → Elementwise epilogue)
// ---------------------------------------------------------------------------

/// Maximum non-special ValueId across all blocks in a kernel.
/// Special VIDs (top bit set — loop-var refs, sub-op-flag refs) are excluded.
fn kernel_max_vid(kernel: &Kernel) -> u32 {
    const SPECIAL: u32 = 0x4000_0000; // covers both 0x4000_xxxx and 0xC000_xxxx loop-var VIDs
    let check = |m: &mut u32, block: &Block| {
        for (op, result) in block.ops.iter().zip(block.results.iter()) {
            op.value_refs().iter().for_each(|v| {
                if v.as_u32() & SPECIAL == 0 {
                    *m = (*m).max(v.as_u32());
                }
            });
            if let Some(r) = result {
                if r.as_u32() & SPECIAL == 0 {
                    *m = (*m).max(r.as_u32());
                }
            }
        }
    };
    let mut m = 0u32;
    check(&mut m, &kernel.body);
    for block in kernel.blocks.values() {
        check(&mut m, block);
    }
    m
}

/// Maximum BlockId across all blocks registered in a kernel.
fn kernel_max_block_id(kernel: &Kernel) -> u32 {
    kernel.blocks.keys().map(|b| b.as_u32()).max().unwrap_or(0)
}

/// Rename string-named param/constexpr references inside an Op.
/// Handles Load.src, Store.dst, StrideReduce.src, StrideStore, VectorLoad/Store, etc.
/// Only names present in `renames` are substituted; others are kept.
fn rename_param_refs_in_op(op: &mut Op, renames: &HashMap<String, String>) {
    if renames.is_empty() {
        return;
    }
    let rename = |s: &mut String| {
        if let Some(new) = renames.get(s.as_str()) {
            s.clone_from(new);
        }
    };
    let rename_opt = |s: &mut Option<String>| {
        if let Some(name) = s {
            if let Some(new) = renames.get(name.as_str()) {
                name.clone_from(new);
            }
        }
    };
    match op {
        Op::Load { src, .. } => rename(src),
        Op::Store { dst, .. } => rename(dst),
        Op::StrideReduce { src, secondary_src, .. } => {
            rename(src);
            rename_opt(secondary_src);
        },
        Op::StrideStore { src, dst, aux_src, .. } => {
            rename(src);
            rename(dst);
            rename_opt(aux_src);
        },
        Op::VectorLoad { src, .. } => rename(src),
        Op::VectorStore { dst, .. } => rename(dst),
        Op::Gather { src, .. } => rename(src),
        Op::Scatter { dst, .. } => rename(dst),
        Op::Atomic { dst, .. } => rename(dst),
        Op::StrideScan { src, dst, .. } => {
            rename(src);
            rename(dst);
        },
        Op::StrideArgReduce { src, .. } => rename(src),
        // Threadgroup-private names are NOT renamed (they're internal to the callee).
        _ => {},
    }
}

/// Clone one block with VID and BlockId offsets applied plus param renames.
///
/// - All non-special ValueIds (`raw < 0x8000_0000`) are shifted by `vid_offset`.
/// - All `BlockId` references in `Op::Loop.body` and `Op::If.then/else_block` are
///   shifted by `block_offset`.
/// - Param string references in Load/Store/etc. are renamed via `param_renames`.
fn clone_block_with_offsets(
    block: &Block,
    vid_offset: u32,
    block_offset: u32,
    param_renames: &HashMap<String, String>,
) -> Block {
    const SPECIAL: u32 = 0x4000_0000; // covers both 0x4000_xxxx and 0xC000_xxxx loop-var VIDs
    let remap_vid = |v: ValueId| -> ValueId {
        if v.as_u32() & SPECIAL == 0 { ValueId::new(v.as_u32() + vid_offset) } else { v }
    };

    let new_bid = BlockId::new(block.id.as_u32() + block_offset);
    let mut new_block = Block::new(new_bid);

    for (op, result) in block.ops.iter().zip(block.results.iter()) {
        let mut new_op = op.clone();

        // Remap BlockId references in Loop and If.
        match &mut new_op {
            Op::Loop { body, .. } => {
                *body = BlockId::new(body.as_u32() + block_offset);
            },
            Op::If { then_block, else_block, .. } => {
                *then_block = BlockId::new(then_block.as_u32() + block_offset);
                if let Some(eb) = else_block {
                    *eb = BlockId::new(eb.as_u32() + block_offset);
                }
            },
            _ => {},
        }

        // Rename param references.
        rename_param_refs_in_op(&mut new_op, param_renames);

        // Remap all non-special ValueIds.
        new_op.for_each_value_id_mut(&mut |v| {
            if v.as_u32() & SPECIAL == 0 {
                *v = ValueId::new(v.as_u32() + vid_offset);
            }
        });

        new_block.ops.push(new_op);
        new_block.results.push(result.map(remap_vid));
    }

    new_block
}

/// Clone the entire callee kernel (body + all sub-blocks) into the host kernel.
///
/// - Callee entry-block ops are **appended** to `host.body` (not added as a
///   separate block).
/// - Callee sub-blocks (non-entry) are added to `host.blocks` with a BlockId offset.
/// - VIDs are shifted so they don't collide with any existing VIDs in the host.
/// - Param string names present in `param_renames` are substituted throughout.
///
/// Returns the VID offset used (= `kernel_max_vid(host_before_clone) + 1`).
fn clone_callee_into_host(
    host: &mut Kernel,
    callee: &Kernel,
    param_renames: &HashMap<String, String>,
) -> u32 {
    let vid_offset = kernel_max_vid(host) + 1;
    let block_offset = kernel_max_block_id(host) + 1;

    // Clone sub-blocks (all blocks except the entry block).
    let sub_block_ids: Vec<BlockId> = callee
        .blocks
        .keys()
        .filter(|&&bid| bid != callee.body.id)
        .copied()
        .collect();
    for bid in sub_block_ids {
        let block = &callee.blocks[&bid];
        let cloned = clone_block_with_offsets(block, vid_offset, block_offset, param_renames);
        host.blocks.insert(cloned.id, cloned);
    }

    // Merge callee entry-block ops into host body (with block_offset on references,
    // but NOT adding block_offset to the entry block's own ID — it's merged, not stored).
    const SPECIAL: u32 = 0x4000_0000; // covers both 0x4000_xxxx and 0xC000_xxxx loop-var VIDs
    let remap_vid = |v: ValueId| -> ValueId {
        if v.as_u32() & SPECIAL == 0 { ValueId::new(v.as_u32() + vid_offset) } else { v }
    };

    for (op, result) in callee.body.ops.iter().zip(callee.body.results.iter()) {
        let mut new_op = op.clone();

        match &mut new_op {
            Op::Loop { body, .. } => {
                *body = BlockId::new(body.as_u32() + block_offset);
            },
            Op::If { then_block, else_block, .. } => {
                *then_block = BlockId::new(then_block.as_u32() + block_offset);
                if let Some(eb) = else_block {
                    *eb = BlockId::new(eb.as_u32() + block_offset);
                }
            },
            _ => {},
        }

        rename_param_refs_in_op(&mut new_op, param_renames);

        new_op.for_each_value_id_mut(&mut |v| {
            if v.as_u32() & SPECIAL == 0 {
                *v = ValueId::new(v.as_u32() + vid_offset);
            }
        });

        host.body.ops.push(new_op);
        host.body.results.push(result.map(remap_vid));
    }

    vid_offset
}

/// For each `Store { dst: store_dst }` found in `kernel.body` or any sub-block,
/// collect `(block_id, op_index, first_index_expr, stored_value_vid)`.
/// Trace a VID through one level of Cast to find the raw underlying value.
///
/// Gemv/bm4 stores bfloat-cast values: `v74 = bfloat(v64)` then `store v74`.
/// `v64` is the f32 SIMD-reduction result, broadcast to all threads and declared
/// at function scope.  `v74` is declared inside the if(tid==0) block.
/// When we capture the stored VID (`v74`) and need to use it OUTSIDE that block,
/// we must trace back to the pre-cast VID (`v64`) which is in function scope.
fn trace_through_cast(vid: ValueId, host: &Kernel) -> ValueId {
    let search_block = |block: &Block| -> Option<ValueId> {
        for (op, result) in block.ops.iter().zip(block.results.iter()) {
            if *result == Some(vid) {
                if let Op::Cast { value, .. } = op {
                    return Some(*value);
                }
                return None;
            }
        }
        None
    };
    if let Some(upstream) = search_block(&host.body) {
        return upstream;
    }
    for block in host.blocks.values() {
        if let Some(upstream) = search_block(block) {
            return upstream;
        }
    }
    vid
}

fn collect_stores_to_param(
    kernel: &Kernel,
    store_dst: &str,
) -> Vec<(BlockId, usize, metaltile_core::ir::IndexExpr, ValueId)> {
    use metaltile_core::ir::IndexExpr;

    let mut result = Vec::new();

    let check = |bid: BlockId,
                 block: &Block,
                 out: &mut Vec<(BlockId, usize, IndexExpr, ValueId)>| {
        for (i, op) in block.ops.iter().enumerate() {
            if let Op::Store { dst, indices, value, .. } = op
                && dst == store_dst
            {
                if let Some(idx) = indices.first() {
                    out.push((bid, i, idx.clone(), *value));
                }
            }
        }
    };

    check(kernel.body.id, &kernel.body, &mut result);
    for (&bid, block) in &kernel.blocks {
        if bid == kernel.body.id {
            continue; // already processed above
        }
        check(bid, block, &mut result);
    }

    result
}

/// Assign or allocate a fresh VID for `op_result`, recording it in `vid_map`.
fn assign_fresh_vid(
    op_result: &Option<ValueId>,
    vid_map: &mut BTreeMap<ValueId, ValueId>,
    next_vid: &mut u32,
) -> Option<ValueId> {
    op_result.map(|r| {
        if let Some(&existing) = vid_map.get(&r) {
            existing
        } else {
            let fresh = ValueId::new(*next_vid);
            *next_vid += 1;
            vid_map.insert(r, fresh);
            fresh
        }
    })
}

/// Inline a **flat** (no sub-blocks) elementwise callee for one specific row.
///
/// Substitutions:
/// - `Op::ProgramId { axis: 0 }` → remapped to `row_index`'s ValueId.
/// - `Op::Load { src }` → if src is a key in `intra_substitutions`, the load
///   is skipped and its result is replaced by the mapped ValueId.
/// - All other `Op::Load` srcs are renamed via `e_param_renames`.
/// - All `Op::Store` dsts are renamed via `e_param_renames`; Store indices are
///   replaced with `row_index` (the row index from the reduction store).
///
/// When `capture_store_dst` is Some, stores to that specific original
/// callee param name are NOT emitted and their stored value VID is captured.
///
/// Returns the ops to splice in place of the reduction's store for this row,
/// plus an optional captured value VID.
fn inline_flat_elementwise_for_row(
    e_callee: &Kernel,
    row_index: &metaltile_core::ir::IndexExpr,
    intra_substitutions: &HashMap<String, ValueId>,
    e_param_renames: &HashMap<String, String>,
    next_vid: &mut u32,
    capture_store_dst: Option<&str>,
) -> (Vec<(Op, Option<ValueId>)>, Option<ValueId>) {
    use metaltile_core::ir::IndexExpr;

    let row_vid = match row_index {
        IndexExpr::Value(v) => *v,
        _ => return (Vec::new(), None),
    };

    let mut vid_map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut captured: Option<ValueId> = None;

    // Pre-pass: seed intra-substitution load results and ProgramId results into vid_map.
    for (op, op_result) in e_callee.body.ops.iter().zip(e_callee.body.results.iter()) {
        match op {
            Op::Load { src, .. } if intra_substitutions.contains_key(src.as_str()) => {
                if let Some(r) = op_result {
                    vid_map.insert(*r, intra_substitutions[src.as_str()]);
                }
            },
            Op::ProgramId { .. } => {
                if let Some(r) = op_result {
                    vid_map.insert(*r, row_vid);
                }
            },
            _ => {},
        }
    }

    let mut result: Vec<(Op, Option<ValueId>)> = Vec::new();

    for (op, op_result) in e_callee.body.ops.iter().zip(e_callee.body.results.iter()) {
        match op {
            // ProgramId: already mapped, skip.
            Op::ProgramId { .. } => continue,

            // Intra-param load: already mapped, skip.
            Op::Load { src, .. } if intra_substitutions.contains_key(src.as_str()) => continue,

            // External input load: rename src, remap indices via VID map.
            // Scalar / constexpr loads have empty indices — keep them empty.
            // Indexed loads remap each IndexExpr::Value VID through vid_map so
            // that program_id(0) references become `row_vid` automatically.
            Op::Load { src, indices, .. } => {
                let renamed = e_param_renames.get(src.as_str()).map(|s| s.as_str()).unwrap_or(src);
                let new_indices: Vec<IndexExpr> = if indices.is_empty() {
                    vec![]
                } else {
                    indices
                        .iter()
                        .map(|ie| match ie {
                            IndexExpr::Value(v) =>
                                IndexExpr::Value(vid_map.get(v).copied().unwrap_or(*v)),
                            IndexExpr::Range(v, n) =>
                                IndexExpr::Range(vid_map.get(v).copied().unwrap_or(*v), *n),
                            IndexExpr::Const(_) => ie.clone(),
                        })
                        .collect()
                };
                let new_op = Op::Load {
                    src: renamed.to_string(),
                    indices: new_indices,
                    mask: None,
                    other: None,
                };
                let new_result = assign_fresh_vid(op_result, &mut vid_map, next_vid);
                result.push((new_op, new_result));
                continue;
            },

            // Output store: rename dst, remap value and indices via VID map.
            // For simple elementwise (store index = program_id), vid_map remaps it
            // to row_vid.  For scattered stores (e.g. kv_cache), the derived VIDs
            // for the index computation are remapped transitively through vid_map.
            Op::Store { dst, indices, value, mask } => {
                // Capture: suppress store and record the stored value VID.
                if let Some(cap_dst) = capture_store_dst
                    && dst == cap_dst
                {
                    let remapped_val = vid_map.get(value).copied().unwrap_or(*value);
                    captured = Some(remapped_val);
                    continue;
                }
                let renamed = e_param_renames.get(dst.as_str()).map(|s| s.as_str()).unwrap_or(dst);
                let remapped_val = vid_map.get(value).copied().unwrap_or(*value);
                let new_indices: Vec<IndexExpr> = indices
                    .iter()
                    .map(|ie| match ie {
                        IndexExpr::Value(v) =>
                            IndexExpr::Value(vid_map.get(v).copied().unwrap_or(*v)),
                        IndexExpr::Range(v, n) =>
                            IndexExpr::Range(vid_map.get(v).copied().unwrap_or(*v), *n),
                        IndexExpr::Const(_) => ie.clone(),
                    })
                    .collect();
                let new_op = Op::Store {
                    dst: renamed.to_string(),
                    indices: new_indices,
                    value: remapped_val,
                    mask: *mask,
                };
                result.push((new_op, None));
                continue;
            },

            _ => {},
        }

        // Default: remap VIDs and emit.
        let mut new_op = op.clone();
        new_op.for_each_value_id_mut(&mut |v| {
            if let Some(&nv) = vid_map.get(v) {
                *v = nv;
            }
        });
        let new_result = assign_fresh_vid(op_result, &mut vid_map, next_vid);
        result.push((new_op, new_result));
    }

    (result, captured)
}

/// Find all `Store { dst: store_dst }` across all blocks in `kernel` and replace
/// each with the inlined elementwise epilogue for that row.
///
/// Stores are processed from the highest (block_id, op_index) to the lowest
/// so that in-block `Vec::remove` + `Vec::insert` don't shift unprocessed indices.
///
/// `intra_substitutions_base` provides the static (non-per-row) intra param → VID
/// mappings (e.g., shared intermediate consumed across rows).  The per-row stored
/// value from the reduction is NOT in this map — it's derived from each store.
///
/// When `capture_store_dst` is Some, stores to that callee param name in the
/// epilogue body are suppressed and their values are captured.
///
/// Returns `Vec<(row_index, captured_value_vid)>` when `capture_store_dst` is set.
fn inject_elementwise_epilogue(
    kernel: &mut Kernel,
    store_dst: &str,
    e_callee: &Kernel,
    e_intra_param: &str,
    e_param_renames: &HashMap<String, String>,
    next_vid: &mut u32,
    extra_intra_subs: &HashMap<String, ValueId>,
    capture_store_dst: Option<&str>,
) -> Vec<(metaltile_core::ir::IndexExpr, ValueId)> {
    use metaltile_core::ir::IndexExpr;

    // Collect all stores first (before any mutation).
    let mut stores: Vec<(BlockId, usize, IndexExpr, ValueId)> =
        collect_stores_to_param(kernel, store_dst);

    // Sort descending by (block_id, op_index) so that the highest index in each
    // block is processed first — preserving validity of lower indices in that block.
    stores.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.as_u32().cmp(&a.0.as_u32())));

    let mut captured: Vec<(IndexExpr, ValueId)> = Vec::new();

    for (bid, store_idx, row_idx, res_vid) in stores {
        // Build per-row intra_substitutions: main intra param + extras.
        let mut intra_substitutions: HashMap<String, ValueId> = extra_intra_subs.clone();
        intra_substitutions.insert(e_intra_param.to_string(), res_vid);

        let (epilogue, row_captured) = inline_flat_elementwise_for_row(
            e_callee,
            &row_idx,
            &intra_substitutions,
            e_param_renames,
            next_vid,
            capture_store_dst,
        );

        if let Some(cap_vid) = row_captured {
            captured.push((row_idx.clone(), cap_vid));
        }

        // Locate the block (entry block vs sub-block).
        let block = if bid == kernel.body.id {
            &mut kernel.body
        } else {
            match kernel.blocks.get_mut(&bid) {
                Some(b) => b,
                None => continue,
            }
        };

        // Replace the store at store_idx with the epilogue ops.
        block.ops.remove(store_idx);
        block.results.remove(store_idx);
        for (j, (ep_op, ep_result)) in epilogue.into_iter().enumerate() {
            block.ops.insert(store_idx + j, ep_op);
            block.results.insert(store_idx + j, ep_result);
        }
    }

    captured
}

/// Attempt **Pattern 3** direct IR synthesis for a diamond [R0, E0, R1, E1] group.
///
/// Handles exactly 4-node groups where:
/// - R0 and R1 are identical-spec Reduction kernels (same num_rows, tpg)
/// - E0 is a flat Elementwise reading R0's intra output and writing an intra
///   tensor consumed by E1
/// - E1 is a flat Elementwise reading R1's intra output AND E0's intra output,
///   writing external output(s)
/// - R0 and R1 may share or differ in external inputs
///
/// Synthesis:
/// 1. Clone R0 into host, inject E0 as epilogue, capture per-row gated VIDs
/// 2. Clone R1 into host, inject E1 as epilogue with dual intra substitutions
/// 3. Build fused node inheriting R0's grid, union I/O bindings
///
/// Returns `None` when the group doesn't match the pattern.
fn synthesize_pattern3(
    group_nodes: &[&DispatchNode],
    group_cached: &[&Kernel],
    intra_tensors: &HashSet<String>,
    dtype: DType,
    group_id: usize,
) -> Option<(DispatchNode, Kernel)> {
    if group_nodes.len() != 4 {
        return None;
    }

    let (r0, e0, r1, e1) = (group_nodes[0], group_nodes[1], group_nodes[2], group_nodes[3]);
    let (c0, ce0, c1, ce1) = (group_cached[0], group_cached[1], group_cached[2], group_cached[3]);

    // Validate: R0, R1 must be Reduction with identical specs.
    let (num_rows, tpg) = match (&r0.mode, &r0.grid) {
        (KernelMode::Reduction, GridSpec::Reduction { num_rows, threads_per_group }) =>
            (*num_rows, *threads_per_group),
        _ => return None,
    };
    match (&r1.mode, &r1.grid) {
        (KernelMode::Reduction, GridSpec::Reduction { num_rows: nr, threads_per_group: t }) =>
            if *nr != num_rows || *t != tpg { return None },
        _ => return None,
    };

    // Validate: E0, E1 must be flat elementwise with n a multiple of num_rows.
    for (en, ec) in [(&e0, &ce0), (&e1, &ce1)] {
        let n = match (&en.mode, &en.grid) {
            (KernelMode::Elementwise, GridSpec::Elementwise { n }) => *n,
            _ => return None,
        };
        if n % num_rows != 0 {
            return None;
        }
        if ec.blocks.len() > 1 {
            return None;
        }
        if ec.body.ops.iter().any(|op| matches!(op, Op::KernelCall { .. })) {
            return None;
        }
    }

    // Safety checks on reduction kernels.
    for rc in [c0, c1] {
        if rc.body.ops.iter().any(|op| matches!(op, Op::KernelCall { .. })) {
            return None;
        }
        if rc.params.iter().any(|p| p.kind == ParamKind::Strided) {
            return None;
        }
    }

    // ── Find intra param names ──────────────────────────────────────────────

    let (r0_out_param, _) = find_single_intra_output_param(r0, c0, intra_tensors)?;
    let e0_intra_in = find_single_intra_input_param(e0, ce0, intra_tensors)?;
    let (e0_intra_out_callee, e0_intra_out_tensor) =
        find_single_intra_output_param(e0, ce0, intra_tensors)?;

    let (r1_out_param, _) = find_single_intra_output_param(r1, c1, intra_tensors)?;

    // E1's intra input from R1 (NOT from E0).
    let e1_intra_in_r1 = {
        let intra_ins: Vec<_> = ce1
            .params
            .iter()
            .filter(|p| !p.is_output)
            .filter(|p| {
                e1.input_bindings
                    .iter()
                    .find(|(n, _)| n == &p.name)
                    .and_then(|(_, slot)| slot_ref_name(slot))
                    .map(|t| intra_tensors.contains(&t) && t != e0_intra_out_tensor)
                    .unwrap_or(false)
            })
            .collect();
        if intra_ins.len() != 1 {
            return None;
        }
        intra_ins[0].name.clone()
    };

    // E1's intra input from E0 (_gated).
    let e1_intra_in_e0 = {
        let intra_ins: Vec<_> = ce1
            .params
            .iter()
            .filter(|p| !p.is_output)
            .filter(|p| {
                e1.input_bindings
                    .iter()
                    .find(|(n, _)| n == &p.name)
                    .and_then(|(_, slot)| slot_ref_name(slot))
                    .map(|t| intra_tensors.contains(&t) && t == e0_intra_out_tensor)
                    .unwrap_or(false)
            })
            .collect();
        if intra_ins.len() != 1 {
            return None;
        }
        intra_ins[0].name.clone()
    };

    // ── Build host kernel ────────────────────────────────────────────────────

    let kernel_name: &'static str =
        Box::leak(format!("fused_group_{group_id}").into_boxed_str());
    let mut host = Kernel::new(kernel_name);
    host.mode = KernelMode::Reduction;

    let fused_grid = GridSpec::Reduction { num_rows, threads_per_group: tpg };
    let grid_dims = grid_to_dims(&fused_grid);

    let mut fused_input_bindings: Vec<(String, SlotRef)> = Vec::new();
    let mut fused_output_bindings: Vec<(String, SlotRef)> = Vec::new();
    let mut fused_cexprs: Vec<(String, ConstexprValue)> = Vec::new();
    let mut used_param_names: HashSet<String> = HashSet::new();

    // ── Step A: Clone R0 into host ────────────────────────────────────────

    let r0_param_renames = build_reduction_renames(
        r0, c0, intra_tensors, &r0_out_param,
        &mut host, &mut fused_input_bindings, &mut fused_output_bindings,
        &mut fused_cexprs, &mut used_param_names, dtype, "n0",
    )?;

    clone_callee_into_host(&mut host, c0, &r0_param_renames);

    // ── Step B: Remove R0's stores and capture raw gate VIDs ─────────────
    //
    // We do NOT inject E0 (silu) here.  Injecting inside R0's if(tid==0)
    // block would put the silu result VIDs in a C++ scope that is not visible
    // in R1's separate if(tid==0) block.  Instead, E0 and E1 are both injected
    // inside R1's if block in step D, where R0's raw reduction results are still
    // in scope (they are broadcast to all threads at function level).

    let e0_param_renames = build_elementwise_renames(
        e0, ce0, intra_tensors, &e0_intra_in,
        &mut host, &mut fused_input_bindings, &mut fused_output_bindings,
        &mut fused_cexprs, &mut used_param_names, dtype, "n1",
    )?;

    // Collect R0's stores (sorted descending so removal preserves indices).
    use metaltile_core::ir::IndexExpr;
    let mut r0_stores: Vec<(BlockId, usize, IndexExpr, ValueId)> =
        collect_stores_to_param(&host, &r0_out_param);
    r0_stores.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.as_u32().cmp(&a.0.as_u32())));

    // Capture gate VIDs: trace through any bfloat Cast to reach the f32 SIMD
    // reduction result (declared in function scope, broadcast to all threads).
    // The stored value is typically `v74 = bfloat(v64)` inside R0's if-block;
    // we need `v64` which is visible in R1's separate if-block.
    let gate_vids: Vec<ValueId> =
        r0_stores.iter().map(|(_, _, _, v)| trace_through_cast(*v, &host)).collect();

    // Remove R0's stores without inserting any replacement ops.
    for (bid, store_idx, _, _) in &r0_stores {
        let block = if *bid == host.body.id {
            &mut host.body
        } else {
            match host.blocks.get_mut(bid) {
                Some(b) => b,
                None => continue,
            }
        };
        block.ops.remove(*store_idx);
        block.results.remove(*store_idx);
    }

    // ── Step C: Clone R1 into host ────────────────────────────────────────

    let r1_param_renames = build_reduction_renames(
        r1, c1, intra_tensors, &r1_out_param,
        &mut host, &mut fused_input_bindings, &mut fused_output_bindings,
        &mut fused_cexprs, &mut used_param_names, dtype, "nr1",
    )?;

    // Record snapshot before cloning R1 so we can rename its DeclareLocal/SetLocal
    // ops — R0 and R1 both use names like "acc0".."acc3", which the MSL emitter
    // renders as __ml_acc0 etc.  Without renaming, the second clone produces a
    // "redefinition of '__ml_acc0'" compile error in Metal.
    let body_ops_before_r1 = host.body.ops.len();
    let blocks_before_r1: HashSet<BlockId> = host.blocks.keys().cloned().collect();

    clone_callee_into_host(&mut host, c1, &r1_param_renames);

    // Append "_r1" to every DeclareLocal/SetLocal name added by R1.
    for op in host.body.ops[body_ops_before_r1..].iter_mut() {
        match op {
            Op::DeclareLocal { name, .. } | Op::SetLocal { name, .. } => name.push_str("_r1"),
            _ => {},
        }
    }
    for (&bid, block) in host.blocks.iter_mut() {
        if blocks_before_r1.contains(&bid) {
            continue;
        }
        for op in block.ops.iter_mut() {
            match op {
                Op::DeclareLocal { name, .. } | Op::SetLocal { name, .. } => name.push_str("_r1"),
                _ => {},
            }
        }
    }

    let mut next_vid_after_r1 = kernel_max_vid(&host) + 1;

    // ── Step D: Inject E0+E1 together into R1's stores ───────────────────
    //
    // E0 (silu) and E1 (mul) are both injected inside R1's if(tid==0) block.
    // gate_vids (R0's raw reduction results) are in function scope (broadcast
    // to all threads), so they are visible here even though R0's if block has
    // already closed.  This avoids the C++ scope error that would occur if
    // silu results were declared in R0's if block and referenced in R1's.

    let e1_param_renames = build_elementwise_renames(
        e1, ce1, intra_tensors, &e1_intra_in_r1,
        &mut host, &mut fused_input_bindings, &mut fused_output_bindings,
        &mut fused_cexprs, &mut used_param_names, dtype, "n2",
    )?;

    // Per-row injection: E0 (silu) then E1 (mul) in R1's store positions.
    let mut r1_stores: Vec<(BlockId, usize, IndexExpr, ValueId)> =
        collect_stores_to_param(&host, &r1_out_param);

    r1_stores.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.as_u32().cmp(&a.0.as_u32())));

    for (i, (bid, store_idx, row_idx, r1_intra_vid)) in r1_stores.iter().enumerate() {
        let (bid, store_idx, row_idx, r1_intra_vid) =
            (*bid, *store_idx, row_idx.clone(), *r1_intra_vid);

        // Apply E0 (silu) using R0's gate VID at position i (same sort order).
        let (e0_epilogue, silu_vid_opt) = if let Some(&gate_vid) = gate_vids.get(i) {
            let mut e0_subs = HashMap::new();
            e0_subs.insert(e0_intra_in.clone(), gate_vid);
            inline_flat_elementwise_for_row(
                ce0,
                &row_idx,
                &e0_subs,
                &e0_param_renames,
                &mut next_vid_after_r1,
                Some(&e0_intra_out_callee),
            )
        } else {
            (Vec::new(), None)
        };

        // Apply E1 (mul) using R1 result + silu result.
        let mut e1_subs = HashMap::new();
        e1_subs.insert(e1_intra_in_r1.clone(), r1_intra_vid);
        if let Some(silu_vid) = silu_vid_opt {
            e1_subs.insert(e1_intra_in_e0.clone(), silu_vid);
        }
        let (e1_epilogue, _) = inline_flat_elementwise_for_row(
            ce1,
            &row_idx,
            &e1_subs,
            &e1_param_renames,
            &mut next_vid_after_r1,
            None,
        );

        let block = if bid == host.body.id {
            &mut host.body
        } else {
            match host.blocks.get_mut(&bid) {
                Some(b) => b,
                None => continue,
            }
        };

        block.ops.remove(store_idx);
        block.results.remove(store_idx);
        let combined = e0_epilogue.into_iter().chain(e1_epilogue.into_iter());
        for (j, (ep_op, ep_result)) in combined.enumerate() {
            block.ops.insert(store_idx + j, ep_op);
            block.results.insert(store_idx + j, ep_result);
        }
    }

    // ── Finalize fused node ────────────────────────────────────────────────

    if fused_output_bindings.is_empty() {
        return None;
    }

    let fused_node = DispatchNode {
        label: format!("fused.group.{group_id}"),
        kernel_name,
        kernel_ir: fused_kernel_placeholder,
        mode: KernelMode::Reduction,
        input_bindings: fused_input_bindings,
        output_bindings: fused_output_bindings,
        cexprs: fused_cexprs,
        grid: fused_grid,
        dtype,
        grid_dims,
        fuse_group: None,
    };

    debug!(
        "fuse group {group_id}: Pattern 3 diamond synthesis (4 nodes → 1, num_rows={num_rows})",
    );

    Some((fused_node, host))
}

// ── Helper: find single intra output param ─────────────────────────────────

fn find_single_intra_output_param(
    node: &DispatchNode,
    callee: &Kernel,
    intra_tensors: &HashSet<String>,
) -> Option<(String, String)> {
    let intra_outs: Vec<_> = callee
        .params
        .iter()
        .filter(|p| p.is_output)
        .filter(|p| {
            node.output_bindings
                .iter()
                .find(|(n, _)| n == &p.name)
                .and_then(|(_, slot)| slot_ref_name(slot))
                .map(|t| intra_tensors.contains(&t))
                .unwrap_or(false)
        })
        .collect();
    if intra_outs.len() != 1 {
        return None;
    }
    let param = &intra_outs[0];
    let tensor_name = node
        .output_bindings
        .iter()
        .find(|(n, _)| n == &param.name)
        .and_then(|(_, slot)| slot_ref_name(slot))?;
    Some((param.name.clone(), tensor_name))
}

// ── Helper: find single intra input param ───────────────────────────────────

fn find_single_intra_input_param(
    node: &DispatchNode,
    callee: &Kernel,
    intra_tensors: &HashSet<String>,
) -> Option<String> {
    let intra_ins: Vec<_> = callee
        .params
        .iter()
        .filter(|p| !p.is_output)
        .filter(|p| {
            node.input_bindings
                .iter()
                .find(|(n, _)| n == &p.name)
                .and_then(|(_, slot)| slot_ref_name(slot))
                .map(|t| intra_tensors.contains(&t))
                .unwrap_or(false)
        })
        .collect();
    if intra_ins.len() != 1 {
        return None;
    }
    Some(intra_ins[0].name.clone())
}

// ── Helper: build reduction kernel renames ─────────────────────────────────

fn build_reduction_renames(
    node: &DispatchNode,
    callee: &Kernel,
    intra_tensors: &HashSet<String>,
    intra_out_param_name: &str,
    host: &mut Kernel,
    fused_input_bindings: &mut Vec<(String, SlotRef)>,
    fused_output_bindings: &mut Vec<(String, SlotRef)>,
    fused_cexprs: &mut Vec<(String, ConstexprValue)>,
    used_param_names: &mut HashSet<String>,
    dtype: DType,
    node_prefix: &str,
) -> Option<HashMap<String, String>> {
    let mut renames: HashMap<String, String> = HashMap::new();

    for param in callee.params.iter().filter(|p| !p.is_output) {
        let (tensor_name, orig_slot_ref) = node
            .input_bindings
            .iter()
            .find(|(n, _)| n == &param.name)
            .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
        let host_name = make_param_name(&tensor_name, used_param_names);
        host.params.push(Param {
            name: host_name.clone(),
            dtype,
            shape: Shape::scalar(),
            is_output: false,
            kind: param.kind.clone(),
        });
        fused_input_bindings.push((host_name.clone(), orig_slot_ref));
        renames.insert(param.name.clone(), host_name);
    }

    for cexpr_decl in &callee.constexprs {
        let cexpr_name = cexpr_decl.name.name();
        let host_cexpr = format!("{node_prefix}_{cexpr_name}");
        let Some((_, cexpr_val)) = node.cexprs.iter().find(|(n, _)| n == cexpr_name) else {
            continue;
        };
        host.constexprs.push(ConstExprDecl {
            name: ConstExpr::new(&host_cexpr),
            dtype: cexpr_decl.dtype,
            value: None,
        });
        fused_cexprs.push((host_cexpr.clone(), cexpr_val.clone()));
        renames.insert(cexpr_name.to_string(), host_cexpr);
    }

    for param in callee.params.iter().filter(|p| p.is_output) {
        let (tensor_name, orig_slot_ref) = node
            .output_bindings
            .iter()
            .find(|(n, _)| n == &param.name)
            .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
        if intra_tensors.contains(&tensor_name) && param.name == intra_out_param_name {
            // Intra output — not renamed; stores keep original callee param name.
            renames.insert(param.name.clone(), param.name.clone());
            continue;
        }
        let host_name = make_param_name(&tensor_name, used_param_names);
        host.params.push(Param {
            name: host_name.clone(),
            dtype,
            shape: Shape::scalar(),
            is_output: true,
            kind: param.kind.clone(),
        });
        fused_output_bindings.push((host_name.clone(), orig_slot_ref));
        renames.insert(param.name.clone(), host_name);
    }

    Some(renames)
}

// ── Helper: build elementwise renames ──────────────────────────────────────

fn build_elementwise_renames(
    node: &DispatchNode,
    callee: &Kernel,
    intra_tensors: &HashSet<String>,
    intra_in_param: &str,
    host: &mut Kernel,
    fused_input_bindings: &mut Vec<(String, SlotRef)>,
    fused_output_bindings: &mut Vec<(String, SlotRef)>,
    fused_cexprs: &mut Vec<(String, ConstexprValue)>,
    used_param_names: &mut HashSet<String>,
    dtype: DType,
    node_prefix: &str,
) -> Option<HashMap<String, String>> {
    let mut renames: HashMap<String, String> = HashMap::new();

    for param in callee.params.iter().filter(|p| !p.is_output) {
        if param.name == intra_in_param {
            renames.insert(param.name.clone(), param.name.clone());
            continue;
        }
        let (tensor_name, orig_slot_ref) = node
            .input_bindings
            .iter()
            .find(|(n, _)| n == &param.name)
            .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
        // Any intra tensor input (e.g. _gated flowing from E0) is substituted
        // in-register during inline_flat_elementwise_for_row — no external param.
        if intra_tensors.contains(&tensor_name) {
            renames.insert(param.name.clone(), param.name.clone());
            continue;
        }
        let host_name = fused_input_bindings
            .iter()
            .find(|(_, slot)| slot_ref_name(slot).as_deref() == Some(&tensor_name))
            .map(|(hn, _)| hn.clone())
            .unwrap_or_else(|| {
                let hn = make_param_name(&tensor_name, used_param_names);
                host.params.push(Param {
                    name: hn.clone(),
                    dtype,
                    shape: Shape::scalar(),
                    is_output: false,
                    kind: param.kind.clone(),
                });
                fused_input_bindings.push((hn.clone(), orig_slot_ref));
                hn
            });
        renames.insert(param.name.clone(), host_name);
    }

    for cexpr_decl in &callee.constexprs {
        let cexpr_name = cexpr_decl.name.name();
        let host_cexpr = format!("{node_prefix}_{cexpr_name}");
        let Some((_, cexpr_val)) = node.cexprs.iter().find(|(n, _)| n == cexpr_name) else {
            continue;
        };
        host.constexprs.push(ConstExprDecl {
            name: ConstExpr::new(&host_cexpr),
            dtype: cexpr_decl.dtype,
            value: None,
        });
        fused_cexprs.push((host_cexpr.clone(), cexpr_val.clone()));
        renames.insert(cexpr_name.to_string(), host_cexpr);
    }

    for param in callee.params.iter().filter(|p| p.is_output) {
        let (tensor_name, orig_slot_ref) = node
            .output_bindings
            .iter()
            .find(|(n, _)| n == &param.name)
            .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
        if intra_tensors.contains(&tensor_name) {
            renames.insert(param.name.clone(), param.name.clone());
        } else {
            let host_name = make_param_name(&tensor_name, used_param_names);
            host.params.push(Param {
                name: host_name.clone(),
                dtype,
                shape: Shape::scalar(),
                is_output: true,
                kind: param.kind.clone(),
            });
            fused_output_bindings.push((host_name.clone(), orig_slot_ref));
            renames.insert(param.name.clone(), host_name);
        }
    }

    Some(renames)
}

/// Attempt **Pattern 1** direct IR synthesis for a linear epilogue chain.
///
/// Handles N ≥ 2 nodes where:
/// - `node[0]` is `Reduction` or `Grid3D` (the "host" that drives dispatch)
/// - Each `node[i]` (i ≥ 1) is `Elementwise` or flat-1D `Grid3D`, has a flat
///   body, and consumes exactly one intra tensor produced by `node[i-1]`
/// - Each node's element count `n` is a multiple of the first node's `num_rows`
///
/// The first node's body is cloned into the host kernel; each subsequent node's
/// body is injected in place of the preceding node's intra-output stores.
/// Intermediate intra tensors never materialise in global memory.
///
/// Returns `None` when the group doesn't match the pattern or synthesis fails.
fn synthesize_pattern1(
    group_nodes: &[&DispatchNode],
    group_cached: &[&Kernel],
    intra_tensors: &HashSet<String>,
    dtype: DType,
    group_id: usize,
) -> Option<(DispatchNode, Kernel)> {
    if group_nodes.len() < 2 {
        return None;
    }

    let (r_node, r_callee) = (group_nodes[0], group_cached[0]);

    // First node: Reduction OR any-shape Grid3D (e.g. rope_llama with grid_y=half_dim).
    // For Grid3D the "num_rows" is the total thread count (x*y*z); the fused
    // kernel retains the same mode and grid spec.
    let (num_rows, r_fused_mode, r_fused_grid) = match (&r_node.mode, &r_node.grid) {
        (KernelMode::Reduction, GridSpec::Reduction { num_rows, threads_per_group }) => (
            *num_rows,
            KernelMode::Reduction,
            GridSpec::Reduction { num_rows: *num_rows, threads_per_group: *threads_per_group },
        ),
        (KernelMode::Grid3D, GridSpec::Grid3D { x, y, z, .. }) => (
            x * y * z,
            KernelMode::Grid3D,
            r_node.grid.clone(),
        ),
        _ => return None,
    };

    // Upfront validation of every subsequent node.
    for (e_node, e_callee) in group_nodes[1..].iter().zip(group_cached[1..].iter()) {
        // Must be Elementwise or flat-1D Grid3D (only program_id::<0>() used).
        let n = match (&e_node.mode, &e_node.grid) {
            (KernelMode::Elementwise, GridSpec::Elementwise { n }) => *n,
            (KernelMode::Grid3D, GridSpec::Grid3D { x, y: 1, z: 1, .. }) => *x,
            _ => return None,
        };
        // n must be an integral multiple of num_rows (ratio ≥ 1 is fine).
        if n % num_rows != 0 {
            return None;
        }
        // Must have a flat body (no nested Loop/If sub-blocks).
        if e_callee.blocks.len() > 1 {
            return None;
        }
        // No nested KernelCall in the epilogue body.
        if e_callee.body.ops.iter().any(|op| matches!(op, Op::KernelCall { .. })) {
            return None;
        }
    }

    // Reject unsupported first-node features.
    if r_callee.body.ops.iter().any(|op| matches!(op, Op::KernelCall { .. })) {
        return None;
    }
    if r_callee.params.iter().any(|p| p.kind == ParamKind::Strided) {
        return None;
    }

    // Find the single intra output param in the first node — the tensor flowing
    // into node[1].
    let r_out_param_original: String = {
        let intra_outs: Vec<_> = r_callee
            .params
            .iter()
            .filter(|p| p.is_output)
            .filter(|p| {
                r_node
                    .output_bindings
                    .iter()
                    .find(|(n, _)| n == &p.name)
                    .and_then(|(_, slot)| slot_ref_name(slot))
                    .map(|t| intra_tensors.contains(&t))
                    .unwrap_or(false)
            })
            .collect();
        if intra_outs.len() != 1 {
            return None;
        }
        intra_outs[0].name.clone()
    };

    // ── Build host kernel ────────────────────────────────────────────────────

    let kernel_name: &'static str =
        Box::leak(format!("fused_group_{group_id}").into_boxed_str());
    let mut host = Kernel::new(kernel_name);
    host.mode = r_fused_mode.clone();

    let fused_grid = r_fused_grid;
    let grid_dims = grid_to_dims(&fused_grid);

    let mut fused_input_bindings: Vec<(String, SlotRef)> = Vec::new();
    let mut fused_output_bindings: Vec<(String, SlotRef)> = Vec::new();
    let mut fused_cexprs: Vec<(String, ConstexprValue)> = Vec::new();
    let mut used_param_names: HashSet<String> = HashSet::new();

    // ── Clone the first node's body ──────────────────────────────────────────
    // Intra output param is NOT renamed → its stores keep the original callee
    // param name so that inject_elementwise_epilogue can find them.
    let mut r_param_renames: HashMap<String, String> = HashMap::new();

    for param in r_callee.params.iter().filter(|p| !p.is_output) {
        let (tensor_name, orig_slot_ref) = r_node
            .input_bindings
            .iter()
            .find(|(n, _)| n == &param.name)
            .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
        let host_name = make_param_name(&tensor_name, &mut used_param_names);
        host.params.push(Param {
            name: host_name.clone(),
            dtype,
            shape: Shape::scalar(),
            is_output: false,
            kind: param.kind.clone(),
        });
        fused_input_bindings.push((host_name.clone(), orig_slot_ref));
        r_param_renames.insert(param.name.clone(), host_name);
    }

    for cexpr_decl in &r_callee.constexprs {
        let cexpr_name = cexpr_decl.name.name();
        let host_cexpr = format!("n0_{cexpr_name}");
        let (_, cexpr_val) = r_node.cexprs.iter().find(|(n, _)| n == cexpr_name)?;
        host.constexprs.push(ConstExprDecl {
            name: ConstExpr::new(&host_cexpr),
            dtype: cexpr_decl.dtype,
            value: None,
        });
        fused_cexprs.push((host_cexpr.clone(), cexpr_val.clone()));
        r_param_renames.insert(cexpr_name.to_string(), host_cexpr);
    }

    for param in r_callee.params.iter().filter(|p| p.is_output) {
        let (tensor_name, orig_slot_ref) = r_node
            .output_bindings
            .iter()
            .find(|(n, _)| n == &param.name)
            .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
        if intra_tensors.contains(&tensor_name) {
            // Intra output — not renamed; stores stay as the original callee param name.
            continue;
        }
        let host_name = make_param_name(&tensor_name, &mut used_param_names);
        host.params.push(Param {
            name: host_name.clone(),
            dtype,
            shape: Shape::scalar(),
            is_output: true,
            kind: param.kind.clone(),
        });
        fused_output_bindings.push((host_name.clone(), orig_slot_ref));
        r_param_renames.insert(param.name.clone(), host_name);
    }

    clone_callee_into_host(&mut host, r_callee, &r_param_renames);

    // ── Inject each subsequent node as an epilogue ───────────────────────────
    // `current_intra_store_name` is the original callee param name whose stores
    // currently appear in the host and will be replaced by the next injection.
    let mut current_intra_store_name = r_out_param_original;
    let mut next_vid = kernel_max_vid(&host) + 1;

    for (ei, (e_node, e_callee)) in
        group_nodes[1..].iter().zip(group_cached[1..].iter()).enumerate()
    {
        let node_prefix = format!("n{}", ei + 1);

        // Find the single intra input param (reads the previous node's output).
        let e_intra_input_param: String = {
            let intra_ins: Vec<_> = e_callee
                .params
                .iter()
                .filter(|p| !p.is_output)
                .filter(|p| {
                    e_node
                        .input_bindings
                        .iter()
                        .find(|(n, _)| n == &p.name)
                        .and_then(|(_, slot)| slot_ref_name(slot))
                        .map(|t| intra_tensors.contains(&t))
                        .unwrap_or(false)
                })
                .collect();
            if intra_ins.len() != 1 {
                return None;
            }
            intra_ins[0].name.clone()
        };

        let mut e_param_renames: HashMap<String, String> = HashMap::new();

        // External input params (not the intra input).
        for param in e_callee.params.iter().filter(|p| !p.is_output) {
            if param.name == e_intra_input_param {
                continue;
            }
            let (tensor_name, orig_slot_ref) = e_node
                .input_bindings
                .iter()
                .find(|(n, _)| n == &param.name)
                .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
            // Reuse a host param if the same tensor is already bound.
            let host_name = fused_input_bindings
                .iter()
                .find(|(_, slot)| slot_ref_name(slot).as_deref() == Some(&tensor_name))
                .map(|(hn, _)| hn.clone())
                .unwrap_or_else(|| {
                    let hn = make_param_name(&tensor_name, &mut used_param_names);
                    host.params.push(Param {
                        name: hn.clone(),
                        dtype,
                        shape: Shape::scalar(),
                        is_output: false,
                        kind: param.kind.clone(),
                    });
                    fused_input_bindings.push((hn.clone(), orig_slot_ref));
                    hn
                });
            e_param_renames.insert(param.name.clone(), host_name);
        }

        // Constexprs for this epilogue node — prefixed to avoid name collisions.
        for cexpr_decl in &e_callee.constexprs {
            let cexpr_name = cexpr_decl.name.name();
            let host_cexpr = format!("{node_prefix}_{cexpr_name}");
            let Some((_, cexpr_val)) = e_node.cexprs.iter().find(|(n, _)| n == cexpr_name) else {
                return None;
            };
            host.constexprs.push(ConstExprDecl {
                name: ConstExpr::new(&host_cexpr),
                dtype: cexpr_decl.dtype,
                value: None,
            });
            fused_cexprs.push((host_cexpr.clone(), cexpr_val.clone()));
            e_param_renames.insert(cexpr_name.to_string(), host_cexpr);
        }

        // Output params: intra outputs keep the original callee param name so the
        // NEXT injection can find the stores.  External outputs are renamed to
        // fresh host params.
        let mut next_intra_store: Option<String> = None;
        for param in e_callee.params.iter().filter(|p| p.is_output) {
            let (tensor_name, orig_slot_ref) = e_node
                .output_bindings
                .iter()
                .find(|(n, _)| n == &param.name)
                .and_then(|(_, slot)| slot_ref_name(slot).map(|n| (n, slot.clone())))?;
            if intra_tensors.contains(&tensor_name) {
                // Intra: do not rename — the next iteration searches for stores
                // to this param's original name in the host IR.
                next_intra_store = Some(param.name.clone());
            } else {
                let host_name = make_param_name(&tensor_name, &mut used_param_names);
                host.params.push(Param {
                    name: host_name.clone(),
                    dtype,
                    shape: Shape::scalar(),
                    is_output: true,
                    kind: param.kind.clone(),
                });
                fused_output_bindings.push((host_name.clone(), orig_slot_ref));
                e_param_renames.insert(param.name.clone(), host_name);
            }
        }

        inject_elementwise_epilogue(
            &mut host,
            &current_intra_store_name,
            e_callee,
            &e_intra_input_param,
            &e_param_renames,
            &mut next_vid,
            &HashMap::new(),  // no extra intra substitutions
            None,              // no capture
        );

        if let Some(next_name) = next_intra_store {
            current_intra_store_name = next_name;
        }
    }

    if fused_output_bindings.is_empty() {
        return None;
    }

    let fused_node = DispatchNode {
        label: format!("fused.group.{group_id}"),
        kernel_name,
        kernel_ir: fused_kernel_placeholder,
        mode: r_fused_mode.clone(),
        input_bindings: fused_input_bindings,
        output_bindings: fused_output_bindings,
        cexprs: fused_cexprs,
        grid: fused_grid,
        dtype,
        grid_dims,
        fuse_group: None,
    };

    debug!(
        "fuse group {group_id}: Pattern 1 {} direct IR synthesis ({} nodes → 1, num_rows={num_rows})",
        if matches!(r_node.mode, KernelMode::Grid3D) { "Grid3D→…" } else { "R→…" },
        group_nodes.len(),
    );

    Some((fused_node, host))
}

// ---------------------------------------------------------------------------
// Sub-pair decomposition — fallback when full-group synthesis fails
// ---------------------------------------------------------------------------

/// When a multi-node fuse group fails both Pattern 1 and Path B synthesis,
/// attempt to find consecutive 2-node `(Reduction → Elementwise)` sub-pairs
/// within the group and synthesize each pair independently via Pattern 1.
///
/// Returns `(local_start, local_end_exclusive, fused_node, fused_kernel)` for
/// every sub-pair that synthesized successfully.  Pairs that fail are skipped;
/// the nodes at those positions remain unchanged in the output of
/// `apply_subpair_decomposition`.
///
/// This handles the `ffn_act` pattern:
///
/// ```text
/// gemv_gate (R) → _gate → silu (E) → _gated ─┐
///                                               └→ mul (E) → _combined
/// gemv_up   (R) ─────────────────── → _up   ───┘
/// ```
///
/// Pairs found: `(gemv_gate, silu)` and `(gemv_up, mul)` — each a valid
/// 2-node R→E group. After synthesis both are fused via Pattern 1, reducing
/// 4 dispatches to 2 per layer (×16 layers = 32 fewer dispatches/token).
fn try_subpair_decomposition(
    group_nodes: &[&DispatchNode],
    group_cached: &[&Kernel],
    node_indices: &[usize],
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
    dtype: DType,
    group_id: usize,
) -> Vec<(usize, usize, DispatchNode, Kernel)> {
    if group_nodes.len() < 2 {
        return Vec::new();
    }

    let mut results: Vec<(usize, usize, DispatchNode, Kernel)> = Vec::new();
    let mut local = 0usize;

    while local + 1 < group_nodes.len() {
        let pair_nodes = [group_nodes[local], group_nodes[local + 1]];
        let pair_cached = [group_cached[local], group_cached[local + 1]];
        let pair_indices = [node_indices[local], node_indices[local + 1]];

        // Compute intra tensors scoped to this 2-node sub-pair.
        let pair_intra = find_pure_intra_tensors(&pair_indices, &HashMap::new(), writes, reads);

        // Use a unique sub-group ID so kernel names don't collide.
        let sub_group_id = group_id.saturating_mul(1000).saturating_add(local);

        let pair_result = synthesize_pattern1(
            &pair_nodes,
            &pair_cached,
            &pair_intra,
            dtype,
            sub_group_id,
        )
        .or_else(|| {
            let (fused_mode, fused_grid) = check_grid_compatibility(&pair_nodes)?;
            build_fused_group(
                &pair_nodes,
                &pair_cached,
                &pair_indices,
                &pair_intra,
                dtype,
                fused_mode,
                fused_grid,
                sub_group_id,
            )
        });

        if let Some((fused_node, fused_kernel)) = pair_result {
            results.push((local, local + 2, fused_node, fused_kernel));
            local += 2;
        } else {
            local += 1;
        }
    }

    results
}

/// Apply the results of `try_subpair_decomposition` to the node/kernel/tracking
/// vecs.  Synthesized pairs are collapsed into one fused node each; the
/// remaining nodes in the group are kept as-is.
///
/// Splices the group range `[start..end]` in place; adjusts `prefill_node_count`
/// if any nodes before the boundary were removed.
fn apply_subpair_decomposition(
    subpairs: Vec<(usize, usize, DispatchNode, Kernel)>,
    node_indices: &[usize],
    nodes: &mut Vec<DispatchNode>,
    cached_kernels: &mut Vec<Kernel>,
    intermediate_outputs: &mut Vec<Vec<(String, usize)>>,
    intermediate_inputs: &mut Vec<Vec<String>>,
    writes: &HashMap<String, Vec<usize>>,
    reads: &HashMap<String, Vec<usize>>,
    prefill_node_count: &mut usize,
    group_id: usize,
) {
    // Build the replacement sequence for the group's global range.
    // For positions covered by a synthesized pair, emit the fused node.
    // For all other positions, clone the original node.

    // Map local_start → (local_end, fused_node, fused_kernel).
    let mut pair_map: HashMap<usize, (usize, DispatchNode, Kernel)> =
        subpairs.into_iter().map(|(ls, le, n, k)| (ls, (le, n, k))).collect();

    let mut repl_nodes: Vec<DispatchNode> = Vec::new();
    let mut repl_kernels: Vec<Kernel> = Vec::new();
    let mut repl_int_out: Vec<Vec<(String, usize)>> = Vec::new();
    let mut repl_int_in: Vec<Vec<String>> = Vec::new();

    let mut local = 0usize;
    while local < node_indices.len() {
        if let Some((le, fused_node, fused_kernel)) = pair_map.remove(&local) {
            // Synthesized sub-pair: compute scoped intra tensors for tracking.
            let sp_indices = &node_indices[local..le];
            let sp_intra = find_pure_intra_tensors(sp_indices, &HashMap::new(), writes, reads);

            let fi_out: Vec<(String, usize)> = {
                let mut seen = HashSet::new();
                (local..le)
                    .flat_map(|j| intermediate_outputs[node_indices[j]].iter().cloned())
                    .filter(|(name, _)| !sp_intra.contains(name.as_str()))
                    .filter(|(name, _)| seen.insert(name.clone()))
                    .collect()
            };
            let fi_in: Vec<String> = {
                let mut seen = HashSet::new();
                (local..le)
                    .flat_map(|j| intermediate_inputs[node_indices[j]].iter().cloned())
                    .filter(|name| !sp_intra.contains(name.as_str()))
                    .filter(|name| seen.insert(name.clone()))
                    .collect()
            };

            repl_nodes.push(fused_node);
            repl_kernels.push(fused_kernel);
            repl_int_out.push(fi_out);
            repl_int_in.push(fi_in);
            local = le;
        } else {
            // Unsynthesized node: keep as-is.
            let ni = node_indices[local];
            repl_nodes.push(nodes[ni].clone());
            repl_kernels.push(cached_kernels[ni].clone());
            repl_int_out.push(intermediate_outputs[ni].clone());
            repl_int_in.push(intermediate_inputs[ni].clone());
            local += 1;
        }
    }

    let n_fused_pairs = repl_nodes.iter().filter(|n| n.label.starts_with("fused.group.")).count();
    debug!(
        "fuse group {group_id}: sub-pair decomposition → {} synthesized pairs, {} nodes total",
        n_fused_pairs,
        repl_nodes.len(),
    );

    let start = node_indices[0];
    let end = *node_indices.last().unwrap() + 1;
    let n_removed = (end - start) - repl_nodes.len();

    if start < *prefill_node_count {
        let removed_before = n_removed.min(*prefill_node_count - start);
        *prefill_node_count -= removed_before;
    }

    nodes.splice(start..end, repl_nodes);
    cached_kernels.splice(start..end, repl_kernels);
    intermediate_outputs.splice(start..end, repl_int_out);
    intermediate_inputs.splice(start..end, repl_int_in);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use metaltile_core::{DType, ir::{BinOpKind, IndexExpr, KernelMode, VarId}};
    use metaltile_runtime::context::GridSpec;

    use super::*;

    fn make_reduction_node(
        label: &str,
        inputs: Vec<(&str, &str)>,
        outputs: Vec<(&str, &str)>,
        cexprs: Vec<(&str, u32)>,
        num_rows: usize,
        tpg: usize,
        fuse_group: Option<usize>,
    ) -> DispatchNode {
        DispatchNode {
            label: label.to_string(),
            kernel_name: "stub_reduction",
            kernel_ir: crate::fuse_group::fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: inputs
                .into_iter()
                .map(|(p, t)| (p.to_string(), SlotRef::Weight(t.to_string())))
                .collect(),
            output_bindings: outputs
                .into_iter()
                .map(|(p, t)| (p.to_string(), SlotRef::Weight(t.to_string())))
                .collect(),
            cexprs: cexprs
                .into_iter()
                .map(|(n, v)| (n.to_string(), ConstexprValue::Static(v)))
                .collect(),
            grid: GridSpec::Reduction { num_rows, threads_per_group: tpg },
            dtype: DType::F32,
            grid_dims: ([num_rows, 1, 1], [tpg, 1, 1]),
            fuse_group,
        }
    }

    fn make_elementwise_node(
        label: &str,
        inputs: Vec<(&str, &str)>,
        outputs: Vec<(&str, &str)>,
        n: usize,
        fuse_group: Option<usize>,
    ) -> DispatchNode {
        DispatchNode {
            label: label.to_string(),
            kernel_name: "stub_elementwise",
            kernel_ir: crate::fuse_group::fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: inputs
                .into_iter()
                .map(|(p, t)| (p.to_string(), SlotRef::Weight(t.to_string())))
                .collect(),
            output_bindings: outputs
                .into_iter()
                .map(|(p, t)| (p.to_string(), SlotRef::Weight(t.to_string())))
                .collect(),
            cexprs: Vec::new(),
            grid: GridSpec::Elementwise { n },
            dtype: DType::F32,
            grid_dims: ([n.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group,
        }
    }

    /// Regression test: `clone_callee_into_host` must NOT apply `vid_offset` to
    /// loop-variable VIDs (those using the `0x4000_0000` prefix family).
    ///
    /// A reduction kernel with a `for` loop has ops in its loop body block that
    /// reference the loop counter as `ValueId(VarId.as_u32() + 0x4000_0000)`.
    /// Before the SPECIAL-mask fix (`0x8000_0000` → `0x4000_0000`), this VID was
    /// incorrectly shifted by `vid_offset`, producing an undeclared variable in
    /// the generated MSL (e.g. `v1073741825`).
    #[test]
    fn clone_callee_preserves_loop_var_vids() {
        // Loop-var VID for VarId(1) — the encoding used by the MSL emitter.
        const LOOP_VAR_VID: u32 = 0x4000_0001;

        // Build a minimal reduction kernel with one loop:
        //   body: ProgramId→v0, Const(0)→v1, Const(8)→v2, Const(1)→v3,
        //         Loop(var=VarId(1), body=BlockId(1))
        //   B1 (loop body): Load(vec[v0x4000_0001])→v4
        let mut callee = Kernel::new("loop_reduction");
        callee.mode = KernelMode::Reduction;
        callee.params = vec![Param {
            name: "vec".to_string(),
            dtype: DType::F32,
            shape: metaltile_core::shape::Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor,
        }];

        let pid = ValueId::new(0);
        callee.body.push_op(Op::ProgramId { axis: 0 }, pid);
        let start = ValueId::new(1);
        callee.body.push_op(Op::Const { value: 0 }, start);
        let end = ValueId::new(2);
        callee.body.push_op(Op::Const { value: 8 }, end);
        let step = ValueId::new(3);
        callee.body.push_op(Op::Const { value: 1 }, step);
        callee.body.push_op_no_result(Op::Loop {
            var: VarId::new(1),
            start,
            end,
            step,
            body: BlockId::new(1),
        });

        // Loop body block — uses the loop-var VID as an index.
        let mut loop_blk = Block::new(BlockId::new(1));
        let loop_var_vid = ValueId::new(LOOP_VAR_VID);
        let loaded = ValueId::new(4);
        loop_blk.push_op(
            Op::Load {
                src: "vec".to_string(),
                indices: vec![IndexExpr::Value(loop_var_vid)],
                mask: None,
                other: None,
            },
            loaded,
        );
        callee.blocks.insert(BlockId::new(1), loop_blk);

        // Clone into a fresh host.
        let mut host = Kernel::new("host");
        let vid_offset = clone_callee_into_host(&mut host, &callee, &HashMap::new());

        // The loop-body block should now be in host.blocks (with block offset applied).
        // Find the Load op and check its index VID.
        let loop_var_in_clone = host.blocks.values().any(|blk| {
            blk.ops.iter().any(|op| {
                if let Op::Load { indices, .. } = op {
                    indices.iter().any(|ix| matches!(ix, IndexExpr::Value(v) if v.as_u32() == LOOP_VAR_VID))
                } else {
                    false
                }
            })
        });
        assert!(
            loop_var_in_clone,
            "loop-var VID 0x{LOOP_VAR_VID:08x} must be preserved unshifted in the cloned host"
        );

        // Verify the WRONG (shifted) VID does NOT appear.
        let shifted = LOOP_VAR_VID + vid_offset;
        let shifted_in_clone =
            host.blocks.values().chain(std::iter::once(&host.body)).any(|blk| {
                blk.ops.iter().any(|op| {
                    if let Op::Load { indices, .. } = op {
                        indices.iter().any(|ix| matches!(ix, IndexExpr::Value(v) if v.as_u32() == shifted))
                    } else {
                        false
                    }
                })
            });
        assert!(
            !shifted_in_clone,
            "shifted loop-var VID 0x{shifted:08x} must NOT appear (would cause 'undeclared identifier' in MSL)"
        );
    }

    #[test]
    fn check_grid_compat_all_reduction_same_dims() {
        let n0 = make_reduction_node("a", vec![], vec![], vec![], 1024, 256, None);
        let n1 = make_reduction_node("b", vec![], vec![], vec![], 1024, 256, None);
        let result = check_grid_compatibility(&[&&n0, &&n1]);
        assert!(result.is_some());
        let (mode, grid) = result.unwrap();
        assert_eq!(mode, KernelMode::Reduction);
        assert!(matches!(grid, GridSpec::Reduction { num_rows: 1024, threads_per_group: 256 }));
    }

    #[test]
    fn check_grid_compat_mixed_ok() {
        let r = make_reduction_node("r", vec![], vec![], vec![], 512, 256, None);
        let e = make_elementwise_node("e", vec![], vec![], 512, None);
        let result = check_grid_compatibility(&[&&r, &&e]);
        assert!(result.is_some());
        let (mode, _) = result.unwrap();
        assert_eq!(mode, KernelMode::Reduction); // Reduction wins
    }

    #[test]
    fn check_grid_compat_mixed_mismatch() {
        let r = make_reduction_node("r", vec![], vec![], vec![], 512, 256, None);
        let e = make_elementwise_node("e", vec![], vec![], 256, None); // different n
        assert!(check_grid_compatibility(&[&&r, &&e]).is_none());
    }

    #[test]
    fn check_grid_compat_mismatched_reduction_rows() {
        let n0 = make_reduction_node("a", vec![], vec![], vec![], 512, 256, None);
        let n1 = make_reduction_node("b", vec![], vec![], vec![], 1024, 256, None);
        assert!(check_grid_compatibility(&[&&n0, &&n1]).is_none());
    }

    #[test]
    fn find_pure_intra_simple() {
        // Node 0 writes _gate; node 1 reads _gate.
        // Node 0 and 1 are both in group 0.
        let n0 = make_reduction_node(
            "gemv",
            vec![("mat", "weight_a"), ("vec", "_x")],
            vec![("out", "_gate")],
            vec![],
            1024,
            256,
            Some(0),
        );
        let n1 = make_elementwise_node(
            "silu",
            vec![("a", "_gate")],
            vec![("out", "_gated")],
            1024,
            Some(0),
        );
        let n2 = make_reduction_node(
            "post",
            vec![("a", "_gated")],
            vec![("out", "_result")],
            vec![],
            1024,
            256,
            None, // outside group
        );

        let nodes = vec![n0, n1, n2];
        let group_node_set: HashMap<usize, usize> =
            [(0usize, 0usize), (1usize, 0usize)].into_iter().collect();
        let (writes, reads) = build_write_read_maps(&nodes);

        let intra = find_pure_intra_tensors(&[0, 1], &group_node_set, &writes, &reads);
        // _gate: written by 0 (group), read by 1 (group) → intra ✓
        // _x, _gated: not written by group members → not intra
        assert!(intra.contains("_gate"), "expected _gate to be intra");
        assert!(!intra.contains("_gated"), "_gated is consumed outside group");
        assert!(!intra.contains("_x"), "_x not written by group");
    }

    // ── Pattern 1 direct IR synthesis tests ───────────────────────────────

    /// Build a minimal gemv-style Reduction kernel:
    /// params: `mat` (input), `vec_in` (input), `out` (output)
    /// constexprs: `k`, `rows`
    /// body: ProgramId(0) → Const(0) → Store(out[pid], const)
    fn make_gemv_reduction_kernel() -> Kernel {
        let mut k = Kernel::new("gemv_bm4");
        k.mode = KernelMode::Reduction;
        k.params = vec![
            Param {
                name: "mat".to_string(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: false,
                kind: ParamKind::Tensor,
            },
            Param {
                name: "vec_in".to_string(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: false,
                kind: ParamKind::Tensor,
            },
            Param {
                name: "out".to_string(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: true,
                kind: ParamKind::Tensor,
            },
        ];
        k.constexprs = vec![
            ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None },
            ConstExprDecl { name: ConstExpr::new("rows"), dtype: DType::U32, value: None },
        ];
        let pid = ValueId::new(0);
        k.body.push_op(Op::ProgramId { axis: 0 }, pid);
        let val = ValueId::new(1);
        k.body.push_op(Op::Const { value: 0 }, val);
        k.body.push_op_no_result(Op::Store {
            dst: "out".to_string(),
            indices: vec![IndexExpr::Value(pid)],
            value: val,
            mask: None,
        });
        k
    }

    /// Build a minimal binary/add Elementwise kernel:
    /// params: `a` (input, will be intra), `b` (input, external), `c` (output)
    /// body: ProgramId(0) → Load a → Load b → BinOp::Add → Store c
    fn make_add_elementwise_kernel() -> Kernel {
        let mut k = Kernel::new("binary_add");
        k.mode = KernelMode::Elementwise;
        k.params = vec![
            Param {
                name: "a".to_string(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: false,
                kind: ParamKind::Tensor,
            },
            Param {
                name: "b".to_string(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: false,
                kind: ParamKind::Tensor,
            },
            Param {
                name: "c".to_string(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: true,
                kind: ParamKind::Tensor,
            },
        ];
        let pid = ValueId::new(0);
        k.body.push_op(Op::ProgramId { axis: 0 }, pid);
        let val_a = ValueId::new(1);
        k.body.push_op(
            Op::Load { src: "a".to_string(), indices: vec![IndexExpr::Value(pid)], mask: None, other: None },
            val_a,
        );
        let val_b = ValueId::new(2);
        k.body.push_op(
            Op::Load { src: "b".to_string(), indices: vec![IndexExpr::Value(pid)], mask: None, other: None },
            val_b,
        );
        let val_sum = ValueId::new(3);
        k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: val_a, rhs: val_b }, val_sum);
        k.body.push_op_no_result(Op::Store {
            dst: "c".to_string(),
            indices: vec![IndexExpr::Value(pid)],
            value: val_sum,
            mask: None,
        });
        k
    }

    #[test]
    fn pattern1_synthesizes_attn_out_group() {
        // Simulate the `attn_out` fuse group from llama_decode.toml:
        //   gemv/bm4  → rows = hidden_dim/4 = 1024,  grid: Reduction(1024, 256)
        //   binary/add → n   = hidden_dim   = 4096,  grid: Elementwise(4096)
        // Pattern 1 should fire (n = 4 × num_rows, strictly mismatched).

        const HIDDEN_DIM: usize = 4096;
        const NUM_ROWS: usize = HIDDEN_DIM / 4; // 1024

        let r_kernel = make_gemv_reduction_kernel();
        let e_kernel = make_add_elementwise_kernel();

        let r_node = DispatchNode {
            label: "gemv_bm4.attn_out".to_string(),
            kernel_name: "gemv_bm4",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("layers.0.attn.o_proj".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_attn_out".to_string())),
            ],
            output_bindings: vec![
                ("out".to_string(), SlotRef::Weight("_o_proj".to_string())),
            ],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(HIDDEN_DIM as u32)),
                ("rows".to_string(), ConstexprValue::Static(NUM_ROWS as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        let e_node = DispatchNode {
            label: "binary_add.attn_out".to_string(),
            kernel_name: "binary_add",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_o_proj".to_string())),
                ("b".to_string(), SlotRef::Weight("_residual".to_string())),
            ],
            output_bindings: vec![
                ("c".to_string(), SlotRef::Weight("_post_attn".to_string())),
            ],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: HIDDEN_DIM },
            dtype: DType::F32,
            grid_dims: ([HIDDEN_DIM.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        // _o_proj is pure intra: written by r_node, read only by e_node.
        let intra: HashSet<String> = ["_o_proj".to_string()].into_iter().collect();

        let result = synthesize_pattern1(&[&r_node, &e_node], &[&r_kernel, &e_kernel], &intra, DType::F32, 0);
        assert!(result.is_some(), "Pattern 1 should synthesize R(1024)→E(4096)");

        let (fused_node, fused_kernel) = result.unwrap();

        // Fused node inherits the Reduction grid.
        assert_eq!(fused_node.mode, KernelMode::Reduction);
        assert!(
            matches!(fused_node.grid, GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 }),
            "fused node must keep the Reduction grid"
        );
        assert_eq!(fused_node.fuse_group, None);

        // Output must be _post_attn (elementwise output), NOT _o_proj (intra).
        let out_tensors: Vec<&str> = fused_node
            .output_bindings
            .iter()
            .filter_map(|(_, s)| if let SlotRef::Weight(n) = s { Some(n.as_str()) } else { None })
            .collect();
        assert!(out_tensors.contains(&"_post_attn"), "output must include _post_attn");
        assert!(!out_tensors.contains(&"_o_proj"), "_o_proj must be eliminated as intra");

        // External inputs: o_proj weight + attn_out vector + residual.
        let in_tensors: Vec<&str> = fused_node
            .input_bindings
            .iter()
            .filter_map(|(_, s)| if let SlotRef::Weight(n) = s { Some(n.as_str()) } else { None })
            .collect();
        assert!(in_tensors.contains(&"_residual"), "fused node must expose _residual as an input");
        assert!(
            in_tensors.contains(&"layers.0.attn.o_proj"),
            "fused node must expose the weight tensor"
        );

        // Fused kernel must be Reduction mode.
        assert_eq!(fused_kernel.mode, KernelMode::Reduction);

        // No store to the intra param original name ("out") should remain.
        let stores_to_out = fused_kernel.body.ops.iter().filter(|op| {
            matches!(op, Op::Store { dst, .. } if dst == "out")
        }).count();
        assert_eq!(stores_to_out, 0, "intra stores to 'out' must be replaced by epilogue");

        // The epilogue store to the external output must be present.
        let has_epilogue_store = fused_kernel.body.ops.iter().any(|op| {
            if let Op::Store { dst, .. } = op {
                fused_node.output_bindings.iter().any(|(hn, _)| hn == dst)
            } else {
                false
            }
        });
        assert!(has_epilogue_store, "fused kernel must contain the epilogue store to the external output");
    }

    #[test]
    fn pattern1_accepts_n_equals_num_rows() {
        // When n == num_rows (ratio = 1), Pattern 1 should synthesize the group directly.
        // Previously this was deferred to build_fused_group; now Pattern 1 handles it.
        const DIM: usize = 1024;
        let r_kernel = make_gemv_reduction_kernel();
        let e_kernel = make_add_elementwise_kernel();

        let r_node = DispatchNode {
            label: "gemv".to_string(),
            kernel_name: "gemv",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("weight".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_x".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight("_y".to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(DIM as u32)),
                ("rows".to_string(), ConstexprValue::Static(DIM as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: DIM, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([DIM, 1, 1], [256, 1, 1]),
            fuse_group: Some(1),
        };
        let e_node = DispatchNode {
            label: "add".to_string(),
            kernel_name: "add",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_y".to_string())),
                ("b".to_string(), SlotRef::Weight("_z".to_string())),
            ],
            output_bindings: vec![("c".to_string(), SlotRef::Weight("_out".to_string()))],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: DIM },
            dtype: DType::F32,
            grid_dims: ([DIM.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(1),
        };

        let intra: HashSet<String> = ["_y".to_string()].into_iter().collect();
        let result =
            synthesize_pattern1(&[&r_node, &e_node], &[&r_kernel, &e_kernel], &intra, DType::F32, 1);
        assert!(result.is_some(), "n == num_rows (ratio=1) must be synthesized by Pattern 1");
        let (fused_node, _) = result.unwrap();
        assert_eq!(fused_node.mode, KernelMode::Reduction);
    }

    #[test]
    fn pattern1_three_node_chain() {
        // Verify that a 3-node linear chain [R, E1, E2] fuses into one kernel.
        // R(num_rows=512) → E1(n=2048) → E2(n=2048)
        // E1's output is intra (flows to E2), E2's output is external.
        const NUM_ROWS: usize = 512;
        const N: usize = 2048;

        let r_kernel = make_gemv_reduction_kernel();
        let e1_kernel = make_add_elementwise_kernel(); // a=intra(_y), b=_ext1, c=_mid (intra to E2)
        let e2_kernel = make_add_elementwise_kernel(); // a=intra(_mid), b=_ext2, c=_final (external)

        let r_node = DispatchNode {
            label: "gemv".to_string(),
            kernel_name: "gemv",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("w".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_x".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight("_y".to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(N as u32)),
                ("rows".to_string(), ConstexprValue::Static(NUM_ROWS as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: Some(5),
        };
        let e1_node = DispatchNode {
            label: "e1".to_string(),
            kernel_name: "add",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_y".to_string())),   // intra from R
                ("b".to_string(), SlotRef::Weight("_ext1".to_string())),
            ],
            output_bindings: vec![("c".to_string(), SlotRef::Weight("_mid".to_string()))],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: N },
            dtype: DType::F32,
            grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(5),
        };
        let e2_node = DispatchNode {
            label: "e2".to_string(),
            kernel_name: "add",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_mid".to_string())),  // intra from E1
                ("b".to_string(), SlotRef::Weight("_ext2".to_string())),
            ],
            output_bindings: vec![("c".to_string(), SlotRef::Weight("_final".to_string()))],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: N },
            dtype: DType::F32,
            grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(5),
        };

        // Both _y (R→E1) and _mid (E1→E2) are intra.
        let intra: HashSet<String> =
            ["_y".to_string(), "_mid".to_string()].into_iter().collect();

        let result = synthesize_pattern1(
            &[&r_node, &e1_node, &e2_node],
            &[&r_kernel, &e1_kernel, &e2_kernel],
            &intra,
            DType::F32,
            5,
        );
        assert!(result.is_some(), "3-node chain R→E1→E2 must be synthesized by Pattern 1");

        let (fused_node, fused_kernel) = result.unwrap();

        // Grid inherits from R.
        assert_eq!(fused_node.mode, KernelMode::Reduction);
        assert!(
            matches!(fused_node.grid, GridSpec::Reduction { num_rows: NUM_ROWS, .. }),
            "fused node must keep R's grid"
        );

        // Only the final external output (_final) appears in output_bindings.
        let out_tensors: Vec<&str> = fused_node
            .output_bindings
            .iter()
            .filter_map(|(_, s)| if let SlotRef::Weight(n) = s { Some(n.as_str()) } else { None })
            .collect();
        assert!(out_tensors.contains(&"_final"), "output must include _final");
        assert!(!out_tensors.contains(&"_y"), "_y is intra, must not appear in outputs");
        assert!(!out_tensors.contains(&"_mid"), "_mid is intra, must not appear in outputs");

        // External inputs: w, _x, _ext1, _ext2.
        let in_tensors: Vec<&str> = fused_node
            .input_bindings
            .iter()
            .filter_map(|(_, s)| if let SlotRef::Weight(n) = s { Some(n.as_str()) } else { None })
            .collect();
        assert!(in_tensors.contains(&"_ext1"), "must expose _ext1");
        assert!(in_tensors.contains(&"_ext2"), "must expose _ext2");

        // No stores to the intra param names should remain in the fused kernel.
        let stores_to_y =
            fused_kernel.body.ops.iter().filter(|op| matches!(op, Op::Store { dst, .. } if dst == "out")).count();
        assert_eq!(stores_to_y, 0, "all intra stores to 'out' (R's param) must be replaced");
    }

    #[test]
    fn pattern1_skips_non_flat_elementwise() {
        // If the elementwise callee has sub-blocks (non-flat), Pattern 1 must decline.
        const NUM_ROWS: usize = 512;
        const N: usize = 2048;

        let r_kernel = make_gemv_reduction_kernel();
        let mut e_kernel = make_add_elementwise_kernel();
        // Inject a dummy sub-block to make e_kernel non-flat.
        let extra = Block::new(BlockId::new(99));
        e_kernel.blocks.insert(BlockId::new(99), extra);

        let r_node = DispatchNode {
            label: "r".to_string(),
            kernel_name: "r",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("w".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_v".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight("_intra".to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(N as u32)),
                ("rows".to_string(), ConstexprValue::Static(NUM_ROWS as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: Some(2),
        };
        let e_node = DispatchNode {
            label: "e".to_string(),
            kernel_name: "e",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_intra".to_string())),
                ("b".to_string(), SlotRef::Weight("_ext".to_string())),
            ],
            output_bindings: vec![("c".to_string(), SlotRef::Weight("_final".to_string()))],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: N },
            dtype: DType::F32,
            grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(2),
        };

        let intra: HashSet<String> = ["_intra".to_string()].into_iter().collect();
        let result = synthesize_pattern1(&[&r_node, &e_node], &[&r_kernel, &e_kernel], &intra, DType::F32, 2);
        assert!(result.is_none(), "non-flat elementwise callee must be rejected by Pattern 1");
    }

    #[test]
    fn subpair_decomposition_fuses_ffn_act_pattern() {
        // Uses make_gemv_reduction_kernel() (params: mat, vec_in, out → k/rows constexprs)
        // and make_add_elementwise_kernel() (params: a [intra], b [external], c [output])
        // as stand-ins.  The synthesis doesn't care about semantics, only param names.
        //
        // Pair 0: gemv_gate (R) → _gate [intra] → silu_standin (E): a=_gate, b=_bias_g, c=_gated
        // Pair 1: gemv_up   (R) → _up   [intra] → mul_standin  (E): a=_gated, b=_up, c=_combined
        const NUM_ROWS: usize = 3456;
        const N: usize = 13824;

        let gemv_gate_kernel = make_gemv_reduction_kernel();
        let silu_kernel = make_add_elementwise_kernel(); // a=intra(_gate), b=external, c=_gated
        let gemv_up_kernel = make_gemv_reduction_kernel();
        let mul_kernel = make_add_elementwise_kernel(); // a=external(_gated), b=intra(_up), c=_combined

        let make_r = |label: &str, mat: &str, out: &str| DispatchNode {
            label: label.to_string(),
            kernel_name: "ffai_gemv_bm4",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight(mat.to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_ffn_normed".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight(out.to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(1024)),
                ("rows".to_string(), ConstexprValue::Static(N as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        // Add a downstream consumer of _combined (down_proj GEMV) so that
        // find_pure_intra_tensors correctly treats _combined as external.
        // Note: vec_in must bind to _combined (not _ffn_normed) to register a read.
        let down_proj = DispatchNode {
            label: "gemv_down".to_string(),
            kernel_name: "ffai_gemv_bm4",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("down_proj".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_combined".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight("_ffn_out".to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(N as u32)),
                ("rows".to_string(), ConstexprValue::Static(1024u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: None,
        };

        let mut nodes = vec![
            make_r("gemv_gate", "gate_proj", "_gate"),
            DispatchNode {
                // silu stand-in: a=_gate (intra), b=_ffn_normed (external), c=_gated
                label: "silu".to_string(),
                kernel_name: "mt_binary_standin",
                kernel_ir: fused_kernel_placeholder,
                mode: KernelMode::Elementwise,
                input_bindings: vec![
                    ("a".to_string(), SlotRef::Weight("_gate".to_string())),
                    ("b".to_string(), SlotRef::Weight("_ffn_normed".to_string())),
                ],
                output_bindings: vec![("c".to_string(), SlotRef::Weight("_gated".to_string()))],
                cexprs: vec![],
                grid: GridSpec::Elementwise { n: N },
                dtype: DType::F32,
                grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
                fuse_group: Some(0),
            },
            make_r("gemv_up", "up_proj", "_up"),
            DispatchNode {
                // mul stand-in: a=_gated (external after pair 0), b=_up (intra), c=_combined
                label: "mul".to_string(),
                kernel_name: "mt_binary_standin",
                kernel_ir: fused_kernel_placeholder,
                mode: KernelMode::Elementwise,
                input_bindings: vec![
                    ("a".to_string(), SlotRef::Weight("_gated".to_string())),
                    ("b".to_string(), SlotRef::Weight("_up".to_string())),
                ],
                output_bindings: vec![("c".to_string(), SlotRef::Weight("_combined".to_string()))],
                cexprs: vec![],
                grid: GridSpec::Elementwise { n: N },
                dtype: DType::F32,
                grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
                fuse_group: Some(0),
            },
            down_proj, // reads _combined → keeps it non-intra
        ];
        let mut cached_kernels =
            vec![gemv_gate_kernel, silu_kernel, gemv_up_kernel, mul_kernel, make_gemv_reduction_kernel()];
        let mut int_out: Vec<Vec<(String, usize)>> = vec![
            vec![("_gate".to_string(), 0)],
            vec![("_gated".to_string(), 1)],
            vec![("_up".to_string(), 2)],
            vec![("_combined".to_string(), 3)],
            vec![("_ffn_out".to_string(), 4)],
        ];
        let mut int_in: Vec<Vec<String>> = vec![
            vec!["_ffn_normed".to_string()],
            vec!["_gate".to_string()],
            vec!["_ffn_normed".to_string()],
            vec!["_gated".to_string(), "_up".to_string()],
            vec!["_combined".to_string()],
        ];
        let mut prefill_count = 5usize;

        synthesize_fuse_groups(
            &mut nodes, &mut cached_kernels, &mut int_out, &mut int_in,
            &mut prefill_count, DType::F32,
        );

        // 4 group nodes → 1 fused node by Pattern 3; downstream node stays → 2 total.
        assert_eq!(nodes.len(), 2, "4-node ffn_act group → 1 fused + 1 downstream");
        assert_eq!(nodes[0].mode, KernelMode::Reduction, "fused 4-node is Reduction");
        assert!(nodes[0].label.starts_with("fused.group."), "node 0: {}", nodes[0].label);

        let all_out: Vec<String> =
            int_out.iter().flat_map(|v| v.iter().map(|(n, _)| n.clone())).collect();
        assert!(!all_out.contains(&"_gate".to_string()), "_gate must be intra-eliminated");
        assert!(!all_out.contains(&"_up".to_string()), "_up must be intra-eliminated");
        // _gated is intra in the 4-node group (read only by mul within the group).
        assert!(!all_out.contains(&"_gated".to_string()), "_gated must be intra-eliminated");

        // _combined is the fused group output (down_proj reads it externally).
        let node0_names: Vec<String> = int_out[0].iter().map(|(n, _)| n.clone()).collect();
        assert!(node0_names.contains(&"_combined".to_string()), "_combined in fused outputs: {node0_names:?}");
    }

    #[test]
    fn find_pure_intra_external_reader_excludes_tensor() {
        // Node 0 (group) writes _gate; node 1 (group) AND node 2 (external) read _gate.
        let n0 =
            make_reduction_node("gemv", vec![], vec![("out", "_gate")], vec![], 1024, 256, Some(0));
        let n1 = make_elementwise_node(
            "silu",
            vec![("a", "_gate")],
            vec![("out", "_gated")],
            1024,
            Some(0),
        );
        let n2 = make_elementwise_node(
            "other",
            vec![("a", "_gate")],
            vec![("out", "_other")],
            1024,
            None,
        );

        let nodes = vec![n0, n1, n2];
        let group_node_set: HashMap<usize, usize> = [(0, 0), (1, 0)].into_iter().collect();
        let (writes, reads) = build_write_read_maps(&nodes);

        let intra = find_pure_intra_tensors(&[0, 1], &group_node_set, &writes, &reads);
        // _gate has an external reader (node 2) → NOT intra
        assert!(!intra.contains("_gate"), "_gate has external reader, must not be intra");
    }

    #[test]
    fn pattern3_synthesizes_ffn_act_diamond() {
        // Verify Pattern 3 synthesizes [R0, E0, R1, E1] diamond:
        // R0: gemv_gate (Reduction 512 rows, tpg=256)
        // E0: silu stand-in (Elementwise n=2048, reads R0's intra, writes _gated)
        // R1: gemv_up   (Reduction 512 rows, tpg=256)
        // E1: mul stand-in (Elementwise n=2048, reads R1's intra + _gated, writes _combined)
        const NUM_ROWS: usize = 512;
        const N: usize = 2048;

        let r_kernel = make_gemv_reduction_kernel();
        let e_kernel = make_add_elementwise_kernel();

        let r0_node = DispatchNode {
            label: "gemv_gate".to_string(),
            kernel_name: "gemv",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("w_gate".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_normed".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight("_gate".to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(N as u32)),
                ("rows".to_string(), ConstexprValue::Static(NUM_ROWS as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        let e0_node = DispatchNode {
            label: "silu".to_string(),
            kernel_name: "silu",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_gate".to_string())),
                ("b".to_string(), SlotRef::Weight("_normed".to_string())),
            ],
            output_bindings: vec![("c".to_string(), SlotRef::Weight("_gated".to_string()))],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: N },
            dtype: DType::F32,
            grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        let r1_node = DispatchNode {
            label: "gemv_up".to_string(),
            kernel_name: "gemv",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Reduction,
            input_bindings: vec![
                ("mat".to_string(), SlotRef::Weight("w_up".to_string())),
                ("vec_in".to_string(), SlotRef::Weight("_normed".to_string())),
            ],
            output_bindings: vec![("out".to_string(), SlotRef::Weight("_up".to_string()))],
            cexprs: vec![
                ("k".to_string(), ConstexprValue::Static(N as u32)),
                ("rows".to_string(), ConstexprValue::Static(NUM_ROWS as u32)),
            ],
            grid: GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 },
            dtype: DType::F32,
            grid_dims: ([NUM_ROWS, 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        let e1_node = DispatchNode {
            label: "mul".to_string(),
            kernel_name: "mul",
            kernel_ir: fused_kernel_placeholder,
            mode: KernelMode::Elementwise,
            input_bindings: vec![
                ("a".to_string(), SlotRef::Weight("_gated".to_string())),
                ("b".to_string(), SlotRef::Weight("_up".to_string())),
            ],
            output_bindings: vec![("c".to_string(), SlotRef::Weight("_combined".to_string()))],
            cexprs: vec![],
            grid: GridSpec::Elementwise { n: N },
            dtype: DType::F32,
            grid_dims: ([N.div_ceil(256), 1, 1], [256, 1, 1]),
            fuse_group: Some(0),
        };

        // _gate, _gated, _up all intra (only consumed within the 4-node group).
        let intra: HashSet<String> =
            ["_gate".to_string(), "_gated".to_string(), "_up".to_string()].into_iter().collect();

        let group_nodes = [&r0_node, &e0_node, &r1_node, &e1_node];
        let group_cached = [&r_kernel, &e_kernel, &r_kernel, &e_kernel];

        let result = synthesize_pattern3(
            &group_nodes, &group_cached, &intra, DType::F32, 0,
        );
        assert!(result.is_some(), "Pattern 3 should synthesize 4-node diamond");

        let (fused_node, fused_kernel) = result.unwrap();
        assert_eq!(fused_node.mode, KernelMode::Reduction);
        assert!(
            matches!(fused_node.grid, GridSpec::Reduction { num_rows: NUM_ROWS, threads_per_group: 256 }),
            "fused node keeps Reduction grid"
        );

        // _combined is the external output.
        let out_tensors: Vec<&str> = fused_node
            .output_bindings
            .iter()
            .filter_map(|(_, s)| if let SlotRef::Weight(n) = s { Some(n.as_str()) } else { None })
            .collect();
        assert!(out_tensors.contains(&"_combined"), "output must include _combined: {out_tensors:?}");
        assert!(!out_tensors.contains(&"_gate"), "_gate must be intra-eliminated");
        assert!(!out_tensors.contains(&"_gated"), "_gated must be intra-eliminated");
        assert!(!out_tensors.contains(&"_up"), "_up must be intra-eliminated");

        // Inputs include w_gate, w_up, _normed.
        let in_tensors: Vec<&str> = fused_node
            .input_bindings
            .iter()
            .filter_map(|(_, s)| if let SlotRef::Weight(n) = s { Some(n.as_str()) } else { None })
            .collect();
        assert!(in_tensors.contains(&"w_gate"), "input must include w_gate");
        assert!(in_tensors.contains(&"w_up"), "input must include w_up");
        assert!(in_tensors.contains(&"_normed"), "input must include _normed");

        // Fused kernel is Reduction mode.
        assert_eq!(fused_kernel.mode, KernelMode::Reduction);
    }
}

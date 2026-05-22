//! TOML fuse-group kernel body synthesis.
//!
//! Synthesizes a single "host kernel" for each compatible fuse group.
//! The host kernel contains `Op::KernelCall` chains that `KernelInlinePass`
//! resolves into a single flat kernel at MSL generation time.
//!
//! ## Approach
//!
//! For each contiguous fuse group (nodes sharing a `fuse_group` ID):
//!
//! 1. **Grid compatibility** — all nodes must be `Elementwise` or `Reduction`
//!    with consistent dimensions.  `Grid3D`, `Tile2D`, and `SimdGroup2D` are
//!    always incompatible.  Mixed Elementwise+Reduction is allowed when all
//!    Elementwise nodes have `n == num_rows` of the Reduction nodes.
//!
//! 2. **Intra-group tensor detection** — an intermediate tensor is "pure intra"
//!    when it is written by a group node and ALL reads between that write and
//!    the next write to the same name come from other group nodes.  Intra
//!    tensors flow as scalar `Value` args instead of round-tripping through
//!    global memory.
//!
//! 3. **Host kernel construction** — a new `Kernel` is built with one
//!    `Op::ProgramId(0)` and one `Op::KernelCall` per group node.  Input
//!    params that are intra tensors are forwarded as `KernelCallArg::Value(vid)`;
//!    all other params (weights, state, non-intra intermediates, constexprs)
//!    become `KernelCallArg::Tensor(host_param_name)`.
//!
//! 4. **Node replacement** — the N group nodes are replaced by 1 fused node
//!    in `nodes` and `cached_kernels`; intermediate tracking is updated to
//!    exclude eliminated intra tensors.

use std::collections::{HashMap, HashSet};

use metaltile_core::{
    DType,
    constexpr::ConstExpr,
    ir::{ConstExprDecl, Kernel, KernelCallArg, KernelMode, Op, Param, ParamKind, ValueId},
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

        // Only synthesize compatible groups.
        let Some((fused_mode, fused_grid)) = check_grid_compatibility(&group_nodes) else {
            debug!("fuse group {group_id}: incompatible grids, skipping synthesis");
            continue;
        };

        let intra_tensors = find_pure_intra_tensors(node_indices, &group_node_set, &writes, &reads);

        let Some((fused_node, fused_kernel)) = build_fused_group(
            &group_nodes,
            &group_cached,
            node_indices,
            &intra_tensors,
            dtype,
            fused_mode,
            fused_grid,
            *group_id,
        ) else {
            debug!("fuse group {group_id}: synthesis failed, skipping");
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use metaltile_core::{DType, ir::KernelMode};
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
}

//! Execution plan executor: dispatches an `ExecutionPlan` on the GPU.
//!
//! Each `DispatchNode` is dispatched individually via a single-spec
//! `Context::dispatch_chain` call. Intermediate tensors are stored
//! directly in GPU-resident `ResidentBuffer`s ("slot_data") — on Apple
//! Silicon's unified memory architecture this avoids the alloc+copy+
//! readback+clone round-trips that `Vec<Vec<u8>>` would incur. Weights
//! live in the caller-provided `resident` map (pre-uploaded GPU buffers)
//! or fall back to `weights` for CPU→GPU upload per dispatch.
//!
//! Constexpr values are placed in `spec.buffers` (as 4-byte LE scalars)
//! so `dispatch_chain` binds them via `setBytes`. The `fn_consts` field
//! is always empty — MetalTile's `#[constexpr]` params use the buffer
//! binding path, not Metal function constants.
//!
//! State tensors (kv_cache, position, etc.) are read from and written
//! back to the mutable `state` map after each dispatch that produces
//! a state output.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer, context::GridSpec};

use crate::{
    DispatchNode,
    error::ModelError,
    plan::{ConstexprValue, ExecutionPlan, SlotRef},
};

/// Watchdog timeout: if a single dispatch_chain call takes longer than this,
/// the process is killed to prevent an indefinite GPU hang from freezing macOS.
const GPU_WATCHDOG_SECS: u64 = 20;

/// GPU buffer storage keyed by tensor name.
pub type WeightMap = HashMap<String, Vec<u8>>;

/// Runtime state buffers (kv_cache, position counters, etc.).
pub type StateMap = HashMap<String, Vec<u8>>;

/// Static empty function-constants map (shared across all dispatches).
static EMPTY_FN_CONSTS: std::sync::OnceLock<BTreeMap<String, u32>> = std::sync::OnceLock::new();

/// Execute a plan on the GPU and read back the final output buffer.
///
/// This is the primary entry point for inference. Each `DispatchNode`
/// is dispatched individually (or as part of a fused group):
/// 1. Intermediate tensors are stored in GPU-resident `ResidentBuffer`s
///    ("slot_data") — no host-side copies on Apple Silicon unified memory.
/// 2. Input slots bind directly via `spec.resident`; output slots allocate
///    fresh `ResidentBuffer`s bound via `spec.output_resident`.
/// 3. Add constexpr values to `spec.buffers` (4-byte LE scalars).
/// 4. For fused groups: collect all nodes with the same `fuse_group`,
///    build a multi-spec chain, and call `ctx.dispatch_chain(&specs)`.
///    Intra-group intermediates are connected through `output_resident`
///    → `resident` without touching `slot_data`.
/// 5. Non-fused nodes use the single-spec path (unchanged).
/// 6. Move output `ResidentBuffer`s into `slot_data`; write state outputs
///    to `state`. Only the final output slot is read back to host.
///
/// Returns the output bytes of the plan's final output slot and the total
/// GPU elapsed time in **microseconds** across all dispatched nodes.
pub fn execute_plan(
    ctx: &Context,
    plan: &ExecutionPlan,
    weights: &WeightMap,
    state: &mut StateMap,
    resident: &BTreeMap<String, ResidentBuffer>,
) -> Result<(Vec<u8>, f64), ModelError> {
    let fn_consts = EMPTY_FN_CONSTS.get_or_init(BTreeMap::new);

    // Pre-allocate GPU-resident slot buffers. On Apple Silicon unified
    // memory, these are just Metal buffer handles — no data copy occurs.
    let mut slot_data: Vec<ResidentBuffer> = plan
        .slots
        .iter()
        .map(|s| ctx.alloc_resident(s.size_bytes))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ModelError::Other(e.to_string()))?;

    // Accumulate GPU elapsed time across all nodes (microseconds).
    let mut total_gpu_us: f64 = 0.0;

    let mut node_idx = 0usize;
    while node_idx < plan.nodes.len() {
        let group = plan.nodes[node_idx].fuse_group;

        if group.is_some() {
            // ── Fused group dispatch ─────────────────────────
            // Collect all adjacent nodes in this group.
            let group_start = node_idx;
            while node_idx < plan.nodes.len() && plan.nodes[node_idx].fuse_group == group {
                node_idx += 1;
            }
            let group_nodes = &plan.nodes[group_start..node_idx];

            // Build a DispatchSpec for each node in the group.
            // Intra-group intermediate buffers are connected through
            // output_resident → resident without touching slot_data.
            // All per-node data is collected into Vecs first so they
            // outlive the `dispatch_chain` borrow.

            // Map from intermediate name → ResidentBuffer produced within
            // the chain (for connecting producer output to consumer input).
            let mut chain_intermediates: HashMap<String, ResidentBuffer> = HashMap::new();

            // Collect per-node data (must live until after dispatch_chain).
            let mut kernels: Vec<metaltile_core::ir::Kernel> = Vec::with_capacity(group_nodes.len());
            let mut all_buffers: Vec<BTreeMap<String, Vec<u8>>> = Vec::new();
            let mut all_resident: Vec<BTreeMap<String, ResidentBuffer>> = Vec::new();
            let mut all_output_resident: Vec<BTreeMap<String, ResidentBuffer>> = Vec::new();
            let mut all_grid: Vec<([usize; 3], [usize; 3])> = Vec::new();

            for node in group_nodes {
                let mut kernel = (node.kernel_ir)(node.dtype);
                kernel.mode = node.mode;
                let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
                let mut spec_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

                // ── Input bindings ─────────────────────────
                for (param_name, slot_ref) in &node.input_bindings {
                    match slot_ref {
                        SlotRef::Slot(idx) => {
                            if let Some(rb) = slot_data.get(*idx) {
                                spec_resident.insert(param_name.clone(), rb.clone());
                            }
                        },
                        SlotRef::Weight(tensor_name) => {
                            if let Some(rb) = chain_intermediates.get(tensor_name) {
                                spec_resident.insert(param_name.clone(), rb.clone());
                            } else if let Some(rb) = resident.get(tensor_name) {
                                spec_resident.insert(param_name.clone(), rb.clone());
                            } else if let Some(bytes) = weights.get(tensor_name) {
                                buffers.insert(param_name.clone(), bytes.clone());
                            }
                        },
                        SlotRef::State(key) => {
                            if let Some(rb) = resident.get(key) {
                                spec_resident.insert(param_name.clone(), rb.clone());
                            } else if let Some(bytes) = state.get(key) {
                                buffers.insert(param_name.clone(), bytes.clone());
                            }
                        },
                    }
                }

                // ── Output bindings ────────────────────────
                let mut spec_output_resident: BTreeMap<String, ResidentBuffer> =
                    BTreeMap::new();

                for (param_name, slot_ref) in &node.output_bindings {
                    match slot_ref {
                        SlotRef::Slot(idx) => {
                            let size =
                                plan.slots.get(*idx).map(|s| s.size_bytes).unwrap_or(0);
                            if size > 0 {
                                let rb = ctx
                                    .alloc_resident(size)
                                    .map_err(|e| ModelError::Other(e.to_string()))?;
                                spec_output_resident.insert(param_name.clone(), rb);
                            }
                        },
                        SlotRef::Weight(tensor_name)
                            if tensor_name.starts_with('_') =>
                        {
                            let size = estimate_intermediate_size(node);
                            if size > 0 {
                                let rb = ctx
                                    .alloc_resident(size)
                                    .map_err(|e| ModelError::Other(e.to_string()))?;
                                chain_intermediates
                                    .insert(tensor_name.clone(), rb.clone());
                                spec_output_resident.insert(param_name.clone(), rb);
                            }
                        },
                        SlotRef::State(key) => {
                            if let Some(rb) = resident.get(key) {
                                spec_output_resident
                                    .insert(param_name.clone(), rb.clone());
                            } else {
                                let size = state.get(key).map(|v| v.len()).unwrap_or(0);
                                if size > 0 {
                                    buffers
                                        .insert(param_name.clone(), vec![0u8; size]);
                                }
                            }
                        },
                        SlotRef::Weight(_) => {},
                    }
                }

                // ── Constexpr values ───────────────────────
                for (name, cv) in &node.cexprs {
                    let bits: u32 = match cv {
                        ConstexprValue::Static(val) => *val,
                        ConstexprValue::State(state_key) => {
                            let bytes = state
                                .get(state_key)
                                .and_then(|b| b.get(..4))
                                .ok_or_else(|| ModelError::UnsafeDispatch {
                                    op: node.label.clone(),
                                    detail: format!(
                                        "runtime state '{state_key}' not found — \
                                         ensure it is set in the state map before the forward pass"
                                    ),
                                })?;
                            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        },
                    };
                    buffers.insert(name.clone(), bits.to_le_bytes().to_vec());
                }

                // ── Pre-dispatch safety checks ─────────────
                for param in kernel.params.iter().filter(|p| !p.is_output) {
                    let in_resident = spec_resident.contains_key(&param.name);
                    let in_buffers =
                        buffers.get(&param.name).is_some_and(|b| !b.is_empty());
                    if !in_resident && !in_buffers {
                        return Err(ModelError::UnsafeDispatch {
                            op: node.label.clone(),
                            detail: format!(
                                "input '{}' has no data — weight missing from checkpoint \
                                 or weight-name mismatch; refusing GPU dispatch",
                                param.name
                            ),
                        });
                    }
                }

                all_grid.push(grid_to_dims(&node.grid));
                all_buffers.push(buffers);
                all_resident.push(spec_resident);
                all_output_resident.push(spec_output_resident);
                kernels.push(kernel);
            }

            // ── Build specs (now all data lives long enough) ──
            let mut chain_specs: Vec<DispatchSpec<'_>> = Vec::with_capacity(group_nodes.len());
            for gi in 0..group_nodes.len() {
                let (grid_groups, threads_per_group) = all_grid[gi];
                chain_specs.push(DispatchSpec {
                    kernel: &kernels[gi],
                    buffers: &all_buffers[gi],
                    fn_consts,
                    grid_groups,
                    threads_per_group,
                    resident: &all_resident[gi],
                    output_resident: &all_output_resident[gi],
                });
            }

            // ── Watchdog ───────────────────────────────────
            let done_flag = Arc::new(AtomicBool::new(false));
            {
                let flag = done_flag.clone();
                let label = group_nodes[0].label.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(GPU_WATCHDOG_SECS));
                    if !flag.load(Ordering::Acquire) {
                        eprintln!(
                            "\n[executor] WATCHDOG: fused dispatch '{}' exceeded {}s — \
                             killing process to prevent system freeze",
                            label, GPU_WATCHDOG_SECS
                        );
                        std::process::exit(1);
                    }
                });
            }

            // ── Multi-spec dispatch ───────────────────────
            let results = ctx
                .dispatch_chain(&chain_specs)
                .map_err(|e| ModelError::Other(e.to_string()))?;
            done_flag.store(true, Ordering::Release);

            if let Some(r) = results.first() {
                total_gpu_us += r.elapsed_us;
            }

            // ── Output readback ───────────────────────────
            // For each node in group, collect its output buffers.
            // all_output_resident buffers were moved into ChainSpecs —
            // retrieve them via the Vec we built.
            let mut results_iter = results.into_iter();
            for (gi, node) in group_nodes.iter().enumerate() {
                let Some(result) = results_iter.next() else { break };
                let mut output_resident = std::mem::take(&mut all_output_resident[gi]);
                for (param_name, slot_ref) in &node.output_bindings {
                    match slot_ref {
                        SlotRef::Slot(idx) => {
                            if let Some(rb) = output_resident.remove(param_name) {
                                slot_data[*idx] = rb;
                            }
                        },
                        SlotRef::State(key) => {
                            if !output_resident.contains_key(param_name) {
                                if let Some(bytes) = result.outputs.get(param_name) {
                                    state.insert(key.clone(), bytes.clone());
                                }
                            }
                        },
                        SlotRef::Weight(_) => {},
                    }
                }
            }
        } else {
            // ── Single-node dispatch (unchanged path) ──────
            let node = &plan.nodes[node_idx];
            node_idx += 1;

            let mut kernel = (node.kernel_ir)(node.dtype);
            kernel.mode = node.mode;
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();

            // ── Input bindings ─────────────────────────────
            let mut spec_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

            for (param_name, slot_ref) in &node.input_bindings {
                match slot_ref {
                    SlotRef::Slot(idx) => {
                        if let Some(rb) = slot_data.get(*idx) {
                            spec_resident.insert(param_name.clone(), rb.clone());
                        }
                    },
                    SlotRef::Weight(tensor_name) =>
                        if let Some(rb) = resident.get(tensor_name) {
                            spec_resident.insert(param_name.clone(), rb.clone());
                        } else if let Some(bytes) = weights.get(tensor_name) {
                            buffers.insert(param_name.clone(), bytes.clone());
                        },
                    SlotRef::State(key) => {
                        if let Some(rb) = resident.get(key) {
                            spec_resident.insert(param_name.clone(), rb.clone());
                        } else if let Some(bytes) = state.get(key) {
                            buffers.insert(param_name.clone(), bytes.clone());
                        }
                    },
                }
            }

            // ── Output bindings ────────────────────────────
            let mut spec_output_resident: BTreeMap<String, ResidentBuffer> =
                BTreeMap::new();

            for (param_name, slot_ref) in &node.output_bindings {
                match slot_ref {
                    SlotRef::Slot(idx) => {
                        let size =
                            plan.slots.get(*idx).map(|s| s.size_bytes).unwrap_or(0);
                        if size > 0 {
                            let rb = ctx
                                .alloc_resident(size)
                                .map_err(|e| ModelError::Other(e.to_string()))?;
                            spec_output_resident.insert(param_name.clone(), rb);
                        }
                    },
                    SlotRef::State(key) => {
                        if let Some(rb) = resident.get(key) {
                            spec_output_resident
                                .insert(param_name.clone(), rb.clone());
                        } else {
                            let size = state.get(key).map(|v| v.len()).unwrap_or(0);
                            if size > 0 {
                                buffers.insert(param_name.clone(), vec![0u8; size]);
                            }
                        }
                    },
                    SlotRef::Weight(_) => {},
                }
            }

            // ── Constexpr values ───────────────────────────
            for (name, cv) in &node.cexprs {
                let bits: u32 = match cv {
                    ConstexprValue::Static(val) => *val,
                    ConstexprValue::State(state_key) => {
                        let bytes = state
                            .get(state_key)
                            .and_then(|b| b.get(..4))
                            .ok_or_else(|| ModelError::UnsafeDispatch {
                                op: node.label.clone(),
                                detail: format!(
                                    "runtime state '{state_key}' not found — \
                                     ensure it is set in the state map before the forward pass"
                                ),
                            })?;
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                    },
                };
                buffers.insert(name.clone(), bits.to_le_bytes().to_vec());
            }

            // ── Pre-dispatch safety checks ─────────────────
            for param in kernel.params.iter().filter(|p| !p.is_output) {
                let in_resident = spec_resident.contains_key(&param.name);
                let in_buffers =
                    buffers.get(&param.name).is_some_and(|b| !b.is_empty());
                if !in_resident && !in_buffers {
                    return Err(ModelError::UnsafeDispatch {
                        op: node.label.clone(),
                        detail: format!(
                            "input '{}' has no data — weight missing from checkpoint \
                             or weight-name mismatch; refusing GPU dispatch",
                            param.name
                        ),
                    });
                }
            }

            let (grid_groups, threads_per_group) = grid_to_dims(&node.grid);

            // ── Watchdog ───────────────────────────────────
            let done_flag = Arc::new(AtomicBool::new(false));
            {
                let flag = done_flag.clone();
                let label = node.label.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(GPU_WATCHDOG_SECS));
                    if !flag.load(Ordering::Acquire) {
                        eprintln!(
                            "\n[executor] WATCHDOG: dispatch '{}' exceeded {}s — \
                             killing process to prevent system freeze",
                            label, GPU_WATCHDOG_SECS
                        );
                        std::process::exit(1);
                    }
                });
            }

            // ── Dispatch ───────────────────────────────────
            let spec = DispatchSpec {
                kernel: &kernel,
                buffers: &buffers,
                fn_consts,
                grid_groups,
                threads_per_group,
                resident: &spec_resident,
                output_resident: &spec_output_resident,
            };

            let results = ctx
                .dispatch_chain(&[spec])
                .map_err(|e| ModelError::Other(e.to_string()))?;
            done_flag.store(true, Ordering::Release);

            if let Some(r) = results.first() {
                total_gpu_us += r.elapsed_us;
            }

            let Some(result) = results.into_iter().next() else { continue };

            for (param_name, slot_ref) in &node.output_bindings {
                match slot_ref {
                    SlotRef::Slot(idx) => {
                        if let Some(rb) = spec_output_resident.remove(param_name) {
                            slot_data[*idx] = rb;
                        }
                    },
                    SlotRef::State(key) => {
                        if !spec_output_resident.contains_key(param_name) {
                            if let Some(bytes) = result.outputs.get(param_name) {
                                state.insert(key.clone(), bytes.clone());
                            }
                        }
                    },
                    SlotRef::Weight(_) => {},
                }
            }
        }
    }

    // Return bytes from the plan's designated output slot via a single
    // readback, plus accumulated GPU time in microseconds.
    let final_slot = &slot_data[plan.output_slot];
    let final_size = plan.slots[plan.output_slot].size_bytes;
    Ok((final_slot.read_bytes(final_size), total_gpu_us))
}


/// Convert a `GridSpec` to `(grid_groups: [usize; 3], threads_per_group: [usize; 3])`.
fn grid_to_dims(grid: &GridSpec) -> ([usize; 3], [usize; 3]) {
    match grid {
        GridSpec::Elementwise { n } => {
            let tpg = 256usize;
            let groups = n.div_ceil(tpg);
            ([groups, 1, 1], [tpg, 1, 1])
        },
        GridSpec::Reduction { num_rows, threads_per_group } =>
            ([*num_rows, 1, 1], [*threads_per_group, 1, 1]),
        GridSpec::Grid3D { x, y, z, threads_per_group } =>
            ([*x, *y, *z], [*threads_per_group, 1, 1]),
    }
}

/// Estimate the byte size of a node's intermediate output.
/// Used for intra-group intermediate buffers in fused dispatch chains.
fn estimate_intermediate_size(node: &DispatchNode) -> usize {
    match &node.grid {
        GridSpec::Elementwise { n } => {
            n * node.dtype.size_bytes()
        },
        GridSpec::Reduction { num_rows, .. } => {
            num_rows * node.dtype.size_bytes() * 32 // rough estimate
        },
        GridSpec::Grid3D { x, y, z, .. } => {
            let elems = x * y * z;
            if elems > 0 {
                elems * node.dtype.size_bytes()
            } else {
                x.max(y).max(z) * node.dtype.size_bytes()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_to_dims_elementwise() {
        let grid = GridSpec::Elementwise { n: 4096 };
        let (groups, tpg) = grid_to_dims(&grid);
        assert_eq!(groups, [16, 1, 1]); // 4096 / 256
        assert_eq!(tpg, [256, 1, 1]);
    }

    #[test]
    fn grid_to_dims_reduction() {
        let grid = GridSpec::Reduction { num_rows: 32, threads_per_group: 1024 };
        let (groups, tpg) = grid_to_dims(&grid);
        assert_eq!(groups, [32, 1, 1]);
        assert_eq!(tpg, [1024, 1, 1]);
    }

    #[test]
    fn grid_to_dims_grid3d() {
        let grid = GridSpec::Grid3D { x: 32, y: 64, z: 1, threads_per_group: 1 };
        let (groups, tpg) = grid_to_dims(&grid);
        assert_eq!(groups, [32, 64, 1]);
        assert_eq!(tpg, [1, 1, 1]);
    }
}

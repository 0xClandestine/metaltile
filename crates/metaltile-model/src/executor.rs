//! Execution plan executor: dispatches an `ExecutionPlan` on the GPU.
//!
//! The entire forward pass is encoded into a **single** `dispatch_chain`
//! call. This eliminates the per-node `waitUntilCompleted()` stall that
//! previously limited throughput to ~16 tok/s on a 1B model.
//!
//! Strategy:
//! 1. Pre-allocate all slot buffers (`slot_data`) and all intra-group
//!    intermediate buffers (`intra_group_bufs`) upfront.
//! 2. Single loop over all `plan.nodes` builds every kernel + spec.
//!    All buffers are fully resolved before `dispatch_chain` is called.
//! 3. `ctx.dispatch_chain(&all_specs)` submits the entire forward pass
//!    in one MTLCommandBuffer and blocks exactly once.
//! 4. CPU-side state outputs (non-KV-cache scalars) are read back from
//!    `results[i].outputs` after the single dispatch.
//!
//! On Apple Silicon unified memory, `ResidentBuffer::clone()` is a cheap
//! handle copy — no data is moved. Slot and intra-group buffers are
//! written directly by the GPU without any host-side copies.

use std::{
    collections::{BTreeMap, HashMap},
    time::{Duration, Instant},
};

use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer, context::GridSpec};

use crate::{
    error::ModelError,
    plan::{ConstexprValue, ExecutionPlan, SlotRef},
};

/// Watchdog timeout: if the full forward pass takes longer than this,
/// the process is killed to prevent an indefinite GPU hang from freezing
/// macOS.  Checked with an elapsed-time guard after every dispatch_chain
/// call (no per-dispatch thread spawn).
const GPU_WATCHDOG_SECS: u64 = 30;

/// GPU buffer storage keyed by tensor name.
pub type WeightMap = HashMap<String, Vec<u8>>;

/// Runtime state buffers (kv_cache, position counters, etc.).
pub type StateMap = HashMap<String, Vec<u8>>;

/// Static empty function-constants map (shared across all dispatches).
static EMPTY_FN_CONSTS: std::sync::OnceLock<BTreeMap<String, u32>> = std::sync::OnceLock::new();

/// Execute a plan on the GPU and read back the final output buffer.
///
/// Submits the entire forward pass as a **single** `dispatch_chain` call,
/// eliminating per-node `waitUntilCompleted()` stalls. All slot buffers
/// and intra-group intermediate buffers are pre-allocated upfront so no
/// allocation occurs inside the hot loop.
///
/// Returns the output bytes of the plan's final output slot and the total
/// GPU elapsed time in **microseconds**.
pub fn execute_plan(
    ctx: &Context,
    plan: &ExecutionPlan,
    weights: &WeightMap,
    state: &mut StateMap,
    resident: &BTreeMap<String, ResidentBuffer>,
) -> Result<(Vec<u8>, f64), ModelError> {
    let fn_consts = EMPTY_FN_CONSTS.get_or_init(BTreeMap::new);

    // ── Pre-allocate slot buffers ──────────────────────────────────────
    // One GPU-resident buffer per plan slot; written by producer kernels,
    // read by consumer kernels within the same command buffer.
    let slot_data: Vec<ResidentBuffer> = plan
        .slots
        .iter()
        .map(|s| ctx.alloc_resident(s.size_bytes))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ModelError::Other(e.to_string()))?;

    // ── Build all kernels + specs in one pass ──────────────────────────
    // Note: the compiler converts all intra-group intermediate Weight("_name")
    // refs to Slot(idx) before returning the plan, so slot_data covers all
    // intermediate tensors — no separate intra-group buffer map is needed.
    let n = plan.nodes.len();
    let mut kernels: Vec<metaltile_core::ir::Kernel> = Vec::with_capacity(n);
    let mut all_buffers: Vec<BTreeMap<String, Vec<u8>>> = Vec::with_capacity(n);
    let mut all_resident: Vec<BTreeMap<String, ResidentBuffer>> = Vec::with_capacity(n);
    let mut all_output_resident: Vec<BTreeMap<String, ResidentBuffer>> = Vec::with_capacity(n);
    let mut all_grid: Vec<([usize; 3], [usize; 3])> = Vec::with_capacity(n);

    for (idx, node) in plan.nodes.iter().enumerate() {
        let mut kernel = plan.cached_kernels[idx].clone();
        kernel.mode = node.mode;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut spec_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        let mut spec_output_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

        // ── Input bindings ─────────────────────────────────────────────
        for (param_name, slot_ref) in &node.input_bindings {
            match slot_ref {
                SlotRef::Slot(i) => {
                    if let Some(rb) = slot_data.get(*i) {
                        spec_resident.insert(param_name.clone(), rb.clone());
                    }
                },
                SlotRef::Weight(tensor_name) => {
                    if let Some(rb) = resident.get(tensor_name) {
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

        // ── Output bindings ────────────────────────────────────────────
        for (param_name, slot_ref) in &node.output_bindings {
            match slot_ref {
                SlotRef::Slot(i) => {
                    // Write directly into the pre-allocated slot buffer.
                    // GPU serial execution within the command buffer
                    // guarantees the producer finishes before any consumer.
                    spec_output_resident
                        .insert(param_name.clone(), slot_data[*i].clone());
                },
                SlotRef::Weight(_) => {},
                SlotRef::State(key) => {
                    if let Some(rb) = resident.get(key) {
                        // GPU-resident state (KV cache): update in-place.
                        spec_output_resident.insert(param_name.clone(), rb.clone());
                    } else {
                        // CPU-side state scalar: pass a zero buffer; read
                        // back from result.outputs after the single dispatch.
                        let size = state.get(key).map(|v| v.len()).unwrap_or(0);
                        if size > 0 {
                            buffers.insert(param_name.clone(), vec![0u8; size]);
                        }
                    }
                },
            }
        }

        // ── Constexpr values ───────────────────────────────────────────
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

        // ── Pre-dispatch safety checks ─────────────────────────────────
        for param in kernel.params.iter().filter(|p| !p.is_output) {
            let in_resident = spec_resident.contains_key(&param.name);
            let in_buffers = buffers.get(&param.name).is_some_and(|b| !b.is_empty());
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

        kernels.push(kernel);
        all_buffers.push(buffers);
        all_resident.push(spec_resident);
        all_output_resident.push(spec_output_resident);
        all_grid.push(grid_to_dims(&node.grid));
    }

    // ── Dispatch ───────────────────────────────────────────────────────
    // When fusion is active (any node has a fuse_group), the entire forward
    // pass is one MTLCommandBuffer — one waitUntilCompleted, maximum GPU
    // utilisation.  When --no-fuse is set no fuse_groups are assigned, so
    // we fall back to one command buffer per node: useful for debugging
    // (you can bisect failures node-by-node) but ~292× slower.
    let all_specs: Vec<DispatchSpec<'_>> = (0..n)
        .map(|i| {
            let (grid_groups, threads_per_group) = all_grid[i];
            DispatchSpec {
                kernel: &kernels[i],
                buffers: &all_buffers[i],
                fn_consts,
                grid_groups,
                threads_per_group,
                resident: &all_resident[i],
                output_resident: &all_output_resident[i],
            }
        })
        .collect();

    let start = Instant::now();
    let fused = plan.nodes.iter().any(|nd| nd.fuse_group.is_some());

    let results = if fused {
        // ── Fused: entire pass in one command buffer ────────────────
        let results = ctx
            .dispatch_chain(&all_specs)
            .map_err(|e| ModelError::Other(e.to_string()))?;
        if start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS) {
            eprintln!(
                "\n[executor] WATCHDOG: forward pass exceeded {}s — \
                 killing process to prevent system freeze",
                GPU_WATCHDOG_SECS
            );
            std::process::exit(1);
        }
        results
    } else {
        // ── Unfused: one command buffer per node (--no-fuse) ────────
        let mut all_results = Vec::with_capacity(n);
        for i in 0..n {
            let mut res = ctx
                .dispatch_chain(&all_specs[i..i + 1])
                .map_err(|e| ModelError::Other(e.to_string()))?;
            all_results.append(&mut res);
            if start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS) {
                eprintln!(
                    "\n[executor] WATCHDOG: forward pass exceeded {}s at node {} — \
                     killing process to prevent system freeze",
                    GPU_WATCHDOG_SECS, i
                );
                std::process::exit(1);
            }
        }
        all_results
    };

    let total_gpu_us = results.first().map(|r| r.elapsed_us).unwrap_or(0.0);

    // ── Read back CPU-side state outputs (non-KV-cache scalars) ───────
    // KV-cache entries live in `resident` and were updated in-place by
    // the GPU. Only non-resident state scalars need a CPU readback here.
    for (idx, node) in plan.nodes.iter().enumerate() {
        let Some(result) = results.get(idx) else { continue };
        for (param_name, slot_ref) in &node.output_bindings {
            if let SlotRef::State(key) = slot_ref {
                // Resident outputs (KV cache) were written in-place; skip.
                if resident.contains_key(key) {
                    continue;
                }
                if let Some(bytes) = result.outputs.get(param_name) {
                    state.insert(key.clone(), bytes.clone());
                }
            }
        }
    }

    // ── Read final output slot ─────────────────────────────────────────
    let final_size = plan.slots[plan.output_slot].size_bytes;
    Ok((slot_data[plan.output_slot].read_bytes(final_size), total_gpu_us))
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

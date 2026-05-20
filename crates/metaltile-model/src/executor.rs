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
/// is dispatched individually:
/// 1. Intermediate tensors are stored in GPU-resident `ResidentBuffer`s
///    ("slot_data") — no host-side copies on Apple Silicon unified memory.
/// 2. Input slots bind directly via `spec.resident`; output slots allocate
///    fresh `ResidentBuffer`s bound via `spec.output_resident`.
/// 3. Add constexpr values to `spec.buffers` (4-byte LE scalars).
/// 4. Call `ctx.dispatch_chain(&[spec])` with a single spec.
/// 5. Move output `ResidentBuffer`s into `slot_data`; write state outputs
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

    for node in &plan.nodes {
        let mut kernel = (node.kernel_ir)(node.dtype);
        kernel.mode = node.mode;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        // ── Input bindings ────────────────────────────────────────
        // Weights/state in `resident` (keyed by tensor name) → skip CPU copy.
        // Intermediate slots → bind directly via `resident` (zero-copy on
        // unified memory). State keys also checked in resident for KV cache.
        let mut spec_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

        for (param_name, slot_ref) in &node.input_bindings {
            match slot_ref {
                SlotRef::Slot(idx) => {
                    // GPU-resident intermediate — bind directly, no copy.
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
                    // GPU-resident state (KV cache) bypasses CPU buffers.
                    if let Some(rb) = resident.get(key) {
                        spec_resident.insert(param_name.clone(), rb.clone());
                    } else if let Some(bytes) = state.get(key) {
                        buffers.insert(param_name.clone(), bytes.clone());
                    }
                },
            }
        }

        // ── Output bindings ───────────────────────────────────────
        // GPU-resident outputs (KV cache, intermediate slots) use
        // output_resident — GPU writes directly, no host buffer.
        let mut spec_output_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

        for (param_name, slot_ref) in &node.output_bindings {
            match slot_ref {
                SlotRef::Slot(idx) => {
                    let size = plan.slots.get(*idx).map(|s| s.size_bytes).unwrap_or(0);
                    if size > 0 {
                        let rb = ctx
                            .alloc_resident(size)
                            .map_err(|e| ModelError::Other(e.to_string()))?;
                        spec_output_resident.insert(param_name.clone(), rb);
                    }
                },
                SlotRef::State(key) => {
                    if let Some(rb) = resident.get(key) {
                        // GPU writes directly into the resident buffer; no readback.
                        spec_output_resident.insert(param_name.clone(), rb.clone());
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

        // ── Constexpr values ──────────────────────────────────────
        // Constexpr params are bound via `setBytes` using values from
        // `spec.buffers[constexpr_name]`. Put 4-byte LE scalars here.
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

        // ── Pre-dispatch safety checks ────────────────────────────
        // Check 1: all non-output kernel params must have data bound.
        // A missing input buffer means the kernel reads uninitialised/null
        // memory → undefined behaviour → potential GPU hang.
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

        let (grid_groups, threads_per_group) = grid_to_dims(&node.grid);

        // ── Watchdog ──────────────────────────────────────────────
        // Arm a watchdog thread before the GPU dispatch.  If dispatch_chain
        // does not return within GPU_WATCHDOG_SECS, the process self-terminates
        // so macOS can reclaim the GPU rather than freezing the machine.
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

        // ── Dispatch ──────────────────────────────────────────────
        let spec = DispatchSpec {
            kernel: &kernel,
            buffers: &buffers,
            fn_consts,
            grid_groups,
            threads_per_group,
            resident: &spec_resident,
            output_resident: &spec_output_resident,
        };

        let results = ctx.dispatch_chain(&[spec]).map_err(|e| ModelError::Other(e.to_string()))?;
        done_flag.store(true, Ordering::Release); // disarm watchdog

        // ── Accumulate GPU time ─────────────────────────────────
        // dispatch_chain attributes the full cmd-buffer elapsed time
        // to the first result when there are multiple specs. For a
        // single-spec chain, results[0].elapsed_us is the node time.
        if let Some(r) = results.first() {
            total_gpu_us += r.elapsed_us;
        }

        // ── Output readback ───────────────────────────────────────
        let Some(result) = results.into_iter().next() else { continue };

        for (param_name, slot_ref) in &node.output_bindings {
            match slot_ref {
                SlotRef::Slot(idx) => {
                    // GPU wrote directly into output_resident — retrieve it.
                    if let Some(rb) = spec_output_resident.remove(param_name) {
                        slot_data[*idx] = rb;
                    }
                },
                SlotRef::State(key) => {
                    // GPU-resident KV cache was updated in-place by the
                    // output_resident path — no readback needed. Host-side
                    // state still goes through DispatchResult.outputs.
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

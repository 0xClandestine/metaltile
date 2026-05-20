//! Execution plan executor: dispatches an `ExecutionPlan` on the GPU.
//!
//! Each `DispatchNode` is dispatched individually via a single-spec
//! `Context::dispatch_chain` call. Intermediate tensors are tracked in
//! `slot_data` (host-side `Vec<Vec<u8>>`). Weights live in the caller-
//! provided `resident` map (pre-uploaded GPU buffers) or fall back to
//! `weights` for CPU→GPU upload per dispatch.
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
/// 1. Build a `BTreeMap<String, Vec<u8>>` with input bytes (slot/weight/state)
///    and correctly-sized zero output buffers.
/// 2. Add constexpr values to the same map (4-byte LE scalars).
/// 3. Call `ctx.dispatch_chain(&[spec])` with a single spec.
/// 4. Read output param bytes from `DispatchResult.outputs`.
/// 5. Write slot outputs to `slot_data`, state outputs to `state`.
///
/// Returns the output bytes of the plan's final output slot.
pub fn execute_plan(
    ctx: &Context,
    plan: &ExecutionPlan,
    weights: &WeightMap,
    state: &mut StateMap,
    resident: &BTreeMap<String, ResidentBuffer>,
) -> Result<Vec<u8>, ModelError> {
    let fn_consts = EMPTY_FN_CONSTS.get_or_init(BTreeMap::new);

    // Pre-allocate slot data with correct sizes from the liveness analysis.
    let mut slot_data: Vec<Vec<u8>> = plan.slots.iter().map(|s| vec![0u8; s.size_bytes]).collect();

    for node in &plan.nodes {
        let mut kernel = (node.kernel_ir)(node.dtype);
        kernel.mode = node.mode;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        // ── Input bindings ────────────────────────────────────────
        // Weights/state in `resident` (keyed by tensor name) → skip CPU copy.
        // State keys also checked in resident (e.g. GPU-resident KV cache).
        let mut spec_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

        for (param_name, slot_ref) in &node.input_bindings {
            match slot_ref {
                SlotRef::Slot(idx) => {
                    buffers.insert(
                        param_name.clone(),
                        slot_data.get(*idx).cloned().unwrap_or_default(),
                    );
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
        // GPU-resident outputs (KV cache) use output_resident — no host buffer.
        // All others get a correctly-sized zero buffer for dispatch_chain.
        let mut spec_output_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

        for (param_name, slot_ref) in &node.output_bindings {
            match slot_ref {
                SlotRef::Slot(idx) => {
                    let size = plan.slots.get(*idx).map(|s| s.size_bytes).unwrap_or(0);
                    if size > 0 {
                        buffers.insert(param_name.clone(), vec![0u8; size]);
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

        // ── Output readback ───────────────────────────────────────
        let Some(result) = results.into_iter().next() else { continue };

        for (param_name, slot_ref) in &node.output_bindings {
            let Some(bytes) = result.outputs.get(param_name) else { continue };

            match slot_ref {
                SlotRef::Slot(idx) =>
                    if let Some(slot) = slot_data.get_mut(*idx) {
                        *slot = bytes.clone();
                    },
                SlotRef::State(key) => {
                    state.insert(key.clone(), bytes.clone());
                },
                SlotRef::Weight(_) => {},
            }
        }
    }

    // Return the bytes of the plan's designated output slot.
    Ok(slot_data.get(plan.output_slot).cloned().unwrap_or_default())
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

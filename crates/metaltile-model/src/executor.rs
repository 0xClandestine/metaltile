//! Execution plan executor: dispatches an `ExecutionPlan` on the GPU.
//!
//! Maps `DispatchNode` → `DispatchSpec` and calls `Context::dispatch_chain`
//! to encode a single command buffer with automatic pass-to-pass buffer
//! aliasing and barrier insertion.
//!
//! Intermediate buffers (those with `SlotRef::Slot`) are provided as
//! empty byte vectors — `dispatch_chain` allocates them in private storage
//! and aliases them where lifetimes don't overlap.

use std::collections::{BTreeMap, HashMap};

use metaltile_core::ir::Kernel;
use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer};
use metaltile_runtime::context::GridSpec;

use crate::{
    error::ModelError,
    plan::{ConstexprValue, ExecutionPlan, SlotRef},
};

/// GPU buffer storage keyed by tensor name.
pub type WeightMap = HashMap<String, Vec<u8>>;

/// Runtime state buffers (kv_cache, position counters, etc.).
pub type StateMap = HashMap<String, Vec<u8>>;

/// Execute a plan on the GPU and read back the final output buffer.
///
/// This is the primary entry point for inference. It:
/// 1. Builds `DispatchSpec`s from the plan + weights + state.
/// 2. Calls `ctx.dispatch_chain`.
/// 3. Reads back the output buffer from the final node's result.
///
/// Returns the output bytes (typically `vocab_size * sizeof(f32)` logits).
pub fn execute_plan(
    ctx: &Context,
    plan: &ExecutionPlan,
    weights: &WeightMap,
    state: &StateMap,
    resident: &BTreeMap<String, ResidentBuffer>,
) -> Result<Vec<u8>, ModelError> {
    // Build storage for the chained dispatch.
    let mut kernels: Vec<Kernel> = Vec::with_capacity(plan.nodes.len());
    let mut buffers: Vec<BTreeMap<String, Vec<u8>>> = Vec::with_capacity(plan.nodes.len());
    let mut fn_consts: Vec<BTreeMap<String, u32>> = Vec::with_capacity(plan.nodes.len());
    let mut specs: Vec<DispatchSpec<'_>> = Vec::with_capacity(plan.nodes.len());

    for node in &plan.nodes {
        let kernel = (node.kernel_ir)(node.dtype);
        let mut buf_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (param_name, slot_ref) in &node.bindings {
            let bytes = match slot_ref {
                SlotRef::Slot(_) => Vec::new(),
                SlotRef::Weight(name) => weights.get(name).cloned().unwrap_or_default(),
                SlotRef::State(name) => state.get(name).cloned().unwrap_or_default(),
            };
            buf_map.insert(param_name.clone(), bytes);
        }
        let fc: BTreeMap<String, u32> = node
            .cexprs
            .iter()
            .map(|(k, v)| {
                let val = match v {
                    ConstexprValue::Static(val) => *val,
                    ConstexprValue::State(state_key) => {
                        let bytes = state.get(state_key).cloned().unwrap_or_default();
                        if bytes.len() >= 4 {
                            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        } else {
                            1
                        }
                    },
                };
                (k.clone(), val)
            })
            .collect();
        buffers.push(buf_map);
        fn_consts.push(fc);
        kernels.push(kernel);
    }

    // Now that all storage vecs are populated, build DispatchSpecs.
    for i in 0..plan.nodes.len() {
        let node = &plan.nodes[i];
        let (grid_groups, threads_per_group) = grid_to_dims(&node.grid);
        // SAFETY: the index i is valid because we just pushed into these vecs.
        specs.push(DispatchSpec {
            kernel: &kernels[i],
            buffers: &buffers[i],
            fn_consts: &fn_consts[i],
            grid_groups,
            threads_per_group,
            resident,
        });
    }

    let results = ctx
        .dispatch_chain(&specs)
        .map_err(|e| ModelError::Other(e.to_string()))?;

    // Collect outputs from the last spec's result.
    if let Some(last) = results.last() {
        if let Some((_, bytes)) = last.outputs.last_key_value() {
            return Ok(bytes.clone());
        }
        // Fallback: try "out" or "logits" key.
        for key in &["logits", "out"] {
            if let Some(bytes) = last.outputs.get(*key) {
                return Ok(bytes.clone());
            }
        }
    }

    Ok(Vec::new())
}

/// Convert a `GridSpec` to `(grid_groups: [usize; 3], threads_per_group: [usize; 3])`.
fn grid_to_dims(grid: &GridSpec) -> ([usize; 3], [usize; 3]) {
    match grid {
        GridSpec::Elementwise { n } => {
            let tpg = 256usize;
            let groups = n.div_ceil(tpg);
            ([groups, 1, 1], [tpg, 1, 1])
        },
        GridSpec::Reduction { num_rows, threads_per_group } => {
            ([*num_rows, 1, 1], [*threads_per_group, 1, 1])
        },
        GridSpec::Grid3D { x, y, z, threads_per_group } => {
            ([*x, *y, *z], [*threads_per_group, 1, 1])
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
        let grid = GridSpec::Reduction {
            num_rows: 32,
            threads_per_group: 1024,
        };
        let (groups, tpg) = grid_to_dims(&grid);
        assert_eq!(groups, [32, 1, 1]);
        assert_eq!(tpg, [1024, 1, 1]);
    }

    #[test]
    fn grid_to_dims_grid3d() {
        let grid = GridSpec::Grid3D {
            x: 32,
            y: 64,
            z: 1,
            threads_per_group: 1,
        };
        let (groups, tpg) = grid_to_dims(&grid);
        assert_eq!(groups, [32, 64, 1]);
        assert_eq!(tpg, [1, 1, 1]);
    }
}

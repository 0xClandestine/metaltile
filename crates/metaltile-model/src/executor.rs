//! Execution plan executor: dispatches an `ExecutionPlan` on the GPU.
//!
//! ## Two execution paths
//!
//! ### `execute_plan` (legacy, one-shot)
//! Rebuilds all binding maps from scratch each call. Kept for testing.
//!
//! ### `PreparedDispatch` + `execute_prepared` (production path)
//! Builds all static binding maps once at session load time. Per token,
//! only the ~82 dynamic entries (runtime-state-derived constexprs and
//! CPU-side state scalars) are updated in-place — reducing per-token
//! work from ~876 BTreeMap operations to ~164.
//!
//! `execute_prepared` accepts a `max_nodes` parameter for prefill
//! optimisation: pass `plan.prefill_node_count` to skip the output-norm +
//! lm_head + sampling tail during non-final prompt tokens.
//!
//! On Apple Silicon unified memory, `ResidentBuffer::clone()` is a cheap
//! handle copy — no data is moved. Slot buffers in `PreparedDispatch` are
//! persistent (session-lifetime scratch), eliminating per-token pool lookups.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer};
use tracing::error;

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
pub type WeightMap = FxHashMap<String, Vec<u8>>;

/// Runtime state buffers (kv_cache, position counters, etc.).
pub type StateMap = FxHashMap<String, Vec<u8>>;

/// Static empty function-constants map (shared across all dispatches).
/// Uses BTreeMap for deterministic sorted iteration in PSO cache key hashing.
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
    let mut all_buffers: Vec<FxHashMap<String, Vec<u8>>> = Vec::with_capacity(n);
    let mut all_resident: Vec<FxHashMap<String, ResidentBuffer>> = Vec::with_capacity(n);
    let mut all_output_resident: Vec<FxHashMap<String, ResidentBuffer>> = Vec::with_capacity(n);

    for (idx, node) in plan.nodes.iter().enumerate() {
        let kernel = &plan.cached_kernels[idx]; // mode pre-set at compile time
        let mut buffers: FxHashMap<String, Vec<u8>> = FxHashMap::default();
        let mut spec_resident: FxHashMap<String, ResidentBuffer> = FxHashMap::default();
        let mut spec_output_resident: FxHashMap<String, ResidentBuffer> = FxHashMap::default();

        // ── Input bindings ─────────────────────────────────────────────
        for (param_name, slot_ref) in &node.input_bindings {
            match slot_ref {
                SlotRef::Slot(i) =>
                    if let Some(rb) = slot_data.get(*i) {
                        spec_resident.insert(param_name.clone(), rb.clone());
                    },
                SlotRef::Weight(tensor_name) =>
                    if let Some(rb) = resident.get(tensor_name) {
                        spec_resident.insert(param_name.clone(), rb.clone());
                    } else if let Some(bytes) = weights.get(tensor_name) {
                        buffers.insert(param_name.clone(), bytes.clone());
                    },
                SlotRef::State(key) =>
                    if let Some(rb) = resident.get(key) {
                        spec_resident.insert(param_name.clone(), rb.clone());
                    } else if let Some(bytes) = state.get(key) {
                        buffers.insert(param_name.clone(), bytes.clone());
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
                    spec_output_resident.insert(param_name.clone(), slot_data[*i].clone());
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
                    let bytes = state.get(state_key).and_then(|b| b.get(..4)).ok_or_else(|| {
                        ModelError::UnsafeDispatch {
                            op: node.label.clone(),
                            detail: format!(
                                "runtime state '{state_key}' not found — \
                                 ensure it is set in the state map before the forward pass"
                            ),
                        }
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

        all_buffers.push(buffers);
        all_resident.push(spec_resident);
        all_output_resident.push(spec_output_resident);
    }

    // ── Dispatch ───────────────────────────────────────────────────────
    let barriers_after = compute_barriers_after(&plan.nodes[..n]);

    let all_specs: Vec<DispatchSpec<'_>> = (0..n)
        .map(|i| {
            let (grid_groups, threads_per_group) = plan.nodes[i].grid_dims;
            DispatchSpec {
                kernel: &plan.cached_kernels[i],
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
    let fused = plan.single_dispatch;

    let results = if fused {
        // ── Fused: entire pass in one command buffer ────────────────
        let results = ctx
            .dispatch_chain(&all_specs, &barriers_after)
            .map_err(|e| ModelError::Other(e.to_string()))?;
        if start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS) {
            error!(
                "WATCHDOG: forward pass exceeded {}s — \
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
                .dispatch_chain(&all_specs[i..i + 1], &[])
                .map_err(|e| ModelError::Other(e.to_string()))?;
            all_results.append(&mut res);
            if start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS) {
                error!(
                    "WATCHDOG: forward pass exceeded {}s at node {} — \
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

// ── PreparedDispatch ───────────────────────────────────────────────────

/// Pre-built per-node binding maps for efficient per-token dispatch.
///
/// All static data (weights, pre-allocated slot buffers, static constexprs)
/// is computed once in `PreparedDispatch::build()`. Per token, only the
/// ~82 dynamic entries (runtime-state-derived constexprs and CPU-side state
/// scalars) are updated in-place — reducing per-token work from ~876
/// BTreeMap operations to ~164.
pub struct PreparedDispatch {
    /// Session-lifetime intermediate tensor scratch buffers. Allocated once,
    /// reused across all tokens (avoids per-token buffer-pool lookups).
    pub(crate) slot_bufs: Vec<ResidentBuffer>,
    /// Static input GPU-resident maps (weights + pre-allocated slots).
    /// Uses FxHashMap for O(1) lookup in the hot per-token dispatch path.
    all_resident: Vec<FxHashMap<String, ResidentBuffer>>,
    /// Static output GPU-resident maps (KV cache + output slots).
    all_output_resident: Vec<FxHashMap<String, ResidentBuffer>>,
    /// CPU-side buffer maps. Pre-populated with static constexprs.
    /// Dynamic entries (state-derived) updated in-place before each dispatch.
    all_buffers: Vec<FxHashMap<String, Vec<u8>>>,
    /// Nodes with dynamic (State-keyed) constexprs: (node_idx, [(param, state_key)]).
    dyn_cexpr: Vec<(usize, Vec<(String, String)>)>,
    /// Nodes with CPU-side state scalar inputs: (node_idx, [(param, state_key)]).
    dyn_state_in: Vec<(usize, Vec<(String, String)>)>,
    /// Nodes with CPU-side state scalar outputs:
    /// (node_idx, [(param, state_key, size_bytes)]).
    cpu_state_out: Vec<(usize, Vec<(String, String, usize)>)>,
    /// Pre-computed barrier mask for `dispatch_chain`. Static — slot
    /// dependencies don't change between tokens.
    barriers_after: Vec<bool>,
}

impl PreparedDispatch {
    /// Build static binding maps from a compiled plan.
    ///
    /// `resident` must contain all weight buffers and KV-cache buffers.
    /// `state` provides sizes for any CPU-side state scalar outputs
    /// (rare; not present in standard LLMs).
    pub fn build(
        ctx: &Context,
        plan: &ExecutionPlan,
        resident: &FxHashMap<String, ResidentBuffer>,
        state: &StateMap,
    ) -> Result<Self, ModelError> {
        // Allocate session-lifetime slot buffers once.
        let slot_bufs: Vec<ResidentBuffer> = plan
            .slots
            .iter()
            .map(|s| ctx.alloc_resident(s.size_bytes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ModelError::Other(e.to_string()))?;

        let n = plan.nodes.len();
        let mut all_resident: Vec<FxHashMap<String, ResidentBuffer>> = Vec::with_capacity(n);
        let mut all_output_resident: Vec<FxHashMap<String, ResidentBuffer>> = Vec::with_capacity(n);
        let mut all_buffers: Vec<FxHashMap<String, Vec<u8>>> = Vec::with_capacity(n);
        let mut dyn_cexpr: Vec<(usize, Vec<(String, String)>)> = Vec::new();
        let mut dyn_state_in: Vec<(usize, Vec<(String, String)>)> = Vec::new();
        let mut cpu_state_out: Vec<(usize, Vec<(String, String, usize)>)> = Vec::new();

        for (idx, node) in plan.nodes.iter().enumerate() {
            let kernel = &plan.cached_kernels[idx];
            let mut spec_res: FxHashMap<String, ResidentBuffer> = FxHashMap::default();
            let mut spec_out_res: FxHashMap<String, ResidentBuffer> = FxHashMap::default();
            let mut buffers: FxHashMap<String, Vec<u8>> = FxHashMap::default();
            let mut node_dyn_cexpr: Vec<(String, String)> = Vec::new();
            let mut node_dyn_state_in: Vec<(String, String)> = Vec::new();
            let mut node_cpu_state_out: Vec<(String, String, usize)> = Vec::new();

            // ── Input bindings ─────────────────────────────────────────
            for (param, slot_ref) in &node.input_bindings {
                match slot_ref {
                    SlotRef::Slot(i) => {
                        spec_res.insert(param.clone(), slot_bufs[*i].clone());
                    },
                    SlotRef::Weight(name) => {
                        if let Some(rb) = resident.get(name) {
                            spec_res.insert(param.clone(), rb.clone());
                        }
                        // Missing weight caught by safety check below.
                    },
                    SlotRef::State(key) =>
                        if let Some(rb) = resident.get(key) {
                            spec_res.insert(param.clone(), rb.clone());
                        } else {
                            let size = state.get(key.as_str()).map(|v| v.len()).unwrap_or(4);
                            buffers.insert(param.clone(), vec![0u8; size]);
                            node_dyn_state_in.push((param.clone(), key.clone()));
                        },
                }
            }

            // ── Output bindings ────────────────────────────────────────
            for (param, slot_ref) in &node.output_bindings {
                match slot_ref {
                    SlotRef::Slot(i) => {
                        spec_out_res.insert(param.clone(), slot_bufs[*i].clone());
                    },
                    SlotRef::Weight(_) => {},
                    SlotRef::State(key) =>
                        if let Some(rb) = resident.get(key) {
                            spec_out_res.insert(param.clone(), rb.clone());
                        } else {
                            let size = state.get(key.as_str()).map(|v| v.len()).unwrap_or(0);
                            if size > 0 {
                                buffers.insert(param.clone(), vec![0u8; size]);
                                node_cpu_state_out.push((param.clone(), key.clone(), size));
                            }
                        },
                }
            }

            // ── Constexprs ─────────────────────────────────────────────
            for (name, cv) in &node.cexprs {
                match cv {
                    ConstexprValue::Static(val) => {
                        buffers.insert(name.clone(), val.to_le_bytes().to_vec());
                    },
                    ConstexprValue::State(key) => {
                        buffers.insert(name.clone(), vec![0u8; 4]);
                        node_dyn_cexpr.push((name.clone(), key.clone()));
                    },
                }
            }

            // ── Safety check at build time (not repeated per-token) ────
            for param in kernel.params.iter().filter(|p| !p.is_output) {
                let in_res = spec_res.contains_key(&param.name);
                let in_buf = buffers.get(&param.name).is_some_and(|b| !b.is_empty());
                let is_dyn = node_dyn_cexpr.iter().any(|(n, _)| n == &param.name)
                    || node_dyn_state_in.iter().any(|(n, _)| n == &param.name);
                if !in_res && !in_buf && !is_dyn {
                    return Err(ModelError::UnsafeDispatch {
                        op: node.label.clone(),
                        detail: format!(
                            "input '{}' has no data — weight missing from checkpoint \
                             or weight-name mismatch",
                            param.name
                        ),
                    });
                }
            }

            if !node_dyn_cexpr.is_empty() {
                dyn_cexpr.push((idx, node_dyn_cexpr));
            }
            if !node_dyn_state_in.is_empty() {
                dyn_state_in.push((idx, node_dyn_state_in));
            }
            if !node_cpu_state_out.is_empty() {
                cpu_state_out.push((idx, node_cpu_state_out));
            }

            all_resident.push(spec_res);
            all_output_resident.push(spec_out_res);
            all_buffers.push(buffers);
        }

        // Pre-compute barrier mask (static — dependencies don't change between tokens).
        let barriers_after = compute_barriers_after(&plan.nodes);

        Ok(PreparedDispatch {
            slot_bufs,
            all_resident,
            all_output_resident,
            all_buffers,
            dyn_cexpr,
            dyn_state_in,
            cpu_state_out,
            barriers_after,
        })
    }
}

/// Execute a plan using pre-built binding maps.
///
/// Much cheaper per-token than `execute_plan` — only dynamic
/// (state-derived) entries are updated before each dispatch.
///
/// `max_nodes`: number of nodes to execute. Pass `plan.nodes.len()` for a
/// full forward pass (decode step), or `plan.prefill_node_count` during
/// non-final prefill steps to skip the vocab-projection + sampling tail.
/// Returns `(vec![], elapsed)` for partial runs — caller discards output
/// and advances position only.
pub fn execute_prepared(
    pd: &mut PreparedDispatch,
    ctx: &Context,
    plan: &ExecutionPlan,
    state: &mut StateMap,
    max_nodes: usize,
) -> Result<(Vec<u8>, f64), ModelError> {
    let fn_consts = EMPTY_FN_CONSTS.get_or_init(BTreeMap::new);
    let n = max_nodes.min(plan.nodes.len());

    // ── Update dynamic entries (per-token) ────────────────────────────

    for (node_idx, cexprs) in &pd.dyn_cexpr {
        if *node_idx >= n {
            continue;
        }
        let buffers = &mut pd.all_buffers[*node_idx];
        for (param, key) in cexprs {
            let bytes = state.get(key).and_then(|b| b.get(..4)).ok_or_else(|| {
                ModelError::UnsafeDispatch {
                    op: plan.nodes[*node_idx].label.clone(),
                    detail: format!(
                        "runtime state '{key}' not found — \
                         ensure it is set in the state map before the forward pass"
                    ),
                }
            })?;
            let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if let Some(buf) = buffers.get_mut(param.as_str()) {
                buf[..4].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }

    for (node_idx, inputs) in &pd.dyn_state_in {
        if *node_idx >= n {
            continue;
        }
        let buffers = &mut pd.all_buffers[*node_idx];
        for (param, key) in inputs {
            if let Some(src) = state.get(key.as_str()) {
                if let Some(buf) = buffers.get_mut(param.as_str()) {
                    let len = src.len().min(buf.len());
                    buf[..len].copy_from_slice(&src[..len]);
                }
            }
        }
    }

    for (node_idx, outputs) in &pd.cpu_state_out {
        if *node_idx >= n {
            continue;
        }
        let buffers = &mut pd.all_buffers[*node_idx];
        for (param, _key, _size) in outputs {
            if let Some(buf) = buffers.get_mut(param.as_str()) {
                buf.fill(0);
            }
        }
    }

    // ── Build dispatch specs ───────────────────────────────────────────
    let all_specs: Vec<DispatchSpec<'_>> = (0..n)
        .map(|i| {
            let (grid_groups, threads_per_group) = plan.nodes[i].grid_dims;
            DispatchSpec {
                kernel: &plan.cached_kernels[i],
                buffers: &pd.all_buffers[i],
                fn_consts,
                grid_groups,
                threads_per_group,
                resident: &pd.all_resident[i],
                output_resident: &pd.all_output_resident[i],
            }
        })
        .collect();

    // ── Dispatch ───────────────────────────────────────────────────────
    let start = Instant::now();

    let results = if plan.single_dispatch {
        let results = ctx
            .dispatch_chain(&all_specs, &pd.barriers_after[..n])
            .map_err(|e| ModelError::Other(e.to_string()))?;
        if start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS) {
            error!(
                "WATCHDOG: forward pass exceeded {}s — \
                 killing process to prevent system freeze",
                GPU_WATCHDOG_SECS
            );
            std::process::exit(1);
        }
        results
    } else {
        let mut all_results = Vec::with_capacity(n);
        for i in 0..n {
            let mut res = ctx
                .dispatch_chain(&all_specs[i..i + 1], &[])
                .map_err(|e| ModelError::Other(e.to_string()))?;
            all_results.append(&mut res);
            if start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS) {
                error!(
                    "WATCHDOG: forward pass exceeded {}s at node {} — \
                     killing process to prevent system freeze",
                    GPU_WATCHDOG_SECS, i
                );
                std::process::exit(1);
            }
        }
        all_results
    };

    let total_gpu_us = results.first().map(|r| r.elapsed_us).unwrap_or(0.0);

    // ── Read back CPU-side state outputs ──────────────────────────────
    for (node_idx, outputs) in &pd.cpu_state_out {
        if *node_idx >= n {
            continue;
        }
        let Some(result) = results.get(*node_idx) else { continue };
        for (param, key, _size) in outputs {
            if let Some(bytes) = result.outputs.get(param) {
                state.insert(key.clone(), bytes.clone());
            }
        }
    }

    // ── Read final output ──────────────────────────────────────────────
    if n == plan.nodes.len() {
        let final_size = plan.slots[plan.output_slot].size_bytes;
        Ok((pd.slot_bufs[plan.output_slot].read_bytes(final_size), total_gpu_us))
    } else {
        // Partial prefill: vocab head not run, no sampled token yet.
        Ok((Vec::new(), total_gpu_us))
    }
}

// ── Barrier helpers ────────────────────────────────────────────────────

use crate::plan::DispatchNode;

/// Compute the per-node barrier mask for `dispatch_chain`.
///
/// A barrier must be inserted after node `i` if node `i+1` reads any
/// resource written by node `i`. This covers both:
/// - Slot-to-Slot: intermediate tensors flowing through the plan's slot array.
/// - State-to-State: GPU-resident buffers (KV cache) written by one kernel
///   and read by the next (e.g. kv_cache_update → sdpa).
///
/// With `MTLDispatchType::Concurrent` and `HazardTrackingModeUntracked`
/// buffers, Metal does not insert implicit barriers — they must be explicit.
/// The `memoryBarrierWithScope(MTLBarrierScope::Buffers)` call covers all
/// buffer writes (tracked and untracked) that precede it in encoding order.
fn compute_barriers_after(nodes: &[DispatchNode]) -> Vec<bool> {
    let n = nodes.len();
    (0..n)
        .map(|i| {
            if i + 1 >= n {
                return false;
            }
            let producer = &nodes[i];
            let consumer = &nodes[i + 1];
            producer.output_bindings.iter().any(|(_, out_sr)| match out_sr {
                SlotRef::Slot(out_idx) => consumer
                    .input_bindings
                    .iter()
                    .any(|(_, in_sr)| matches!(in_sr, SlotRef::Slot(in_idx) if in_idx == out_idx)),
                SlotRef::State(out_key) => consumer
                    .input_bindings
                    .iter()
                    .any(|(_, in_sr)| matches!(in_sr, SlotRef::State(in_key) if in_key == out_key)),
                SlotRef::Weight(_) => false,
            })
        })
        .collect()
}

//! Execution plan executor: dispatches an `ExecutionPlan` on the GPU.
//!
//! ## Two execution paths
//!
//! ### `execute_plan` (legacy, one-shot)
//! Thin wrapper over `PreparedDispatch::build` + `execute_prepared`.
//! Kept for backward compatibility.
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

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer};
use rustc_hash::FxHashMap;
use tracing::error;

use crate::{
    error::ModelError,
    plan::{ConstexprValue, DispatchNode, ExecutionPlan, SlotRef},
};

/// GPU watchdog timeout: if the full forward pass exceeds this, the process
/// is killed to prevent an indefinite GPU hang from freezing macOS.
const GPU_WATCHDOG_SECS: u64 = 30;

/// Check GPU watchdog every N dispatch calls to avoid per-call syscalls.
const WATCHDOG_CHECK_INTERVAL: u64 = 100;

fn check_watchdog(start: &Instant, counter: &mut u64, label: &str) {
    *counter += 1;
    if *counter % WATCHDOG_CHECK_INTERVAL == 0
        && start.elapsed() > Duration::from_secs(GPU_WATCHDOG_SECS)
    {
        error!(
            "WATCHDOG: forward pass exceeded {}s at {label} — \
             killing process to prevent system freeze",
            GPU_WATCHDOG_SECS
        );
        std::process::exit(1);
    }
}

/// GPU buffer storage keyed by tensor name.
pub type WeightMap = FxHashMap<String, Vec<u8>>;

/// Runtime state buffers (kv_cache, position counters, etc.).
pub type StateMap = FxHashMap<String, Vec<u8>>;

/// Static empty function-constants map (shared across all dispatches).
/// Uses BTreeMap for deterministic sorted iteration in PSO cache key hashing.
static EMPTY_FN_CONSTS: std::sync::OnceLock<BTreeMap<String, u32>> = std::sync::OnceLock::new();

/// Per-node binding maps produced by `bind_node`.
struct NodeBindings {
    resident: FxHashMap<String, ResidentBuffer>,
    out_resident: FxHashMap<String, ResidentBuffer>,
    buffers: FxHashMap<String, Vec<u8>>,
    /// Dynamic (state-derived) constexpr entries: (param_name, state_key).
    dyn_cexpr: Vec<(String, String)>,
    /// CPU-side state scalar inputs: (param_name, state_key).
    dyn_state_in: Vec<(String, String)>,
    /// CPU-side state scalar outputs: (param_name, state_key, size_bytes).
    cpu_state_out: Vec<(String, String, usize)>,
}

/// Build the three binding maps (+ dynamic tracking) for one dispatch node.
///
/// Shared by both `execute_plan` and `PreparedDispatch::build`.  When
/// `track_dynamic` is false the dyn_* fields are always empty (caller can
/// ignore them).
fn bind_node(
    node: &DispatchNode,
    kernel: &metaltile_core::ir::Kernel,
    slot_bufs: &[ResidentBuffer],
    weights: Option<&WeightMap>,
    resident: &FxHashMap<String, ResidentBuffer>,
    state: &StateMap,
    track_dynamic: bool,
) -> Result<NodeBindings, ModelError> {
    let mut sp = FxHashMap::default();
    let mut sp_out = FxHashMap::default();
    let mut bufs = FxHashMap::default();
    let mut dc = Vec::new();
    let mut dsi = Vec::new();
    let mut cso = Vec::new();

    // ── Input bindings ─────────────────────────────────────────────────
    for (param, slot_ref) in &node.input_bindings {
        match slot_ref {
            SlotRef::Slot(i) => {
                sp.insert(param.clone(), slot_bufs[*i].clone());
            },
            SlotRef::Weight(name) => {
                if let Some(rb) = resident.get(name) {
                    sp.insert(param.clone(), rb.clone());
                } else if let Some(wm) = weights {
                    if let Some(bytes) = wm.get(name) {
                        bufs.insert(param.clone(), bytes.clone());
                    }
                }
                // Missing weight caught by safety check below.
            },
            SlotRef::State(key) =>
                if let Some(rb) = resident.get(key) {
                    sp.insert(param.clone(), rb.clone());
                } else if let Some(bytes) = state.get(key) {
                    bufs.insert(param.clone(), bytes.clone());
                    if track_dynamic {
                        dsi.push((param.clone(), key.clone()));
                    }
                },
        }
    }

    // ── Output bindings ────────────────────────────────────────────────
    for (param, slot_ref) in &node.output_bindings {
        match slot_ref {
            SlotRef::Slot(i) => {
                sp_out.insert(param.clone(), slot_bufs[*i].clone());
            },
            SlotRef::Weight(_) => {},
            SlotRef::State(key) =>
                if let Some(rb) = resident.get(key) {
                    sp_out.insert(param.clone(), rb.clone());
                } else {
                    let size = state.get(key.as_str()).map(|v| v.len()).unwrap_or(0);
                    if size > 0 {
                        bufs.insert(param.clone(), vec![0u8; size]);
                        if track_dynamic {
                            cso.push((param.clone(), key.clone(), size));
                        }
                    }
                },
        }
    }

    // ── Constexpr values ───────────────────────────────────────────────
    for (name, cv) in &node.cexprs {
        match cv {
            ConstexprValue::Static(val) => {
                bufs.insert(name.clone(), val.to_le_bytes().to_vec());
            },
            ConstexprValue::State(key) => {
                bufs.insert(name.clone(), vec![0u8; 4]);
                if track_dynamic {
                    dc.push((name.clone(), key.clone()));
                }
            },
        }
    }

    // ── Safety check ───────────────────────────────────────────────────
    for param in kernel.params.iter().filter(|p| !p.is_output) {
        let in_sp = sp.contains_key(&param.name);
        let in_buf = bufs.get(&param.name).is_some_and(|b| !b.is_empty());
        let is_dyn = track_dynamic
            && (dc.iter().any(|(n, _)| n == &param.name)
                || dsi.iter().any(|(n, _)| n == &param.name));
        if !in_sp && !in_buf && !is_dyn {
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

    Ok(NodeBindings { resident: sp, out_resident: sp_out, buffers: bufs, dyn_cexpr: dc, dyn_state_in: dsi, cpu_state_out: cso })
}

/// Execute a plan on the GPU and read back the final output buffer.
///
/// Wraps `PreparedDispatch::build` + `execute_prepared` for backward
/// compatibility with the old one-shot API.
pub fn execute_plan(
    ctx: &Context,
    plan: &ExecutionPlan,
    weights: &WeightMap,
    state: &mut StateMap,
    resident: &BTreeMap<String, ResidentBuffer>,
) -> Result<(Vec<u8>, f64), ModelError> {
    let mut fx: FxHashMap<String, ResidentBuffer> =
        resident.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    // Upload any weights not already GPU-resident.
    for (name, bytes) in weights.iter() {
        fx.entry(name.clone()).or_insert_with(|| ctx.upload_resident(bytes).unwrap());
    }
    let mut pd = PreparedDispatch::build(ctx, plan, &fx, state)?;
    execute_prepared(&mut pd, ctx, plan, state, plan.nodes.len())
}

// ── PreparedDispatch ───────────────────────────────────────────────────

/// Pre-built per-node binding maps for efficient per-token dispatch.
pub struct PreparedDispatch {
    pub(crate) slot_bufs: Vec<ResidentBuffer>,
    all_resident: Vec<FxHashMap<String, ResidentBuffer>>,
    all_output_resident: Vec<FxHashMap<String, ResidentBuffer>>,
    all_buffers: Vec<FxHashMap<String, Vec<u8>>>,
    dyn_cexpr: Vec<(usize, Vec<(String, String)>)>,
    dyn_state_in: Vec<(usize, Vec<(String, String)>)>,
    cpu_state_out: Vec<(usize, Vec<(String, String, usize)>)>,
    barriers_after: Vec<bool>,
}

impl PreparedDispatch {
    /// Build static binding maps from a compiled plan.
    pub fn build(
        ctx: &Context,
        plan: &ExecutionPlan,
        resident: &FxHashMap<String, ResidentBuffer>,
        state: &StateMap,
    ) -> Result<Self, ModelError> {
        let slot_bufs: Vec<ResidentBuffer> = plan
            .slots
            .iter()
            .map(|s| ctx.alloc_resident(s.size_bytes))
            .collect::<Result<Vec<_>, _>>()?;

        let n = plan.nodes.len();
        let mut all_resident = Vec::with_capacity(n);
        let mut all_output_resident = Vec::with_capacity(n);
        let mut all_buffers = Vec::with_capacity(n);
        let mut dyn_cexpr = Vec::new();
        let mut dyn_state_in = Vec::new();
        let mut cpu_state_out = Vec::new();

        for (idx, node) in plan.nodes.iter().enumerate() {
            let kernel = &plan.cached_kernels[idx];
            let nb = bind_node(node, kernel, &slot_bufs, None, resident, state, true)?;
            if !nb.dyn_cexpr.is_empty() {
                dyn_cexpr.push((idx, nb.dyn_cexpr));
            }
            if !nb.dyn_state_in.is_empty() {
                dyn_state_in.push((idx, nb.dyn_state_in));
            }
            if !nb.cpu_state_out.is_empty() {
                cpu_state_out.push((idx, nb.cpu_state_out));
            }
            all_resident.push(nb.resident);
            all_output_resident.push(nb.out_resident);
            all_buffers.push(nb.buffers);
        }

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
        if *node_idx >= n { continue; }
        let bufs = &mut pd.all_buffers[*node_idx];
        for (param, key) in cexprs {
            let bytes = state.get(key).and_then(|b| b.get(..4)).ok_or_else(|| {
                ModelError::UnsafeDispatch {
                    op: plan.nodes[*node_idx].label.clone(),
                    detail: format!("runtime state '{key}' not found"),
                }
            })?;
            let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if let Some(buf) = bufs.get_mut(param.as_str()) {
                buf[..4].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }

    for (node_idx, inputs) in &pd.dyn_state_in {
        if *node_idx >= n { continue; }
        let bufs = &mut pd.all_buffers[*node_idx];
        for (param, key) in inputs {
            if let Some(src) = state.get(key.as_str()) {
                if let Some(buf) = bufs.get_mut(param.as_str()) {
                    let len = src.len().min(buf.len());
                    buf[..len].copy_from_slice(&src[..len]);
                }
            }
        }
    }

    for (node_idx, outputs) in &pd.cpu_state_out {
        if *node_idx >= n { continue; }
        let bufs = &mut pd.all_buffers[*node_idx];
        for (param, _key, _size) in outputs {
            if let Some(buf) = bufs.get_mut(param.as_str()) {
                buf.fill(0);
            }
        }
    }

    // ── Build dispatch specs ───────────────────────────────────────────
    let all_specs: Vec<DispatchSpec<'_>> = (0..n)
        .map(|i| {
            let (g, t) = plan.nodes[i].grid_dims;
            DispatchSpec {
                kernel: &plan.cached_kernels[i],
                buffers: &pd.all_buffers[i],
                fn_consts,
                grid_groups: g,
                threads_per_group: t,
                resident: &pd.all_resident[i],
                output_resident: &pd.all_output_resident[i],
            }
        })
        .collect();

    // ── Dispatch ───────────────────────────────────────────────────────
    let start = Instant::now();
    let mut watchdog_ctr: u64 = 0;

    #[inline]
    fn dispatch_one(ctx: &Context, specs: &[DispatchSpec], barriers: &[bool], label: &str, ctr: &mut u64, start: &Instant) -> Result<Vec<metaltile_runtime::DispatchResult>, ModelError> {
        let r = ctx.dispatch_chain(specs, barriers)?;
        check_watchdog(start, ctr, label);
        Ok(r)
    }

    let results = if plan.single_dispatch {
        dispatch_one(ctx, &all_specs, &pd.barriers_after[..n], "execute_prepared (fused)", &mut watchdog_ctr, &start)?
    } else {
        let mut all = Vec::with_capacity(n);
        for i in 0..n {
            all.append(&mut dispatch_one(ctx, &all_specs[i..i + 1], &[], &format!("execute_prepared node {i}"), &mut watchdog_ctr, &start)?);
        }
        all
    };

    let total_gpu_us = results.first().map(|r| r.elapsed_us).unwrap_or(0.0);

    // ── Read back CPU-side state outputs ──────────────────────────────
    for (node_idx, outputs) in &pd.cpu_state_out {
        if *node_idx >= n { continue; }
        let Some(result) = results.get(*node_idx) else { continue };
        for (param, key, _size) in outputs {
            if let Some(bytes) = result.outputs.get(param) {
                state.insert(key.clone(), bytes.clone());
            }
        }
    }

    // ── Read final output ──────────────────────────────────────────────
    if n == plan.nodes.len() {
        let sz = plan.slots[plan.output_slot].size_bytes;
        Ok((pd.slot_bufs[plan.output_slot].read_bytes(sz), total_gpu_us))
    } else {
        Ok((Vec::new(), total_gpu_us))
    }
}

// ── Barrier helpers ────────────────────────────────────────────────────

/// Compute the per-node barrier mask for `dispatch_chain`.
fn compute_barriers_after(nodes: &[DispatchNode]) -> Vec<bool> {
    let n = nodes.len();
    (0..n)
        .map(|i| {
            if i + 1 >= n { return false; }
            let p = &nodes[i];
            let c = &nodes[i + 1];
            p.output_bindings.iter().any(|(_, out)| match out {
                SlotRef::Slot(oi) => c.input_bindings.iter().any(|(_, inn)| matches!(inn, SlotRef::Slot(ii) if ii == oi)),
                SlotRef::State(ok) => c.input_bindings.iter().any(|(_, inn)| matches!(inn, SlotRef::State(ik) if ik == ok)),
                SlotRef::Weight(_) => false,
            })
        })
        .collect()
}
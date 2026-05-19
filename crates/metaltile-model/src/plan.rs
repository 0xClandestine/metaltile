//! Execution plan types: the result of compiling a `ModelDef` into a
//! dispatchable sequence.
//!
//! An `ExecutionPlan` is a flattened, resolved, GPU-ready list of
//! kernel dispatches — all `$var` references resolved, buffer slots
//! assigned, grid dimensions computed. It can be dispatched repeatedly
//! (once per token for autoregressive inference).

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::context::GridSpec;

// ── ExecutionPlan ──────────────────────────────────────────────────────

/// A fully resolved model forward-pass plan.
///
/// Build once at model-load time, dispatch per token.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Ordered dispatch nodes, one per kernel invocation after unrolling.
    pub nodes: Vec<DispatchNode>,
    /// Named buffer slots. Slots can be shared across nodes where lifetimes
    /// don't overlap (see `liveness.rs`).
    pub slots: Vec<BufferSlot>,
    /// Index into `slots` for the final output tensor (e.g. logits).
    pub output_slot: usize,
    /// The total number of layers (unroll count).
    pub n_layers: usize,
}

// ── ConstexprValue ────────────────────────────────────────────────────

/// A constexpr value that may be static (resolved at compile time)
/// or dynamic (resolved per-token from runtime state).
#[derive(Debug, Clone)]
pub enum ConstexprValue {
    /// Resolved at graph-compile time.
    Static(u32),
    /// Resolved per-dispatch from state map key.
    State(String),
}

// ── DispatchNode ───────────────────────────────────────────────────────

/// A single kernel dispatch within the plan.
///
/// Maps 1:1 to a `Context::dispatch_chain` pass (one `DispatchSpec`).
/// The node holds enough information to build a `DispatchSpec` at
/// dispatch time, resolving `SlotRef`s to actual GPU buffers.
#[derive(Debug)]
pub struct DispatchNode {
    /// Human-readable label for debugging (e.g. "layer.3.rms_norm").
    pub label: String,
    /// Kernel name for PSO caching and debug output.
    pub kernel_name: &'static str,
    /// IR constructor — same field as `BenchSpec.kernel_ir`.
    pub kernel_ir: fn(DType) -> Kernel,
    /// KernelMode for this dispatch (sets Metal built-in attributes).
    pub mode: KernelMode,
    /// Buffer parameter bindings: param name → slot/weight/state ref.
    pub bindings: Vec<(String, SlotRef)>,
    /// Constexpr values bound at graph-compile time (Static) or
    /// resolved per-dispatch from runtime state (State).
    pub cexprs: Vec<(String, ConstexprValue)>,
    /// Grid sizing, computed from constexprs at compile time.
    pub grid: GridSpec,
    /// Dtype for this node (typically inherits model activation dtype).
    pub dtype: DType,
}

// ── SlotRef ────────────────────────────────────────────────────────────

/// Reference to a buffer needed by a `DispatchNode`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotRef {
    /// Index into `ExecutionPlan.slots` (intermediate buffer).
    Slot(usize),
    /// Weight tensor, bound by name at inference time.
    Weight(String),
    /// Runtime state tensor (kv_cache, position, etc.).
    State(String),
}

// ── BufferSlot ─────────────────────────────────────────────────────────

/// A reusable intermediate buffer slot.
#[derive(Debug, Clone)]
pub struct BufferSlot {
    /// Human-readable name (for debugging).
    pub name: String,
    /// Size in bytes (computed at compile time from shapes × dtype).
    pub size_bytes: usize,
    /// Node index of first use (inclusive).
    pub first_use: usize,
    /// Node index of last use (inclusive).
    pub last_use: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_ref_equality() {
        assert_eq!(SlotRef::Slot(0), SlotRef::Slot(0));
        assert_ne!(SlotRef::Slot(0), SlotRef::Slot(1));
        assert_eq!(SlotRef::Weight("w".into()), SlotRef::Weight("w".into()));
        assert_ne!(SlotRef::Weight("a".into()), SlotRef::Weight("b".into()));
    }

    #[test]
    fn buffer_slot_lifetime_range() {
        let slot = BufferSlot {
            name: "test".into(),
            size_bytes: 1024,
            first_use: 3,
            last_use: 7,
        };
        assert_eq!(slot.first_use, 3);
        assert_eq!(slot.last_use, 7);
    }
}

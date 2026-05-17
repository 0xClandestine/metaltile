//! Register Estimation — conservative linear-scan liveness analysis.
//!
//! Performs a conservative forward liveness analysis over IR blocks to
//! estimate the maximum number of simultaneously-live ValueIds, which
//! approximates register pressure per thread.  Used by [`occupancy`] to
//! compare tile size candidates.
//!
//! ## Caveats
//!
//! This is a *static estimate*.  The Metal compiler performs the actual
//! register allocation.  The estimate is useful for comparing tile size
//! candidates: lower max_live → higher occupancy potential.
//!
//! ## Algorithm
//!
//! For each block, iterate forward through ops:
//! - When an op references a ValueId, add it to the live set.
//! - When an op defines a ValueId, add it to the live set.
//! - Track the maximum live set size at each step.
//!
//! Phase 1 uses conservative liveness: values are never killed within a block
//! (they remain live through the block end).  This over-estimates register
//! pressure, which is the safe direction for occupancy decisions.
//!
//! ## References
//! - Poletto & Sarkar (1999), "Linear scan register allocation",
//!   ACM TOPLAS 21(5):895–913.  The foundational paper on linear-scan
//!   liveness analysis for fast register allocation.
//!   https://dl.acm.org/doi/10.1145/330249.330250

use metaltile_core::ir::{Kernel, ValueId};

use super::remap;

/// Estimated register pressure for a kernel.
#[derive(Debug, Clone)]
pub struct RegisterEstimate {
    /// Maximum simultaneously-live ValueIds across all blocks.
    pub max_live: usize,
    /// Estimated registers per thread (max_live × avg_regs_per_value).
    /// Uses 1.5 as a heuristic: some values are scalars (1 reg),
    /// some are vectors (4 regs), and many are short-lived.
    pub regs_per_thread: usize,
}

/// Run a conservative liveness analysis and return the register estimate.
pub fn estimate_registers(kernel: &Kernel) -> RegisterEstimate {
    let mut max_live = 0usize;

    // Process body.
    max_live = max_live.max(block_max_live(&kernel.body));

    // Process nested blocks.
    for block in kernel.blocks.values() {
        max_live = max_live.max(block_max_live(block));
    }

    let regs_per_thread = (max_live as f64 * 1.5).ceil() as usize;

    RegisterEstimate { max_live, regs_per_thread }
}

/// Compute the maximum live ValueId count in a single block.
fn block_max_live(block: &metaltile_core::ir::Block) -> usize {
    let mut live: std::collections::BTreeSet<ValueId> = std::collections::BTreeSet::new();
    let mut max = 0usize;

    for (i, op) in block.ops.iter().enumerate() {
        // References → add to live set.
        for vid in remap::op_value_refs(op) {
            live.insert(vid);
        }

        // Definitions → add to live set.
        if let Some(Some(vid)) = block.results.get(i) {
            live.insert(*vid);
        }

        max = max.max(live.len());
    }

    max
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{BinOpKind, Op};

    use super::*;

    #[test]
    fn empty_kernel_has_zero_live() {
        let k = Kernel::new("empty");
        let est = estimate_registers(&k);
        assert_eq!(est.max_live, 0);
    }

    #[test]
    fn single_binop_has_three_live() {
        // v0 = Const, v1 = Const, v2 = Add(v0, v1) → 3 live at peak
        let mut k = Kernel::new("binop");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );

        let est = estimate_registers(&k);
        assert_eq!(est.max_live, 3); // v0, v1, v2 all live at the Add
    }

    #[test]
    fn estimates_regs_per_thread() {
        let mut k = Kernel::new("regs");
        // Push 10 constant ops → 10 live at peak.
        for i in 0..10u32 {
            k.body.push_op(Op::Const { value: i as i64 }, ValueId::new(i));
        }

        let est = estimate_registers(&k);
        assert_eq!(est.max_live, 10);
        assert_eq!(est.regs_per_thread, 15); // 10 * 1.5 = 15
    }
}

//! Dead Store Elimination — remove writes that are overwritten before any read.
//!
//! Eliminates `Op::Store` / `Op::VectorStore` operations that write to a
//! location subsequently overwritten before any read, or writes that are never
//! read at all.  This is block-local overwrite detection; global store-to-load
//! forwarding is handled by a separate pass.
//!
//! ## Algorithm
//!
//! 1. Collect all load locations (reads) from the block.
//! 2. Backward scan: for each store, check if the same location is stored
//!    again later without an intervening read.  If so, the earlier store is dead.
//! 3. Never eliminate the last store to an output param.
//! 4. Conservatively preserve masked stores (may-write semantics).
//!
//! ## Safety
//!
//! - Output params: stores to `is_output` params are always preserved.
//! - Masked stores: treated as "may write" — never eliminated.
//! - Partial overlap: Phase 1 handles exact overlap only (same `(dst, indices)`).
//! - Phase 2 (future): partial overlap via range analysis.
//!
//! ## References
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §9.1.3.  Dead-code elimination, including store removal.
//! - Ferrante, Ottenstein & Warren (1987), "The program dependence graph and
//!   its use in optimization", ACM TOPLAS 9(3):319–349.  PDG-based dead-store
//!   detection.

use std::collections::BTreeSet;

use metaltile_core::ir::{Block, BlockId, IndexExpr, Kernel, Op, ValueId};
use rustc_hash::{FxHashMap, FxHashSet};

use super::block_util;
use crate::error::{Error, Result};

pub struct DeadStoreElimPass;

impl super::Pass for DeadStoreElimPass {
    fn name(&self) -> &str { "dead_store_elim" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Collect output param names (never eliminate these).
        let output_params: BTreeSet<String> =
            kernel.params.iter().filter(|p| p.is_output).map(|p| p.name.clone()).collect();

        dse_block(&mut kernel.body, &output_params);

        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            let mut block =
                kernel.blocks.remove(&bid).ok_or_else(|| Error::BlockNotFound(bid.as_u32()))?;
            dse_block(&mut block, &output_params);
            kernel.blocks.insert(bid, block);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// store location key
// ---------------------------------------------------------------------------

/// A key identifying a unique store/load location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StoreKey {
    Scalar { dst: String, indices: Vec<IndexExpr> },
    Vector { dst: String, byte_offset: ValueId, len: u32 },
}

impl StoreKey {
    fn from_store(op: &Op) -> Option<Self> {
        match op {
            Op::Store { dst, indices, .. } =>
                Some(StoreKey::Scalar { dst: dst.clone(), indices: indices.clone() }),
            Op::VectorStore { dst, byte_offset, len, .. } =>
                Some(StoreKey::Vector { dst: dst.clone(), byte_offset: *byte_offset, len: *len }),
            _ => None,
        }
    }

    fn from_load(op: &Op) -> Option<Self> {
        match op {
            Op::Load { src, indices, .. } =>
                Some(StoreKey::Scalar { dst: src.clone(), indices: indices.clone() }),
            Op::VectorLoad { src, byte_offset, len } =>
                Some(StoreKey::Vector { dst: src.clone(), byte_offset: *byte_offset, len: *len }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// main DSE logic
// ---------------------------------------------------------------------------

fn dse_block(block: &mut Block, output_params: &BTreeSet<String>) {
    let n = block.ops.len();
    if n == 0 {
        return;
    }

    // Phase 1: collect all load locations (reads). Pre-size to op count —
    // upper bound on distinct load locations, avoids the 4 → 8 → 16 → … grow
    // sequence on every block.
    let mut reads: FxHashSet<StoreKey> = FxHashSet::with_capacity_and_hasher(n, Default::default());
    for op in &block.ops {
        if let Some(key) = StoreKey::from_load(op) {
            reads.insert(key);
        }
    }

    // Phase 2: backward scan for dead stores.
    let mut dead: Vec<usize> = Vec::new();
    // Maps a StoreKey to (position of most recent store seen, is_alive).
    // Scanned backwards: "most recent" = earliest in execution order.
    let mut last_store_at: FxHashMap<StoreKey, (usize, bool)> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    for i in (0..n).rev() {
        let op = &block.ops[i];
        let Some(key) = StoreKey::from_store(op) else { continue };

        // Don't eliminate masked stores (may-write semantics).
        if has_mask(op) {
            continue;
        }

        // Don't eliminate stores to output params.
        if store_target_is_output(op, output_params) {
            continue;
        }

        if reads.contains(&key) {
            // This location is loaded somewhere → keep this store alive.
            // The previous store (if any, at higher index) is NOT overwritten
            // by this one (it executes before the load).
            last_store_at.insert(key, (i, true));
        } else if let Some(&(_later_idx, later_alive)) = last_store_at.get(&key) {
            // A later store exists at later_idx.
            // The CURRENT store (at i) is overwritten by the later one → dead.
            dead.push(i);
            // But DON'T overwrite last_store_at — the later store (later_idx)
            // is the one that actually matters. However, if the later store
            // is also dead, make this one the new candidate.
            if !later_alive {
                // Later store is dead too → this one becomes the new candidate.
                last_store_at.insert(key, (i, false));
            }
            // If later_alive, this store (i) is dead and the later store stays.
        } else {
            // No later store for this key. This is the last write → it's alive.
            last_store_at.insert(key, (i, true));
        }
    }

    if dead.is_empty() {
        return;
    }

    // Remove dead stores from the block.
    dead.sort_unstable();
    block_util::remove_ops(block, &dead);
}

/// Check if a store has a mask (may-write semantics).
fn has_mask(op: &Op) -> bool {
    match op {
        Op::Store { mask, .. } => mask.is_some(),
        Op::VectorStore { .. } => false, // VectorStore has no mask field
        _ => false,
    }
}

/// Check if a store targets an output param.
fn store_target_is_output(op: &Op, output_params: &BTreeSet<String>) -> bool {
    let dst = match op {
        Op::Store { dst, .. } => dst,
        Op::VectorStore { dst, .. } => dst,
        _ => return false,
    };
    output_params.contains(dst.as_str())
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        Shape,
        dtype::DType,
        ir::{Param, ParamKind},
    };

    use super::*;
    use crate::passes::Pass;

    fn make_kernel_with_output(name: &str, out_name: &str) -> Kernel {
        let mut k = Kernel::new(name);
        k.params.push(Param {
            name: out_name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: ParamKind::Tensor,
        });
        k
    }

    #[test]
    fn eliminates_overwritten_scalar_store() {
        let mut k = Kernel::new("dse_overwrite");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: None,
        });
        k.body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(2),
            mask: None,
        });

        DeadStoreElimPass.run(&mut k).unwrap();

        // Only the second store remains.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        assert_eq!(stores.len(), 1);
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 2);
        }
    }

    #[test]
    fn preserves_output_param_store() {
        let mut k = make_kernel_with_output("dse_output", "out");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: None,
        });

        DeadStoreElimPass.run(&mut k).unwrap();
        assert_eq!(k.body.ops.len(), 3); // ProgramId + Const + Store preserved
    }

    #[test]
    fn preserves_store_with_intervening_load() {
        let mut k = Kernel::new("dse_load_between");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(2));
        // Store to buf[0]
        k.body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: None,
        });
        // Load from buf[0]
        k.body.push_op(
            Op::Load {
                src: "buf".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(3),
        );
        // Store to buf[0] again
        k.body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(2),
            mask: None,
        });

        DeadStoreElimPass.run(&mut k).unwrap();

        // Both stores preserved; the first is read by the Load.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        assert_eq!(stores.len(), 2);
    }

    #[test]
    fn preserves_masked_store() {
        let mut k = Kernel::new("dse_masked");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: Some(ValueId::new(100)), // masked → not eliminated
        });
        k.body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(2),
            mask: None,
        });

        DeadStoreElimPass.run(&mut k).unwrap();

        // Both preserved: first is masked, second is the final store.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        assert_eq!(stores.len(), 2);
    }
}

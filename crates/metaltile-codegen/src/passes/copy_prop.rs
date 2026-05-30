//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copy Propagation — forward source values through identity operations.
//!
//! Eliminates no-op operations and propagates the underlying source value
//! through chains of copies and identity casts.  Shortens use-def chains so
//! downstream passes (CSE, Fusion, AlgebraicSimplify) see the "real" values.
//!
//! ## Identity Patterns
//! - `Cast(dtype, x)` → `x`  when `x` is already that dtype
//! - `Broadcast(x, [1])` → `x`  when broadcasting a scalar with shape [1]
//! - `Reshape(x, s)` → `x`  when shapes are identical
//! - `Select(cond, x, x)` → `x`  (also in AlgebraicSimplify, but cheap to re-check)
//!
//! ## Copy Forwarding
//! When an op's result is used through a chain of identity operations,
//! forward the source value through. The downstream CSE pass then eliminates the
//! now-dead identity ops.
//!
//! ## Algorithm
//!
//! Iterates to fixpoint.  Each iteration:
//! 1. Find identity ops (result == source).
//! 2. Replace all uses of the identity result with the source ValueId.
//! 3. DCE cleans up the dead identity ops (ran after this pass).
//!
//! ## References
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §9.1.1.  Canonical treatment of copy propagation.
//! - Wegman & Zadeck (1991), "Constant propagation with conditional branches",
//!   ACM TOPLAS 13(2):181–210.  Sparse conditional constant propagation framework
//!   that subsumes copy propagation.

use metaltile_core::{
    dtype::DType,
    ir::{Block, BlockId, Kernel, Op, ValueId},
};
use rustc_hash::{FxHashMap, FxHashSet};

use super::remap;
use crate::error::{Error, Result};

pub struct CopyPropPass;

impl super::Pass for CopyPropPass {
    fn name(&self) -> &str { "copy_prop" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();

        for bid in &block_ids {
            let mut block =
                kernel.blocks.remove(bid).ok_or_else(|| Error::BlockNotFound(bid.as_u32()))?;
            copy_prop_block_fixpoint(&mut block);
            kernel.blocks.insert(*bid, block);
        }

        copy_prop_block_fixpoint(&mut kernel.body);

        super::dead_value_elim::eliminate_dead_values(kernel)?;
        Ok(())
    }
}

/// Resolve transitive replacement chains: {v2→v1, v1→v0} becomes {v2→v0, v1→v0}.
fn resolve_transitive(map: &FxHashMap<ValueId, ValueId>) -> FxHashMap<ValueId, ValueId> {
    let mut resolved = FxHashMap::with_capacity_and_hasher(map.len(), Default::default());
    for (&key, &val) in map.iter() {
        let mut terminal = val;
        let mut visited = FxHashSet::default();
        visited.insert(key);
        while let Some(&next) = map.get(&terminal) {
            if !visited.insert(terminal) {
                break; // cycle detected
            }
            terminal = next;
        }
        resolved.insert(key, terminal);
    }
    resolved
}

fn copy_prop_block_fixpoint(block: &mut Block) {
    loop {
        if !copy_prop_block_once(block) {
            break;
        }
    }
}

fn copy_prop_block_once(block: &mut Block) -> bool {
    let n = block.ops.len();
    // Pre-sized with `block.ops.len()` per playbook §"Pre-size with
    // `with_capacity`" — half the dead_store_elim PR #38 win.
    let mut vid_replacements: FxHashMap<ValueId, ValueId> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    for i in 0..n {
        let op = &block.ops[i];
        if let Some(source_vid) = is_identity(op, block)
            && let Some(Some(result_vid)) = block.results.get(i)
        {
            vid_replacements.insert(*result_vid, source_vid);
        }
    }

    if vid_replacements.is_empty() {
        return false;
    }

    // Resolve transitive replacement chains: v2→v1→v0 becomes v2→v0.
    let vid_replacements = resolve_transitive(&vid_replacements);

    // Remap ValueIds in all ops via the FxHashMap-keyed sibling of
    // `remap_value_ids` (playbook §"FxHashMap wins on codegen").
    for op in block.ops.iter_mut() {
        remap::remap_value_ids_fx(op, &vid_replacements);
    }

    // Remove dead ops whose results were redirected via identity propagation.
    // Without this, the same identity pattern re-matches on the next iteration,
    // producing the same replacement and causing an infinite fixpoint loop.
    let dead_vids: FxHashSet<ValueId> = vid_replacements.keys().copied().collect();
    if !dead_vids.is_empty() {
        let mut new_ops = Vec::new();
        let mut new_results = Vec::new();
        for (i, op) in block.ops.iter().enumerate() {
            let is_dead =
                block.results.get(i).is_some_and(|r| r.is_some_and(|v| dead_vids.contains(&v)));
            if !is_dead {
                new_ops.push(op.clone());
                new_results.push(block.results[i]);
            }
        }
        block.ops = new_ops;
        block.results = new_results;
    }

    true
}

/// Check if an op is an identity (output equals one of its inputs in all cases).
fn is_identity(op: &Op, _block: &Block) -> Option<ValueId> {
    match op {
        // Cast(float, x) → x  when x is already float
        Op::Cast { value, dtype } => {
            let inferred = infer_value_dtype(*value, _block);
            if inferred == Some(*dtype) { Some(*value) } else { None }
        },

        // Broadcast(x, [1]) → x  — broadcasting a scalar by shape [1] is a no-op
        Op::Broadcast { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                Some(*value)
            } else {
                None
            }
        },

        // Reshape(x, s) → x  when shape s has the same total elements and same layout
        // For now: only when the value is already a scalar or single-element tile.
        Op::Reshape { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                // Reshape to [1] is identity for scalars
                Some(*value)
            } else {
                None
            }
        },

        // Select(cond, x, x) → x  — same value both sides
        Op::Select { on_true, on_false, .. } =>
            if on_true == on_false {
                Some(*on_true)
            } else {
                None
            },

        // ExpandDims with shape [1] is effectively an identity for a scalar
        // (handled by Reshape already; but cover base case)
        _ => None,
    }
}

/// Naive dtype inference for a value.  Only detects `Cast` and `Const` patterns.
fn infer_value_dtype(vid: ValueId, block: &Block) -> Option<DType> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) != Some(&Some(vid)) {
            continue;
        }
        match op {
            Op::Cast { dtype, .. } => return Some(*dtype),
            Op::Const { .. } =>
            // Constants are integers; they'll be cast to target dtype at use.
            {
                return None;
            },
            Op::Zeros { dtype, .. } | Op::Splat { dtype, .. } => return Some(*dtype),
            Op::Load { .. } => return None, // dtype comes from param
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod perf {
    //! `#[ignore]`'d microbench for the cascade — runs under
    //!
    //! ```text
    //! cargo test -p metaltile-codegen --release perf_copy_prop_btreemap_vs_fxhash \
    //!     -- --ignored --nocapture
    //! ```
    //!
    //! Reconstructs the pre-cascade `resolve_transitive` + `dead_vids`
    //! collection + `remap_value_ids` shape using `BTreeMap`/`BTreeSet`
    //! against the new `FxHashMap`/`FxHashSet` versions and times the
    //! per-block hot path. Covers the same shape change applied across
    //! `copy_prop.rs` / `algebraic_simplify.rs` / `unroll.rs` /
    //! `vectorize.rs::result_remap` — the four call sites of
    //! `remap::remap_value_ids_fx` introduced in this PR.
    use std::{
        collections::{BTreeMap, BTreeSet},
        hint::black_box,
        time::Instant,
    };

    use metaltile_core::ir::{BinOpKind, Op, ValueId};
    use rustc_hash::{FxHashMap, FxHashSet};

    use super::remap;

    fn synthetic_block(n: usize) -> (Vec<Op>, Vec<Option<ValueId>>) {
        // Build n BinOp(Add, v_{i-1}, v_{i-2}) ops to exercise the
        // 2-refs-per-op iteration that drives `remap_value_ids` cost.
        let mut ops = Vec::with_capacity(n);
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let lhs = ValueId::new(if i >= 2 { (i - 2) as u32 } else { 0 });
            let rhs = ValueId::new(if i >= 1 { (i - 1) as u32 } else { 0 });
            ops.push(Op::BinOp { op: BinOpKind::Add, lhs, rhs });
            results.push(Some(ValueId::new(i as u32 + 100)));
        }
        (ops, results)
    }

    #[test]
    #[ignore]
    fn perf_copy_prop_btreemap_vs_fxhash() {
        const N_OPS: usize = 2_000;
        const N_REPLACEMENTS: u32 = 256;
        const N_ITERS: usize = 500;

        let (base_ops, _results) = synthetic_block(N_OPS);

        // ── BTreeMap (pre-cascade) ──
        let bt_seed: Vec<(ValueId, ValueId)> =
            (0..N_REPLACEMENTS).map(|i| (ValueId::new(i), ValueId::new(i + 1000))).collect();

        for _ in 0..3 {
            let map: BTreeMap<ValueId, ValueId> = bt_seed.iter().copied().collect();
            let mut ops = base_ops.clone();
            for op in ops.iter_mut() {
                remap::remap_value_ids(op, &map);
            }
            let _dead: BTreeSet<ValueId> = map.keys().copied().collect();
            black_box(&ops);
        }
        let t0 = Instant::now();
        for _ in 0..N_ITERS {
            let map: BTreeMap<ValueId, ValueId> = bt_seed.iter().copied().collect();
            let mut ops = base_ops.clone();
            for op in ops.iter_mut() {
                remap::remap_value_ids(op, &map);
            }
            let dead: BTreeSet<ValueId> = map.keys().copied().collect();
            black_box(&ops);
            black_box(dead.len());
        }
        let bt_elapsed = t0.elapsed();

        // ── FxHashMap (this cascade) ──
        for _ in 0..3 {
            let map: FxHashMap<ValueId, ValueId> = bt_seed.iter().copied().collect();
            let mut ops = base_ops.clone();
            for op in ops.iter_mut() {
                remap::remap_value_ids_fx(op, &map);
            }
            let _dead: FxHashSet<ValueId> = map.keys().copied().collect();
            black_box(&ops);
        }
        let t0 = Instant::now();
        for _ in 0..N_ITERS {
            let map: FxHashMap<ValueId, ValueId> =
                FxHashMap::with_capacity_and_hasher(N_REPLACEMENTS as usize, Default::default());
            let mut map = map;
            for (k, v) in &bt_seed {
                map.insert(*k, *v);
            }
            let mut ops = base_ops.clone();
            for op in ops.iter_mut() {
                remap::remap_value_ids_fx(op, &map);
            }
            let dead: FxHashSet<ValueId> = map.keys().copied().collect();
            black_box(&ops);
            black_box(dead.len());
        }
        let fx_elapsed = t0.elapsed();

        let bt_ns_per = bt_elapsed.as_nanos() as f64 / (N_ITERS * N_OPS) as f64;
        let fx_ns_per = fx_elapsed.as_nanos() as f64 / (N_ITERS * N_OPS) as f64;
        let speedup = bt_ns_per / fx_ns_per;
        let delta_pct = (1.0 - fx_ns_per / bt_ns_per) * 100.0;
        println!();
        println!(
            "=== copy_prop hot path: build map + remap {N_OPS} ops + collect dead set ({N_ITERS} iters) ==="
        );
        println!("  BTreeMap (old)  : {bt_elapsed:?}  ({bt_ns_per:.2} ns/op)");
        println!("  FxHashMap (new) : {fx_elapsed:?}  ({fx_ns_per:.2} ns/op)");
        println!("  speedup         : {speedup:.2}× ({delta_pct:+.1}%)");

        assert!(
            fx_ns_per <= bt_ns_per * 1.05,
            "FxHashMap regressed vs BTreeMap (fx={fx_ns_per:.2} ns, bt={bt_ns_per:.2} ns)"
        );
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        shape::{Dim, Shape},
    };

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn eliminates_cast_of_same_dtype() {
        let mut k = Kernel::new("cast_id");
        // Create a float-producing op
        k.body.push_op(Op::Zeros { dtype: DType::F32, shape: Shape::scalar() }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        // Cast(f32, f32_val) → f32_val; uses of v1 redirected to v0.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "Cast(f32, f32) should redirect to original");
        }
    }

    #[test]
    fn eliminates_broadcast_scalar_shape1() {
        let mut k = Kernel::new("broadcast_id");
        k.body.push_op(Op::Const { value: 5 }, ValueId::new(0));
        k.body.push_op(
            Op::Broadcast { value: ValueId::new(0), shape: Shape::new([Dim::Known(1)]) },
            ValueId::new(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "Broadcast(x, [1]) should redirect to x");
        }
    }

    #[test]
    fn eliminates_select_with_same_branches() {
        let mut k = Kernel::new("select_id");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(1));
        k.body.push_op(
            Op::Select {
                cond: ValueId::new(0),
                on_true: ValueId::new(1),
                on_false: ValueId::new(1),
            },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 1, "Select(cond,a,a) should redirect to a");
        }
    }

    #[test]
    fn preserves_non_identity_cast() {
        let mut k = Kernel::new("cast_real");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0)); // i32
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        // Cast(i32→f32) is NOT an identity (dtype inferred from Const is None), should be kept.
        let has_cast = k.body.ops.iter().any(|op| matches!(op, Op::Cast { .. }));
        assert!(has_cast, "Cast to different dtype should be preserved");
    }

    #[test]
    fn fixpoint_propagates_through_chain() {
        let mut k = Kernel::new("copy_chain");
        k.body.push_op(Op::Zeros { dtype: DType::F32, shape: Shape::scalar() }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op(Op::Cast { value: ValueId::new(1), dtype: DType::F32 }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        // Both Casts are identities → all redirect to v0.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "chain of identity casts should propagate to source");
        }
    }
}

//! Loop Invariant Code Motion (LICM) pass.
//!
//! Identifies computations inside loop bodies whose operands are all defined
//! outside the loop, and hoists them to before the loop. This eliminates
//! redundant re-computation of index arithmetic and const-buffer loads
//! across loop iterations.
//!
//! ## Algorithm
//!
//! For each `Op::Loop` in every block:
//! 1. Build the initial invariant set: all ValueIds defined in the parent block
//!    before the loop (or in ancestor blocks).
//! 2. Iterate to fixpoint: any op in the loop body whose operands are all
//!    invariant AND which has no side effects is marked as hoistable.
//! 3. Hoist: remove hoistable ops from the loop body and insert them before the loop
//!    in the parent block, respecting topological order among hoisted ops.
//!
//! ## Safety
//!
//! Only pure ops are hoisted. The following are NOT hoisted:
//! - `Store`, `Atomic`, `Barrier`, `ThreadgroupStore` (side effects)
//! - `SetLocal` (writes to mutable loop-carried state)
//! - `DeclareLocal` inside loops (mutable variable declaration)
//! - `Load` from mutable/unknown params
//! - Any op whose operands include the loop induction variable

use std::collections::{BTreeMap, BTreeSet};

use metaltile_core::{
    error::Result,
    ir::{Block, BlockId, IndexExpr, Kernel, Op, ParamKind, ValueId},
};

pub struct LicmPass;

impl super::Pass for LicmPass {
    fn name(&self) -> &str { "licm" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Determine which params are read-only (Load-safe for hoisting).
        let read_only: BTreeSet<String> = kernel
            .params
            .iter()
            .filter(|p| !p.is_output && matches!(p.kind, ParamKind::Tensor | ParamKind::Strided))
            .map(|p| p.name.clone())
            .collect();

        // Build a definition map: ValueId -> BlockId where it's defined.
        let mut def_block: BTreeMap<ValueId, BlockId> = BTreeMap::new();
        for vid in kernel.body.results.iter().flatten() {
            def_block.insert(*vid, kernel.body.id);
        }
        for (bid, block) in &kernel.blocks {
            for vid in block.results.iter().flatten() {
                def_block.insert(*vid, *bid);
            }
        }

        // Take all blocks out so we can mutate them freely.
        let mut blocks = std::mem::take(&mut kernel.blocks);

        // Process the body first, then nested blocks.
        // Nested blocks are processed inside-out (post-order): child loops first.
        licm_block(&mut kernel.body, &mut blocks, &def_block, &read_only);

        // Process each block. We need to handle them in post-order:
        // blocks that are children of other blocks should be processed first.
        // Collect them all, sort by dependency depth.
        let mut block_ids: Vec<BlockId> = blocks.keys().copied().collect();
        // Simple heuristic: sort by BlockId descending (newer blocks have higher IDs)
        // This isn't perfect but works for typical linear block allocation.
        block_ids.sort_by_key(|bid| -(bid.as_u32() as i32));

        for bid in block_ids {
            let mut block = blocks.remove(&bid).unwrap();
            licm_block(&mut block, &mut blocks, &def_block, &read_only);
            blocks.insert(bid, block);
        }

        kernel.blocks = blocks;
        Ok(())
    }
}

/// Process a single block, hoisting invariants from any `Op::Loop` children.
/// `blocks` is the mutable block map so loop bodies can be modified.
fn licm_block(
    block: &mut Block,
    blocks: &mut BTreeMap<BlockId, Block>,
    def_block: &BTreeMap<ValueId, BlockId>,
    read_only: &BTreeSet<String>,
) {
    let n = block.ops.len();

    // Phase 1: for each Op::Loop, find which ops to hoist.
    // (loop_idx, (hoisted_ops, hoisted_results, body_block_id, removal_indices))
    struct HoistPlan {
        loop_idx: usize,
        hoisted_ops: Vec<Op>,
        hoisted_results: Vec<Option<ValueId>>,
        body_id: BlockId,
        removal_indices: Vec<usize>,
    }

    let mut plans: Vec<HoistPlan> = Vec::new();

    for i in 0..n {
        if let Op::Loop { body, .. } = &block.ops[i] {
            let Some(loop_body) = blocks.get(body) else {
                continue;
            };

            // Build the initial invariant set: ValueIds defined before position `i`
            // in the parent block, plus any from ancestor blocks.
            let mut invariant: BTreeSet<ValueId> = BTreeSet::new();
            for j in 0..i {
                if let Some(Some(vid)) = block.results.get(j) {
                    invariant.insert(*vid);
                }
            }
            // Also include values from other blocks (ancestors) referenced by the loop.
            for op in &loop_body.ops {
                for vid in op_value_refs(op) {
                    if let Some(&def_bid) = def_block.get(&vid) {
                        if def_bid != *body {
                            invariant.insert(vid);
                        }
                    }
                }
            }

            // Fixpoint: find hoistable ops.
            let mut hoist_indices: Vec<usize> = Vec::new();
            let m = loop_body.ops.len();
            loop {
                let mut changed = false;
                for j in 0..m {
                    if hoist_indices.contains(&j) {
                        continue;
                    }
                    let op = &loop_body.ops[j];
                    if !is_pure_op(op, read_only) {
                        continue;
                    }
                    let op_refs = op_value_refs(op);
                    if op_refs.iter().all(|v| invariant.contains(v)) {
                        if let Some(Some(vid)) = loop_body.results.get(j) {
                            invariant.insert(*vid);
                            hoist_indices.push(j);
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }

            if hoist_indices.is_empty() {
                continue;
            }

            // Sort ascending for topological order.
            hoist_indices.sort();

            let hoisted_ops: Vec<Op> =
                hoist_indices.iter().map(|&j| loop_body.ops[j].clone()).collect();
            let hoisted_results: Vec<Option<ValueId>> =
                hoist_indices.iter().map(|&j| loop_body.results[j]).collect();

            plans.push(HoistPlan {
                loop_idx: i,
                hoisted_ops,
                hoisted_results,
                body_id: *body,
                removal_indices: hoist_indices,
            });
        }
    }

    if plans.is_empty() {
        return;
    }

    // Phase 2: remove hoisted ops from loop bodies.
    for plan in &plans {
        if let Some(loop_body) = blocks.get_mut(&plan.body_id) {
            remove_ops_from_block(loop_body, &plan.removal_indices);
        }
    }

    // Phase 3: rebuild the parent block with hoisted ops inserted before each loop.
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let n2 = old_ops.len();

    // Map: loop_idx -> (ops, results) to insert before it.
    let mut insert_at: BTreeMap<usize, (&[Op], &[Option<ValueId>])> = BTreeMap::new();
    for plan in &plans {
        insert_at.insert(plan.loop_idx, (&plan.hoisted_ops, &plan.hoisted_results));
    }

    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    for i in 0..n2 {
        // Insert hoisted ops before position i if any.
        if let Some(&(hoisted_ops, hoisted_results)) = insert_at.get(&i) {
            for (op, result) in hoisted_ops.iter().zip(hoisted_results.iter()) {
                new_ops.push(op.clone());
                new_results.push(*result);
            }
        }

        new_ops.push(old_ops[i].clone());
        new_results.push(old_results[i]);
    }

    block.ops = new_ops;
    block.results = new_results;
}

/// Remove ops at given indices from a block. Indices must be sorted ascending.
fn remove_ops_from_block(block: &mut Block, indices: &[usize]) {
    let skip: BTreeSet<usize> = indices.iter().copied().collect();
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let mut new_ops = Vec::new();
    let mut new_results = Vec::new();
    for (i, op) in old_ops.into_iter().enumerate() {
        if !skip.contains(&i) {
            new_ops.push(op);
            new_results.push(old_results[i]);
        }
    }
    block.ops = new_ops;
    block.results = new_results;
}

/// Return true if the op is pure (no side effects) and safe to hoist.
fn is_pure_op(op: &Op, read_only: &BTreeSet<String>) -> bool {
    match op {
        Op::BinOp { .. }
        | Op::UnaryOp { .. }
        | Op::Cast { .. }
        | Op::Activation { .. }
        | Op::Select { .. }
        | Op::Const { .. }
        | Op::ProgramId { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Broadcast { .. }
        | Op::Transpose { .. }
        | Op::ExpandDims { .. }
        | Op::Reshape { .. }
        | Op::Slice { .. } => true,

        // Load from a read-only (const) param is pure.
        Op::Load { src, .. } => read_only.contains(src.as_str()),

        // NOT pure — side effects or loop-dependent:
        Op::Store { .. }
        | Op::Atomic { .. }
        | Op::Barrier
        | Op::ThreadgroupStore { .. }
        | Op::SetLocal { .. }
        | Op::DeclareLocal { .. }
        | Op::Loop { .. }
        | Op::If { .. }
        | Op::InlineMsl { .. }
        | Op::VectorStore { .. }
        | Op::Scatter { .. }
        | Op::ThreadgroupLoad { .. }
        | Op::ThreadgroupAlloc { .. }
        | Op::StrideStore { .. }
        | Op::Dequantize { .. }
        | Op::SimdReduce { .. }
        | Op::ArgReduce { .. }
        | Op::FusedElementwise { .. }
        | Op::VectorLoad { .. }
        | Op::StrideReduce { .. }
        | Op::StrideScan { .. }
        | Op::StrideArgReduce { .. }
        | Op::Cat { .. }
        | Op::Gather { .. }
        | Op::Scan { .. }
        | Op::Reduce { .. }
        | Op::Dot { .. }
        | Op::FlashAttention { .. }
        | Op::SlidingWindowAttention { .. }
        | Op::RmsNorm { .. }
        | Op::GatedMlp { .. } => false,
    }
}

/// Return all ValueId references used by an op.
fn op_value_refs(op: &Op) -> Vec<ValueId> {
    let mut refs = Vec::new();
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            refs.push(*lhs);
            refs.push(*rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            refs.push(*value);
        },
        Op::Select { cond, on_true, on_false } => {
            refs.push(*cond);
            refs.push(*on_true);
            refs.push(*on_false);
        },
        Op::Dot { a, b } => {
            refs.push(*a);
            refs.push(*b);
        },
        Op::Load { indices, mask, .. } => {
            for ix in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    refs.push(*v);
                }
            }
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::Store { indices, value, mask, .. } => {
            for ix in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    refs.push(*v);
                }
            }
            refs.push(*value);
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::Loop { start, end, step, .. } => {
            refs.push(*start);
            refs.push(*end);
            refs.push(*step);
        },
        Op::InlineMsl { inputs, .. } => {
            refs.extend(inputs);
        },
        Op::FlashAttention { q, k, v, .. } => {
            refs.push(*q);
            refs.push(*k);
            refs.push(*v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            refs.push(*q);
            refs.push(*k);
            refs.push(*v);
        },
        Op::RmsNorm { x, scale, .. } => {
            refs.push(*x);
            refs.push(*scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            refs.push(*x);
            refs.push(*gate_proj);
            refs.push(*up_proj);
            refs.push(*down_proj);
        },
        Op::FusedElementwise { ops } =>
            for sub in ops {
                refs.extend(op_value_refs(sub));
            },
        Op::VectorLoad { byte_offset, .. } => {
            refs.push(*byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            refs.push(*byte_offset);
            refs.push(*value);
        },
        Op::StrideReduce { offset, stride, end, .. } => {
            refs.push(*offset);
            refs.push(*stride);
            refs.push(*end);
        },
        Op::If { cond, .. } => {
            refs.push(*cond);
        },
        Op::ExpandDims { value, .. } => {
            refs.push(*value);
        },
        Op::Reshape { value, .. } => {
            refs.push(*value);
        },
        Op::Cat { values, .. } => {
            refs.extend(values);
        },
        Op::Gather { indices, .. } => {
            refs.push(*indices);
        },
        Op::Scatter { indices, value, .. } => {
            refs.push(*indices);
            refs.push(*value);
        },
        Op::Atomic { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },
        Op::Scan { value, .. } => {
            refs.push(*value);
        },
        Op::StrideScan { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            refs.push(*value);
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            refs.push(*offset);
            refs.push(*end);
            refs.push(*scalar);
        },
        Op::ThreadgroupLoad { index, .. } => {
            refs.push(*index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => {
            refs.push(*value);
        },
        _ => {},
    }
    refs
}

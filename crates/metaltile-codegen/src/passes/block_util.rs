//! Block Utilities — shared primitives for IR block manipulation.
//!
//! Provides `remove_ops` and `insert_ops` for safely mutating op sequences
//! within a [`Block`].  Used by Unroll, LICM, DSE, IfConversion, ValueSink, and
//! any pass that rewrites the op/results arrays of a block.

use std::collections::BTreeSet;

use metaltile_core::ir::{Block, Op, ValueId};

/// Remove ops at the given indices from a block.
///
/// Indices must be sorted in ascending order. The `ops` and `results` arrays
/// are rebuilt without the marked positions.
pub fn remove_ops(block: &mut Block, indices: &[usize]) {
    if indices.is_empty() {
        return;
    }
    let skip: BTreeSet<usize> = indices.iter().copied().collect();
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let mut new_ops = Vec::with_capacity(old_ops.len().saturating_sub(indices.len()));
    let mut new_results = Vec::with_capacity(old_results.len().saturating_sub(indices.len()));

    for (i, op) in old_ops.into_iter().enumerate() {
        if !skip.contains(&i) {
            new_ops.push(op);
            new_results.push(old_results[i]);
        }
    }

    block.ops = new_ops;
    block.results = new_results;
}

/// Insert ops at position `at` in a block, shifting existing ops down.
///
/// `at` is the insertion position. The newly inserted ops will occupy
/// positions `at .. at + new_ops.len()`. Existing ops at and after `at`
/// are shifted.
pub fn insert_ops(block: &mut Block, at: usize, new_ops: Vec<(Op, Option<ValueId>)>) {
    if new_ops.is_empty() {
        return;
    }
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);

    let mut result_ops = Vec::with_capacity(old_ops.len() + new_ops.len());
    let mut result_vids = Vec::with_capacity(old_results.len() + new_ops.len());

    // Copy ops before the insertion point
    for (i, op) in old_ops.iter().enumerate().take(at) {
        result_ops.push(op.clone());
        result_vids.push(old_results[i]);
    }

    // Insert new ops
    for (op, vid) in new_ops {
        result_ops.push(op);
        result_vids.push(vid);
    }

    // Copy remaining ops
    for (i, op) in old_ops.iter().enumerate().skip(at) {
        result_ops.push(op.clone());
        result_vids.push(old_results[i]);
    }

    block.ops = result_ops;
    block.results = result_vids;
}

/// A single action in a block rebuild plan.
#[derive(Debug, Clone)]
pub enum BlockAction {
    /// Keep this op (from the original block at the corresponding position).
    Keep,
    /// Insert the given ops at this position before the kept op.
    Insert(Vec<(Op, Option<ValueId>)>),
    /// Remove this op (skip it in the output).
    Remove,
}

/// Rebuild a block according to a plan.
///
/// `plan` maps original positions (0..block.ops.len()) to actions.
/// Positions not in the map are kept. Actions are applied in order.
/// This is the most general block manipulation — Unroll's inline_at,
/// LICM's hoist insertion, and DSE's dead-store removal all use
/// patterns that map to this.
pub fn rebuild_block(block: &mut Block, plan: &std::collections::BTreeMap<usize, BlockAction>) {
    if plan.is_empty() {
        return;
    }

    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let n = old_ops.len();

    let mut new_ops = Vec::new();
    let mut new_results = Vec::new();

    for i in 0..n {
        match plan.get(&i) {
            Some(BlockAction::Remove) => {
                // skip this op
            },
            Some(BlockAction::Insert(ops)) => {
                // insert before the current op
                for (op, vid) in ops.iter() {
                    new_ops.push(op.clone());
                    new_results.push(*vid);
                }
                new_ops.push(old_ops[i].clone());
                new_results.push(old_results[i]);
            },
            _ => {
                // Keep (or not in plan)
                new_ops.push(old_ops[i].clone());
                new_results.push(old_results[i]);
            },
        }
    }

    // Insert-after-last: if any plan entry is for position == n
    if let Some(BlockAction::Insert(ops)) = plan.get(&n) {
        for (op, vid) in ops.iter() {
            new_ops.push(op.clone());
            new_results.push(*vid);
        }
    }

    block.ops = new_ops;
    block.results = new_results;
}

//! Loop Unrolling — replicate loop body for known-constant trip counts.
//!
//! Replicates a loop body `trip_count` times when the trip count is a known
//! compile-time constant and ≤ `MAX_UNROLL_TRIP` (8).  This exposes consecutive
//! loads to the vectorizer, eliminates loop overhead (induction variable
//! updates, branch), and increases instruction-level parallelism.
//!
//! ## Induction variable
//!
//! The DSL parser binds loop variables with the convention
//! `ValueId::new(var_id + 1000)`.  Inside the body the variable appears as a
//! direct ValueId reference rather than a Load op.  For each iteration *k* we
//! emit `Op::Const { value: start + k*step }` and remap the IV ValueId to that
//! Const's result.
//!
//! ## Alpha-renaming
//!
//! Every op result defined inside the loop body gets a fresh ValueId for each
//! cloned iteration.  Operands that point into the body are remapped to the
//! clone's fresh IDs; operands pointing **outside** the body pass through
//! unchanged.
//!
//! ## Limitations
//!
//! - Max trip count is 8; loops larger than this remain rolled.
//! - Only handles innermost loops; nested loop unrolling is deferred.
//! - Does not perform partial unrolling with cleanup epilogue.
//!
//! ## References
//! - Bacon, Graham & Sharp (1994), "Compiler Transformations for High-
//!   Performance Computing", ACM Computing Surveys 26(4):345–420.
//!   Surveys loop unrolling and its interactions with other optimizations.
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §9.4.  Standard treatment of loop unrolling.

use std::collections::BTreeMap;

use metaltile_core::{
    error::Result,
    ir::{Block, BlockId, Kernel, Op, ValueId},
};

use super::remap;

const MAX_UNROLL_TRIP: i64 = 8;

pub struct UnrollPass {
    factor: u32,
}

impl UnrollPass {
    pub fn new(factor: u32) -> Self { UnrollPass { factor: factor.min(MAX_UNROLL_TRIP as u32) } }
}

impl Default for UnrollPass {
    fn default() -> Self { UnrollPass::new(4) }
}

impl super::Pass for UnrollPass {
    fn name(&self) -> &str { "unroll" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let max_vid = remap::find_max_vid(kernel);
        let mut next_vid = (max_vid + 1).max(10_000);

        unroll_block(&mut kernel.body, &mut kernel.blocks, &mut next_vid, self.factor);

        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in &block_ids {
            let mut block = kernel.blocks.remove(bid).unwrap();
            unroll_block(&mut block, &mut kernel.blocks, &mut next_vid, self.factor);
            kernel.blocks.insert(*bid, block);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn has_nested_loop_or_barrier(block: &Block) -> bool {
    block.ops.iter().any(|op| matches!(op, Op::Loop { .. } | Op::Barrier))
}

fn find_const_in_block(block: &Block, vid: ValueId) -> Option<i64> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid))
            && let Op::Const { value } = op
        {
            return Some(*value);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// main unroll logic
// ---------------------------------------------------------------------------

fn unroll_block(
    block: &mut Block,
    blocks: &mut BTreeMap<BlockId, Block>,
    next_vid: &mut u32,
    factor: u32,
) {
    let n = block.ops.len();

    struct Plan {
        loop_idx: usize,
        trip_count: i64,
        start_val: i64,
        step_val: i64,
        var_id: u32,
        body_id: BlockId,
    }

    let mut plans: Vec<Plan> = Vec::new();

    for i in 0..n {
        if let Op::Loop { var, start, end, step, body } = &block.ops[i] {
            let Some(body_block) = blocks.get(body) else { continue };
            let Some(start_val) = find_const_in_block(block, *start) else {
                continue;
            };
            let Some(end_val) = find_const_in_block(block, *end) else {
                continue;
            };
            let Some(step_val) = find_const_in_block(block, *step) else {
                continue;
            };
            if step_val <= 0 {
                continue;
            };
            let tc = (end_val - start_val) / step_val;
            if tc <= 0 || tc > factor as i64 {
                continue;
            };
            if has_nested_loop_or_barrier(body_block) {
                continue;
            };
            plans.push(Plan {
                loop_idx: i,
                trip_count: tc,
                start_val,
                step_val,
                var_id: var.as_u32(),
                body_id: *body,
            });
        }
    }

    if plans.is_empty() {
        return;
    }

    // ---- rebuild parent block --------------------------------------------
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);

    // inline_at: loop_idx → inlined ops
    let mut inline_at: BTreeMap<usize, Vec<(Op, Option<ValueId>)>> = BTreeMap::new();

    for plan in &plans {
        let body = blocks.get(&plan.body_id).unwrap();
        let body_n = body.ops.len();

        // IV ValueId (convention: var_id + 1000).
        let iv_vid = ValueId::new(plan.var_id + 1000);

        let mut inlined: Vec<(Op, Option<ValueId>)> = Vec::new();

        for k in 0..plan.trip_count {
            let iv_val = plan.start_val + k * plan.step_val;

            // ---- emit IV Const for this iteration --------------------------
            let iv_const_vid = ValueId::new(*next_vid);
            *next_vid += 1;
            inlined.push((Op::Const { value: iv_val }, Some(iv_const_vid)));

            // ---- build vid-map for this clone ----------------------------
            let mut vid_map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
            vid_map.insert(iv_vid, iv_const_vid);

            for j in 0..body_n {
                if let Some(old_v) = body.results[j] {
                    let new_v = ValueId::new(*next_vid);
                    *next_vid += 1;
                    vid_map.insert(old_v, new_v);
                }
            }

            // ---- clone and remap each body op -----------------------------
            for j in 0..body_n {
                let mut new_op = body.ops[j].clone();
                remap::remap_value_ids(&mut new_op, &vid_map);

                let new_vid = body.results[j].map(|_| vid_map[&body.results[j].unwrap()]);
                inlined.push((new_op, new_vid));
            }
        }

        inline_at.insert(plan.loop_idx, inlined);
    }

    // Assemble.
    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    for i in 0..old_ops.len() {
        if let Some(inlined) = inline_at.get(&i) {
            for (op, vid) in inlined {
                new_ops.push(op.clone());
                new_results.push(*vid);
            }
        } else {
            new_ops.push(old_ops[i].clone());
            new_results.push(old_results[i]);
        }
    }

    block.ops = new_ops;
    block.results = new_results;

    for plan in &plans {
        blocks.remove(&plan.body_id);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use metaltile_core::ir::VarId;

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn unrolls_trip_count_4_loop() {
        let mut k = Kernel::new("unroll_4");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0)); // start
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1)); // end
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2)); // step

        // Loop body: a single Const (uses the IV).
        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 0 }, ValueId::new(100));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        UnrollPass::default().run(&mut k).unwrap();

        // Loop should be gone, body should be removed from blocks.
        let has_loop = k.body.ops.iter().any(|op| matches!(op, Op::Loop { .. }));
        assert!(!has_loop, "Loop should be unrolled and removed");
        assert!(!k.blocks.contains_key(&body_id), "loop body block should be removed");

        // Unrolled body: for trip_count=4, we get: IV Const(0), body clone(0),
        // IV Const(1), body clone(1), IV Const(2), body clone(2), IV Const(3), body clone(3).
        // That's 8 ops (4 IV consts + 4 body clones).
        let iv_consts: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Const { .. })).collect();
        // Original 3 consts (start, end, step) + new 4 IV consts = 7, plus 4 body consts = 11 total
        assert!(iv_consts.len() >= 7, "should have start/end/step consts + 4 IV consts");
    }

    #[test]
    fn does_not_unroll_loop_with_barrier() {
        let mut k = Kernel::new("unroll_barrier");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Barrier, ValueId::new(100)); // Barrier prevents unrolling
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        UnrollPass::default().run(&mut k).unwrap();

        // Loop should still be present.
        let has_loop = k.body.ops.iter().any(|op| matches!(op, Op::Loop { .. }));
        assert!(has_loop, "Loop with Barrier should not be unrolled");
    }

    #[test]
    fn does_not_unroll_large_trip_count() {
        let mut k = Kernel::new("unroll_large");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 100 }, ValueId::new(1)); // trip = 100 > factor(4)
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 0 }, ValueId::new(100));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        UnrollPass::default().run(&mut k).unwrap();

        // Loop should remain because trip_count 100 > default factor 4.
        let has_loop = k.body.ops.iter().any(|op| matches!(op, Op::Loop { .. }));
        assert!(has_loop, "Large trip count loop should not be unrolled");
    }

    #[test]
    fn respects_custom_unroll_factor() {
        let mut k = Kernel::new("unroll_factor_8");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 8 }, ValueId::new(1)); // trip = 8
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 0 }, ValueId::new(100));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        // Default factor 4 won't unroll 8; use factor 8.
        UnrollPass::new(8).run(&mut k).unwrap();

        let has_loop = k.body.ops.iter().any(|op| matches!(op, Op::Loop { .. }));
        assert!(!has_loop, "Trip count 8 should be unrolled with factor 8");
    }

    #[test]
    fn alpha_renames_body_values() {
        // Verify that body values get fresh ValueIds to avoid conflicts.
        let mut k = Kernel::new("unroll_alpha");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 42 }, ValueId::new(10));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        UnrollPass::default().run(&mut k).unwrap();
        // Check that we have more ops now (unrolled) and they use fresh VIDs.
        assert!(k.body.ops.len() > 4, "should have more ops after unrolling");

        // All result VIDs should be unique.
        let mut seen = BTreeSet::new();
        for vid in k.body.results.iter().flatten() {
            assert!(
                seen.insert(vid.as_u32()),
                "duplicate ValueId {} after unrolling",
                vid.as_u32()
            );
        }
    }
}

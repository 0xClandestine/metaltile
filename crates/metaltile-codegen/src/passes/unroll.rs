//! Loop unrolling pass.
//!
//! Replicates a loop body `trip_count` times when the trip count is a known
//! compile-time constant and ≤ `MAX_UNROLL_TRIP`.  This exposes consecutive
//! loads to the vectorizer and eliminates loop overhead.
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

use std::collections::BTreeMap;

use metaltile_core::{
    error::Result,
    ir::{Block, BlockId, IndexExpr, Kernel, Op, ValueId},
};

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
        let max_vid = find_max_vid(kernel);
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

fn find_max_vid(kernel: &Kernel) -> u32 {
    let mut m = 0u32;
    for op in &kernel.body.ops {
        m = m.max(max_vid_in_op(op));
    }
    for block in kernel.blocks.values() {
        for op in &block.ops {
            m = m.max(max_vid_in_op(op));
        }
    }
    for vid in kernel.body.results.iter().flatten() {
        m = m.max(vid.as_u32());
    }
    for block in kernel.blocks.values() {
        for vid in block.results.iter().flatten() {
            m = m.max(vid.as_u32());
        }
    }
    m
}

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
                remap_value_ids(&mut new_op, &vid_map);

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

// ---------------------------------------------------------------------------
// value-id remapping (covers every Op variant)
// ---------------------------------------------------------------------------

fn remap_value_ids(op: &mut Op, map: &BTreeMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&nv) = map.get(v) {
            *v = nv;
        }
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => s(value),
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },
        Op::Store { value, indices, .. } => {
            s(value);
            for ix in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    s(v);
                }
            }
        },
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::Load { indices, .. } =>
            for ix in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    s(v);
                }
            },
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                s(v);
            },
        Op::FlashAttention { q, k, v, .. } => {
            s(q);
            s(k);
            s(v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            s(q);
            s(k);
            s(v);
        },
        Op::RmsNorm { x, scale, .. } => {
            s(x);
            s(scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            s(x);
            s(gate_proj);
            s(up_proj);
            s(down_proj);
        },
        Op::FusedElementwise { ops } =>
            for o in ops {
                remap_value_ids(o, map);
            },
        Op::VectorLoad { byte_offset, .. } => s(byte_offset),
        Op::VectorStore { byte_offset, value, .. } => {
            s(byte_offset);
            s(value);
        },
        Op::StrideReduce { offset, stride, end, .. } => {
            s(offset);
            s(stride);
            s(end);
        },
        Op::If { cond, .. } => s(cond),
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => s(value),
        Op::Cat { values, .. } =>
            for v in values {
                s(v);
            },
        Op::Gather { indices, .. } => s(indices),
        Op::Scatter { indices, value, .. } => {
            s(indices);
            s(value);
        },
        Op::Atomic { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::Scan { value, .. } => s(value),
        Op::StrideScan { offset, end, .. } | Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            s(offset);
            s(end);
            s(scalar);
        },
        Op::Dequantize { .. } => {},
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => s(value),
        Op::ThreadgroupLoad { index, .. } => s(index),
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier => {},
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => s(value),
        _ => {},
    }
}

// ---------------------------------------------------------------------------
// max ValueId in an op
// ---------------------------------------------------------------------------

fn max_vid_in_op(op: &Op) -> u32 {
    let mut m = 0;
    let mut push = |v: ValueId| m = m.max(v.as_u32());
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            push(*lhs);
            push(*rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => push(*value),
        Op::Select { cond, on_true, on_false } => {
            push(*cond);
            push(*on_true);
            push(*on_false);
        },
        Op::Dot { a, b } => {
            push(*a);
            push(*b);
        },
        Op::Store { value, indices, .. } => {
            push(*value);
            for ix in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    push(*v);
                }
            }
        },
        Op::Loop { start, end, step, .. } => {
            push(*start);
            push(*end);
            push(*step);
        },
        Op::Load { indices, .. } =>
            for ix in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    push(*v);
                }
            },
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                push(*v);
            },
        Op::FlashAttention { q, k, v, .. } => {
            push(*q);
            push(*k);
            push(*v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            push(*q);
            push(*k);
            push(*v);
        },
        Op::RmsNorm { x, scale, .. } => {
            push(*x);
            push(*scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            push(*x);
            push(*gate_proj);
            push(*up_proj);
            push(*down_proj);
        },
        Op::FusedElementwise { ops } =>
            for o in ops {
                m = m.max(max_vid_in_op(o));
            },
        Op::VectorLoad { byte_offset, .. } => push(*byte_offset),
        Op::VectorStore { byte_offset, value, .. } => {
            push(*byte_offset);
            push(*value);
        },
        Op::StrideReduce { offset, stride, end, .. } => {
            push(*offset);
            push(*stride);
            push(*end);
        },
        Op::If { cond, .. } => push(*cond),
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => push(*value),
        Op::Cat { values, .. } =>
            for v in values {
                push(*v);
            },
        Op::Gather { indices, .. } => push(*indices),
        Op::Scatter { indices, value, .. } => {
            push(*indices);
            push(*value);
        },
        Op::Atomic { index, value, .. } => {
            push(*index);
            push(*value);
        },
        Op::Scan { value, .. } => push(*value),
        Op::StrideScan { offset, end, .. } | Op::StrideArgReduce { offset, end, .. } => {
            push(*offset);
            push(*end);
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            push(*offset);
            push(*end);
            push(*scalar);
        },
        Op::Dequantize { .. } => {},
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => push(*value),
        Op::ThreadgroupLoad { index, .. } => push(*index),
        Op::ThreadgroupStore { index, value, .. } => {
            push(*index);
            push(*value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier => {},
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => push(*value),
        _ => {},
    }
    m
}

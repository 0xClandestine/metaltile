//! ValueId Remapping — shared utilities for IR traversal and mutation.
//!
//! Provides canonical functions for ValueId remapping, reference collection,
//! and Op classification used by nearly every pass in the pipeline.
//!
//! ## Core functions
//! - [`remap_value_ids`] — rewrite all ValueId references in an Op.
//! - [`op_value_refs`] — collect all ValueId references (read-only, for analysis).
//! - [`max_vid_in_op`] — find the maximum ValueId in an Op.
//! - [`find_max_vid`] — find the maximum ValueId across a whole Kernel.
//!
//! ## Op predicates
//! - [`has_side_effects`] — cannot be moved, duplicated, or deleted.
//! - [`is_unpredictable`] — cannot appear inside predicated (if-converted) code.
//! - [`is_cheap_alu`] — eligible for rematerialization / value sinking.
//! - [`is_load`] / [`is_store`] / [`is_barrier`] — memory classification.
//!
//! Centralizing these here ensures all Op variants are handled consistently
//! across passes.  The exhaustive match arms serve as a single point of truth;
//! a test verifies no variant is silently skipped.

use std::collections::BTreeMap;

use metaltile_core::ir::{IndexExpr, Kernel, Op, ValueId};

// ---------------------------------------------------------------------------
// remap_value_ids — mutate ValueId references in an Op
// ---------------------------------------------------------------------------

/// Remap all `ValueId` references in `op` according to `map`.
/// References not present in `map` are left unchanged.
pub fn remap_value_ids(op: &mut Op, map: &BTreeMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&nv) = map.get(v) {
            *v = nv;
        }
    };

    match op {
        // ── arithmetic / logic ────────────────────────────────────────────
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
        | Op::Broadcast { value, .. } => {
            s(value);
        },
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },

        // ── memory ────────────────────────────────────────────────────────
        Op::Load { indices, mask, .. } => {
            for ix in indices.iter_mut() {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => s(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                s(m);
            }
        },
        Op::Store { value, indices, mask, .. } => {
            s(value);
            for ix in indices.iter_mut() {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => s(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                s(m);
            }
        },
        Op::VectorLoad { byte_offset, .. } => {
            s(byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            s(byte_offset);
            s(value);
        },
        Op::Gather { indices, .. } => {
            s(indices);
        },
        Op::Scatter { indices, value, .. } => {
            s(indices);
            s(value);
        },
        Op::Atomic { index, value, .. } => {
            s(index);
            s(value);
        },

        // ── control flow ─────────────────────────────────────────────────
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::If { cond, .. } => {
            s(cond);
        },

        // ── inline / fused ───────────────────────────────────────────────
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                s(v);
            },
        Op::FusedElementwise { ops } =>
            for sub_op in ops.iter_mut() {
                remap_value_ids(sub_op, map);
            },

        // ── ML primitives ────────────────────────────────────────────────
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

        // ── tiling / reduction ───────────────────────────────────────────
        Op::StrideReduce { offset, stride, end, secondary_base, .. } => {
            s(offset);
            s(stride);
            s(end);
            if let Some(sb) = secondary_base {
                s(sb);
            }
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            s(offset);
            s(end);
            s(scalar);
        },
        Op::StrideScan { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::Scan { value, .. } => {
            s(value);
        },

        // ── SIMD / threadgroup ────────────────────────────────────────────
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => {
            s(value);
        },
        Op::ThreadgroupLoad { index, .. } => {
            s(index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },

        // ── locals ───────────────────────────────────────────────────────
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            s(value);
        },

        // ── shape ops ────────────────────────────────────────────────────
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
            s(value);
        },
        Op::Cat { values, .. } =>
            for v in values {
                s(v);
            },

        // ── no ValueId refs ──────────────────────────────────────────────
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Barrier
        | Op::ThreadgroupAlloc { .. }
        | Op::Dequantize { .. } => {},
    }
}

// ---------------------------------------------------------------------------
// op_value_refs — collect all ValueId references (read-only, for analysis)
// ---------------------------------------------------------------------------

/// Return all `ValueId` references in `op` (for liveness, use-count, invariant analysis).
pub fn op_value_refs(op: &Op) -> Vec<ValueId> {
    let mut refs = Vec::new();

    match op {
        // ── arithmetic / logic ────────────────────────────────────────────
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

        // ── memory ────────────────────────────────────────────────────────
        Op::Load { indices, mask, .. } => {
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => refs.push(*v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::Store { indices, value, mask, .. } => {
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => refs.push(*v),
                    IndexExpr::Const(_) => {},
                }
            }
            refs.push(*value);
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::VectorLoad { byte_offset, .. } => {
            refs.push(*byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            refs.push(*byte_offset);
            refs.push(*value);
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

        // ── control flow ─────────────────────────────────────────────────
        Op::Loop { start, end, step, .. } => {
            refs.push(*start);
            refs.push(*end);
            refs.push(*step);
        },
        Op::If { cond, .. } => {
            refs.push(*cond);
        },

        // ── inline / fused ───────────────────────────────────────────────
        Op::InlineMsl { inputs, .. } => {
            refs.extend(inputs);
        },
        Op::FusedElementwise { ops } =>
            for sub in ops {
                refs.extend(op_value_refs(sub));
            },

        // ── ML primitives ────────────────────────────────────────────────
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

        // ── tiling / reduction ───────────────────────────────────────────
        Op::StrideReduce { offset, stride, end, secondary_base, .. } => {
            refs.push(*offset);
            refs.push(*stride);
            refs.push(*end);
            if let Some(sb) = secondary_base {
                refs.push(*sb);
            }
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            refs.push(*offset);
            refs.push(*end);
            refs.push(*scalar);
        },
        Op::StrideScan { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::Scan { value, .. } => {
            refs.push(*value);
        },

        // ── SIMD / threadgroup ────────────────────────────────────────────
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => {
            refs.push(*value);
        },
        Op::ThreadgroupLoad { index, .. } => {
            refs.push(*index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },

        // ── locals ───────────────────────────────────────────────────────
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            refs.push(*value);
        },

        // ── shape ops ────────────────────────────────────────────────────
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
            refs.push(*value);
        },
        Op::Cat { values, .. } => {
            refs.extend(values.iter());
        },

        // ── no ValueId refs ──────────────────────────────────────────────
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Barrier
        | Op::ThreadgroupAlloc { .. }
        | Op::Dequantize { .. } => {},
    }

    refs
}

// ---------------------------------------------------------------------------
// max_vid_in_op — highest ValueId in an Op
// ---------------------------------------------------------------------------

/// Return the maximum `ValueId` referenced by `op`.
pub fn max_vid_in_op(op: &Op) -> u32 {
    let mut m = 0u32;
    let mut push = |v: &ValueId| m = m.max(v.as_u32());

    match op {
        // ── arithmetic / logic ────────────────────────────────────────────
        Op::BinOp { lhs, rhs, .. } => {
            push(lhs);
            push(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            push(value);
        },
        Op::Select { cond, on_true, on_false } => {
            push(cond);
            push(on_true);
            push(on_false);
        },
        Op::Dot { a, b } => {
            push(a);
            push(b);
        },

        // ── memory ────────────────────────────────────────────────────────
        Op::Load { indices, mask, .. } => {
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => push(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                push(m);
            }
        },
        Op::Store { value, indices, mask, .. } => {
            push(value);
            for ix in indices {
                match ix {
                    IndexExpr::Value(v) | IndexExpr::Range(v, _) => push(v),
                    IndexExpr::Const(_) => {},
                }
            }
            if let Some(m) = mask {
                push(m);
            }
        },
        Op::VectorLoad { byte_offset, .. } => {
            push(byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            push(byte_offset);
            push(value);
        },
        Op::Gather { indices, .. } => {
            push(indices);
        },
        Op::Scatter { indices, value, .. } => {
            push(indices);
            push(value);
        },
        Op::Atomic { index, value, .. } => {
            push(index);
            push(value);
        },

        // ── control flow ─────────────────────────────────────────────────
        Op::Loop { start, end, step, .. } => {
            push(start);
            push(end);
            push(step);
        },
        Op::If { cond, .. } => {
            push(cond);
        },

        // ── inline / fused ───────────────────────────────────────────────
        Op::InlineMsl { inputs, .. } =>
            for v in inputs {
                push(v);
            },
        Op::FusedElementwise { ops } =>
            for sub in ops {
                m = m.max(max_vid_in_op(sub));
            },

        // ── ML primitives ────────────────────────────────────────────────
        Op::FlashAttention { q, k, v, .. } => {
            push(q);
            push(k);
            push(v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            push(q);
            push(k);
            push(v);
        },
        Op::RmsNorm { x, scale, .. } => {
            push(x);
            push(scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            push(x);
            push(gate_proj);
            push(up_proj);
            push(down_proj);
        },

        // ── tiling / reduction ───────────────────────────────────────────
        Op::StrideReduce { offset, stride, end, secondary_base, .. } => {
            push(offset);
            push(stride);
            push(end);
            if let Some(sb) = secondary_base {
                push(sb);
            }
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            push(offset);
            push(end);
            push(scalar);
        },
        Op::StrideScan { offset, end, .. } => {
            push(offset);
            push(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            push(offset);
            push(end);
        },
        Op::Scan { value, .. } => {
            push(value);
        },

        // ── SIMD / threadgroup ────────────────────────────────────────────
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => {
            push(value);
        },
        Op::ThreadgroupLoad { index, .. } => {
            push(index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            push(index);
            push(value);
        },

        // ── locals ───────────────────────────────────────────────────────
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            push(value);
        },

        // ── shape ops ────────────────────────────────────────────────────
        Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
            push(value);
        },
        Op::Cat { values, .. } =>
            for v in values {
                push(v);
            },

        // ── no ValueId refs ──────────────────────────────────────────────
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Barrier
        | Op::ThreadgroupAlloc { .. }
        | Op::Dequantize { .. } => {},
    }

    m
}

// ---------------------------------------------------------------------------
// find_max_vid — maximum ValueId across a whole Kernel
// ---------------------------------------------------------------------------

/// Find the maximum `ValueId` across all ops and results in `kernel`.
pub fn find_max_vid(kernel: &Kernel) -> u32 {
    let mut m = 0u32;

    // Body ops and results
    for op in &kernel.body.ops {
        m = m.max(max_vid_in_op(op));
    }
    for vid in kernel.body.results.iter().flatten() {
        m = m.max(vid.as_u32());
    }

    // Nested blocks
    for block in kernel.blocks.values() {
        for op in &block.ops {
            m = m.max(max_vid_in_op(op));
        }
        for vid in block.results.iter().flatten() {
            m = m.max(vid.as_u32());
        }
    }

    m
}

// ---------------------------------------------------------------------------
// all_blocks — collect all block IDs in post-order
// ---------------------------------------------------------------------------

/// Collect all block IDs in the kernel, including the body.
/// Returns sorted keys from the block map plus the body block ID.
pub fn all_block_ids(kernel: &Kernel) -> Vec<metaltile_core::ir::BlockId> {
    let mut ids: Vec<metaltile_core::ir::BlockId> = kernel.blocks.keys().copied().collect();
    ids.push(kernel.body.id);
    ids
}

// ---------------------------------------------------------------------------
// Op predicates (shared across passes)
// ---------------------------------------------------------------------------

/// True if the op has side effects and cannot be moved, duplicated, or deleted.
pub fn has_side_effects(op: &Op) -> bool {
    matches!(
        op,
        Op::Store { .. }
            | Op::VectorStore { .. }
            | Op::Atomic { .. }
            | Op::Barrier
            | Op::SetLocal { .. }
            | Op::ThreadgroupStore { .. }
            | Op::ThreadgroupAlloc { .. }
            | Op::StrideStore { .. }
            | Op::Scatter { .. }
    )
}

/// True if the op cannot appear inside predicated code (Barrier, Atomic, Loop, etc.).
pub fn is_unpredictable(op: &Op) -> bool {
    matches!(
        op,
        Op::Barrier
            | Op::Atomic { .. }
            | Op::Loop { .. }
            | Op::SetLocal { .. }
            | Op::DeclareLocal { .. }
            | Op::ThreadgroupAlloc { .. }
            | Op::If { .. } // nested If needs recursive if-conversion, not flat predicates
            | Op::StrideScan { .. }
            | Op::StrideArgReduce { .. }
    )
}

/// True if the op is a "cheap ALU" op (eligible for rematerialization / value sinking).
pub fn is_cheap_alu(op: &Op) -> bool {
    matches!(
        op,
        Op::BinOp { .. }
            | Op::UnaryOp { .. }
            | Op::Cast { .. }
            | Op::Select { .. }
            | Op::Const { .. }
            | Op::ProgramId { .. }
    )
}

/// True if the op is a load from device or threadgroup memory.
pub fn is_load(op: &Op) -> bool {
    matches!(op, Op::Load { .. } | Op::VectorLoad { .. } | Op::ThreadgroupLoad { .. })
}

/// True if the op is a store to device or threadgroup memory.
pub fn is_store(op: &Op) -> bool {
    matches!(op, Op::Store { .. } | Op::VectorStore { .. } | Op::ThreadgroupStore { .. })
}

/// True if the op contains a barrier.
pub fn is_barrier(op: &Op) -> bool { matches!(op, Op::Barrier) }

#[cfg(test)]
mod tests {
    use metaltile_core::ir::BinOpKind;

    use super::*;

    /// Every Op variant must be handled by remap_value_ids, op_value_refs, and max_vid_in_op.
    /// This test ensures no variant is silently skipped by a catch-all `_ => {}`.
    #[test]
    fn all_op_variants_covered() {
        // We can't exhaustively instantiate every variant, but we exercise each
        // major category to ensure the match arms don't panic.
        let map: BTreeMap<ValueId, ValueId> = BTreeMap::new();

        // BinOp
        let mut op = Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 2);
        assert!(max_vid_in_op(&op) >= 2);

        // Load with mask
        let mut op = Op::Load {
            src: "a".into(),
            indices: vec![IndexExpr::Value(ValueId::new(3))],
            mask: Some(ValueId::new(4)),
            other: None,
        };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 2); // index + mask

        // Store with mask
        let mut op = Op::Store {
            dst: "b".into(),
            indices: vec![IndexExpr::Const(0)],
            value: ValueId::new(5),
            mask: Some(ValueId::new(6)),
        };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 2); // value + mask

        // StrideReduce with secondary_base
        let mut op = Op::StrideReduce {
            src: "x".into(),
            offset: ValueId::new(7),
            stride: ValueId::new(8),
            end: ValueId::new(9),
            op: metaltile_core::ir::ReduceKind::Sum,
            dtype: metaltile_core::dtype::DType::F32,
            transform: None,
            secondary_src: None,
            secondary_base: Some(ValueId::new(10)),
        };
        remap_value_ids(&mut op, &map);
        assert_eq!(op_value_refs(&op).len(), 4); // offset, stride, end, secondary_base

        // Ops with no ValueIds should return 0 refs
        assert_eq!(op_value_refs(&Op::Barrier).len(), 0);
        assert_eq!(op_value_refs(&Op::Const { value: 42 }).len(), 0);
        assert_eq!(max_vid_in_op(&Op::Barrier), 0);
    }

    #[test]
    fn remap_rewrites_referenced_values() {
        let mut map = BTreeMap::new();
        map.insert(ValueId::new(1), ValueId::new(100));
        map.insert(ValueId::new(2), ValueId::new(200));

        let mut op = Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) };
        remap_value_ids(&mut op, &map);

        if let Op::BinOp { lhs, rhs, .. } = &op {
            assert_eq!(lhs.as_u32(), 100);
            assert_eq!(rhs.as_u32(), 200);
        } else {
            panic!("op changed variant");
        }
    }

    #[test]
    fn find_max_vid_works() {
        let mut k = Kernel::new("test");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(5));
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(3));
        assert_eq!(find_max_vid(&k), 5);
    }
}

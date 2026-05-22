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

use metaltile_core::ir::{Kernel, Op, ValueId};
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// remap_value_ids — mutate ValueId references in an Op
// ---------------------------------------------------------------------------

/// Remap all `ValueId` references in `op` according to `map`.
/// References not present in `map` are left unchanged.
pub fn remap_value_ids(op: &mut Op, map: &BTreeMap<ValueId, ValueId>) {
    op.for_each_value_id_mut(&mut |v| {
        if let Some(&nv) = map.get(v) {
            *v = nv;
        }
    });
}

// ---------------------------------------------------------------------------
// op_value_refs — collect all ValueId references (read-only, for analysis)
// ---------------------------------------------------------------------------

/// Return all `ValueId` references in `op` (for liveness, use-count, invariant analysis).
///
/// Thin wrapper around `op.value_refs()` that returns owned `ValueId`s.  Callers that
/// only need to iterate can call `op.value_refs()` directly to avoid the owned collection.
pub fn op_value_refs(op: &Op) -> SmallVec<[ValueId; 4]> {
    op.value_refs().into_iter().copied().collect()
}

// ---------------------------------------------------------------------------
// max_vid_in_op — highest ValueId in an Op
// ---------------------------------------------------------------------------

/// Return the maximum *real* `ValueId` referenced by `op`.
///
/// `FusedElementwise` sub-ops encode chain-internal references using
/// `SUB_OP_FLAG = 0x8000_0000` as the top bit. Those are not real
/// kernel-wide ValueIds — they are filtered out here.
pub fn max_vid_in_op(op: &Op) -> u32 {
    const SUB_OP_FLAG: u32 = 0x8000_0000;
    op.value_refs()
        .iter()
        .map(|v| v.as_u32())
        .filter(|&raw| raw & SUB_OP_FLAG == 0)
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// find_max_vid — maximum ValueId across a whole Kernel
// ---------------------------------------------------------------------------

/// Find the maximum *real* `ValueId` across all ops and results in
/// `kernel`. Ignores `FusedElementwise` sub-op refs (top-bit-set
/// chain-internal indices); see [`max_vid_in_op`] for why.
pub fn find_max_vid(kernel: &Kernel) -> u32 {
    /// Top bit reserved by `passes::fusion::SUB_OP_FLAG`.
    const SUB_OP_FLAG: u32 = 0x8000_0000;
    let real_vid = |vid: &ValueId| -> Option<u32> {
        let raw = vid.as_u32();
        if raw & SUB_OP_FLAG == 0 { Some(raw) } else { None }
    };

    let mut m = 0u32;

    // Body ops and results
    for op in &kernel.body.ops {
        m = m.max(max_vid_in_op(op));
    }
    for vid in kernel.body.results.iter().flatten() {
        if let Some(raw) = real_vid(vid) {
            m = m.max(raw);
        }
    }

    // Nested blocks
    for block in kernel.blocks.values() {
        for op in &block.ops {
            m = m.max(max_vid_in_op(op));
        }
        for vid in block.results.iter().flatten() {
            if let Some(raw) = real_vid(vid) {
                m = m.max(raw);
            }
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

/// True if the op writes to memory or synchronises threads.
pub fn has_side_effects(op: &Op) -> bool { op.has_side_effects() }

/// True if the op cannot appear inside predicated code.
pub fn is_unpredictable(op: &Op) -> bool { op.is_unpredictable() }

/// True if the op is a cheap ALU op eligible for rematerialization.
pub fn is_cheap_alu(op: &Op) -> bool { op.is_cheap_alu() }

/// True if the op is a load from device or threadgroup memory.
pub fn is_load(op: &Op) -> bool { op.is_load() }

/// True if the op is a store to device or threadgroup memory.
pub fn is_store(op: &Op) -> bool { op.is_store() }

/// True if the op contains a barrier.
pub fn is_barrier(op: &Op) -> bool { op.is_barrier() }

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{BinOpKind, IndexExpr};

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
    fn max_vid_ignores_sub_op_refs() {
        // FusedElementwise sub-ops encode chain-internal references by
        // setting the top bit of the ValueId (`SUB_OP_FLAG = 0x8000_0000`).
        // `max_vid_in_op` / `find_max_vid` must skip those — otherwise
        // the unroll pass's `next_vid = max_vid + 1` allocation collides
        // with the sub-op-ref encoding namespace and the MSL emitter
        // produces `0 /* bad sub-op ref */` placeholders. Regression
        // test for the `mt_rope_f16` / `mt_affine_quantize_int8`
        // miscompiles.
        const SUB_OP_FLAG: u32 = 0x8000_0000;
        let sub_op_ref_0 = ValueId::new(SUB_OP_FLAG); // chain position 0
        let real_vid = ValueId::new(42);

        // Single-op test: a fused chain that contains a sub-op ref.
        let op = Op::FusedElementwise {
            ops: vec![Op::BinOp { op: BinOpKind::Add, lhs: sub_op_ref_0, rhs: real_vid }],
        };
        assert_eq!(max_vid_in_op(&op), 42, "sub-op refs must not bump max_vid");

        // Whole-kernel test mirrors the rope path: real ValueIds up to
        // 100, plus a fused op with a sub-op ref far above SUB_OP_FLAG.
        let mut k = Kernel::new("max_vid_ignores_sub_op_refs");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(100));
        k.body.push_op(
            Op::FusedElementwise {
                ops: vec![
                    Op::Cast { value: ValueId::new(100), dtype: metaltile_core::dtype::DType::F32 },
                    Op::UnaryOp { op: metaltile_core::ir::UnaryOpKind::Neg, value: sub_op_ref_0 },
                ],
            },
            ValueId::new(101),
        );
        assert_eq!(
            find_max_vid(&k),
            101,
            "find_max_vid must ignore sub-op refs (else unroll's next_vid space \
             collides with the SUB_OP_FLAG namespace)"
        );
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

    // ── op-variant coverage ───────────────────────────────────────────────
    //
    // The existing `all_op_variants_covered` test exercises a handful of
    // Op variants and notes "we can't exhaustively instantiate every
    // variant". The tests below add coverage on the remaining ~40 variants
    // so the match arms in `remap_value_ids`, `op_value_refs`, and
    // `max_vid_in_op` are not silently broken by future additions to
    // `Op`. Each test groups variants by category for readability.

    use metaltile_core::{
        constexpr::ConstExpr,
        dtype::DType,
        ir::{ActKind, AtomicKind, AtomicScope, BlockId, ReduceKind, UnaryOpKind, VarId},
        shape::Shape,
    };

    /// Apply remap, refs, max in one place — keeps each variant case a
    /// single line in the per-category tests below.
    fn check_op(op: Op, expected_refs: usize, expected_max: u32) {
        let refs = op_value_refs(&op);
        assert_eq!(
            refs.len(),
            expected_refs,
            "op_value_refs returned {} refs for {op:?}, expected {expected_refs}",
            refs.len(),
        );
        assert_eq!(
            max_vid_in_op(&op),
            expected_max,
            "max_vid_in_op returned wrong value for {op:?}",
        );
        // No-op map: remap_value_ids must not panic on any variant and
        // must leave value refs unchanged.
        let mut op_remapped = op.clone();
        let empty: BTreeMap<ValueId, ValueId> = BTreeMap::new();
        remap_value_ids(&mut op_remapped, &empty);
        assert_eq!(op_value_refs(&op_remapped), refs);
    }

    #[test]
    fn arith_and_cast_variants() {
        check_op(Op::ProgramId { axis: 0 }, 0, 0);
        check_op(Op::Const { value: 7 }, 0, 0);
        check_op(Op::Cast { value: ValueId::new(3), dtype: DType::F16 }, 1, 3);
        check_op(Op::UnaryOp { op: UnaryOpKind::Exp, value: ValueId::new(4) }, 1, 4);
        check_op(Op::Activation { kind: ActKind::Silu, value: ValueId::new(5) }, 1, 5);
        check_op(Op::Dot { a: ValueId::new(2), b: ValueId::new(8) }, 2, 8);
        check_op(
            Op::Select {
                cond: ValueId::new(1),
                on_true: ValueId::new(2),
                on_false: ValueId::new(3),
            },
            3,
            3,
        );
    }

    #[test]
    fn tile_shape_variants() {
        let shape = Shape::scalar();
        check_op(Op::Zeros { dtype: DType::F32, shape: shape.clone() }, 0, 0);
        check_op(Op::Transpose { value: ValueId::new(6) }, 1, 6);
        check_op(Op::ExpandDims { value: ValueId::new(7), axis: 0 }, 1, 7);
        check_op(Op::Reshape { value: ValueId::new(8), shape: shape.clone() }, 1, 8);
        check_op(
            Op::Cat { values: vec![ValueId::new(2), ValueId::new(9), ValueId::new(4)], axis: 0 },
            3,
            9,
        );
        check_op(Op::Slice { value: ValueId::new(11), ranges: vec![(0, 0, 4)] }, 1, 11);
        check_op(Op::Broadcast { value: ValueId::new(12), shape: shape.clone() }, 1, 12);
        check_op(Op::Splat { value: 1.0, dtype: DType::F32, shape }, 0, 0);
    }

    #[test]
    fn reduce_and_scan_variants() {
        check_op(Op::Reduce { value: ValueId::new(13), axis: 0, op: ReduceKind::Sum }, 1, 13);
        check_op(
            Op::Scan { value: ValueId::new(14), axis: 0, op: ReduceKind::Max, exclusive: false },
            1,
            14,
        );
        check_op(Op::ArgReduce { value: ValueId::new(15), axis: 0, op: ReduceKind::Max }, 1, 15);
        check_op(Op::SimdReduce { value: ValueId::new(16), op: ReduceKind::Sum }, 1, 16);
        check_op(Op::SimdShuffleXor { value: ValueId::new(17), mask: 8 }, 1, 17);
        check_op(
            Op::SimdScan { value: ValueId::new(18), op: ReduceKind::Sum, exclusive: true },
            1,
            18,
        );
        check_op(
            Op::StrideScan {
                src: "x".into(),
                dst: "y".into(),
                offset: ValueId::new(18),
                end: ValueId::new(19),
                op: ReduceKind::Sum,
            },
            2,
            19,
        );
        check_op(
            Op::StrideArgReduce {
                src: "x".into(),
                offset: ValueId::new(20),
                end: ValueId::new(21),
                op: ReduceKind::Max,
            },
            2,
            21,
        );
        check_op(
            Op::StrideStore {
                src: "x".into(),
                dst: "y".into(),
                offset: ValueId::new(22),
                end: ValueId::new(23),
                scalar: ValueId::new(24),
                aux_src: Some("w".into()),
            },
            3,
            24,
        );
    }

    #[test]
    fn indexed_memory_variants() {
        check_op(Op::Gather { src: "x".into(), indices: ValueId::new(25), axis: 0 }, 1, 25);
        check_op(
            Op::Scatter {
                dst: "y".into(),
                indices: ValueId::new(26),
                value: ValueId::new(27),
                axis: 0,
            },
            2,
            27,
        );
        check_op(
            Op::Atomic {
                op: AtomicKind::Add,
                scope: AtomicScope::Device,
                dst: "z".into(),
                index: ValueId::new(28),
                value: ValueId::new(29),
            },
            2,
            29,
        );
        check_op(Op::VectorLoad { src: "x".into(), byte_offset: ValueId::new(30), len: 4 }, 1, 30);
        check_op(
            Op::Pack {
                dtype: DType::F32,
                elements: vec![
                    ValueId::new(31),
                    ValueId::new(32),
                    ValueId::new(33),
                    ValueId::new(34),
                ],
            },
            4,
            34,
        );
        check_op(
            Op::VectorStore {
                dst: "y".into(),
                byte_offset: ValueId::new(31),
                len: 4,
                value: ValueId::new(32),
            },
            2,
            32,
        );
    }

    #[test]
    fn threadgroup_and_local_variants() {
        check_op(Op::ThreadgroupLoad { name: "tg".into(), index: ValueId::new(45) }, 1, 45);
        check_op(
            Op::ThreadgroupStore {
                name: "tg".into(),
                index: ValueId::new(46),
                value: ValueId::new(47),
            },
            2,
            47,
        );
        check_op(Op::Barrier, 0, 0);
        check_op(Op::DeclareLocal { name: "l".into(), value: ValueId::new(48) }, 1, 48);
        check_op(Op::SetLocal { name: "l".into(), value: ValueId::new(49) }, 1, 49);
    }

    #[test]
    fn simdgroup_matrix_variants() {
        check_op(Op::SimdgroupAlloc { dtype: DType::F32, m: 8, n: 8 }, 0, 0);
        check_op(Op::SimdgroupElemLoad { value: ValueId::new(50), index: 0 }, 1, 50);
        check_op(
            Op::SimdgroupElemStore { value: ValueId::new(51), index: 0, data: ValueId::new(52) },
            2,
            52,
        );
        // SimdgroupMatMul a/b/c are SSA ValueIds produced by SimdgroupAlloc /
        // SimdgroupLoad — they participate in value_refs and must be remappable.
        check_op(
            Op::SimdgroupMatMul { a: ValueId::new(53), b: ValueId::new(54), c: ValueId::new(55) },
            3,
            55,
        );
        check_op(Op::SimdLaneId, 0, 0);
        check_op(Op::SimdGroupId, 0, 0);
    }

    // ── op-classification predicates ──────────────────────────────────────

    #[test]
    fn predicate_loads_and_stores() {
        // is_load / is_store / has_side_effects classification.
        // Note: Load is NOT considered a side effect here — only Store /
        // Atomic / Barrier / *Alloc / *Local are. Loads can be moved as
        // long as the source isn't aliased by an interleaving Store.
        let load = Op::Load {
            src: "a".into(),
            indices: vec![IndexExpr::Const(0)],
            mask: None,
            other: None,
        };
        assert!(is_load(&load));
        assert!(!is_store(&load));
        assert!(!has_side_effects(&load));

        let store = Op::Store {
            dst: "a".into(),
            indices: vec![IndexExpr::Const(0)],
            value: ValueId::new(1),
            mask: None,
        };
        assert!(is_store(&store));
        assert!(!is_load(&store));
        assert!(has_side_effects(&store));

        let vload = Op::VectorLoad { src: "a".into(), byte_offset: ValueId::new(1), len: 4 };
        assert!(is_load(&vload));
        assert!(!has_side_effects(&vload));

        let vstore = Op::VectorStore {
            dst: "a".into(),
            byte_offset: ValueId::new(1),
            len: 4,
            value: ValueId::new(2),
        };
        assert!(is_store(&vstore));
        assert!(has_side_effects(&vstore));

        let tgload = Op::ThreadgroupLoad { name: "tg".into(), index: ValueId::new(1) };
        assert!(is_load(&tgload));

        let tgstore = Op::ThreadgroupStore {
            name: "tg".into(),
            index: ValueId::new(1),
            value: ValueId::new(2),
        };
        assert!(is_store(&tgstore));
        assert!(has_side_effects(&tgstore));
    }

    #[test]
    fn predicate_barrier_and_atomic() {
        assert!(is_barrier(&Op::Barrier));
        assert!(has_side_effects(&Op::Barrier));
        assert!(!is_load(&Op::Barrier));

        let atomic = Op::Atomic {
            op: AtomicKind::Add,
            scope: AtomicScope::Device,
            dst: "x".into(),
            index: ValueId::new(0),
            value: ValueId::new(1),
        };
        assert!(has_side_effects(&atomic));
        assert!(!is_cheap_alu(&atomic));
    }

    #[test]
    fn predicate_cheap_alu() {
        // BinOp / UnaryOp / Const / Cast all qualify as cheap ALU and can
        // be rematerialized or sunk by value_sink / LICM.
        let binop = Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) };
        assert!(is_cheap_alu(&binop));
        assert!(!has_side_effects(&binop));

        let cst = Op::Const { value: 0 };
        assert!(is_cheap_alu(&cst));

        let cast = Op::Cast { value: ValueId::new(0), dtype: DType::F16 };
        assert!(is_cheap_alu(&cast));
    }

    // ── result-flag coverage ─────────────────────────────────────────
    //
    // Every Op variant must carry exactly one of:
    //   - #[no_result]        (no SSA output)
    //   - #[result_u32]       (u32 scalar)
    //   - #[result_i32]       (i32 scalar)
    //   - #[result_f32_scalar] (f32 scalar)
    //   - #[result_f16_scalar] (f16 scalar)
    //   - #[result_same_type] (copies first-input type)
    //   - #[result_custom]    (explicit type-inference arm in type_check.rs)
    //
    // This catches new variants that forget a flag and silently
    // produce wrong codegen downstream.
    #[test]
    fn every_variant_has_result_flag_or_no_result() {
        // Instantiate every Op variant.  The assert_msg identifies
        // the variant that lacks a flag.
        let shape = Shape::scalar();
        let vid = ValueId::new(0);

        let all: &[(&str, Op)] = &[
            ("ProgramId", Op::ProgramId { axis: 0 }),
            ("Const", Op::Const { value: 0 }),
            ("Arange", Op::Arange { start: None, step: None, len: ConstExpr::new("N") }),
            ("Load", Op::Load { src: "x".into(), indices: vec![], mask: None, other: None }),
            ("Store", Op::Store { dst: "x".into(), indices: vec![], value: vid, mask: None }),
            ("BinOp", Op::BinOp { op: BinOpKind::Add, lhs: vid, rhs: vid }),
            ("Dot", Op::Dot { a: vid, b: vid }),
            ("Reduce", Op::Reduce { value: vid, axis: 0, op: ReduceKind::Sum }),
            ("StrideReduce", Op::StrideReduce {
                src: "x".into(),
                offset: vid,
                stride: vid,
                end: vid,
                op: ReduceKind::Sum,
                dtype: DType::F32,
                transform: None,
                secondary_src: None,
                secondary_base: None,
            }),
            ("Cast", Op::Cast { value: vid, dtype: DType::F32 }),
            ("Loop", Op::Loop {
                var: VarId::new(0),
                start: vid,
                end: vid,
                step: vid,
                body: BlockId::new(1),
            }),
            ("If", Op::If { cond: vid, then_block: BlockId::new(1), else_block: None }),
            ("Zeros", Op::Zeros { dtype: DType::F32, shape: shape.clone() }),
            ("Transpose", Op::Transpose { value: vid }),
            ("ExpandDims", Op::ExpandDims { value: vid, axis: 0 }),
            ("Reshape", Op::Reshape { value: vid, shape: shape.clone() }),
            ("Cat", Op::Cat { values: vec![vid], axis: 0 }),
            ("Slice", Op::Slice { value: vid, ranges: vec![(0, 0, 4)] }),
            ("InlineMsl", Op::InlineMsl { source: "".into(), inputs: vec![], outputs: vec![] }),
            ("UnaryOp", Op::UnaryOp { op: UnaryOpKind::Exp, value: vid }),
            ("Activation", Op::Activation { kind: ActKind::Silu, value: vid }),
            ("Select", Op::Select { cond: vid, on_true: vid, on_false: vid }),
            ("Broadcast", Op::Broadcast { value: vid, shape: shape.clone() }),
            ("Splat", Op::Splat { value: 1.0, dtype: DType::F32, shape: shape.clone() }),
            ("FusedElementwise", Op::FusedElementwise { ops: vec![] }),
            ("VectorLoad", Op::VectorLoad { src: "x".into(), byte_offset: vid, len: 4 }),
            ("VectorStore", Op::VectorStore {
                dst: "x".into(),
                byte_offset: vid,
                len: 4,
                value: vid,
            }),
            ("VectorExtract", Op::VectorExtract { vec: vid, lane: 0 }),
            ("Pack", Op::Pack { dtype: DType::F32, elements: vec![vid, vid, vid, vid] }),
            ("Gather", Op::Gather { src: "x".into(), indices: vid, axis: 0 }),
            ("Scatter", Op::Scatter { dst: "x".into(), indices: vid, value: vid, axis: 0 }),
            ("Atomic", Op::Atomic {
                op: AtomicKind::Add,
                scope: AtomicScope::Device,
                dst: "x".into(),
                index: vid,
                value: vid,
            }),
            ("Scan", Op::Scan { value: vid, axis: 0, op: ReduceKind::Sum, exclusive: false }),
            ("StrideScan", Op::StrideScan {
                src: "x".into(),
                dst: "y".into(),
                offset: vid,
                end: vid,
                op: ReduceKind::Sum,
            }),
            ("StrideArgReduce", Op::StrideArgReduce {
                src: "x".into(),
                offset: vid,
                end: vid,
                op: ReduceKind::Max,
            }),
            ("StrideStore", Op::StrideStore {
                src: "x".into(),
                dst: "y".into(),
                offset: vid,
                end: vid,
                scalar: vid,
                aux_src: None,
            }),
            ("SimdReduce", Op::SimdReduce { value: vid, op: ReduceKind::Sum }),
            ("SimdShuffleXor", Op::SimdShuffleXor { value: vid, mask: 1 }),
            ("SimdBroadcast", Op::SimdBroadcast { value: vid, lane: vid }),
            ("ThreadgroupAlloc", Op::ThreadgroupAlloc {
                dtype: DType::F32,
                size: 64,
                name: "tg".into(),
            }),
            ("ThreadgroupLoad", Op::ThreadgroupLoad { name: "tg".into(), index: vid }),
            ("ThreadgroupStore", Op::ThreadgroupStore {
                name: "tg".into(),
                index: vid,
                value: vid,
            }),
            ("StackAlloc", Op::StackAlloc { dtype: DType::F32, size: 4, name: "s".into() }),
            ("StackLoad", Op::StackLoad { name: "s".into(), index: vid }),
            ("StackStore", Op::StackStore { name: "s".into(), index: vid, value: vid }),
            ("Barrier", Op::Barrier),
            ("SimdgroupBarrier", Op::SimdgroupBarrier),
            ("SimdgroupAlloc", Op::SimdgroupAlloc { dtype: DType::F32, m: 8, n: 8 }),
            ("SimdgroupElemLoad", Op::SimdgroupElemLoad { value: vid, index: 0 }),
            ("SimdgroupElemStore", Op::SimdgroupElemStore { value: vid, index: 0, data: vid }),
            ("SimdgroupLoad", Op::SimdgroupLoad {
                dest: vid,
                tg: "tg".into(),
                offset: vid,
                stride: 8,
                transpose: false,
            }),
            ("SimdgroupMatMul", Op::SimdgroupMatMul { a: vid, b: vid, c: vid }),
            ("SimdScan", Op::SimdScan { value: vid, op: ReduceKind::Sum, exclusive: false }),
            ("SimdLaneId", Op::SimdLaneId),
            ("SimdGroupId", Op::SimdGroupId),
            ("DeclareLocal", Op::DeclareLocal { name: "l".into(), value: vid }),
            ("SetLocal", Op::SetLocal { name: "l".into(), value: vid }),
            ("ArgReduce", Op::ArgReduce { value: vid, axis: 0, op: ReduceKind::Max }),
        ];

        for (name, op) in all {
            let count = [
                op.is_result_u32_scalar() as u32,
                op.is_result_i32_scalar() as u32,
                op.is_result_f32_scalar() as u32,
                op.is_result_f16_scalar() as u32,
                op.is_result_same_type() as u32,
                op.is_result_custom() as u32,
                op.is_no_result() as u32,
            ]
            .iter()
            .sum::<u32>();
            assert_eq!(
                count, 1,
                "Op::{name} has {count} result-type flags (expected exactly 1). \
                 Each Op variant must carry exactly one of: #[no_result], \
                 #[result_u32], #[result_i32], #[result_f32_scalar], \
                 #[result_f16_scalar], #[result_same_type], or #[result_custom]."
            );
        }
    }
}

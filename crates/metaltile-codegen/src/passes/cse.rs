//! Common Subexpression Elimination (CSE) pass.
//!
//! Performs local value numbering on each block: when two ops compute the same
//! result (identical opcode and operands), the second is eliminated and all
//! downstream uses are rerouted to the first.
//!
//! ## CSE-eligible ops
//!
//! | Op | Notes |
//! |----|-------|
//! | `BinOp` | Commutative ops (Add, Mul, Max, Min, BitAnd, BitOr, BitXor, CmpEq, CmpNe) are canonicalized |
//! | `UnaryOp` | |
//! | `Cast` | |
//! | `Activation` | |
//! | `Select` | All three operands must match |
//! | `Load` | Only from read-only (const) params |
//!
//! Never eligible: `Store`, `Reduce`, `StrideReduce`, `Loop`, `Barrier`, `Atomic`,
//! and any other op with side effects.

use std::collections::HashMap;

use metaltile_core::{
    error::Result,
    ir::{BinOpKind, Block, BlockId, IndexExpr, Kernel, Op, ParamKind, ValueId},
};

use metaltile_core::ir::UnaryOpKind;
use metaltile_core::ir::ActKind;
use metaltile_core::dtype::DType;

/// A structural key for CSE: captures the opcode and operands in a hashable form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum OpKey {
    BinOp { op: BinOpKind, lhs: u32, rhs: u32 },
    UnaryOp { op: UnaryOpKind, value: u32 },
    Cast { dtype: DType, value: u32 },
    Activation { kind: ActKind, value: u32 },
    Select { cond: u32, on_true: u32, on_false: u32 },
    Load { src: String, idx0: IndexExpr },
}

pub struct CsePass;

impl super::Pass for CsePass {
    fn name(&self) -> &str { "cse" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Determine which params are read-only.
        let read_only: std::collections::BTreeSet<String> = kernel
            .params
            .iter()
            .filter(|p| !p.is_output && matches!(p.kind, ParamKind::Tensor | ParamKind::Strided))
            .map(|p| p.name.clone())
            .collect();

        // CSE on the body block.
        cse_block(&mut kernel.body, &read_only);

        // CSE on all nested blocks.
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get_mut(bid) {
                cse_block(block, &read_only);
            }
        }

        Ok(())
    }
}

fn cse_block(block: &mut Block, read_only: &std::collections::BTreeSet<String>) {
    let n = block.ops.len();

    // Phase 1: find duplicates and build old_vid -> replacement_vid map.
    let mut table: HashMap<OpKey, ValueId> = HashMap::new();
    let mut old_to_new: HashMap<ValueId, ValueId> = HashMap::new();
    let mut skip: Vec<bool> = vec![false; n];

    for i in 0..n {
        let Some(key) = op_key(&block.ops[i], read_only) else {
            continue;
        };
        let Some(&Some(vid)) = block.results.get(i) else {
            continue;
        };

        if let Some(&existing_vid) = table.get(&key) {
            // Duplicate found: remap `vid` to `existing_vid`.
            old_to_new.insert(vid, existing_vid);
            skip[i] = true;
        } else {
            table.insert(key, vid);
        }
    }

    if old_to_new.is_empty() {
        return;
    }

    // Phase 2: remap ValueId references in all surviving ops.
    for op in block.ops.iter_mut() {
        replace_values(op, &old_to_new);
    }

    // Phase 3: rebuild the block without skipped ops.
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    for i in 0..n {
        if !skip[i] {
            new_ops.push(old_ops[i].clone());
            new_results.push(old_results[i]);
        }
    }

    block.ops = new_ops;
    block.results = new_results;
}

/// Build an OpKey for an op if it's CSE-eligible.
fn op_key(op: &Op, read_only: &std::collections::BTreeSet<String>) -> Option<OpKey> {
    match op {
        Op::BinOp { op: kind, lhs, rhs } => {
            let (l, r) = canonicalize_binop(*kind, lhs.as_u32(), rhs.as_u32());
            Some(OpKey::BinOp { op: *kind, lhs: l, rhs: r })
        },
        Op::UnaryOp { op: kind, value } => {
            Some(OpKey::UnaryOp { op: *kind, value: value.as_u32() })
        },
        Op::Cast { dtype, value } => Some(OpKey::Cast { dtype: *dtype, value: value.as_u32() }),
        Op::Activation { kind, value } => {
            Some(OpKey::Activation { kind: *kind, value: value.as_u32() })
        },
        Op::Select { cond, on_true, on_false } => Some(OpKey::Select {
            cond: cond.as_u32(),
            on_true: on_true.as_u32(),
            on_false: on_false.as_u32(),
        }),
        Op::Load { src, indices, .. } =>
            if read_only.contains(src.as_str()) && indices.len() == 1 {
                Some(OpKey::Load { src: src.clone(), idx0: indices[0].clone() })
            } else {
                None
            },
        _ => None,
    }
}

/// For commutative binary ops, sort operands so that `a+b` and `b+a` hash identically.
fn canonicalize_binop(op: BinOpKind, lhs: u32, rhs: u32) -> (u32, u32) {
    let is_commutative = matches!(
        op,
        BinOpKind::Add
            | BinOpKind::Mul
            | BinOpKind::Max
            | BinOpKind::Min
            | BinOpKind::And
            | BinOpKind::Or
            | BinOpKind::Xor
            | BinOpKind::BitAnd
            | BinOpKind::BitOr
            | BinOpKind::BitXor
            | BinOpKind::CmpEq
            | BinOpKind::CmpNe
    );
    if is_commutative && lhs > rhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

/// Replace all ValueId references in `op` using the remapping map.
fn replace_values(op: &mut Op, map: &HashMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new_v) = map.get(v) {
            *v = new_v;
        }
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. } => s(value),
        Op::Activation { value, .. } => s(value),
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Broadcast { value, .. } => s(value),
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },
        Op::Store { value, indices, .. } => {
            s(value);
            for idx in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    s(v);
                }
            }
        },
        Op::Cast { value, .. } => s(value),
        Op::Reduce { value, .. } => s(value),
        Op::Transpose { value } => s(value),
        Op::Slice { value, .. } => s(value),
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::Load { indices, .. } =>
            for idx in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    s(v);
                }
            },
        Op::InlineMsl { inputs, .. } =>
            for v in inputs.iter_mut() {
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
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. } => {},
        Op::FusedElementwise { ops } =>
            for op in ops.iter_mut() {
                replace_values(op, map);
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
        Op::ExpandDims { value, .. } => s(value),
        Op::Reshape { value, .. } => s(value),
        Op::Cat { values, .. } =>
            for v in values.iter_mut() {
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
        Op::StrideScan { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            s(offset);
            s(end);
            s(scalar);
        },
        Op::Dequantize { .. } => {},
        Op::SimdReduce { value, .. } => s(value),
        Op::ThreadgroupLoad { index, .. } => s(index),
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier => {},
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => s(value),
        Op::ArgReduce { value, .. } => s(value),
    }
}

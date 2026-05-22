//! Fused elementwise chain emission.
//!
//! Handles `Op::FusedElementwise` lowering: walks the fused op tree and emits
//! a single compound MSL expression without intermediate temporaries.

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, Block, Kernel, Op, ValueId},
};

use super::MslGenerator;
use crate::passes::{
    fusion::{SUB_OP_FLAG, is_sub_op_ref},
    type_check::TypeEnv,
};

/// Recover the dtype of a fused sub-op's result. Walks the fused chain when
/// the operand ValueId points to another sub-op (marked by `SUB_OP_FLAG`),
/// falls back to the global `type_env` for cross-op operands. Returns `None`
/// when the dtype can't be determined locally — caller should treat that as
/// "don't apply dtype-dependent peepholes."
fn infer_fused_op_dtype(vid: ValueId, fused_ops: &[Op], type_env: &TypeEnv) -> Option<DType> {
    let raw = vid.as_u32();
    if is_sub_op_ref(raw) {
        let idx = (raw & !SUB_OP_FLAG) as usize;
        let op = fused_ops.get(idx)?;
        match op {
            Op::Cast { dtype, .. } => Some(*dtype),
            Op::BinOp { lhs, .. } => infer_fused_op_dtype(*lhs, fused_ops, type_env),
            Op::UnaryOp { value, .. } | Op::Activation { value, .. } =>
                infer_fused_op_dtype(*value, fused_ops, type_env),
            Op::Select { on_true, .. } => infer_fused_op_dtype(*on_true, fused_ops, type_env),
            _ => None,
        }
    } else {
        type_env.get(&vid).map(|tv| tv.dtype)
    }
}

impl MslGenerator {
    /// Emit a fused expression rooted at `fused_ops[idx]`, returning the MSL expression string.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_fused_expr(
        &self,
        out: &mut String,
        pad: &str,
        fused_ops: &[Op],
        block: &Block,
        kernel: &Kernel,
        type_env: &TypeEnv,
        extra_names: &BTreeMap<ValueId, String>,
        resolved_vid: ValueId,
        idx: usize,
    ) -> String {
        let op = &fused_ops[idx];

        match op {
            Op::BinOp { op: kind, lhs, rhs } => {
                let l = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *lhs,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                let r = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *rhs,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                match kind {
                    BinOpKind::Max
                    | BinOpKind::Min
                    | BinOpKind::Pow
                    | BinOpKind::ATan2
                    | BinOpKind::Rem => format!("{}({l}, {r})", kind.msl_symbol()),
                    BinOpKind::And => format!("({l} && {r})"),
                    BinOpKind::Or => format!("({l} || {r})"),
                    BinOpKind::Xor => format!("((bool){l} != (bool){r})"),
                    BinOpKind::BitAnd => format!("({l} & {r})"),
                    BinOpKind::BitOr => format!("({l} | {r})"),
                    BinOpKind::BitXor => format!("({l} ^ {r})"),
                    BinOpKind::Shl => format!("({l} << {r})"),
                    BinOpKind::Shr => format!("({l} >> {r})"),
                    _ => format!("({l} {} {r})", kind.msl_symbol()),
                }
            },
            Op::UnaryOp { op: kind, value } => {
                let v = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *value,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                kind.msl_emit(&v)
            },
            Op::Activation { kind, value } => {
                let v = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *value,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                format!("{}({v})", kind.msl_fn())
            },
            Op::Cast { dtype, value } => {
                let v = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *value,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                // Inside a fused chain the operand may be a sub-op without a
                // type_env entry. Walk the fused chain to infer the source
                // dtype — falls back to type_env lookup for cross-op operands.
                let src_dtype = infer_fused_op_dtype(*value, fused_ops, type_env);
                self.emit_cast_expr_with_src(*dtype, src_dtype, &v)
            },
            Op::Select { cond, on_true, on_false } => {
                let c = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *cond,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                let t = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *on_true,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                let f = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *on_false,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                format!("select({f}, {t}, bool({c}))")
            },
            Op::Broadcast { .. } => {
                // Broadcast in a fused chain is unusual; fall back.
                "0 /* broadcast-in-fused */".into()
            },
            _ => {
                let vid = super::helpers::op_to_vid(op);
                self.vname(Some(vid), block, extra_names)
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_operand(
        &self,
        out: &mut String,
        pad: &str,
        fused_ops: &[Op],
        vid: ValueId,
        block: &Block,
        kernel: &Kernel,
        type_env: &TypeEnv,
        extra_names: &BTreeMap<ValueId, String>,
        resolved_vid: ValueId,
    ) -> String {
        if is_sub_op_ref(vid.as_u32()) {
            let sub_idx = (vid.as_u32() & !SUB_OP_FLAG) as usize;
            if sub_idx < fused_ops.len() {
                self.emit_fused_expr(
                    out,
                    pad,
                    fused_ops,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                    sub_idx,
                )
            } else {
                "0 /* bad sub-op ref */".into()
            }
        } else {
            // Plain value — an ordinary SSA name or a loop-var ValueId
            // (`LOOP_VAR_FLAG | VarId`); `vname` resolves both via
            // `extra_names` (the loop body's `inner_names`).
            self.vname(Some(vid), block, extra_names)
        }
    }
}

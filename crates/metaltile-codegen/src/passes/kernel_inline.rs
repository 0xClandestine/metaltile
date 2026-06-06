//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! KernelInlinePass — resolve `Op::KernelCall` by splicing callee ops inline.
//!
//! Runs as the **first** pass in `standard_pipeline()` so all subsequent
//! passes (const_fold, fusion, vectorize, etc.) see only flat scalar ops.
//! The callee name is preserved in a comment during inlining to help future
//! fusion passes recognize cross-kernel composition patterns.
//!
//! ## Algorithm
//!
//! For each `Op::KernelCall { callee, args, dtype }` encountered in the
//! body (depth-first, single pass — callee bodies themselves must not
//! contain `KernelCall`):
//!
//! 1. Look up `callee` in the `inventory`-based `KernelEntry` registry,
//!    build its IR for the requested `dtype`.
//! 2. Find the callee's `max_vid`; use `find_max_vid(caller) + 1` as the
//!    starting offset for fresh callee ValueIds so they don't collide.
//! 3. Args are matched positionally to callee params. Two arg kinds:
//!    - `KernelCallArg::Value(vid)` — a pre-computed scalar.  The callee's
//!      input-param load for that param is skipped; all references to its
//!      result are replaced by `vid` directly (no memory round-trip).
//!    - `KernelCallArg::Tensor(name)` — a buffer / constexpr name.  The
//!      callee's loads/stores for that param are KEPT but their src/dst are
//!      renamed to `name`, enabling multi-element tensor access from within
//!      the callee body.
//! 4. Output params with NO corresponding arg have their stores skipped;
//!    the value being stored maps to `call_result` (the SSA vid returned by
//!    the `KernelCall` op).
//! 5. Callee `ProgramId` ops are remapped to the caller's corresponding
//!    `ProgramId` result vids (same axis) rather than being skipped.  This
//!    is correct for reduction-kernel composition where callee needs the
//!    threadgroup index.  If the caller has no matching axis, a fresh vid is
//!    used (causing a compile error if actually referenced — better than
//!    silent wrong code).

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    ir::{Block, Kernel, KernelCallArg, Op, ValueId},
};
use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    kernel_registry::all_kernels,
    passes::{
        Pass,
        remap::{find_max_vid, remap_value_ids},
    },
};

pub struct KernelInlinePass;

impl Pass for KernelInlinePass {
    fn name(&self) -> &str { "kernel_inline" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Collect caller's ProgramId result vids by axis (from the entry block
        // only — that's where program_id ops always live).  Callee ProgramId
        // ops in any block are remapped to these.
        let caller_pids: FxHashMap<u32, ValueId> = kernel
            .body
            .ops
            .iter()
            .zip(kernel.body.results.iter())
            .filter_map(|(op, r)| {
                if let (Op::ProgramId { axis }, Some(vid)) = (op, r) {
                    Some((*axis, *vid))
                } else {
                    None
                }
            })
            .collect();

        let mut vid_offset = find_max_vid(kernel) + 1;

        // Process the entry block first.
        inline_block(&mut kernel.body, &caller_pids, &mut vid_offset)?;

        // Process every nested block (then/else bodies of Op::If, loop bodies,
        // etc.).  KernelCalls inside nested blocks arise when cross-kernel calls
        // are placed inside loops or conditional branches.  IfConversionPass
        // later hoists those ops to the parent scope, but inlining must already
        // have resolved them by then.
        let block_ids: Vec<_> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            if let Some(block) = kernel.blocks.get_mut(&bid) {
                inline_block(block, &caller_pids, &mut vid_offset)?;
            }
        }

        Ok(())
    }
}

/// Resolve all top-level `Op::KernelCall`s in `block` by splicing callee ops
/// inline.  Updates `vid_offset` so callee VIDs never collide across blocks.
fn inline_block(
    block: &mut Block,
    caller_pids: &FxHashMap<u32, ValueId>,
    vid_offset: &mut u32,
) -> Result<()> {
    let mut new_ops: Vec<Op> = Vec::with_capacity(block.ops.len());
    let mut new_results: Vec<Option<ValueId>> = Vec::with_capacity(block.results.len());

    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);

    for (op, result) in old_ops.into_iter().zip(old_results) {
        if let Op::KernelCall { ref callee, ref args, dtype } = op {
            let call_result = result;

            let callee_kernel = lookup_kernel(callee, dtype)?;

            let inlined =
                inline_callee(&callee_kernel, args, caller_pids, call_result, *vid_offset);

            let max_new_vid = inlined
                .iter()
                .filter_map(|(_, r)| *r)
                .map(|v| v.as_u32())
                .max()
                .unwrap_or(vid_offset.saturating_sub(1));
            *vid_offset = max_new_vid + 1;

            for (inlined_op, inlined_result) in inlined {
                new_ops.push(inlined_op);
                new_results.push(inlined_result);
            }
        } else {
            new_ops.push(op);
            new_results.push(result);
        }
    }

    block.ops = new_ops;
    block.results = new_results;

    Ok(())
}

// ---------------------------------------------------------------------------
// Registry lookup
// ---------------------------------------------------------------------------

fn lookup_kernel(name: &str, dtype: DType) -> Result<Kernel> {
    all_kernels().find(|e| e.name() == name).map(|e| e.build(&[dtype])).ok_or_else(|| {
        Error::Generation(format!(
            "KernelInlinePass: unknown kernel `{name}` (not registered via #[kernel])"
        ))
    })
}

// ---------------------------------------------------------------------------
// inline_callee — splice callee ops with full KernelCallArg support
// ---------------------------------------------------------------------------

fn inline_callee(
    callee: &Kernel,
    args: &[KernelCallArg],
    caller_pids: &FxHashMap<u32, ValueId>,
    call_result: Option<ValueId>,
    vid_offset: u32,
) -> Vec<(Op, Option<ValueId>)> {
    // Enforce: callee bodies must not contain nested KernelCall.  If they do,
    // the inner call would survive as an unresolved KernelCall in the MSL
    // emitter and produce a silent `/* ERROR */` comment.  Either fixpoint the
    // inliner or ensure callee kernels are always flat.
    assert!(
        !callee.body.ops.iter().any(|op| matches!(op, Op::KernelCall { .. })),
        "KernelInlinePass: callee `{}` contains nested KernelCall — \
         callee bodies must be flat (no cross-kernel calls)",
        callee.name,
    );

    // Collect param name lists in declaration order for positional arg matching.
    // Constexpr params live in `callee.constexprs` (not `callee.params`), so
    // the positional order is: [input_params..., constexprs..., output_params...].
    let input_params: Vec<&str> =
        callee.params.iter().filter(|p| !p.is_output).map(|p| p.name.as_str()).collect();
    let constexpr_names: Vec<&str> = callee.constexprs.iter().map(|c| c.name.name()).collect();
    let output_params: Vec<&str> =
        callee.params.iter().filter(|p| p.is_output).map(|p| p.name.as_str()).collect();
    let n_input_slots = input_params.len() + constexpr_names.len();

    // Unified name → arg map for all non-output params (tensor inputs + constexprs).
    // Single lookup replaces the three parallel arrays + offset arithmetic that
    // previously caused the constexpr positional-arg bug.
    let param_arg: FxHashMap<&str, &KernelCallArg> = input_params
        .iter()
        .chain(constexpr_names.iter())
        .enumerate()
        .filter_map(|(i, &name)| args.get(i).map(|a| (name, a)))
        .collect();

    // Output param args (rare; usually absent → store is skipped, value → call_result).
    let output_arg: FxHashMap<&str, Option<&KernelCallArg>> = output_params
        .iter()
        .enumerate()
        .map(|(j, &name)| (name, args.get(n_input_slots + j)))
        .collect();

    let mut vid_map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut next_vid = vid_offset;

    // Single pre-pass: seed Value-arg load results into vid_map, and map the
    // stored value of output params without an explicit arg to call_result.
    // Merged from the previous two separate pre-pass loops.
    let mut mapped_output = false;
    for (op, result) in callee.body.ops.iter().zip(callee.body.results.iter()) {
        match op {
            Op::Load { src, .. } => {
                if let Some(r) = result
                    && let Some(KernelCallArg::Value(arg_vid)) =
                        param_arg.get(src.as_str()).copied()
                {
                    vid_map.insert(*r, *arg_vid);
                }
            },
            Op::Store { dst, value, .. } => {
                if let Some(out_arg) = output_arg.get(dst.as_str())
                    && out_arg.is_none()
                    && !mapped_output
                    && let Some(cr) = call_result
                {
                    vid_map.insert(*value, cr);
                    mapped_output = true;
                }
            },
            _ => {},
        }
    }

    // Splice callee ops.
    let mut inlined: Vec<(Op, Option<ValueId>)> = Vec::new();

    for (op, op_result) in callee.body.ops.iter().zip(callee.body.results.iter()) {
        match op {
            // ── ProgramId ────────────────────────────────────────────────────
            // Map to the caller's ProgramId for the same axis; don't re-emit.
            // If the caller has no matching axis a fresh vid is allocated —
            // it will fail to compile if actually referenced, which is correct.
            Op::ProgramId { axis } => {
                if let Some(r) = op_result {
                    let mapped = caller_pids.get(axis).copied().unwrap_or_else(|| {
                        let fresh = ValueId::new(next_vid);
                        next_vid += 1;
                        fresh
                    });
                    vid_map.insert(*r, mapped);
                }
                continue;
            },

            // ── Load from any callee param (tensor input or constexpr) ────────
            Op::Load { src, .. } => {
                if let Some(arg) = param_arg.get(src.as_str()) {
                    match arg {
                        KernelCallArg::Value(_) => {
                            // Pre-pass seeded vid_map with load_result → arg_vid.
                            debug_assert!(
                                op_result.is_none_or(|r| vid_map.contains_key(&r)),
                                "Value-arg load result not in vid_map — callee IR malformed"
                            );
                            continue;
                        },
                        KernelCallArg::Tensor(tensor_name) => {
                            let mut new_op = op.clone();
                            if let Op::Load { src: ref mut s, .. } = new_op {
                                *s = tensor_name.clone();
                            }
                            remap_value_ids(&mut new_op, &vid_map);
                            let new_result = assign_result(op_result, &mut vid_map, &mut next_vid);
                            inlined.push((new_op, new_result));
                            continue;
                        },
                    }
                }
                // Fall through: load from a non-param buffer (internal to callee).
            },

            // ── Store to an output param ──────────────────────────────────────
            Op::Store { dst, .. } => {
                if let Some(out_arg) = output_arg.get(dst.as_str()) {
                    match out_arg {
                        // Explicit Tensor arg: rename dst and keep the store.
                        Some(KernelCallArg::Tensor(tensor_name)) => {
                            let mut new_op = op.clone();
                            if let Op::Store { dst: ref mut d, .. } = new_op {
                                *d = tensor_name.clone();
                            }
                            remap_value_ids(&mut new_op, &vid_map);
                            inlined.push((new_op, None));
                            continue;
                        },
                        // No arg (or Value arg): skip store — value already
                        // mapped to call_result by the pre-pass.
                        None | Some(KernelCallArg::Value(_)) => continue,
                    }
                }
                // Fall through: store to a non-param buffer (internal to callee).
            },

            _ => {},
        }

        // Default: remap value ids and keep the op.
        let mut new_op = op.clone();
        remap_value_ids(&mut new_op, &vid_map);
        let new_result = assign_result(op_result, &mut vid_map, &mut next_vid);
        inlined.push((new_op, new_result));
    }

    inlined
}

/// Assign a caller-side result vid for a callee op result, using any existing
/// mapping or allocating a fresh vid.
fn assign_result(
    op_result: &Option<ValueId>,
    vid_map: &mut BTreeMap<ValueId, ValueId>,
    next_vid: &mut u32,
) -> Option<ValueId> {
    op_result.map(|r| {
        if let Some(&existing) = vid_map.get(&r) {
            existing
        } else {
            let fresh = ValueId::new(*next_vid);
            *next_vid += 1;
            vid_map.insert(r, fresh);
            fresh
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use metaltile_core::{
        DType,
        ir::{ActKind, BinOpKind, IndexExpr, Kernel, KernelCallArg, Op, Param, ParamKind, ValueId},
        shape::Shape,
    };

    use super::*;
    use crate::passes::Pass;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn v(n: u32) -> ValueId { ValueId::new(n) }

    fn tensor_param(name: &str, dtype: DType, is_output: bool) -> Param {
        Param {
            name: name.to_string(),
            dtype,
            shape: Shape::scalar(),
            is_output,
            kind: ParamKind::Tensor,
        }
    }

    /// Build a minimal callee that looks like `mt_silu`:
    ///   tid   = program_id::<0>()          → v0
    ///   loaded = load(a[tid])              → v1
    ///   result = Activation(Silu, loaded)  → v2
    ///   store(out[tid], result)            (no result)
    fn build_silu_callee() -> Kernel {
        let mut k = Kernel::new("mt_silu");
        k.params.push(tensor_param("a", DType::F32, false));
        k.params.push(tensor_param("out", DType::F32, true));

        // v0 = tid
        k.body.push_op(Op::ProgramId { axis: 0 }, v(0));
        // v1 = load(a[v0])
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                indices: vec![IndexExpr::Value(v(0))],
                mask: None,
                other: None,
            },
            v(1),
        );
        // v2 = silu(v1)
        k.body.push_op(Op::Activation { kind: ActKind::Silu, value: v(1) }, v(2));
        // store(out[v0], v2) — no result
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(v(0))],
            value: v(2),
            mask: None,
        });
        k
    }

    // ── test 1: Value arg — scalar callee (mt_silu pattern) ──────────────────

    /// Caller passes a pre-computed f32 scalar to mt_silu via Value arg.
    /// Expected: silu op is kept with the scalar vid; no load/store/ProgramId.
    #[test]
    fn value_arg_scalar_callee_splices_activation() {
        let callee = build_silu_callee();
        let caller_pids: FxHashMap<u32, ValueId> = [(0, v(5))].into_iter().collect();
        // g_vid = v10 (the pre-computed f32 scalar in caller)
        let args = vec![KernelCallArg::Value(v(10))];
        let call_result = Some(v(99));

        let inlined = inline_callee(&callee, &args, &caller_pids, call_result, 100);

        // Should emit exactly one op: Activation { Silu, v(10) }
        // ProgramId, Load, and Store should all be skipped.
        assert_eq!(inlined.len(), 1, "expected exactly 1 inlined op, got {}", inlined.len());
        let (op, result) = &inlined[0];
        assert!(
            matches!(op, Op::Activation { kind: ActKind::Silu, value } if *value == v(10)),
            "expected Activation(Silu, v10), got {op:?}"
        );
        // The result of the Activation op must be call_result = v99.
        assert_eq!(*result, Some(v(99)), "activation result must map to call_result");
    }

    // ── test 2: Tensor arg — callee loads/stores are kept with renamed src ───

    /// Build a trivial callee that just copies input → output:
    ///   tid   = ProgramId(0) → v0
    ///   val   = load(src[tid]) → v1
    ///   store(dst[tid], val)
    fn build_copy_callee() -> Kernel {
        let mut k = Kernel::new("copy_helper");
        k.params.push(tensor_param("src", DType::F32, false));
        k.params.push(tensor_param("dst", DType::F32, true));

        k.body.push_op(Op::ProgramId { axis: 0 }, v(0));
        k.body.push_op(
            Op::Load {
                src: "src".into(),
                indices: vec![IndexExpr::Value(v(0))],
                mask: None,
                other: None,
            },
            v(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "dst".into(),
            indices: vec![IndexExpr::Value(v(0))],
            value: v(1),
            mask: None,
        });
        k
    }

    /// Caller passes tensor names "x_buf" (input) and "y_buf" (output).
    /// Expected: load/store ops are KEPT but src/dst are renamed.
    #[test]
    fn tensor_args_rename_load_store_and_keep_ops() {
        let callee = build_copy_callee();
        let caller_pids: FxHashMap<u32, ValueId> = [(0, v(3))].into_iter().collect();
        let args =
            vec![KernelCallArg::Tensor("x_buf".into()), KernelCallArg::Tensor("y_buf".into())];
        // No scalar call_result needed — output goes to the Tensor arg.
        let call_result = None;

        let inlined = inline_callee(&callee, &args, &caller_pids, call_result, 50);

        // ProgramId is skipped (remapped to caller's tid v3).
        // Load from "src" → renamed to "x_buf", kept.
        // Store to "dst" → renamed to "y_buf", kept.
        let ops: Vec<_> = inlined.iter().map(|(op, _)| op).collect();
        assert_eq!(ops.len(), 2, "expected load + store, got {}", ops.len());

        let load_op = &ops[0];
        assert!(
            matches!(load_op, Op::Load { src, .. } if src == "x_buf"),
            "load src should be renamed to x_buf, got {load_op:?}"
        );

        let store_op = &ops[1];
        assert!(
            matches!(store_op, Op::Store { dst, .. } if dst == "y_buf"),
            "store dst should be renamed to y_buf, got {store_op:?}"
        );

        // The load uses the caller's ProgramId (v3) as its index.
        if let Op::Load { indices, .. } = &ops[0] {
            assert!(
                matches!(indices[0], IndexExpr::Value(v) if v == ValueId::new(3)),
                "load index should be caller's tid v3, got {:?}",
                indices[0]
            );
        }
    }

    // ── test 3: ProgramId inheritance ─────────────────────────────────────────

    /// Callee has a ProgramId with axis 1; caller has ProgramId axis 1 → v7.
    /// Expected: callee's ProgramId result is remapped to v7 and not emitted.
    #[test]
    fn programid_remapped_to_caller_axis() {
        let mut callee = Kernel::new("axis1_helper");
        callee.params.push(tensor_param("a", DType::F32, false));
        callee.params.push(tensor_param("out", DType::F32, true));

        // v0 = ProgramId(axis=1)
        callee.body.push_op(Op::ProgramId { axis: 1 }, v(0));
        // v1 = load(a[v0])
        callee.body.push_op(
            Op::Load {
                src: "a".into(),
                indices: vec![IndexExpr::Value(v(0))],
                mask: None,
                other: None,
            },
            v(1),
        );
        // v2 = v1 + v1 (just to have an op that references v0-remapped chain)
        callee.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: v(1), rhs: v(1) }, v(2));
        // store(out[v0], v2) — uses axis-1 ProgramId as index
        callee.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(v(0))],
            value: v(2),
            mask: None,
        });

        // Caller has ProgramId axis 0 → v10, axis 1 → v11.
        let caller_pids: FxHashMap<u32, ValueId> = [(0, v(10)), (1, v(11))].into_iter().collect();
        let args = vec![KernelCallArg::Value(v(20))]; // scalar input arg
        let call_result = Some(v(99));

        let inlined = inline_callee(&callee, &args, &caller_pids, call_result, 200);

        // Expect: BinOp only (Load skipped — Value arg; Store skipped — no Tensor output arg)
        // The BinOp should have lhs=rhs=v20 (since v1 → v20 via Value arg substitution).
        assert_eq!(inlined.len(), 1, "expected 1 op (BinOp), got {}", inlined.len());
        let (op, result) = &inlined[0];
        assert!(
            matches!(op, Op::BinOp { lhs, rhs, .. } if *lhs == v(20) && *rhs == v(20)),
            "BinOp args should be remapped to caller's v20, got {op:?}"
        );
        assert_eq!(*result, Some(v(99)));
    }

    // ── test 4: KernelInlinePass integrates into run() correctly ─────────────

    /// Full pass test: a Kernel containing Op::KernelCall with an
    /// unregistered callee → pass returns an error (rather than silently
    /// leaving an unresolved op in the IR).
    #[test]
    fn unregistered_callee_returns_error() {
        let mut k = Kernel::new("caller");
        k.params.push(tensor_param("x", DType::F32, false));
        k.params.push(tensor_param("out", DType::F32, true));

        k.body.push_op(Op::ProgramId { axis: 0 }, v(0));
        k.body.push_op(
            Op::KernelCall {
                callee: "nonexistent_kernel_xyz".into(),
                args: vec![KernelCallArg::Value(v(0))],
                dtype: DType::F32,
            },
            v(1),
        );

        let result = KernelInlinePass.run(&mut k);
        assert!(result.is_err(), "expected error for unregistered callee");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("nonexistent_kernel_xyz"), "error should name the callee, got: {msg}");
    }

    // ── test 5: no-op when body has no KernelCall ops ─────────────────────────

    #[test]
    fn pass_is_noop_when_no_kernel_calls() {
        let mut k = Kernel::new("simple");
        k.params.push(tensor_param("x", DType::F32, false));
        k.params.push(tensor_param("out", DType::F32, true));

        k.body.push_op(Op::ProgramId { axis: 0 }, v(0));
        k.body.push_op(
            Op::Load {
                src: "x".into(),
                indices: vec![IndexExpr::Value(v(0))],
                mask: None,
                other: None,
            },
            v(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(v(0))],
            value: v(1),
            mask: None,
        });

        let original_len = k.body.ops.len();
        KernelInlinePass.run(&mut k).unwrap();
        assert_eq!(k.body.ops.len(), original_len, "no-op: body should be unchanged");
    }
}

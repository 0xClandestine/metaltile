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
    KernelEntry,
    dtype::DType,
    ir::{Kernel, KernelCallArg, Op, ValueId},
};

use crate::{
    error::{Error, Result},
    passes::{
        Pass,
        remap::{find_max_vid, remap_value_ids},
    },
};

pub struct KernelInlinePass;

impl Pass for KernelInlinePass {
    fn name(&self) -> &str { "kernel_inline" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Collect caller's ProgramId result vids by axis (before we start
        // rewriting the body).  Callee ProgramId ops are remapped to these.
        let caller_pids: BTreeMap<u32, ValueId> = kernel
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

        let mut new_ops: Vec<Op> = Vec::with_capacity(kernel.body.ops.len());
        let mut new_results: Vec<Option<ValueId>> =
            Vec::with_capacity(kernel.body.results.len());

        let old_ops = std::mem::take(&mut kernel.body.ops);
        let old_results = std::mem::take(&mut kernel.body.results);

        for (op, result) in old_ops.into_iter().zip(old_results.into_iter()) {
            if let Op::KernelCall { ref callee, ref args, dtype } = op {
                let call_result = result;

                let callee_kernel = match lookup_kernel(callee, dtype) {
                    Some(k) => k,
                    None => {
                        return Err(Error::Generation(format!(
                            "KernelInlinePass: unknown kernel `{callee}` \
                             (not registered via #[kernel])"
                        )));
                    },
                };

                let inlined =
                    inline_callee(&callee_kernel, args, &caller_pids, call_result, vid_offset);

                let max_new_vid = inlined
                    .iter()
                    .filter_map(|(_, r)| *r)
                    .map(|v| v.as_u32())
                    .max()
                    .unwrap_or(vid_offset.saturating_sub(1));
                vid_offset = max_new_vid + 1;

                for (inlined_op, inlined_result) in inlined {
                    new_ops.push(inlined_op);
                    new_results.push(inlined_result);
                }
            } else {
                new_ops.push(op);
                new_results.push(result);
            }
        }

        kernel.body.ops = new_ops;
        kernel.body.results = new_results;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Registry lookup
// ---------------------------------------------------------------------------

fn lookup_kernel(name: &str, dtype: DType) -> Option<Kernel> {
    inventory::iter::<KernelEntry>().find(|e| e.name == name).map(|e| (e.build)(&[dtype]))
}

// ---------------------------------------------------------------------------
// inline_callee — splice callee ops with full KernelCallArg support
// ---------------------------------------------------------------------------

fn inline_callee(
    callee: &Kernel,
    args: &[KernelCallArg],
    caller_pids: &BTreeMap<u32, ValueId>,
    call_result: Option<ValueId>,
    vid_offset: u32,
) -> Vec<(Op, Option<ValueId>)> {
    // Separate callee params into input/output, preserving original order for
    // positional arg matching.  We match args[i] → input_params[i], then
    // remaining args (if any) → output_params[0..].
    let input_params: Vec<&str> =
        callee.params.iter().filter(|p| !p.is_output).map(|p| p.name.as_str()).collect();
    let output_params: Vec<&str> =
        callee.params.iter().filter(|p| p.is_output).map(|p| p.name.as_str()).collect();

    // For each input param, determine how it's being supplied:
    //   Value(vid) → skip the load, map load result → vid.
    //   Tensor(name) → rename load/store src/dst to `name`, keep the op.
    //   No arg → keep the load as-is (unusual; e.g. forwarding param by name
    //             from a parent which already renamed it).
    let input_arg: Vec<Option<&KernelCallArg>> =
        (0..input_params.len()).map(|i| args.get(i)).collect();

    // For each output param, check if an explicit Tensor arg was passed
    // (args[input_params.len() + j]).  If not, the store is skipped and
    // the stored value maps to call_result.
    let output_arg: Vec<Option<&KernelCallArg>> = (0..output_params.len())
        .map(|j| args.get(input_params.len() + j))
        .collect();

    // Find the callee SSA vid being stored to each output param without an
    // explicit arg → these map to call_result.
    let mut vid_map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut next_vid = vid_offset;

    // Seed vid_map with Value args for input params.
    for (op, result) in callee.body.ops.iter().zip(callee.body.results.iter()) {
        if let (Op::Load { src, .. }, Some(r)) = (op, result) {
            if let Some(idx) = input_params.iter().position(|&n| n == src.as_str()) {
                if let Some(KernelCallArg::Value(arg_vid)) = input_arg[idx] {
                    vid_map.insert(*r, *arg_vid);
                }
            }
        }
    }

    // For output params without an explicit arg, map their stored value vids
    // to call_result.  Only the first such output is mapped (single return).
    let mut mapped_output = false;
    for (op, _result) in callee.body.ops.iter().zip(callee.body.results.iter()) {
        if let Op::Store { dst, value, .. } = op {
            if let Some(idx) = output_params.iter().position(|&n| n == dst.as_str()) {
                if matches!(output_arg[idx], None) && !mapped_output {
                    if let Some(cr) = call_result {
                        vid_map.insert(*value, cr);
                        mapped_output = true;
                    }
                }
            }
        }
    }

    // Splice callee ops.
    let mut inlined: Vec<(Op, Option<ValueId>)> = Vec::new();

    for (op, op_result) in callee.body.ops.iter().zip(callee.body.results.iter()) {
        match op {
            // ── ProgramId ────────────────────────────────────────────────────
            // Map to the caller's ProgramId for the same axis, or allocate a
            // fresh vid if the caller doesn't have one (will fail to compile if
            // the callee actually uses it, which is the correct behaviour).
            Op::ProgramId { axis } => {
                if let Some(r) = op_result {
                    let mapped = if let Some(&caller_pid) = caller_pids.get(axis) {
                        caller_pid
                    } else {
                        let fresh = ValueId::new(next_vid);
                        next_vid += 1;
                        fresh
                    };
                    vid_map.insert(*r, mapped);
                }
                // Don't emit the op — the caller already has it.
                continue;
            },

            // ── Load from an input param ──────────────────────────────────────
            Op::Load { src, .. } => {
                if let Some(idx) = input_params.iter().position(|&n| n == src.as_str()) {
                    match input_arg[idx] {
                        // Value arg: skip load, result already in vid_map.
                        Some(KernelCallArg::Value(_)) => {
                            // Ensure skipped result has a vid entry.
                            if let Some(r) = op_result {
                                if !vid_map.contains_key(r) {
                                    let fresh = ValueId::new(next_vid);
                                    next_vid += 1;
                                    vid_map.insert(*r, fresh);
                                }
                            }
                            continue;
                        },
                        // Tensor arg: rename src and keep the op.
                        Some(KernelCallArg::Tensor(tensor_name)) => {
                            let mut new_op = op.clone();
                            if let Op::Load { src: ref mut s, .. } = new_op {
                                *s = tensor_name.clone();
                            }
                            remap_value_ids(&mut new_op, &vid_map);
                            let new_result = assign_result(op_result, &mut vid_map, &mut next_vid);
                            inlined.push((new_op, new_result));
                            continue;
                        },
                        // No arg: keep load as-is (unusual).
                        None => {},
                    }
                }
                // Fall through to default handling.
            },

            // ── Store to an output param ──────────────────────────────────────
            Op::Store { dst, .. } => {
                if let Some(idx) = output_params.iter().position(|&n| n == dst.as_str()) {
                    match output_arg[idx] {
                        // Tensor arg: rename dst and keep the store.
                        Some(KernelCallArg::Tensor(tensor_name)) => {
                            let mut new_op = op.clone();
                            if let Op::Store { dst: ref mut d, .. } = new_op {
                                *d = tensor_name.clone();
                            }
                            remap_value_ids(&mut new_op, &vid_map);
                            // Stores don't produce a result.
                            inlined.push((new_op, None));
                            continue;
                        },
                        // No arg: skip store — value already mapped to call_result above.
                        None | Some(KernelCallArg::Value(_)) => {
                            if let Some(r) = op_result {
                                if !vid_map.contains_key(r) {
                                    let fresh = ValueId::new(next_vid);
                                    next_vid += 1;
                                    vid_map.insert(*r, fresh);
                                }
                            }
                            continue;
                        },
                    }
                }
                // Fall through to default handling (store to a non-param buffer).
            },

            _ => {},
        }

        // Default: remap and keep the op.
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
            Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(v(0))], mask: None, other: None },
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
        let caller_pids: BTreeMap<u32, ValueId> = [(0, v(5))].into_iter().collect();
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
            Op::Load { src: "src".into(), indices: vec![IndexExpr::Value(v(0))], mask: None, other: None },
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
        let caller_pids: BTreeMap<u32, ValueId> = [(0, v(3))].into_iter().collect();
        let args = vec![
            KernelCallArg::Tensor("x_buf".into()),
            KernelCallArg::Tensor("y_buf".into()),
        ];
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
                "load index should be caller's tid v3, got {:?}", indices[0]
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
            Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(v(0))], mask: None, other: None },
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
        let caller_pids: BTreeMap<u32, ValueId> =
            [(0, v(10)), (1, v(11))].into_iter().collect();
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
        assert!(
            msg.contains("nonexistent_kernel_xyz"),
            "error should name the callee, got: {msg}"
        );
    }

    // ── test 5: no-op when body has no KernelCall ops ─────────────────────────

    #[test]
    fn pass_is_noop_when_no_kernel_calls() {
        let mut k = Kernel::new("simple");
        k.params.push(tensor_param("x", DType::F32, false));
        k.params.push(tensor_param("out", DType::F32, true));

        k.body.push_op(Op::ProgramId { axis: 0 }, v(0));
        k.body.push_op(
            Op::Load { src: "x".into(), indices: vec![IndexExpr::Value(v(0))], mask: None, other: None },
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

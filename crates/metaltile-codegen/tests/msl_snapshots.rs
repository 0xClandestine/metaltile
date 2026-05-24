//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Golden MSL snapshots — the full MSL output for a small zoo of
//! hand-built kernels is pinned via `insta`, so any change to codegen
//! (op lowering, preamble emission, scheduling, vectorization) shows
//! up as a reviewable diff instead of having to be guessed at by
//! grepping for substrings.
//!
//! Refresh after intentional codegen changes:
//!   cargo insta test --accept --workspace
//!
//! Or interactively:
//!   cargo insta review
//!
//! The fixtures here aim to exercise distinct emit paths rather than to
//! be exhaustive (per-kernel benches in `metaltile-cli`/`metaltile-std`
//! cover the real production kernels). Add a fixture when a new emit
//! path lands that the existing snapshots don't cover.

use insta::assert_snapshot;
use metaltile_codegen::{MslGenerator, msl::MslConfig};
use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, IndexExpr, Kernel, KernelMode, Op, Param, ReduceKind, ValueId},
    shape::Shape,
};

// ── Kernel builders ──────────────────────────────────────────────────────────

/// Three-buffer 1-D elementwise add: `c[idx] = a[idx] + b[idx]`. The
/// minimal kernel that covers ProgramId / Load / BinOp / Store and the
/// scalar `tid` mapping.
fn vadd(dtype: DType) -> Kernel {
    let mut k = Kernel::new("vector_add");
    for (name, is_output) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype,
            shape: Shape::scalar(),
            is_output,
            kind: Default::default(),
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.name_value(ValueId::new(0), "idx");
    k.body.push_op(
        Op::Load {
            src: "a".into(),
            mask: None,
            other: None,
            indices: vec![IndexExpr::Value(ValueId::new(0))],
        },
        ValueId::new(1),
    );
    k.body.name_value(ValueId::new(1), "x");
    k.body.push_op(
        Op::Load {
            src: "b".into(),
            mask: None,
            other: None,
            indices: vec![IndexExpr::Value(ValueId::new(0))],
        },
        ValueId::new(2),
    );
    k.body.name_value(ValueId::new(2), "y");
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.name_value(ValueId::new(3), "sum");
    k.body.push_op_no_result(Op::Store {
        mask: None,
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(3),
    });
    k
}

/// Cast chain that touches every dtype's lowering: f32 → f16 → bf16
/// → f32. Useful for catching regressions in `static_cast`, the bf16
/// compat constructor, and the dtype-name table at once.
fn cast_chain_f32_f16_bf16() -> Kernel {
    let mut k = Kernel::new("cast_chain");
    k.params.push(Param {
        name: "x".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: false,
        kind: Default::default(),
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: true,
        kind: Default::default(),
    });
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load {
            src: "x".into(),
            mask: None,
            other: None,
            indices: vec![IndexExpr::Value(ValueId::new(0))],
        },
        ValueId::new(1),
    );
    // f32 -> f16
    k.body.push_op(Op::Cast { value: ValueId::new(1), dtype: DType::F16 }, ValueId::new(2));
    // f16 -> bf16
    k.body.push_op(Op::Cast { value: ValueId::new(2), dtype: DType::BF16 }, ValueId::new(3));
    // bf16 -> f32
    k.body.push_op(Op::Cast { value: ValueId::new(3), dtype: DType::F32 }, ValueId::new(4));
    k.body.push_op_no_result(Op::Store {
        mask: None,
        dst: "out".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(4),
    });
    k
}

/// Bare kernel with a single bf16-typed parameter. Exercises just the
/// preamble / buffer-decl path, no body — used to pin the bf16 compat
/// vs native emission decision.
fn bf16_param_only() -> Kernel {
    let mut k = Kernel::new("bf16_param");
    k.params.push(Param {
        name: "a".into(),
        dtype: DType::BF16,
        shape: Shape::scalar(),
        is_output: false,
        kind: Default::default(),
    });
    k
}

// ── Snapshots ────────────────────────────────────────────────────────────────

#[test]
fn vadd_f32_default_config() {
    let msl = MslGenerator::default().generate(&vadd(DType::F32)).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn vadd_f16_default_config() {
    let msl = MslGenerator::default().generate(&vadd(DType::F16)).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn vadd_bf16_native() {
    let cfg = MslConfig { native_bfloat: true, ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&vadd(DType::BF16)).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn vadd_bf16_compat_preamble() {
    // native_bfloat: false ⇒ emitter falls back to the `struct bfloat16_t`
    // compatibility preamble for pre-Metal-3.1 toolchains.
    let cfg = MslConfig { native_bfloat: false, ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&vadd(DType::BF16)).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn cast_chain_default_config() {
    let msl = MslGenerator::default().generate(&cast_chain_f32_f16_bf16()).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn cast_chain_bf16_compat() {
    let cfg = MslConfig { native_bfloat: false, ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&cast_chain_f32_f16_bf16()).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn bf16_param_only_native_emits_native_type() {
    let cfg = MslConfig { native_bfloat: true, ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&bf16_param_only()).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn bf16_param_only_compat_emits_preamble() {
    let cfg = MslConfig { native_bfloat: false, ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&bf16_param_only()).unwrap();
    assert_snapshot!(msl);
}

// ── Regression tests: codegen bug fixes from the SDPA prefill PR ────────────

/// Kernel that computes a uint index `simd_lane * 128 + tgid_x` and stores
/// `1.0` at that offset. The emitted MSL must NOT contain `fma(...)`: Metal's
/// `fma` overload set is float-only, and the prior fma-fusion check looked
/// only at result type — for an index chain whose operands are uint, the
/// integer mul+add must lower to `*` and `+`, not `fma`.
///
/// Regression for: codegen bug "fma fusion fires on integer operands"
/// (fixed in this PR, see `emit_block.rs::BinOp::Add` arm).
fn uint_index_chain_no_fma() -> Kernel {
    let mut k = Kernel::new("uint_index_chain");
    k.params.push(Param {
        name: "out".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: true,
        kind: Default::default(),
    });
    // tgid_x (uint builtin) — typed via the type_check builtin-uint table.
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.name_value(ValueId::new(0), "tg");
    k.body.push_op(
        Op::Load { src: "simd_lane".into(), mask: None, other: None, indices: vec![] },
        ValueId::new(1),
    );
    k.body.name_value(ValueId::new(1), "lane");
    k.body.push_op(Op::Const { value: 128 }, ValueId::new(2));
    // idx = lane * 128 + tg  — mul then add, both uint.
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(3), rhs: ValueId::new(0) },
        ValueId::new(4),
    );
    k.body.name_value(ValueId::new(4), "idx");
    k.body.push_op(Op::Const { value: 1 }, ValueId::new(5));
    k.body.push_op(Op::Cast { value: ValueId::new(5), dtype: DType::F32 }, ValueId::new(6));
    k.body.push_op_no_result(Op::Store {
        mask: None,
        dst: "out".into(),
        indices: vec![IndexExpr::Value(ValueId::new(4))],
        value: ValueId::new(6),
    });
    k
}

#[test]
fn uint_index_chain_does_not_emit_fma() {
    let msl = MslGenerator::default().generate(&uint_index_chain_no_fma()).unwrap();
    assert!(
        !msl.contains("fma("),
        "uint index chain must not lower to fma(); Metal has no int fma overload.\n{msl}"
    );
}

/// Kernel that produces `simd_shuffle_xor(value, 8)` against a loaded f32.
/// Asserts the MSL emits the exact intrinsic call. Regression for the new
/// `Op::SimdShuffleXor` DSL primitive added in this PR.
fn simd_shuffle_xor_kernel() -> Kernel {
    let mut k = Kernel::new("simd_shuffle_xor_smoke");
    k.params.push(Param {
        name: "a".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: false,
        kind: Default::default(),
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: true,
        kind: Default::default(),
    });
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load {
            src: "a".into(),
            mask: None,
            other: None,
            indices: vec![IndexExpr::Value(ValueId::new(0))],
        },
        ValueId::new(1),
    );
    k.body.push_op(Op::SimdShuffleXor { value: ValueId::new(1), mask: 8 }, ValueId::new(2));
    k.body.push_op_no_result(Op::Store {
        mask: None,
        dst: "out".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(2),
    });
    k
}

#[test]
fn simd_shuffle_xor_lowers_to_intrinsic() {
    let msl = MslGenerator::default().generate(&simd_shuffle_xor_kernel()).unwrap();
    assert!(
        msl.contains("simd_shuffle_xor("),
        "Op::SimdShuffleXor must lower to `simd_shuffle_xor(v, mask)` in MSL.\n{msl}"
    );
    assert!(
        msl.contains("simd_shuffle_xor(v1, 8"),
        "shuffle mask 8 must appear as the second arg to simd_shuffle_xor.\n{msl}"
    );
}

// ── Reduction-mode single-simdgroup specialization ─────────────────────────

/// Minimal Reduction-mode kernel that loads one element and reduces it.
/// Exercises `emit_reduce`'s threadgroup-scope path so we can pin the
/// per-`expected_tpg` specialization (fast / slow / default-safe).
fn reduction_kernel(kind: ReduceKind) -> Kernel {
    let mut k = Kernel::new("reduction_smoke");
    k.mode = KernelMode::Reduction;
    k.params.push(Param {
        name: "x".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: false,
        kind: Default::default(),
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: true,
        kind: Default::default(),
    });
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load {
            src: "x".into(),
            mask: None,
            other: None,
            indices: vec![IndexExpr::Value(ValueId::new(0))],
        },
        ValueId::new(1),
    );
    k.body.push_op(Op::Reduce { value: ValueId::new(1), axis: 0, op: kind }, ValueId::new(2));
    k.body.push_op_no_result(Op::Store {
        mask: None,
        dst: "out".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(2),
    });
    k
}

/// `expected_tpg = Some(32)` (or any value ≤ simd_size) emits only the
/// fast path: a single `simd_*(value)` call, no threadgroup buffer, no
/// barriers, no runtime `lsize` branch left behind.
#[test]
fn reduction_with_small_tpg_emits_only_fast_path() {
    let cfg = MslConfig { expected_tpg: Some(32), ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&reduction_kernel(ReduceKind::Sum)).unwrap();
    assert!(msl.contains("simd_sum(float(v1))"), "fast path must call simd_sum directly:\n{msl}");
    assert!(
        !msl.contains("threadgroup float"),
        "fast path must not declare the 32-slot tg buffer:\n{msl}"
    );
    assert!(
        !msl.contains("threadgroup_barrier"),
        "fast path must not emit any threadgroup barriers:\n{msl}"
    );
    assert!(
        !msl.contains("if (lsize"),
        "compile-time spec must not leave a runtime branch behind:\n{msl}"
    );
}

/// `expected_tpg = Some(256)` (above simd_size) emits only the slow path:
/// full two-level threadgroup reduction with both barriers.
#[test]
fn reduction_with_large_tpg_emits_only_slow_path() {
    let cfg = MslConfig { expected_tpg: Some(256), ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&reduction_kernel(ReduceKind::Sum)).unwrap();
    assert!(
        msl.contains("threadgroup float v2_sg[32]"),
        "slow path must declare the 32-slot tg buffer:\n{msl}"
    );
    assert_eq!(
        msl.matches("threadgroup_barrier(mem_flags::mem_threadgroup)").count(),
        2,
        "slow path emits exactly two barriers (post-write, post-broadcast):\n{msl}"
    );
    assert!(
        !msl.contains("if (lsize"),
        "compile-time spec must not leave a runtime branch behind:\n{msl}"
    );
}

/// `expected_tpg = None` (the default) falls back to the slow path — the
/// only choice that's correct at any TPG without runtime information.
#[test]
fn reduction_with_unknown_tpg_defaults_to_slow_path() {
    let msl = MslGenerator::default().generate(&reduction_kernel(ReduceKind::Sum)).unwrap();
    assert!(msl.contains("threadgroup float v2_sg[32]"), "default must emit slow path:\n{msl}");
    assert_eq!(
        msl.matches("threadgroup_barrier(mem_flags::mem_threadgroup)").count(),
        2,
        "default must keep both slow-path barriers:\n{msl}"
    );
}

/// Max in the fast path uses `simd_max` with no slow-path padding.
#[test]
fn reduction_max_small_tpg_emits_simd_max_only() {
    let cfg = MslConfig { expected_tpg: Some(32), ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&reduction_kernel(ReduceKind::Max)).unwrap();
    assert!(msl.contains("simd_max(float(v1))"), "fast-path max must call simd_max:\n{msl}");
    assert!(
        !msl.contains("-INFINITY"),
        "fast path needs no `-INFINITY` lane-pad (no slow-path code emitted):\n{msl}"
    );
}

/// Mean divides by `lsize` in whichever path is emitted.
#[test]
fn reduction_mean_divides_by_lsize_in_both_specializations() {
    let small = MslGenerator::new(MslConfig { expected_tpg: Some(32), ..MslConfig::default() })
        .generate(&reduction_kernel(ReduceKind::Mean))
        .unwrap();
    assert!(small.contains("/ float(lsize)"), "fast-path mean must divide by lsize:\n{small}");

    let large = MslGenerator::new(MslConfig { expected_tpg: Some(256), ..MslConfig::default() })
        .generate(&reduction_kernel(ReduceKind::Mean))
        .unwrap();
    assert!(large.contains("/ float(lsize)"), "slow-path mean must divide by lsize:\n{large}");
}

#[test]
fn reduction_sum_snapshot_default() {
    let msl = MslGenerator::default().generate(&reduction_kernel(ReduceKind::Sum)).unwrap();
    assert_snapshot!(msl);
}

#[test]
fn reduction_sum_snapshot_small_tpg() {
    let cfg = MslConfig { expected_tpg: Some(32), ..MslConfig::default() };
    let msl = MslGenerator::new(cfg).generate(&reduction_kernel(ReduceKind::Sum)).unwrap();
    assert_snapshot!(msl);
}

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
    ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ValueId},
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

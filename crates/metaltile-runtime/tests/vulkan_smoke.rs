//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Phase-1 Vulkan smoke test (`VULKAN_BACKEND_SPEC.md`): prove the pipeline
//! end-to-end on a real Vulkan device —
//!   IR → GlslGenerator (GLSL compute) → shaderc (SPIR-V) → VkPipeline →
//!   dispatch → read-back → compare against the CPU oracle.
//!
//! Runs only with `--features vulkan`. Skips cleanly if no Vulkan device
//! is present (CPU-only CI).
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;

use metaltile_codegen::{CodegenBackend, GlslGenerator};
use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{
        BinOpKind, ConstExprDecl, IndexExpr, Kernel, KernelMode, Op, Param, ParamKind,
        ReduceKind, UnaryOpKind, ValueId,
    },
    shape::Shape,
};
use metaltile_runtime::VulkanDevice;

fn vector_add_ir() -> Kernel {
    let mut k = Kernel::new("vector_add");
    for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.name_value(ValueId::new(0), "idx");
    k.body.push_op(
        Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
        ValueId::new(1),
    );
    k.body.name_value(ValueId::new(1), "x");
    k.body.push_op(
        Op::Load { src: "b".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
        ValueId::new(2),
    );
    k.body.name_value(ValueId::new(2), "y");
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.name_value(ValueId::new(3), "sum");
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(3),
        mask: None,
    });
    k
}

#[test]
fn glsl_codegen_emits_valid_source() {
    let g = GlslGenerator::new();
    let src = g.generate(&vector_add_ir()).unwrap();
    assert!(src.contains("#version 460"));
    assert!(src.contains("layout(set = 0, binding = 0, scalar) readonly buffer Buf_a"));
}

#[test]
fn shaderc_glsl_to_spv_for_vector_add() {
    // shaderc is statically linked, so this test can run even without
    // a Vulkan device — it exercises the GLSL→SPIR-V path in isolation.
    let g = GlslGenerator::new();
    let glsl = g.generate(&vector_add_ir()).unwrap();
    let spv = metaltile_runtime::compile_glsl_to_spv(&glsl, "vector_add.comp")
        .expect("shaderc compile");
    // SPIR-V magic number `0x07230203` (little-endian).
    assert!(spv.len() >= 4);
    assert_eq!(&spv[..4], &[0x03, 0x02, 0x23, 0x07]);
    assert_eq!(spv.len() % 4, 0);
    eprintln!("vulkan_smoke: vector_add SPIR-V = {} bytes", spv.len());
}

#[test]
fn vulkan_vector_add_f32_bit_exact() {
    let dev = match VulkanDevice::create() {
        Ok(Some(d)) => d,
        Ok(None) => {
            eprintln!("vulkan_smoke: no Vulkan device — skipping");
            return;
        }
        Err(e) => {
            eprintln!("vulkan_smoke: Vulkan init failed ({e:?}) — skipping");
            return;
        }
    };
    eprintln!(
        "vulkan_smoke: device='{}' qfam={}",
        dev.name(),
        dev.queue_family()
    );

    const N: usize = 16 * 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32) * -0.25 + 7.0).collect();
    let oracle: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let mut bufs = BTreeMap::new();
    let to_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    bufs.insert("a".to_string(), to_bytes(&a));
    bufs.insert("b".to_string(), to_bytes(&b));
    bufs.insert("c".to_string(), vec![0u8; N * 4]);

    let block = 256u32;
    let grid = (N as u32).div_ceil(block);
    let k = vector_add_ir();
    let out = dev
        .run_kernel(&k, &bufs, [grid, 1, 1], [block, 1, 1])
        .expect("vulkan run_kernel");

    let c_bytes = out.get("c").expect("output `c` present");
    let c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut max_abs: f32 = 0.0;
    for (got, want) in c.iter().zip(&oracle) {
        max_abs = max_abs.max((got - want).abs());
    }
    eprintln!("vulkan_smoke: max|Δ| = {max_abs:e}");
    assert!(max_abs == 0.0, "vector_add bit-exact violated: max|Δ| = {max_abs:e}");
}

/// Same `scale_add_exp` shape as the HIP/CUDA smoke — constexpr scalar +
/// `exp` intrinsic. Vulkan/GLSL maps `exp` directly to GLSL.std.450's
/// `Exp` extension instruction, so this also validates the unary intrinsic
/// table for the SPIR-V profile.
fn scale_add_exp_ir() -> Kernel {
    let mut k = Kernel::new("scale_add_exp");
    for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("scale"),
        dtype: DType::F32,
        value: None,
    });
    let (idx, x, sc, mul, y, sum, e) = (
        ValueId::new(0), ValueId::new(1), ValueId::new(2),
        ValueId::new(3), ValueId::new(4), ValueId::new(5), ValueId::new(6),
    );
    k.body.push_op(Op::ProgramId { axis: 0 }, idx);
    k.body.name_value(idx, "idx");
    k.body.push_op(
        Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(idx)], mask: None, other: None },
        x,
    );
    k.body.name_value(x, "x");
    k.body.push_op(Op::Load { src: "scale".into(), indices: vec![], mask: None, other: None }, sc);
    k.body.name_value(sc, "sc");
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: x, rhs: sc }, mul);
    k.body.name_value(mul, "mul");
    k.body.push_op(
        Op::Load { src: "b".into(), indices: vec![IndexExpr::Value(idx)], mask: None, other: None },
        y,
    );
    k.body.name_value(y, "y");
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: mul, rhs: y }, sum);
    k.body.name_value(sum, "sum");
    k.body.push_op(Op::UnaryOp { op: UnaryOpKind::Exp, value: sum }, e);
    k.body.name_value(e, "e");
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(idx)],
        value: e,
        mask: None,
    });
    k
}

#[test]
fn vulkan_scale_add_exp_f32_tight_tol() {
    let dev = match VulkanDevice::create() {
        Ok(Some(d)) => d,
        _ => {
            eprintln!("vulkan_smoke: no Vulkan device — skipping");
            return;
        }
    };

    const N: usize = 4096;
    let scale: f32 = 0.0125;
    let a: Vec<f32> = (0..N).map(|i| ((i as f32) - 2048.0) * 0.01).collect();
    let b: Vec<f32> = (0..N).map(|i| ((i % 17) as f32) * 0.02).collect();
    let oracle: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x * scale + y).exp()).collect();

    let mut bufs = BTreeMap::new();
    let to_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    bufs.insert("a".into(), to_bytes(&a));
    bufs.insert("b".into(), to_bytes(&b));
    bufs.insert("c".into(), vec![0u8; N * 4]);
    bufs.insert("scale".into(), scale.to_le_bytes().to_vec());

    let block = 256u32;
    let grid = (N as u32).div_ceil(block);
    let k = scale_add_exp_ir();
    let out = dev
        .run_kernel(&k, &bufs, [grid, 1, 1], [block, 1, 1])
        .expect("vulkan run_kernel");

    let c: Vec<f32> = out["c"]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    // Vulkan/GLSL `exp` is implementation-defined precision (typically a
    // few ULP) — pick a tolerance roughly matching the CUDA --fmad=false
    // path, with some headroom for vendor variation.
    let max_rel: f32 = c
        .iter()
        .zip(&oracle)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-30))
        .fold(0.0f32, f32::max);
    eprintln!("vulkan_smoke: scale_add_exp max_rel = {max_rel:e}");
    assert!(max_rel < 1e-5, "scale_add_exp tol broken: max_rel = {max_rel:e}");
}

/// row_reduce_sum — same IR shape as the HIP smoke. Validates the Phase-2
/// Vulkan Reduction-mode lowering: per-thread grid-stride accumulator +
/// workgroup-shared barrier-tree reduction (subgroup-width agnostic;
/// `VULKAN_BACKEND_SPEC.md §4.1`).
fn row_reduce_sum_ir() -> Kernel {
    let mut k = Kernel::new("row_reduce_sum");
    k.mode = KernelMode::Reduction;
    k.params.push(Param {
        name: "inp".into(), dtype: DType::F32, shape: Shape::scalar(),
        is_output: false, kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "out".into(), dtype: DType::F32, shape: Shape::scalar(),
        is_output: true, kind: ParamKind::Tensor,
    });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n"),
        dtype: DType::U32,
        value: None,
    });
    let (row, nv, rs, re, acc, res) = (
        ValueId::new(0), ValueId::new(1), ValueId::new(2),
        ValueId::new(3), ValueId::new(4), ValueId::new(5),
    );
    k.body.push_op(Op::ProgramId { axis: 0 }, row);
    k.body.name_value(row, "row");
    k.body.push_op(Op::Load { src: "n".into(), indices: vec![], mask: None, other: None }, nv);
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: row, rhs: nv }, rs);
    k.body.name_value(rs, "rs");
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: rs, rhs: nv }, re);
    k.body.name_value(re, "re");
    k.body.push_op(
        Op::StrideReduce {
            src: "inp".into(), offset: rs, stride: nv, end: re,
            op: ReduceKind::Sum, dtype: DType::F32,
            transform: None, secondary_src: None, secondary_base: None,
        },
        acc,
    );
    k.body.name_value(acc, "acc");
    k.body.push_op(Op::Reduce { value: acc, axis: 0, op: ReduceKind::Sum }, res);
    k.body.name_value(res, "result");
    k.body.push_op_no_result(Op::Store {
        dst: "out".into(), indices: vec![IndexExpr::Value(row)], value: res, mask: None,
    });
    k
}

#[test]
fn vulkan_row_reduce_sum_f32() {
    let dev = match VulkanDevice::create() {
        Ok(Some(d)) => d,
        _ => {
            eprintln!("vulkan_smoke: no Vulkan device — skipping");
            return;
        }
    };

    const ROWS: usize = 32;
    const COLS: usize = 4096;
    let inp: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as i32 % 257) as f32) * 0.001 - 0.1)
        .collect();
    let oracle: Vec<f32> = (0..ROWS)
        .map(|r| inp[r * COLS..(r + 1) * COLS].iter().sum::<f32>())
        .collect();

    let mut bufs = BTreeMap::new();
    let to_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    bufs.insert("inp".into(), to_bytes(&inp));
    bufs.insert("out".into(), vec![0u8; ROWS * 4]);
    bufs.insert("n".into(), (COLS as u32).to_le_bytes().to_vec());

    let block = 256u32;
    let grid = ROWS as u32;
    let k = row_reduce_sum_ir();
    let out = dev
        .run_kernel(&k, &bufs, [grid, 1, 1], [block, 1, 1])
        .expect("vulkan run_kernel");

    let got: Vec<f32> = out["out"]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let max_rel: f32 = got
        .iter()
        .zip(&oracle)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-30))
        .fold(0.0f32, f32::max);
    eprintln!("vulkan_smoke: row_reduce_sum max_rel = {max_rel:e}");
    assert!(max_rel < 1e-5, "row_reduce_sum tol broken: max_rel = {max_rel:e}");
}

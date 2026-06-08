//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Phase-1 HIP smoke test (`AMD_BACKEND_SPEC.md`): prove the pipeline
//! end-to-end on a real AMD device —
//!   IR → HipGenerator (HIP C++) → hipRTC (AMDGPU code-object) → module
//!   → launch → read-back → compare against the CPU oracle.
//!
//! Runs only with `--features hip`. When no AMD device is present, it
//! skips (no failure) so CI without a GPU is unaffected.
#![cfg(feature = "hip")]

use std::collections::BTreeMap;

use metaltile_codegen::{CodegenBackend, HipGenerator};
use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{
        BinOpKind, ConstExprDecl, IndexExpr, Kernel, KernelMode, Op, Param, ParamKind,
        ReduceKind, UnaryOpKind, ValueId,
    },
    shape::Shape,
};
use metaltile_runtime::HipDevice;

/// out[i] = a[i] + b[i]  (KernelMode::Elementwise, f32). Same IR shape as
/// the CUDA smoke test.
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
fn hip_codegen_emits_valid_source() {
    // Always-runnable correctness check on the emitter — no device needed.
    let g = HipGenerator::new();
    let src = g.generate(&vector_add_ir()).unwrap();
    assert!(src.contains("extern \"C\" __global__ void vector_add("));
    assert!(src.contains("#include <hip/hip_fp16.h>"));
    assert!(!src.contains("cuda_fp16.h"));
}

#[test]
fn hip_vector_add_f32_bit_exact() {
    // Spin up the device; skip if no AMD GPU / ROCm runtime is present.
    let dev = match HipDevice::create() {
        Ok(Some(d)) => d,
        Ok(None) => {
            eprintln!("hip_smoke: no HIP device — skipping");
            return;
        }
        Err(e) => {
            eprintln!("hip_smoke: HIP init failed ({e:?}) — skipping");
            return;
        }
    };
    eprintln!(
        "hip_smoke: device='{}' gfx={} warp_size={}",
        dev.name(),
        dev.gfx_arch(),
        dev.warp_size()
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
        .expect("hip run_kernel");

    let c_bytes = out.get("c").expect("output `c` present");
    let c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Bit-exact: vector_add has no FMA / reduction round-off; HIP single-
    // precision + the IEEE oracle agree to 0 ULP.
    let mut max_abs: f32 = 0.0;
    for (got, want) in c.iter().zip(&oracle) {
        max_abs = max_abs.max((got - want).abs());
    }
    eprintln!("hip_smoke: max|Δ| = {max_abs:e}");
    assert!(max_abs == 0.0, "vector_add bit-exact violated: max|Δ| = {max_abs:e}");
}

/// out[i] = exp(a[i] * scale + b[i]) — exercises a Scalar constexpr arg,
/// Mul/Add chain, and the `expf` intrinsic. Mirrors the CUDA smoke set.
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
    // Constexpr scalar referenced by name (Load with empty indices on a
    // constexpr name lowers to the kernel-arg scalar).
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
fn hip_scale_add_exp_f32_tight_tol() {
    let dev = match HipDevice::create() {
        Ok(Some(d)) => d,
        _ => {
            eprintln!("hip_smoke: no HIP device — skipping");
            return;
        }
    };
    const N: usize = 4096;
    // Small magnitudes so exp doesn't saturate; tests `expf` precision.
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
        .expect("hip run_kernel");

    let c: Vec<f32> = out["c"]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let max_rel: f32 = c
        .iter()
        .zip(&oracle)
        .map(|(g, w)| ((g - w).abs() / w.abs().max(1e-30)))
        .fold(0.0f32, f32::max);
    eprintln!("hip_smoke: scale_add_exp max_rel = {max_rel:e}");
    assert!(max_rel < 5e-7, "scale_add_exp tol broken: max_rel = {max_rel:e}");
}

/// out[row] = sum(inp[row*n .. row*n+n]). Reduction-mode kernel; ports the
/// CUDA `row_reduce_sum` IR verbatim. Validates warp-shuffle reduction
/// (`__shfl_down_sync`) + shared-mem tree + `__syncthreads` on RDNA 4
/// wave32 — the same path that was Phase-2 GREEN on the GX10 Blackwell.
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
fn hip_row_reduce_sum_f32() {
    let dev = match HipDevice::create() {
        Ok(Some(d)) => d,
        _ => {
            eprintln!("hip_smoke: no HIP device — skipping");
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
        .expect("hip run_kernel");

    let got: Vec<f32> = out["out"]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    // Reduction has tree-summation reordering vs the CPU oracle's left-fold.
    // Phase-2-style tolerance — a few ULP per accumulated element.
    let max_rel: f32 = got
        .iter()
        .zip(&oracle)
        .map(|(g, w)| ((g - w).abs() / w.abs().max(1e-30)))
        .fold(0.0f32, f32::max);
    eprintln!("hip_smoke: row_reduce_sum max_rel = {max_rel:e}");
    assert!(max_rel < 1e-5, "row_reduce_sum tol broken: max_rel = {max_rel:e}");
}

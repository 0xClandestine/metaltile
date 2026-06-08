//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Phase-1 CUDA smoke test (CUDA_BACKEND_SPEC §5.1): prove the pipeline
//! end-to-end on a real NVIDIA device —
//!   IR → CudaGenerator (CUDA C++) → NVRTC (PTX) → module → launch →
//!   read-back → compare against the CPU oracle.
//!
//! Runs only with `--features cuda` on a CUDA host (the GX10 / sm_121).
//! When no device is present, it skips (no failure) so CI without a GPU
//! is unaffected.
#![cfg(feature = "cuda")]

use std::ffi::c_void;

use metaltile_codegen::{CodegenBackend, CudaGenerator};
use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{
        BinOpKind, ConstExprDecl, IndexExpr, Kernel, Op, Param, ParamKind, ReduceKind, UnaryOpKind,
        ValueId,
    },
    shape::Shape,
};
use metaltile_runtime::CudaDevice;

/// out[i] = a[i] + b[i]  (KernelMode::Elementwise, f32). Mirrors the
/// codegen `vector_add` reference kernel.
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

/// out[i] = exp(a[i] * scale + b[i]) — exercises a Scalar constexpr arg,
/// Mul/Add chain, and the `__expf` intrinsic (UnaryOp::Exp). Proves the
/// walker + constexpr kernel-arg marshalling generalize past `vector_add`.
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
    let idx = ValueId::new(0);
    let x = ValueId::new(1);
    let y = ValueId::new(2);
    let s = ValueId::new(3);
    let m = ValueId::new(4);
    let sum = ValueId::new(5);
    let e = ValueId::new(6);
    k.body.push_op(Op::ProgramId { axis: 0 }, idx);
    k.body.name_value(idx, "idx");
    k.body.push_op(
        Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(idx)], mask: None, other: None },
        x,
    );
    k.body.push_op(
        Op::Load { src: "b".into(), indices: vec![IndexExpr::Value(idx)], mask: None, other: None },
        y,
    );
    // Scalar constexpr load: empty indices → by-value kernel arg `scale`.
    k.body.push_op(Op::Load { src: "scale".into(), indices: vec![], mask: None, other: None }, s);
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: x, rhs: s }, m);
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: m, rhs: y }, sum);
    k.body.push_op(Op::UnaryOp { op: UnaryOpKind::Exp, value: sum }, e);
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(idx)],
        value: e,
        mask: None,
    });
    k
}

/// out[row] = sum(inp[row*n .. row*n+n]) — KernelMode::Reduction. Exercises
/// the Phase-2 reduction path: StrideReduce (per-thread grid-stride accum) +
/// Reduce (warp-shuffle + shared-mem tree).
fn row_reduce_sum_ir() -> Kernel {
    let mut k = Kernel::new("row_reduce_sum");
    k.mode = metaltile_core::ir::KernelMode::Reduction;
    k.params.push(Param {
        name: "inp".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output: true,
        kind: ParamKind::Tensor,
    });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n"),
        dtype: DType::U32,
        value: None,
    });
    let row = ValueId::new(0);
    let nval = ValueId::new(1);
    let rs = ValueId::new(2);
    let re = ValueId::new(3);
    let acc = ValueId::new(4);
    let res = ValueId::new(5);
    k.body.push_op(Op::ProgramId { axis: 0 }, row);
    k.body.name_value(row, "row");
    k.body.push_op(Op::Load { src: "n".into(), indices: vec![], mask: None, other: None }, nval);
    k.body.name_value(nval, "n");
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: row, rhs: nval }, rs);
    k.body.name_value(rs, "rs");
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: rs, rhs: nval }, re);
    k.body.name_value(re, "re");
    k.body.push_op(
        Op::StrideReduce {
            src: "inp".into(),
            offset: rs,
            stride: nval, // ignored by the CUDA emitter (uses lsize); any valid vid
            end: re,
            op: ReduceKind::Sum,
            dtype: DType::F32,
            transform: None,
            secondary_src: None,
            secondary_base: None,
        },
        acc,
    );
    k.body.name_value(acc, "acc");
    k.body.push_op(Op::Reduce { value: acc, axis: 0, op: ReduceKind::Sum }, res);
    k.body.name_value(res, "result");
    k.body.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(row)],
        value: res,
        mask: None,
    });
    k
}

/// out[r,i] = x[r,i] * rsqrt(mean(x[r,:]²) + eps) * w[i] — RMSNorm.
/// One element per thread (N == blockDim), Reduction mode. The real fused
/// LLM kernel, built from ops the Phase-2 emitter already supports:
/// square = Mul(x,x), threadgroup Reduce(sum), Cast, Div, Rsqrt, weighted
/// Store. No StrideReduce/StrideStore needed for the 1-elem/thread layout.
fn rms_norm_ir() -> Kernel {
    let mut k = Kernel::new("rms_norm");
    k.mode = metaltile_core::ir::KernelMode::Reduction;
    for (name, is_out) in [("x", false), ("w", false), ("out", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("eps"), dtype: DType::F32, value: None });

    let v = |i| ValueId::new(i);
    let (row, tidv, nval, rs, col, x, sq, ssq, nf, msq, epsv, t, inv, w, xn, outv) =
        (v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11), v(12), v(13), v(14), v(15));
    let ld = |src: &str, idx: Vec<IndexExpr>| Op::Load { src: src.into(), indices: idx, mask: None, other: None };

    k.body.push_op(Op::ProgramId { axis: 0 }, row);
    k.body.name_value(row, "row");
    k.body.push_op(ld("tid", vec![]), tidv);
    k.body.name_value(tidv, "tid");
    k.body.push_op(ld("n", vec![]), nval);
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: row, rhs: nval }, rs);
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: rs, rhs: tidv }, col);
    k.body.name_value(col, "col");
    k.body.push_op(ld("x", vec![IndexExpr::Value(col)]), x);
    k.body.name_value(x, "x");
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: x, rhs: x }, sq); // square
    k.body.push_op(Op::Reduce { value: sq, axis: 0, op: ReduceKind::Sum }, ssq);
    k.body.name_value(ssq, "ssq");
    k.body.push_op(Op::Cast { value: nval, dtype: DType::F32 }, nf);
    k.body.push_op(Op::BinOp { op: BinOpKind::Div, lhs: ssq, rhs: nf }, msq);
    k.body.push_op(ld("eps", vec![]), epsv);
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: msq, rhs: epsv }, t);
    k.body.push_op(Op::UnaryOp { op: UnaryOpKind::Rsqrt, value: t }, inv);
    k.body.name_value(inv, "inv");
    k.body.push_op(ld("w", vec![IndexExpr::Value(tidv)]), w);
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: x, rhs: inv }, xn);
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: xn, rhs: w }, outv);
    k.body.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(col)],
        value: outv,
        mask: None,
    });
    k
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn vector_add_cuda_end_to_end() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping CUDA smoke test");
        return;
    };
    let (maj, min) = dev.compute_capability();
    eprintln!("CUDA device compute capability: sm_{maj}{min}");

    // 1. IR → CUDA C++.
    let src = CudaGenerator::new().generate(&vector_add_ir()).expect("cuda codegen");
    eprintln!("--- generated CUDA ---\n{src}\n----------------------");

    // 2. NVRTC compile + load.
    let module = dev.compile(&src, "vector_add.cu").expect("nvrtc compile");
    let func = module.function("vector_add").expect("get function");

    // 3. Host data + CPU oracle.
    const N: usize = 4096;
    let a: Vec<f32> = (0..N).map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.2).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017 + 0.1).cos() * 0.8).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    // 4. Upload, allocate output, launch.
    let da = dev.upload(&f32s_to_bytes(&a)).expect("upload a");
    let db = dev.upload(&f32s_to_bytes(&b)).expect("upload b");
    let dc = dev.alloc(N * 4).expect("alloc c");

    let mut pa = da.device_ptr();
    let mut pb = db.device_ptr();
    let mut pc = dc.device_ptr();
    let mut n: u32 = N as u32;
    let mut args: [*mut c_void; 4] = [
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (N as u32).div_ceil(block);
    dev.launch_1d(func, grid, block, &mut args).expect("launch");

    // 5. Read back + compare to oracle.
    let mut out_bytes = vec![0u8; N * 4];
    dev.download(&dc, &mut out_bytes).expect("download c");
    let got = bytes_to_f32s(&out_bytes);

    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        max_err = max_err.max((g - e).abs());
    }
    eprintln!("max|Δ| = {max_err:.3e} over {N} elements");
    assert!(max_err <= 1e-6, "CUDA vector_add mismatch: max|Δ|={max_err:.3e}");
}

#[test]
fn scale_add_exp_cuda_end_to_end() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping CUDA smoke test");
        return;
    };

    let src = CudaGenerator::new().generate(&scale_add_exp_ir()).expect("cuda codegen");
    eprintln!("--- generated CUDA ---\n{src}\n----------------------");
    let module = dev.compile(&src, "scale_add_exp.cu").expect("nvrtc compile");
    let func = module.function("scale_add_exp").expect("get function");

    const N: usize = 4096;
    let scale: f32 = 0.5;
    let a: Vec<f32> = (0..N).map(|i| (i as f32 * 0.011).sin() * 0.7).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32 * 0.019).cos() * 0.3).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x * scale + y).exp()).collect();

    let da = dev.upload(&f32s_to_bytes(&a)).expect("upload a");
    let db = dev.upload(&f32s_to_bytes(&b)).expect("upload b");
    let dc = dev.alloc(N * 4).expect("alloc c");

    let mut pa = da.device_ptr();
    let mut pb = db.device_ptr();
    let mut pc = dc.device_ptr();
    let mut sc = scale;
    let mut n: u32 = N as u32;
    // Arg order matches the emitted signature: a, b, c, <constexpr scale>, _n_elems.
    let mut args: [*mut c_void; 5] = [
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut sc as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    dev.launch_1d(func, (N as u32).div_ceil(block), block, &mut args).expect("launch");

    let mut out_bytes = vec![0u8; N * 4];
    dev.download(&dc, &mut out_bytes).expect("download c");
    let got = bytes_to_f32s(&out_bytes);

    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        max_err = max_err.max((g - e).abs());
    }
    // __expf is the fast-math intrinsic — allow a small abs tolerance.
    eprintln!("max|Δ| = {max_err:.3e} over {N} elements (scale_add_exp, __expf)");
    assert!(max_err <= 1e-3, "CUDA scale_add_exp mismatch: max|Δ|={max_err:.3e}");
}

#[test]
fn row_reduce_sum_cuda_end_to_end() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping CUDA smoke test");
        return;
    };

    let src = CudaGenerator::new().generate(&row_reduce_sum_ir()).expect("cuda codegen");
    eprintln!("--- generated CUDA ---\n{src}\n----------------------");
    let module = dev.compile(&src, "row_reduce_sum.cu").expect("nvrtc compile");
    let func = module.function("row_reduce_sum").expect("get function");

    const ROWS: usize = 128;
    const N: usize = 256;
    let inp: Vec<f32> = (0..ROWS * N).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let expected: Vec<f32> = (0..ROWS)
        .map(|r| inp[r * N..(r + 1) * N].iter().sum::<f32>())
        .collect();

    let dinp = dev.upload(&f32s_to_bytes(&inp)).expect("upload inp");
    let dout = dev.alloc(ROWS * 4).expect("alloc out");

    let mut pin = dinp.device_ptr();
    let mut pout = dout.device_ptr();
    let mut n: u32 = N as u32;
    // Arg order: inp, out, <constexpr n>. (Reduction mode → no _n_elems.)
    let mut args: [*mut c_void; 3] = [
        &mut pin as *mut _ as *mut c_void,
        &mut pout as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    // One block per output row; 256 threads per block (8 warps).
    dev.launch_1d(func, ROWS as u32, 256, &mut args).expect("launch");

    let mut out_bytes = vec![0u8; ROWS * 4];
    dev.download(&dout, &mut out_bytes).expect("download out");
    let got = bytes_to_f32s(&out_bytes);

    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        max_err = max_err.max((g - e).abs());
    }
    eprintln!("max|Δ| = {max_err:.3e} over {ROWS} rows (row_reduce_sum)");
    assert!(max_err <= 1e-3, "CUDA row_reduce_sum mismatch: max|Δ|={max_err:.3e}");
}

#[test]
fn rms_norm_cuda_end_to_end() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping CUDA smoke test");
        return;
    };

    let src = CudaGenerator::new().generate(&rms_norm_ir()).expect("cuda codegen");
    eprintln!("--- generated CUDA ---\n{src}\n----------------------");
    let module = dev.compile(&src, "rms_norm.cu").expect("nvrtc compile");
    let func = module.function("rms_norm").expect("get function");

    const ROWS: usize = 64;
    const N: usize = 256; // one element per thread → blockDim = 256
    let eps: f32 = 1e-5;
    let x: Vec<f32> = (0..ROWS * N).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
    let w: Vec<f32> = (0..N).map(|i| 0.5 + (i as f32 * 0.01).sin() * 0.2).collect();

    // CPU oracle.
    let mut expected = vec![0.0f32; ROWS * N];
    for r in 0..ROWS {
        let row = &x[r * N..(r + 1) * N];
        let ssq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0f32 / (ssq / N as f32 + eps).sqrt();
        for i in 0..N {
            expected[r * N + i] = row[i] * inv * w[i];
        }
    }

    let dx = dev.upload(&f32s_to_bytes(&x)).expect("upload x");
    let dw = dev.upload(&f32s_to_bytes(&w)).expect("upload w");
    let dout = dev.alloc(ROWS * N * 4).expect("alloc out");

    let mut px = dx.device_ptr();
    let mut pw = dw.device_ptr();
    let mut pout = dout.device_ptr();
    let mut n: u32 = N as u32;
    let mut ep = eps;
    // Arg order: x, w, out, <constexpr n>, <constexpr eps>.
    let mut args: [*mut c_void; 5] = [
        &mut px as *mut _ as *mut c_void,
        &mut pw as *mut _ as *mut c_void,
        &mut pout as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
        &mut ep as *mut _ as *mut c_void,
    ];
    dev.launch_1d(func, ROWS as u32, N as u32, &mut args).expect("launch");

    let mut out_bytes = vec![0u8; ROWS * N * 4];
    dev.download(&dout, &mut out_bytes).expect("download out");
    let got = bytes_to_f32s(&out_bytes);

    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        max_err = max_err.max((g - e).abs());
    }
    eprintln!("max|Δ| = {max_err:.3e} over {} elems (rms_norm)", ROWS * N);
    assert!(max_err <= 1e-3, "CUDA rms_norm mismatch: max|Δ|={max_err:.3e}");
}

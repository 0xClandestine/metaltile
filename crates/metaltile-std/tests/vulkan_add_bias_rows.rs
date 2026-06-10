//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Vulkan/RDNA4 correctness for the per-feature bias broadcast used by the
//! batched prefill path: `out[r, c] = x[r, c] + bias[c]` over an
//! `[n_rows, n]` activation, expressed elementwise as `out[i] = x[i] +
//! bias[i % n]`.
//!
//! Historically the FFAI consumer routed this through a host round-trip
//! (download bias, tile on CPU, re-upload, plain `add`) because the GLSL
//! `%` broadcast was believed wrong on Vulkan. This test drives the
//! fully-on-device broadcast kernel directly through `run_kernel` so the
//! modulo/index codegen is guarded — both the `Mod` (integer `%`) and the
//! `Div/Mul/Sub` (`i - (i/n)*n`) index forms, against a CPU oracle.
//!
//!   cargo test -p metaltile-std --features vulkan \
//!       --test vulkan_add_bias_rows -- --nocapture
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;

use metaltile::{
    VulkanDevice,
    core::{
        dtype::DType,
        ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ParamKind, ValueId},
        shape::Shape,
    },
};

fn p(name: &str, is_output: bool) -> Param {
    Param {
        name: name.into(),
        dtype: DType::F32,
        shape: Shape::scalar(),
        is_output,
        kind: ParamKind::Tensor,
    }
}

/// `out[i] = x[i] + bias[i % n]`.
/// `index_mode = "mod"`  → col via `BinOpKind::Mod` (integer `%`).
/// `index_mode = "divmul"` → col via `i - (i / n) * n`.
fn bias_rows_kernel(n: u32, index_mode: &str) -> Kernel {
    let mut k = Kernel::new("ffai_add_bias_rows_test");
    k.params.push(p("x", false));
    k.params.push(p("bias", false));
    k.params.push(p("out", true));

    let mut next = 0u32;
    let mut nid = || {
        let v = ValueId::new(next);
        next += 1;
        v
    };

    let i = nid();
    k.body.push_op(Op::ProgramId { axis: 0 }, i);

    let n_const = nid();
    k.body.push_op(Op::Const { value: n as i64 }, n_const);

    let col = nid();
    match index_mode {
        "mod" => {
            k.body.push_op(Op::BinOp { op: BinOpKind::Mod, lhs: i, rhs: n_const }, col);
        },
        "divmul" => {
            let q = nid();
            k.body.push_op(Op::BinOp { op: BinOpKind::Div, lhs: i, rhs: n_const }, q);
            let qn = nid();
            k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: q, rhs: n_const }, qn);
            k.body.push_op(Op::BinOp { op: BinOpKind::Sub, lhs: i, rhs: qn }, col);
        },
        _ => unreachable!(),
    }

    let xv = nid();
    k.body.push_op(
        Op::Load { src: "x".into(), indices: vec![IndexExpr::Value(i)], mask: None, other: None },
        xv,
    );
    let bv = nid();
    k.body.push_op(
        Op::Load {
            src: "bias".into(),
            indices: vec![IndexExpr::Value(col)],
            mask: None,
            other: None,
        },
        bv,
    );
    let sum = nid();
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: xv, rhs: bv }, sum);
    k.body.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(i)],
        value: sum,
        mask: None,
    });
    k
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn read_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    bytes.chunks_exact(4).take(n).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn run_case(dev: &VulkanDevice, n_rows: usize, n: usize, index_mode: &str) {
    let total = n_rows * n;
    let x: Vec<f32> = (0..total).map(|i| (i as f32 * 0.013) % 3.0 - 1.0).collect();
    let bias: Vec<f32> = (0..n).map(|c| (c as f32 * 0.07) % 2.0 - 1.0).collect();
    let expected: Vec<f32> = (0..total).map(|i| x[i] + bias[i % n]).collect();

    let k = bias_rows_kernel(n as u32, index_mode);
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), f32_bytes(&x));
    buffers.insert("bias".into(), f32_bytes(&bias));
    buffers.insert("out".into(), vec![0u8; total * 4]);

    let grid = [(total as u32).div_ceil(256), 1, 1];
    let tpg = [256u32, 1, 1];
    let outputs = dev
        .run_kernel(&k, &buffers, grid, tpg)
        .unwrap_or_else(|e| panic!("[{index_mode} n_rows={n_rows} n={n}] run_kernel failed: {e}"));
    let got = read_f32(outputs.get("out").expect("out missing"), total);

    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = (g - e).abs();
        if d > worst {
            worst = d;
            worst_idx = idx;
        }
    }
    eprintln!(
        "[{index_mode}] n_rows={n_rows} n={n}  max|Δ|={worst:.3e}  \
         (idx {worst_idx} col {}: got {:.5} want {:.5})",
        worst_idx % n,
        got[worst_idx],
        expected[worst_idx]
    );
    assert!(
        (worst as f64) <= 1e-5,
        "[{index_mode}] add_bias_rows broadcast WRONG on Vulkan: max|Δ|={worst:.3e} (n={n})"
    );
}

#[test]
fn add_bias_rows_broadcast_correct_on_vulkan() {
    let Some(dev) = VulkanDevice::create().expect("Vulkan init") else {
        eprintln!("no Vulkan device — skipping");
        return;
    };
    eprintln!("=== add_bias_rows broadcast correctness on Vulkan/RDNA4 ===");
    // Qwen2.5-1.5B prefill projection shapes:
    //   q: n = nq*hd = 12*128 = 1536 ; k/v: n = nkv*hd = 2*128 = 256.
    for mode in ["mod", "divmul"] {
        run_case(&dev, 8, 1536, mode); // q bias, S=8
        run_case(&dev, 8, 256, mode); // k/v bias, S=8
        run_case(&dev, 16, 1536, mode); // larger block
        run_case(&dev, 3, 100, mode); // n not a multiple of warp/256
        run_case(&dev, 7, 13, mode); // tiny, awkward stride
    }
}

//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_grouped_gemm_q8` — the GROUPED multi-row
//! Q8_0 tiled GEMM (O-LoRA-A prefill). Oracle: triple loop with the SAME Q8
//! dequant as `gemm_q8_correctness`, but output column `o` belongs to group
//! `g = o / rows_per_group` and reads the `g`-th `in_dim`-slice of an
//! `n_groups*in_dim`-wide activation row.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemm_q8::ffai_grouped_gemm_q8;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dt: Dt,
    in_dim: usize,
    out_dim: usize,
    n_rows: usize,
    n_groups: usize,
    rows_per_group: usize,
    tol: f32,
) {
    let _g = gpu_lock();
    assert_eq!(out_dim, n_groups * rows_per_group, "out_dim must = n_groups*rows_per_group");
    // A 32-wide output tile must lie entirely within one group.
    assert_eq!(rows_per_group % 32, 0, "rows_per_group must be a multiple of 32");

    let n_blocks = out_dim * in_dim / 32;
    let mut st = 0x6C1E_2026u32;
    let qs: Vec<u32> = (0..n_blocks * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_blocks).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0002 + 0.001).collect();
    // Activation row is n_groups*in_dim wide.
    let mut input: Vec<f32> = (0..n_rows * n_groups * in_dim)
        .map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0)
        .collect();
    if matches!(dt, Dt::F16) {
        input = unpack_bytes(&pack_bytes(&input, Dt::F16), Dt::F16);
    }

    // Q8 weight [out_dim, in_dim].
    let deq = |vidx: usize| -> f32 {
        let block = vidx / 32;
        let lane = vidx % 32;
        let word = qs[block * 8 + lane / 4];
        let by = ((word >> ((lane % 4) * 8)) & 0xff) as i32;
        let q = if by > 127 { by - 256 } else { by };
        d[block] * q as f32
    };
    let row_in_stride = n_groups * in_dim;
    let mut want = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let g = o / rows_per_group;
            let in_col_off = g * in_dim;
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += deq(o * in_dim + k) * input[r * row_in_stride + in_col_off + k];
            }
            want[r * out_dim + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qs".into(), pack_u32_bytes(&qs));
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("input".into(), pack_bytes(&input, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_rows * out_dim], dt));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_rows".into(), (n_rows as u32).to_le_bytes().to_vec());
    buffers.insert("n_groups".into(), (n_groups as u32).to_le_bytes().to_vec());
    buffers.insert("rows_per_group".into(), (rows_per_group as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut kernel = ffai_grouped_gemm_q8::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let gx = (out_dim as u32).div_ceil(32);
    let gy = (n_rows as u32).div_ceil(32);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [gx as usize, gy as usize, 1], [
            1024, 1, 1,
        ])
        .expect("dispatch");
    let got = unpack_bytes(result.outputs.get("out").expect("out"), dt);

    for i in 0..(n_rows * out_dim) {
        let w = want[i];
        let denom = w.abs().max(1.0);
        assert!((got[i] - w).abs() / denom < tol, "i={i}: got={} want={}", got[i], w);
    }
}

#[test]
fn grouped_gemm_q8_f32() {
    // 2 groups, each in_dim=256, out_dim=64 (rows_per_group=32), 32 rows.
    run_case(Dt::F32, 256, 64, 32, 2, 32, 1e-3);
    // 4 groups, edge n_rows not a multiple of 32.
    run_case(Dt::F32, 128, 128, 40, 4, 32, 1e-3);
}

#[test]
fn grouped_gemm_q8_f16() { run_case(Dt::F16, 256, 64, 32, 2, 32, 2e-2); }

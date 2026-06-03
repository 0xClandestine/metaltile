//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_gemm_q8` — the multi-row Q8_0 tiled GEMM
//! used by DSv4 prefill. Oracle: triple-loop with the same Q8 dequant.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemm_q8::ffai_gemm_q8;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

fn run_case(dt: Dt, in_dim: usize, out_dim: usize, n_rows: usize, tol: f32) {
    let _g = gpu_lock();
    let n_blocks = out_dim * in_dim / 32;
    let mut st = 0x515E_2026u32;
    let qs: Vec<u32> = (0..n_blocks * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_blocks).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0002 + 0.001).collect();
    let mut input: Vec<f32> =
        (0..n_rows * in_dim).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();
    // Round inputs to the kernel's dtype so the f32 oracle sees the same
    // values the kernel loads (f16 input rounding, amplified by cancellation
    // over in_dim terms, otherwise dominates the diff). Matches ffai_gemm's test.
    if matches!(dt, Dt::F16) {
        input = unpack_bytes(&pack_bytes(&input, Dt::F16), Dt::F16);
    }

    // Dequant weight to a dense [out_dim, in_dim] f32 for the oracle.
    let deq = |vidx: usize| -> f32 {
        let block = vidx / 32;
        let lane = vidx % 32;
        let word = qs[block * 8 + lane / 4];
        let by = ((word >> ((lane % 4) * 8)) & 0xff) as i32;
        let q = if by > 127 { by - 256 } else { by };
        d[block] * q as f32
    };
    let mut want = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += deq(o * in_dim + k) * input[r * in_dim + k];
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

    let ctx = Context::new().expect("ctx");
    let mut kernel = ffai_gemm_q8::kernel_ir_for(dt.to_dtype());
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
fn gemm_q8_f32() {
    run_case(Dt::F32, 256, 64, 32, 1e-3); // aligned
    run_case(Dt::F32, 512, 96, 40, 1e-3); // edge: out_dim/n_rows not mult of 32
}

#[test]
fn gemm_q8_f16() {
    run_case(Dt::F16, 256, 64, 32, 2e-2);
    run_case(Dt::F16, 512, 96, 40, 2e-2);
}

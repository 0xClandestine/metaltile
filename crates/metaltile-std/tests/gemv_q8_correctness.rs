//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_gemv_q8` — Q8_0 inline-dequant gemv
//! vs a CPU reference using the same dequant (value = d * int8).
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemv_q8::ffai_gemv_q8;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

fn run_case(dt: Dt, k_in: usize, m_out: usize, tol: f32) {
    let _g = gpu_lock();
    let bpr = k_in / 32;
    let mut st = 0xC0FFEEu32;
    // qs: m_out*bpr*8 u32 (random int8 payloads). d: per-block scale. x: input.
    let qs: Vec<u32> = (0..m_out * bpr * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..m_out * bpr).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0001 + 0.001).collect();
    let x: Vec<f32> =
        (0..k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    // CPU reference.
    let mut want = vec![0.0f32; m_out];
    for r in 0..m_out {
        let mut acc = 0.0f32;
        for b in 0..bpr {
            let dv = d[r * bpr + b];
            for w in 0..8 {
                let packed = qs[r * bpr * 8 + b * 8 + w];
                for i in 0..4 {
                    let by = ((packed >> (i * 8)) & 0xff) as i32;
                    let q = if by > 127 { by - 256 } else { by };
                    acc += dv * q as f32 * x[b * 32 + w * 4 + i];
                }
            }
        }
        want[r] = acc;
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qs".into(), pack_u32_bytes(&qs));
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("x".into(), pack_bytes(&x, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; m_out], dt));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut kernel = ffai_gemv_q8::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, 1, 1], [32, 1, 1])
        .expect("dispatch");
    let got = unpack_bytes(result.outputs.get("out").expect("out"), dt);

    for (r, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        assert!((g - w).abs() / denom < tol, "dt={dt:?} r={r}: got={g} want={w}");
    }
}

#[test]
fn gemv_q8_f32() {
    run_case(Dt::F32, 1024, 64, 1e-3);
    run_case(Dt::F32, 8192, 32, 1e-3);
}

#[test]
fn gemv_q8_f16() {
    run_case(Dt::F16, 1024, 64, 3e-2);
    run_case(Dt::F16, 8192, 32, 4e-2);
}

#[test]
fn grouped_gemv_q8_f32() {
    use metaltile_std::ffai::gemv_q8::ffai_grouped_gemv_q8;
    let _g = gpu_lock();
    let k_in = 4096usize;
    let rows_per_group = 1024usize;
    let n_groups = 8usize;
    let m_out = rows_per_group * n_groups;
    let bpr = k_in / 32;
    let mut st = 0xABCDu32;
    let qs: Vec<u32> = (0..m_out * bpr * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..m_out * bpr).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0001 + 0.001).collect();
    let x: Vec<f32> =
        (0..n_groups * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();
    let mut want = vec![0.0f32; m_out];
    for r in 0..m_out {
        let xb = (r / rows_per_group) * k_in;
        let mut acc = 0.0f32;
        for b in 0..bpr {
            let dv = d[r * bpr + b];
            for w in 0..8 {
                let packed = qs[r * bpr * 8 + b * 8 + w];
                for i in 0..4 {
                    let by = ((packed >> (i * 8)) & 0xff) as i32;
                    let q = if by > 127 { by - 256 } else { by };
                    acc += dv * q as f32 * x[xb + b * 32 + w * 4 + i];
                }
            }
        }
        want[r] = acc;
    }
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("qs".into(), pack_u32_bytes(&qs));
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("rows_per_group".into(), (rows_per_group as u32).to_le_bytes().to_vec());
    let ctx = Context::new().expect("ctx");
    let mut kernel = ffai_grouped_gemv_q8::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, 1, 1], [32, 1, 1])
        .expect("dispatch");
    let got = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    for (r, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() / w.abs().max(1.0) < 1e-3, "r={r}: got={g} want={w}");
    }
}

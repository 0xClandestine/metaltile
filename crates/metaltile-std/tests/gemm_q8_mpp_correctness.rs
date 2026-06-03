//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_gemm_q8_mpp` and `ffai_grouped_gemm_q8_mpp`
//! — the cooperative-tensor MMA Q8_0 GEMMs (dense + grouped). Oracle: triple
//! loop with the SAME Q8 dequant as `gemm_q8_correctness`; the MMA kernels
//! differ only in the matmul structure (64×64×32 coop_tile vs scalar). Both
//! live-compile (name has `_mpp_`). out[r,o] = Σ_k dequant(W[o,k]) · x[r,k].
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gemm_q8_mpp::{ffai_gemm_q8_mpp, ffai_grouped_gemm_q8_mpp};

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

/// Q8 weight dequant: weight is [out_dim, k_in] row-major over values.
fn deq(qs: &[u32], d: &[f32], vidx: usize) -> f32 {
    let block = vidx / 32;
    let lane = vidx % 32;
    let word = qs[block * 8 + lane / 4];
    let by = ((word >> ((lane % 4) * 8)) & 0xff) as i32;
    let q = if by > 127 { by - 256 } else { by };
    d[block] * q as f32
}

fn need_apple10() -> bool {
    let probe = Context::new().expect("Context::new");
    let ok = probe.chip_family().is_some_and(|lvl| lvl >= 10);
    if !ok {
        eprintln!("skip: needs Apple10+ GPU for coop_tile MMA");
    }
    ok
}

#[test]
fn gemm_q8_mpp_matches_oracle() {
    let _g = gpu_lock();
    if !need_apple10() {
        return;
    }
    // Aligned to the 64×64×32 tile.
    let n_rows = 128usize;
    let out_dim = 128usize;
    let k_in = 128usize;
    let n_blocks = out_dim * k_in / 32;

    let mut st = 0x4A3D_2026u32;
    let qs: Vec<u32> = (0..n_blocks * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_blocks).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0002 + 0.001).collect();
    let x: Vec<f32> =
        (0..n_rows * k_in).map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0).collect();

    let mut want = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(&qs, &d, o * k_in + k) * x[r * k_in + k];
            }
            want[r * out_dim + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("qs".into(), pack_u32_bytes(&qs));
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_rows * out_dim], Dt::F32));
    buffers.insert("n_rows".into(), (n_rows as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut kernel = ffai_gemm_q8_mpp::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let gx = (out_dim as u32).div_ceil(64) as usize;
    let gy = (n_rows as u32).div_ceil(64) as usize;
    let r = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [gx, gy, 1], [128, 1, 1])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut maxd = 0.0f32;
    for i in 0..(n_rows * out_dim) {
        assert!(got[i].is_finite(), "non-finite at {i}");
        let denom = want[i].abs().max(1.0);
        maxd = maxd.max((got[i] - want[i]).abs() / denom);
    }
    eprintln!("gemm_q8_mpp maxRelDiff={maxd:.6}");
    assert!(maxd < 1e-3, "gemm_q8_mpp diverges from oracle: maxRelDiff={maxd}");
}

#[test]
fn grouped_gemm_q8_mpp_matches_oracle() {
    let _g = gpu_lock();
    if !need_apple10() {
        return;
    }
    // 2 groups, rows_per_group=64 (64-wide feature tile is uniform-group).
    let n_rows = 64usize;
    let k_in = 64usize;
    let n_groups = 2usize;
    let rows_per_group = 64usize;
    let out_dim = n_groups * rows_per_group; // 128
    let n_blocks = out_dim * k_in / 32;

    let mut st = 0x5B7C_2026u32;
    let qs: Vec<u32> = (0..n_blocks * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_blocks).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0002 + 0.001).collect();
    // Activation row is n_groups*k_in wide.
    let x: Vec<f32> = (0..n_rows * n_groups * k_in)
        .map(|_| ((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0)
        .collect();

    let row_in_stride = n_groups * k_in;
    let mut want = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let g = o / rows_per_group;
            let in_col_off = g * k_in;
            let mut acc = 0.0f32;
            for k in 0..k_in {
                acc += deq(&qs, &d, o * k_in + k) * x[r * row_in_stride + in_col_off + k];
            }
            want[r * out_dim + o] = acc;
        }
    }

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("qs".into(), pack_u32_bytes(&qs));
    buffers.insert("d_f32".into(), pack_bytes(&d, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_rows * out_dim], Dt::F32));
    buffers.insert("n_rows".into(), (n_rows as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("n_groups".into(), (n_groups as u32).to_le_bytes().to_vec());
    buffers.insert("rows_per_group".into(), (rows_per_group as u32).to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut kernel = ffai_grouped_gemm_q8_mpp::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let gx = (out_dim as u32).div_ceil(64) as usize;
    let gy = (n_rows as u32).div_ceil(64) as usize;
    let r = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [gx, gy, 1], [128, 1, 1])
        .unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), Dt::F32);

    let mut maxd = 0.0f32;
    for i in 0..(n_rows * out_dim) {
        assert!(got[i].is_finite(), "non-finite at {i}");
        let denom = want[i].abs().max(1.0);
        maxd = maxd.max((got[i] - want[i]).abs() / denom);
    }
    eprintln!("grouped_gemm_q8_mpp maxRelDiff={maxd:.6}");
    assert!(maxd < 1e-3, "grouped_gemm_q8_mpp diverges from oracle: maxRelDiff={maxd}");
}

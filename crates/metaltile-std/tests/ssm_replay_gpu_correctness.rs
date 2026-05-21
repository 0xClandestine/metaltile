//! GPU correctness for `ffai::ssm_replay` — the Mamba 2 SSD
//! tape-capture (`ssm_step_record`) and tape-replay (`ssm_replay`)
//! kernels for speculative-decode rollback.
//!
//! `record` pins the SSD recurrence (`y = C·state + D·x`,
//! `state ← dA·state + dBx`) and the `(dA, dBx)` tape it surfaces.
//! `ssm_replay` pins the re-fold of the first `k` tape entries.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::ssm_replay::{ssm_replay_d16_64_4, ssm_step_record_d16_64_4_2};

const DH: usize = 16;
const DS: usize = 64;
const H: usize = 4;
const G: usize = 2;

fn src(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20_000) as f32 / 20_000.0 * scale - scale * 0.5
        })
        .collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

struct Tape {
    y: Vec<f32>,
    state_out: Vec<f32>,
    da_log: Vec<f32>,
    dbx_log: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn naive_record(
    x: &[f32],
    a_log: &[f32],
    bmat: &[f32],
    cmat: &[f32],
    dvec: &[f32],
    dt: &[f32],
    state_in: &[f32],
    mask: &[u32],
    batch: usize,
    t_total: usize,
    has_mask: bool,
) -> Tape {
    let mut y = vec![0.0_f32; batch * t_total * H * DH];
    let mut da_log = vec![0.0_f32; batch * t_total * H * DS];
    let mut dbx_log = vec![0.0_f32; batch * t_total * H * DH * DS];
    let mut state = state_in.to_vec();
    for n in 0..batch * H {
        let b = n / H;
        let h = n % H;
        let g = h / (H / G);
        let a_neg = -a_log[h].exp();
        for t in 0..t_total {
            let bt = b * t_total + t;
            let bt_h = bt * H + h;
            let bt_g = bt * G + g;
            let active = !has_mask || mask[bt] != 0;
            let dt_v = dt[bt_h];
            let dt_eff = if active { dt_v } else { 0.0 };
            let d_a = if active { (a_neg * dt_v).exp() } else { 1.0 };
            for ds in 0..DS {
                da_log[bt_h * DS + ds] = d_a;
            }
            for dh in 0..DH {
                let x_v = x[bt_h * DH + dh];
                let mut y_acc = 0.0_f32;
                for ds in 0..DS {
                    let dbx = x_v * dt_eff * bmat[bt_g * DS + ds];
                    dbx_log[(bt_h * DH + dh) * DS + ds] = dbx;
                    let s0 = (n * DH + dh) * DS + ds;
                    state[s0] = d_a * state[s0] + dbx;
                    y_acc += state[s0] * cmat[bt_g * DS + ds];
                }
                y[bt_h * DH + dh] = y_acc + x_v * dvec[h];
            }
        }
    }
    Tape { y, state_out: state, da_log, dbx_log }
}

#[allow(clippy::too_many_arguments)]
fn naive_replay(
    snapshot: &[f32],
    da_log: &[f32],
    dbx_log: &[f32],
    mask: &[u32],
    batch: usize,
    t_total: usize,
    k: usize,
    has_mask: bool,
) -> Vec<f32> {
    let mut state = snapshot.to_vec();
    for n in 0..batch * H {
        let b = n / H;
        let h = n % H;
        for t in 0..k {
            let bt = b * t_total + t;
            if has_mask && mask[bt] == 0 {
                continue;
            }
            let bt_h = bt * H + h;
            for dh in 0..DH {
                for ds in 0..DS {
                    let s0 = (n * DH + dh) * DS + ds;
                    state[s0] =
                        da_log[bt_h * DS + ds] * state[s0] + dbx_log[(bt_h * DH + dh) * DS + ds];
                }
            }
        }
    }
    state
}

fn dispatch(
    kernel_ir: fn(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel,
    buffers: &BTreeMap<String, Vec<u8>>,
    batch: usize,
    want: &[&str],
) -> Vec<Vec<f32>> {
    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(metaltile_core::dtype::DType::F32);
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, buffers, &BTreeMap::new(), [1, DH, batch * H], [32, 1, 1])
        .expect("ssm_replay dispatch");
    want.iter().map(|w| unpack_bytes(result.outputs.get(*w).expect(w), Dt::F32)).collect()
}

#[test]
fn ssm_step_record_captures_tape_f32() {
    let _g = gpu_lock();
    let (batch, t) = (1usize, 4usize);
    let x = src(batch * t * H * DH, 0x1, 1.0);
    let a_log = src(H, 0x2, 1.0);
    let bmat = src(batch * t * G * DS, 0x3, 1.0);
    let cmat = src(batch * t * G * DS, 0x4, 1.0);
    let dvec = src(H, 0x5, 0.5);
    let dt: Vec<f32> = src(batch * t * H, 0x6, 0.1).iter().map(|v| 0.2 + v).collect();
    let state_in = src(batch * H * DH * DS, 0x7, 0.3);
    let exp = naive_record(&x, &a_log, &bmat, &cmat, &dvec, &dt, &state_in, &[], batch, t, false);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("x".into(), pack_bytes(&x, Dt::F32));
    b.insert("a_log".into(), pack_bytes(&a_log, Dt::F32));
    b.insert("b".into(), pack_bytes(&bmat, Dt::F32));
    b.insert("c".into(), pack_bytes(&cmat, Dt::F32));
    b.insert("d".into(), pack_bytes(&dvec, Dt::F32));
    b.insert("dt".into(), pack_bytes(&dt, Dt::F32));
    b.insert("state_in".into(), pack_bytes(&state_in, Dt::F32));
    b.insert("mask".into(), u32_bytes(&vec![1; batch * t]));
    b.insert("y".into(), pack_bytes(&vec![0.0; exp.y.len()], Dt::F32));
    b.insert("state_out".into(), pack_bytes(&vec![0.0; state_in.len()], Dt::F32));
    b.insert("da_log".into(), pack_bytes(&vec![0.0; exp.da_log.len()], Dt::F32));
    b.insert("dbx_log".into(), pack_bytes(&vec![0.0; exp.dbx_log.len()], Dt::F32));
    b.insert("t_total".into(), (t as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), 0u32.to_le_bytes().to_vec());

    let got = dispatch(ssm_step_record_d16_64_4_2::kernel_ir_for, &b, batch, &[
        "y",
        "state_out",
        "da_log",
        "dbx_log",
    ]);
    assert!(got[3].iter().any(|&v| v != 0.0), "dBx tape all zeros");
    assert!(max_abs_diff(&got[0], &exp.y) < 2e-3, "y mismatch");
    assert!(max_abs_diff(&got[1], &exp.state_out) < 2e-3, "state mismatch");
    assert!(max_abs_diff(&got[2], &exp.da_log) < 2e-3, "dA tape mismatch");
    assert!(max_abs_diff(&got[3], &exp.dbx_log) < 2e-3, "dBx tape mismatch");
}

fn run_replay(k: usize, has_mask: bool, mask: &[u32]) {
    let _g = gpu_lock();
    let (batch, t) = (1usize, 5usize);
    let snapshot = src(batch * H * DH * DS, 0x21, 0.3);
    let da_log: Vec<f32> = src(batch * t * H * DS, 0x22, 0.1).iter().map(|v| 0.9 + v).collect();
    let dbx_log = src(batch * t * H * DH * DS, 0x23, 0.4);
    let expected = naive_replay(&snapshot, &da_log, &dbx_log, mask, batch, t, k, has_mask);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("state_snapshot".into(), pack_bytes(&snapshot, Dt::F32));
    b.insert("da_log".into(), pack_bytes(&da_log, Dt::F32));
    b.insert("dbx_log".into(), pack_bytes(&dbx_log, Dt::F32));
    b.insert("mask".into(), u32_bytes(mask));
    b.insert("state_after_k".into(), pack_bytes(&vec![0.0; snapshot.len()], Dt::F32));
    b.insert("k_steps".into(), (k as u32).to_le_bytes().to_vec());
    b.insert("t_total".into(), (t as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), u32::from(has_mask).to_le_bytes().to_vec());

    let got = dispatch(ssm_replay_d16_64_4::kernel_ir_for, &b, batch, &["state_after_k"]);
    assert!(
        max_abs_diff(&got[0], &expected) < 2e-3,
        "replay k={k} mask={has_mask}: state mismatch",
    );
}

#[test]
fn ssm_replay_full_prefix_f32() { run_replay(5, false, &[1; 5]); }

#[test]
fn ssm_replay_partial_prefix_f32() { run_replay(2, false, &[1; 5]); }

#[test]
fn ssm_replay_masked_steps_f32() { run_replay(5, true, &[1, 0, 1, 1, 0]); }

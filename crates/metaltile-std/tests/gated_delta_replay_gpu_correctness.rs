//! GPU correctness for `ffai::gated_delta_replay` — the GatedDeltaNet
//! tape-capture (`gated_delta_step_record`) and tape-replay
//! (`state_replay`) kernels for speculative-decode rollback.
//!
//! `record` pins that the forward step still matches the plain
//! recurrence *and* surfaces `delta_t` to the tape. `state_replay`
//! pins the branchless `select`-gated re-fold of the accepted prefix.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta_replay::{
    gated_delta_step_record_d64_32_2_2,
    state_replay_d64_32_2_2,
};

const DK: usize = 64;
const DV: usize = 32;
const HK: usize = 2;
const HV: usize = 2;

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

/// Forward recurrence + delta capture (the `record` reference).
#[allow(clippy::too_many_arguments)]
fn naive_record(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    batch: usize,
    t_val: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0_f32; batch * t_val * HV * DV];
    let mut delta_log = vec![0.0_f32; batch * t_val * HV * DV];
    let mut state = state_in.to_vec();
    for n in 0..batch * HV {
        let b = n / HV;
        let hvh = n % HV;
        let hkh = hvh / (HV / HK);
        for t in 0..t_val {
            let qk = (b * t_val + t) * HK * DK + hkh * DK;
            let vb = (b * t_val + t) * HV * DV + hvh * DV;
            let gb = (b * t_val + t) * HV + hvh;
            for dv in 0..DV {
                let s0 = (n * DV + dv) * DK;
                let mut kv = 0.0_f32;
                for dk in 0..DK {
                    state[s0 + dk] *= g[gb];
                    kv += state[s0 + dk] * k[qk + dk];
                }
                let delta = (v[vb + dv] - kv) * beta[gb];
                delta_log[vb + dv] = delta;
                let mut out = 0.0_f32;
                for dk in 0..DK {
                    state[s0 + dk] += k[qk + dk] * delta;
                    out += state[s0 + dk] * q[qk + dk];
                }
                y[vb + dv] = out;
            }
        }
    }
    (y, state, delta_log)
}

/// Branchless tape re-fold (the `state_replay` reference).
#[allow(clippy::too_many_arguments)]
fn naive_replay(
    delta_log: &[f32],
    k_log: &[f32],
    g_log: &[f32],
    state_in: &[f32],
    mask: &[u32],
    batch: usize,
    t_log: usize,
    accepted: usize,
    has_mask: bool,
) -> Vec<f32> {
    let mut state = state_in.to_vec();
    for n in 0..batch * HV {
        let b = n / HV;
        let hvh = n % HV;
        for t in 0..t_log {
            let do_step = t < accepted && (!has_mask || mask[b * t_log + t] != 0);
            if !do_step {
                continue;
            }
            let dr = (b * t_log + t) * HV * DV + hvh * DV;
            let kr = (b * t_log + t) * HV * DK + hvh * DK;
            let g = g_log[(b * t_log + t) * HV + hvh];
            for dv in 0..DV {
                let s0 = (n * DV + dv) * DK;
                for dk in 0..DK {
                    state[s0 + dk] = state[s0 + dk] * g + k_log[kr + dk] * delta_log[dr + dv];
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
        .dispatch_with_grid(&kernel, buffers, &BTreeMap::new(), [1, DV, batch * HV], [32, 1, 1])
        .expect("gated_delta_replay dispatch");
    want.iter().map(|w| unpack_bytes(result.outputs.get(*w).expect(w), Dt::F32)).collect()
}

#[test]
fn gated_delta_record_captures_tape_f32() {
    let _g = gpu_lock();
    let (batch, t_val) = (1usize, 3usize);
    let q = src(batch * t_val * HK * DK, 0x1, 0.4);
    let k = src(batch * t_val * HK * DK, 0x2, 0.4);
    let v = src(batch * t_val * HV * DV, 0x3, 1.0);
    let g: Vec<f32> = src(batch * t_val * HV, 0x4, 0.1).iter().map(|x| 0.9 + x).collect();
    let beta: Vec<f32> = src(batch * t_val * HV, 0x5, 0.1).iter().map(|x| 0.5 + x).collect();
    let state_in = src(batch * HV * DV * DK, 0x6, 0.2);
    let (exp_y, exp_s, exp_d) = naive_record(&q, &k, &v, &g, &beta, &state_in, batch, t_val);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q".into(), pack_bytes(&q, Dt::F32));
    b.insert("k".into(), pack_bytes(&k, Dt::F32));
    b.insert("v".into(), pack_bytes(&v, Dt::F32));
    b.insert("g".into(), pack_bytes(&g, Dt::F32));
    b.insert("beta".into(), pack_bytes(&beta, Dt::F32));
    b.insert("state_in".into(), pack_bytes(&state_in, Dt::F32));
    b.insert("mask".into(), u32_bytes(&vec![1; batch * t_val]));
    b.insert("y".into(), pack_bytes(&vec![0.0; exp_y.len()], Dt::F32));
    b.insert("state_out".into(), pack_bytes(&vec![0.0; state_in.len()], Dt::F32));
    b.insert("delta_log".into(), pack_bytes(&vec![0.0; exp_d.len()], Dt::F32));
    b.insert("t_val".into(), (t_val as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), 0u32.to_le_bytes().to_vec());

    let got = dispatch(gated_delta_step_record_d64_32_2_2::kernel_ir_for, &b, batch, &[
        "y",
        "state_out",
        "delta_log",
    ]);
    assert!(got[2].iter().any(|&x| x != 0.0), "tape is all zeros");
    assert!(max_abs_diff(&got[0], &exp_y) < 2e-3, "y mismatch");
    assert!(max_abs_diff(&got[1], &exp_s) < 2e-3, "state mismatch");
    assert!(max_abs_diff(&got[2], &exp_d) < 2e-3, "delta tape mismatch");
}

fn run_replay(accepted: usize, has_mask: bool, mask: &[u32]) {
    let _g = gpu_lock();
    let (batch, t_log) = (1usize, 5usize);
    let delta_log = src(batch * t_log * HV * DV, 0x21, 0.5);
    let k_log = src(batch * t_log * HV * DK, 0x22, 0.4);
    let g_log: Vec<f32> = src(batch * t_log * HV, 0x23, 0.1).iter().map(|x| 0.9 + x).collect();
    let state_in = src(batch * HV * DV * DK, 0x24, 0.3);
    let expected =
        naive_replay(&delta_log, &k_log, &g_log, &state_in, mask, batch, t_log, accepted, has_mask);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("delta_log".into(), pack_bytes(&delta_log, Dt::F32));
    b.insert("k_log".into(), pack_bytes(&k_log, Dt::F32));
    b.insert("g_log".into(), pack_bytes(&g_log, Dt::F32));
    b.insert("state_in".into(), pack_bytes(&state_in, Dt::F32));
    b.insert("mask".into(), u32_bytes(mask));
    b.insert("state_out".into(), pack_bytes(&vec![0.0; state_in.len()], Dt::F32));
    b.insert("t_log".into(), (t_log as u32).to_le_bytes().to_vec());
    b.insert("accepted".into(), (accepted as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), u32::from(has_mask).to_le_bytes().to_vec());

    let got = dispatch(state_replay_d64_32_2_2::kernel_ir_for, &b, batch, &["state_out"]);
    assert!(
        max_abs_diff(&got[0], &expected) < 2e-3,
        "replay accepted={accepted} mask={has_mask}: state mismatch",
    );
}

#[test]
fn state_replay_full_prefix_f32() { run_replay(5, false, &[1; 5]); }

#[test]
fn state_replay_partial_prefix_f32() {
    // Only the first 3 tape steps are accepted; the rest must not fold.
    run_replay(3, false, &[1; 5]);
}

#[test]
fn state_replay_masked_steps_f32() {
    // Accepted prefix of 5, but steps 1 and 3 are masked out.
    run_replay(5, true, &[1, 0, 1, 0, 1]);
}

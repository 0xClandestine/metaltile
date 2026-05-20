//! GPU correctness for `ffai::gated_delta` — the GatedDeltaNet decode
//! recurrence, standard and fused kernels.
//!
//! Recurrence: `S_t = g_t·S_{t-1} + β_t·k_t·(v_t − kᵀ_t·S_{t-1})ᵀ`,
//! `y_t = q_t·S_t`. The naive f32 oracle runs the same sequential
//! recurrence; the standard kernel takes pre-normalized q/k + explicit
//! g/beta, the fused kernel derives RMSNorm(q)/RMSNorm(k)/g/beta itself.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta::{
    gated_delta_step_d192_128_4_4,
    gated_delta_step_fused_d192_128_4_4,
};

const DK: usize = 192;
const DV: usize = 128;
const HK: usize = 4;
const HV: usize = 4;

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

/// Naive recurrence shared shape; `derive` controls whether q/k/g/beta
/// are taken raw (fused: RMSNorm + softplus-gate applied here) or used
/// as-is (standard).
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state_in: Vec<f32>,
    mask: Vec<u32>,
}

fn naive(inp: &Inputs, batch: usize, t_val: usize, has_mask: bool) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0_f32; batch * t_val * HV * DV];
    let mut state = inp.state_in.clone();
    for n in 0..batch * HV {
        let b = n / HV;
        let hvh = n % HV;
        let hkh = hvh / (HV / HK);
        for t in 0..t_val {
            if has_mask && inp.mask[b * t_val + t] == 0 {
                continue;
            }
            let qk_base = (b * t_val + t) * HK * DK + hkh * DK;
            let v_base = (b * t_val + t) * HV * DV + hvh * DV;
            let gb = (b * t_val + t) * HV + hvh;
            for dv in 0..DV {
                let s0 = (n * DV + dv) * DK;
                let mut kv = 0.0_f32;
                for dk in 0..DK {
                    state[s0 + dk] *= inp.g[gb];
                    kv += state[s0 + dk] * inp.k[qk_base + dk];
                }
                let delta = (inp.v[v_base + dv] - kv) * inp.beta[gb];
                let mut out = 0.0_f32;
                for dk in 0..DK {
                    state[s0 + dk] += inp.k[qk_base + dk] * delta;
                    out += state[s0 + dk] * inp.q[qk_base + dk];
                }
                y[v_base + dv] = out;
            }
        }
    }
    (y, state)
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    kernel_ir: fn(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel,
    dt: Dt,
    buffers: BTreeMap<String, Vec<u8>>,
    batch: usize,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, DV, batch * HV], [32, 1, 1])
        .expect("gated_delta dispatch");
    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let st = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    (y, st)
}

#[test]
fn gated_delta_standard_matches_naive_f32() {
    let _g = gpu_lock();
    let (batch, t_val) = (1usize, 3usize);
    let inp = Inputs {
        q: src(batch * t_val * HK * DK, 0x1, 0.4),
        k: src(batch * t_val * HK * DK, 0x2, 0.4),
        v: src(batch * t_val * HV * DV, 0x3, 1.0),
        g: src(batch * t_val * HV, 0x4, 0.1).iter().map(|v| 0.9 + v).collect(),
        beta: src(batch * t_val * HV, 0x5, 0.1).iter().map(|v| 0.5 + v).collect(),
        state_in: src(batch * HV * DV * DK, 0x6, 0.2),
        mask: vec![1; batch * t_val],
    };
    let (exp_y, exp_s) = naive(&inp, batch, t_val, false);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q".into(), pack_bytes(&inp.q, Dt::F32));
    b.insert("k".into(), pack_bytes(&inp.k, Dt::F32));
    b.insert("v".into(), pack_bytes(&inp.v, Dt::F32));
    b.insert("g".into(), pack_bytes(&inp.g, Dt::F32));
    b.insert("beta".into(), pack_bytes(&inp.beta, Dt::F32));
    b.insert("state_in".into(), pack_bytes(&inp.state_in, Dt::F32));
    b.insert("mask".into(), inp.mask.iter().flat_map(|m| m.to_le_bytes()).collect());
    b.insert("y".into(), pack_bytes(&vec![0.0; batch * t_val * HV * DV], Dt::F32));
    b.insert("state_out".into(), pack_bytes(&vec![0.0; inp.state_in.len()], Dt::F32));
    b.insert("t_val".into(), (t_val as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), 0u32.to_le_bytes().to_vec());

    let (y, s) = dispatch(gated_delta_step_d192_128_4_4::kernel_ir_for, Dt::F32, b, batch);
    assert!(y.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&y, &exp_y) < 2e-3, "y mismatch");
    assert!(max_abs_diff(&s, &exp_s) < 2e-3, "state mismatch");
}

#[test]
fn gated_delta_standard_respects_mask_f32() {
    let _g = gpu_lock();
    let (batch, t_val) = (1usize, 4usize);
    let inp = Inputs {
        q: src(batch * t_val * HK * DK, 0x11, 0.4),
        k: src(batch * t_val * HK * DK, 0x12, 0.4),
        v: src(batch * t_val * HV * DV, 0x13, 1.0),
        g: src(batch * t_val * HV, 0x14, 0.1).iter().map(|v| 0.9 + v).collect(),
        beta: src(batch * t_val * HV, 0x15, 0.1).iter().map(|v| 0.5 + v).collect(),
        state_in: src(batch * HV * DV * DK, 0x16, 0.2),
        mask: vec![1, 0, 1, 0],
    };
    let (_exp_y, exp_s) = naive(&inp, batch, t_val, true);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q".into(), pack_bytes(&inp.q, Dt::F32));
    b.insert("k".into(), pack_bytes(&inp.k, Dt::F32));
    b.insert("v".into(), pack_bytes(&inp.v, Dt::F32));
    b.insert("g".into(), pack_bytes(&inp.g, Dt::F32));
    b.insert("beta".into(), pack_bytes(&inp.beta, Dt::F32));
    b.insert("state_in".into(), pack_bytes(&inp.state_in, Dt::F32));
    b.insert("mask".into(), inp.mask.iter().flat_map(|m| m.to_le_bytes()).collect());
    b.insert("y".into(), pack_bytes(&vec![0.0; batch * t_val * HV * DV], Dt::F32));
    b.insert("state_out".into(), pack_bytes(&vec![0.0; inp.state_in.len()], Dt::F32));
    b.insert("t_val".into(), (t_val as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), 1u32.to_le_bytes().to_vec());

    let (_y, s) = dispatch(gated_delta_step_d192_128_4_4::kernel_ir_for, Dt::F32, b, batch);
    // The masked steps must leave the state untouched between them — the
    // final state still reflects only steps 0 and 2.
    assert!(max_abs_diff(&s, &exp_s) < 2e-3, "masked-state mismatch");
}

#[test]
fn gated_delta_fused_matches_naive_f32() {
    let _g = gpu_lock();
    let (batch, t_val) = (1usize, 3usize);
    let q_raw = src(batch * t_val * HK * DK, 0x21, 1.0);
    let k_raw = src(batch * t_val * HK * DK, 0x22, 1.0);
    let v = src(batch * t_val * HV * DV, 0x23, 1.0);
    let a = src(batch * t_val * HV, 0x24, 1.0);
    let b_in = src(batch * t_val * HV, 0x25, 1.0);
    let a_log = src(HV, 0x26, 0.5);
    let dt_bias = src(HV, 0x27, 0.5);
    let state_in = src(batch * HV * DV * DK, 0x28, 0.2);

    // Mirror the fused kernel's derivations for the oracle.
    let inv_sq = 1.0 / DK as f32;
    let inv_single = (DK as f32).powf(-0.5);
    let rmsnorm = |row: &[f32], post: f32| -> Vec<f32> {
        let ss: f32 = row.iter().map(|x| x * x).sum();
        let rms = (ss / DK as f32 + 1e-6).powf(-0.5);
        row.iter().map(|x| x * rms * post).collect()
    };
    let mut q = vec![0.0_f32; q_raw.len()];
    let mut k = vec![0.0_f32; k_raw.len()];
    for r in 0..batch * t_val * HK {
        q[r * DK..(r + 1) * DK].copy_from_slice(&rmsnorm(&q_raw[r * DK..(r + 1) * DK], inv_sq));
        k[r * DK..(r + 1) * DK].copy_from_slice(&rmsnorm(&k_raw[r * DK..(r + 1) * DK], inv_single));
    }
    let mut g = vec![0.0_f32; a.len()];
    let mut beta = vec![0.0_f32; b_in.len()];
    for b_idx in 0..batch {
        for t in 0..t_val {
            for hv in 0..HV {
                let idx = (b_idx * t_val + t) * HV + hv;
                let dt = a[idx] + dt_bias[hv];
                let sp = if dt > 20.0 { dt } else { (1.0 + dt.exp()).ln() };
                g[idx] = (-a_log[hv].exp() * sp).exp();
                beta[idx] = 1.0 / (1.0 + (-b_in[idx]).exp());
            }
        }
    }
    let inp = Inputs {
        q,
        k,
        v: v.clone(),
        g,
        beta,
        state_in: state_in.clone(),
        mask: vec![1; batch * t_val],
    };
    let (exp_y, exp_s) = naive(&inp, batch, t_val, false);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q_raw".into(), pack_bytes(&q_raw, Dt::F32));
    b.insert("k_raw".into(), pack_bytes(&k_raw, Dt::F32));
    b.insert("v".into(), pack_bytes(&v, Dt::F32));
    b.insert("a".into(), pack_bytes(&a, Dt::F32));
    b.insert("b_input".into(), pack_bytes(&b_in, Dt::F32));
    b.insert("a_log".into(), pack_bytes(&a_log, Dt::F32));
    b.insert("dt_bias".into(), pack_bytes(&dt_bias, Dt::F32));
    b.insert("state_in".into(), pack_bytes(&state_in, Dt::F32));
    b.insert("mask".into(), [1u8, 0, 0, 0].repeat(batch * t_val));
    b.insert("y".into(), pack_bytes(&vec![0.0; batch * t_val * HV * DV], Dt::F32));
    b.insert("state_out".into(), pack_bytes(&vec![0.0; state_in.len()], Dt::F32));
    b.insert("t_val".into(), (t_val as u32).to_le_bytes().to_vec());
    b.insert("has_mask".into(), 0u32.to_le_bytes().to_vec());

    let (y, s) = dispatch(gated_delta_step_fused_d192_128_4_4::kernel_ir_for, Dt::F32, b, batch);
    assert!(y.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&y, &exp_y) < 5e-3, "fused y mismatch");
    assert!(max_abs_diff(&s, &exp_s) < 5e-3, "fused state mismatch");
}

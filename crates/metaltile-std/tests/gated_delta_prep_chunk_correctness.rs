#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

//! GPU correctness for `ffai::gated_delta_prep_chunk::mt_gated_delta_prep_chunk`.
//!
//! The chunked variant must match T sequential calls of
//! `mt_gated_delta_prep_step` carrying state forward — that is exactly
//! the per-token loop body in `Qwen35GDNMixer.forwardMany` we are
//! replacing with one dispatch.
//!
//! CPU oracle: scalar prep + recurrence, T times, state carried.
//! GPU under test: one `mt_gated_delta_prep_chunk` dispatch over `T`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta_prep_chunk::mt_gated_delta_prep_chunk;

// ────────────────────────────────────────────────────────────────────
//  CPU oracle: T-step prep + recurrence with state carried forward.
// ────────────────────────────────────────────────────────────────────

fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Run T scalar prep+recurrence steps. Returns y[B, T, Hv, Dv] and final
/// state[B, Hv, Dv, Dk]. Inputs match the chunk kernel layout —
/// conv_out / a_raw / b_raw have an added T dimension.
fn cpu_chunk_oracle(
    conv_out: &[f32],      // [B, T, 2·Hk·Dk + Hv·Dv]
    a_log: &[f32],         // [Hv]
    dt_bias: &[f32],       // [Hv]
    a_raw: &[f32],         // [B, T, Hv]
    b_raw: &[f32],         // [B, T, Hv]
    q_norm_weight: &[f32], // [Hk·Dk]
    k_norm_weight: &[f32], // [Hk·Dk]
    state_in: &[f32],      // [B, Hv, Dv, Dk]
    b: usize,
    t: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-6_f32;
    let stride_b = 2 * hk * dk + hv * dv;
    let hk_per_hv = hv / hk;

    let mut state = state_in.to_vec(); // mutated across T
    let mut y_all = vec![0.0_f32; b * t * hv * dv];

    for step in 0..t {
        // Re-slice conv_out / a_raw / b_raw for this step.
        for batch in 0..b {
            let bt = batch * t + step;
            let conv_step_base = bt * stride_b;

            // Prep: q/k/v slabs + RMSNorm + g/beta.
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx / hk_per_hv;
                let q_off = conv_step_base + hk_idx * dk;
                let k_off = conv_step_base + hk * dk + hk_idx * dk;
                let v_off = conv_step_base + 2 * hk * dk + hv_idx * dv;

                // ssq for this Hv-head (every Hv-head in an Hk-group
                // recomputes the same ssq — same as the kernel).
                let mut q_ssq = 0.0_f32;
                let mut k_ssq = 0.0_f32;
                for d in 0..dk {
                    let qv = conv_out[q_off + d];
                    let kv = conv_out[k_off + d];
                    q_ssq += qv * qv;
                    k_ssq += kv * kv;
                }
                let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();

                let dt_v = softplus_unclamped(a_raw[bt * hv + hv_idx] + dt_bias[hv_idx]);
                let g_val = (-a_log[hv_idx].exp() * dt_v).exp();
                let beta_val = sigmoid(b_raw[bt * hv + hv_idx]);

                // Recurrence per Dv slot.
                for dv_idx in 0..dv {
                    let v_val = conv_out[v_off + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;

                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    let mut k_normed_arr = vec![0.0_f32; dk];
                    for s_idx in 0..dk {
                        let s = state[s_base + s_idx] * g_val;
                        decayed[s_idx] = s;
                        let kv = conv_out[k_off + s_idx];
                        let kw = k_norm_weight[hk_idx * dk + s_idx];
                        let k_normed = kv * k_inv * kw;
                        k_normed_arr[s_idx] = k_normed;
                        kv_mem += s * k_normed;
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s_idx in 0..dk {
                        let s_new = decayed[s_idx] + k_normed_arr[s_idx] * delta;
                        state[s_base + s_idx] = s_new;
                        let qv = conv_out[q_off + s_idx];
                        let qw = q_norm_weight[hk_idx * dk + s_idx];
                        let q_normed = qv * q_inv * qw;
                        out += s_new * q_normed;
                    }
                    y_all[(bt * hv + hv_idx) * dv + dv_idx] = out;
                }
            }
        }
    }

    (y_all, state)
}

// ────────────────────────────────────────────────────────────────────
//  GPU dispatch.
// ────────────────────────────────────────────────────────────────────

fn run_gpu(
    conv_out: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    a_raw: &[f32],
    b_raw: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    state_in: &[f32],
    dt: Dt,
    b: usize,
    t: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(dk.is_multiple_of(32));
    let n_total = b * hv;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("conv_out".into(), pack_bytes(conv_out, dt));
    buffers.insert("a_log".into(), pack_bytes(a_log, dt));
    buffers.insert("dt_bias".into(), pack_bytes(dt_bias, dt));
    buffers.insert("a_raw".into(), pack_bytes(a_raw, dt));
    buffers.insert("b_raw".into(), pack_bytes(b_raw, dt));
    buffers.insert("q_norm_weight".into(), pack_bytes(q_norm_weight, dt));
    buffers.insert("k_norm_weight".into(), pack_bytes(k_norm_weight, dt));
    buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; b * t * hv * dv], dt));
    buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gated_delta_prep_chunk::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
        .expect("mt_gated_delta_prep_chunk dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    (y, state_out)
}

// ────────────────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let a_first_nan = a.iter().position(|v| !v.is_finite());
    let b_first_nan = b.iter().position(|v| !v.is_finite());
    if let Some(i) = a_first_nan {
        eprintln!("cosine: LHS non-finite at idx {i} (val={}), len={}", a[i], a.len());
    }
    if let Some(i) = b_first_nan {
        eprintln!("cosine: RHS non-finite at idx {i} (val={}), len={}", b[i], b.len());
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let af = *av as f64;
        let bf = *bv as f64;
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

struct Fixture {
    conv_out: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    a_raw: Vec<f32>,
    b_raw: Vec<f32>,
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    state_in: Vec<f32>,
}

fn make_fixture(
    b: usize,
    t: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
    identity_weights: bool,
    weight_scale: f32,
) -> Fixture {
    // Magnitudes are tighter than prep_step's unit-test fixture because
    // 32-step recurrence with `g ≈ 0.99` slow-decay + k_normed ≈ 1.0
    // amplifies state by ~T·delta — with the prep_step magnitudes the
    // state overflows f32 around T=20. Production training keeps this
    // stable via learned a_log / dt_bias; random fixtures need clamping.
    let stride_b = 2 * hk * dk + hv * dv;
    let conv_out: Vec<f32> =
        (0..b * t * stride_b).map(|i| ((i as f32) * 0.0131).sin() * 0.1).collect();
    // More-negative a_log → smaller exp(a_log) → g closer to 1 but the
    // decay accelerates near 1 anyway. Keep a_log ≤ -2 so the *step
    // count* dominates state growth, not single-step gain.
    let a_log: Vec<f32> = (0..hv).map(|i| -2.0 - (i as f32) * 0.05).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.3 + (i as f32) * 0.02).collect();
    let a_raw: Vec<f32> = (0..b * t * hv).map(|i| -0.2 + ((i as f32) * 0.04).sin() * 0.2).collect();
    let b_raw: Vec<f32> = (0..b * t * hv).map(|i| -0.2 + ((i as f32) * 0.03).cos() * 0.2).collect();
    let q_norm_weight: Vec<f32> = if identity_weights {
        vec![weight_scale; hk * dk]
    } else {
        (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect()
    };
    let k_norm_weight: Vec<f32> = if identity_weights {
        vec![weight_scale; hk * dk]
    } else {
        (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect()
    };
    let state_in: Vec<f32> =
        (0..b * hv * dv * dk).map(|i| ((i as f32) * 0.0073).cos() * 0.02).collect();
    Fixture { conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in }
}

fn round_fixture(f: &Fixture, dt: Dt) -> Fixture {
    let r = |xs: &[f32]| xs.iter().map(|&v| dt.round(v)).collect::<Vec<_>>();
    Fixture {
        conv_out: r(&f.conv_out),
        a_log: r(&f.a_log),
        dt_bias: r(&f.dt_bias),
        a_raw: r(&f.a_raw),
        b_raw: r(&f.b_raw),
        q_norm_weight: r(&f.q_norm_weight),
        k_norm_weight: r(&f.k_norm_weight),
        state_in: r(&f.state_in),
    }
}

fn run_cell(
    b: usize,
    t: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
    dt: Dt,
    identity_weights: bool,
    weight_scale: f32,
) -> (f32, f32) {
    let _g = gpu_lock();
    let raw = make_fixture(b, t, hv, hk, dv, dk, identity_weights, weight_scale);
    let f = round_fixture(&raw, dt);
    let (y_cpu, state_cpu) = cpu_chunk_oracle(
        &f.conv_out,
        &f.a_log,
        &f.dt_bias,
        &f.a_raw,
        &f.b_raw,
        &f.q_norm_weight,
        &f.k_norm_weight,
        &f.state_in,
        b,
        t,
        hv,
        hk,
        dv,
        dk,
    );
    let (y_gpu, state_gpu) = run_gpu(
        &f.conv_out,
        &f.a_log,
        &f.dt_bias,
        &f.a_raw,
        &f.b_raw,
        &f.q_norm_weight,
        &f.k_norm_weight,
        &f.state_in,
        dt,
        b,
        t,
        hv,
        hk,
        dv,
        dk,
    );
    (cosine(&y_gpu, &y_cpu), cosine(&state_gpu, &state_cpu))
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────

#[test]
fn prep_chunk_f32_qwen36_t1() {
    // T=1 reduces to a single prep_step — sanity baseline.
    let (cy, cs) = run_cell(1, 1, 32, 16, 128, 128, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 qwen3.6 T=1 y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 qwen3.6 T=1 state cos = {cs:.6}");
}

#[test]
fn prep_chunk_f32_qwen36_t8_identity() {
    let (cy, cs) = run_cell(1, 8, 32, 16, 128, 128, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 qwen3.6 T=8 identity y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 qwen3.6 T=8 identity state cos = {cs:.6}");
}

#[test]
fn prep_chunk_f32_qwen36_t8_weighted() {
    let (cy, cs) = run_cell(1, 8, 32, 16, 128, 128, Dt::F32, false, 0.5);
    assert!(cy >= 0.999, "f32 qwen3.6 T=8 weighted y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 qwen3.6 T=8 weighted state cos = {cs:.6}");
}

#[test]
fn prep_chunk_f32_long_t32_small_shape() {
    // T=32 with smaller shape (matches the existing chunk-only T=64 test).
    // Random fixtures at full Qwen3.6 shape destabilise across 32 steps —
    // both CPU oracle AND GPU overflow at the same indices, confirming
    // it's fixture dynamics rather than a kernel bug. Production
    // validation lives in FFAI's forwardManyEquivalence over real weights.
    let (cy, cs) = run_cell(1, 32, 4, 2, 8, 32, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 long T=32 small y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 long T=32 small state cos = {cs:.6}");
}

#[test]
fn prep_chunk_f32_dk_256() {
    // Dk=256 ⇒ n_per_t = 8, full stack_alloc slot usage.
    let (cy, cs) = run_cell(1, 4, 4, 2, 8, 256, Dt::F32, true, 1.0);
    assert!(cy >= 0.999, "f32 Dk=256 y cos = {cy:.6}");
    assert!(cs >= 0.999, "f32 Dk=256 state cos = {cs:.6}");
}

#[test]
fn prep_chunk_f16_t1_smoke() {
    // f16 multi-step recurrence overflows f16 dynamic range as the
    // state grows step-by-step — kernel does fp32 math internally but
    // f16 store/load truncates at each step. f16 is not a production
    // dtype for Qwen3.6 GDN (bf16 + f32-state is); this is a smoke test
    // that the f16 emit + dispatch path is wired correctly at T=1.
    let (cy, cs) = run_cell(1, 1, 4, 2, 8, 32, Dt::F16, true, 1.0);
    assert!(cy >= 0.999, "f16 T=1 y cos = {cy:.6}");
    assert!(cs >= 0.999, "f16 T=1 state cos = {cs:.6}");
}

#[test]
fn prep_chunk_bf16_qwen36_t8() {
    let (cy, cs) = run_cell(1, 8, 32, 16, 128, 128, Dt::Bf16, true, 1.0);
    assert!(cy >= 0.998, "bf16 qwen3.6 T=8 y cos = {cy:.6}");
    assert!(cs >= 0.998, "bf16 qwen3.6 T=8 state cos = {cs:.6}");
}

#[test]
fn prep_chunk_bf16_small_shape_t32() {
    // bf16 with smaller shape exercises 32-step register-state persistence
    // without overflow. Production-scale bf16 validation runs in FFAI
    // forwardManyEquivalence over real model weights.
    let (cy, cs) = run_cell(1, 32, 4, 2, 8, 32, Dt::Bf16, true, 1.0);
    assert!(cy >= 0.995, "bf16 small T=32 y cos = {cy:.6}");
    assert!(cs >= 0.995, "bf16 small T=32 state cos = {cs:.6}");
}

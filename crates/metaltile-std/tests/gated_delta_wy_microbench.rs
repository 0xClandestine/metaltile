//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Microbench: `mt_gated_delta_wy_chunk` vs `mt_gated_delta_step` (MLX baseline).
//!
//! Compares the chunked-WY kernel against running `mt_gated_delta_step`
//! T times (the current production GDN prefill path). Both kernels live
//! in metaltile-std; we drive them via the same Context and time wall
//! clock.
//!
//! Constraints (scalar foundation):
//!   - Dk = Dv = 32 (TG buffer cap)
//!   - C ∈ {8, 16}  (TG buffer cap)
//!
//! Marked `#[ignore]` so it doesn't run on every cargo test.
//! Run with: `cargo test -p metaltile-std --test gated_delta_wy_microbench --release -- --ignored --nocapture`.

#![cfg(target_os = "macos")]

mod common;

use std::{collections::BTreeMap, time::Instant};

use common::{Dt, gpu_lock, pack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::{
    gated_delta::{mt_gated_delta_chunk, mt_gated_delta_step},
    gated_delta_wy::mt_gated_delta_wy_chunk,
};

/// Run the WY chunked kernel once on the given inputs, returning wall time
/// in microseconds. The inputs/outputs are full T-length, so this is one
/// dispatch covering the entire prefill.
#[allow(clippy::too_many_arguments)]
fn time_wy_chunk(
    ctx: &Context,
    b: usize,
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    c: usize,
    iters: usize,
) -> f64 {
    let n_total = b * hv;
    let q: Vec<f32> = (0..t * hk * dk).map(|i| (i as f32 * 0.0173).sin() * 0.1).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| (i as f32 * 0.0211).cos() * 0.1).collect();
    let v: Vec<f32> = (0..t * n_total * dv).map(|i| (i as f32 * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..t * n_total).map(|i| 0.9 + 0.05 * (i as f32 * 0.013).sin()).collect();
    let beta: Vec<f32> = (0..t * n_total).map(|i| 0.5 + 0.2 * (i as f32 * 0.017).cos()).collect();
    let state_in = vec![0.0_f32; n_total * dv * dk];

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(&q, Dt::F32));
    buffers.insert("k".into(), pack_bytes(&k, Dt::F32));
    buffers.insert("v".into(), pack_bytes(&v, Dt::F32));
    buffers.insert("g".into(), pack_bytes(&g, Dt::F32));
    buffers.insert("beta".into(), pack_bytes(&beta, Dt::F32));
    buffers.insert("state_in".into(), pack_bytes(&state_in, Dt::F32));
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], Dt::F32));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; t * n_total * dv], Dt::F32));
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
    buffers.insert("c".into(), (c as u32).to_le_bytes().to_vec());
    buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());

    let mut kernel = mt_gated_delta_wy_chunk::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Warm-up dispatch so the PSO cache is hot.
    let _ = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, n_total, 1], [32, 1, 1])
        .expect("WY dispatch");

    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, n_total, 1], [32, 1, 1])
            .expect("WY dispatch");
    }
    let elapsed_us = start.elapsed().as_micros() as f64;
    elapsed_us / iters as f64
}

/// Fair baseline: `mt_gated_delta_chunk` runs the full T-sequence in ONE
/// dispatch via its inner T-loop. This is the kernel `mt_gated_delta_wy_chunk`
/// is actually trying to beat (both single dispatch over T tokens).
#[allow(clippy::too_many_arguments)]
fn time_chunk_kernel(
    ctx: &Context,
    b: usize,
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    iters: usize,
) -> f64 {
    let n_total = b * hv;
    let q: Vec<f32> = (0..t * hk * dk).map(|i| (i as f32 * 0.0173).sin() * 0.1).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| (i as f32 * 0.0211).cos() * 0.1).collect();
    let v: Vec<f32> = (0..t * n_total * dv).map(|i| (i as f32 * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..t * n_total).map(|i| 0.9 + 0.05 * (i as f32 * 0.013).sin()).collect();
    let beta: Vec<f32> = (0..t * n_total).map(|i| 0.5 + 0.2 * (i as f32 * 0.017).cos()).collect();
    let state_in = vec![0.0_f32; n_total * dv * dk];

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(&q, Dt::F32));
    buffers.insert("k".into(), pack_bytes(&k, Dt::F32));
    buffers.insert("v".into(), pack_bytes(&v, Dt::F32));
    buffers.insert("g".into(), pack_bytes(&g, Dt::F32));
    buffers.insert("beta".into(), pack_bytes(&beta, Dt::F32));
    buffers.insert("state_in".into(), pack_bytes(&state_in, Dt::F32));
    buffers.insert("state_out".into(), pack_bytes(&state_in, Dt::F32));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; t * n_total * dv], Dt::F32));
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
    buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());

    let mut kernel = mt_gated_delta_chunk::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let _ = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
        .expect("chunk dispatch");

    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
            .expect("chunk dispatch");
    }
    let elapsed_us = start.elapsed().as_micros() as f64;
    elapsed_us / iters as f64
}

/// Looped per-step baseline (mostly to show what we'd pay with T separate
/// dispatches — kept for visibility, not the fair comparison).
#[allow(clippy::too_many_arguments)]
fn time_step_loop(
    ctx: &Context,
    b: usize,
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    iters: usize,
) -> f64 {
    let n_total = b * hv;
    // Per-step inputs: pre-allocate one buffer per step.
    let q_all: Vec<f32> = (0..t * hk * dk).map(|i| (i as f32 * 0.0173).sin() * 0.1).collect();
    let k_all: Vec<f32> = (0..t * hk * dk).map(|i| (i as f32 * 0.0211).cos() * 0.1).collect();
    let v_all: Vec<f32> = (0..t * n_total * dv).map(|i| (i as f32 * 0.029).sin() * 0.3).collect();
    let g_all: Vec<f32> = (0..t * n_total).map(|i| 0.9 + 0.05 * (i as f32 * 0.013).sin()).collect();
    let beta_all: Vec<f32> =
        (0..t * n_total).map(|i| 0.5 + 0.2 * (i as f32 * 0.017).cos()).collect();

    let mut kernel = mt_gated_delta_step::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Single dispatch warm-up.
    {
        let q_step = &q_all[..(b * hk * dk)];
        let k_step = &k_all[..(b * hk * dk)];
        let v_step = &v_all[..(n_total * dv)];
        let g_step = &g_all[..n_total];
        let beta_step = &beta_all[..n_total];
        let state = vec![0.0_f32; n_total * dv * dk];
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("q".into(), pack_bytes(q_step, Dt::F32));
        buffers.insert("k".into(), pack_bytes(k_step, Dt::F32));
        buffers.insert("v".into(), pack_bytes(v_step, Dt::F32));
        buffers.insert("g".into(), pack_bytes(g_step, Dt::F32));
        buffers.insert("beta".into(), pack_bytes(beta_step, Dt::F32));
        buffers.insert("state_in".into(), pack_bytes(&state, Dt::F32));
        buffers.insert("state_out".into(), pack_bytes(&state, Dt::F32));
        buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_total * dv], Dt::F32));
        buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
        buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
        buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
        buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
        let _ = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
            .expect("step dispatch");
    }

    let start = Instant::now();
    for _ in 0..iters {
        let mut state = vec![0.0_f32; n_total * dv * dk];
        for ts in 0..t {
            let qs = ts * b * hk * dk;
            let ks = ts * b * hk * dk;
            let vs = ts * n_total * dv;
            let gs = ts * n_total;
            let bs = ts * n_total;
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            buffers.insert("q".into(), pack_bytes(&q_all[qs..qs + b * hk * dk], Dt::F32));
            buffers.insert("k".into(), pack_bytes(&k_all[ks..ks + b * hk * dk], Dt::F32));
            buffers.insert("v".into(), pack_bytes(&v_all[vs..vs + n_total * dv], Dt::F32));
            buffers.insert("g".into(), pack_bytes(&g_all[gs..gs + n_total], Dt::F32));
            buffers.insert("beta".into(), pack_bytes(&beta_all[bs..bs + n_total], Dt::F32));
            buffers.insert("state_in".into(), pack_bytes(&state, Dt::F32));
            buffers.insert("state_out".into(), pack_bytes(&state, Dt::F32));
            buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_total * dv], Dt::F32));
            buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
            buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
            buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
            buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
            let res = ctx
                .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [
                    32, 1, 1,
                ])
                .expect("step dispatch");
            state = res
                .outputs
                .get("state_out")
                .expect("state_out")
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
        }
    }
    let elapsed_us = start.elapsed().as_micros() as f64;
    elapsed_us / iters as f64
}

#[ignore]
#[test]
fn wy_vs_step_bench_t128_dk32_dv32_c16() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (b, t, hk, hv, dk, dv, c) = (1, 128, 1, 1, 32, 32, 16);

    let iters = 5;
    let wy_us = time_wy_chunk(&ctx, b, t, hk, hv, dk, dv, c, iters);
    let chunk_us = time_chunk_kernel(&ctx, b, t, hk, hv, dk, dv, iters);
    let step_us = time_step_loop(&ctx, b, t, hk, hv, dk, dv, 2.min(iters));
    let speedup_vs_chunk = chunk_us / wy_us;
    let speedup_vs_step = step_us / wy_us;
    eprintln!(
        "T={t} Dk={dk} Dv={dv} C={c}: WY={wy_us:.1}us  chunk={chunk_us:.1}us ({speedup_vs_chunk:.2}×)  step×{t}={step_us:.1}us ({speedup_vs_step:.2}×)"
    );
}

#[ignore]
#[test]
fn wy_vs_step_bench_t256_dk32_dv32_c16() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (b, t, hk, hv, dk, dv, c) = (1, 256, 1, 1, 32, 32, 16);

    let iters = 5;
    let wy_us = time_wy_chunk(&ctx, b, t, hk, hv, dk, dv, c, iters);
    let chunk_us = time_chunk_kernel(&ctx, b, t, hk, hv, dk, dv, iters);
    let step_us = time_step_loop(&ctx, b, t, hk, hv, dk, dv, 2.min(iters));
    let speedup_vs_chunk = chunk_us / wy_us;
    let speedup_vs_step = step_us / wy_us;
    eprintln!(
        "T={t} Dk={dk} Dv={dv} C={c}: WY={wy_us:.1}us  chunk={chunk_us:.1}us ({speedup_vs_chunk:.2}×)  step×{t}={step_us:.1}us ({speedup_vs_step:.2}×)"
    );
}

#[ignore]
#[test]
fn wy_vs_step_bench_t512_dk32_dv32_c16() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (b, t, hk, hv, dk, dv, c) = (1, 512, 1, 1, 32, 32, 16);

    let iters = 5;
    let wy_us = time_wy_chunk(&ctx, b, t, hk, hv, dk, dv, c, iters);
    let chunk_us = time_chunk_kernel(&ctx, b, t, hk, hv, dk, dv, iters);
    let step_us = time_step_loop(&ctx, b, t, hk, hv, dk, dv, 2.min(iters));
    let speedup_vs_chunk = chunk_us / wy_us;
    let speedup_vs_step = step_us / wy_us;
    eprintln!(
        "T={t} Dk={dk} Dv={dv} C={c}: WY={wy_us:.1}us  chunk={chunk_us:.1}us ({speedup_vs_chunk:.2}×)  step×{t}={step_us:.1}us ({speedup_vs_step:.2}×)"
    );
}

#[ignore]
#[test]
fn wy_vs_chunk_t2048() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (b, t, hk, hv, dk, dv, c) = (1, 2048, 1, 1, 32, 32, 16);

    let iters = 3;
    let wy_us = time_wy_chunk(&ctx, b, t, hk, hv, dk, dv, c, iters);
    let chunk_us = time_chunk_kernel(&ctx, b, t, hk, hv, dk, dv, iters);
    let speedup_vs_chunk = chunk_us / wy_us;
    eprintln!(
        "T={t} Dk={dk} Dv={dv} C={c}: WY={wy_us:.1}us  chunk={chunk_us:.1}us  WY/chunk={speedup_vs_chunk:.2}×"
    );
}

#[ignore]
#[test]
fn wy_vs_chunk_t4096() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (b, t, hk, hv, dk, dv, c) = (1, 4096, 1, 1, 32, 32, 16);

    let iters = 3;
    let wy_us = time_wy_chunk(&ctx, b, t, hk, hv, dk, dv, c, iters);
    let chunk_us = time_chunk_kernel(&ctx, b, t, hk, hv, dk, dv, iters);
    let speedup_vs_chunk = chunk_us / wy_us;
    eprintln!(
        "T={t} Dk={dk} Dv={dv} C={c}: WY={wy_us:.1}us  chunk={chunk_us:.1}us  WY/chunk={speedup_vs_chunk:.2}×"
    );
}

#[ignore]
#[test]
fn wy_vs_step_bench_t1024_b4_hv4() {
    // Multi-slot test: B=1 Hv=4 (typical Qwen3.6 linear-attn layer), T=1024.
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (b, t, hk, hv, dk, dv, c) = (1, 1024, 1, 4, 32, 32, 16);

    let iters = 2;
    let wy_us = time_wy_chunk(&ctx, b, t, hk, hv, dk, dv, c, iters);
    let step_us = time_step_loop(&ctx, b, t, hk, hv, dk, dv, iters);
    let speedup = step_us / wy_us;
    eprintln!(
        "T={t} Hv={hv} Dk={dk} Dv={dv} C={c}: WY={wy_us:.1}us  step×{t}={step_us:.1}us  speedup={speedup:.2}×"
    );
}

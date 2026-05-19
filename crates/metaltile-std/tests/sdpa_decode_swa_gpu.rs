//! Sliding-window + sink-token perf bench for `ffai::sdpa_decode`.
//!
//! Companion to `sdpa_decode_gpu_correctness.rs`. The correctness file
//! pins that the dense path (`sink_end = 0, window_start = 0`) is
//! bit-identical to the pre-SWA kernel and that the SWA bound split
//! matches a CPU naive reference. This file measures the resulting
//! decode speedup at Qwen3-class GQA shapes and the long-context
//! regimes where sliding window is actually deployed (industry
//! SWA config: `window = 4096` over `n_kv ∈ {8192, 16384, 32768}`).
//!
//! Ignored by default. Run manually:
//!
//!     cargo test --release -p metaltile-std --test sdpa_decode_swa_gpu \
//!         -- --ignored --nocapture
//!
//! Reports median GPU µs and effective GB/s (computed from the
//! *attended* KV bytes, not the full cache — sliding window doesn't
//! touch the masked range, so the bandwidth metric reflects what the
//! kernel actually does). The dense / SWA rows share a Q / K / V
//! buffer trio per shape; only the `sink_end` / `window_start`
//! constexprs differ between dispatches.
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, pack_bytes, ramp};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode::sdpa_decode;

// (n_q_heads, n_kv_heads, n_kv, window_size, sink_tokens). Qwen3-class
// GQA shape (32 Q heads / 8 KV heads) at the long-context regimes where
// sliding window is deployed. Window = 4096 mirrors the industry
// SWA configs; the sink-tokens column mirrors the "attention sink"
// findings (4 is the canonical count from Xiao et al. 2023).
const SHAPES_SWA: &[(usize, usize, usize, usize, usize)] =
    &[(32, 8, 8192, 4096, 4), (32, 8, 16384, 4096, 4), (32, 8, 32768, 4096, 4)];

const WARMUP_ITERS: usize = 20;
const MEASURED_ITERS: usize = 100;

struct DispatchCfg {
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    sink_end: u32,
    window_start: u32,
    scale: f32,
}

fn bench_one(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    cfg: &DispatchCfg,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    dt: Dt,
) -> f64 {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), q_bytes.to_vec());
    buffers.insert("k".into(), k_bytes.to_vec());
    buffers.insert("v".into(), v_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; cfg.n_q_heads * cfg.head_dim * dt.bytes()]);
    buffers.insert("head_dim".into(), (cfg.head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (cfg.n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (cfg.kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (cfg.heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("sink_end".into(), cfg.sink_end.to_le_bytes().to_vec());
    buffers.insert("window_start".into(), cfg.window_start.to_le_bytes().to_vec());
    buffers.insert("scale".into(), cfg.scale.to_le_bytes().to_vec());

    let mut samples = Vec::with_capacity(MEASURED_ITERS);
    for i in 0..(WARMUP_ITERS + MEASURED_ITERS) {
        let r = ctx
            .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [cfg.n_q_heads, 1, 1], [
                1024, 1, 1,
            ])
            .expect("dispatch_with_grid should succeed");
        if i >= WARMUP_ITERS {
            samples.push(r.elapsed_us);
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn run_dense_vs_swa(label: &str, dt: Dt) {
    let head_dim = 128usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let dtype = dt.to_dtype();
    let dt_bytes = dt.bytes();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;

    println!();
    println!("{label} — Apple M-series (median of {MEASURED_ITERS} iters, head_dim=128, gqa=4)");
    println!(
        "  {:>5} {:>6} {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "n_kv", "window", "sinks", "dense µs", "SWA µs", "dense GB/s", "SWA GB/s", "speedup"
    );

    for &(n_q_heads, n_kv_heads, n_kv, window, sinks) in SHAPES_SWA {
        let kv_stride = n_kv;
        let heads_per_group = n_q_heads / n_kv_heads;
        let q = ramp(n_q_heads * head_dim, 17, 8.0);
        let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
        let v = ramp(n_kv_heads * kv_stride * head_dim, 11, 5.0);
        let q_b = pack_bytes(&q, dt);
        let k_b = pack_bytes(&k, dt);
        let v_b = pack_bytes(&v, dt);

        // Sliding window keeps sinks + the last `window` positions.
        // `window_start = max(sink_end, n_kv - window)` — saturating
        // sub guards short-context regimes where window >= n_kv.
        let sink_end = sinks as u32;
        let window_start = n_kv.saturating_sub(window).max(sinks) as u32;
        let attended = (window_start as usize..n_kv).len() + sinks;

        let dense_cfg = DispatchCfg {
            n_q_heads,
            head_dim,
            n_kv,
            kv_stride,
            heads_per_group,
            sink_end: 0,
            window_start: 0,
            scale,
        };
        let swa_cfg = DispatchCfg {
            n_q_heads,
            head_dim,
            n_kv,
            kv_stride,
            heads_per_group,
            sink_end,
            window_start,
            scale,
        };

        let dense_us = bench_one(&ctx, &kernel, &dense_cfg, &q_b, &k_b, &v_b, dt);
        let swa_us = bench_one(&ctx, &kernel, &swa_cfg, &q_b, &k_b, &v_b, dt);

        // Bandwidth model: Q + K + V + O, with K/V sized by the
        // *attended* position count for the SWA row (the kernel
        // doesn't touch masked positions, so charging full n_kv would
        // inflate the GB/s figure and hide actual register/SLC stall
        // patterns). Dense row uses full n_kv.
        let dense_bytes =
            (n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
                * dt_bytes;
        let swa_bytes =
            (n_q_heads * head_dim + 2 * n_kv_heads * attended * head_dim + n_q_heads * head_dim)
                * dt_bytes;
        let dense_gbps = (dense_bytes as f64) / (dense_us * 1e-6) / 1e9;
        let swa_gbps = (swa_bytes as f64) / (swa_us * 1e-6) / 1e9;
        let speedup = dense_us / swa_us;

        println!(
            "  {:>5} {:>6} {:>5}  {:>9.2}  {:>9.2}  {:>9.1}  {:>9.1}  {:>6.2}x",
            n_kv, window, sinks, dense_us, swa_us, dense_gbps, swa_gbps, speedup,
        );
    }
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_swa_perf_bench_f32() {
    run_dense_vs_swa("sdpa_decode SWA f32 (dense vs sliding-window+sinks)", Dt::F32);
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_swa_perf_bench_f16() {
    run_dense_vs_swa("sdpa_decode SWA f16 (dense vs sliding-window+sinks)", Dt::F16);
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn sdpa_decode_swa_perf_bench_bf16() {
    run_dense_vs_swa("sdpa_decode SWA bf16 (dense vs sliding-window+sinks)", Dt::Bf16);
}

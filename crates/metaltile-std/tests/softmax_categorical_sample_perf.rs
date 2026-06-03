//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Quick perf-only timing for `softmax_categorical_sample` at vocab=152K.
//!
//! Ignored by default — run with `--ignored` to measure. Times 1000
//! sequential dispatches on the same buffers and prints the median +
//! min per-dispatch latency. Reference numbers on M5 Max at vocab=152K:
//! single-thread CDF walk ~8370µs, parallel-prefix replacement ~563µs
//! (~15× speedup, dominated by collapsing the O(n) inner loop).

#![cfg(target_os = "macos")]

mod common;

use std::{collections::BTreeMap, time::Instant};

use common::{Dt, gpu_lock, pack_bytes};
use metaltile::core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sampling::softmax_categorical_sample;

#[test]
#[ignore]
fn perf_softmax_categorical_sample_vocab_152k() {
    let _g = gpu_lock();
    let n = 152_064usize;
    let logits: Vec<f32> = (0..n).map(|i| ((i % 257) as f32 - 128.0) * 0.01).collect();
    let temperature: f32 = 1.0;
    let uniform: f32 = 0.4321;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(&logits, Dt::F32));
    buffers.insert("out".into(), vec![0u8; 4]);
    buffers.insert("temperature_in".into(), temperature.to_le_bytes().to_vec());
    buffers.insert("uniform_in".into(), uniform.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = softmax_categorical_sample::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // Warmup.
    for _ in 0..16 {
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
            .expect("warmup dispatch");
    }

    let mut samples: Vec<u128> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let t0 = Instant::now();
        ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
            .expect("perf dispatch");
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let median_ns = samples[samples.len() / 2];
    let min_ns = samples[0];
    println!(
        "softmax_categorical_sample vocab={n}: median={:.1}µs min={:.1}µs",
        median_ns as f64 / 1000.0,
        min_ns as f64 / 1000.0,
    );
}

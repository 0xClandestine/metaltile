//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Time the registered `#[test_kernel]` corpus on the CUDA backend.
//!
//! Companion to `cuda_kernel_corpus.rs` (correctness): same `KernelTest`
//! inventory, but each supported (kernel × dtype) is launched on a real
//! CUDA device under `cuEvent` timing via `CudaDevice::bench_kernel`, and
//! reported as effective DRAM bandwidth (GB/s) using `BenchStats` for the
//! min/median/p95/cv distribution.
//!
//! This is a measurement run, not a pass/fail gate — marked `#[ignore]` so
//! plain `cargo test` skips it. Run explicitly on the CUDA host:
//!
//! ```sh
//! cargo test -p metaltile-std --features cuda --test cuda_bench_corpus \
//!     -- --ignored --nocapture
//! ```
//!
//! ## Read this as a LATENCY profile, not throughput
//!
//! The corpus uses *correctness-sized* tensors (a few KB) — far too small
//! to saturate DRAM. Every kernel floors at the ~2.4µs launch-overhead
//! floor, so the **GB/s column is launch-bound noise** at these sizes; it
//! only starts to mean something for the heavier kernels. The honest signal
//! is **min µs** (steady-state per-launch latency on the live GB10). A true
//! throughput sweep needs the large-input bench harness (Phase 6 CLI
//! wiring); this reuses the correctness inventory so the *mechanism*
//! (cuEvent timing end-to-end) is proven on the full 4164.
//!
//! Caveat: cooperative-matmul ops (qmm / moe / attention) are still
//! **software-emulated** on CUDA, so their numbers reflect the emulation,
//! not the achievable `mma.sync` path. Elementwise / reduction / norm
//! kernels bench honestly.
#![cfg(feature = "cuda")]

use metaltile::runner::BenchStats;
use metaltile_core::dtype::DType;
use metaltile_runtime::CudaDevice;
use std::collections::BTreeMap;

/// Warmup launches (JIT/cache/clock ramp) before timing.
const WARMUP: u32 = 5;
/// Timed launches per (kernel × dtype).
const ITERS: u32 = 50;

struct Row {
    name: String,
    dt: &'static str,
    stats: BenchStats,
    gbps: f64,
}

fn dt_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        _ => "?",
    }
}

#[test]
#[ignore = "measurement run; needs a CUDA host. Run with --ignored --nocapture"]
fn bench_corpus_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    let (maj, min) = dev.compute_capability();
    eprintln!("CUDA device: sm_{maj}{min}  warmup={WARMUP} iters={ITERS}\n");

    let mut rows: Vec<Row> = Vec::new();
    let mut skipped = 0u32;

    for entry in metaltile_std::all_tests() {
        let t = entry.test();
        for &dt in t.dtypes() {
            let setup = t.setup(dt);
            let kernel = setup.kernel();

            // GPU-vs-GPU reference setups need two dispatches; skip.
            if setup.ref_setup().is_some() {
                skipped += 1;
                continue;
            }

            // Buffers: inputs + pre-sized outputs + constexprs.
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            let mut moved = 0usize;
            for inp in setup.inputs() {
                let bytes = inp.data().to_vec();
                moved += bytes.len();
                buffers.insert(inp.name().to_string(), bytes);
            }
            for exp in setup.expected() {
                moved += exp.data().len();
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes());
            }

            let grid = setup.grid();
            match dev.bench_kernel(kernel, &buffers, grid.grid, grid.tpg, WARMUP, ITERS) {
                Ok(samples) => {
                    let stats = BenchStats::from_samples(samples);
                    if !stats.is_valid() || stats.min_us <= 0.0 {
                        skipped += 1;
                        continue;
                    }
                    // Effective bandwidth: bytes moved ÷ steady-state (min) latency.
                    let gbps = moved as f64 / (stats.min_us * 1_000.0);
                    rows.push(Row { name: t.name().to_string(), dt: dt_label(dt), stats, gbps });
                }
                // Unsupported / codegen-gap kernels just don't get a number.
                Err(_) => skipped += 1,
            }
        }
    }

    // Sort heaviest-first by steady-state latency — the meaningful signal at
    // corpus sizes (GB/s is launch-bound noise; see the module docs).
    rows.sort_by(|a, b| b.stats.min_us.partial_cmp(&a.stats.min_us).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!(
        "{:<34} {:>5} {:>10} {:>10} {:>10} {:>8} {:>9}",
        "kernel", "dtype", "min µs", "med µs", "p95 µs", "cv%", "GB/s*"
    );
    eprintln!("{}", "─".repeat(92));
    for r in &rows {
        eprintln!(
            "{:<34} {:>5} {:>10.2} {:>10.2} {:>10.2} {:>8.1} {:>9.1}",
            r.name, r.dt, r.stats.min_us, r.stats.median_us, r.stats.p95_us, r.stats.cv_pct, r.gbps,
        );
    }
    eprintln!("{}", "─".repeat(92));
    eprintln!("timed {} (kernel × dtype), skipped {skipped}", rows.len());
    eprintln!("* GB/s is launch-bound at corpus sizes — read min µs (latency). See module docs.");

    assert!(!rows.is_empty(), "no kernels timed on CUDA — bench pipeline broken");
}

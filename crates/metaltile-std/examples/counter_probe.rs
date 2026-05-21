//! Counter-sampling probe.
//!
//! Runs a tiny elementwise kernel (`out[i] = in[i] + 1`) across a sweep of
//! input sizes and dumps `MTLCommonCounterSetStageUtilization` cycle counts
//! per iteration. Goal is to answer: do the counter values reported by the
//! `GpuRunner::measure_with_counters` shim actually scale with workload,
//! and which stage fields are populated on this device?
//!
//! Usage:
//!     cargo run -p metaltile-std --example counter_probe --release
//!
//! macOS-only. On Intel macOS (no StageUtilization counter set) the probe
//! exits with a message; that's the same condition the shim itself
//! surfaces as `Err`.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("counter_probe requires macOS — Metal counter sampling is platform-specific.");
}

#[cfg(target_os = "macos")]
fn main() {
    use metaltile_std::runner::GpuRunner;

    // Single-thread-per-element add_one. Linear scaling with `n` is what
    // we want to see in the totals column.
    const MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_one(device float* in_  [[buffer(0)]],
                            device float* out_ [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
            out_[gid] = in_[gid] + 1.0;
        }
    "#;

    let runner = GpuRunner::new().expect("Metal device");
    println!("device: {}", runner.device_name);

    // Diagnostic header — tells us what counter sets & sampling points the
    // device exposes, before we even attempt a measurement. Critical for
    // understanding why a measurement might fail.
    let sets = runner.counter_set_names();
    if sets.is_empty() {
        println!("counter sets: (none exposed)");
    } else {
        println!("counter sets:");
        for s in &sets {
            println!("  - {s}");
        }
    }
    println!("sampling-point support:");
    for (name, ok) in runner.counter_sampling_support() {
        println!("  {:<24} {}", name, if ok { "yes" } else { "no" });
    }
    let tick_ns = runner.gpu_tick_period_ns();
    println!("gpu_tick_period_ns: {:.6} ns/tick", tick_ns);

    let kernel = runner.compile(MSL, "add_one").expect("compile add_one");

    // 64K → 16M f32 elements (256 KB → 64 MB).
    let sizes: &[usize] = &[
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
    ];
    const TPG: usize = 256;
    const WARMUP: usize = 3;
    const ITERS: usize = 5;

    println!();
    println!(
        "{:>10}  {:>10}  {:>16}  {:>14}  {:>14}",
        "n_elem", "mean_us", "mean_gpu_ticks", "ticks_min", "ticks_max",
    );
    println!("{}", "-".repeat(78));

    for &n in sizes {
        assert!(n % TPG == 0, "size {n} not divisible by TPG={TPG}");
        let in_buf = runner.buffer_f32(&vec![1.0f32; n]);
        let out_buf = runner.buffer_zeros(n * 4);
        let tgs = [n / TPG, 1, 1];
        let tpg_arr = [TPG, 1, 1];

        runner.flush_slc();
        let samples = match runner.measure_with_counters(
            &kernel,
            &[&in_buf, &out_buf],
            tgs,
            tpg_arr,
            WARMUP,
            ITERS,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("size={n}: counter sampling unavailable: {e}");
                return;
            },
        };

        let mean_us = samples.iter().map(|s| s.gpu_us).sum::<f64>() / samples.len() as f64;
        let ticks: Vec<u64> = samples.iter().map(|s| s.ts.gpu_ticks).collect();
        let mean_ticks = ticks.iter().sum::<u64>() / ticks.len() as u64;
        let min_ticks = *ticks.iter().min().unwrap();
        let max_ticks = *ticks.iter().max().unwrap();
        println!(
            "{:>10}  {:>10.2}  {:>16}  {:>14}  {:>14}",
            n, mean_us, mean_ticks, min_ticks, max_ticks,
        );
    }
}

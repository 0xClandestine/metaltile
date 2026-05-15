//! Scan benchmarks — metal/scan.metal  (MLX, Apache-2.0)
//!
//! Inclusive prefix-sum (cumsum) over rows.
//!
//! MLX reference: `contig_scan_inclusive_sum_float32_float32`
//!   Params: (in, out, axis_size: constant size_t&)
//!   Grid: [1, rows, 1] × [256, 1, 1]
//!
//! MetalTile: `mt_scan_f32` — parallel SIMD two-phase scan, #[kernel] DSL.
//!   Grid: [1, rows, 1] × [256, 1, 1]
//!   N_READS = 1 element per thread per outer iteration.
//!   Phase 1: SIMD exclusive scan within warp; inclusive = exclusive + val.
//!   Phase 2: first warp exclusive-scans the n_simd warp totals.
//!   Phase 3: combine running_prefix + warp_excl + val_incl.
//!   KernelMode::Reduction
//!
//! Note: f32-only (the MLX reference is f32-only).  DSL codegen now supports
//! simd_prefix_exclusive_sum, so the scan kernel is pure DSL.

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{OpBench, OpResult, bench_gbps, check_equiv, run_f32_once},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/scan.metal");

const BENCH: OpBench = OpBench::new("scan", "GB/s");
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)]; // (rows, cols)
const CHECK_ROWS: usize = 4;
const CHECK_N: usize = 256;
const TPG: usize = 256;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Parallel inclusive prefix-sum over a row using two-phase SIMD scan, N_READS=4.
///
/// Each thread processes 4 consecutive elements per outer iteration.
/// `sgs[9]`: slots 0..n_simd-1 hold per-warp exclusive prefixes;
///            slot n_simd holds the running prefix across outer iterations.
///
/// Algorithm per iteration:
///   1. Load 4 values; compute per-thread inclusive prefix sum (s1,s2,s3).
///   2. SIMD exclusive scan on thread totals (s3).
///   3. barrier + lane-31 writes warp total to sgs[sg].
///   4. barrier + warp-0 exclusive-scans warp totals in sgs.
///   5. Combine: out[i] = running_prefix + warp_excl + thread_excl + per_thread_cumsum[i].
///   6. barrier + last thread updates running prefix + barrier.
///
/// Total barriers: 4 × ceil(N/1024) + 1 init  (vs 4 × ceil(N/256) + 1 for N_READS=1).
/// For N=4096, lsize=256: 4×4+1=17 barriers  (vs 4×16+1=65 for N_READS=1).
///
/// Grid: [1, rows, 1] × [256, 1, 1]
#[kernel]
pub fn mt_scan_f32(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let lane = simd_lane;
    let sg = simd_id;
    let ns = n_simd; // = lsize / 32 = 8 for lsize=256
    let row_off = row * n;
    // 8 warp slots (indices 0..7) + 1 running-prefix slot (index ns=8).
    threadgroup_alloc("sgs", 9);
    if lid == 0 {
        threadgroup_store("sgs", ns, 0);
    }
    threadgroup_barrier();
    // Read back 0 as float — used as the zero-pad for OOB elements.
    let zero_f = threadgroup_load("sgs", ns);
    // N_READS=4: process 4 elements per thread per outer iteration.
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        // Load 4 consecutive elements (OOB → 0 to avoid affecting the prefix).
        let v0 = select(base < n, load(inp[row_off + base]), zero_f);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]), zero_f);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]), zero_f);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]), zero_f);
        // Per-thread inclusive prefix: s1=v0+v1, s2=v0+v1+v2, s3=total.
        let s1 = v0 + v1;
        let s2 = s1 + v2;
        let s3 = s2 + v3;
        // Phase 1: SIMD exclusive scan on per-thread totals.
        let thread_excl = simd_scan_exclusive(s3);
        // Phase 2: lane 31 writes the warp's inclusive total to sgs[sg].
        if lane == 31 {
            threadgroup_store("sgs", sg, thread_excl + s3);
        }
        threadgroup_barrier();
        // Phase 3: first warp exclusive-scans the n_simd warp totals.
        if sg == 0 {
            let wt = select(lane < ns, threadgroup_load("sgs", lane), zero_f);
            let wt_excl = simd_scan_exclusive(wt);
            if lane < ns {
                threadgroup_store("sgs", lane, wt_excl);
            }
        }
        threadgroup_barrier();
        // Phase 4: combine and write 4 outputs.
        // base_prefix = running_prefix + warp_excl + thread_excl (exclusive prefix for this thread).
        let cur_prefix = threadgroup_load("sgs", ns);
        let warp_excl = threadgroup_load("sgs", sg);
        let base_prefix = cur_prefix + warp_excl + thread_excl;
        // Each element's inclusive sum = base_prefix + per-thread cumsum up to that element.
        if base < n {
            store(out[row_off + base], base_prefix + v0);
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], base_prefix + s1);
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], base_prefix + s2);
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], base_prefix + s3);
        }
        // Phase 5: last thread updates running prefix for next iteration.
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, base_prefix + s3);
        }
        threadgroup_barrier();
    }
}

fn scan_msl() -> String {
    let mut k = mt_scan_f32::kernel_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[scan]: {e}");
        String::new()
    })
}

// ── Bench ─────────────────────────────────────────────────────────────────────

fn cpu_cumsum(inp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += inp[r * cols + c];
            out[r * cols + c] = acc;
        }
    }
    out
}

pub fn bench_scan(runner: &GpuRunner) -> Vec<OpResult> {
    let mt_msl = scan_msl();
    let mk = if mt_msl.is_empty() { None } else { runner.compile(&mt_msl, "mt_scan_f32").ok() };

    // MLX reference: contig_scan_inclusive_sum_float32_float32
    // Params: (in, out, axis_size: constant size_t [u64])
    let rk = runner.compile(SRC, "contig_scan_inclusive_sum_float32_float32").ok();

    let mut results = Vec::new();
    for &(rows, n) in SHAPES {
        let inp_vals: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();

        let equiv = mk.as_ref().map(|mk| {
            let ref_vals = cpu_cumsum(&inp_vals[..CHECK_ROWS * CHECK_N], CHECK_ROWS, CHECK_N);
            let inp_b = runner.buffer_f32(&inp_vals[..CHECK_ROWS * CHECK_N]);
            let out_b = runner.buffer_zeros(CHECK_ROWS * CHECK_N * 4);
            let ns = runner.buffer_u32(CHECK_N as u32);
            let mt_vals = run_f32_once(
                runner,
                mk,
                &[&inp_b, &out_b, &ns],
                &out_b,
                CHECK_ROWS * CHECK_N,
                [1, CHECK_ROWS, 1],
                [TPG, 1, 1],
            );
            check_equiv(&ref_vals, &mt_vals, 1e-3)
        });

        let inp_buf = runner.buffer_f32(&inp_vals);
        let bytes = (rows * n * 8) as f64; // read + write
        let ns_u64 = runner.buffer_u64(n as u64); // MLX uses size_t (u64)
        let ns_u32 = runner.buffer_u32(n as u32);

        let ref_out = runner.buffer_zeros(rows * n * 4);
        let ref_perf = rk.as_ref().and_then(|rk| {
            bench_gbps(runner, rk, &[&inp_buf, &ref_out, &ns_u64], [1, rows, 1], [TPG, 1, 1], bytes)
        });

        let mt_out = runner.buffer_zeros(rows * n * 4);
        let mt_perf = mk.as_ref().and_then(|mk| {
            bench_gbps(runner, mk, &[&inp_buf, &mt_out, &ns_u32], [1, rows, 1], [TPG, 1, 1], bytes)
        });

        let shape = format!("B={rows} N={n} f32");
        results.push(BENCH.result(shape, ref_perf, mt_perf, equiv));
    }
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_msl_generates() {
        let msl = scan_msl();
        assert!(!msl.trim().is_empty(), "scan MSL should not be empty");
        assert!(msl.contains("mt_scan_f32"), "kernel name missing");
        assert!(msl.contains("simd_prefix_exclusive_sum"), "SIMD scan missing");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn scan_kernel_compiles() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        let msl = scan_msl();
        runner
            .compile(&msl, "mt_scan_f32")
            .unwrap_or_else(|e| panic!("mt_scan_f32 compile error: {e}\nMSL:\n{msl}"));
    }
}

use crate::ops::{KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![
        KernelSpec {
            op: "scan",
            mt_kernel: "mt_scan_f32".into(),
            metal_file: "scan.metal",
            ref_spec: RefSpec::Literal("contig_scan_inclusive_sum_float32_float32"),
            dtypes: &["f32"],
        },
        KernelSpec {
            op: "scan",
            mt_kernel: "mt_scan_f16".into(),
            metal_file: "scan.metal",
            ref_spec: RefSpec::None(
                "f16/bf16 scan not yet in MT bench;                  MLX contig_scan_inclusive_sum_float16_float16 exists",
            ),
            dtypes: &["f16"],
        },
        KernelSpec {
            op: "scan",
            mt_kernel: "mt_scan_bf16".into(),
            metal_file: "scan.metal",
            ref_spec: RefSpec::None(
                "f16/bf16 scan not yet in MT bench;                  MLX contig_scan_inclusive_sum_bfloat16_bfloat16 exists",
            ),
            dtypes: &["bf16"],
        },
    ]
}

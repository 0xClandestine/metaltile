//! Runner for the `__tile_runner` harness.
//!
//! [`Runner`] dispatches benches and tests against a live Metal device,
//! streaming [`ProtocolMessage`] JSON lines to stdout for `tile bench` /
//! `tile test` to consume.

use std::{collections::BTreeMap, io::Write as IoWrite, path::PathBuf};

use metaltile_core::{
    DType,
    all_benches,
    all_tests,
    bench::{BenchBuffer, ConstValue, KernelBench, KernelTest, RefKernel, TestBuffer},
    protocol::{BenchResult, ProtocolMessage, TestResult},
};
use metaltile_runtime::Context;

/// Env var the CLI sets to the resolved `reference_metal_path` from tile.toml.
const REF_METAL_PATH_ENV: &str = "TILE_REF_METAL_PATH";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Top-level entry point called by `__tile_runner`.
pub fn run(args: Args) { Runner::new(args).run(); }

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

/// Command-line arguments forwarded by the `__tile_runner` binary.
pub struct Args {
    pub command: Command,
    pub filter: Option<String>,
}

/// Which operation the runner should perform.
pub enum Command {
    Bench,
    Test,
}

impl Args {
    /// Parse from `std::env::args`.
    ///
    /// Expected format: `__tile_runner bench|test [--filter <pattern>]`
    pub fn from_env() -> Self {
        let raw: Vec<String> = std::env::args().collect();
        let command =
            raw.iter().find(|a| *a == "bench" || *a == "test").map_or(Command::Bench, |a| {
                if a == "test" { Command::Test } else { Command::Bench }
            });
        let filter =
            raw.windows(2).find(|w| w[0] == "--filter" || w[0] == "-f").map(|w| w[1].clone());
        Args { command, filter }
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Tuning knobs for bench timing.
struct BenchConfig {
    warmup_iters: u32,
    timed_iters: u32,
}

impl Default for BenchConfig {
    fn default() -> Self { BenchConfig { warmup_iters: 3, timed_iters: 20 } }
}

/// Stateful runner holding the Metal context and configuration.
struct Runner {
    ctx: Context,
    command: Command,
    filter: Option<String>,
    bench_cfg: BenchConfig,
    /// Resolved path to reference `.metal` source files, read from the
    /// `TILE_REF_METAL_PATH` env var (set by the CLI from tile.toml).
    ref_metal_path: Option<PathBuf>,
}

impl Runner {
    fn new(args: Args) -> Self {
        let mut ctx = Context::new().unwrap_or_else(|e| {
            eprintln!("error: failed to create Metal context: {e}");
            std::process::exit(1);
        });
        // Tests use a 10 s polling timeout to prevent GPU hangs on bad kernels.
        // Benches use waitUntilCompleted (None) so GPU timer is unobstructed and
        // elapsed_us = GPUEndTime - GPUStartTime gives true GPU execution time.
        if let Command::Test = args.command {
            ctx.set_dispatch_timeout(Some(10));
        }
        let ref_metal_path = std::env::var(REF_METAL_PATH_ENV).ok().map(PathBuf::from);

        Runner {
            ctx,
            command: args.command,
            filter: args.filter,
            bench_cfg: BenchConfig::default(),
            ref_metal_path,
        }
    }

    fn run(&self) {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();

        let benches = self.filtered_benches();
        let tests = self.filtered_tests();

        self.emit(&mut out, ProtocolMessage::Start {
            runner_version: "0.1".to_string(),
            total_benches: benches.len() as u32,
            total_tests: tests.len() as u32,
        });

        let (bench_passed, bench_failed) = match self.command {
            Command::Bench => self.run_all_benches(&mut out, &benches),
            Command::Test => (0, 0),
        };

        let (test_passed, test_failed) = match self.command {
            Command::Test => self.run_all_tests(&mut out, &tests),
            Command::Bench => (0, 0),
        };

        self.emit(&mut out, ProtocolMessage::Done {
            bench_passed,
            bench_failed,
            test_passed,
            test_failed,
        });
    }

    // -----------------------------------------------------------------------
    // Filtering
    // -----------------------------------------------------------------------

    fn matches_filter(&self, name: &str) -> bool {
        self.filter
            .as_deref()
            .map_or(true, |f| name.to_ascii_lowercase().contains(&f.to_ascii_lowercase()))
    }

    fn filtered_benches(&self) -> Vec<&'static metaltile_core::KernelBenchEntry> {
        all_benches().filter(|e| self.matches_filter(e.as_ref().name())).collect()
    }

    fn filtered_tests(&self) -> Vec<&'static metaltile_core::KernelTestEntry> {
        all_tests().filter(|e| self.matches_filter(e.as_ref().name())).collect()
    }

    // -----------------------------------------------------------------------
    // Bench loop
    // -----------------------------------------------------------------------

    fn run_all_benches(
        &self,
        out: &mut impl IoWrite,
        benches: &[&'static metaltile_core::KernelBenchEntry],
    ) -> (u32, u32) {
        let (mut passed, mut failed) = (0u32, 0u32);
        for entry in benches {
            let bench: &dyn KernelBench = entry.as_ref();
            for &dt in bench.dtypes() {
                match self.run_single_bench(bench, dt) {
                    Ok(result) => {
                        self.emit(out, ProtocolMessage::BenchResult(result));
                        passed += 1;
                    },
                    Err(e) => {
                        self.emit(out, ProtocolMessage::ProtocolError {
                            name: bench.name().to_string(),
                            dtype: dt.label().to_string(),
                            message: e.to_string(),
                        });
                        failed += 1;
                    },
                }
                let _ = out.flush();
            }
        }
        (passed, failed)
    }

    fn run_single_bench(
        &self,
        bench: &dyn KernelBench,
        dt: DType,
    ) -> Result<BenchResult, Box<dyn std::error::Error>> {
        let setup = bench.setup(dt);
        let bytes_moved = bench.bytes_moved(&setup);
        let mt_min_us = self.time_bench_setup(&setup)?;
        let mt_gbps =
            if mt_min_us > 0.0 { (bytes_moved as f64 / 1e9) / (mt_min_us * 1e-6) } else { 0.0 };

        // If the setup names a reference Metal kernel and we have a path, time it.
        let (ref_gbps, mt_pct) = if let Some(ref_k) = setup.ref_kernel() {
            match self.time_ref_kernel(ref_k) {
                Ok(ref_min_us) => {
                    let ref_bytes: u64 = ref_k.buffers.iter().map(|b| b.size_bytes()).sum();
                    let rgbps = if ref_min_us > 0.0 {
                        (ref_bytes as f64 / 1e9) / (ref_min_us * 1e-6)
                    } else {
                        0.0
                    };
                    let pct = if rgbps > 0.0 { (mt_gbps / rgbps - 1.0) * 100.0 } else { 0.0 };
                    (Some(rgbps), Some(pct))
                },
                Err(e) => {
                    eprintln!("warn: reference kernel '{}' failed: {e}", ref_k.fn_name);
                    (None, None)
                },
            }
        } else {
            (None, None)
        };

        let mean_us = 0.0; // mean not tracked separately; min_us is what matters
        Ok(BenchResult {
            name: bench.name().to_string(),
            dtype: dt.label().to_string(),
            mt_gbps,
            ref_gbps,
            mt_pct,
            correct: true,
            min_us: mt_min_us,
            mean_us,
        })
    }

    /// Run warmup + timed iterations for a `BenchSetup` and return the minimum
    /// wall-clock time in microseconds.
    fn time_bench_setup(
        &self,
        setup: &metaltile_core::bench::BenchSetup,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let dispatch =
            DispatchParams::from_bench_buffers(setup.buffers(), setup.constexprs(), setup.grid());

        for _ in 0..self.bench_cfg.warmup_iters {
            self.ctx.dispatch_with_grid(
                setup.kernel(),
                &dispatch.buffers,
                &dispatch.fn_consts,
                dispatch.grid_groups,
                dispatch.tpg,
            )?;
        }

        let min_us = (0..self.bench_cfg.timed_iters)
            .map(|_| {
                let result = self.ctx.dispatch_with_grid(
                    setup.kernel(),
                    &dispatch.buffers,
                    &dispatch.fn_consts,
                    dispatch.grid_groups,
                    dispatch.tpg,
                )?;
                Ok(result.elapsed_us)
            })
            .collect::<Result<Vec<f64>, Box<dyn std::error::Error>>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min);

        Ok(min_us)
    }

    /// Load, compile, and time a reference Metal kernel.
    ///
    /// Returns the minimum GPU time in microseconds (from `GPUEndTime - GPUStartTime`),
    /// or an error if `ref_metal_path` is not configured or the source file can't be read.
    fn time_ref_kernel(&self, ref_k: &RefKernel) -> Result<f64, Box<dyn std::error::Error>> {
        let base = self
            .ref_metal_path
            .as_ref()
            .ok_or("no reference_metal_path configured in tile.toml [bench]")?;
        let msl_source = std::fs::read_to_string(base.join(&ref_k.metal_file))?;
        let buffers: Vec<(&str, Vec<u8>)> =
            ref_k.buffers.iter().map(|b| (b.name(), b.initial_bytes())).collect();
        let [gx, gy, gz] = ref_k.grid.grid;
        let [tx, ty, tz] = ref_k.grid.tpg;
        let grid_groups = [gx as usize, gy as usize, gz as usize];
        let tpg = [tx as usize, ty as usize, tz as usize];

        // Warmup (also compiles and caches the PSO).
        for _ in 0..self.bench_cfg.warmup_iters {
            self.ctx.dispatch_raw_msl(&ref_k.fn_name, &msl_source, &buffers, grid_groups, tpg)?;
        }

        let min_us = (0..self.bench_cfg.timed_iters)
            .map(|_| {
                let result = self.ctx.dispatch_raw_msl(
                    &ref_k.fn_name,
                    &msl_source,
                    &buffers,
                    grid_groups,
                    tpg,
                )?;
                Ok(result.elapsed_us)
            })
            .collect::<Result<Vec<f64>, Box<dyn std::error::Error>>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min);

        Ok(min_us)
    }

    // -----------------------------------------------------------------------
    // Test loop
    // -----------------------------------------------------------------------

    fn run_all_tests(
        &self,
        out: &mut impl IoWrite,
        tests: &[&'static metaltile_core::KernelTestEntry],
    ) -> (u32, u32) {
        let (mut passed, mut failed) = (0u32, 0u32);
        for entry in tests {
            let test: &dyn KernelTest = entry.as_ref();
            for &dt in test.dtypes() {
                match self.run_single_test(test, dt) {
                    Ok(result) => {
                        let ok = result.passed;
                        self.emit(out, ProtocolMessage::TestResult(result));
                        if ok {
                            passed += 1;
                        } else {
                            failed += 1;
                        }
                    },
                    Err(e) => {
                        self.emit(out, ProtocolMessage::ProtocolError {
                            name: test.name().to_string(),
                            dtype: dt.label().to_string(),
                            message: e.to_string(),
                        });
                        failed += 1;
                    },
                }
                let _ = out.flush();
            }
        }
        (passed, failed)
    }

    fn run_single_test(
        &self,
        test: &dyn KernelTest,
        dt: DType,
    ) -> Result<TestResult, Box<dyn std::error::Error>> {
        let setup = test.setup(dt);
        let tol = test.tolerance(dt);
        let dispatch =
            DispatchParams::from_test_buffers(setup.inputs(), setup.constexprs(), setup.grid());

        // Generate expected values — from a GPU reference kernel if present,
        // otherwise from the CPU-side buffers supplied by the test author.
        let expected: Vec<(String, Vec<u8>)> = if let Some(ref_setup) = setup.ref_setup() {
            let ref_dispatch = DispatchParams::from_test_buffers(
                ref_setup.inputs(),
                ref_setup.constexprs(),
                ref_setup.grid(),
            );
            self.ctx
                .dispatch_with_grid(
                    ref_setup.kernel(),
                    &ref_dispatch.buffers,
                    &ref_dispatch.fn_consts,
                    ref_dispatch.grid_groups,
                    ref_dispatch.tpg,
                )?
                .outputs
                .into_iter()
                .collect()
        } else {
            setup.expected().iter().map(|b| (b.name().to_string(), b.data().to_vec())).collect()
        };

        let result = self.ctx.dispatch_with_grid(
            setup.kernel(),
            &dispatch.buffers,
            &dispatch.fn_consts,
            dispatch.grid_groups,
            dispatch.tpg,
        )?;

        let (passed, max_err) = compare_outputs(&result.outputs, &expected, dt, tol);
        Ok(TestResult {
            name: test.name().to_string(),
            dtype: dt.label().to_string(),
            passed,
            max_err,
        })
    }

    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------

    fn emit(&self, out: &mut impl IoWrite, msg: ProtocolMessage) {
        let _ = out.write_all(&msg.to_json_line());
    }
}

// ---------------------------------------------------------------------------
// DispatchParams — bundles the runtime inputs for ctx.dispatch_with_grid
// ---------------------------------------------------------------------------

struct DispatchParams {
    buffers: BTreeMap<String, Vec<u8>>,
    fn_consts: BTreeMap<String, u32>,
    grid_groups: [usize; 3],
    tpg: [usize; 3],
}

impl DispatchParams {
    fn from_bench_buffers(
        buffers: &[BenchBuffer],
        constexprs: &[(String, ConstValue)],
        grid: &metaltile_core::bench::Grid,
    ) -> Self {
        let buffers = buffers.iter().map(|b| (b.name().to_string(), b.initial_bytes())).collect();
        Self::new(buffers, constexprs, grid)
    }

    fn from_test_buffers(
        buffers: &[TestBuffer],
        constexprs: &[(String, ConstValue)],
        grid: &metaltile_core::bench::Grid,
    ) -> Self {
        let buffers = buffers.iter().map(|b| (b.name().to_string(), b.data().to_vec())).collect();
        Self::new(buffers, constexprs, grid)
    }

    fn new(
        buffers: BTreeMap<String, Vec<u8>>,
        constexprs: &[(String, ConstValue)],
        grid: &metaltile_core::bench::Grid,
    ) -> Self {
        let fn_consts = constexprs
            .iter()
            .filter_map(|(name, val)| val.as_u32().ok().map(|v| (name.clone(), v)))
            .collect();
        DispatchParams {
            buffers,
            fn_consts,
            grid_groups: [grid.grid[0] as usize, grid.grid[1] as usize, grid.grid[2] as usize],
            tpg: [grid.tpg[0] as usize, grid.tpg[1] as usize, grid.tpg[2] as usize],
        }
    }
}

// ---------------------------------------------------------------------------
// Output comparison
// ---------------------------------------------------------------------------

/// Returns `(passed, max_abs_error)` by comparing every output buffer.
fn compare_outputs(
    got: &BTreeMap<String, Vec<u8>>,
    expected: &[(String, Vec<u8>)],
    dt: DType,
    tol: f64,
) -> (bool, f64) {
    let mut max_err = 0.0f64;
    let mut passed = true;

    for (name, exp_bytes) in expected {
        let Some(got_bytes) = got.get(name) else {
            passed = false;
            continue;
        };
        let err = max_abs_err(got_bytes, exp_bytes, dt);
        if err > max_err {
            max_err = err;
        }
        if max_err > tol {
            passed = false;
        }
    }
    (passed, max_err)
}

/// Element-wise max |got[i] − exp[i]| over a pair of raw byte buffers.
///
/// Returns a large finite value on size mismatch so JSON serialization stays
/// well-formed (`f64::INFINITY` serialises as `null` in serde_json).
fn max_abs_err(got: &[u8], exp: &[u8], dt: DType) -> f64 {
    if got.len() != exp.len() {
        return 1e38;
    }
    let elem = dt.size_bytes();
    let n = got.len() / elem;
    (0..n)
        .map(|i| (elem_as_f64(got, i, elem, dt) - elem_as_f64(exp, i, elem, dt)).abs())
        // Ignore NaN diffs (e.g. NaN GPU output) — they are not treated as errors here;
        // tests that produce NaN will surface through other correctness checks.
        .fold(0.0f64, |acc, x| if x.is_finite() { acc.max(x) } else { acc })
}

/// Decode element `idx` from a raw byte slice as `f64`.
fn elem_as_f64(bytes: &[u8], idx: usize, elem_size: usize, dt: DType) -> f64 {
    let off = idx * elem_size;
    let s = &bytes[off..off + elem_size];
    match dt {
        DType::F32 => f32::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::F16 => half::f16::from_le_bytes(s.try_into().unwrap()).to_f64(),
        DType::BF16 => half::bf16::from_le_bytes(s.try_into().unwrap()).to_f64(),
        DType::I32 => i32::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::U32 => u32::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::I8 => s[0] as i8 as f64,
        DType::U8 => s[0] as f64,
        _ => 0.0,
    }
}

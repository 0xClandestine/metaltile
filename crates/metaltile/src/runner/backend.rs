//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Non-Metal backend dispatch for `__tile_runner` — `--backend cuda|hip|vulkan`.
//!
//! Routes the same `#[test_kernel]` / `#[bench]` inventories the Metal
//! harness uses through the device runtimes' generic `run_kernel` /
//! `bench_kernel` entry points, streaming the same `ProtocolMessage`s. Each
//! backend is compiled in via its cargo feature; requesting one the runner
//! wasn't built with produces a single actionable error instead of a panic.

use metaltile_core::protocol::ProtocolMessage;

use crate::runner::{args::RunnerArgs, emit::emit_stdout};

/// The backend explicitly requested via `--backend`, normalised. `None`
/// means the platform default (Metal) — including an explicit
/// `--backend metal`.
pub(crate) fn requested(args: &RunnerArgs) -> Option<&str> {
    match args.backend.as_deref() {
        None | Some("metal") => None,
        Some(other) => Some(other),
    }
}

/// Error text for a backend this runner binary cannot serve.
fn unavailable_message(backend: &str) -> String {
    match backend {
        "cuda" | "hip" | "vulkan" => format!(
            "runner was built without the `{backend}` feature — rebuild the \
             __tile_runner binary with `--features {backend}`"
        ),
        other => format!("unknown backend '{other}' (expected metal, cuda, hip, or vulkan)"),
    }
}

fn emit_backend_error(message: String) -> bool {
    emit_stdout(&ProtocolMessage::Start {
        runner_version: env!("CARGO_PKG_VERSION").into(),
        command: "backend".into(),
        total: 0,
        device: None,
    });
    emit_stdout(&ProtocolMessage::ProtocolError {
        name: "backend".into(),
        dtype: "".into(),
        message,
    });
    emit_stdout(&ProtocolMessage::Done {
        ok: false,
        bench_passed: 0,
        bench_failed: 0,
        test_passed: 0,
        test_failed: 0,
        test_skipped: 0,
    });
    false
}

// ── Fallback when no backend feature is compiled in ──────────────────────

#[cfg(not(any(feature = "cuda", feature = "hip", feature = "vulkan")))]
pub(crate) fn run_test(_args: &RunnerArgs, backend: &str) -> bool {
    emit_backend_error(unavailable_message(backend))
}

#[cfg(not(any(feature = "cuda", feature = "hip", feature = "vulkan")))]
pub(crate) fn run_bench(_args: &RunnerArgs, backend: &str) -> bool {
    emit_backend_error(unavailable_message(backend))
}

// ── Real dispatch (any backend feature on) ────────────────────────────────

#[cfg(any(feature = "cuda", feature = "hip", feature = "vulkan"))]
pub(crate) use enabled::{run_bench, run_test};

#[cfg(any(feature = "cuda", feature = "hip", feature = "vulkan"))]
mod enabled {
    use std::collections::BTreeMap;

    use metaltile_core::{
        DType,
        ir::Kernel,
        protocol::{BenchResult, ProtocolMessage, TestResult},
    };
    use metaltile_runtime::MetalTileError;

    use super::{emit_backend_error, unavailable_message};
    use crate::{
        harness::{
            bench::KernelBench,
            registry::{all_benches, all_tests},
        },
        runner::{
            args::RunnerArgs,
            emit::emit_stdout,
            gpu::{BENCH_ITERS, BENCH_WARMUP, BenchStats},
            harness::{
                NameFilter,
                RunnerHarness,
                bench_display_name,
                kernel_family,
                read_raw_f32,
                suffixed_kernel_name,
            },
        },
    };

    /// One handle over the feature-gated device runtimes.
    enum BackendDevice {
        #[cfg(feature = "cuda")]
        Cuda(Box<metaltile_runtime::CudaDevice>),
        #[cfg(feature = "hip")]
        Hip(Box<metaltile_runtime::HipDevice>),
        #[cfg(feature = "vulkan")]
        Vulkan(Box<metaltile_runtime::VulkanDevice>),
    }

    impl BackendDevice {
        /// Initialise the requested backend. `Ok(None)` = feature compiled
        /// in but no device present; `Err` = unknown name, feature not
        /// compiled in, or device init failure.
        fn create(backend: &str) -> Result<Option<Self>, String> {
            match backend {
                "cuda" => {
                    #[cfg(feature = "cuda")]
                    {
                        metaltile_runtime::CudaDevice::create()
                            .map(|d| d.map(|d| BackendDevice::Cuda(Box::new(d))))
                            .map_err(|e| format!("CUDA init failed: {e}"))
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        Err(unavailable_message(backend))
                    }
                },
                "hip" => {
                    #[cfg(feature = "hip")]
                    {
                        metaltile_runtime::HipDevice::create()
                            .map(|d| d.map(|d| BackendDevice::Hip(Box::new(d))))
                            .map_err(|e| format!("HIP init failed: {e}"))
                    }
                    #[cfg(not(feature = "hip"))]
                    {
                        Err(unavailable_message(backend))
                    }
                },
                "vulkan" => {
                    #[cfg(feature = "vulkan")]
                    {
                        metaltile_runtime::VulkanDevice::create()
                            .map(|d| d.map(|d| BackendDevice::Vulkan(Box::new(d))))
                            .map_err(|e| format!("Vulkan init failed: {e}"))
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        Err(unavailable_message(backend))
                    }
                },
                other => Err(unavailable_message(other)),
            }
        }

        fn name(&self) -> String {
            match self {
                #[cfg(feature = "cuda")]
                BackendDevice::Cuda(d) => {
                    let (maj, min) = d.compute_capability();
                    format!("NVIDIA CUDA cc{maj}.{min}")
                },
                #[cfg(feature = "hip")]
                BackendDevice::Hip(d) => format!("{} ({})", d.name(), d.gfx_arch()),
                #[cfg(feature = "vulkan")]
                BackendDevice::Vulkan(d) => d.name().to_string(),
            }
        }

        fn run_kernel(
            &self,
            kernel: &Kernel,
            buffers: &BTreeMap<String, Vec<u8>>,
            grid: [u32; 3],
            block: [u32; 3],
        ) -> Result<BTreeMap<String, Vec<u8>>, MetalTileError> {
            match self {
                #[cfg(feature = "cuda")]
                BackendDevice::Cuda(d) => d.run_kernel(kernel, buffers, grid, block),
                #[cfg(feature = "hip")]
                BackendDevice::Hip(d) => d.run_kernel(kernel, buffers, grid, block),
                #[cfg(feature = "vulkan")]
                BackendDevice::Vulkan(d) => d.run_kernel(kernel, buffers, grid, block),
            }
        }

        /// Event-timed per-launch samples in µs. `Ok(None)` when the
        /// backend has no GPU-timestamp bench path yet (Vulkan).
        fn bench_kernel(
            &self,
            kernel: &Kernel,
            buffers: &BTreeMap<String, Vec<u8>>,
            grid: [u32; 3],
            block: [u32; 3],
            warmup: u32,
            iters: u32,
        ) -> Result<Option<Vec<f64>>, MetalTileError> {
            match self {
                #[cfg(feature = "cuda")]
                BackendDevice::Cuda(d) =>
                    d.bench_kernel(kernel, buffers, grid, block, warmup, iters).map(Some),
                #[cfg(feature = "hip")]
                BackendDevice::Hip(d) =>
                    d.bench_kernel(kernel, buffers, grid, block, warmup, iters).map(Some),
                #[cfg(feature = "vulkan")]
                BackendDevice::Vulkan(_) => {
                    // Consume the launch params — Vulkan has no event-timed
                    // bench path yet, but the signature stays uniform.
                    let _ = (kernel, buffers, grid, block, warmup, iters);
                    Ok(None)
                },
            }
        }
    }

    /// UNSUPPORTED is decided on the TYPED error, not message sniffing:
    /// `Codegen(UnsupportedOp)` covers every codegen coverage gap
    /// (cooperative MMA, ops not wired yet); `DeviceCapability` covers
    /// kernels the codegen emits but the arch physically cannot run. Both
    /// become protocol skips — anything else on a kernel we claim to
    /// support stays a hard error.
    fn is_unsupported(e: &MetalTileError) -> bool {
        use metaltile_codegen::error::Error as CodegenError;
        matches!(
            e,
            MetalTileError::Codegen(CodegenError::UnsupportedOp(_))
                | MetalTileError::DeviceCapability(_)
        )
    }

    /// Non-finite-aware element compare. Bitwise-equal values (covers equal
    /// infinities) and NaN-on-both-sides count as agreement; any one-sided
    /// NaN/inf maps to +inf so garbage fails loudly instead of `f32::max`
    /// silently discarding NaN operands.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                if x == y || (x.is_nan() && y.is_nan()) {
                    0.0
                } else {
                    let d = (x - y).abs();
                    if d.is_nan() { f32::INFINITY } else { d }
                }
            })
            .fold(0.0f32, f32::max)
    }

    /// `tile test --backend <name>` — run the `#[test_kernel]` corpus on a
    /// non-Metal device via its generic `run_kernel`, comparing against the
    /// same CPU oracles the Metal harness uses.
    pub(crate) fn run_test(args: &RunnerArgs, backend: &str) -> bool {
        let name_filter = match NameFilter::from_args(args) {
            Ok(f) => f,
            Err(msg) => return emit_backend_error(msg),
        };
        let dev = match BackendDevice::create(backend) {
            Ok(Some(d)) => d,
            Ok(None) => return emit_backend_error(format!("no {backend} device found")),
            Err(msg) => return emit_backend_error(msg),
        };

        let entries: Vec<_> = all_tests()
            .filter(|e| {
                name_filter.matches(e.test().name(), e.test().name(), kernel_family(e.file()))
            })
            .collect();

        let dtypes = RunnerHarness::dtype_list(args);
        let allowed = |dt: &DType| dtypes.as_ref().is_none_or(|ds| ds.contains(dt));
        let total: u32 = entries
            .iter()
            .map(|e| e.test().dtypes().iter().filter(|dt| allowed(dt)).count() as u32)
            .sum();

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "test".into(),
            total,
            device: Some(dev.name()),
        });

        let (mut passed, mut failed, mut skipped) = (0u32, 0u32, 0u32);

        for entry in entries {
            let test = entry.test();
            for &dt in test.dtypes() {
                if !allowed(&dt) {
                    continue;
                }
                let setup = test.setup(dt);
                let tol = test.tolerance(dt);
                let name = test.name().to_string();
                let dtype_str = format!("{dt:?}").to_lowercase();

                // GPU-vs-GPU reference setups need two dispatches; not
                // wired on the non-Metal paths yet.
                if setup.ref_setup().is_some() {
                    skipped += 1;
                    emit_stdout(&ProtocolMessage::TestResult(TestResult {
                        name,
                        dtype: dtype_str,
                        passed: false,
                        max_err: 0.0,
                        skipped: true,
                    }));
                    continue;
                }

                let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
                for inp in setup.inputs() {
                    buffers.insert(inp.name().to_string(), inp.data().to_vec());
                }
                for (k, v) in setup.constexprs() {
                    buffers.insert(k.clone(), v.to_le_bytes());
                }

                let grid = setup.grid();
                match dev.run_kernel(setup.kernel(), &buffers, grid.grid, grid.tpg) {
                    Ok(outputs) => {
                        let mut worst = 0.0f32;
                        for exp in setup.expected() {
                            let Some(got_bytes) = outputs.get(exp.name()) else {
                                worst = f32::INFINITY;
                                break;
                            };
                            let n = exp.len();
                            let got = read_raw_f32(got_bytes, exp.dtype(), n);
                            let want = read_raw_f32(exp.data(), exp.dtype(), n);
                            worst = worst.max(max_abs_diff(&got, &want));
                        }
                        let ok = (worst as f64) <= tol;
                        if ok {
                            passed += 1;
                        } else {
                            failed += 1;
                        }
                        emit_stdout(&ProtocolMessage::TestResult(TestResult {
                            name,
                            dtype: dtype_str,
                            passed: ok,
                            max_err: worst as f64,
                            skipped: false,
                        }));
                    },
                    Err(e) if is_unsupported(&e) => {
                        skipped += 1;
                        emit_stdout(&ProtocolMessage::TestResult(TestResult {
                            name,
                            dtype: dtype_str,
                            passed: false,
                            max_err: 0.0,
                            skipped: true,
                        }));
                    },
                    Err(e) => {
                        failed += 1;
                        emit_stdout(&ProtocolMessage::ProtocolError {
                            name,
                            dtype: dtype_str,
                            message: e.to_string(),
                        });
                    },
                }
            }
        }

        emit_stdout(&ProtocolMessage::Done {
            ok: failed == 0,
            bench_passed: 0,
            bench_failed: 0,
            test_passed: passed,
            test_failed: failed,
            test_skipped: skipped,
        });
        failed == 0
    }

    /// `tile bench --backend <name>` — event-timed kernel throughput on a
    /// non-Metal device. No reference A/B (the MSL reference kernels are
    /// Metal-only); correctness stays `tile test`'s job.
    pub(crate) fn run_bench(args: &RunnerArgs, backend: &str) -> bool {
        let warmup = args.warmup.unwrap_or(BENCH_WARMUP) as u32;
        let iters = args.iters.unwrap_or(BENCH_ITERS) as u32;

        let name_filter = match NameFilter::from_args(args) {
            Ok(f) => f,
            Err(msg) => return emit_backend_error(msg),
        };
        let dev = match BackendDevice::create(backend) {
            Ok(Some(d)) => d,
            Ok(None) => return emit_backend_error(format!("no {backend} device found")),
            Err(msg) => return emit_backend_error(msg),
        };

        let dtype_restrict = RunnerHarness::dtype_list(args);
        let mut work: Vec<(&'static dyn KernelBench, DType, &'static str)> = Vec::new();
        for entry in all_benches() {
            let bench = entry.bench();
            let family = kernel_family(entry.file());
            for &dt in bench.dtypes() {
                if dtype_restrict.as_ref().is_none_or(|ds| ds.contains(&dt)) {
                    work.push((bench, dt, family));
                }
            }
        }
        if !name_filter.is_empty() {
            work.retain(|&(bench, dt, family)| {
                let name = bench_display_name(bench, dt);
                name_filter.matches(&name, bench.name(), family)
            });
        }

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "bench".into(),
            total: work.len() as u32,
            device: Some(dev.name()),
        });

        let (mut passed, mut failed, mut skipped) = (0u32, 0u32, 0u32);

        for (bench, dt, family) in work {
            let setup = bench.setup(dt);
            let bytes_moved = bench.bytes_moved(&setup);
            let kernel = setup.kernel();
            let name = suffixed_kernel_name(&kernel.name, dt);
            let dtype_str = format!("{dt:?}").to_lowercase();

            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for buf in setup.buffers() {
                buffers.insert(buf.name().to_string(), buf.initial_bytes());
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes());
            }

            let grid = setup.grid();
            match dev.bench_kernel(kernel, &buffers, grid.grid, grid.tpg, warmup, iters) {
                Ok(Some(samples)) if !samples.is_empty() => {
                    let stats = BenchStats::from_samples(samples);
                    // Effective bandwidth from steady-state (min) latency.
                    let mt_gbps = if stats.min_us > 0.0 {
                        bytes_moved as f64 / (stats.min_us * 1_000.0)
                    } else {
                        0.0
                    };
                    // Same fallback shape rule as the Metal path: largest
                    // buffer element count.
                    let shape = setup.shape_label().map(|s| s.to_string()).unwrap_or_else(|| {
                        let n = setup.buffers().iter().map(|b| b.len()).max().unwrap_or(0);
                        let suffix = if n >= 1 << 20 && n % (1 << 20) == 0 {
                            format!("{}M", n >> 20)
                        } else if n >= 1 << 10 && n % (1 << 10) == 0 {
                            format!("{}K", n >> 10)
                        } else {
                            n.to_string()
                        };
                        format!("N={suffix}")
                    });
                    passed += 1;
                    emit_stdout(&ProtocolMessage::BenchResult(BenchResult {
                        name,
                        group: family.to_string(),
                        dtype: dtype_str,
                        shape,
                        mt_gbps,
                        ref_gbps: None,
                        mt_pct: None,
                        correct: true,
                        min_us: stats.min_us,
                        mean_us: stats.mean_us,
                        profile: None,
                    }));
                },
                Ok(Some(_)) => {
                    skipped += 1;
                },
                Ok(None) => {
                    failed += 1;
                    emit_stdout(&ProtocolMessage::ProtocolError {
                        name,
                        dtype: dtype_str,
                        message: format!(
                            "kernel-time benchmarking is not wired for the {backend} \
                             backend yet (no GPU-timestamp path) — use `tile test \
                             --backend {backend}` for correctness"
                        ),
                    });
                },
                Err(e) if is_unsupported(&e) => {
                    skipped += 1;
                },
                Err(e) => {
                    failed += 1;
                    emit_stdout(&ProtocolMessage::ProtocolError {
                        name,
                        dtype: dtype_str,
                        message: e.to_string(),
                    });
                },
            }
        }

        emit_stdout(&ProtocolMessage::Done {
            ok: failed == 0,
            bench_passed: passed,
            bench_failed: failed,
            test_passed: 0,
            test_failed: 0,
            test_skipped: skipped,
        });
        failed == 0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn unknown_backend_is_an_error() {
            let e = BackendDevice::create("opencl").err().unwrap();
            assert!(e.contains("unknown backend"));
        }

        #[test]
        fn nan_aware_compare() {
            assert_eq!(max_abs_diff(&[f32::NAN], &[f32::NAN]), 0.0);
            assert_eq!(max_abs_diff(&[f32::NEG_INFINITY], &[f32::NEG_INFINITY]), 0.0);
            assert_eq!(max_abs_diff(&[f32::NAN], &[1.0]), f32::INFINITY);
            assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.0, 2.5]), 0.5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_and_none_mean_default() {
        let mut a = RunnerArgs::parse(vec!["bench".into()]).unwrap();
        assert!(requested(&a).is_none());
        a.backend = Some("metal".into());
        assert!(requested(&a).is_none());
        a.backend = Some("cuda".into());
        assert_eq!(requested(&a), Some("cuda"));
    }

    #[test]
    fn unavailable_message_distinguishes_unknown_from_missing_feature() {
        assert!(unavailable_message("cuda").contains("--features cuda"));
        assert!(unavailable_message("opencl").contains("unknown backend"));
    }
}

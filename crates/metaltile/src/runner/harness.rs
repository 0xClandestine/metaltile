//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `RunnerHarness` — orchestrates bench / test / build / inspect in the
//! `__tile_runner` subprocess and streams [`ProtocolMessage`]s to stdout.
//!
//! This is the only place that calls the `inventory` registries. The CLI
//! process never imports this module — it only reads the JSON lines.

use metaltile_codegen::{
    emit as codegen_emit,
    generator_for_mode,
    msl::MslGenerator,
    passes::{PassStats, PipelineBuilder, run_passes_with_stats},
};
use metaltile_core::{
    DType,
    ir::{Kernel, ParamKind},
    protocol::{ArtifactKind, BenchResult, BuildError, BuildResult, ProtocolMessage, TestResult},
};

use crate::{
    harness::{
        bench::{BenchSetup, KernelBench, RefKernel},
        registry::{all_benches, all_kernels, all_tests},
        test::TestSetup,
    },
    runner::{
        args::{RunnerArgs, RunnerCommand},
        emit::emit_stdout,
        gpu::{BENCH_ITERS, BENCH_WARMUP, GpuBuffer, GpuRunner, bench_gbps_with, read_typed},
    },
};

/// Entry-point for the `__tile_runner` subprocess.
///
/// Parses the `RunnerArgs`, initialises the GPU runner (if needed), runs the
/// requested suite, and streams `ProtocolMessage` JSON lines to stdout.
pub struct RunnerHarness;

impl RunnerHarness {
    /// Run the full harness.  Returns `true` if every item passed / compiled.
    pub fn run(args: &RunnerArgs) -> bool {
        match args.command {
            RunnerCommand::Bench => Self::run_bench(args),
            RunnerCommand::Test => Self::run_test(args),
            RunnerCommand::Build => Self::run_build(args),
            RunnerCommand::Inspect => Self::run_inspect(args),
        }
    }

    // ── bench ─────────────────────────────────────────────────────────────────

    fn run_bench(args: &RunnerArgs) -> bool {
        let warmup = args.warmup.unwrap_or(BENCH_WARMUP);
        let iters = args.iters.unwrap_or(BENCH_ITERS);

        let entries: Vec<_> = all_benches()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.bench().name().contains(f)))
            .collect();

        let dtypes = Self::dtype_list(args);
        let total = (entries.len() * dtypes.len()) as u32;

        let runner = match GpuRunner::new() {
            Ok(r) => r,
            Err(e) => {
                emit_stdout(&ProtocolMessage::Start {
                    runner_version: env!("CARGO_PKG_VERSION").into(),
                    command: "bench".into(),
                    total,
                    device: None,
                });
                emit_stdout(&ProtocolMessage::ProtocolError {
                    name: "GpuRunner".into(),
                    dtype: "".into(),
                    message: format!("GPU init failed: {e}"),
                });
                emit_stdout(&ProtocolMessage::Done {
                    ok: false,
                    bench_passed: 0,
                    bench_failed: total,
                    test_passed: 0,
                    test_failed: 0,
                    test_skipped: 0,
                });
                return false;
            },
        };

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "bench".into(),
            total,
            device: Some(runner.device_name.clone()),
        });

        let mut passed = 0u32;
        let mut failed = 0u32;

        for entry in entries {
            let bench = entry.bench();
            for &dt in &dtypes {
                if let Some(result) = run_one_bench(&runner, bench, dt, warmup, iters) {
                    if result.correct {
                        passed += 1;
                    } else {
                        failed += 1;
                    }
                    emit_stdout(&ProtocolMessage::BenchResult(result));
                } else {
                    failed += 1;
                    emit_stdout(&ProtocolMessage::ProtocolError {
                        name: entry.bench().name().into(),
                        dtype: format!("{dt:?}").to_lowercase(),
                        message: "bench failed (compile error or GPU unavailable)".into(),
                    });
                }
            }
        }

        emit_stdout(&ProtocolMessage::Done {
            ok: failed == 0,
            bench_passed: passed,
            bench_failed: failed,
            test_passed: 0,
            test_failed: 0,
            test_skipped: 0,
        });
        failed == 0
    }

    // ── test ──────────────────────────────────────────────────────────────────

    fn run_test(args: &RunnerArgs) -> bool {
        use rayon::prelude::*;

        let entries: Vec<_> = all_tests()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.test().name().contains(f)))
            .collect();

        let dtypes = Self::dtype_list(args);
        let total: u32 = entries
            .iter()
            .map(|e| e.test().dtypes().iter().filter(|dt| dtypes.contains(dt)).count() as u32)
            .sum();

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "test".into(),
            total,
            device: None,
        });

        let ctx = match metaltile_runtime::Context::new() {
            Ok(c) => c,
            Err(e) => {
                emit_stdout(&ProtocolMessage::ProtocolError {
                    name: "Context".into(),
                    dtype: "".into(),
                    message: format!("runtime init: {e}"),
                });
                emit_stdout(&ProtocolMessage::Done {
                    ok: false,
                    bench_passed: 0,
                    bench_failed: 0,
                    test_passed: 0,
                    test_failed: total,
                    test_skipped: 0,
                });
                return false;
            },
        };

        // Phase 1 (parallel): build all TestSetups on the CPU.
        // `test.setup(dt)` computes expected output buffers without touching the
        // GPU, so all (entry × dtype) pairs can run concurrently via rayon.
        // Only run each test for its registered dtypes, intersected with the
        // CLI dtype filter — a test with dtypes=[f32] must not run with f16.
        let work: Vec<Vec<(String, DType, TestSetup, f64)>> = entries
            .par_iter()
            .map(|entry| {
                let test = entry.test();
                test.dtypes()
                    .iter()
                    .filter(|dt| dtypes.contains(dt))
                    .map(|&dt| {
                        let setup = test.setup(dt);
                        let tol = test.tolerance(dt);
                        (test.name().to_string(), dt, setup, tol)
                    })
                    .collect()
            })
            .collect();

        // Phase 2 (serial): GPU dispatch + comparison.
        // Metal Context is not Send, so all dispatches run on the main thread.
        let mut passed = 0u32;
        let mut failed = 0u32;
        let mut skipped = 0u32;

        for group in &work {
            for (name, dt, setup, tol) in group {
                match run_one_test_with_setup(&ctx, setup, *tol, name, *dt) {
                    Ok(result) => {
                        if result.passed {
                            passed += 1;
                        } else {
                            failed += 1;
                        }
                        emit_stdout(&ProtocolMessage::TestResult(result));
                    },
                    Err(msg) => {
                        // Cooperative-tensor (NAX/MPP matmul2d) kernels fail to
                        // build on macOS <26.5 Metal toolchains with
                        // "unsupported deferred-static-alloca-size". Skip rather
                        // than fail — they'll auto-enable on qualifying runners.
                        if setup.kernel().requires_cooperative_tensors()
                            && msg.contains("deferred-static-alloca")
                        {
                            skipped += 1;
                            emit_stdout(&ProtocolMessage::TestResult(TestResult {
                                name: name.clone(),
                                dtype: format!("{dt:?}").to_lowercase(),
                                passed: false,
                                max_err: 0.0,
                                skipped: true,
                            }));
                        } else {
                            failed += 1;
                            emit_stdout(&ProtocolMessage::ProtocolError {
                                name: name.clone(),
                                dtype: format!("{dt:?}").to_lowercase(),
                                message: msg,
                            });
                        }
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

    // ── build ─────────────────────────────────────────────────────────────────

    fn run_build(args: &RunnerArgs) -> bool {
        use std::{collections::BTreeSet, path::PathBuf};

        use rayon::prelude::*;

        // --time-passes: pure-CPU pass timing, prints table directly to stdout
        // (no JSON protocol — CLI uses run() with inherited stdout for this path).
        if args.time_passes {
            return Self::run_time_passes(args);
        }

        let sdk = args.sdk.as_deref().unwrap_or("macosx");

        // Parse --emit kinds; "metallib" implies "msl" (needs .metal files on disk).
        let emit_kinds: BTreeSet<&str> = {
            let mut s = BTreeSet::new();
            if let Some(raw) = args.emit.as_deref() {
                for tok in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                    match tok {
                        "msl" => {
                            s.insert("msl");
                        },
                        "metallib" => {
                            s.insert("msl");
                            s.insert("metallib");
                        },
                        "swift" => {
                            s.insert("swift");
                        },
                        "ir" => {
                            s.insert("ir");
                        },
                        "all" => {
                            s.insert("msl");
                            s.insert("metallib");
                            s.insert("swift");
                            s.insert("ir");
                        },
                        other => eprintln!("[runner] unknown --emit kind '{other}', skipping"),
                    }
                }
            }
            s
        };
        let out_root = args.out_dir.as_ref().map(PathBuf::from);
        let kernels_dir = out_root.as_ref().map(|r| r.join("Resources").join("kernels"));

        // Collect unique kernels from the bench registry.  Each bench's setup
        // carries the kernel IR (with mode applied), the dtype set, and the
        // threadgroup geometry — everything emission needs.  Keyed by the
        // kernel's generic name; multiple benches for the same kernel union
        // their dtype sets.
        let mut kernel_map: std::collections::BTreeMap<
            String,
            (&'static dyn KernelBench, Vec<DType>),
        > = std::collections::BTreeMap::new();
        for entry in all_benches()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.bench().name().contains(f)))
        {
            let bench = entry.bench();
            let Some(&first_dt) = bench.dtypes().first() else { continue };
            let base_name = bench.setup(first_dt).kernel().name.to_string();
            let e = kernel_map.entry(base_name).or_insert((bench, Vec::new()));
            for &dt in bench.dtypes() {
                if !e.1.contains(&dt) {
                    e.1.push(dt);
                }
            }
        }

        // Dtype filter from --dtype (comma-separated).
        let dtype_filter: Option<Vec<DType>> = args
            .dtype
            .as_ref()
            .map(|s| s.split(',').filter_map(|t| parse_dtype(t.trim())).collect());

        let total = kernel_map.len() as u32;
        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "build".into(),
            total,
            device: None,
        });

        // --names: emit BuildResult messages with kernel names/dtypes, no compilation.
        if args.names {
            for (name, (_, dtypes)) in &kernel_map {
                let ok: Vec<String> = match &dtype_filter {
                    Some(df) => dtypes
                        .iter()
                        .filter(|dt| df.contains(dt))
                        .map(|dt| format!("{dt:?}").to_lowercase())
                        .collect(),
                    None => dtypes.iter().map(|dt| format!("{dt:?}").to_lowercase()).collect(),
                };
                emit_stdout(&ProtocolMessage::BuildResult(BuildResult {
                    name: name.clone(),
                    dtypes_ok: ok,
                    dtypes_err: Vec::new(),
                }));
            }
            emit_stdout(&ProtocolMessage::Done {
                ok: true,
                bench_passed: 0,
                bench_failed: 0,
                test_passed: 0,
                test_failed: 0,
                test_skipped: 0,
            });
            return true;
        }

        // Create output directory if needed.
        if let Some(dir) = &kernels_dir {
            let _ = std::fs::create_dir_all(dir);
        }

        struct WorkItem {
            name: String,
            bench: &'static dyn KernelBench,
            dtypes: Vec<DType>,
            n_dtypes: usize,
        }
        let work_items: Vec<WorkItem> = kernel_map
            .into_iter()
            .map(|(name, (bench, dtypes))| {
                let dtypes_to_check = match &dtype_filter {
                    Some(df) => dtypes.iter().filter(|dt| df.contains(dt)).copied().collect(),
                    None => dtypes.clone(),
                };
                let n_dtypes = dtypes.len();
                WorkItem { name, bench, dtypes: dtypes_to_check, n_dtypes }
            })
            .collect();

        struct KernelResult {
            name: String,
            dtypes_ok: Vec<String>,
            dtypes_err: Vec<BuildError>,
            emitted_kernels: Vec<Kernel>,
            emitted_paths: Vec<PathBuf>,
        }

        // Compile all kernels in parallel (each xcrun call is ~50-200 ms and
        // fully independent, so parallelism across N cores gives a near-N× speedup).
        let par_results: Vec<KernelResult> = work_items
            .par_iter()
            .map(|item| {
                let mut dtypes_ok = Vec::new();
                let mut dtypes_err = Vec::new();
                let mut item_kernels = Vec::new();
                let mut item_paths = Vec::new();

                for &dt in &item.dtypes {
                    let setup = item.bench.setup(dt);
                    let mut k = setup.kernel().clone();
                    let mode = k.mode;
                    let suffix = codegen_emit::dtype_suffix(dt);
                    k.name = if item.n_dtypes == 1 && item.name.ends_with(&format!("_{suffix}")) {
                        item.name.clone()
                    } else {
                        format!("{}_{suffix}", item.name)
                    };
                    let expected_tpg = Some(setup.grid().tpg[0]);
                    let generator = generator_for_mode(mode, expected_tpg);

                    let _msl = match generator.generate(&k) {
                        Ok(msl) => msl,
                        Err(e) => {
                            dtypes_err.push(BuildError {
                                dtype: format!("{dt:?}").to_lowercase(),
                                message: e.to_string(),
                            });
                            continue;
                        },
                    };

                    // Metal compile-check via xcrun (macOS only).
                    #[cfg(target_os = "macos")]
                    if let Err(e) = build_metal_compile_check(&_msl, &k.name, sdk) {
                        dtypes_err.push(BuildError {
                            dtype: format!("{dt:?}").to_lowercase(),
                            message: e,
                        });
                        continue;
                    }

                    dtypes_ok.push(format!("{dt:?}").to_lowercase());

                    if let Some(dir) = &kernels_dir
                        && emit_kinds.contains("msl")
                    {
                        match codegen_emit::write_msl(&k, dir, &generator) {
                            Ok(path) => item_paths.push(path),
                            Err(e) => {
                                dtypes_err.push(BuildError {
                                    dtype: format!("{dt:?}").to_lowercase(),
                                    message: e.to_string(),
                                });
                                dtypes_ok.pop();
                                continue;
                            },
                        }
                    }
                    if !emit_kinds.is_empty() {
                        item_kernels.push(k);
                    }
                }

                KernelResult {
                    name: item.name.clone(),
                    dtypes_ok,
                    dtypes_err,
                    emitted_kernels: item_kernels,
                    emitted_paths: item_paths,
                }
            })
            .collect();

        let mut any_err = false;
        let mut all_emitted_kernels: Vec<Kernel> = Vec::new();
        let mut all_emitted_paths: Vec<PathBuf> = Vec::new();

        for result in par_results {
            any_err |= !result.dtypes_err.is_empty();
            emit_stdout(&ProtocolMessage::BuildResult(BuildResult {
                name: result.name,
                dtypes_ok: result.dtypes_ok,
                dtypes_err: result.dtypes_err,
            }));
            all_emitted_kernels.extend(result.emitted_kernels);
            all_emitted_paths.extend(result.emitted_paths);
        }

        // ── Emit pass ─────────────────────────────────────────────────────────
        if let Some(out) = &out_root {
            let resources_dir = out.join("Resources");
            let generated_dir = out.join("Generated");

            if emit_kinds.contains("ir") {
                let manifest_path = resources_dir.join("manifest.json");
                let _ = std::fs::create_dir_all(&resources_dir);
                match codegen_emit::write_manifest(&all_emitted_kernels, &manifest_path) {
                    Ok(()) => emit_stdout(&ProtocolMessage::Artifact {
                        kind: ArtifactKind::Ir,
                        path: manifest_path.to_string_lossy().into_owned(),
                    }),
                    Err(e) => eprintln!("[runner] write manifest: {e}"),
                }
            }

            if emit_kinds.contains("swift") {
                let path = generated_dir.join("MetalTileKernels.swift");
                let _ = std::fs::create_dir_all(&generated_dir);
                match codegen_emit::write_swift_wrappers(&all_emitted_kernels, &path) {
                    Ok(()) => emit_stdout(&ProtocolMessage::Artifact {
                        kind: ArtifactKind::Swift,
                        path: path.to_string_lossy().into_owned(),
                    }),
                    Err(e) => eprintln!("[runner] write swift: {e}"),
                }
            }

            // Emit Artifact messages for each written .metal file.
            for path in &all_emitted_paths {
                emit_stdout(&ProtocolMessage::Artifact {
                    kind: ArtifactKind::Msl,
                    path: path.to_string_lossy().into_owned(),
                });
            }

            if emit_kinds.contains("metallib") {
                let metallib_path = resources_dir.join("kernels.metallib");
                let air_dir = std::env::var("CARGO_TARGET_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("target"))
                    .join("tile-build-air");
                let _ = std::fs::create_dir_all(&air_dir);
                let air_result: Result<Vec<PathBuf>, _> = all_emitted_paths
                    .par_iter()
                    .map(|m| codegen_emit::compile_metal_to_air(m, sdk, &air_dir))
                    .collect();
                match air_result {
                    Ok(airs) => {
                        match codegen_emit::link_air_to_metallib(&airs, &metallib_path, sdk) {
                            Ok(()) => emit_stdout(&ProtocolMessage::Artifact {
                                kind: ArtifactKind::Metallib,
                                path: metallib_path.to_string_lossy().into_owned(),
                            }),
                            Err(e) => {
                                eprintln!("[runner] link metallib: {e}");
                                any_err = true;
                            },
                        }
                    },
                    Err(e) => {
                        eprintln!("[runner] compile .metal: {e}");
                        any_err = true;
                    },
                }
            }
        }

        emit_stdout(&ProtocolMessage::Done {
            ok: !any_err,
            bench_passed: 0,
            bench_failed: 0,
            test_passed: 0,
            test_failed: 0,
            test_skipped: 0,
        });
        !any_err
    }

    // ── time-passes ───────────────────────────────────────────────────────────

    /// Run the standard pass pipeline over the filtered kernel corpus and print
    /// a per-pass median wall-time table directly to stdout (no JSON protocol).
    fn run_time_passes(args: &RunnerArgs) -> bool {
        const WARMUP: usize = 5;
        const ITERS: usize = 25;

        let dtype_filter: Option<Vec<DType>> = args
            .dtype
            .as_ref()
            .map(|s| s.split(',').filter_map(|t| parse_dtype(t.trim())).collect());

        let kernels: Vec<_> = all_benches()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.bench().name().contains(f)))
            .map(|e| e.bench())
            .flat_map(|b| {
                b.dtypes()
                    .iter()
                    .filter(|dt| dtype_filter.as_ref().is_none_or(|df| df.contains(dt)))
                    .map(|&dt| b.setup(dt).kernel().clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        if kernels.is_empty() {
            eprintln!("[runner] no kernels matched filter");
            return false;
        }

        let pipeline = PipelineBuilder::standard().build();
        let total_iters = WARMUP + ITERS;
        let mut pass_names: Vec<String> = Vec::new();
        let mut samples: Vec<Vec<u64>> = Vec::new();

        for iter in 0..total_iters {
            let mut pass_totals: Vec<u64> = Vec::new();
            for k in &kernels {
                let mut kc = k.clone();
                let stats: Vec<PassStats> = match run_passes_with_stats(&mut kc, &pipeline) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if pass_totals.is_empty() {
                    pass_totals = vec![0u64; stats.len()];
                    if pass_names.is_empty() {
                        pass_names = stats.iter().map(|s| s.name.clone()).collect();
                        samples = vec![Vec::with_capacity(ITERS); pass_names.len()];
                    }
                }
                for (i, s) in stats.iter().enumerate() {
                    pass_totals[i] += s.wall_us;
                }
            }
            if iter >= WARMUP {
                for (i, t) in pass_totals.iter().enumerate() {
                    samples[i].push(*t);
                }
            }
        }

        let n_kernels = kernels.len() as f64;
        println!(
            "tile build --time-passes · {} kernels × {} iters ({} warmup)",
            kernels.len(),
            ITERS,
            WARMUP,
        );
        println!("  {:<24}  {:>14}  {:>18}", "pass", "median_us", "median_us/kernel");
        for (i, name) in pass_names.iter().enumerate() {
            samples[i].sort_unstable();
            let median = samples[i][samples[i].len() / 2];
            let per_kernel = median as f64 / n_kernels;
            println!("  {name:<24}  {median:>14}  {per_kernel:>18.1}");
        }
        true
    }

    // ── inspect ───────────────────────────────────────────────────────────────

    fn run_inspect(args: &RunnerArgs) -> bool {
        use metaltile_core::protocol::InspectKind;

        let kind = match args.inspect_kind.as_deref().unwrap_or("msl") {
            "msl" => InspectKind::Msl,
            "ir" => InspectKind::Ir,
            "stats" => InspectKind::Stats,
            "listing" => InspectKind::Listing,
            other => {
                eprintln!("unknown inspect kind '{other}'; defaulting to msl");
                InspectKind::Msl
            },
        };

        let entries: Vec<_> = all_kernels()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.name().contains(f)))
            .collect();

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "inspect".into(),
            total: entries.len() as u32,
            device: None,
        });

        let mut ok = true;
        for entry in &entries {
            let dt = args.dtype.as_deref().and_then(parse_dtype).unwrap_or(DType::F32);
            let kernel = entry.build(&[dt]);

            let content = match kind {
                InspectKind::Msl => match MslGenerator::default().generate(&kernel) {
                    Ok(msl) => msl,
                    Err(e) => {
                        ok = false;
                        emit_stdout(&ProtocolMessage::ProtocolError {
                            name: entry.name().to_string(),
                            dtype: format!("{dt:?}").to_lowercase(),
                            message: e.to_string(),
                        });
                        continue;
                    },
                },
                InspectKind::Ir => format!("{kernel:#?}"),
                InspectKind::Stats | InspectKind::Listing => "not yet implemented".into(),
            };

            emit_stdout(&ProtocolMessage::Inspect {
                name: entry.name().to_string(),
                kind: kind.clone(),
                content,
            });
        }

        emit_stdout(&ProtocolMessage::Done {
            ok,
            bench_passed: 0,
            bench_failed: 0,
            test_passed: 0,
            test_failed: 0,
            test_skipped: 0,
        });
        ok
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn dtype_list(args: &RunnerArgs) -> Vec<DType> {
        if let Some(s) = &args.dtype {
            parse_dtype(s).map(|d| vec![d]).unwrap_or_else(|| {
                eprintln!("unknown dtype '{s}'; using f32");
                vec![DType::F32]
            })
        } else {
            vec![DType::F32, DType::F16, DType::BF16]
        }
    }
}

fn parse_dtype(s: &str) -> Option<DType> {
    match s {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        "i32" => Some(DType::I32),
        "u32" => Some(DType::U32),
        "i8" => Some(DType::I8),
        "u8" => Some(DType::U8),
        _ => None,
    }
}

// ── Metal compile-check ───────────────────────────────────────────────────────

/// Quickly compile an MSL source string via xcrun to catch type errors.
/// Returns `Ok(())` on success, `Err(msg)` on failure.
#[cfg(target_os = "macos")]
fn build_metal_compile_check(msl: &str, kernel_name: &str, sdk: &str) -> Result<(), String> {
    use std::process::Command;
    let dir = std::env::temp_dir().join("tile-build-check");
    let _ = std::fs::create_dir_all(&dir);
    let metal_path = dir.join(format!("{kernel_name}.metal"));
    let air_path = dir.join(format!("{kernel_name}.air"));
    if let Err(e) = std::fs::write(&metal_path, msl) {
        return Err(format!("write temp .metal: {e}"));
    }
    let output = Command::new("xcrun")
        .args(["-sdk", sdk, "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(&air_path)
        .output()
        .map_err(|e| format!("invoke xcrun metal: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let short =
            stderr.lines().filter(|l| l.contains("error:")).take(3).collect::<Vec<_>>().join("\n");
        return Err(if short.is_empty() { stderr.into_owned() } else { short });
    }
    let _ = std::fs::remove_file(&metal_path);
    let _ = std::fs::remove_file(&air_path);
    Ok(())
}

// ── per-item execution ────────────────────────────────────────────────────────

/// Run one bench entry for one dtype; returns `None` on compile/GPU error.
fn run_one_bench(
    runner: &GpuRunner,
    bench: &'static dyn KernelBench,
    dt: DType,
    warmup: usize,
    iters: usize,
) -> Option<BenchResult> {
    let setup: BenchSetup = bench.setup(dt);
    let bytes_moved = bench.bytes_moved(&setup);
    let kernel = setup.kernel();
    let name = bench.name().to_string();
    let dtype_str = format!("{dt:?}").to_lowercase();

    let msl = MslGenerator::default().generate(kernel).ok()?;
    let compiled = runner.compile(&msl, &kernel.name).ok()?;

    // Build positional GPU buffers: tensor params → constexprs.
    let mut bufs: Vec<GpuBuffer> =
        Vec::with_capacity(kernel.params.len() + kernel.constexprs.len());
    let mut input_bytes: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();
    let mut mt_out_idx: Option<usize> = None;
    let mut mt_out_n = 0usize;
    let mut mt_out_dt = dt;

    for param in &kernel.params {
        let buf = setup.buffers().iter().find(|b| b.name() == param.name).or_else(|| {
            eprintln!(
                "[runner] bench '{}' dt={dt:?}: no buffer named '{}' in setup",
                bench.name(),
                param.name,
            );
            None
        })?;
        let bytes = buf.initial_bytes();
        if param.is_output && mt_out_idx.is_none() {
            mt_out_idx = Some(bufs.len());
            mt_out_n = buf.len();
            mt_out_dt = buf.dtype();
        }
        bufs.push(runner.buffer_bytes(&bytes));
        input_bytes.insert(param.name.clone(), bytes);
        if param.kind == ParamKind::Strided {
            let shape_name = format!("{}_shape", param.name);
            let strides_name = format!("{}_strides", param.name);
            let shape = setup.buffers().iter().find(|b| b.name() == shape_name)?;
            let strides = setup.buffers().iter().find(|b| b.name() == strides_name)?;
            bufs.push(runner.buffer_bytes(&shape.initial_bytes()));
            bufs.push(runner.buffer_bytes(&strides.initial_bytes()));
        }
    }
    for decl in &kernel.constexprs {
        let n = decl.name.name();
        let (_, value) = setup.constexprs().iter().find(|(k, _)| k == n)?;
        bufs.push(runner.buffer_bytes(&value.to_le_bytes()));
    }
    let refs: Vec<&GpuBuffer> = bufs.iter().collect();

    let grid = setup.grid();
    let g = grid.grid.map(|x| x as usize);
    let t = grid.tpg.map(|x| x as usize);
    let (mt_gbps, stats) =
        bench_gbps_with(runner, &compiled, &refs, g, t, bytes_moved as f64, warmup, iters)?;

    // Reference comparison (optional).
    let (ref_gbps, mt_pct, correct) =
        if let (Some(rk), Some(out_idx)) = (setup.ref_kernel(), mt_out_idx) {
            match run_reference(
                runner,
                rk,
                &bufs,
                out_idx,
                mt_out_n,
                mt_out_dt,
                &input_bytes,
                bytes_moved,
                warmup,
                iters,
            ) {
                Some((rgbps, pass)) => {
                    let pct = mt_gbps / rgbps * 100.0;
                    (Some(rgbps), Some(pct), pass)
                },
                None => (None, None, true),
            }
        } else {
            (None, None, true)
        };

    let shape = setup.shape_label().map(|s| s.to_string()).unwrap_or_else(|| {
        let n = setup.buffers().iter().map(|b| b.len()).max().unwrap_or(0);
        let suffix = if n >= 1 << 20 && n % (1 << 20) == 0 {
            format!("{}M", n >> 20)
        } else if n >= 1 << 10 && n % (1 << 10) == 0 {
            format!("{}K", n >> 10)
        } else {
            n.to_string()
        };
        format!("N={suffix} {dtype_str}")
    });

    Some(BenchResult {
        name,
        dtype: dtype_str,
        shape,
        mt_gbps,
        ref_gbps,
        mt_pct,
        correct,
        min_us: stats.min_us,
        mean_us: stats.mean_us,
        profile: None,
    })
}

const COMPARE_ELEM_CAP: usize = 1 << 15;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn run_reference(
    runner: &GpuRunner,
    rk: &RefKernel,
    mt_bufs: &[GpuBuffer],
    mt_out_idx: usize,
    mt_out_n: usize,
    mt_out_dt: DType,
    input_bytes: &std::collections::HashMap<String, Vec<u8>>,
    bytes_moved: u64,
    warmup: usize,
    iters: usize,
) -> Option<(f64, bool)> {
    let compiled = if rk.bool_constants.is_empty() {
        runner.compile(&rk.source, &rk.fn_name).ok()?
    } else {
        runner.compile_with_bool_constants(&rk.source, &rk.fn_name, &rk.bool_constants).ok()?
    };

    let mut ref_bufs: Vec<GpuBuffer> = Vec::with_capacity(rk.buffers.len());
    let mut ref_out_idx = None;
    let mut ref_out_n = 0usize;
    let mut ref_out_dt = DType::F32;
    for b in &rk.buffers {
        if b.is_output() && ref_out_idx.is_none() {
            ref_out_idx = Some(ref_bufs.len());
            ref_out_n = b.len();
            ref_out_dt = b.dtype();
        }
        let bytes = match (b.is_output(), input_bytes.get(b.name())) {
            (false, Some(shared)) => shared.clone(),
            _ => b.initial_bytes(),
        };
        ref_bufs.push(runner.buffer_bytes(&bytes));
    }
    let ref_out_idx = ref_out_idx?;
    let ref_refs: Vec<&GpuBuffer> = ref_bufs.iter().collect();
    let g = rk.grid.grid.map(|x| x as usize);
    let t = rk.grid.tpg.map(|x| x as usize);
    let (ref_gbps, _) =
        bench_gbps_with(runner, &compiled, &ref_refs, g, t, bytes_moved as f64, warmup, iters)?;

    let n = mt_out_n.min(ref_out_n).min(COMPARE_ELEM_CAP);
    let mt_vals = read_typed(runner, &mt_bufs[mt_out_idx], n, mt_out_dt);
    let ref_vals = read_typed(runner, &ref_bufs[ref_out_idx], n, ref_out_dt);
    let err = max_abs_diff(&mt_vals, &ref_vals);
    let passed = (err as f64) <= rk.tol.into();

    Some((ref_gbps, passed))
}

/// GPU dispatch + comparison for a pre-built `TestSetup` (parallel-friendly).
///
/// The `setup` is built in Phase 1 (possibly on a rayon thread); this
/// function runs in Phase 2 on the main thread where the Metal `Context` lives.
fn run_one_test_with_setup(
    ctx: &metaltile_runtime::Context,
    setup: &TestSetup,
    tol: f64,
    name: &str,
    dt: DType,
) -> Result<TestResult, String> {
    use std::collections::BTreeMap;

    use crate::runner::gpu::elem_bytes;

    let dtype_str = format!("{dt:?}").to_lowercase();
    let no_consts: BTreeMap<String, u32> = BTreeMap::new();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for inp in setup.inputs() {
        buffers.insert(inp.name().to_string(), inp.data().to_vec());
    }
    for (k, v) in setup.constexprs() {
        buffers.insert(k.clone(), v.to_le_bytes());
    }

    let grid = setup.grid();
    let g = grid.grid.map(|x| x as usize);
    let t = grid.tpg.map(|x| x as usize);
    let result = ctx
        .dispatch_with_grid(setup.kernel(), &buffers, &no_consts, g, t)
        .map_err(|e| format!("dispatch failed: {e}"))?;

    let expected: Vec<(String, Vec<u8>, DType)> = if let Some(reference) = setup.ref_setup() {
        let mut ref_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for inp in reference.inputs() {
            ref_bufs.insert(inp.name().to_string(), inp.data().to_vec());
        }
        for (k, v) in reference.constexprs() {
            ref_bufs.insert(k.clone(), v.to_le_bytes());
        }
        let rg = reference.grid();
        let rgg = rg.grid.map(|x| x as usize);
        let rgt = rg.tpg.map(|x| x as usize);
        let ref_result = ctx
            .dispatch_with_grid(reference.kernel(), &ref_bufs, &no_consts, rgg, rgt)
            .map_err(|e| format!("reference dispatch failed: {e}"))?;
        ref_result
            .outputs
            .into_iter()
            .map(|(n, bytes)| {
                let d =
                    setup.inputs().iter().find(|b| b.name() == n).map_or(DType::F32, |b| b.dtype());
                (n, bytes, d)
            })
            .collect()
    } else {
        setup
            .expected()
            .iter()
            .map(|b| (b.name().to_string(), b.data().to_vec(), b.dtype()))
            .collect()
    };

    let mut worst = 0.0f32;
    for (bname, exp_bytes, bdt) in &expected {
        let out_bytes =
            result.output(bname).ok_or_else(|| format!("expected output '{bname}' missing"))?;
        let n = out_bytes.len() / elem_bytes(*bdt).max(1);
        let got = read_raw_f32(out_bytes, *bdt, n);
        let exp = read_raw_f32(exp_bytes, *bdt, n);
        let err = max_abs_diff(&got, &exp);
        worst = worst.max(err);
    }

    Ok(TestResult {
        name: name.to_string(),
        dtype: dtype_str,
        passed: (worst as f64) <= tol,
        max_err: worst as f64,
        skipped: false,
    })
}

// ── Public in-process test runner (legacy CLI compat) ─────────────────────────

/// Outcome of running a single `#[test_kernel]` setup in-process.
#[derive(Debug, Clone, Copy)]
pub struct TestOutcome {
    /// Whether every compared element was within tolerance.
    pub passed: bool,
    /// Largest absolute error observed across all expected buffers.
    pub max_abs_err: f32,
    /// Total number of elements compared.
    pub n_checked: usize,
}

/// Run a `TestSetup` in-process via the given runtime context.
///
/// This is the legacy API used by `tile test` and integration test harnesses
/// before the subprocess migration. It dispatches the kernel, then compares
/// each expected output buffer against the GPU result within `tol` (absolute).
pub fn run_kernel_test(
    ctx: &metaltile_runtime::Context,
    setup: &crate::harness::test::TestSetup,
    tol: f64,
) -> Result<TestOutcome, String> {
    use std::collections::BTreeMap;

    use crate::runner::gpu::elem_bytes;

    let no_consts: BTreeMap<String, u32> = BTreeMap::new();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for inp in setup.inputs() {
        buffers.insert(inp.name().to_string(), inp.data().to_vec());
    }
    for (k, v) in setup.constexprs() {
        buffers.insert(k.clone(), v.to_le_bytes());
    }

    let grid = setup.grid();
    let g = grid.grid.map(|x| x as usize);
    let t = grid.tpg.map(|x| x as usize);
    let result = ctx
        .dispatch_with_grid(setup.kernel(), &buffers, &no_consts, g, t)
        .map_err(|e| format!("dispatch failed: {e}"))?;

    let expected: Vec<(String, Vec<u8>, DType)> = if let Some(reference) = setup.ref_setup() {
        let mut ref_bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for inp in reference.inputs() {
            ref_bufs.insert(inp.name().to_string(), inp.data().to_vec());
        }
        for (k, v) in reference.constexprs() {
            ref_bufs.insert(k.clone(), v.to_le_bytes());
        }
        let rg = reference.grid();
        let rgg = rg.grid.map(|x| x as usize);
        let rgt = rg.tpg.map(|x| x as usize);
        let ref_result = ctx
            .dispatch_with_grid(reference.kernel(), &ref_bufs, &no_consts, rgg, rgt)
            .map_err(|e| format!("reference dispatch failed: {e}"))?;
        ref_result
            .outputs
            .into_iter()
            .map(|(n, bytes)| {
                let d =
                    setup.inputs().iter().find(|b| b.name() == n).map_or(DType::F32, |b| b.dtype());
                (n, bytes, d)
            })
            .collect()
    } else {
        setup
            .expected()
            .iter()
            .map(|b| (b.name().to_string(), b.data().to_vec(), b.dtype()))
            .collect()
    };

    let mut worst = 0.0f32;
    let mut n_checked = 0usize;
    for (bname, exp_bytes, bdt) in &expected {
        let out_bytes =
            result.output(bname).ok_or_else(|| format!("expected output '{bname}' missing"))?;
        let n = out_bytes.len() / elem_bytes(*bdt).max(1);
        let got = read_raw_f32(out_bytes, *bdt, n);
        let exp = read_raw_f32(exp_bytes, *bdt, n);
        let err = max_abs_diff(&got, &exp);
        worst = worst.max(err);
        n_checked += n;
    }

    Ok(TestOutcome { passed: (worst as f64) <= tol, max_abs_err: worst, n_checked })
}

fn read_raw_f32(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                // simple f16→f32 via half-float bit pattern
                let sign = ((bits as u32) >> 15) << 31;
                let exp5 = ((bits as u32) >> 10) & 0x1f;
                let mant = (bits as u32) & 0x3ff;
                if exp5 == 0 {
                    return f32::from_bits(sign);
                }
                if exp5 == 31 {
                    return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
                }
                let exp8 = (exp5 as i32 - 15 + 127) as u32;
                f32::from_bits(sign | (exp8 << 23) | (mant << 13))
            })
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        DType::I8 => bytes.iter().take(n).map(|&b| b as i8 as f32).collect(),
        DType::U8 => bytes.iter().take(n).map(|&b| b as f32).collect(),
        _ => vec![0.0; n],
    }
}

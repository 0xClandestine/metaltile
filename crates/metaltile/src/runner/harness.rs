//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `RunnerHarness` — orchestrates bench / test / build / inspect in the
//! `__tile_runner` subprocess and streams [`ProtocolMessage`]s to stdout.
//!
//! This is the only place that calls the `inventory` registries. The CLI
//! process never imports this module — it only reads the JSON lines.

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{
    DType,
    ir::ParamKind,
    protocol::{BenchResult, BuildError, BuildResult, ProtocolMessage, TestResult},
};

use crate::{
    harness::{
        bench::{BenchSetup, KernelBench, RefKernel},
        registry::{all_benches, all_kernels, all_tests},
        test::{KernelTest, TestSetup},
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

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "bench".into(),
            total,
        });

        let runner = match GpuRunner::new() {
            Ok(r) => r,
            Err(e) => {
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
                });
                return false;
            },
        };

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
        });
        failed == 0
    }

    // ── test ──────────────────────────────────────────────────────────────────

    fn run_test(args: &RunnerArgs) -> bool {
        let entries: Vec<_> = all_tests()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.test().name().contains(f)))
            .collect();

        let dtypes = Self::dtype_list(args);
        let total = (entries.len() * dtypes.len()) as u32;

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "test".into(),
            total,
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
                });
                return false;
            },
        };

        let mut passed = 0u32;
        let mut failed = 0u32;

        for entry in entries {
            let test = entry.test();
            for &dt in &dtypes {
                match run_one_test(&ctx, test, dt) {
                    Ok(result) => {
                        if result.passed {
                            passed += 1;
                        } else {
                            failed += 1;
                        }
                        emit_stdout(&ProtocolMessage::TestResult(result));
                    },
                    Err(msg) => {
                        failed += 1;
                        emit_stdout(&ProtocolMessage::ProtocolError {
                            name: entry.test().name().into(),
                            dtype: format!("{dt:?}").to_lowercase(),
                            message: msg,
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
        });
        failed == 0
    }

    // ── build ─────────────────────────────────────────────────────────────────

    fn run_build(args: &RunnerArgs) -> bool {
        let entries: Vec<_> = all_kernels()
            .filter(|e| args.filter.as_deref().is_none_or(|f| e.name().contains(f)))
            .collect();

        let dtypes = Self::dtype_list(args);
        let total = entries.len() as u32;

        emit_stdout(&ProtocolMessage::Start {
            runner_version: env!("CARGO_PKG_VERSION").into(),
            command: "build".into(),
            total,
        });

        let mut any_err = false;
        for entry in &entries {
            let mut dtypes_ok = Vec::new();
            let mut dtypes_err = Vec::new();

            for &dt in &dtypes {
                let kernel = entry.build(&[dt]);
                match MslGenerator::default().generate(&kernel) {
                    Ok(_msl) => dtypes_ok.push(format!("{dt:?}").to_lowercase()),
                    Err(e) => {
                        any_err = true;
                        dtypes_err.push(BuildError {
                            dtype: format!("{dt:?}").to_lowercase(),
                            message: e.to_string(),
                        });
                    },
                }
            }

            emit_stdout(&ProtocolMessage::BuildResult(BuildResult {
                name: entry.name().to_string(),
                dtypes_ok,
                dtypes_err,
            }));
        }

        emit_stdout(&ProtocolMessage::Done {
            ok: !any_err,
            bench_passed: 0,
            bench_failed: 0,
            test_passed: 0,
            test_failed: 0,
        });
        !any_err
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

    Some(BenchResult {
        name,
        dtype: dtype_str,
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

/// Run one test entry for one dtype.
fn run_one_test(
    ctx: &metaltile_runtime::Context,
    test: &'static dyn KernelTest,
    dt: DType,
) -> Result<TestResult, String> {
    use std::collections::BTreeMap;

    use crate::runner::gpu::elem_bytes;

    let setup: TestSetup = test.setup(dt);
    let name = test.name().to_string();
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
    let tol = test.tolerance(dt);
    for (bname, exp_bytes, bdt) in &expected {
        let out_bytes =
            result.output(bname).ok_or_else(|| format!("expected output '{bname}' missing"))?;
        let n = out_bytes.len() / elem_bytes(*bdt).max(1);
        let got = read_raw_f32(out_bytes, *bdt, n);
        let exp = read_raw_f32(exp_bytes, *bdt, n);
        let err = max_abs_diff(&got, &exp);
        worst = worst.max(err);
    }

    Ok(TestResult { name, dtype: dtype_str, passed: (worst as f64) <= tol, max_err: worst as f64 })
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

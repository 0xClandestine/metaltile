//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Harness library: entry point for external-project bench binaries.

pub use metaltile_core::bench::types::DType;
use metaltile_core::bench::{
    spec::BenchSpec,
    types::{CorrectnessStatus, dtype_label},
};
use metaltile_runtime::runner::GpuRunner as GpuContext;

use crate::{
    bench::run::run as run_spec,
    harness::{CorrectnessResult, filters::Filters},
};

/// Alias so external code that already imports `HarnessFilters` keeps compiling.
pub type HarnessFilters = Filters;

pub fn run_single_correctness_check(
    ctx: &GpuContext,
    specs: &[&BenchSpec],
    op_name: &str,
    dt: DType,
) -> CorrectnessResult {
    let spec = match specs.iter().find(|s| s.op == op_name) {
        Some(s) => s,
        None => {
            return CorrectnessResult {
                op_name: op_name.to_string(),
                dtype: dtype_label(dt).to_string(),
                passed: false,
                max_err: f32::MAX,
                cosine_sim: 0.0,
            };
        },
    };
    let results = run_spec(spec, ctx, dt);
    let equiv = results.iter().find_map(|r| r.equiv().copied());
    match equiv {
        Some(e) => CorrectnessResult {
            op_name: op_name.to_string(),
            dtype: dtype_label(dt).to_string(),
            passed: e.passed,
            max_err: e.max_abs_err,
            cosine_sim: e.cosine_sim,
        },
        None => CorrectnessResult {
            op_name: op_name.to_string(),
            dtype: dtype_label(dt).to_string(),
            passed: false,
            max_err: 0.0,
            cosine_sim: 0.0,
        },
    }
}

#[macro_export]
macro_rules! tile_harness {
    ($specs_fn:path) => {
        fn main() -> Result<(), Box<dyn std::error::Error>> {
            let args = std::env::args().collect::<Vec<_>>();
            let mut protocol = "jsonl";
            let mut action = "bench";
            let (mut raw_filter, mut mk, mut mm, mut nmk, mut nmm): (
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
            ) = (None, None, None, None, None, None);

            let mut dtypes_arg: Option<String> = None;
            let mut i = 1;
            while i < args.len() {
                let arg = &args[i];
                macro_rules! flag {
                    ($long:literal, $var:expr) => {
                        if arg == $long && i + 1 < args.len() {
                            $var = Some(args[i + 1].clone());
                            i += 2;
                            continue;
                        }
                        if let Some(v) = arg.strip_prefix(concat!($long, "=")) {
                            $var = Some(v.to_string());
                            i += 1;
                            continue;
                        }
                    };
                }
                flag!("--tile-protocol", protocol);
                flag!("--action", action);
                flag!("--filter", raw_filter);
                flag!("-f", raw_filter);
                flag!("--match-kernel", mk);
                flag!("--match-module", mm);
                flag!("--no-match-kernel", nmk);
                flag!("--no-match-module", nmm);
                flag!("--dtypes", dtypes_arg);
                i += 1;
            }

            if protocol != "jsonl" {
                eprintln!("unsupported protocol: {protocol}");
                std::process::exit(1);
            }

            let filters = match $crate::harness::HarnessFilters::build(
                raw_filter.as_deref(),
                mk.as_deref(),
                mm.as_deref(),
                nmk.as_deref(),
                nmm.as_deref(),
            ) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                },
            };

            let specs = $specs_fn();
            let dtypes_filter: Option<Vec<$crate::harness::DType>> =
                dtypes_arg.as_deref().map(|s| {
                    s.split(',')
                        .filter_map(|t| t.trim().parse::<$crate::harness::DType>().ok())
                        .collect()
                });
            match action {
                "bench" => $crate::harness::run_bench(&specs, filters, dtypes_filter),
                "build" => $crate::harness::run_build(&specs, filters, dtypes_filter),
                "test" => $crate::harness::run_test(&specs, filters, dtypes_filter),
                other => {
                    eprintln!("unknown action: {other}");
                    std::process::exit(1);
                },
            }
        }
    };
}

pub fn run_bench(
    specs: &[&'static BenchSpec],
    filters: HarnessFilters,
    dtypes_filter: Option<Vec<DType>>,
) -> ! {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GpuContext::new() failed: {e}");
            std::process::exit(1);
        },
    };
    let mut ok = 0u32;
    let mut errors = 0u32;
    for spec in specs {
        if !filters.matches_kernel(spec.kernel_name, spec.op) {
            continue;
        }
        for &dt in spec.dtypes {
            if let Some(ref df) = dtypes_filter {
                if !df.contains(&dt) {
                    continue;
                }
            }
            let results = run_spec(spec, &ctx, dt);
            for r in &results {
                let passed = matches!(r.correctness_status(), CorrectnessStatus::Passed { .. });
                let max_err = r.equiv().map(|e| e.max_abs_err);
                let json = serde_json::json!({
                    "type": "result",
                    "op": r.op(),
                    "subop": spec.subop,
                    "kernel_name": spec.kernel_name,
                    "dtype": dt.label(),
                    "shape": r.shape(),
                    "metric": r.metric(),
                    "ref_gbps": r.ref_perf(),
                    "mt_gbps": r.mt_perf(),
                    "passed": passed,
                    "max_err": max_err,
                });
                println!("{}", json);
                if r.mt_perf().is_some() {
                    ok += 1;
                } else {
                    errors += 1;
                }
            }
        }
    }
    println!("{}", serde_json::json!({"type":"done","ok":ok,"errors":errors}));
    std::process::exit(0);
}

pub fn run_build(
    specs: &[&'static BenchSpec],
    filters: HarnessFilters,
    dtypes_filter: Option<Vec<DType>>,
) -> ! {
    use metaltile_core::bench::spec::effective_mode;
    let mut ok = 0u32;
    let mut errors = 0u32;
    for spec in specs {
        if !filters.matches_kernel(spec.kernel_name, spec.op) {
            continue;
        }
        let mode = effective_mode(spec);
        for &dt in spec.dtypes {
            if let Some(ref df) = dtypes_filter {
                if !df.contains(&dt) {
                    continue;
                }
            }
            let mut k = (spec.kernel_ir)(dt);
            k.mode = mode;
            let msl_gen = metaltile_codegen::generator_for_mode(mode, spec.dispatch.tpg_hint());
            match msl_gen.generate(&k) {
                Ok(_) => {
                    ok += 1;
                    println!(
                        "{}",
                        serde_json::json!({
                            "type": "result",
                            "op": spec.op,
                            "kernel_name": spec.kernel_name,
                            "dtype": dt.label(),
                            "ok": true,
                        })
                    );
                },
                Err(e) => {
                    errors += 1;
                    println!(
                        "{}",
                        serde_json::json!({
                            "type": "result",
                            "op": spec.op,
                            "kernel_name": spec.kernel_name,
                            "dtype": dt.label(),
                            "ok": false,
                            "error": e.to_string(),
                        })
                    );
                },
            }
        }
    }
    println!("{}", serde_json::json!({"type":"done","ok":ok,"errors":errors}));
    std::process::exit(if errors > 0 { 1 } else { 0 });
}

pub fn run_test(
    specs: &[&'static BenchSpec],
    filters: HarnessFilters,
    dtypes_filter: Option<Vec<DType>>,
) -> ! {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GpuContext::new() failed: {e}");
            std::process::exit(1);
        },
    };
    let mut ok = 0u32;
    let mut errors = 0u32;
    for spec in specs {
        if !filters.matches_kernel(spec.kernel_name, spec.op) {
            continue;
        }
        for &dt in spec.dtypes {
            if let Some(ref df) = dtypes_filter {
                if !df.contains(&dt) {
                    continue;
                }
            }
            let results = run_spec(spec, &ctx, dt);
            let passed = results
                .iter()
                .all(|r| matches!(r.correctness_status(), CorrectnessStatus::Passed { .. }));
            let max_err = results
                .iter()
                .filter_map(|r| r.equiv().map(|e| e.max_abs_err))
                .fold(0.0f32, f32::max);
            println!(
                "{}",
                serde_json::json!({
                    "type": "test_result",
                    "op": spec.op,
                    "kernel_name": spec.kernel_name,
                    "subop": spec.subop,
                    "dtype": dt.label(),
                    "passed": passed,
                    "max_err": max_err,
                })
            );
            if passed {
                ok += 1;
            } else {
                errors += 1;
            }
        }
    }
    println!("{}", serde_json::json!({"type":"done","ok":ok,"errors":errors}));
    std::process::exit(if errors > 0 { 1 } else { 0 });
}

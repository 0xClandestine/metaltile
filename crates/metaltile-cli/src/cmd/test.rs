//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile test` — Run GPU correctness checks.

use metaltile::harness::TestRow;
use metaltile_core::bench::types::{CorrectnessStatus, dtype_label};
use metaltile::bench::run::run as run_spec;
use metaltile_runtime::runner::GpuRunner;

use crate::{
    CliError,
    FilterFlags,
    project::{self, harness::HarnessMessage, CompileService, RealCompileService},
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct TestArgs {
    #[command(flatten)]
    pub filters: FilterFlags,
    /// Comma-separated list of dtypes to test (f32,f16,bf16).
    #[arg(long = "dtypes")]
    pub dtypes: Option<String>,
    /// Print per-element error detail (-v) or full output diff (-vv) on failure.
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    pub verbose: u8,
    /// Skip GPU dispatch; only verify that kernels compile.
    #[arg(long = "no-gpu")]
    pub no_gpu: bool,
}

impl TestArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let _span =
            tracing::info_span!("test", filter = ?args.filters.filter, verbose = args.verbose)
                .entered();

        let filters = args.filters.to_filters()?;

        // External-project path
        if project::has_tile_toml() {
            return run_external(args);
        }

        // In-tree path
        let runner = match GpuRunner::new() {
            Ok(r) => r,
            Err(e) => return Err(CliError::GpuInit(e)),
        };

        println!(
            "{} {}",
            paint_stdout("tile test", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(format!("· {}", runner.device_name), Style::new().fg(Color::BrightBlack)),
        );

        let specs = metaltile_core::bench::spec::bench_specs();
        let mut rows: Vec<TestRow> = Vec::new();

        for spec in specs {
            if !filters.matches_kernel(spec.kernel_name, spec.op) {
                continue;
            }
            for &dt in spec.dtypes {
                let results = run_spec(spec, &runner, dt);
                let passed = results
                    .iter()
                    .all(|r| matches!(r.correctness_status(), CorrectnessStatus::Passed { .. }));
                let max_err = results
                    .iter()
                    .filter_map(|r| r.equiv().map(|e| e.max_abs_err))
                    .fold(0.0f32, f32::max);
                rows.push(TestRow {
                    kernel_name: spec.kernel_name.to_string(),
                    dtype: dtype_label(dt).to_string(),
                    passed,
                    max_err: Some(max_err),
                });
            }
        }

        render_test_results(&rows);
        finalize_test(&rows)
    }
}

pub(crate) fn render_test_results(rows: &[TestRow]) {
    for row in rows {
        let icon = if row.passed { "✓" } else { "✗" };
        let color = if row.passed { Color::Green } else { Color::Red };
        match row.max_err {
            Some(e) => println!(
                "  {:20} {:>4}   {} (max_err={:.2e})",
                paint_stdout(&row.kernel_name, Style::new().fg(Color::Cyan).bold()),
                paint_stdout(&row.dtype, Style::new().fg(Color::Blue).bold()),
                paint_stdout(icon, Style::new().fg(color).bold()),
                e,
            ),
            None => println!(
                "  {:20} {:>4}   {}",
                paint_stdout(&row.kernel_name, Style::new().fg(Color::Cyan).bold()),
                paint_stdout(&row.dtype, Style::new().fg(Color::Blue).bold()),
                paint_stdout(icon, Style::new().fg(color).bold()),
            ),
        }
    }
}

fn finalize_test(rows: &[TestRow]) -> Result<(), CliError> {
    let passed = rows.iter().filter(|r| r.passed).count();
    let failed = rows.iter().filter(|r| !r.passed).count();

    println!();
    if failed > 0 {
        println!(
            "  {}  {}",
            paint_stdout(format!("{passed} passed"), Style::new().fg(Color::Green).bold()),
            paint_stderr(format!("{failed} failed"), Style::new().fg(Color::Red).bold()),
        );
        Err(CliError::TestFailed(failed as u32))
    } else {
        println!(
            "  {}",
            paint_stdout(
                format!("{passed} passed, 0 failed"),
                Style::new().fg(Color::Green).bold()
            ),
        );
        Ok(())
    }
}

fn run_external(args: &TestArgs) -> Result<(), CliError> {
    let compile = RealCompileService;
    let project_dir = std::path::Path::new(".");
    let binary = compile
        .compile_harness(project_dir, false)
        .map_err(|e| CliError::Config(format!("harness build: {e}")))?;

    let filters = args.filters.to_filters()?;
    let mut cmd = std::process::Command::new(&binary);
    cmd.arg("--tile-protocol=jsonl").arg("--action=test");
    filters.forward_to_cmd(&mut cmd);
    if let Some(dtypes) = &args.dtypes {
        cmd.arg("--dtypes").arg(dtypes);
    }

    let profile = project::active_profile();
    println!(
        "{} {}  profile={}",
        paint_stdout("tile test", Style::new().fg(Color::Cyan).bold()),
        paint_stdout("· Tile.toml", Style::new().fg(Color::BrightBlack)),
        profile,
    );

    let messages = crate::project::harness::run_harness(&mut cmd)?;

    let mut rows: Vec<TestRow> = Vec::new();
    for msg in messages {
        if let HarnessMessage::TestResult(val) = msg {
            let kernel_name = val
                .get("kernel_name")
                .or_else(|| val.get("op"))
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            let dtype = val.get("dtype").and_then(|v| v.as_str()).unwrap_or("?").to_string();
            let passed = val.get("passed").and_then(|v| v.as_bool()).unwrap_or(false);
            let max_err = val.get("max_err").and_then(|v| v.as_f64()).map(|v| v as f32);
            rows.push(TestRow { kernel_name, dtype, passed, max_err });
        }
    }

    render_test_results(&rows);
    finalize_test(&rows)
}

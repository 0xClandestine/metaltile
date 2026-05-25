//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile bench` — Benchmark suite: MetalTile vs MLX reference.

use std::collections::HashMap;

use metaltile::harness::BenchRow;
use crate::bench::report::BenchReport;
use metaltile_core::bench::types::CorrectnessStatus;
use metaltile::bench::run::run as run_spec;
use metaltile_runtime::runner::GpuRunner;

use crate::{
    CliError,
    FilterFlags,
    bench::profile::{ProfileMap, ProfileRow},
    diff,
    git::{GitProvider, RealGit},
    project::{self, harness::HarnessMessage, CompileService, RealCompileService},
    bench::printer::SuitePrinter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    #[command(flatten)]
    pub filters: FilterFlags,
    /// Show occupancy and register profile (-v) and GPU timing stats (-vv).
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    pub verbose: u8,
    /// Write results as JSON to <path>.
    #[arg(long = "json", short = 'o', value_hint = clap::ValueHint::FilePath)]
    pub json: Option<std::path::PathBuf>,
    /// Run even if the working tree has uncommitted changes.
    #[arg(long = "allow-dirty")]
    pub allow_dirty: bool,
    /// Show a perf diff against the target-branch baseline after running.
    #[arg(long = "diff")]
    pub diff: bool,
    /// Git ref whose baselines/<chip>.json to diff against (default: origin/dev).
    #[arg(long = "baseline-ref", value_name = "REF")]
    pub baseline_ref: Option<String>,
}

impl BenchArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let _span =
            tracing::info_span!("bench", filter = ?args.filters.filter, verbose = args.verbose)
                .entered();

        let filters = args.filters.to_filters()?;
        let git = RealGit;

        // Dirty-tree check — uses GitProvider trait.
        dirty_tree_check(&git, args.allow_dirty)?;

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
            paint_stdout("tile bench", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(format!("· {}", runner.device_name), Style::new().fg(Color::BrightBlack)),
        );

        let verbose = args.verbose;
        let profile_map: Option<HashMap<(String, String), ProfileRow>> =
            if verbose > 0 { Some(ProfileMap::compute(&filters).into_inner()) } else { None };

        let mut printer = SuitePrinter::new(true);
        let mut all: Vec<BenchRow> = Vec::new();
        let mut matched_filter = false;

        let specs = metaltile_core::bench::spec::bench_specs();
        for spec in specs {
            if !filters.matches_kernel(spec.kernel_name, spec.op) {
                continue;
            }
            matched_filter = true;
            for &dt in spec.dtypes {
                let _kspan = tracing::debug_span!("kernel", op = spec.op, dtype = %dt).entered();
                tracing::debug!(op = spec.op, dtype = %dt, "running benchmark");
                runner.flush_slc();
                let results = run_spec(spec, &runner, dt);
                let rows: Vec<BenchRow> = results.iter().map(BenchRow::from).collect();
                printer.print_batch(&rows, profile_map.as_ref(), verbose);
                all.extend(rows);
            }
        }

        if all.is_empty() {
            warn_no_results(args.filters.filter.as_deref(), matched_filter);
            return Ok(());
        }

        BenchReport::new(all.clone()).validate().unwrap_or_else(|err| panic!("{err}"));
        finalize_bench(&all, args, &runner.device_name, &mut printer)
    }
}

// ── Dirty-tree check (extracted for testability) ─────────────────────────

fn dirty_tree_check(git: &impl GitProvider, allow_dirty: bool) -> Result<(), CliError> {
    if allow_dirty {
        return Ok(());
    }
    if let Some(true) = git.working_tree_dirty() {
        let files = git.list_dirty_files();
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                "working tree has uncommitted changes; bench numbers \
                 would not tie back to a clean commit.",
                Style::new().fg(Color::BrightWhite),
            ),
        );
        if !files.is_empty() {
            let preview: Vec<&str> = files.iter().take(8).map(String::as_str).collect();
            let overflow = if files.len() > 8 {
                format!(" (+{} more)", files.len() - 8)
            } else {
                String::new()
            };
            eprintln!(
                "  {} {}{}",
                paint_stderr("Dirty:", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(preview.join(", "), Style::new().fg(Color::BrightWhite)),
                overflow,
            );
        }
        eprintln!(
            "  {} {}",
            paint_stderr("Override:", Style::new().fg(Color::BrightBlack).bold()),
            paint_stderr(
                "re-run with --allow-dirty to bench anyway.",
                Style::new().fg(Color::BrightBlack),
            ),
        );
        return Err(CliError::Git("uncommitted changes".into()));
    }
    Ok(())
}

// ── Shared post-processing ────────────────────────────────────────────────

fn finalize_bench(
    all: &[BenchRow],
    args: &BenchArgs,
    device_name: &str,
    printer: &mut SuitePrinter,
) -> Result<(), CliError> {
    printer.finish();

    let impl_count = all.iter().filter(|r| r.mt_perf().is_some()).count();
    let nyi_count = all.iter().filter(|r| r.mt_perf().is_none()).count();
    let checked_count = all.iter().filter(|r| row_has_equiv(r)).count();
    let equiv_pass = all
        .iter()
        .filter(|r| matches!(r.correctness_status(), CorrectnessStatus::Passed { .. }))
        .count();
    let equiv_fail = all
        .iter()
        .filter(|r| matches!(r.correctness_status(), CorrectnessStatus::Failed { .. }))
        .count();
    let unchecked: Vec<String> = all
        .iter()
        .filter(|r| r.is_unchecked())
        .map(|r| format!("{} [{}]", r.op_display(), r.shape()))
        .collect();
    let avg_pct: Option<f64> = {
        let valid: Vec<f64> = all.iter().filter_map(|r| r.pct()).collect();
        if valid.is_empty() { None } else { Some(valid.iter().sum::<f64>() / valid.len() as f64) }
    };

    let mut parts: Vec<String> = Vec::new();
    let sep = format!("  {}  ", paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()));

    parts.push(format!(
        "{} impl",
        paint_stdout(impl_count.to_string(), Style::new().fg(Color::Green).bold()),
    ));
    if nyi_count > 0 {
        parts.push(format!(
            "{} NYI",
            paint_stdout(nyi_count.to_string(), Style::new().fg(Color::Yellow).bold()),
        ));
    }
    if let Some(p) = avg_pct {
        parts.push(format!("avg {}", paint_stdout(format!("{p:.0}% MT"), pct_style(p))));
    }
    if checked_count > 0 {
        let corr_style = if equiv_fail == 0 {
            Style::new().fg(Color::Green).bold()
        } else {
            Style::new().fg(Color::Yellow).bold()
        };
        parts.push(format!(
            "{} correct",
            paint_stdout(format!("{equiv_pass}/{checked_count}"), corr_style),
        ));
    }
    if !unchecked.is_empty() {
        parts.push(format!(
            "{} unchecked",
            paint_stdout(unchecked.len().to_string(), Style::new().fg(Color::Yellow).bold()),
        ));
    }

    println!("\n  {}", parts.join(&sep));

    if equiv_fail > 0 {
        println!(
            "  {} {}",
            paint_stdout("Failures:", Style::new().fg(Color::Red).bold()),
            paint_stdout(equiv_fail.to_string(), Style::new().fg(Color::Red).bold()),
        );
    }
    if !unchecked.is_empty() {
        println!(
            "  {} {}",
            paint_stdout("Unchecked:", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(unchecked.join(", "), Style::new().fg(Color::Yellow).bold()),
        );
    }
    println!();

    let report = BenchReport::new(all.to_vec());

    if args.diff {
        try_auto_diff(
            device_name,
            &report,
            args.filters.filter.as_deref(),
            args.baseline_ref.as_deref(),
        );
    }

    if let Some(path) = &args.json {
        let json = report.to_json(device_name);
        match std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")))
            .and_then(|_| std::fs::write(path, &json))
        {
            Ok(()) => println!(
                "  {} {}",
                paint_stdout("Saved →", Style::new().fg(Color::Cyan).bold()),
                paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightWhite)),
            ),
            Err(e) => eprintln!(
                "  {} {}",
                paint_stderr("save failed:", Style::new().fg(Color::Red).bold()),
                paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite)),
            ),
        }
    }

    if equiv_fail > 0 {
        return Err(CliError::Correctness(equiv_fail as u32));
    }
    Ok(())
}

fn row_has_equiv(r: &BenchRow) -> bool {
    matches!(
        r.correctness_status(),
        CorrectnessStatus::Passed { .. } | CorrectnessStatus::Failed { .. }
    )
}

fn warn_no_results(filter: Option<&str>, matched_filter: bool) {
    if let Some(pattern) = filter {
        if matched_filter {
            eprintln!(
                "{} {}",
                paint_stderr("[error]", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    format!(
                        "Kernel matched --filter {pattern:?} but all shapes failed to compile or run"
                    ),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    format!("No benchmarks matched --filter {pattern:?}"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        }
    } else {
        eprintln!(
            "{} {}",
            paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
            paint_stderr("No benchmarks ran", Style::new().fg(Color::BrightWhite)),
        );
    }
}

// ── External-project path ─────────────────────────────────────────────────

fn run_external(args: &BenchArgs) -> Result<(), CliError> {
    use crate::project::harness::HarnessMessage;

    let filters = args.filters.to_filters()?;
    let compile = RealCompileService;

    let project_dir = std::path::Path::new(".");
    let binary = compile
        .compile_harness(project_dir, true)
        .map_err(|e| CliError::Config(format!("harness build: {e}")))?;

    let mut cmd = std::process::Command::new(&binary);
    cmd.arg("--tile-protocol=jsonl").arg("--action=bench");
    filters.forward_to_cmd(&mut cmd);
    if args.verbose > 0 {
        cmd.arg("-v");
    }

    let profile = project::active_profile();
    println!(
        "{} {}  profile={}",
        paint_stdout("tile bench", Style::new().fg(Color::Cyan).bold()),
        paint_stdout("· Tile.toml", Style::new().fg(Color::BrightBlack)),
        profile,
    );

    let messages = crate::project::harness::run_harness(&mut cmd)?;

    let mut printer = SuitePrinter::new(true);
    let mut all: Vec<BenchRow> = Vec::new();
    let mut device_name = String::from("external");

    for msg in messages {
        match msg {
            HarnessMessage::BenchResult(val) =>
                if let Some(row) = BenchRow::from_json(&val) {
                    printer.print_batch(&[row.clone()], None, args.verbose);
                    all.push(row);
                },
            HarnessMessage::Done { device } =>
                if let Some(d) = device {
                    device_name = d;
                },
            _ => {},
        }
    }

    if all.is_empty() {
        eprintln!(
            "  {} no results from harness",
            paint_stderr("warn:", Style::new().fg(Color::Yellow).bold()),
        );
        return Ok(());
    }

    finalize_bench(&all, args, &device_name, &mut printer)
}

// ── Auto-diff helpers ─────────────────────────────────────────────────────

fn try_auto_diff(
    device: &str,
    report: &BenchReport,
    filter: Option<&str>,
    baseline_ref_override: Option<&str>,
) {
    use crate::git::GitProvider;

    let git = RealGit;
    let slug = BenchReport::chip_slug(device);
    let baseline_path = format!("baselines/{slug}.json");

    let candidates: Vec<&str> = match baseline_ref_override {
        Some(r) => vec![r],
        None => vec!["origin/dev", "upstream/dev", "dev"],
    };
    let Some(reference) = git.resolve_baseline_ref(&candidates) else {
        log_skip("baseline auto-diff: no target-branch ref — skipping");
        return;
    };
    let Some(sha) = git.merge_base_with(&reference) else {
        log_skip(&format!("baseline auto-diff: merge-base HEAD..{reference} failed — skipping"));
        return;
    };
    let Some(content) = git.show_file_at(&sha, &baseline_path) else {
        log_skip(&format!(
            "baseline auto-diff: no {baseline_path} at {reference} ({}…) — skipping",
            sha.chars().take(7).collect::<String>()
        ));
        return;
    };

    let baseline_json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            log_skip(&format!(
                "baseline auto-diff: {baseline_path} at {reference} is not valid JSON ({e}) — skipping"
            ));
            return;
        },
    };
    let Some(baseline_rows) = baseline_json.get("results").and_then(|v| v.as_array()).cloned()
    else {
        log_skip(&format!(
            "baseline auto-diff: {baseline_path} at {reference} has no 'results' array — skipping"
        ));
        return;
    };

    let current_rows: Vec<serde_json::Value> = report.to_json_values();

    let short_sha: String = sha.chars().take(7).collect();
    let heading = format!("tile bench · diff vs {reference} @ {short_sha} ({baseline_path})");
    let opts = diff::RenderOpts {
        heading: Some(&heading),
        sort: "regression",
        filter,
        ..diff::RenderOpts::default()
    };
    let outcome = diff::render(&baseline_rows, &current_rows, &opts);
    if outcome.total_rows == 0 {
        log_skip(&format!(
            "baseline auto-diff: no overlapping rows with {baseline_path} at {reference}"
        ));
    }
}

fn log_skip(msg: &str) {
    eprintln!("  {}", paint_stderr(msg, Style::new().fg(Color::BrightBlack)));
}

fn pct_style(pct: f64) -> Style {
    if pct >= 90.0 {
        Style::new().fg(Color::Green).bold()
    } else if pct >= 60.0 {
        Style::new().fg(Color::Yellow).bold()
    } else {
        Style::new().fg(Color::Red).bold()
    }
}

#[cfg(test)]
mod tests {
    use crate::bench::report::BenchReport;
    use metaltile_core::bench::types::{EquivResult, OpBench};

    use super::*;

    fn pass_equiv() -> EquivResult {
        EquivResult { n_checked: 1, max_abs_err: 0.0, cosine_sim: 1.0, passed: true }
    }

    fn fail_equiv() -> EquivResult {
        EquivResult { n_checked: 1, max_abs_err: 1e3, cosine_sim: 0.5, passed: false }
    }

    fn make_row(ref_p: Option<f64>, mt_p: Option<f64>, equiv: Option<EquivResult>) -> BenchRow {
        let b = OpBench::new("op", "GB/s");
        BenchRow::from(&b.result("shape", ref_p, mt_p, equiv))
    }

    #[test]
    fn summary_counts_per_category() {
        let report = BenchReport::new(vec![
            make_row(Some(100.0), Some(95.0), Some(pass_equiv())),
            make_row(Some(100.0), Some(40.0), Some(fail_equiv())),
            make_row(Some(100.0), None, None),
        ]);
        let s = report.summary();
        assert_eq!(s.total, 3);
        assert_eq!(s.implemented, 2);
        assert_eq!(s.correct, 1);
        assert_eq!(s.unchecked, 0);
    }

    #[test]
    fn summary_on_empty_input_is_all_zero() {
        let s = BenchReport::new(vec![]).summary();
        assert_eq!(s.total, 0);
        assert_eq!(s.implemented, 0);
        assert_eq!(s.correct, 0);
        assert_eq!(s.unchecked, 0);
    }

    #[test]
    fn json_serialises_row_without_subop() {
        let report = BenchReport::new(vec![make_row(Some(323.9), Some(325.6), Some(pass_equiv()))]);
        let json = report.to_json("test-device");
        assert!(json.contains(r#""op":"op""#));
        assert!(!json.contains(r#""subop""#));
    }

    #[test]
    fn bench_row_from_json_parses_full_result() {
        let val = serde_json::json!({
            "type": "result",
            "op": "unary",
            "subop": "exp",
            "shape": "N=64M f32",
            "metric": "GB/s",
            "ref_gbps": 544.8,
            "mt_gbps": 512.1,
            "passed": true,
            "max_err": 0.0001,
            "cosine_sim": 1.0
        });
        let row = BenchRow::from_json(&val).unwrap();
        assert_eq!(row.op, "unary");
        assert_eq!(row.subop.as_deref(), Some("exp"));
        assert_eq!(row.metric, "GB/s");
        assert!(matches!(row.correctness, CorrectnessStatus::Passed { .. }));
    }

    #[test]
    fn bench_row_from_json_handles_missing_mt_perf() {
        let val = serde_json::json!({
            "type": "result",
            "op": "fft",
            "shape": "N=1024",
            "ref_gbps": 100.0,
            "passed": false
        });
        let row = BenchRow::from_json(&val).unwrap();
        assert!(matches!(row.correctness, CorrectnessStatus::Unavailable));
    }
}

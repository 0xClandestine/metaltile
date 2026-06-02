//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile test` — run `#[test_kernel]` correctness setups against a CPU oracle.
//!
//! ## Output format (forge-style)
//!
//! ```text
//! Ran 3 tests for mt_add
//! [PASS] mt_add [f32]   (err=0.00e0)
//! [PASS] mt_add [f16]   (err=2.38e-7)
//! [PASS] mt_add [bf16]  (err=1.56e-3)
//! Suite result: ok. 3 passed; 0 failed; finished in 45.12ms
//!
//! Ran 2 test suites in 57.46ms: 3 tests passed, 1 failed (4 total tests)
//! ```

use std::{io::Write as _, time::Instant};

use metaltile::runner::run_kernel_test;
use metaltile_codegen::generator_for_mode;
use rayon::prelude::*;

use crate::{
    FilterSpec,
    TestArgs,
    term::{Color, Spinner, Style, paint_stderr, paint_stdout},
};

// ── Summary tracking ─────────────────────────────────────────────────────

struct SuiteSummary {
    name: String,
    passed: usize,
    failed: usize,
    elapsed_ms: f64,
    /// (label, passed) per individual test — populated only with --detailed.
    rows: Vec<(String, bool)>,
}

// ── MSL generation helper ────────────────────────────────────────────────

fn generate_msl_for_setup(setup: &metaltile::harness::test::TestSetup) -> Option<String> {
    let k = setup.kernel().clone();
    let generator = generator_for_mode(k.mode, None);
    generator.generate(&k).ok()
}

/// `TileCommand` wrapper for `tile test`.
pub struct TestCommand<'a>(pub &'a TestArgs);

impl<'a> super::TileCommand for TestCommand<'a> {
    fn run(&self, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
        run(self.0, harness)
    }
}

pub fn run(args: &TestArgs, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
    let _span = tracing::info_span!("test", filter = ?args.filter_args.filter).entered();
    let verbosity = harness.verbosity();

    // Merge positional path into filter args.
    let mut filter_args = args.filter_args.clone();
    if let Some(p) = &args.path {
        if p.contains('/') || p.contains('*') {
            if filter_args.match_path.is_none() {
                filter_args.match_path = Some(p.clone());
            }
        } else if filter_args.filter.is_none() {
            filter_args.filter = Some(p.clone());
        }
    }
    let spec = FilterSpec::from_args(&filter_args);

    let ctx = match metaltile::Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "{} {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite)),
            );
            return Err(crate::CliError::GpuInit(e.to_string()));
        },
    };

    // Collect all matching entries.
    let entries: Vec<_> = metaltile::harness::registry::all_tests()
        .filter(|entry| spec.matches(entry.test().name(), entry.file()))
        .collect();

    if entries.is_empty() {
        if let Some(pattern) = &filter_args.filter {
            eprintln!(
                "{} {}",
                paint_stderr("warning:", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    format!("no tests matched filter {pattern:?}"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else if !spec.is_empty() {
            eprintln!(
                "{} {}",
                paint_stderr("warning:", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    "no tests matched the given filter flags",
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("warning:", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    "no #[test_kernel] tests registered",
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        }
        return Ok(());
    }

    // --list: print matching tests without running them.
    if args.list {
        for entry in &entries {
            let t = entry.test();
            for &dt in t.dtypes() {
                println!("{} [{dt}]", t.name());
            }
        }
        return Ok(());
    }

    // Phase 1 (parallel): run CPU oracle for every (entry, dtype) pair.
    // `t.setup(dt)` computes expected output buffers on the CPU — no GPU
    // involvement — so all pairs can run concurrently.
    let mut spinner = Spinner::new("Preparing tests...");
    let work: Vec<Vec<_>> = entries
        .par_iter()
        .map(|entry| {
            let t = entry.test();
            t.dtypes()
                .iter()
                .map(|&dt| {
                    let label = format!("{} [{dt}]", t.name());
                    let setup = t.setup(dt);
                    let tol = t.tolerance(dt);
                    (label, setup, tol)
                })
                .collect()
        })
        .collect();

    spinner.stop();
    let wall_start = Instant::now();
    let mut total = 0usize;
    let mut total_passed = 0usize;
    let mut total_failed = 0usize;

    // failure_lines: pre-formatted "[FAIL: reason] label" strings for the summary.
    // failure_labels: plain label strings for JSON output.
    let mut failure_lines: Vec<String> = Vec::new();
    let mut failure_labels: Vec<String> = Vec::new();
    let mut suite_summaries: Vec<SuiteSummary> = Vec::new();

    // Progress counter — used when --show-progress is set.
    let total_tests: usize = work.iter().map(|g| g.len()).sum();
    let mut test_idx = 0usize;

    // Phase 2 (serial): GPU dispatch + comparison.
    // `run_kernel_test` uses the Metal `Context` which is not `Send`, so all
    // dispatches happen on the main thread in deterministic order.
    'suites: for (entry, group) in entries.iter().zip(work.iter()) {
        let suite_name = entry.test().name();
        let n = group.len();
        let noun = if n == 1 { "test" } else { "tests" };

        println!();
        println!(
            "Ran {n} {noun} for {}",
            paint_stdout(suite_name, Style::new().fg(Color::Cyan).bold()),
        );

        let suite_start = Instant::now();
        let mut suite_passed = 0usize;
        let mut suite_failed = 0usize;
        let mut suite_rows: Vec<(String, bool)> = Vec::new();

        for (label, setup, tol) in group {
            test_idx += 1;

            // Live progress indicator on stderr (clears before the result line).
            if args.show_progress {
                eprint!("\r\x1b[K[{test_idx}/{total_tests}] {label}");
                let _ = std::io::stderr().flush();
            }

            total += 1;
            match run_kernel_test(&ctx, setup, *tol) {
                Ok(o) if o.passed => {
                    if args.show_progress {
                        eprint!("\r\x1b[K");
                    }
                    suite_passed += 1;
                    total_passed += 1;
                    if args.detailed {
                        suite_rows.push((label.clone(), true));
                    }
                    println!(
                        "{}  {}  {}",
                        paint_stdout("[PASS]", Style::new().fg(Color::Green).bold()),
                        paint_stdout(label, Style::new().fg(Color::BrightWhite)),
                        paint_stdout(
                            format!("(err={:.2e}, tol={tol:.2e})", o.max_abs_err),
                            Style::new().fg(Color::BrightBlack),
                        ),
                    );
                    // -vvv: show MSL for passing tests too.
                    if verbosity >= 3
                        && let Some(msl) = generate_msl_for_setup(setup)
                    {
                        println!("// MSL for {label}\n{msl}");
                    }
                },
                Ok(o) => {
                    if args.show_progress {
                        eprint!("\r\x1b[K");
                    }
                    suite_failed += 1;
                    total_failed += 1;
                    if args.detailed {
                        suite_rows.push((label.clone(), false));
                    }
                    let reason = format!("err={:.2e}, tol={tol:.2e}", o.max_abs_err);
                    let line = format!(
                        "{}  {}",
                        paint_stdout(
                            format!("[FAIL: {reason}]"),
                            Style::new().fg(Color::Red).bold(),
                        ),
                        paint_stdout(label, Style::new().fg(Color::BrightWhite)),
                    );
                    failure_lines.push(line.clone());
                    failure_labels.push(label.clone());
                    println!("{line}");
                    // -vv or higher: show MSL for failing tests.
                    if verbosity >= 2
                        && let Some(msl) = generate_msl_for_setup(setup)
                    {
                        println!("// MSL for {label}\n{msl}");
                    }
                    if args.fail_fast {
                        break 'suites;
                    }
                },
                Err(e) => {
                    if args.show_progress {
                        eprint!("\r\x1b[K");
                    }
                    suite_failed += 1;
                    total_failed += 1;
                    if args.detailed {
                        suite_rows.push((label.clone(), false));
                    }
                    let line = format!(
                        "{}  {}",
                        paint_stdout(format!("[FAIL: {e}]"), Style::new().fg(Color::Red).bold()),
                        paint_stdout(label, Style::new().fg(Color::BrightWhite)),
                    );
                    failure_lines.push(line.clone());
                    failure_labels.push(label.clone());
                    println!("{line}");
                    // -vv or higher: show MSL for failing tests.
                    if verbosity >= 2
                        && let Some(msl) = generate_msl_for_setup(setup)
                    {
                        println!("// MSL for {label}\n{msl}");
                    }
                    if args.fail_fast {
                        break 'suites;
                    }
                },
            }
        }

        let suite_elapsed = suite_start.elapsed();
        let (result_word, passed_paint, failed_paint) = if suite_failed == 0 {
            (
                paint_stdout("ok", Style::new().fg(Color::Green).bold()),
                paint_stdout(suite_passed.to_string(), Style::new().fg(Color::Green)),
                paint_stdout("0", Style::new().fg(Color::BrightBlack)),
            )
        } else {
            (
                paint_stderr("FAILED", Style::new().fg(Color::Red).bold()),
                paint_stdout(suite_passed.to_string(), Style::new().fg(Color::BrightBlack)),
                paint_stderr(suite_failed.to_string(), Style::new().fg(Color::Red)),
            )
        };
        println!(
            "Suite result: {result_word}. {passed_paint} passed; {failed_paint} failed; finished in {suite_elapsed:.2?}",
        );

        if args.summary || args.detailed {
            suite_summaries.push(SuiteSummary {
                name: suite_name.to_string(),
                passed: suite_passed,
                failed: suite_failed,
                elapsed_ms: suite_elapsed.as_secs_f64() * 1000.0,
                rows: suite_rows,
            });
        }
    }

    // Failing tests section — mirrors forge's "Failing tests:" block.
    if !failure_lines.is_empty() {
        println!("\nFailing tests:");
        for line in &failure_lines {
            println!("  {line}");
        }
    }

    // --summary / --detailed table.
    if args.summary && !suite_summaries.is_empty() {
        println!();
        let name_w = suite_summaries.iter().map(|s| s.name.len()).max().unwrap_or(8).max(5);
        if args.detailed {
            // Detailed: show every individual test row grouped by suite.
            let label_w = suite_summaries
                .iter()
                .flat_map(|s| s.rows.iter().map(|(l, _)| l.len()))
                .max()
                .unwrap_or(8)
                .max(4);
            println!("  {:<name_w$}  {:<label_w$}  Result", "Suite", "Test");
            println!("  {}  {}  ------", "-".repeat(name_w), "-".repeat(label_w));
            for s in &suite_summaries {
                for (label, passed) in &s.rows {
                    let result = if *passed {
                        paint_stdout("pass", Style::new().fg(Color::Green))
                    } else {
                        paint_stderr("FAIL", Style::new().fg(Color::Red))
                    };
                    println!(
                        "  {:<name_w$}  {:<label_w$}  {result}",
                        paint_stdout(&s.name, Style::new().fg(Color::Cyan)),
                        label,
                    );
                }
            }
        } else {
            // Summary: one row per suite.
            println!(
                "  {:<name_w$}  {:>6}  {:>6}  {:>5}  Time",
                "Suite", "Passed", "Failed", "Tests"
            );
            println!("  {}  ------  ------  -----  --------", "-".repeat(name_w));
            for s in &suite_summaries {
                let passed_col = paint_stdout(
                    format!("{:>6}", s.passed),
                    if s.failed == 0 {
                        Style::new().fg(Color::Green)
                    } else {
                        Style::new().fg(Color::BrightBlack)
                    },
                );
                let failed_col = if s.failed > 0 {
                    paint_stderr(format!("{:>6}", s.failed), Style::new().fg(Color::Red))
                } else {
                    paint_stdout(format!("{:>6}", 0), Style::new().fg(Color::BrightBlack))
                };
                println!(
                    "  {:<name_w$}  {passed_col}  {failed_col}  {:>5}  {:.1}ms",
                    paint_stdout(&s.name, Style::new().fg(Color::Cyan)),
                    s.passed + s.failed,
                    s.elapsed_ms,
                );
            }
        }
    }

    // JSON summary (global --json flag).
    if harness.json_output() {
        let json = serde_json::json!({
            "command": "test",
            "passed": total_passed,
            "failed": total_failed,
            "total": total,
            "failures": failure_labels,
        });
        println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
    }

    // Overall summary line — mirrors forge's "Ran N test suites in X.XXs" line.
    let wall_elapsed = wall_start.elapsed();
    let n_suites = entries.len();
    let suite_noun = if n_suites == 1 { "suite" } else { "suites" };
    let passed_overall = paint_stdout(total_passed.to_string(), Style::new().fg(Color::Green));
    let failed_overall = if total_failed > 0 {
        paint_stderr(total_failed.to_string(), Style::new().fg(Color::Red))
    } else {
        paint_stdout("0", Style::new().fg(Color::BrightBlack))
    };
    println!(
        "\nRan {n_suites} test {suite_noun} in {wall_elapsed:.2?}: {passed_overall} passed, {failed_overall} failed ({total} total tests)",
    );

    if total_failed > 0 {
        return Err(crate::CliError::TestFailure);
    }
    Ok(())
}

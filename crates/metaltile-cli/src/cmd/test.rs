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
//! Suite result: ok. 3 passed; 0 failed
//!
//! Ran 2 test suites in 57.46ms: 3 tests passed, 1 failed (4 total tests)
//! ```
//!
//! This command spawns `__tile_runner test` as a subprocess and streams
//! `ProtocolMessage` JSON lines. All inventory lookup happens in the runner
//! process where `metaltile-std` is linked.

use std::time::Instant;

use metaltile_core::protocol::{ProtocolMessage, TestResult as ProtoTestResult};

use crate::{
    TestArgs,
    project_runner::{ProjectRunner, RunnerInvocation},
    term::{Color, Style, paint_stderr, paint_stdout},
};

/// `TileCommand` wrapper for `tile test`.
pub struct TestCommand<'a>(pub &'a TestArgs);

impl<'a> super::TileCommand for TestCommand<'a> {
    fn run(&self, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
        run(self.0, harness)
    }
}

pub fn run(args: &TestArgs, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
    let _span = tracing::info_span!("test", filter = ?args.filter_args.filter).entered();

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

    let inv = RunnerInvocation {
        command: "test".into(),
        filter: filter_args.filter.clone(),
        ..Default::default()
    };

    let mut results: Vec<ProtoTestResult> = Vec::new();
    let mut error_msgs: Vec<(String, String, String)> = Vec::new();
    let wall_start = Instant::now();

    let runner_ok = ProjectRunner::new(harness).run_streaming(&inv, |msg| match msg {
        ProtocolMessage::Start { total, .. } => {
            println!(
                "{} {}",
                paint_stdout("tile test", Style::new().fg(Color::Cyan).bold()),
                paint_stdout(format!("({total} items)"), Style::new().fg(Color::BrightBlack)),
            );
        },
        ProtocolMessage::TestResult(tr) => {
            results.push(tr);
        },
        ProtocolMessage::ProtocolError { name, dtype, message } => {
            error_msgs.push((name, dtype, message));
        },
        _ => {},
    });

    if !runner_ok && results.is_empty() && error_msgs.is_empty() {
        return Err(crate::CliError::Other("__tile_runner failed".into()));
    }

    if results.is_empty() && error_msgs.is_empty() {
        if let Some(pattern) = &filter_args.filter {
            eprintln!(
                "{} no tests matched filter {pattern:?}",
                paint_stderr("warning:", Style::new().fg(Color::Yellow).bold()),
            );
        } else {
            eprintln!(
                "{} no #[test_kernel] tests registered",
                paint_stderr("warning:", Style::new().fg(Color::Yellow).bold()),
            );
        }
        return Ok(());
    }

    // Group consecutive results by kernel name to produce forge-style suite blocks.
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, r) in results.iter().enumerate() {
        match groups.last_mut() {
            Some((name, idxs)) if name == &r.name => idxs.push(i),
            _ => groups.push((r.name.clone(), vec![i])),
        }
    }

    let mut total_passed = 0usize;
    let mut total_failed = 0usize;
    let mut total_skipped = 0usize;
    let mut failure_lines: Vec<String> = Vec::new();

    for (name, idxs) in &groups {
        let n = idxs.len();
        let noun = if n == 1 { "test" } else { "tests" };
        println!(
            "\nRan {n} {noun} for {}",
            paint_stdout(name, Style::new().fg(Color::Cyan).bold()),
        );

        let mut suite_passed = 0usize;
        let mut suite_failed = 0usize;
        let mut suite_skipped = 0usize;

        for &i in idxs {
            let r = &results[i];
            let label = format!("{} [{}]", r.name, r.dtype);
            if r.skipped {
                suite_skipped += 1;
                total_skipped += 1;
                println!(
                    "{}  {}",
                    paint_stdout("[SKIP]", Style::new().fg(Color::Yellow).bold()),
                    paint_stdout(&label, Style::new().fg(Color::BrightBlack)),
                );
            } else if r.passed {
                suite_passed += 1;
                total_passed += 1;
                println!(
                    "{}  {}  {}",
                    paint_stdout("[PASS]", Style::new().fg(Color::Green).bold()),
                    paint_stdout(&label, Style::new().fg(Color::BrightWhite)),
                    paint_stdout(
                        format!("(err={:.2e})", r.max_err),
                        Style::new().fg(Color::BrightBlack),
                    ),
                );
            } else {
                suite_failed += 1;
                total_failed += 1;
                let line = format!(
                    "{}  {}",
                    paint_stdout(
                        format!("[FAIL: err={:.2e}]", r.max_err),
                        Style::new().fg(Color::Red).bold(),
                    ),
                    paint_stdout(&label, Style::new().fg(Color::BrightWhite)),
                );
                failure_lines.push(line.clone());
                println!("{line}");
                if args.fail_fast {
                    break;
                }
            }
        }

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
        let skipped_note = if suite_skipped > 0 {
            format!(
                "; {}",
                paint_stdout(format!("{suite_skipped} skipped"), Style::new().fg(Color::Yellow),)
            )
        } else {
            String::new()
        };
        println!(
            "Suite result: {result_word}. {passed_paint} passed; {failed_paint} failed{skipped_note}"
        );

        if args.fail_fast && suite_failed > 0 {
            break;
        }
    }

    // Protocol-level errors (GPU init failure, compile errors, etc.).
    for (name, dtype, msg) in &error_msgs {
        eprintln!(
            "{} {} [{}]: {}",
            paint_stderr("[error]", Style::new().fg(Color::Red).bold()),
            name,
            dtype,
            msg,
        );
        total_failed += 1;
    }

    if !failure_lines.is_empty() {
        println!("\nFailing tests:");
        for line in &failure_lines {
            println!("  {line}");
        }
    }

    let wall_elapsed = wall_start.elapsed();
    let n_suites = groups.len();
    let suite_noun = if n_suites == 1 { "suite" } else { "suites" };
    let total = total_passed + total_failed + total_skipped;
    let skipped_overall = if total_skipped > 0 {
        format!(
            ", {}",
            paint_stdout(format!("{total_skipped} skipped"), Style::new().fg(Color::Yellow),)
        )
    } else {
        String::new()
    };
    println!(
        "\nRan {n_suites} test {suite_noun} in {wall_elapsed:.2?}: {} passed, {} failed{skipped_overall} ({total} total tests)",
        paint_stdout(total_passed.to_string(), Style::new().fg(Color::Green)),
        if total_failed > 0 {
            paint_stderr(total_failed.to_string(), Style::new().fg(Color::Red))
        } else {
            paint_stdout("0", Style::new().fg(Color::BrightBlack))
        },
    );

    if total_failed > 0 {
        return Err(crate::CliError::TestFailure);
    }
    Ok(())
}

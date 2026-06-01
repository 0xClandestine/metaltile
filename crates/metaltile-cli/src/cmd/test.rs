//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile test` — run `#[test_kernel]` correctness setups against a CPU oracle.
//!
//! Iterates the `KernelTest` inventory and dispatches each setup in-process via
//! the shared name-keyed runner, comparing GPU output to the expected buffers
//! within each test's tolerance. Replaces the former
//! `tests/*_gpu_correctness.rs` suite (removed in #240; now in-source
//! `#[test_kernel]`s).

use metaltile::runner::run_kernel_test;

use crate::{
    TestArgs,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

/// `TileCommand` wrapper for `tile test`.
pub struct TestCommand<'a>(pub &'a TestArgs);

impl<'a> super::TileCommand for TestCommand<'a> {
    fn run(&self, _harness: &crate::harness::Harness) -> Result<(), crate::CliError> { run(self.0) }
}

pub fn run(args: &TestArgs) -> Result<(), crate::CliError> {
    let _span = tracing::info_span!("test", filter = ?args.filter).entered();
    let filter = &args.filter;

    let ctx = match metaltile::Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite)),
            );
            return Err(crate::CliError::GpuInit(e.to_string()));
        },
    };

    println!("{}", paint_stdout("tile test", Style::new().fg(Color::Cyan).bold()));

    let mut total = 0usize;
    let mut passed = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut matched_filter = false;

    for entry in metaltile::harness::registry::all_tests() {
        let t = entry.test();
        if !matches_filter(filter.as_deref(), t.name()) {
            continue;
        }
        matched_filter = true;
        for &dt in t.dtypes() {
            let setup = t.setup(dt);
            let tol = t.tolerance(dt);
            total += 1;
            let label = format!("{} [{dt}]", t.name());
            match run_kernel_test(&ctx, &setup, tol) {
                Ok(o) if o.passed => {
                    passed += 1;
                    println!(
                        "  {} {}  {}",
                        paint_stdout("✓", Style::new().fg(Color::Green).bold()),
                        paint_stdout(&label, Style::new().fg(Color::BrightWhite)),
                        paint_stdout(
                            format!("max|Δ|={:.2e}", o.max_abs_err),
                            Style::new().fg(Color::BrightBlack),
                        ),
                    );
                },
                Ok(o) => {
                    failures.push(label.clone());
                    println!(
                        "  {} {}  {}",
                        paint_stdout("✗", Style::new().fg(Color::Red).bold()),
                        paint_stdout(&label, Style::new().fg(Color::BrightWhite)),
                        paint_stdout(
                            format!("max|Δ|={:.2e} > tol {tol:.2e}", o.max_abs_err),
                            Style::new().fg(Color::Red),
                        ),
                    );
                },
                Err(e) => {
                    failures.push(label.clone());
                    println!(
                        "  {} {}  {}",
                        paint_stdout("✗", Style::new().fg(Color::Red).bold()),
                        paint_stdout(&label, Style::new().fg(Color::BrightWhite)),
                        paint_stderr(e, Style::new().fg(Color::Red)),
                    );
                },
            }
        }
    }

    if total == 0 {
        if let Some(pattern) = filter {
            let _ = matched_filter;
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    format!("No tests matched --filter {pattern:?}"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    "No #[test_kernel] tests registered",
                    Style::new().fg(Color::BrightWhite)
                ),
            );
        }
        return Ok(());
    }

    let style = if failures.is_empty() {
        Style::new().fg(Color::Green).bold()
    } else {
        Style::new().fg(Color::Red).bold()
    };
    println!("\n  {}", paint_stdout(format!("{passed}/{total} passed"), style));

    if !failures.is_empty() {
        return Err(crate::CliError::Other(format!("{} test(s) failed", failures.len())));
    }
    Ok(())
}

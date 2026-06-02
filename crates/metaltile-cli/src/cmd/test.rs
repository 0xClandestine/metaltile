//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile test` — run `#[test_kernel]` correctness setups against a CPU oracle.
//!
//! Iterates the `KernelTest` inventory and dispatches each setup in-process via
//! the shared name-keyed runner, comparing GPU output to the expected buffers
//! within each test's tolerance. Replaces the former
//! `tests/*_gpu_correctness.rs` suite (removed in #240; now in-source
//! `#[test_kernel]`s).
//!
//! ## Two-phase execution
//!
//! CPU oracle work (`t.setup(dt)` — generating expected buffers) is run in
//! parallel across all (test, dtype) pairs via rayon.  GPU dispatch
//! (`run_kernel_test`) is then performed serially on the main thread, since
//! `Context` wraps a non-`Send` Metal device.

use metaltile::runner::run_kernel_test;
use rayon::prelude::*;

use crate::{
    FilterSpec,
    TestArgs,
    term::{Color, Style, paint_stderr, paint_stdout},
};

/// `TileCommand` wrapper for `tile test`.
pub struct TestCommand<'a>(pub &'a TestArgs);

impl<'a> super::TileCommand for TestCommand<'a> {
    fn run(&self, _harness: &crate::harness::Harness) -> Result<(), crate::CliError> { run(self.0) }
}

pub fn run(args: &TestArgs) -> Result<(), crate::CliError> {
    let _span = tracing::info_span!("test", filter = ?args.filter_args.filter).entered();
    let spec = FilterSpec::from_args(&args.filter_args);

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

    // Collect all matching entries up front so we can detect empty results
    // before starting any work and so rayon can index into the slice.
    let entries: Vec<_> = metaltile::harness::registry::all_tests()
        .filter(|entry| spec.matches(entry.test().name(), entry.file()))
        .collect();

    if entries.is_empty() {
        if let Some(pattern) = &args.filter_args.filter {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    format!("No tests matched filter {pattern:?}"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else if !spec.is_empty() {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    "No tests matched the given filter flags",
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    "No #[test_kernel] tests registered",
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        }
        return Ok(());
    }

    // Phase 1 (parallel): run CPU oracle for every (entry, dtype) pair.
    // `t.setup(dt)` computes expected output buffers on the CPU — no GPU
    // involvement — so all pairs can run concurrently.  Results are collected
    // in input order (rayon preserves order with par_iter + collect).
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

    // Phase 2 (serial): GPU dispatch + comparison.
    // `run_kernel_test` uses the Metal `Context` which is not `Send`, so all
    // dispatches happen on the main thread in deterministic order.
    let mut total = 0usize;
    let mut passed = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for group in work {
        for (label, setup, tol) in group {
            total += 1;
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

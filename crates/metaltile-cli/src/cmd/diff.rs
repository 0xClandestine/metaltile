//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile diff` — Compare bench results against a saved baseline.
//!
//! Delegates the core diff computation to [`crate::diff::render`].

use std::path::Path;

use serde_json::Value;

use crate::{
    CliError,
    FilterFlags,
    bench::runner::{BenchRunOpts, BenchRunner},
    diff::{RenderOutcome},
    term::{Color, Style, paint_stderr},
};

#[derive(clap::Args, Debug)]
pub struct DiffArgs {
    /// Baseline JSON file to compare against.
    #[arg(value_hint = clap::ValueHint::FilePath)]
    pub baseline: std::path::PathBuf,
    /// Current results JSON file (re-runs bench if omitted).
    #[arg(value_hint = clap::ValueHint::FilePath)]
    pub current: Option<std::path::PathBuf>,
    #[command(flatten)]
    pub filters: FilterFlags,
    /// Highlight regressions larger than this percentage (default: 5).
    #[arg(long = "threshold", default_value = "5.0")]
    pub threshold: f64,
    /// Sort rows by: name, delta, pct, or regression.
    #[arg(long = "sort", default_value = "name")]
    pub sort: String,
    /// Show only regressions.
    #[arg(long = "only-regressions")]
    pub only_regressions: bool,
    /// Show only improvements.
    #[arg(long = "only-improvements")]
    pub only_improvements: bool,
}

impl DiffArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let _span = tracing::info_span!(
            "diff",
            baseline = ?args.baseline,
            threshold = args.threshold,
        )
        .entered();

        let baseline = load_results(&args.baseline, "baseline")?;

        let current = if let Some(path) = &args.current {
            load_results(path.as_ref(), "current")?
        } else {
            eprintln!(
                "  {}",
                paint_stderr(
                    "Running bench suite for current...",
                    Style::new().fg(Color::Cyan).bold()
                ),
            );
            BenchRunner::run_results(&BenchRunOpts {
                filter: args.filters.filter.clone(),
                json_path: None,
            })?
        };

        let opts = crate::diff::RenderOpts {
            filter: args.filters.filter.as_deref(),
            threshold: args.threshold,
            sort: &args.sort,
            only_regressions: args.only_regressions,
            only_improvements: args.only_improvements,
            heading: Some("tile diff"),
        };
        let RenderOutcome { regressions, total_rows } =
            crate::diff::render(&baseline, &current, &opts);

        if total_rows == 0 {
            println!(
                "  {}",
                paint_stderr("No matching results to diff.", Style::new().fg(Color::BrightBlack)),
            );
            return Ok(());
        }

        if regressions > 0 {
            return Err(CliError::Regression(regressions));
        }
        Ok(())
    }
}

fn load_results(path: &Path, label: &str) -> Result<Vec<Value>, CliError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                format!("cannot read {label} {}: {e}", path.display()),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        CliError::Io(e)
    })?;
    let json: Value = serde_json::from_str(&content).map_err(|e| {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(format!("invalid {label} JSON: {e}"), Style::new().fg(Color::BrightWhite)),
        );
        CliError::Json(e)
    })?;
    // Support both bench_dump format (results array at top level) and snapshot format
    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        Ok(results.clone())
    } else if let Some(results) = json.as_array() {
        Ok(results.clone())
    } else {
        eprintln!(
            "{} {}",
            paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
            paint_stderr(
                format!("{label} has no 'results' array"),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        Err(CliError::Config(format!("{label} has no 'results' array")))
    }
}

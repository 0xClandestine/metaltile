//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Service to run `tile bench` as a subprocess and capture its JSON output.
//!
//! Eliminates the duplicated subprocess-spawn logic in `diff.rs` and `snap.rs`.

use std::{path::PathBuf, process::Command};

use serde_json::Value;

use crate::{
    error::CliError,
    term::{Color, Style, paint_stderr},
};

/// Options for [`BenchRunner::run`].
pub struct BenchRunOpts {
    /// Optional filter string forwarded as `--filter`.
    pub filter: Option<String>,
    /// Optional JSON output path (overrides the temp file).
    pub json_path: Option<PathBuf>,
}

/// Runs `tile bench` as a subprocess and captures structured results.
///
/// This service exists because both `tile snap` and `tile diff` need to
/// re-run the bench suite and parse the JSON output.
pub struct BenchRunner;

impl BenchRunner {
    /// Run `tile bench --json <tmpfile>`, wait for completion, and return
    /// the parsed JSON value (the full `BenchReport` document).
    pub fn run(opts: &BenchRunOpts) -> Result<Value, CliError> {
        let temp_path = opts.json_path.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!(".tile-bench-runner-{}.json", std::process::id()))
        });

        let mut cmd = Command::new(std::env::current_exe().map_err(CliError::Io)?);
        cmd.arg("bench").arg("--json").arg(
            temp_path
                .to_str()
                .ok_or_else(|| CliError::HarnessProtocol("non-UTF8 temp path".into()))?,
        );

        if let Some(f) = &opts.filter {
            cmd.arg("--filter").arg(f);
        }

        let status = cmd
            .spawn()
            .map_err(|e| CliError::Subprocess(format!("spawn tile bench: {e}")))?
            .wait()
            .map_err(|e| CliError::Subprocess(format!("tile bench wait: {e}")))?;

        if !status.success() {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr("bench suite failed", Style::new().fg(Color::BrightWhite)),
            );
            let _ = std::fs::remove_file(&temp_path);
            return Err(CliError::Subprocess("bench suite failed".into()));
        }

        let content = std::fs::read_to_string(&temp_path).map_err(|e| {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    format!("cannot read bench results: {e}"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            CliError::Io(e)
        })?;

        // Clean up temp file if we created one.
        if opts.json_path.is_none() {
            let _ = std::fs::remove_file(&temp_path);
        }

        let json: Value = serde_json::from_str(&content).map_err(|e| {
            eprintln!(
                "{} {}",
                paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    format!("invalid bench JSON: {e}"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            CliError::Json(e)
        })?;

        Ok(json)
    }

    /// Convenience wrapper: run bench, extract the `results` array.
    pub fn run_results(opts: &BenchRunOpts) -> Result<Vec<Value>, CliError> {
        let json = Self::run(opts)?;
        json.get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .ok_or_else(|| CliError::HarnessProtocol("bench output has no 'results' array".into()))
    }
}

/// Print a "running bench..." message to stderr.
pub fn print_running_bench() {
    eprintln!("  {}", paint_stderr("Running bench suite...", Style::new().fg(Color::Cyan).bold()),);
}

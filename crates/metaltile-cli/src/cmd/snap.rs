//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile snap` — Save current bench results as a regression baseline.

use metaltile_core::GpuFamily;
use serde::Serialize;
use serde_json::Value;

use crate::{
    CliError,
    FilterFlags,
    bench::runner::{BenchRunOpts, BenchRunner, print_running_bench},
    git::{GitProvider, RealGit},
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct SnapArgs {
    #[command(flatten)]
    pub filters: FilterFlags,
    /// Write snapshot to <file> (default: .tile-snapshots/<timestamp>.json).
    #[arg(long = "out", short = 'o', value_hint = clap::ValueHint::FilePath)]
    pub out: Option<std::path::PathBuf>,
    /// Promote an existing JSON bench result file instead of re-running.
    #[arg(long = "from", value_hint = clap::ValueHint::FilePath)]
    pub from: Option<std::path::PathBuf>,
    /// Attach a human-readable note to the snapshot.
    #[arg(long = "note")]
    pub note: Option<String>,
}

#[derive(Serialize)]
struct Snapshot {
    device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_sha: Option<String>,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
    results: Vec<Value>,
}

impl SnapArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let _span = tracing::info_span!("snap", out = ?args.out, from = ?args.from).entered();
        let note = &args.note;

        let out_path: std::path::PathBuf = match &args.out {
            Some(p) => p.clone(),
            None => {
                let date = iso_now();
                std::path::PathBuf::from(format!(".tile-snapshots/{}.json", date))
            },
        };

        let results_json: Value = if let Some(from) = &args.from {
            let content = std::fs::read_to_string(from).map_err(|e| {
                eprintln!(
                    "{} {}",
                    paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                    paint_stderr(
                        format!("cannot read {}: {e}", from.display()),
                        Style::new().fg(Color::BrightWhite)
                    ),
                );
                CliError::Io(e)
            })?;
            serde_json::from_str(&content).map_err(|e| {
                eprintln!(
                    "{} {}",
                    paint_stderr("Error:", Style::new().fg(Color::Red).bold()),
                    paint_stderr(format!("invalid JSON: {e}"), Style::new().fg(Color::BrightWhite)),
                );
                CliError::Json(e)
            })?
        } else {
            print_running_bench();
            BenchRunner::run(&BenchRunOpts {
                filter: args.filters.filter.clone(),
                json_path: None,
            })?
        };

        let device =
            results_json.get("device").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

        let mut results: Vec<Value> =
            results_json.get("results").and_then(|v| v.as_array()).cloned().unwrap_or_default();

        // Apply filter
        if let Some(f) = &args.filters.filter {
            let f_lower = f.to_ascii_lowercase();
            results.retain(|r| {
                r.get("op")
                    .and_then(|v| v.as_str())
                    .map(|op| op.to_ascii_lowercase().contains(&f_lower))
                    .unwrap_or(false)
            });
        }

        let git = RealGit;
        let git_sha = git.short_sha();
        let gpu_family = GpuFamily::from_device_name(&device).code().map(|s| s.to_string());

        let timestamp = iso_now();
        let result_count = results.len();
        let note_suffix = note.as_ref().map(|n| format!(", \"{n}\"")).unwrap_or_default();

        let snapshot =
            Snapshot { device, gpu_family, git_sha, timestamp, note: note.clone(), results };

        let dir = out_path.parent().unwrap_or(std::path::Path::new("."));
        std::fs::create_dir_all(dir).map_err(|e| {
            eprintln!("cannot create directory: {e}");
            CliError::Io(e)
        })?;

        let json = serde_json::to_string_pretty(&snapshot).map_err(CliError::Json)?;
        std::fs::write(&out_path, &json).map_err(|e| {
            eprintln!("cannot write snapshot: {e}");
            CliError::Io(e)
        })?;

        println!(
            "  {} {}  ({} results{})",
            paint_stdout("Saved →", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(out_path.display().to_string(), Style::new().fg(Color::BrightWhite)),
            result_count,
            note_suffix,
        );
        Ok(())
    }
}

/// RFC 3339 timestamp using standard library types.
fn iso_now() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();

    // Calculate date from Unix timestamp using the civil-time algorithm.
    // Based on Howard Hinnant's public-domain algorithm.
    let z = (secs / 86400) as i64;
    let era = if z >= 0 { z } else { z - 146096 };
    let era_days = era % 146097;
    let era_offset = if era_days == 146096 { 0i64 } else { era_days };
    let yoe = (era_offset - era_offset / 1460 + era_offset / 36524 - era_offset / 146096) / 365;
    let y = yoe + 1970 + (era / 146097) * 400;
    let doy = era_offset - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let month = if m <= 12 { m } else { m - 12 };
    let year = if month <= 2 { y + 1 } else { y };

    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let mins = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    format!("{year:04}-{month:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::iso_now;

    #[test]
    fn iso_now_returns_valid_format() {
        let s = iso_now();
        assert_eq!(s.len(), 20, "expected ISO 8601 length, got {s}");
        assert!(s.ends_with('Z'), "expected Z suffix, got {s}");
    }
}

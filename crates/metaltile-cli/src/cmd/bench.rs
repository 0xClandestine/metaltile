//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile bench` — Benchmark MetalTile kernels (latency / GB/s / GFLOP·s /
//! roofline). The MLX reference A/B (speed + output-equivalence) is opt-in via
//! `--mlx`; by default only the metaltile kernels are benched.

use metaltile_core::protocol::{BenchResult as ProtoBenchResult, ProtocolMessage};
use serde_json::Value;

use crate::{
    BenchArgs,
    FilterSpec,
    cmd::diff as diff_cmd,
    git,
    project_runner::{ProjectRunner, RunnerInvocation},
    suite_printer::SuitePrinter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &BenchArgs, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
    let verbose = harness.verbosity();
    let runs = args.runs.unwrap_or_else(|| harness.config.effective_runs());
    let warmup_runs = args.warmup.unwrap_or_else(|| harness.config.effective_warmup_runs());
    let _span = tracing::info_span!("bench", filter = ?args.filter_args.filter, verbose).entered();
    let json_out = &args.out;
    // Merge positional path into filter: '/' → match-path glob, else → --filter.
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

    // Refuse to bench on a dirty tree: a stale `target/` binary against
    // a dirty source tree silently decouples the numbers from any
    // commit SHA we'd record in a snapshot. `working_tree_dirty()`
    // returns None outside a repo — skip the check there.
    if !args.allow_dirty
        && let Some(true) = git::working_tree_dirty()
    {
        let files = git::list_dirty_files();
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
        return Err(crate::CliError::Other("uncommitted changes".into()));
    }

    // Spawn __tile_runner bench and stream protocol results.
    let inv = RunnerInvocation {
        command: "bench".into(),
        filter: filter_args.filter.clone(),
        warmup_runs: Some(warmup_runs),
        runs: Some(runs),
        profile: verbose >= 1,
        ..Default::default()
    };

    let mut all: Vec<ProtoBenchResult> = Vec::new();
    let mut matched_filter = false;
    let mut device_name = String::new();

    let mut printer = SuitePrinter::new(true);
    printer.set_verbose(verbose);

    let runner_ok = ProjectRunner::new(harness).run_streaming(&inv, |msg| match msg {
        ProtocolMessage::Start { device, total, .. } => {
            device_name = device.unwrap_or_default();
            let dev_part =
                if device_name.is_empty() { String::new() } else { format!("· {} ", device_name) };
            println!(
                "{} {}{}",
                paint_stdout("tile bench", Style::new().fg(Color::Cyan).bold()),
                paint_stdout(&dev_part, Style::new().fg(Color::BrightBlack)),
                paint_stdout(
                    format!("warmup={warmup_runs} runs={runs}  ({total} items)"),
                    Style::new().fg(Color::BrightBlack),
                ),
            );
        },
        ProtocolMessage::BenchResult(br) => {
            if spec.matches_name(&br.name) {
                matched_filter = true;
                printer.print_bench_result(&br);
            }
            all.push(br);
        },
        ProtocolMessage::ProtocolError { name, dtype, message } => {
            eprintln!(
                "{} {} [{}]: {}",
                paint_stderr("[error]", Style::new().fg(Color::Red).bold()),
                name,
                dtype,
                message,
            );
        },
        _ => {},
    });

    if !runner_ok && all.is_empty() {
        return Err(crate::CliError::Other("__tile_runner failed".into()));
    }

    if all.is_empty() {
        if let Some(pattern) = &filter_args.filter {
            if matched_filter {
                eprintln!(
                    "{} {}",
                    paint_stderr("[error]", Style::new().fg(Color::Red).bold()),
                    paint_stderr(
                        format!(
                            "Kernel matched filter {pattern:?} but all shapes failed to compile or run"
                        ),
                        Style::new().fg(Color::BrightWhite),
                    ),
                );
            } else {
                eprintln!(
                    "{} {}",
                    paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                    paint_stderr(
                        format!("No benchmarks matched filter {pattern:?}"),
                        Style::new().fg(Color::BrightWhite),
                    ),
                );
            }
        } else if !spec.is_empty() {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr(
                    "No benchmarks matched the given filter flags",
                    Style::new().fg(Color::BrightWhite),
                ),
            );
        } else {
            eprintln!(
                "{} {}",
                paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
                paint_stderr("No benchmarks ran", Style::new().fg(Color::BrightWhite)),
            );
        }
        return Ok(());
    }

    printer.finish();

    // Counters derived from protocol BenchResult.
    let impl_count = all.iter().filter(|r| r.mt_gbps > 0.0).count();
    let equiv_fail = all.iter().filter(|r| !r.correct).count();
    let checked_count = all.len();
    let equiv_pass = all.iter().filter(|r| r.correct).count();
    let avg_pct: Option<f64> = {
        let valid: Vec<f64> = all.iter().filter_map(|r| r.mt_pct).collect();
        if valid.is_empty() { None } else { Some(valid.iter().sum::<f64>() / valid.len() as f64) }
    };

    // Summary — compact single line.
    let mut parts: Vec<String> = Vec::new();
    let sep = format!("  {}  ", paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()));

    parts.push(format!(
        "{} impl",
        paint_stdout(impl_count.to_string(), Style::new().fg(Color::Green).bold()),
    ));
    if let Some(p) = avg_pct {
        parts.push(format!("avg {}", paint_stdout(format!("{p:.0}% MT"), pct_style(p)),));
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

    println!("\n  {}", parts.join(&sep));

    if equiv_fail > 0 {
        println!(
            "  {} {}",
            paint_stdout("Failures:", Style::new().fg(Color::Red).bold()),
            paint_stdout(equiv_fail.to_string(), Style::new().fg(Color::Red).bold()),
        );
    }
    println!();

    if args.diff {
        try_auto_diff(
            &device_name,
            &all,
            filter_args.filter.as_deref(),
            args.baseline_ref.as_deref(),
        );
    }

    if let Some(path) = json_out {
        save_json(&device_name, &all, path);
    }

    if equiv_fail > 0 {
        return Err(crate::CliError::TestFailure);
    }
    Ok(())
}

/// Resolve a baseline file from the target branch and diff the
/// just-finished bench against it. Best-effort: any failure (no git
/// repo, no resolved ref, no baseline file at that ref, etc.) logs a
/// one-line skip note and returns. Never aborts the bench.
fn try_auto_diff(
    device: &str,
    results: &[ProtoBenchResult],
    filter: Option<&str>,
    baseline_ref_override: Option<&str>,
) {
    let slug = chip_slug(device);
    let baseline_path = format!("baselines/{slug}.json");

    let candidates: Vec<&str> = match baseline_ref_override {
        Some(r) => vec![r],
        None => vec!["origin/dev", "upstream/dev", "dev"],
    };
    let Some(reference) = git::resolve_baseline_ref(&candidates) else {
        log_skip(&format!(
            "baseline auto-diff: no target-branch ref ({}) — skipping",
            candidates.join("/")
        ));
        return;
    };
    let Some(sha) = git::merge_base_with(&reference) else {
        log_skip(&format!("baseline auto-diff: merge-base HEAD..{reference} failed — skipping"));
        return;
    };
    let Some(content) = git::show_file_at(&sha, &baseline_path) else {
        log_skip(&format!(
            "baseline auto-diff: no {baseline_path} at {reference} ({}…) — skipping",
            sha.chars().take(7).collect::<String>()
        ));
        return;
    };

    let baseline_json: Value = match serde_json::from_str(&content) {
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

    let current_rows: Vec<Value> = results.iter().map(result_to_value).collect();

    let short_sha: String = sha.chars().take(7).collect();
    let heading = format!("tile bench · diff vs {reference} @ {short_sha} ({baseline_path})");
    let opts = diff_cmd::RenderOpts {
        heading: Some(&heading),
        sort: "regression",
        filter,
        ..diff_cmd::RenderOpts::default()
    };
    let outcome = diff_cmd::render(&baseline_rows, &current_rows, &opts);
    if outcome.total_rows == 0 {
        log_skip(&format!(
            "baseline auto-diff: no overlapping rows with {baseline_path} at {reference}"
        ));
    }
}

/// Lowercase + collapse whitespace runs into a single dash, dropping
/// any character that isn't alphanumeric or `-`. Yields slugs like
/// `apple-m5-max` from `Apple M5 Max`, matching the naming convention
/// established by `baselines/apple-m5-max.json`.
fn chip_slug(device: &str) -> String {
    let mut out = String::with_capacity(device.len());
    let mut prev_dash = false;
    for ch in device.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            out.push(lowered);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// Convert a `ProtoBenchResult` to the legacy JSON schema used by the
/// diff/baseline system: `op`/`subop`/`shape`/`metric`/`ref`/`mt`.
///
/// The kernel `name` (e.g. `"unary/exp"`) is split on the first `'/'`:
/// the leading component becomes `op`, any remainder becomes `subop`.
fn result_to_value(r: &ProtoBenchResult) -> Value {
    let mut obj = serde_json::Map::new();
    let (op, subop) = match r.name.split_once('/') {
        Some((a, b)) => (a, Some(b)),
        None => (r.name.as_str(), None),
    };
    obj.insert("op".into(), Value::from(op));
    if let Some(sub) = subop {
        obj.insert("subop".into(), Value::from(sub));
    }
    obj.insert("shape".into(), Value::from(r.shape.as_str()));
    obj.insert("metric".into(), Value::from("GB/s"));
    obj.insert("ref".into(), r.ref_gbps.map(Value::from).unwrap_or(Value::Null));
    obj.insert("mt".into(), Value::from(r.mt_gbps));
    Value::Object(obj)
}

fn log_skip(msg: &str) {
    eprintln!("  {}", paint_stderr(msg, Style::new().fg(Color::BrightBlack)));
}

fn save_json(device: &str, results: &[ProtoBenchResult], path: &str) {
    use std::io::Write;
    let s = summarize(results);
    let mut out = String::new();
    out.push_str(&format!(
        "{{\"device\":{:?},\"summary\":{{\"total\":{},\"implemented\":{},\"correct\":{}}},\"results\":[\n",
        device, s.total, s.implemented, s.correct,
    ));
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        let (op, subop) = match r.name.split_once('/') {
            Some((a, b)) => (a, Some(b)),
            None => (r.name.as_str(), None),
        };
        out.push_str(&format!(
            "  {}{}\n",
            format_result_row(op, subop, &r.shape, "GB/s", r.ref_gbps, Some(r.mt_gbps)),
            comma
        ));
    }
    out.push_str("]}");
    match std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(".".as_ref()))
        .and_then(|_| std::fs::File::create(path))
        .and_then(|mut f| f.write_all(out.as_bytes()))
    {
        Ok(()) => println!(
            "  {} {}",
            paint_stdout("Saved →", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(path, Style::new().fg(Color::BrightWhite)),
        ),
        Err(e) => eprintln!(
            "  {} {}",
            paint_stderr("save failed:", Style::new().fg(Color::Red).bold()),
            paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite)),
        ),
    }
}

/// Aggregate counts mirroring the terminal banner. Persisted alongside
/// the per-row results in the JSON so CI and dashboards can consume
/// kernel-correctness as a single signal without re-parsing every row.
struct Summary {
    total: usize,
    implemented: usize,
    correct: usize,
}

fn summarize(results: &[ProtoBenchResult]) -> Summary {
    Summary {
        total: results.len(),
        implemented: results.iter().filter(|r| r.mt_gbps > 0.0).count(),
        correct: results.iter().filter(|r| r.correct).count(),
    }
}

/// Format one bench result as a single-line JSON object. The `subop` field is
/// emitted only when present, keeping the schema additive — existing consumers
/// that only read `op`/`shape`/`metric`/`ref`/`mt` are unaffected.
fn format_result_row(
    op: &str,
    subop: Option<&str>,
    shape: &str,
    metric: &str,
    ref_perf: Option<f64>,
    mt_perf: Option<f64>,
) -> String {
    match subop {
        Some(s) => format!(
            "{{\"op\":{:?},\"subop\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}",
            op,
            s,
            shape,
            metric,
            json_f(ref_perf),
            json_f(mt_perf),
        ),
        None => format!(
            "{{\"op\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}",
            op,
            shape,
            metric,
            json_f(ref_perf),
            json_f(mt_perf),
        ),
    }
}

fn json_f(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "null".into())
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

// ── TileCommand impl ──────────────────────────────────────────────────────

pub struct BenchCommand<'a>(pub &'a BenchArgs);

impl<'a> super::TileCommand for BenchCommand<'a> {
    fn run(&self, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
        run(self.0, harness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, dtype: &str, mt_gbps: f64, correct: bool) -> ProtoBenchResult {
        ProtoBenchResult {
            name: name.into(),
            dtype: dtype.into(),
            shape: format!("N=1M {dtype}"),
            mt_gbps,
            ref_gbps: None,
            mt_pct: None,
            correct,
            min_us: 1.0,
            mean_us: 1.0,
            profile: None,
        }
    }

    #[test]
    fn summary_counts_per_category() {
        let implemented_correct = make_result("unary/exp", "f32", 100.0, true);
        let implemented_wrong = make_result("unary/log", "f32", 40.0, false);
        let nyi = make_result("unary/sin", "f32", 0.0, false);

        let s = summarize(&[implemented_correct, implemented_wrong, nyi]);
        assert_eq!(s.total, 3);
        assert_eq!(s.implemented, 2); // exp + log (mt_gbps > 0)
        assert_eq!(s.correct, 1); // only exp
    }

    #[test]
    fn summary_on_empty_input_is_all_zero() {
        let s = summarize(&[]);
        assert_eq!(s.total, 0);
        assert_eq!(s.implemented, 0);
        assert_eq!(s.correct, 0);
    }

    #[test]
    fn result_to_value_splits_name_into_op_subop() {
        let r = make_result("unary/exp", "f32", 325.6, true);
        let v = result_to_value(&r);
        assert_eq!(v["op"], "unary");
        assert_eq!(v["subop"], "exp");
        assert_eq!(v["metric"], "GB/s");
        assert_eq!(v["mt"], 325.6);
        assert!(v["ref"].is_null());
    }

    #[test]
    fn result_to_value_no_subop_for_simple_name() {
        let r = make_result("rms_norm", "f32", 323.9, true);
        let v = result_to_value(&r);
        assert_eq!(v["op"], "rms_norm");
        assert!(v.get("subop").is_none());
    }

    #[test]
    fn json_f_formats_finite_and_none() {
        assert_eq!(json_f(Some(12.345)), "12.345");
        assert_eq!(json_f(Some(0.0)), "0.000");
        assert_eq!(json_f(None), "null");
    }

    #[test]
    fn row_without_subop_matches_legacy_schema() {
        // Pre-existing consumers rely on this exact key set + ordering.
        let row = format_result_row(
            "rms_norm",
            None,
            "B=1024 N=4096 f32",
            "GB/s",
            Some(323.9),
            Some(325.6),
        );
        assert_eq!(
            row,
            r#"{"op":"rms_norm","shape":"B=1024 N=4096 f32","metric":"GB/s","ref":323.900,"mt":325.600}"#,
        );
        assert!(!row.contains("\"subop\""));
    }

    #[test]
    fn row_with_subop_emits_disambiguated_field() {
        // The motivating bug: many `unary` subops collapse to identical
        // (op, shape) tuples in the legacy schema. The `subop` field
        // disambiguates them without breaking schema additively.
        let row =
            format_result_row("unary", Some("sin"), "N=64M f32", "GB/s", Some(544.8), Some(114.5));
        assert_eq!(
            row,
            r#"{"op":"unary","subop":"sin","shape":"N=64M f32","metric":"GB/s","ref":544.800,"mt":114.500}"#,
        );
    }

    #[test]
    fn row_handles_missing_perf_values() {
        let row = format_result_row("sdpa", Some("sdpa_vector"), "H=8 N=2048", "GB/s", None, None);
        assert!(row.contains(r#""ref":null"#));
        assert!(row.contains(r#""mt":null"#));
        assert!(row.contains(r#""subop":"sdpa_vector""#));
    }

    #[test]
    fn row_quotes_strings_containing_special_chars() {
        // Shape strings sometimes embed `=`, spaces, and parens; ensure they
        // round-trip via Debug-quoting so the row is valid JSON.
        let row = format_result_row("foo", None, "k=2 (warm)", "GB/s", Some(1.0), Some(2.0));
        let parsed: serde_json::Value = serde_json::from_str(&row).unwrap();
        assert_eq!(parsed["shape"], "k=2 (warm)");
    }

    // The slug must match the filenames committed under `baselines/`
    // (e.g. `apple-m5-max.json`) so auto-diff can find them.
    #[test]
    fn chip_slug_matches_apple_m5_max_filename() {
        assert_eq!(chip_slug("Apple M5 Max"), "apple-m5-max");
    }

    #[test]
    fn chip_slug_collapses_runs_of_punctuation() {
        // Hypothetical messy device string — make sure we don't emit
        // double dashes or trailing dashes.
        assert_eq!(chip_slug("  Apple  --M1 (Pro)  "), "apple-m1-pro");
        assert_eq!(chip_slug("Apple_M2_Max!"), "apple-m2-max");
    }
}

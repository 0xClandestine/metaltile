//! `tile autotune` — controller for the cache-warming pipeline.
//!
//! Thin wrapper over [`metaltile_autotune::run_autotune`]: parses args,
//! brings up the `GpuRunner` only when `--measure` is set, calls the
//! engine, and renders the colored progress + summary block. All
//! measurement, search, cost-model, and JSONL-export logic lives in
//! `metaltile-autotune` so it stays independently testable.

use std::path::PathBuf;

use metaltile::autotune::Autotuner;
use metaltile_autotune::{
    AutotuneError, AutotuneOptions, AutotuneSummary, KernelTuneResult, collect_training_rows,
    default_training_data_path, write_training_jsonl, write_training_jsonl_to_file,
};
use metaltile_std::spec::{BenchDispatch, BenchSpec};

use crate::{
    AutotuneArgs,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &AutotuneArgs) -> Result<(), crate::CliError> {
    if args.clear {
        return clear_cache();
    }
    if args.dump {
        return dump_cache();
    }
    if args.list {
        return list_kernels(args.filter.as_deref());
    }
    if let Some(dest) = args.export_training_data.as_deref() {
        return export_training_data(dest);
    }

    let shape_overrides = parse_shape_overrides(&args.shapes)?;

    // Bring up a GpuRunner only when we actually need one — `--measure`
    // is the only mode that dispatches kernels. Static cost runs
    // headless and works fine on machines without Metal (CI).
    let runner = if args.measure {
        Some(metaltile_std::runner::GpuRunner::new().map_err(crate::CliError::GpuInit)?)
    } else {
        None
    };

    let options = AutotuneOptions {
        measure: args.measure,
        quick: args.quick,
        filter: args.filter.clone(),
        shape_overrides: shape_overrides.clone(),
    };

    print_header(args, shape_overrides.as_deref());

    let summary = metaltile_autotune::run_autotune(&options, runner.as_ref(), render_kernel_result)
        .map_err(map_autotune_error)?;

    print_summary(args, &summary);
    Ok(())
}

fn print_header(args: &AutotuneArgs, shape_overrides: Option<&[usize]>) {
    println!(
        "{} {}",
        paint_stdout("tile autotune", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            format!("· cache: {}", Autotuner::default_cache_dir().display()),
            Style::new().fg(Color::BrightBlack),
        ),
    );
    let mode_note = match (args.measure, args.quick) {
        (true, true) =>
            "cost = GPU-measured median μs (Generic + SdpaVector + SdpaPrefill; --quick: 3 warmup + 11 iters).",
        (true, false) =>
            "cost = GPU-measured median μs (Generic + SdpaVector + SdpaPrefill; 20 warmup + 100 iters); other → static.",
        (false, _) => "cost = static occupancy estimate. Pass --measure for real GPU timing.",
    };
    println!(
        "  {} {}",
        paint_stdout("[mode]", Style::new().fg(Color::BrightBlack).bold()),
        paint_stdout(mode_note, Style::new().fg(Color::BrightBlack)),
    );
    if let Some(list) = shape_overrides {
        let ns = list.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        println!(
            "  {} {}",
            paint_stdout("[shapes]", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(
                format!("tuning N ∈ {{{ns}}} — {} entries per (kernel, dtype)", list.len()),
                Style::new().fg(Color::BrightBlack),
            ),
        );
    }
}

fn render_kernel_result(result: &KernelTuneResult) {
    // Per-kernel lines only fire on failure, matching the prior
    // behavior where tuned/measured/estimated only appeared in the
    // closing summary.
    if let Err(e) = &result.outcome {
        let n_tag = result.n_override.map(|n| format!(" (N={n})")).unwrap_or_default();
        eprintln!(
            "  {} {}{}: {}",
            paint_stderr("skip", Style::new().fg(Color::Yellow).bold()),
            paint_stderr(result.kernel_name, Style::new().fg(Color::BrightWhite)),
            paint_stderr(n_tag, Style::new().fg(Color::BrightBlack)),
            paint_stderr(e, Style::new().fg(Color::BrightBlack)),
        );
    }
}

fn print_summary(args: &AutotuneArgs, summary: &AutotuneSummary) {
    let sep = format!("  {}  ", paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()));
    let fallback_segment = if args.measure {
        format!(
            "{sep}{}",
            paint_stdout(
                format!("{} config fallbacks", summary.fallbacks),
                if summary.fallbacks > 0 {
                    Style::new().fg(Color::Yellow)
                } else {
                    Style::new().fg(Color::BrightBlack)
                },
            ),
        )
    } else {
        String::new()
    };
    println!(
        "\n  {tuned}{sep}{measured}{sep}{estimated}{sep}{skipped}{fallback_segment}{sep}{disk}",
        tuned = paint_stdout(
            format!("{} tuned", summary.tuned),
            Style::new().fg(Color::Green).bold(),
        ),
        measured = paint_stdout(
            format!("{} measured", summary.measured),
            if summary.measured > 0 {
                Style::new().fg(Color::Green)
            } else {
                Style::new().fg(Color::BrightBlack)
            },
        ),
        estimated = paint_stdout(
            format!("{} estimated", summary.estimated),
            Style::new().fg(Color::BrightBlack),
        ),
        skipped = paint_stdout(
            format!("{} skipped", summary.skipped),
            if summary.skipped > 0 {
                Style::new().fg(Color::Yellow).bold()
            } else {
                Style::new().fg(Color::BrightBlack)
            },
        ),
        disk = paint_stdout(
            format!("{} entries on disk", summary.cache_entries),
            Style::new().fg(Color::Cyan).bold(),
        ),
    );
}

/// Validate the `--shapes` CLI flag (clap has already parsed it into a
/// `Vec<usize>`). Returns `Ok(None)` when the flag was absent, so the
/// caller can fall back to the legacy single-shape path. Returns
/// `Err` if any N is zero — usize parse already rejects negatives.
fn parse_shape_overrides(shapes: &[usize]) -> Result<Option<Vec<usize>>, crate::CliError> {
    if shapes.is_empty() {
        return Ok(None);
    }
    if let Some(&bad) = shapes.iter().find(|&&n| n == 0) {
        return Err(crate::CliError::Other(format!("--shapes: N must be > 0 (got {bad})")));
    }
    Ok(Some(shapes.to_vec()))
}

/// Emit `Autotuner::export_training_data()` as JSONL — one row per
/// line, in cache-key order so diffs across runs are reviewable.
///
/// `dest` semantics, following the CLI surface:
/// - `""`  → default file `~/.cache/metaltile/training_data.jsonl`
///   (clap's `default_missing_value` fires when the user passes
///   `--export-training-data` with no value).
/// - `"-"` → stdout (good for piping into `jq`, pandas, etc.).
/// - any other value → that path.
fn export_training_data(dest: &str) -> Result<(), crate::CliError> {
    let rows = collect_training_rows();

    if dest == "-" {
        write_training_jsonl(std::io::stdout().lock(), &rows).map_err(map_autotune_error)?;
        return Ok(());
    }

    let path = if dest.is_empty() {
        default_training_data_path()
    } else {
        PathBuf::from(dest)
    };
    write_training_jsonl_to_file(&path, &rows).map_err(map_autotune_error)?;
    println!(
        "  {} {} {}",
        paint_stdout("exported", Style::new().fg(Color::Green).bold()),
        paint_stdout(format!("{} rows →", rows.len()), Style::new().fg(Color::BrightBlack),),
        paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightWhite)),
    );
    Ok(())
}

fn dump_cache() -> Result<(), crate::CliError> {
    let path = Autotuner::default_cache_dir().join("tuning_cache.json");
    if !path.exists() {
        eprintln!(
            "  {} no cache file at {}",
            paint_stderr("[info]", Style::new().fg(Color::BrightBlack).bold()),
            paint_stderr(path.display().to_string(), Style::new().fg(Color::BrightBlack)),
        );
        return Ok(());
    }
    let s = std::fs::read_to_string(&path)?;
    println!("{s}");
    Ok(())
}

fn clear_cache() -> Result<(), crate::CliError> {
    let path = Autotuner::default_cache_dir().join("tuning_cache.json");
    if path.exists() {
        std::fs::remove_file(&path)?;
        println!(
            "  {} {}",
            paint_stdout("cleared", Style::new().fg(Color::Green).bold()),
            paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightWhite)),
        );
    } else {
        println!(
            "  {} no cache file at {}",
            paint_stdout("[info]", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightBlack)),
        );
    }
    Ok(())
}

fn list_kernels(filter: Option<&str>) -> Result<(), crate::CliError> {
    let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
    specs.sort_unstable_by_key(|s| (s.kernel_name,));
    let mut count = 0;
    for spec in specs {
        if !matches_filter(filter, spec.kernel_name) {
            continue;
        }
        let family = Autotuner::infer_family(spec.kernel_name);
        let n_configs = family.config_space().len();
        let dispatch_tag = match spec.dispatch {
            BenchDispatch::Generic => "Generic",
            _ => "non-Generic",
        };
        println!(
            "  {}  →  family={:?}  configs={}  dispatch={}",
            paint_stdout(spec.kernel_name, Style::new().fg(Color::Cyan).bold()),
            family,
            n_configs,
            dispatch_tag,
        );
        count += 1;
    }
    if count == 0 {
        eprintln!(
            "  {} no kernels matched",
            paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
        );
    }
    Ok(())
}

fn map_autotune_error(e: AutotuneError) -> crate::CliError {
    match e {
        AutotuneError::Io(io) => crate::CliError::Io(io),
        AutotuneError::Json(j) => crate::CliError::Json(j),
        AutotuneError::MetalTile(m) => crate::CliError::Other(m.to_string()),
        AutotuneError::Other(s) => crate::CliError::Other(s),
    }
}

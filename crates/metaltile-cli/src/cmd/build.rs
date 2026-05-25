//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile build` — Compile all registered kernels.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use metaltile::harness::BuildRow;
use metaltile_codegen::{
    emit::{self, compile_metallib, write_manifest, write_msl, write_swift_wrappers},
    generator_for_mode,
    passes::{PassStats, PipelineBuilder, run_passes_with_stats},
};
use metaltile_core::bench::types::DType;
use metaltile_core::bench::spec::{BenchSpec, effective_mode};
use metaltile_core::ir::Kernel;

use crate::{
    CliError,
    FilterFlags,
    project::harness::HarnessMessage,
    project::{self, CompileService, RealCompileService},
    term::{Color, Style, paint_stderr, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct BuildArgs {
    #[command(flatten)]
    pub filters: FilterFlags,
    /// Comma-separated list of dtypes to build (f32,f16,bf16).
    #[arg(long = "dtypes")]
    pub dtypes: Option<String>,
    /// Print generated MSL (-v), IR before passes (-vv), IR after each pass (-vvv).
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    pub verbose: u8,
    /// Comma-separated emit targets: msl, metallib, swift, ir, all.
    #[arg(long = "emit")]
    pub emit: Option<String>,
    /// Output directory for emitted artifacts (required when --emit is set).
    #[arg(long = "out", short = 'o', value_hint = clap::ValueHint::DirPath)]
    pub out: Option<PathBuf>,
    /// xcrun SDK to use for metallib compilation (default: macosx).
    #[arg(long = "sdk", default_value = "macosx")]
    pub sdk: String,
    /// Rebuild when source files change (watch mode).
    #[arg(long = "watch")]
    pub watch: bool,
    /// Run the pass pipeline 25× per kernel and print per-pass median wall_us.
    #[arg(long = "time-passes", short = 't')]
    pub time_passes: bool,
}

impl BuildArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let _span = tracing::info_span!("build", filter = ?args.filters.filter, emit = ?args.emit)
            .entered();

        let filters = args.filters.to_filters()?;

        // External-project path
        if crate::project::has_tile_toml() {
            return run_build_external(args);
        }

        if args.time_passes {
            return run_time_passes(&args.filters, args.dtypes.as_deref());
        }

        let emit_kinds: BTreeSet<EmitKind> = match args.emit.as_deref() {
            None => BTreeSet::new(),
            Some(raw) => parse_emit_list(raw)?,
        };

        let out_root: Option<PathBuf> = match (&emit_kinds.is_empty(), &args.out) {
            (true, _) => None,
            (false, Some(p)) => Some(p.clone()),
            (false, None) => {
                eprintln!(
                    "  {} {}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                    paint_stderr(
                        "--emit requires --out <dir>",
                        Style::new().fg(Color::BrightWhite)
                    ),
                );
                return Err(CliError::Config("--emit requires --out <dir>".into()));
            },
        };

        let dtypes_filter: Option<Vec<DType>> = args
            .dtypes
            .as_ref()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse::<DType>().ok()).collect());

        // Collect unique kernel specs.
        let specs: Vec<&'static BenchSpec> = metaltile_core::bench::spec::bench_specs().to_vec();
        let mut kernels: BTreeMap<&str, (&'static BenchSpec, Vec<DType>)> = BTreeMap::new();
        for spec in &specs {
            let entry = kernels.entry(spec.kernel_name).or_insert_with(|| (*spec, Vec::new()));
            for &dt in spec.dtypes {
                if !entry.1.contains(&dt) {
                    entry.1.push(dt);
                }
            }
        }

        let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> = kernels.into_iter().collect();
        sorted.sort_unstable_by_key(|(name, _)| *name);

        // Header.
        println!(
            "{} {}",
            paint_stdout("tile build", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(
                format!("· {} kernels", sorted.len()),
                Style::new().fg(Color::BrightBlack)
            ),
        );

        let name_w = sorted.iter().map(|(n, _)| n.len()).max().unwrap_or(20).clamp(8, 48);
        let dt_w = sorted
            .iter()
            .map(|(_, (_, dtypes))| {
                dtypes.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/").len()
            })
            .max()
            .unwrap_or(12)
            .clamp(8, 24);

        let sep = col_sep();
        let bold = Style::new().fg(Color::BrightWhite).bold();
        let hdr = format!(
            "  {} {} {} {} {}",
            paint_stdout(pad_left("Kernel", name_w), bold),
            sep,
            paint_stdout(pad_left("Dtypes", dt_w), bold),
            sep,
            paint_stdout(pad_right("ok", 2), bold),
        );
        println!("{hdr}");

        let total_w = 4 + name_w + 3 + dt_w + 3 + 2;
        let sep_line = paint_stdout("─".repeat(total_w), Style::new().fg(Color::BrightBlack).dim());
        println!("  {sep_line}");

        let kernels_dir = out_root.as_ref().map(|r| r.join("Resources").join("kernels"));
        if let Some(dir) = &kernels_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!(
                    "  {} create {}: {}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                    dir.display(),
                    e
                );
                return Err(CliError::Io(e));
            }
        }
        let mut emitted_kernels: Vec<Kernel> = Vec::new();
        let mut emitted_paths: Vec<PathBuf> = Vec::new();

        let mut ok = 0u32;
        let mut errors = 0u32;

        for (name, (spec, dtypes)) in &sorted {
            if !filters.matches_kernel(spec.kernel_name, spec.op) {
                continue;
            }
            let _kspan = tracing::debug_span!("kernel", name).entered();
            tracing::debug!(kernel = name, "building kernel");

            let dtypes_to_check: Vec<DType> = match &dtypes_filter {
                Some(df) => dtypes.iter().filter(|dt| df.contains(dt)).copied().collect(),
                None => dtypes.clone(),
            };

            let mode = effective_mode(spec);

            let mut dtypes_ok = Vec::new();
            let mut dtypes_err = Vec::new();
            for &dt in &dtypes_to_check {
                let mut k = (spec.kernel_ir)(dt);
                k.mode = mode;
                k.name = monomorphized_name(spec.kernel_name, dt, dtypes.len());

                let expected_tpg =
                    spec.shapes.first().map(|s| s.tpg as u32).or_else(|| spec.dispatch.tpg_hint());
                let generator = generator_for_mode(mode, expected_tpg);

                let msl = match generator.generate(&k) {
                    Ok(msl) => msl,
                    Err(e) => {
                        tracing::warn!(kernel = %k.name, dtype = %dt, error = %e, "codegen failed");
                        dtypes_err.push((dt, format!("{e:?}")));
                        errors += 1;
                        continue;
                    },
                };

                // Metal compile-check on macOS (catches invalid simdgroup signatures, etc.)
                if cfg!(target_os = "macos") {
                    let air_check = check_metal_compile(&msl, &k.name);
                    if let Err(e) = air_check {
                        dtypes_err.push((dt, e));
                        errors += 1;
                        continue;
                    }
                }

                dtypes_ok.push(dt);

                // Emit on success.
                if let Some(dir) = &kernels_dir {
                    if emit_kinds.contains(&EmitKind::Msl) {
                        match write_msl(&k, dir, &generator) {
                            Ok(path) => emitted_paths.push(path),
                            Err(e) => {
                                eprintln!(
                                    "  {} emit msl for {}: {}",
                                    paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                                    k.name,
                                    e
                                );
                                return Err(CliError::Config(e.to_string()));
                            },
                        }
                    }
                }
                if !emit_kinds.is_empty() {
                    if metaltile_std::ffai::dequant_gemv::dequant_gemv_wants_indirect(&k.name) {
                        k.wants_indirect_variant = true;
                    }
                    emitted_kernels.push(k.clone());
                }

                if args.verbose > 0 {
                    if let Ok(msl) = generator.generate(&k) {
                        println!("// ══ {} {} ══\n{}", k.name, dt.label(), msl);
                    }
                }
            }

            if !dtypes_err.is_empty() {
                let kernel_cell =
                    paint_stdout(pad_left(name, name_w), Style::new().fg(Color::Cyan).bold());
                let dt_str: String =
                    dtypes_err.iter().map(|(dt, _)| dt.label()).collect::<Vec<_>>().join("/");
                let dt_cell =
                    paint_stdout(pad_left(&dt_str, dt_w), Style::new().fg(Color::Blue).bold());
                let ck_cell = paint_stderr("✗", Style::new().fg(Color::Red).bold());
                println!("  {kernel_cell} {sep} {dt_cell} {sep}  {ck_cell}");
                for (dt, err_msg) in &dtypes_err {
                    let label = format!("{}:", dt.label());
                    eprintln!(
                        "    {} {}",
                        paint_stdout(
                            pad_right(&label, dt_w + 2),
                            Style::new().fg(Color::BrightBlack)
                        ),
                        paint_stderr(
                            err_msg.lines().next().unwrap_or(err_msg),
                            Style::new().fg(Color::BrightWhite)
                        ),
                    );
                }
            } else if !dtypes_ok.is_empty() {
                ok += 1;
                let kernel_cell =
                    paint_stdout(pad_left(name, name_w), Style::new().fg(Color::Cyan).bold());
                let dtype_str = dtypes_ok.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/");
                let dt_cell =
                    paint_stdout(pad_left(&dtype_str, dt_w), Style::new().fg(Color::Blue).bold());
                let ck_cell = paint_stdout("✓", Style::new().fg(Color::Green).bold());
                println!("  {kernel_cell} {sep} {dt_cell} {sep}  {ck_cell}");
            }
        }

        // Emit pass (manifest, Swift wrappers, metallib)
        if let Some(out) = &out_root {
            emit_artifacts(out, &emit_kinds, &emitted_kernels, &emitted_paths, &args.sdk)?;
        }

        // Summary
        println!();
        if errors > 0 {
            println!(
                "  {}  {}",
                paint_stdout(format!("{ok} ok"), Style::new().fg(Color::Green).bold()),
                paint_stderr(
                    format!("{errors} error{}", if errors == 1 { "" } else { "s" }),
                    Style::new().fg(Color::Red).bold()
                ),
            );
            Err(CliError::BuildFailed(errors))
        } else {
            println!(
                "  {}",
                paint_stdout(format!("{ok} ok"), Style::new().fg(Color::Green).bold())
            );
            Ok(())
        }
    }
}

// ── EmitKind ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EmitKind {
    Msl,
    Metallib,
    Swift,
    Ir,
}

fn parse_emit_list(raw: &str) -> Result<BTreeSet<EmitKind>, CliError> {
    let mut kinds = BTreeSet::new();
    for tok in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match tok {
            "msl" => {
                kinds.insert(EmitKind::Msl);
            },
            "metallib" => {
                kinds.insert(EmitKind::Metallib);
                kinds.insert(EmitKind::Msl);
            },
            "swift" => {
                kinds.insert(EmitKind::Swift);
            },
            "ir" => {
                kinds.insert(EmitKind::Ir);
            },
            "all" => {
                kinds.insert(EmitKind::Msl);
                kinds.insert(EmitKind::Metallib);
                kinds.insert(EmitKind::Swift);
                kinds.insert(EmitKind::Ir);
            },
            other => return Err(CliError::Config(format!("unknown --emit kind '{other}'"))),
        }
    }
    Ok(kinds)
}

fn monomorphized_name(base: &str, dt: DType, n_dtypes: usize) -> String {
    let suffix = emit::dtype_suffix(dt);
    if n_dtypes == 1 && base.ends_with(&format!("_{suffix}")) {
        base.to_string()
    } else {
        format!("{base}_{suffix}")
    }
}

fn emit_artifacts(
    out_root: &Path,
    kinds: &BTreeSet<EmitKind>,
    kernels: &[Kernel],
    metal_files: &[PathBuf],
    sdk: &str,
) -> Result<(), CliError> {
    let resources_dir = out_root.join("Resources");
    let generated_dir = out_root.join("Generated");

    if kinds.contains(&EmitKind::Ir) {
        if let Err(e) = std::fs::create_dir_all(&resources_dir).and_then(|_| {
            write_manifest(kernels, &resources_dir.join("manifest.json"))
                .map_err(|e| std::io::Error::other(e.to_string()))
        }) {
            return Err(CliError::Io(e));
        }
    }

    if kinds.contains(&EmitKind::Swift) {
        if let Err(e) = std::fs::create_dir_all(&generated_dir) {
            return Err(CliError::Io(e));
        }
        let path = generated_dir.join("MetalTileKernels.swift");
        if let Err(e) = write_swift_wrappers(kernels, &path) {
            return Err(CliError::Config(e.to_string()));
        }
    }

    if kinds.contains(&EmitKind::Metallib) {
        let metallib_path = resources_dir.join("kernels.metallib");
        let air_dir = project::air_cache_dir();
        if let Err(e) = compile_metallib(metal_files, &metallib_path, sdk, &air_dir) {
            return Err(CliError::MetalCompile(e.to_string()));
        }
    }
    Ok(())
}

// ── Quick Metal compile check ─────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn check_metal_compile(msl: &str, kernel_name: &str) -> Result<(), String> {
    use std::process::Command;

    let dir = std::env::temp_dir().join("tile-build-check");
    let _ = std::fs::create_dir_all(&dir);
    let metal_path = dir.join(format!("{kernel_name}.metal"));
    let air_path = dir.join(format!("{kernel_name}.air"));

    std::fs::write(&metal_path, msl).map_err(|e| format!("write temp .metal: {e}"))?;

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(&air_path)
        .output()
        .map_err(|e| format!("invoke xcrun metal: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let short =
            stderr.lines().filter(|l| l.contains("error:")).take(3).collect::<Vec<_>>().join("\n");
        return if short.is_empty() { Err(stderr.into_owned()) } else { Err(short) };
    }

    let _ = std::fs::remove_file(&metal_path);
    let _ = std::fs::remove_file(&air_path);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn check_metal_compile(_msl: &str, _kernel_name: &str) -> Result<(), String> { Ok(()) }

// ── External-project build ───────────────────────────────────────────────

fn run_build_external(args: &BuildArgs) -> Result<(), CliError> {
    let filters = args.filters.to_filters()?;
    let compile = RealCompileService;

    let project_dir = std::path::Path::new(".");
    let binary = compile
        .compile_harness(project_dir, false)
        .map_err(|e| CliError::Config(format!("harness build: {e}")))?;

    let mut cmd = std::process::Command::new(&binary);
    cmd.arg("--tile-protocol=jsonl").arg("--action=build");
    filters.forward_to_cmd(&mut cmd);
    if let Some(dtypes) = &args.dtypes {
        cmd.arg("--dtypes").arg(dtypes);
    }

    let profile = project::active_profile();
    println!(
        "{} {}  profile={}",
        paint_stdout("tile build", Style::new().fg(Color::Cyan).bold()),
        paint_stdout("· Tile.toml", Style::new().fg(Color::BrightBlack)),
        profile,
    );

    let messages = crate::project::harness::run_harness(&mut cmd)?;

    let mut rows: Vec<BuildRow> = Vec::new();
    for msg in messages {
        if let HarnessMessage::BuildResult(val) = msg {
            let kernel_name = val
                .get("kernel_name")
                .or_else(|| val.get("op"))
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            let dtype = val.get("dtype").and_then(|v| v.as_str()).unwrap_or("?").to_string();
            let ok = val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
            let error = val.get("error").and_then(|v| v.as_str()).map(String::from);
            rows.push(BuildRow { kernel_name, dtype, ok, error });
        }
    }

    render_build_rows(&rows);
    finalize_build(&rows)
}

fn render_build_rows(rows: &[BuildRow]) {
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<&str, (Vec<&str>, Vec<(&str, Option<&str>)>)> = BTreeMap::new();
    for row in rows {
        let entry = groups.entry(row.kernel_name.as_str()).or_default();
        if row.ok {
            entry.0.push(row.dtype.as_str());
        } else {
            entry.1.push((row.dtype.as_str(), row.error.as_deref()));
        }
    }

    if groups.is_empty() {
        return;
    }

    let name_w = groups.keys().map(|n| n.len()).max().unwrap_or(20).clamp(8, 48);
    let dt_w = rows.iter().map(|r| r.dtype.len()).max().unwrap_or(3).clamp(3, 24) * 3;
    let dt_w = dt_w.clamp(8, 24);

    let sep = col_sep();
    let bold = Style::new().fg(Color::BrightWhite).bold();
    let hdr = format!(
        "  {} {} {} {} {}",
        paint_stdout(pad_left("Kernel", name_w), bold),
        sep,
        paint_stdout(pad_left("Dtypes", dt_w), bold),
        sep,
        paint_stdout(pad_right("ok", 2), bold),
    );
    println!("{hdr}");
    let total_w = 4 + name_w + 3 + dt_w + 3 + 2;
    let sep_line = paint_stdout("─".repeat(total_w), Style::new().fg(Color::BrightBlack).dim());
    println!("  {sep_line}");

    for (name, (ok_dtypes, err_dtypes)) in &groups {
        if !err_dtypes.is_empty() {
            let kernel_cell =
                paint_stdout(pad_left(name, name_w), Style::new().fg(Color::Cyan).bold());
            let dt_str = err_dtypes.iter().map(|(dt, _)| *dt).collect::<Vec<_>>().join("/");
            let dt_cell =
                paint_stdout(pad_left(&dt_str, dt_w), Style::new().fg(Color::Blue).bold());
            let ck_cell = paint_stderr("✗", Style::new().fg(Color::Red).bold());
            println!("  {kernel_cell} {sep} {dt_cell} {sep}  {ck_cell}");
            for (dt, err_msg) in err_dtypes {
                if let Some(msg) = err_msg {
                    let label = format!("{}:", dt);
                    eprintln!(
                        "    {} {}",
                        paint_stdout(
                            pad_right(&label, dt_w + 2),
                            Style::new().fg(Color::BrightBlack)
                        ),
                        paint_stderr(
                            msg.lines().next().unwrap_or(msg),
                            Style::new().fg(Color::BrightWhite)
                        ),
                    );
                }
            }
        }
        if !ok_dtypes.is_empty() {
            let kernel_cell =
                paint_stdout(pad_left(name, name_w), Style::new().fg(Color::Cyan).bold());
            let dt_str = ok_dtypes.join("/");
            let dt_cell =
                paint_stdout(pad_left(&dt_str, dt_w), Style::new().fg(Color::Blue).bold());
            let ck_cell = paint_stdout("✓", Style::new().fg(Color::Green).bold());
            println!("  {kernel_cell} {sep} {dt_cell} {sep}  {ck_cell}");
        }
    }
}

fn finalize_build(rows: &[BuildRow]) -> Result<(), CliError> {
    let ok_count = rows.iter().filter(|r| r.ok).count() as u32;
    let error_count = rows.iter().filter(|r| !r.ok).count() as u32;
    println!();
    if error_count > 0 {
        println!(
            "  {}  {}",
            paint_stdout(format!("{ok_count} ok"), Style::new().fg(Color::Green).bold()),
            paint_stderr(
                format!("{error_count} error{}", if error_count == 1 { "" } else { "s" }),
                Style::new().fg(Color::Red).bold()
            ),
        );
        Err(CliError::BuildFailed(error_count))
    } else {
        println!(
            "  {}",
            paint_stdout(format!("{ok_count} ok"), Style::new().fg(Color::Green).bold())
        );
        Ok(())
    }
}

// ── --time-passes ────────────────────────────────────────────────────────

const TIME_PASSES_WARMUP: usize = 5;
const TIME_PASSES_ITERS: usize = 25;

fn run_time_passes(
    filter_flags: &crate::FilterFlags,
    dtypes_arg: Option<&str>,
) -> Result<(), CliError> {
    let dtypes_filter: Option<Vec<DType>> =
        dtypes_arg.map(|s| s.split(',').filter_map(|t| t.trim().parse::<DType>().ok()).collect());

    let kernels: Vec<_> = metaltile_core::bench::spec::bench_specs()
        .iter()
        .copied()
        .filter(|s| filter_flags.matches_filter(s.kernel_name))
        .flat_map(|s| {
            s.dtypes
                .iter()
                .filter(|dt| dtypes_filter.as_ref().is_none_or(|df| df.contains(dt)))
                .map(|&dt| (s.kernel_ir)(dt))
        })
        .collect();

    if kernels.is_empty() {
        return Err(CliError::Config("no kernels matched filter".into()));
    }

    let pipeline = PipelineBuilder::standard().build();
    let total_iters = TIME_PASSES_WARMUP + TIME_PASSES_ITERS;
    let mut pass_names: Vec<String> = Vec::new();
    let mut samples: Vec<Vec<u64>> = Vec::new();

    for iter in 0..total_iters {
        let mut pass_totals: Vec<u64> = Vec::new();
        for k in &kernels {
            let mut kc = k.clone();
            let stats: Vec<PassStats> = match run_passes_with_stats(&mut kc, &pipeline) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if pass_totals.is_empty() {
                pass_totals = vec![0u64; stats.len()];
                if pass_names.is_empty() {
                    pass_names = stats.iter().map(|s| s.name.clone()).collect();
                    samples = vec![Vec::with_capacity(TIME_PASSES_ITERS); pass_names.len()];
                }
            }
            for (i, s) in stats.iter().enumerate() {
                pass_totals[i] += s.wall_us;
            }
        }
        if iter >= TIME_PASSES_WARMUP {
            for (i, t) in pass_totals.iter().enumerate() {
                samples[i].push(*t);
            }
        }
    }

    let n_kernels = kernels.len() as f64;
    println!(
        "{} {}",
        paint_stdout("tile build --time-passes", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            format!(
                "· {} kernels × {} iters ({} warmup)",
                kernels.len(),
                TIME_PASSES_ITERS,
                TIME_PASSES_WARMUP
            ),
            Style::new().fg(Color::BrightBlack),
        ),
    );
    println!("  {:<24}  {:>14}  {:>18}", "pass", "median_us", "median_us/kernel");
    for (i, name) in pass_names.iter().enumerate() {
        samples[i].sort_unstable();
        let median = samples[i][samples[i].len() / 2];
        let per_kernel = median as f64 / n_kernels;
        println!("  {name:<24}  {median:>14}  {per_kernel:>18.1}");
    }
    Ok(())
}

// ── Table styling helpers ────────────────────────────────────────────────

fn col_sep() -> String { paint_stdout("│", Style::new().fg(Color::BrightBlack).dim()) }
fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }
fn pad_right(text: &str, width: usize) -> String { format!("{text:>width$}") }

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use metaltile_core::bench::spec::{BenchSpec, effective_mode};

    use super::*;

    fn collect_all_kernels() -> Vec<Kernel> {
        let mut by_name: BTreeMap<&str, (&'static BenchSpec, Vec<DType>)> = BTreeMap::new();
        for spec in metaltile_core::bench::spec::bench_specs().iter().copied() {
            let entry = by_name.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
            for &dt in spec.dtypes {
                if !entry.1.contains(&dt) {
                    entry.1.push(dt);
                }
            }
        }
        let total: usize = by_name.values().map(|(_, dtypes)| dtypes.len()).sum();
        let mut kernels = Vec::with_capacity(total);
        for (spec, dtypes) in by_name.values() {
            let mode = effective_mode(spec);
            for &dt in dtypes {
                let mut k = (spec.kernel_ir)(dt);
                k.name = monomorphized_name(spec.kernel_name, dt, dtypes.len());
                k.mode = mode;
                if metaltile_std::ffai::dequant_gemv::dequant_gemv_wants_indirect(&k.name) {
                    k.wants_indirect_variant = true;
                }
                kernels.push(k);
            }
        }
        kernels
    }

    #[test]
    fn build_keeps_indirect_wrappers_for_dequant_gemv_int4() {
        let kernels = collect_all_kernels();
        assert!(
            kernels.iter().any(|k| k.name == "dequant_gemv_int4_f16"),
            "dequant_gemv_int4_f16 missing from kernel set"
        );
        assert!(
            kernels.iter().any(|k| k.name == "dequant_gemv_int4_bf16"),
            "dequant_gemv_int4_bf16 missing from kernel set"
        );

        let swift = metaltile_codegen::emit::render_swift_wrappers(&kernels);
        assert!(
            swift.contains("func dequant_gemv_int4_f16_indirect("),
            "indirect Swift wrapper for dequant_gemv_int4_f16 dropped"
        );
        assert!(
            swift.contains("func dequant_gemv_int4_bf16_indirect("),
            "indirect Swift wrapper for dequant_gemv_int4_bf16 dropped"
        );
        assert!(
            swift.contains("dispatchThreadgroups(indirectBuffer:"),
            "indirect wrappers must dispatch from an indirect buffer"
        );
    }
}

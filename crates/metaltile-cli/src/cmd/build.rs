//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile build` — Compile all registered kernels.
//!
//! Default behavior is a compile-check (codegen MSL, report errors,
//! no I/O). With `--emit <list> --out <dir>` it also writes artifacts:
//!
//!   --emit msl       Write per-kernel `<dir>/Resources/kernels/<name>.metal`
//!   --emit metallib  Compile + write `<dir>/Resources/kernels.metallib`
//!                    (implies msl)
//!   --emit swift     Write `<dir>/Generated/MetalTileKernels.swift`
//!                    dispatch wrappers
//!   --emit ir        Write `<dir>/Resources/manifest.json` IR descriptor
//!   --emit all       Shorthand for msl,metallib,swift,ir
//!
//! Multiple kinds may be combined via comma: `--emit msl,swift,ir`.
//!
//! The output layout intentionally matches a Swift Package's
//! `Sources/<Target>/{Resources,Generated}/` convention so a consumer
//! can point `--out` at their target directory and have the artifacts
//! land in the right place for SwiftPM resource bundling.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use metaltile::harness::{bench::KernelBench, registry::all_benches};
use metaltile_codegen::{
    emit::{
        self,
        compile_metal_to_air,
        dtype_suffix,
        link_air_to_metallib,
        write_manifest,
        write_msl,
        write_swift_wrappers,
    },
    generator_for_mode,
    passes::{PassStats, PipelineBuilder, run_passes_with_stats},
};
use metaltile_core::ir::Kernel;
use metaltile_std::bench_types::DType;
use rayon::prelude::*;

use crate::{
    BuildArgs,
    CliError,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

// ── Table helpers ────────────────────────────────────────────────────

fn col_sep() -> String { paint_stdout("│", Style::new().fg(Color::BrightBlack).dim()) }

fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }

fn pad_right(text: &str, width: usize) -> String { format!("{text:>width$}") }

/// A kernel to emit: its bench (carrying IR + mode + grid) and the union of
/// dtypes it should be monomorphized over.
type EmitKernel = (&'static dyn KernelBench, Vec<DType>);

/// `TileCommand` wrapper for `tile build`.
pub struct BuildCommand<'a>(pub &'a BuildArgs);

impl<'a> super::TileCommand for BuildCommand<'a> {
    fn run(&self, _harness: &crate::harness::Harness) -> Result<(), crate::CliError> { run(self.0) }
}

pub fn run(args: &BuildArgs) -> Result<(), CliError> {
    let _span = tracing::info_span!("build", filter = ?args.filter, emit = ?args.emit).entered();
    let filter = &args.filter;
    let dtypes_arg = &args.dtypes;
    let verbose = args.verbose > 0;
    let emit_arg = &args.emit;
    let out_arg = &args.out;
    let sdk = &args.sdk;

    if args.time_passes {
        run_time_passes(filter.as_deref(), dtypes_arg.as_deref())?;
        return Ok(());
    }

    let emit_kinds: BTreeSet<EmitKind> = match emit_arg.as_deref() {
        None => BTreeSet::new(),
        Some(raw) => match parse_emit_list(raw) {
            Ok(k) => k,
            Err(e) => {
                eprintln!(
                    "  {} {}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                    paint_stderr(e.to_string(), Style::new().fg(Color::BrightWhite)),
                );
                eprintln!("  valid kinds: msl, metallib, swift, ir, all");
                return Err(e);
            },
        },
    };

    let out_root: Option<PathBuf> = match (&emit_kinds.is_empty(), &out_arg) {
        (true, _) => None,
        (false, Some(p)) => Some(PathBuf::from(p)),
        (false, None) => {
            eprintln!(
                "  {} {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                paint_stderr("--emit requires --out <dir>", Style::new().fg(Color::BrightWhite)),
            );
            return Err(CliError::Other("--emit requires --out <dir>".into()));
        },
    };

    // Parse --dtypes list
    let dtypes_filter: Option<Vec<DType>> = dtypes_arg
        .as_ref()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse::<DType>().ok()).collect());

    // Collect unique kernels from the new `#[bench]` registry. Each bench's
    // setup carries the kernel IR (with mode applied via `.mode()`), the dtype
    // set, and the threadgroup geometry — everything emission needs — so the
    // emitted kernel set no longer depends on the legacy `BenchSpec` inventory.
    // (`BenchBuffer`s are lazy metadata, so building a setup just to read its
    // kernel/grid is cheap.) Keyed by the kernel's generic name; a kernel with
    // multiple benches unions their dtype sets.
    let mut kernels: BTreeMap<String, EmitKernel> = BTreeMap::new();
    for entry in all_benches() {
        let bench = entry.bench();
        let Some(&first_dt) = bench.dtypes().first() else { continue };
        let base_name = bench.setup(first_dt).kernel().name.to_string();
        let e = kernels.entry(base_name).or_insert((bench, Vec::new()));
        for &dt in bench.dtypes() {
            if !e.1.contains(&dt) {
                e.1.push(dt);
            }
        }
    }

    let mut sorted: Vec<(String, EmitKernel)> = kernels.into_iter().collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    // Header.
    println!(
        "{} {}",
        paint_stdout("tile build", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(format!("· {} kernels", sorted.len()), Style::new().fg(Color::BrightBlack)),
    );

    // Compute column widths.
    let name_w = sorted.iter().map(|(n, _)| n.len()).max().unwrap_or(20).clamp(8, 48);
    let dt_w = sorted
        .iter()
        .map(|(_, (_, dtypes))| {
            dtypes.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/").len()
        })
        .max()
        .unwrap_or(12)
        .clamp(8, 24);
    let ck_w = 2usize;

    let sep = col_sep();
    let bold = Style::new().fg(Color::BrightWhite).bold();
    let hdr = format!(
        "  {} {} {} {} {}",
        paint_stdout(pad_left("Kernel", name_w), bold),
        sep,
        paint_stdout(pad_left("Dtypes", dt_w), bold),
        sep,
        paint_stdout(pad_right("ok", ck_w), bold),
    );
    println!("{hdr}");

    let total_w = 4 + name_w + 3 + dt_w + 3 + ck_w;
    let sep_line = paint_stdout("─".repeat(total_w), Style::new().fg(Color::BrightBlack).dim());
    println!("  {sep_line}");

    // Per-output collectors for the emit step.
    let kernels_dir = out_root.as_ref().map(|r| r.join("Resources").join("kernels"));
    if let Some(dir) = &kernels_dir
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        eprintln!(
            "  {} create {}: {}",
            paint_stderr("error:", Style::new().fg(Color::Red).bold()),
            dir.display(),
            e
        );
        return Err(CliError::Io(e));
    }

    // Collect work items for kernels that pass the name filter.
    struct WorkItem<'a> {
        name: &'a str,
        bench: &'static dyn metaltile::harness::bench::KernelBench,
        dtypes_to_check: Vec<DType>,
        n_dtypes: usize,
    }
    let work_items: Vec<WorkItem<'_>> = sorted
        .iter()
        .filter(|(name, _)| matches_filter(filter.as_deref(), name.as_str()))
        .map(|(name, (bench, dtypes))| {
            let dtypes_to_check: Vec<DType> = match &dtypes_filter {
                Some(df) => dtypes.iter().filter(|dt| df.contains(dt)).copied().collect(),
                None => dtypes.clone(),
            };
            WorkItem { name: name.as_str(), bench: *bench, dtypes_to_check, n_dtypes: dtypes.len() }
        })
        .collect();

    // Compile all kernels in parallel. Each xcrun metal -c call is ~50-200 ms
    // and fully independent, so parallelism across N CPU cores gives a near-N×
    // speedup on the typical 200+ kernel corpus.
    struct KernelResult {
        name: String,
        dtypes_ok: Vec<DType>,
        dtypes_err: Vec<(DType, String)>,
        emitted_kernels: Vec<Kernel>,
        emitted_paths: Vec<PathBuf>,
        verbose_lines: Vec<String>,
    }

    let par_results: Vec<KernelResult> = work_items
        .par_iter()
        .map(|item| {
            let _kspan = tracing::debug_span!("kernel", name = item.name).entered();
            tracing::debug!(kernel = item.name, "building kernel");

            let mut dtypes_ok = Vec::new();
            let mut dtypes_err = Vec::new();
            let mut item_kernels = Vec::new();
            let mut item_paths = Vec::new();
            let mut verbose_lines = Vec::new();

            for &dt in &item.dtypes_to_check {
                let setup = item.bench.setup(dt);
                let mut k = setup.kernel().clone();
                let mode = k.mode;
                k.name = monomorphized_name(item.name, dt, item.n_dtypes);

                // Codegen hint so the emitted MSL matches exactly what `tile
                // bench` measures (and what production callers will dispatch
                // at, per the kernel's DISPATCH INVARIANTS):
                //   1. `Generic` dispatch carries TPG on `ShapeSpec`; prefer that.
                //   2. Other variants (`Sort`, `SdpaVector`, `Attention`, …) carry
                //      TPG on the `BenchDispatch` variant itself.
                //   3. `None` (a few Grid3D/Elementwise variants with no fixed
                //      TPG) → safe slow path. Reduction-mode kernels with no TPG
                //      signal anywhere fall through to the conservative emit.
                let expected_tpg = Some(setup.grid().tpg[0]);
                let generator = generator_for_mode(mode, expected_tpg);

                let msl = match generator.generate(&k) {
                    Ok(msl) => {
                        tracing::debug!(kernel = %k.name, dtype = %dt, bytes = msl.len(), "codegen ok");
                        msl
                    },
                    Err(e) => {
                        tracing::warn!(kernel = %k.name, dtype = %dt, error = %e, "codegen failed");
                        dtypes_err.push((dt, format!("{e:?}")));
                        continue;
                    },
                };

                // Metal compile-check on macOS (catches invalid simdgroup signatures, etc.)
                if cfg!(target_os = "macos")
                    && let Err(e) = check_metal_compile(&msl, &k.name)
                {
                    dtypes_err.push((dt, e));
                    continue;
                }

                if verbose {
                    verbose_lines.push(format!("// ══ {} {} ══\n{}", k.name, dt.label(), msl));
                }

                dtypes_ok.push(dt);

                // Emit on success.
                if let Some(dir) = &kernels_dir
                    && emit_kinds.contains(&EmitKind::Msl)
                {
                    match write_msl(&k, dir, &generator) {
                        Ok(path) => item_paths.push(path),
                        Err(e) => {
                            dtypes_err.push((dt, e.to_string()));
                            dtypes_ok.pop();
                            continue;
                        },
                    }
                }
                if !emit_kinds.is_empty() {
                    // Per-kernel opt-in for the `_indirect` Swift wrapper.
                    if metaltile_std::ffai::dequant_gemv::dequant_gemv_wants_indirect(&k.name) {
                        k.wants_indirect_variant = true;
                    }
                    item_kernels.push(k);
                }
            }

            KernelResult {
                name: item.name.to_string(),
                dtypes_ok,
                dtypes_err,
                emitted_kernels: item_kernels,
                emitted_paths: item_paths,
                verbose_lines,
            }
        })
        .collect();

    // Sequential pass: print table rows and accumulate emit artifacts.
    let mut ok = 0u32;
    let mut errors = 0u32;
    let mut emitted_kernels: Vec<Kernel> = Vec::new();
    let mut emitted_paths: Vec<PathBuf> = Vec::new();

    for result in par_results {
        for line in &result.verbose_lines {
            println!("{line}");
        }
        if !result.dtypes_err.is_empty() {
            let kernel_cell =
                paint_stdout(pad_left(&result.name, name_w), Style::new().fg(Color::Cyan).bold());
            let dt_str: String =
                result.dtypes_err.iter().map(|(dt, _)| dt.label()).collect::<Vec<_>>().join("/");
            let dt_cell =
                paint_stdout(pad_left(&dt_str, dt_w), Style::new().fg(Color::Blue).bold());
            let ck_cell = paint_stderr("✗", Style::new().fg(Color::Red).bold());
            println!("  {kernel_cell} {sep} {dt_cell} {sep}  {ck_cell}");
            for (dt, err_msg) in &result.dtypes_err {
                let label = format!("{}:", dt.label());
                eprintln!(
                    "    {} {}",
                    paint_stdout(pad_right(&label, dt_w + 2), Style::new().fg(Color::BrightBlack)),
                    paint_stderr(
                        err_msg.lines().next().unwrap_or(err_msg),
                        Style::new().fg(Color::BrightWhite)
                    ),
                );
            }
            errors += result.dtypes_err.len() as u32;
        } else if !result.dtypes_ok.is_empty() {
            ok += 1;
            let kernel_cell =
                paint_stdout(pad_left(&result.name, name_w), Style::new().fg(Color::Cyan).bold());
            let dtype_str =
                result.dtypes_ok.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/");
            let dt_cell =
                paint_stdout(pad_left(&dtype_str, dt_w), Style::new().fg(Color::Blue).bold());
            let ck_cell = paint_stdout("✓", Style::new().fg(Color::Green).bold());
            println!("  {kernel_cell} {sep} {dt_cell} {sep}  {ck_cell}");
        }
        emitted_kernels.extend(result.emitted_kernels);
        emitted_paths.extend(result.emitted_paths);
    }

    // ─── Emit pass (manifest, Swift wrappers, metallib) ─────────────────
    if let Some(out) = &out_root {
        emit_artifacts(out, &emit_kinds, &emitted_kernels, &emitted_paths, sdk)?;
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
        Err(CliError::Other(format!("{errors} kernel(s) failed to compile")))
    } else {
        println!("  {}", paint_stdout(format!("{ok} ok"), Style::new().fg(Color::Green).bold()));
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

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
                kinds.insert(EmitKind::Msl); // metallib needs the .metal source files on disk
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
            other => return Err(CliError::Other(format!("unknown --emit kind '{other}'"))),
        }
    }
    Ok(kinds)
}

/// Build the per-dtype monomorphized kernel symbol name.
///
/// `mt_add` (2+ dtypes) → `mt_add_f32` / `mt_add_f16` / `mt_add_bf16`
/// `mt_argmax_f32` (1 dtype, name already has suffix) → `mt_argmax_f32`
fn monomorphized_name(base: &str, dt: DType, n_dtypes: usize) -> String {
    let suffix = dtype_suffix(dt);
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

    if kinds.contains(&EmitKind::Ir)
        && let Err(e) = std::fs::create_dir_all(&resources_dir).and_then(|_| {
            write_manifest(kernels, &resources_dir.join("manifest.json"))
                .map_err(|e| std::io::Error::other(e.to_string()))
        })
    {
        eprintln!(
            "  {} write manifest.json: {}",
            paint_stderr("error:", Style::new().fg(Color::Red).bold()),
            e
        );
        return Err(CliError::Io(e));
    }

    if kinds.contains(&EmitKind::Swift) {
        if let Err(e) = std::fs::create_dir_all(&generated_dir) {
            eprintln!(
                "  {} create {}: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                generated_dir.display(),
                e
            );
            return Err(CliError::Io(e));
        }
        let path = generated_dir.join("MetalTileKernels.swift");
        if let Err(e) = write_swift_wrappers(kernels, &path) {
            eprintln!(
                "  {} write {}: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                path.display(),
                e
            );
            return Err(CliError::Other(e.to_string()));
        }
    }

    if kinds.contains(&EmitKind::Metallib) {
        let metallib_path = resources_dir.join("kernels.metallib");
        let air_dir = std::env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target"))
            .join("tile-build-air");
        if let Err(e) = std::fs::create_dir_all(&air_dir) {
            eprintln!(
                "  {} create air dir: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                e
            );
            return Err(CliError::Io(e));
        }
        // Compile each .metal → .air in parallel; each xcrun call is ~50-200 ms
        // and fully independent. The metallib link remains a single serial step.
        let air_files: Vec<PathBuf> = match metal_files
            .par_iter()
            .map(|m| compile_metal_to_air(m, sdk, &air_dir))
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "  {} compile .metal: {}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                    e
                );
                return Err(CliError::MetalCompile(e.to_string()));
            },
        };
        if let Err(e) = link_air_to_metallib(&air_files, &metallib_path, sdk) {
            eprintln!(
                "  {} link metallib: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                e
            );
            return Err(CliError::MetalCompile(e.to_string()));
        }
    }
    let _ = emit::dtype_suffix; // anchor public API surface for re-export checks
    Ok(())
}

/// Quickly compile a Metal shader with xcrun to catch type errors.
/// Returns Ok(()) if compilation succeeds, Err(msg) if it fails.
#[cfg(target_os = "macos")]
fn check_metal_compile(msl: &str, kernel_name: &str) -> Result<(), String> {
    use std::process::Command;

    let dir = std::env::temp_dir().join("tile-build-check");
    let _ = std::fs::create_dir_all(&dir);
    let metal_path = dir.join(format!("{kernel_name}.metal"));
    let air_path = dir.join(format!("{kernel_name}.air"));

    if let Err(e) = std::fs::write(&metal_path, msl) {
        return Err(format!("write temp .metal: {e}"));
    }

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(&air_path)
        .output()
        .map_err(|e| format!("invoke xcrun metal: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Keep just the first meaningful error line.
        let short =
            stderr.lines().filter(|l| l.contains("error:")).take(3).collect::<Vec<_>>().join("\n");
        let msg = if short.is_empty() { stderr.into_owned() } else { short };
        return Err(msg);
    }

    // Clean up temp files on success.
    let _ = std::fs::remove_file(&metal_path);
    let _ = std::fs::remove_file(&air_path);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn check_metal_compile(_msl: &str, _kernel_name: &str) -> Result<(), String> {
    Ok(()) // Skip Metal compile-check on non-macOS.
}

// ── --time-passes ────────────────────────────────────────────────────────

const TIME_PASSES_WARMUP: usize = 5;
const TIME_PASSES_ITERS: usize = 25;

/// Run the standard pass pipeline `TIME_PASSES_ITERS` times over the
/// filtered `BenchSpec × dtype` corpus and print per-pass median wall_us
/// (after `TIME_PASSES_WARMUP` discarded warmup iters).
///
/// Output schema matches `rustc -Z time-passes`-style tables:
/// `pass_name  median_total_us  median_per_kernel_us`.
fn run_time_passes(filter: Option<&str>, dtypes_arg: Option<&str>) -> Result<(), CliError> {
    let dtypes_filter: Option<Vec<DType>> =
        dtypes_arg.map(|s| s.split(',').filter_map(|t| t.trim().parse::<DType>().ok()).collect());

    let kernels: Vec<_> = all_benches()
        .map(|e| e.bench())
        .filter(|b| {
            let first = b.dtypes().first().copied().unwrap_or(DType::F32);
            matches_filter(filter, &b.setup(first).kernel().name)
        })
        .flat_map(|b| {
            b.dtypes()
                .iter()
                .filter(|dt| dtypes_filter.as_ref().is_none_or(|df| df.contains(dt)))
                .map(|&dt| b.setup(dt).kernel().clone())
                .collect::<Vec<_>>()
        })
        .collect();

    if kernels.is_empty() {
        eprintln!(
            "  {} no kernels matched filter",
            paint_stderr("error:", Style::new().fg(Color::Red).bold()),
        );
        return Err(CliError::Other("no kernels matched filter".into()));
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
                TIME_PASSES_WARMUP,
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// Collect all registered kernels (unfiltered) with indirect-dispatch
    /// flags applied. Used only by tests to verify the full kernel corpus.
    /// Mirrors the `--emit` path: kernels are discovered via the `#[bench]`
    /// registry, with the mode baked into each bench's setup IR.
    fn collect_all_kernels() -> Vec<Kernel> {
        let mut by_name: BTreeMap<String, EmitKernel> = BTreeMap::new();
        for entry in all_benches() {
            let b = entry.bench();
            let Some(&first_dt) = b.dtypes().first() else { continue };
            let name = b.setup(first_dt).kernel().name.to_string();
            let e = by_name.entry(name).or_insert((b, Vec::new()));
            for &dt in b.dtypes() {
                if !e.1.contains(&dt) {
                    e.1.push(dt);
                }
            }
        }
        let total: usize = by_name.values().map(|(_, dtypes)| dtypes.len()).sum();
        let mut kernels = Vec::with_capacity(total);
        for (name, (b, dtypes)) in &by_name {
            for &dt in dtypes {
                let mut k = b.setup(dt).kernel().clone();
                k.name = monomorphized_name(name, dt, dtypes.len());
                if metaltile_std::ffai::dequant_gemv::dequant_gemv_wants_indirect(&k.name) {
                    k.wants_indirect_variant = true;
                }
                kernels.push(k);
            }
        }
        kernels
    }

    /// Regression guard for the `_indirect` Swift wrappers.
    ///
    /// FFAI's GPU-router dispatches `dequant_gemv_int4` indirectly (grid
    /// shape from an `MTLBuffer` rather than a host `MTLSize`). If the
    /// indirect-wrapper flag is dropped or `dequant_gemv_int4` is
    /// unregistered/renamed, the FFAI build breaks.
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

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile inspect` — Print IR and/or MSL for kernels.

use std::{collections::BTreeMap, str::FromStr};

use metaltile_codegen::generator_for_mode;
use metaltile_core::bench::types::DType;
use metaltile_core::bench::spec::{BenchSpec, effective_mode};

use crate::{
    CliError,
    FilterFlags,
    term::{Color, Style, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct InspectArgs {
    /// Kernel name or substring to inspect (lists all if omitted).
    pub kernel: Option<String>,
    #[command(flatten)]
    pub filters: FilterFlags,
    /// Inspect all registered kernels.
    #[arg(long = "all")]
    pub all: bool,
    /// Print the raw IR before any compiler passes.
    #[arg(long = "ir")]
    pub ir: bool,
    /// Print a per-pass op-count reduction table.
    #[arg(long = "stats")]
    pub stats: bool,
    /// Print IR after a specific pass name, or "all" for every stage.
    #[arg(long = "pass")]
    pub pass: Option<String>,
    /// Dtype override for the kernel (f32, f16, bf16, i32, u32).
    #[arg(long = "dtype")]
    pub dtype: Option<String>,
    /// Write output files to <dir> instead of stdout.
    #[arg(long = "dir", short = 'o', value_hint = clap::ValueHint::DirPath)]
    pub dir: Option<std::path::PathBuf>,
}

impl InspectArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let filter_val = args.filters.filter.as_ref().or(args.kernel.as_ref());
        let _span = tracing::info_span!(
            "inspect",
            filter = ?filter_val,
            ir = args.ir,
            stats = args.stats,
        )
        .entered();

        // Collect all specs grouped by kernel_name.
        let mut kernels: BTreeMap<&str, (&'static BenchSpec, Vec<DType>)> = BTreeMap::new();
        for spec in metaltile_core::bench::spec::bench_specs().iter().copied() {
            let entry = kernels.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
            for &dt in spec.dtypes {
                if !entry.1.contains(&dt) {
                    entry.1.push(dt);
                }
            }
        }

        if kernels.is_empty() {
            eprintln!("No kernels registered.");
            return Ok(());
        }

        let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> = kernels.into_iter().collect();
        sorted.sort_unstable_by_key(|(name, _)| *name);

        if args.all {
            return dump_all(&sorted, args);
        }

        let Some(filter) = filter_val else {
            return list_kernels(&sorted);
        };

        let matched: Vec<_> =
            sorted.iter().filter(|(name, _)| args.filters.matches_filter(name)).collect();

        if matched.is_empty() {
            eprintln!(
                "{} {}",
                paint_stdout("error:", Style::new().fg(Color::Red).bold()),
                paint_stdout(
                    format!("no kernel matched '{filter}'"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            eprintln!(
                "\n{} {}",
                paint_stdout("Available:", Style::new().fg(Color::BrightBlack)),
                paint_stdout(
                    sorted.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", "),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            return Err(CliError::Config(format!("no kernel matched '{filter}'")));
        }

        let dtype_override: Option<DType> =
            args.dtype.as_deref().and_then(|s| DType::from_str(s).ok());

        for (name, (spec, dtypes)) in &matched {
            let dt =
                dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));

            if args.ir {
                let k = (spec.kernel_ir)(dt);
                if let Some(d) = &args.dir {
                    let path = d.join(format!("{}.ir", name));
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                    println!("wrote {}", path.display());
                } else {
                    println!("{k}");
                }
            } else if args.stats {
                let mut k = (spec.kernel_ir)(dt);
                k.mode = effective_mode(spec);
                let expected_tpg =
                    spec.shapes.first().map(|s| s.tpg as u32).or_else(|| spec.dispatch.tpg_hint());
                let generator = generator_for_mode(effective_mode(spec), expected_tpg);
                match generator.generate_with_stats(&k) {
                    Ok((_, stats)) => print_stats_table(&stats),
                    Err(e) => eprintln!("error: {e}"),
                }
            } else if let Some(pass) = &args.pass {
                let mut k = (spec.kernel_ir)(dt);
                k.mode = effective_mode(spec);

                match pass.as_str() {
                    "all" => {
                        println!("// ── BEFORE PASSES ───────────────────────────");
                        println!("{k}");
                        run_all_passes_and_print(&mut k);
                    },
                    pass_name => match metaltile_codegen::passes::PassRegistry::get(pass_name) {
                        Some(pass_obj) => {
                            if let Err(e) = pass_obj.run(&mut k) {
                                eprintln!("Pass {pass_name} failed: {e}");
                                return Ok(());
                            }
                            println!("// ── AFTER {pass_name} ────────────────────────");
                            println!("{k}");
                        },
                        None => {
                            let valid: Vec<_> = metaltile_codegen::passes::PassRegistry::names();
                            eprintln!("Unknown pass: {pass_name}. Valid: {} all", valid.join(", "));
                            return Ok(());
                        },
                    },
                }
            } else {
                let msl = generate_msl_dt(spec, dt);
                if let Some(d) = &args.dir {
                    let path = d.join(format!("{}.metal", name));
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, &msl).map_err(CliError::Io)?;
                    println!("wrote {}", path.display());
                } else {
                    let mode_str = effective_mode(spec).to_string();
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("// kernel: {}  mode: {}", name, mode_str);
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("{}", msl);
                }
            }
        }
        Ok(())
    }
}

// ── Listing & dumping ────────────────────────────────────────────────────

fn list_kernels(sorted: &[(&str, (&BenchSpec, Vec<DType>))]) -> Result<(), CliError> {
    eprintln!("{}", paint_stdout("tile inspect", Style::new().fg(Color::Cyan).bold()));
    eprintln!();
    for (name, (spec, dtypes)) in sorted {
        let dtype_str = dtypes.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/");
        let mode_str = effective_mode(spec).to_string();
        eprintln!(
            "  {}   {}   {dtype_str}",
            paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
            paint_stdout(mode_str, Style::new().fg(Color::BrightBlack)),
        );
    }
    let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());
    eprintln!();
    eprintln!(
        "  {} {sep} {}",
        paint_stdout(format!("{} kernels", sorted.len()), Style::new().fg(Color::BrightBlack)),
        paint_stdout("<kernel> for MSL", Style::new().fg(Color::BrightBlack)),
    );
    Ok(())
}

fn dump_all(sorted: &[(&str, (&BenchSpec, Vec<DType>))], args: &InspectArgs) -> Result<(), CliError> {
    for (name, (spec, dtypes)) in sorted {
        let dt = dtypes.first().copied().unwrap_or(DType::F32);
        if args.ir {
            let k = (spec.kernel_ir)(dt);
            if let Some(d) = &args.dir {
                let path = d.join(format!("{}.ir", name));
                std::fs::create_dir_all(d).map_err(CliError::Io)?;
                std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                println!("wrote {}", path.display());
            } else {
                println!("{k}");
            }
        } else {
            let msl = generate_msl(spec, dtypes);
            if let Some(d) = &args.dir {
                let path = d.join(format!("{}.metal", name));
                std::fs::create_dir_all(d).map_err(CliError::Io)?;
                std::fs::write(&path, &msl).map_err(CliError::Io)?;
                println!("wrote {}", path.display());
            } else {
                let mode_str = effective_mode(spec).to_string();
                println!("// ═══════════════════════════════════════════════════════");
                println!("// kernel: {}  mode: {}", name, mode_str);
                println!("// ═══════════════════════════════════════════════════════");
                println!("{}", msl);
            }
        }
    }
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn run_all_passes_and_print(k: &mut metaltile_core::ir::Kernel) {
    use metaltile_codegen::msl::MslGenerator;

    let passes = metaltile_codegen::passes::PassRegistry::standard_with_names();

    for (name, pass) in &passes {
        if let Err(e) = pass.run(k) {
            println!("\n// ── AFTER {name} ──────── ERROR ──");
            println!("// {e}");
            return;
        }
        println!("\n// ── AFTER {name} ────────────────────────");
        println!("{k}");
    }

    let generator = MslGenerator::default();
    match generator.generate(k) {
        Ok(msl) => {
            println!("\n// ── FINAL MSL ───────────────────────────────");
            println!("{msl}");
        },
        Err(e) => {
            println!("\n// ── MSL ERROR ───────────────────────────────");
            println!("// {e}");
        },
    }
}

fn generate_msl(spec: &BenchSpec, dtypes: &[DType]) -> String {
    generate_msl_dt(spec, dtypes.first().copied().unwrap_or(DType::F32))
}

fn generate_msl_dt(spec: &BenchSpec, dt: DType) -> String {
    let mut k = (spec.kernel_ir)(dt);
    if spec.kernel_name == "mt_qmm_mma" {
        metaltile_std::mlx::quantized::patch_qmm_mma_dtype_aware_skew(&mut k, dt);
    }
    let mode = effective_mode(spec);
    k.mode = mode;
    let expected_tpg =
        spec.shapes.first().map(|s| s.tpg as u32).or_else(|| spec.dispatch.tpg_hint());
    generator_for_mode(mode, expected_tpg)
        .generate(&k)
        .unwrap_or_else(|e| format!("// ERROR: {e}\n"))
}

fn print_stats_table(stats: &[metaltile_codegen::passes::PassStats]) {
    println!(
        "{:<20}  {:>10}  {:>9}  {:>6}  {:>7}",
        "pass", "ops_before", "ops_after", "delta", "time_us"
    );
    println!("{:-<20}  {:->10}  {:->9}  {:->6}  {:->7}", "", "", "", "", "");
    for s in stats {
        let delta = s.ops_after as isize - s.ops_before as isize;
        let delta_str = if delta == 0 { "  +0".to_string() } else { format!("{:>+4}", delta) };
        println!(
            "{:<20}  {:>10}  {:>9}  {:>6}  {:>7}",
            s.name, s.ops_before, s.ops_after, delta_str, s.wall_us
        );
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile build` — Compile all registered kernels.
//!
//! Spawns `__tile_runner build` as a subprocess. All kernel discovery,
//! MSL generation, xcrun compile-checks, and artifact emission happen in
//! the runner process where `metaltile-std` is linked.
//!
//! Default behavior is a compile-check (codegen MSL via xcrun, report errors,
//! no I/O). With `--emit <list> --out <dir>` it also writes artifacts:
//!
//!   --emit msl       Write per-kernel `<dir>/Resources/kernels/<name>.metal`
//!   --emit metallib  Compile + write `<dir>/Resources/kernels.metallib`
//!                    (implies msl)
//!   --emit swift     Write `<dir>/Generated/MetalTileKernels.swift`
//!   --emit ir        Write `<dir>/Resources/manifest.json` IR descriptor
//!   --emit all       Shorthand for msl,metallib,swift,ir
//!
//! `--time-passes` bypasses the JSON protocol: the runner prints the timing
//! table directly to stdout (the CLI uses `run()` with inherited stdout).

use metaltile_core::protocol::{ArtifactKind, BuildResult, ProtocolMessage};

use crate::{
    BuildArgs,
    CliError,
    project_runner::{ProjectRunner, RunnerInvocation},
    term::{Color, Style, paint_stderr, paint_stdout},
};

// ── Column formatting ─────────────────────────────────────────────────────

fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }

/// `TileCommand` wrapper for `tile build`.
pub struct BuildCommand<'a>(pub &'a BuildArgs);

impl<'a> super::TileCommand for BuildCommand<'a> {
    fn run(&self, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
        run(self.0, harness)
    }
}

pub fn run(args: &BuildArgs, harness: &crate::harness::Harness) -> Result<(), CliError> {
    let _span = tracing::info_span!("build", filter = ?args.filter_args.filter, emit = ?args.emit)
        .entered();
    let verbose = harness.verbosity() > 0;
    // CLI --sdk overrides tile.toml sdk.
    let sdk = args.sdk.clone().unwrap_or_else(|| harness.config.effective_sdk().to_string());

    // Validate --emit + --out combination early in the CLI (before spawning runner).
    if args.emit.is_some() && args.out.is_none() {
        eprintln!(
            "  {} {}",
            paint_stderr("error:", Style::new().fg(Color::Red).bold()),
            paint_stderr("--emit requires --out <dir>", Style::new().fg(Color::BrightWhite)),
        );
        eprintln!("  valid kinds: msl, metallib, swift, ir, all");
        return Err(CliError::Other("--emit requires --out <dir>".into()));
    }

    let inv = RunnerInvocation {
        command: "build".into(),
        filter: args.filter_args.filter.clone(),
        dtype: args.dtypes.clone(),
        emit: args.emit.clone(),
        out_dir: args.out.clone(),
        sdk: Some(sdk),
        names: args.names,
        time_passes: args.time_passes,
        ..Default::default()
    };

    // --time-passes: runner prints the timing table directly to stdout (no JSON).
    if args.time_passes {
        ProjectRunner::new(harness).run(&inv);
        return Ok(());
    }

    // Stream BuildResult / Artifact messages from the runner.
    let mut name_w = 20usize;
    let mut dt_w = 12usize;
    let mut total_kernels = 0u32;
    let mut errors = 0u32;
    let mut artifacts: Vec<(ArtifactKind, String)> = Vec::new();

    let runner_ok = ProjectRunner::new(harness).run_streaming(&inv, |msg| match msg {
        ProtocolMessage::Start { total, .. } => {
            total_kernels = total;
            println!(
                "{}",
                paint_stdout(
                    format!("Compiling {total} kernels"),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            if verbose {
                println!();
            }
        },
        ProtocolMessage::BuildResult(BuildResult { name, dtypes_ok, dtypes_err }) => {
            if args.names {
                // --names mode: print "name  f32/f16/bf16"
                println!("{name}  {}", dtypes_ok.join("/"));
                return;
            }
            if !dtypes_err.is_empty() {
                let dt_err_str =
                    dtypes_err.iter().map(|e| e.dtype.as_str()).collect::<Vec<_>>().join("/");
                let kernel_col =
                    paint_stdout(pad_left(&name, name_w), Style::new().fg(Color::Cyan));
                let dt_col =
                    paint_stdout(pad_left(&dt_err_str, dt_w), Style::new().fg(Color::BrightBlack));
                let status = paint_stderr("FAILED", Style::new().fg(Color::Red).bold());
                println!("  {kernel_col}  {dt_col}  {status}");
                for e in &dtypes_err {
                    eprintln!(
                        "    {}  {}",
                        paint_stdout(format!("{}:", e.dtype), Style::new().fg(Color::BrightBlack),),
                        paint_stderr(
                            e.message.lines().next().unwrap_or(&e.message),
                            Style::new().fg(Color::BrightWhite),
                        ),
                    );
                }
                errors += dtypes_err.len() as u32;
            } else if verbose && !dtypes_ok.is_empty() {
                let kernel_col =
                    paint_stdout(pad_left(&name, name_w), Style::new().fg(Color::Cyan));
                let dt_col = paint_stdout(
                    pad_left(&dtypes_ok.join("/"), dt_w),
                    Style::new().fg(Color::BrightBlack),
                );
                println!(
                    "  {kernel_col}  {dt_col}  {}",
                    paint_stdout("ok", Style::new().fg(Color::Green))
                );
            }
            // Update column widths for subsequent lines.
            name_w = name_w.max(name.len());
            dt_w = dt_w.max(dtypes_ok.join("/").len());
        },
        ProtocolMessage::Artifact { kind, path } => {
            artifacts.push((kind, path));
        },
        ProtocolMessage::ProtocolError { name, dtype, message } => {
            eprintln!(
                "  {} {} [{}]: {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                name,
                dtype,
                message,
            );
            errors += 1;
        },
        _ => {},
    });

    if !artifacts.is_empty() {
        println!();
        for (kind, path) in &artifacts {
            let kind_str = match kind {
                ArtifactKind::Msl => "msl",
                ArtifactKind::Metallib => "metallib",
                ArtifactKind::Swift => "swift",
                ArtifactKind::Ir => "ir",
            };
            println!(
                "  {} {}",
                paint_stdout(format!("{kind_str}:"), Style::new().fg(Color::BrightBlack)),
                path,
            );
        }
    }

    println!();
    let _ = total_kernels; // used in header
    if errors > 0 || !runner_ok {
        println!(
            "{}",
            paint_stderr(
                format!("Build FAILED. {} error{}.", errors, if errors == 1 { "" } else { "s" }),
                Style::new().fg(Color::Red).bold(),
            ),
        );
        Err(CliError::BuildFailure)
    } else {
        println!(
            "{}",
            paint_stdout("Compiler run successful!", Style::new().fg(Color::Green).bold()),
        );
        Ok(())
    }
}

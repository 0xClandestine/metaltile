//! MetalTile CLI — `tile` binary.
//!
//! Subcommands:
//!   bench     Benchmark suite: MetalTile vs MLX reference
//!   build     Compile all kernels to MSL and report errors
//!   inspect   Print IR and/or MSL for one kernel
//!   profile   Estimate GPU occupancy and register pressure
//!   device    Show GPU device info and supported features

mod cmd;
pub mod kernel_utils;
pub mod measure;
pub mod run_spec;
pub mod runner;
pub mod stats;
pub mod term;

use crate::term::{Color, Style, paint_stderr, paint_stdout};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit(&args[0]);
    }

    let subcommand = &args[1];
    let rest = &args[2..];

    let wants_help = rest.iter().any(|a| a == "--help" || a == "-h");

    match subcommand.as_str() {
        "--help" | "-h" => print_usage_and_exit(&args[0]),
        "bench" if wants_help => cmd::bench::help(),
        "bench" => cmd::bench::run(rest),
        "build" if wants_help => cmd::build::help(),
        "build" => cmd::build::run(rest),
        "inspect" if wants_help => cmd::inspect::help(),
        "inspect" => cmd::inspect::run(rest),
        "device" if wants_help => cmd::device::help(),
        "device" => cmd::device::run(rest),
        "test" if wants_help => cmd::test::help(),
        "test" => cmd::test::run(rest),
        "profile" if wants_help => cmd::profile::help(),
        "profile" => cmd::profile::run(rest),
        "snap" if wants_help => cmd::snap::help(),
        "snap" => cmd::snap::run(rest),
        "diff" if wants_help => cmd::diff::help(),
        "diff" => cmd::diff::run(rest),
        _ => {
            eprintln!(
                "{} {}",
                paint_stderr("error:", Style::new().fg(Color::Red).bold()),
                paint_stderr(
                    format!("unknown subcommand '{}'", subcommand),
                    Style::new().fg(Color::BrightWhite),
                ),
            );
            eprintln!();
            print_usage_and_exit(&args[0]);
            std::process::exit(1);
        },
    }
}

fn print_usage_and_exit(program: &str) {
    let name = std::path::Path::new(program)
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "tile".into());
    eprintln!(
        "{}",
        paint_stderr(
            "MetalTile CLI — benchmark, test, and inspect GPU kernels",
            Style::new().fg(Color::BrightWhite).bold(),
        ),
    );
    eprintln!();
    eprintln!(
        "{}",
        paint_stderr(
            format!("Usage: {name} <subcommand> [options]"),
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!(
        "  {}  {}",
        paint_stdout("bench", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Benchmark suite: MetalTile vs MLX reference",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("build", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Compile all kernels to MSL and report errors",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("inspect", Style::new().fg(Color::Cyan).bold()),
        paint_stdout("Print IR and/or MSL for one kernel", Style::new().fg(Color::BrightWhite),),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("profile", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Estimate GPU occupancy and register pressure",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("device", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Show GPU device info and supported features",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("test", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Run correctness checks: interpreter ↔ GPU",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("snap", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Save bench results as a regression baseline",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!(
        "  {}  {}",
        paint_stdout("diff", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            "Compare bench results to a saved baseline",
            Style::new().fg(Color::BrightWhite),
        ),
    );
    eprintln!();
    eprintln!(
        "Run '{}' for subcommand-specific options.",
        paint_stdout(format!("{name} <sub> --help"), Style::new().fg(Color::BrightBlack)),
    );

    std::process::exit(1);
}

/// Parse a `--flag <value>` pair from args.
pub(crate) fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

/// Check if `--flag` is present (boolean flag).
pub(crate) fn flag_present(args: &[String], name: &str) -> bool { args.iter().any(|a| a == name) }

/// Return the first positional argument that doesn't start with `-`.
pub(crate) fn positional(args: &[String]) -> Option<String> {
    args.iter().find(|a| !a.starts_with('-')).cloned()
}

/// Filter helper: case-insensitive substring match.
pub(crate) fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    label.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())
}

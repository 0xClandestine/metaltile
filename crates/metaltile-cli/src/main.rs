//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile CLI — `tile` binary.
//!
//! Subcommands live in `cmd/` and own their own `Args` structs.
//! This file is pure dispatch.

pub mod bench;
mod cmd;
pub mod diff;
mod error;
pub mod git;
pub mod project;
pub mod term;

use anstyle::AnsiColor;
use clap::{Parser, builder::Styles};
pub use cmd::bench::BenchArgs;
pub use cmd::build::BuildArgs;
pub use cmd::clean::CleanArgs;
pub use cmd::completions::CompletionsArgs;
pub use cmd::config::ConfigArgs;
pub use cmd::device::DeviceArgs;
pub use cmd::diff::DiffArgs;
pub use cmd::init::InitArgs;
pub use cmd::inspect::InspectArgs;
pub use cmd::snap::SnapArgs;
pub use cmd::test::TestArgs;
pub use cmd::update::UpdateArgs;
pub use error::CliError;

const CLAP_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Cyan.on_default().bold())
    .usage(AnsiColor::Cyan.on_default())
    .literal(AnsiColor::Green.on_default())
    .placeholder(AnsiColor::BrightBlack.on_default())
    .error(AnsiColor::Red.on_default().bold())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

/// MetalTile CLI — benchmark and inspect GPU kernels on Apple Silicon.
#[derive(Parser, Debug)]
#[command(name = "tile", version, about, styles = CLAP_STYLES)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Benchmark suite: MetalTile vs MLX reference.
    Bench(BenchArgs),
    /// Compile kernels to MSL; emit metallib/Swift/manifest with --emit.
    Build(BuildArgs),
    /// Run GPU correctness checks.
    Test(TestArgs),
    /// Bootstrap a new MetalTile project.
    Init(InitArgs),
    /// Print resolved Tile.toml configuration.
    Config(ConfigArgs),
    /// Remove build artifacts and cache.
    Clean(CleanArgs),
    /// Print IR and/or MSL for registered kernels.
    Inspect(InspectArgs),
    /// Show GPU device info and supported feature flags.
    Device(DeviceArgs),
    /// Save bench results as a regression baseline.
    Snap(SnapArgs),
    /// Compare bench results against a saved baseline.
    Diff(DiffArgs),
    /// Install the latest tile binary, or build from a PR / commit.
    Update(UpdateArgs),
    /// Emit shell completion scripts for bash, zsh, fish, or elvish.
    Completions(CompletionsArgs),
}

// ── Shared filter flags ──────────────────────────────────────────────────

#[derive(clap::Args, Debug, Clone, Default)]
pub struct FilterFlags {
    /// Filter by kernel name (substring, case-insensitive).
    #[arg(long = "filter", short = 'f')]
    pub filter: Option<String>,

    /// Only include kernels whose name matches this regex.
    #[arg(long = "match-kernel")]
    pub match_kernel: Option<String>,

    /// Only include kernels whose op group matches this regex.
    #[arg(long = "match-module")]
    pub match_module: Option<String>,

    /// Exclude kernels whose name matches this regex.
    #[arg(long = "no-match-kernel")]
    pub no_match_kernel: Option<String>,

    /// Exclude kernels whose op group matches this regex.
    #[arg(long = "no-match-module")]
    pub no_match_module: Option<String>,
}

impl FilterFlags {
    /// Compile flag strings into a validated `Filters` instance.
    pub fn to_filters(&self) -> Result<metaltile::harness::Filters, CliError> {
        metaltile::harness::Filters::build(
            self.filter.as_deref(),
            self.match_kernel.as_deref(),
            self.match_module.as_deref(),
            self.no_match_kernel.as_deref(),
            self.no_match_module.as_deref(),
        )
        .map_err(|e| CliError::Config(e.to_string()))
    }

    /// Returns `true` if `label` passes the `--filter` substring check.
    pub fn matches_filter(&self, label: &str) -> bool {
        self.filter
            .as_deref()
            .is_none_or(|f| label.to_ascii_lowercase().contains(&f.to_ascii_lowercase()))
    }
}

// ── Dispatch ─────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let debug_level = std::env::var("METALTILE_DEBUG").ok();
    let filter = match debug_level.as_deref() {
        Some("1") | Some("debug") => "metaltile=debug",
        Some("trace") => "metaltile=trace",
        _ => "off",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .compact()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Bench(args) => args.run(),
        Command::Build(args) => args.run(),
        Command::Test(args) => args.run(),
        Command::Init(args) => args.run(),
        Command::Config(args) => args.run(),
        Command::Clean(args) => args.run(),
        Command::Inspect(args) => args.run(),
        Command::Device(args) => args.run(),
        Command::Snap(args) => args.run(),
        Command::Diff(args) => args.run(),
        Command::Update(args) => args.run(),
        Command::Completions(args) => args.run(),
    }
    .map_err(Into::into)
}

/// Returns `true` if `label` passes a substring filter (or filter is `None`).
pub(crate) fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    filter.is_none_or(|f| label.to_ascii_lowercase().contains(&f.to_ascii_lowercase()))
}

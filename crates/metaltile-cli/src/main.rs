//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile CLI — `tile` binary.
//!
//! Subcommands:
//!   bench     Benchmark suite: MetalTile vs MLX reference
//!   test      Run #[test_kernel] correctness tests
//!   build     Compile kernels to MSL; emit metallib/Swift/manifest with --emit
//!   inspect   Print IR and/or MSL for one kernel
//!   device    Show GPU device info and supported feature flags
//!   snap      Save bench results as a regression baseline
//!   diff      Compare bench results to a saved baseline
//!   update    Install the latest tile binary (or build from a PR / commit)
//!   init      Scaffold a new MetalTile kernel project

mod cmd;
pub mod config;
mod error;
pub mod git;
pub mod harness;
pub mod project_runner;
pub mod suite_printer;
pub mod term;
use anstyle::AnsiColor;
use clap::{Parser, builder::Styles};
use cmd::TileCommand as _;
pub use error::CliError;
use regex::Regex;

const CLAP_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Cyan.on_default().bold())
    .usage(AnsiColor::Cyan.on_default())
    .literal(AnsiColor::Green.on_default())
    .placeholder(AnsiColor::BrightBlack.on_default())
    .error(AnsiColor::Red.on_default().bold())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

/// MetalTile CLI — benchmark and inspect GPU kernels on Apple Silicon.
#[derive(Parser)]
#[command(name = "tile", version, about, styles = CLAP_STYLES)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Benchmark suite: MetalTile vs MLX reference
    Bench(BenchArgs),
    /// Run #[test_kernel] correctness tests against their CPU oracle
    Test(TestArgs),
    /// Compile kernels to MSL; emit metallib/Swift/manifest with --emit
    Build(BuildArgs),
    /// Print IR and/or MSL for registered kernels
    Inspect(InspectArgs),
    /// Show GPU device info and supported feature flags
    Device(DeviceArgs),
    /// Save bench results as a regression baseline
    Snap(SnapArgs),
    /// Compare bench results against a saved baseline
    Diff(DiffArgs),
    /// Install the latest tile binary, or build from a PR / commit
    Update(UpdateArgs),
    /// Scaffold a new MetalTile kernel project in the current directory
    Init(InitArgs),
}

// ── Shared filter flags ───────────────────────────────────────────────────

/// Filter flags shared across all `tile` subcommands.
///
/// Flatten into any `*Args` struct with `#[command(flatten)]`.
#[derive(clap::Args, Debug, Default)]
pub(crate) struct FilterArgs {
    /// Only run entries whose name contains this text (substring, case-insensitive)
    #[arg(long = "filter", short = 'f')]
    pub filter: Option<String>,
    /// Only run entries whose name matches this regex (case-insensitive)
    #[arg(long = "match-name", alias = "mn")]
    pub match_name: Option<String>,
    /// Exclude entries whose name matches this regex (case-insensitive)
    #[arg(long = "no-match-name", alias = "nmn")]
    pub no_match_name: Option<String>,
    /// Only run entries in the given op group (regex, case-insensitive).
    /// Group is the path component before `/`, e.g. `ffai` from `ffai/gemv`.
    #[arg(long = "match-group", alias = "mg")]
    pub match_group: Option<String>,
    /// Exclude entries in the given op group (regex, case-insensitive)
    #[arg(long = "no-match-group", alias = "nmg")]
    pub no_match_group: Option<String>,
    /// Only run entries whose source file matches this glob pattern
    #[arg(long = "match-path", alias = "mp")]
    pub match_path: Option<String>,
    /// Exclude entries whose source file matches this glob pattern
    #[arg(long = "no-match-path", alias = "nmp")]
    pub no_match_path: Option<String>,
}

// ── FilterSpec (runtime evaluator) ───────────────────────────────────────

/// Compiled filter spec built from [`FilterArgs`]. All predicates must pass (AND logic).
pub(crate) struct FilterSpec {
    filter: Option<String>,
    match_name: Option<Regex>,
    no_match_name: Option<Regex>,
    match_group: Option<Regex>,
    no_match_group: Option<Regex>,
    match_path: Option<glob::Pattern>,
    no_match_path: Option<glob::Pattern>,
}

impl FilterSpec {
    pub fn from_args(args: &FilterArgs) -> Self {
        fn re(s: Option<&str>, flag: &str) -> Option<Regex> {
            s.map(|p| {
                Regex::new(&format!("(?i){p}")).unwrap_or_else(|e| {
                    eprintln!("tile: invalid regex for {flag}: {e}");
                    std::process::exit(2);
                })
            })
        }
        fn gl(s: Option<&str>, flag: &str) -> Option<glob::Pattern> {
            s.map(|p| {
                glob::Pattern::new(p).unwrap_or_else(|e| {
                    eprintln!("tile: invalid glob for {flag}: {e}");
                    std::process::exit(2);
                })
            })
        }
        FilterSpec {
            filter: args.filter.clone(),
            match_name: re(args.match_name.as_deref(), "--match-name"),
            no_match_name: re(args.no_match_name.as_deref(), "--no-match-name"),
            match_group: re(args.match_group.as_deref(), "--match-group"),
            no_match_group: re(args.no_match_group.as_deref(), "--no-match-group"),
            match_path: gl(args.match_path.as_deref(), "--match-path"),
            no_match_path: gl(args.no_match_path.as_deref(), "--no-match-path"),
        }
    }

    /// All filters pass for this name + source file (used for bench/test/build).
    pub fn matches(&self, name: &str, file: &str) -> bool {
        self.matches_name(name) && self.matches_file(file)
    }

    /// Name-only match (used for snap/diff which have no file info).
    pub fn matches_name(&self, name: &str) -> bool {
        if let Some(f) = &self.filter
            && !name.to_ascii_lowercase().contains(&f.to_ascii_lowercase())
        {
            return false;
        }
        if let Some(re) = &self.match_name
            && !re.is_match(name)
        {
            return false;
        }
        if let Some(re) = &self.no_match_name
            && re.is_match(name)
        {
            return false;
        }
        let group = extract_op_group(name);
        if let Some(re) = &self.match_group
            && !re.is_match(group)
        {
            return false;
        }
        if let Some(re) = &self.no_match_group
            && re.is_match(group)
        {
            return false;
        }
        true
    }

    fn matches_file(&self, file: &str) -> bool {
        if let Some(pat) = &self.match_path
            && !pat.matches(file)
        {
            return false;
        }
        if let Some(pat) = &self.no_match_path
            && pat.matches(file)
        {
            return false;
        }
        true
    }

    /// Returns the legacy `--filter` string for passing to APIs that take `Option<&str>`.
    pub fn legacy_filter(&self) -> Option<&str> { self.filter.as_deref() }

    /// True if no filter flags were set (everything passes).
    pub fn is_empty(&self) -> bool {
        self.filter.is_none()
            && self.match_name.is_none()
            && self.no_match_name.is_none()
            && self.match_group.is_none()
            && self.no_match_group.is_none()
            && self.match_path.is_none()
            && self.no_match_path.is_none()
    }
}

/// Extract the op group from a kernel name:
/// - `"ffai/gemv"` → `"ffai"`
/// - `"mt_softmax_f32"` → `"softmax"` (strips `mt_` prefix and dtype suffix)
/// - `"softmax"` → `"softmax"`
fn extract_op_group(name: &str) -> &str {
    if let Some(pos) = name.find('/') {
        return &name[..pos];
    }
    let name = name.strip_prefix("mt_").unwrap_or(name);
    name.strip_suffix("_f32")
        .or_else(|| name.strip_suffix("_f16"))
        .or_else(|| name.strip_suffix("_bf16"))
        .unwrap_or(name)
}

// ── Bench ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct BenchArgs {
    #[command(flatten)]
    filter_args: FilterArgs,
    /// Show occupancy and register profile (-v) and GPU timing stats (-vv).
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    verbose: u8,
    /// Write results as JSON to this file
    #[arg(long = "json", short = 'o')]
    json: Option<String>,
    /// Run even if the working tree has tracked-file modifications.
    /// Without this flag, bench refuses to run on a dirty tree so the
    /// numbers always tie back to a clean commit SHA.
    #[arg(long = "allow-dirty")]
    allow_dirty: bool,
    /// Opt into the post-bench diff against the target-branch baseline.
    #[arg(long = "diff")]
    diff: bool,
    /// Git ref whose `baselines/<chip>.json` to diff against (default:
    /// first of `origin/dev`, `upstream/dev`, `dev` that resolves).
    #[arg(long = "baseline-ref")]
    baseline_ref: Option<String>,
}

// ── Test ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct TestArgs {
    #[command(flatten)]
    filter_args: FilterArgs,
}

// ── Build ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct BuildArgs {
    #[command(flatten)]
    filter_args: FilterArgs,
    /// Comma-separated list of dtypes to build (f32,f16,bf16)
    #[arg(long = "dtypes")]
    dtypes: Option<String>,
    /// Print generated MSL for each kernel (-v for verbose)
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    verbose: u8,
    /// Comma-separated: msl,metallib,swift,ir,all
    #[arg(long = "emit")]
    emit: Option<String>,
    /// Output directory (required when --emit is set)
    #[arg(long = "out", short = 'o')]
    out: Option<String>,
    /// xcrun SDK (default: macosx)
    #[arg(long = "sdk", default_value = "macosx")]
    sdk: String,
    /// Run the standard pass pipeline 25× per kernel and print per-pass
    /// median wall_us instead of emitting MSL (after 5 warmup iters).
    /// Inherits `--filter` and `--dtypes`.
    #[arg(long = "time-passes", short = 't')]
    time_passes: bool,
}

// ── Inspect ──────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct InspectArgs {
    /// Kernel name to inspect (list all if omitted)
    kernel: Option<String>,
    #[command(flatten)]
    filter_args: FilterArgs,
    /// Process all kernels
    #[arg(long = "all")]
    all: bool,
    /// Print raw IR before any passes
    #[arg(long = "ir")]
    ir: bool,
    /// Print per-pass op-count reduction table
    #[arg(long = "stats")]
    stats: bool,
    /// Print IR after a specific pass name (or 'all' for every stage)
    #[arg(long = "pass")]
    pass: Option<String>,
    /// Dtype override (f32, f16, bf16, i32, u32)
    #[arg(long = "dtype")]
    dtype: Option<String>,
    /// Write output files to <path> instead of stdout
    #[arg(long = "dir", short = 'o')]
    dir: Option<String>,
}

// ── Device ───────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct DeviceArgs {
    /// Output as JSON
    #[arg(long = "json")]
    json: bool,
}

// ── Snap ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct SnapArgs {
    /// Write snapshot to <file> (default: .tile-snapshots/<sha>.json)
    #[arg(long = "out", short = 'o')]
    out: Option<String>,
    /// Promote an existing JSON file instead of re-running bench
    #[arg(long = "from")]
    from: Option<String>,
    /// Attach a note to the snapshot
    #[arg(long = "note")]
    note: Option<String>,
    #[command(flatten)]
    filter_args: FilterArgs,
}

// ── Diff ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct DiffArgs {
    /// Baseline JSON file
    baseline: String,
    /// Current JSON file (runs bench if omitted)
    current: Option<String>,
    #[command(flatten)]
    filter_args: FilterArgs,
    /// Highlight regressions larger than this percentage (default: 5)
    #[arg(long = "threshold", default_value = "5.0")]
    threshold: f64,
    /// Sort by: name, delta, pct (default: name)
    #[arg(long = "sort", default_value = "name")]
    sort: String,
    /// Only show regressions
    #[arg(long = "only-regressions")]
    only_regressions: bool,
    /// Only show improvements
    #[arg(long = "only-improvements")]
    only_improvements: bool,
}

// ── Init ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct InitArgs {
    /// Project directory name to create (default: "my-tile-kernels")
    #[arg(default_value = "my-tile-kernels")]
    name: String,
    /// Overwrite if the directory already exists
    #[arg(long = "force")]
    force: bool,
}

// ── Update ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct UpdateArgs {
    /// Print what would be installed without modifying anything
    #[arg(long = "check")]
    check: bool,
    /// Build and install from the head of this PR number (requires git + cargo)
    #[arg(long = "pr", value_name = "N", conflicts_with = "commit")]
    pr: Option<u32>,
    /// Build and install from this commit SHA (requires git + cargo)
    #[arg(long = "commit", value_name = "SHA", conflicts_with = "pr")]
    commit: Option<String>,
}

// ── Dispatch ─────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise tracing. METALTILE_DEBUG=1 enables debug-level output for all
    // metaltile crates; METALTILE_DEBUG=trace enables trace level.
    // When the env-var is absent the subscriber is still installed but the filter
    // rejects everything, so library crates pay only the ~1 ns no-subscriber cost.
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
        // Diagnostics go to stderr so they don't interleave with bench/build
        // output on stdout. `with_target` shows which crate/module emitted
        // each event — useful when tracing spans multiple crates.
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        // Print a line when each span closes so you see elapsed wall time.
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .compact()
        .init();

    let cli = Cli::parse();
    let _span = tracing::info_span!("tile", command = ?cli.command).entered();

    let h = harness::Harness::from_config();

    match cli.command {
        Command::Bench(ref args) => cmd::BenchCommand(args).run(&h)?,
        Command::Test(ref args) => cmd::TestCommand(args).run(&h)?,
        Command::Build(ref args) => cmd::BuildCommand(args).run(&h)?,
        Command::Inspect(ref args) => cmd::InspectCommand(args).run(&h)?,
        Command::Device(args) => cmd::device::run(&args)?,
        Command::Snap(args) => cmd::snap::run(&args)?,
        Command::Diff(args) => cmd::diff::run(&args)?,
        Command::Update(args) => cmd::update::run(&args)?,
        Command::Init(ref args) => cmd::InitCommand(args).run(&h)?,
    }
    Ok(())
}

/// Legacy filter helper: case-insensitive substring match on a single string.
/// Prefer [`FilterSpec`] for new code; this remains for `diff` render internals
/// that only have a name, not a full entry.
pub(crate) fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    label.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())
}

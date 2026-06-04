//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile CLI — `tile` binary.
//!
//! Subcommands:
//!   bench         Benchmark MetalTile kernels (--mlx adds the MLX A/B)
//!   test          Run #[test_kernel] correctness tests
//!   build         Compile kernels to MSL; emit metallib/Swift/manifest
//!   inspect       Print IR and/or MSL for registered kernels
//!   device        Show GPU device info and feature flags
//!   snap          Save bench results as a regression baseline
//!   diff          Compare bench results against a saved baseline
//!   update        Install the latest tile binary
//!   init          Scaffold a new MetalTile kernel project
//!   clean         Remove build artifacts and cached snapshots
//!   config        Display effective merged configuration
//!   completions   Generate shell completion scripts

pub mod bench_types;
mod cmd;
pub mod config;
mod error;
pub mod git;
pub mod harness;
pub mod project_runner;
pub mod suite_printer;
pub mod term;
use anstyle::AnsiColor;
use clap::{CommandFactory, Parser, builder::Styles};
use cmd::TileCommand as _;
pub use error::{CliError, TileExitCode};
use regex::Regex;

const CLAP_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Cyan.on_default().bold())
    .usage(AnsiColor::Cyan.on_default())
    .literal(AnsiColor::Green.on_default())
    .placeholder(AnsiColor::BrightBlack.on_default())
    .error(AnsiColor::Red.on_default().bold())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

// ── Color choice ──────────────────────────────────────────────────────────

/// When to emit coloured output.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum ColorChoice {
    /// Always emit ANSI colour codes.
    Always,
    /// Emit colour only when stdout/stderr is a TTY (default).
    #[default]
    Auto,
    /// Never emit colour codes.
    Never,
}

fn apply_color_choice(choice: ColorChoice) {
    // SAFETY: called once at startup before any threads are spawned,
    // so there are no concurrent reads of these env vars.
    unsafe {
        match choice {
            ColorChoice::Always => {
                std::env::set_var("CLICOLOR_FORCE", "1");
                std::env::remove_var("NO_COLOR");
            },
            ColorChoice::Never => {
                std::env::set_var("NO_COLOR", "1");
                std::env::remove_var("CLICOLOR_FORCE");
            },
            ColorChoice::Auto => {},
        }
    }
}

// ── Global flags ──────────────────────────────────────────────────────────

/// Global flags available to every `tile` subcommand.
///
/// All fields carry `global = true` so clap propagates them through
/// subcommand boundaries automatically.
#[derive(clap::Args, Debug, Clone, Default)]
#[command(next_help_heading = "Global options")]
pub struct GlobalArgs {
    /// Suppress all non-essential terminal output.
    #[arg(short = 'q', long, global = true, help_heading = "Global options")]
    pub quiet: bool,

    /// Output machine-readable JSON (where supported by the subcommand).
    #[arg(long, global = true, help_heading = "Global options")]
    pub json: bool,

    /// Control coloured output.
    #[arg(
        long,
        global = true,
        value_name = "WHEN",
        default_value = "auto",
        help_heading = "Global options"
    )]
    pub color: ColorChoice,

    /// Number of parallel threads (defaults to logical CPU count).
    #[arg(short = 'j', long, global = true, value_name = "N", help_heading = "Global options")]
    pub threads: Option<usize>,

    /// Increase verbosity: -v profile columns, -vv timing stats, -vvv dumps IR.
    #[arg(
        short = 'v',
        long = "verbose",
        global = true,
        action = clap::ArgAction::Count,
        help_heading = "Global options"
    )]
    pub verbose: u8,

    /// Select a named config profile from `[profiles.<name>]` in tile.toml.
    ///
    /// Can also be set via the `TILE_PROFILE` environment variable.
    #[arg(
        long,
        global = true,
        value_name = "NAME",
        env = "TILE_PROFILE",
        help_heading = "Global options"
    )]
    pub profile: Option<String>,
}

// ── CLI root ──────────────────────────────────────────────────────────────

/// MetalTile CLI — benchmark and inspect GPU kernels on Apple Silicon.
///
/// Run `tile <COMMAND> --help` for per-command documentation.
#[derive(Parser)]
#[command(name = "tile", version, about, long_about = None, styles = CLAP_STYLES)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,

    #[command(subcommand)]
    command: Command,
}

// ── Subcommand enum ───────────────────────────────────────────────────────

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Benchmark MetalTile kernels (throughput, GFLOP/s, roofline).
    ///
    /// Measures throughput (GB/s) for every registered `#[bench_kernel]` entry.
    /// By default it benches only the metaltile kernels; pass `--mlx` to also run
    /// the MLX reference kernels for an A/B speed + output-equivalence comparison.
    #[command(visible_alias = "b")]
    Bench(BenchArgs),

    /// Run `#[test_kernel]` correctness tests against their CPU oracle.
    ///
    /// Each test computes expected output on the CPU and compares GPU output
    /// element-wise within the kernel's declared tolerance.
    #[command(visible_alias = "t")]
    Test(TestArgs),

    /// Compile all registered kernels to MSL; optionally emit build artifacts.
    ///
    /// Default: compile-check only (no output written).
    /// With `--emit <kinds> --out <dir>`: write MSL, metallib, Swift wrappers,
    /// and/or an IR manifest to disk.
    #[command(visible_alias = "c", visible_alias = "compile")]
    Build(BuildArgs),

    /// Print IR and/or MSL for registered kernels.
    ///
    /// Lists all kernels when called without arguments.  Pass a kernel name
    /// to inspect a specific kernel, or use `--all` to dump every kernel.
    #[command(visible_alias = "in")]
    Inspect(InspectArgs),

    /// Show GPU device info and supported Metal feature flags.
    #[command(visible_alias = "d")]
    Device(DeviceArgs),

    /// Save current bench results as a regression baseline JSON file.
    #[command(visible_alias = "s")]
    Snap(SnapArgs),

    /// Compare bench results against a saved baseline.
    ///
    /// Pass two JSON files to diff offline, or omit the second argument to
    /// run bench live and compare against the baseline.
    Diff(DiffArgs),

    /// Install the latest `tile` binary, or build from a PR / commit.
    #[command(visible_alias = "u")]
    Update(UpdateArgs),

    /// Scaffold a new MetalTile kernel project in a new directory.
    ///
    /// Creates a minimal Cargo workspace with an example `#[bench_kernel]`.
    Init(InitArgs),

    /// Remove build artifacts (`*.air`, `*.metallib`) and optionally snapshots.
    ///
    /// By default only removes generated build outputs. Pass `--snapshots` to
    /// also wipe `.tile-snapshots/`, or `--all` to remove everything.
    #[command(visible_alias = "cl")]
    Clean(CleanArgs),

    /// Display the effective merged configuration.
    ///
    /// Prints the config as TOML (or JSON with `--json`), showing the result
    /// of merging defaults → extends base → tile.toml → profile → env vars.
    /// Useful for debugging why a setting is not taking effect.
    #[command(visible_alias = "co")]
    Config(ConfigArgs),

    /// Generate shell completion script and print it to stdout.
    ///
    /// Pipe into your shell's completion directory, e.g.:
    ///   tile completions zsh > ~/.zfunc/_tile
    ///   tile completions bash > /etc/bash_completion.d/tile
    #[command(visible_alias = "com")]
    Completions(CompletionsArgs),
}

// ── Shared filter flags ───────────────────────────────────────────────────

/// Filter flags shared across all `tile` subcommands.
///
/// All predicates must pass (AND semantics).  Flatten into any `*Args`
/// struct with `#[command(flatten)]`.
#[derive(clap::Args, Debug, Default, Clone)]
#[command(next_help_heading = "Filter options")]
pub(crate) struct FilterArgs {
    /// Only run entries whose name contains this text (substring, case-insensitive).
    #[arg(long = "filter", short = 'f', help_heading = "Filter options")]
    pub filter: Option<String>,
    /// Only run entries whose name matches this regex (case-insensitive).
    #[arg(long = "match-name", visible_alias = "mn", help_heading = "Filter options")]
    pub match_name: Option<String>,
    /// Exclude entries whose name matches this regex (case-insensitive).
    #[arg(long = "no-match-name", visible_alias = "nmn", help_heading = "Filter options")]
    pub no_match_name: Option<String>,
    /// Only run entries in this op group (regex). Group = path component before `/`.
    #[arg(long = "match-group", visible_alias = "mg", help_heading = "Filter options")]
    pub match_group: Option<String>,
    /// Exclude entries in this op group (regex).
    #[arg(long = "no-match-group", visible_alias = "nmg", help_heading = "Filter options")]
    pub no_match_group: Option<String>,
    /// Only run entries whose source file matches this glob pattern.
    #[arg(long = "match-path", visible_alias = "mp", help_heading = "Filter options")]
    pub match_path: Option<String>,
    /// Exclude entries whose source file matches this glob pattern.
    #[arg(long = "no-match-path", visible_alias = "nmp", help_heading = "Filter options")]
    pub no_match_path: Option<String>,
}

// ── FilterSpec (runtime evaluator) ───────────────────────────────────────

/// Compiled filter spec built from [`FilterArgs`].  All predicates must pass (AND).
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

    /// All filters pass for this name + source file.
    #[allow(dead_code)]
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
#[command(
    after_help = "EXAMPLES:\n  tile bench                      # run all benchmarks\n  tile bench softmax              # run benchmarks matching 'softmax'\n  tile bench -f gemv --vv         # gemv with timing stats\n  tile bench --diff               # bench + auto-diff vs target branch baseline"
)]
struct BenchArgs {
    /// Kernel name filter (shorthand for --filter; if it contains '/' it is
    /// treated as a --match-path glob instead).
    #[arg(value_name = "FILTER", value_hint = clap::builder::ValueHint::Other)]
    path: Option<String>,

    #[command(flatten)]
    filter_args: FilterArgs,

    /// Override the number of timed iterations (default: from tile.toml or 3).
    #[arg(long, value_name = "N", help_heading = "Bench options")]
    runs: Option<usize>,

    /// Override the number of warmup iterations (default: from tile.toml or 1).
    #[arg(long, value_name = "N", help_heading = "Bench options")]
    warmup: Option<usize>,

    /// Save results as JSON to this file path.
    #[arg(long, short = 'o', value_name = "FILE", help_heading = "Output options")]
    out: Option<String>,

    /// Run even if the working tree has uncommitted changes.
    ///
    /// By default bench refuses to run on a dirty tree so numbers always
    /// tie back to a clean commit SHA.
    #[arg(long, help_heading = "Bench options")]
    allow_dirty: bool,

    /// After bench, auto-diff against the target-branch baseline.
    #[arg(long, help_heading = "Bench options")]
    diff: bool,

    /// Git ref to use for baseline auto-diff (default: origin/dev or dev).
    #[arg(long, value_name = "REF", help_heading = "Bench options")]
    baseline_ref: Option<String>,

    /// Also bench the MLX reference kernels — an A/B speed comparison plus an
    /// output-equivalence check — for kernels that have one.
    ///
    /// Off by default: the metaltile kernels have superseded the MLX references,
    /// per-kernel correctness is covered by `tile test`, and running the MLX side
    /// roughly doubles bench time. Pass `--mlx` when you specifically want the
    /// side-by-side comparison.
    #[arg(long, visible_alias = "reference", help_heading = "Bench options")]
    mlx: bool,
}

// ── Test ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
#[command(
    after_help = "EXAMPLES:\n  tile test                   # run all tests\n  tile test softmax           # tests matching 'softmax'\n  tile test -f rms_norm       # tests matching 'rms_norm'"
)]
struct TestArgs {
    /// Kernel name filter (shorthand for --filter; if it contains '/' treated
    /// as a --match-path glob).
    #[arg(value_name = "FILTER", value_hint = clap::builder::ValueHint::Other)]
    path: Option<String>,

    /// Stop running tests after the first failure.
    #[arg(long, help_heading = "Test options")]
    fail_fast: bool,

    /// List matching tests without running them.
    #[arg(long, short = 'l', help_heading = "Display options")]
    list: bool,

    /// Show a live per-test progress indicator while tests run.
    #[arg(long, help_heading = "Display options")]
    show_progress: bool,

    /// Print a per-suite summary table after all tests finish.
    #[arg(long, help_heading = "Display options")]
    summary: bool,

    /// With --summary: show individual test rows in the table.
    #[arg(long, requires = "summary", help_heading = "Display options")]
    detailed: bool,

    #[command(flatten)]
    filter_args: FilterArgs,
}

// ── Build ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
#[command(
    after_help = "EXAMPLES:\n  tile build                       # compile-check all kernels\n  tile build -f softmax            # compile-check softmax kernels\n  tile build --emit msl --out out/ # emit MSL source files\n  tile build --emit all --out out/ # emit MSL + metallib + Swift + IR"
)]
struct BuildArgs {
    #[command(flatten)]
    filter_args: FilterArgs,

    /// Comma-separated list of dtypes to compile (e.g. f32,f16,bf16).
    #[arg(long, value_name = "LIST", help_heading = "Build options")]
    dtypes: Option<String>,

    /// Comma-separated emit kinds: msl, metallib, swift, ir, all.
    #[arg(long, value_name = "KINDS", help_heading = "Output options")]
    emit: Option<String>,

    /// Output directory (required when --emit is set).
    #[arg(long, short = 'o', value_name = "DIR", help_heading = "Output options")]
    out: Option<String>,

    /// xcrun SDK to use for Metal compilation (default: from tile.toml or macosx).
    #[arg(long, value_name = "SDK", help_heading = "Build options")]
    sdk: Option<String>,

    /// List kernel names that would be compiled, without compiling.
    #[arg(long, short = 'n', help_heading = "Build options")]
    names: bool,

    /// Run the standard pass pipeline and report per-pass median wall time.
    #[arg(long, short = 't', help_heading = "Build options")]
    time_passes: bool,
}

// ── Inspect ──────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
#[command(
    after_help = "EXAMPLES:\n  tile inspect                     # list all kernels\n  tile inspect mt_softmax_f32      # print final MSL\n  tile inspect mt_softmax_f32 --ir # print raw IR\n  tile inspect --all -o /tmp/out   # dump all kernels to disk"
)]
struct InspectArgs {
    /// Kernel name to inspect (lists all registered kernels when omitted).
    kernel: Option<String>,

    #[command(flatten)]
    filter_args: FilterArgs,

    /// Process all kernels (required when no kernel name is given and --dir is set).
    #[arg(long, help_heading = "Inspect options")]
    all: bool,

    /// Print raw IR before any passes.
    #[arg(long, help_heading = "Inspect options")]
    ir: bool,

    /// Print per-pass op-count reduction table.
    #[arg(long, help_heading = "Inspect options")]
    stats: bool,

    /// Print IR after a specific pass name, or 'all' for every stage.
    #[arg(long, value_name = "NAME", help_heading = "Inspect options")]
    pass: Option<String>,

    /// Dtype override (f32, f16, bf16, i32, u32).
    #[arg(long, value_name = "DTYPE", help_heading = "Inspect options")]
    dtype: Option<String>,

    /// Write output files to <path> instead of stdout.
    #[arg(long, short = 'o', value_name = "DIR", help_heading = "Output options")]
    dir: Option<String>,
}

// ── Device ───────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct DeviceArgs {
    /// Output as JSON (also enabled by global --json flag).
    #[arg(long)]
    json: bool,
}

// ── Snap ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct SnapArgs {
    /// Write snapshot to this file (default: `.tile-snapshots/<sha>.json`).
    #[arg(long, short = 'o', value_name = "FILE")]
    out: Option<String>,

    /// Promote an existing JSON bench result file instead of re-running bench.
    #[arg(long, value_name = "FILE")]
    from: Option<String>,

    /// Attach a freeform note to the snapshot.
    #[arg(long, value_name = "TEXT")]
    note: Option<String>,

    #[command(flatten)]
    filter_args: FilterArgs,
}

// ── Diff ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct DiffArgs {
    /// Baseline JSON file.
    baseline: String,

    /// Current JSON file (runs bench if omitted).
    current: Option<String>,

    #[command(flatten)]
    filter_args: FilterArgs,

    /// Highlight regressions larger than this percentage (default: 5).
    #[arg(long, default_value = "5.0", value_name = "PCT")]
    threshold: f64,

    /// Sort by: name, delta, pct (default: name).
    #[arg(long, default_value = "name", value_name = "KEY")]
    sort: String,

    /// Only show regressions (hide improvements and unchanged rows).
    #[arg(long)]
    only_regressions: bool,

    /// Only show improvements (hide regressions and unchanged rows).
    #[arg(long)]
    only_improvements: bool,
}

// ── Init ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct InitArgs {
    /// Name for the new project directory (default: my-tile-kernels).
    #[arg(default_value = "my-tile-kernels")]
    name: String,

    /// Overwrite an existing directory if it already exists.
    #[arg(long)]
    force: bool,
}

// ── Clean ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct CleanArgs {
    /// Also remove regression baselines in `.tile-snapshots/`.
    #[arg(long)]
    snapshots: bool,

    /// Remove all build artifacts and regression baselines.
    #[arg(long)]
    all: bool,
}

// ── Config ───────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct ConfigArgs {
    /// Output as JSON instead of TOML (also enabled by global --json).
    #[arg(long)]
    json: bool,
}

// ── Completions ───────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct CompletionsArgs {
    /// Shell to generate completions for.
    #[arg(value_enum)]
    shell: clap_complete::Shell,
}

// ── Update ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
struct UpdateArgs {
    /// Print what would be installed without making any changes.
    #[arg(long)]
    check: bool,

    /// Build and install from the head of this PR number (requires git + cargo).
    #[arg(long, value_name = "N", conflicts_with = "commit")]
    pr: Option<u32>,

    /// Build and install from this commit SHA (requires git + cargo).
    #[arg(long, value_name = "SHA", conflicts_with = "pr")]
    commit: Option<String>,
}

// ── Dispatch ─────────────────────────────────────────────────────────────

fn main() {
    // Initialise tracing. METALTILE_DEBUG=1 → debug level; =trace → trace.
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

    // Apply --color choice before any paint calls (sets NO_COLOR / CLICOLOR_FORCE).
    apply_color_choice(cli.global.color);

    let _span = tracing::info_span!("tile", command = ?cli.command).entered();

    let h = harness::Harness::new(cli.global);

    let result: Result<(), CliError> = match cli.command {
        Command::Bench(ref args) => cmd::BenchCommand(args).run(&h),
        Command::Test(ref args) => cmd::TestCommand(args).run(&h),
        Command::Build(ref args) => cmd::BuildCommand(args).run(&h),
        Command::Inspect(ref args) => cmd::InspectCommand(args).run(&h),
        Command::Device(ref args) => cmd::device::run(args, &h),
        Command::Snap(ref args) => cmd::snap::run(args),
        Command::Diff(ref args) => cmd::diff::run(args),
        Command::Update(ref args) => cmd::update::run(args),
        Command::Init(ref args) => cmd::InitCommand(args).run(&h),
        Command::Clean(ref args) => cmd::clean::run(args),
        Command::Config(ref args) => cmd::config::run(args, &h),
        Command::Completions(ref args) => {
            clap_complete::generate(
                args.shell,
                &mut Cli::command(),
                "tile",
                &mut std::io::stdout(),
            );
            Ok(())
        },
    };

    // Print config-load warnings (unknown keys, deprecated fields).
    if !matches!(cli.command, Command::Config(_) | Command::Completions(_)) {
        h.print_warnings();
    }

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(e.exit_code());
    }
}

/// Legacy filter helper: case-insensitive substring match on a single string.
/// Prefer [`FilterSpec`] for new code.
pub(crate) fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    label.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())
}

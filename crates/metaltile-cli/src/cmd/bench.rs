//! `tile bench` command — benchmark GPU kernels.
//!
//! Orchestrates the bench lifecycle:
//! 1. Find `tile.toml` walking up from CWD.
//! 2. Generate runner harness source into `$CARGO_TARGET_DIR/tile/`.
//! 3. Spawn `cargo run --bin __tile_runner -- bench [--filter …]`.
//! 4. Stream JSON lines → render live table.
//! 5. Optionally write `results.json`.

use std::{
    io::{BufRead, Write},
    path::PathBuf,
    process::{Command, Stdio},
};

use metaltile_core::{protocol::ProtocolMessage, tile_config::TileConfig};

use crate::{
    CliError,
    term::{Color, Style, paint_stderr},
};

/// Run the bench subcommand.
pub fn run(
    filter: Option<String>,
    verbose: u8,
    json_path: Option<String>,
    allow_dirty: bool,
    _diff: bool,
    _baseline_ref: Option<String>,
) -> Result<(), CliError> {
    // ── 1. Find tile.toml ───────────────────────────────────────────────
    let cwd = std::env::current_dir()?;
    let config =
        TileConfig::discover(&cwd).map_err(|e| CliError::Other(e.to_string()))?.unwrap_or_default();

    // ── 2. Check dirty tree ─────────────────────────────────────────────
    if !allow_dirty {
        let dirty = is_working_tree_dirty()?;
        if dirty {
            eprintln!(
                "{} Working tree has tracked-file modifications.\n  \
                 Pass {} to bench anyway, or commit/stash your changes first.\n  \
                 Bench results tie back to a clean commit SHA.",
                paint_stderr("warning:", Style::new().fg(Color::Yellow).bold()),
                paint_stderr("--allow-dirty", Style::new().fg(Color::Green).bold()),
            );
            return Err(CliError::Other(
                "dirty working tree (pass --allow-dirty to override)".into(),
            ));
        }
    }

    // ── 3. Generate runner harness ──────────────────────────────────────
    let target_dir = get_target_dir()?;
    let harness_dir = target_dir.join("tile");
    std::fs::create_dir_all(&harness_dir)?;
    let harness_path = harness_dir.join("__runner.rs");
    generate_harness(&harness_path, &config)?;

    // ── 4. Build and spawn runner ───────────────────────────────────────
    let cargo_args = &config.runner.cargo_args;
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--bin", "__tile_runner"]);
    for arg in cargo_args {
        cmd.arg(arg);
    }
    cmd.arg("--");
    cmd.arg("bench");
    if let Some(ref f) = filter {
        cmd.args(["--filter", f]);
    }
    // Forward reference_metal_path so the runner can locate reference kernels.
    if let Some(ref path) = config.bench.reference_metal_path {
        cmd.env("TILE_REF_METAL_PATH", cwd.join(path));
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit()); // compiler errors go to stderr

    let mut child =
        cmd.spawn().map_err(|e| CliError::Subprocess(format!("failed to spawn runner: {e}")))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let reader = std::io::BufReader::new(stdout);

    // ── 5. Stream JSON lines → render ──────────────────────────────────
    let mut results: Vec<ProtocolMessage> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| CliError::Subprocess(format!("read error: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        match ProtocolMessage::from_json_line(line.as_bytes()) {
            Ok(msg) => {
                render_message(&msg, verbose);
                results.push(msg);
            },
            Err(e) => {
                eprintln!(
                    "{} Failed to parse runner output: {e}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold())
                );
                eprintln!("  Raw line: {line:.100}");
            },
        }
    }

    let status = child
        .wait()
        .map_err(|e| CliError::Subprocess(format!("runner process wait failed: {e}")))?;

    if !status.success() {
        return Err(CliError::Subprocess(format!(
            "runner exited with code {}",
            status.code().unwrap_or(-1)
        )));
    }

    // ── 6. Optionally write results.json ────────────────────────────────
    if let Some(path) = json_path {
        let out_path = PathBuf::from(&path);
        let mut file = std::fs::File::create(&out_path)?;
        for msg in &results {
            file.write_all(&msg.to_json_line())?;
        }
        eprintln!(
            "{} wrote {} messages to {}",
            paint_stderr("saved:", Style::new().fg(Color::Green).bold()),
            results.len(),
            out_path.display(),
        );
    }

    Ok(())
}

/// Render a single protocol message to stderr.
fn render_message(msg: &ProtocolMessage, _verbose: u8) {
    match msg {
        ProtocolMessage::Start { runner_version, total_benches, total_tests } => {
            eprintln!(
                "{} runner v{}, {} benches, {} tests",
                paint_stderr("start", Style::new().fg(Color::Cyan).bold()),
                runner_version,
                total_benches,
                total_tests,
            );
        },
        ProtocolMessage::BenchResult(b) => {
            let color = if b.correct { Color::Green } else { Color::Red };
            eprintln!(
                "  {} {:<30} {:>6}  {:>8.1} GB/s  {:>7.1} µs",
                paint_stderr("bench", Style::new().fg(color).bold()),
                b.name,
                b.dtype,
                b.mt_gbps,
                b.mean_us,
            );
            if let Some(ref_gbps) = b.ref_gbps {
                let pct = b.mt_pct.unwrap_or(0.0);
                eprintln!("  {:>34} ref {:>8.1} GB/s  {:>+5.1}%", "", ref_gbps, pct,);
            }
        },
        ProtocolMessage::TestResult(t) => {
            let color = if t.passed { Color::Green } else { Color::Red };
            let status = if t.passed { "PASS" } else { "FAIL" };
            eprintln!(
                "  {} {:<30} {:>6}  {:<4}  max_err={:.2e}",
                paint_stderr("test", Style::new().fg(color).bold()),
                t.name,
                t.dtype,
                status,
                t.max_err,
            );
        },
        ProtocolMessage::ProtocolError { name, dtype, message } => {
            eprintln!(
                "  {} {:<30} {:>6}  {message}",
                paint_stderr("error", Style::new().fg(Color::Red).bold()),
                name,
                dtype,
            );
        },
        ProtocolMessage::Done { bench_passed, bench_failed, test_passed, test_failed } => {
            eprintln!(
                "{} benches: {} passed, {} failed  |  tests: {} passed, {} failed",
                paint_stderr("done", Style::new().fg(Color::Cyan).bold()),
                bench_passed,
                bench_failed,
                test_passed,
                test_failed,
            );
        },
    }
}

/// Generate the runner harness source file.
///
/// The harness is a single `fn main()` that calls `metaltile::runner::run`.
fn generate_harness(path: &std::path::Path, _config: &TileConfig) -> Result<(), CliError> {
    // Only generate if absent (or stale — simple mtime check).
    let needs_generation = if path.exists() {
        let metadata = std::fs::metadata(path)?;
        let elapsed = metadata.modified().ok().and_then(|m| m.elapsed().ok());
        // Regenerate if older than 1 hour (conservative).
        elapsed.map(|e| e.as_secs() > 3600).unwrap_or(true)
    } else {
        true
    };

    if needs_generation {
        let source = r#"// auto-generated by tile — do not edit, do not check in
fn main() {
    metaltile::runner::run(metaltile::runner::Args::from_env());
}
"#;
        std::fs::write(path, source)?;
    }

    Ok(())
}

/// Get the Cargo target directory.
fn get_target_dir() -> Result<PathBuf, CliError> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .output()
        .map_err(|e| CliError::Subprocess(format!("cargo metadata failed: {e}")))?;
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let target_dir = metadata["target_directory"]
        .as_str()
        .ok_or_else(|| CliError::Other("could not determine target directory".into()))?;
    let mut path = PathBuf::from(target_dir);
    // Canonicalize to resolve any symlinks
    if let Ok(canonical) = path.canonicalize() {
        path = canonical;
    }
    Ok(path)
}

/// Check if the working tree has tracked-file modifications.
fn is_working_tree_dirty() -> Result<bool, CliError> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|e| CliError::Subprocess(format!("git status failed: {e}")))?;
    // If there's any output, the tree is dirty.
    Ok(!output.stdout.is_empty())
}

//! `tile test` command — run GPU kernel correctness tests.
//!
//! Same lifecycle as `tile bench` but invokes the runner with `-- test`.

use std::{
    io::BufRead,
    path::PathBuf,
    process::{Command, Stdio},
};

use metaltile_core::{protocol::ProtocolMessage, tile_config::TileConfig};

use crate::{
    CliError,
    term::{Color, Style, paint_stderr},
};

/// Run the test subcommand.
pub fn run(filter: Option<String>, verbose: u8, allow_dirty: bool) -> Result<(), CliError> {
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
                 Pass {} to proceed anyway, or commit/stash first.",
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
    cmd.arg("test");
    if let Some(ref f) = filter {
        cmd.args(["--filter", f]);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());

    let mut child =
        cmd.spawn().map_err(|e| CliError::Subprocess(format!("failed to spawn runner: {e}")))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let reader = std::io::BufReader::new(stdout);

    // ── 5. Stream JSON lines → render ──────────────────────────────────
    for line in reader.lines() {
        let line = line.map_err(|e| CliError::Subprocess(format!("read error: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        match ProtocolMessage::from_json_line(line.as_bytes()) {
            Ok(msg) => render_message(&msg, verbose),
            Err(e) => {
                eprintln!(
                    "{} Failed to parse runner output: {e}",
                    paint_stderr("error:", Style::new().fg(Color::Red).bold())
                );
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

    Ok(())
}

/// Render a single protocol message to stderr.
fn render_message(msg: &ProtocolMessage, _verbose: u8) {
    match msg {
        ProtocolMessage::Start { runner_version, total_benches: _, total_tests } => {
            eprintln!(
                "{} runner v{}, {} tests",
                paint_stderr("start", Style::new().fg(Color::Cyan).bold()),
                runner_version,
                total_tests,
            );
        },
        ProtocolMessage::BenchResult(_) => {
            // Tests don't emit bench results — silently ignore.
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
        ProtocolMessage::Done { bench_passed: _, bench_failed: _, test_passed, test_failed } => {
            eprintln!(
                "{} tests: {} passed, {} failed",
                paint_stderr("done", Style::new().fg(Color::Cyan).bold()),
                test_passed,
                test_failed,
            );
        },
    }
}

/// Generate the runner harness source file.
fn generate_harness(path: &std::path::Path, _config: &TileConfig) -> Result<(), CliError> {
    let needs_generation = if path.exists() {
        let metadata = std::fs::metadata(path)?;
        let elapsed = metadata.modified().ok().and_then(|m| m.elapsed().ok());
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
    Ok(PathBuf::from(target_dir))
}

/// Check if the working tree has tracked-file modifications.
fn is_working_tree_dirty() -> Result<bool, CliError> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|e| CliError::Subprocess(format!("git status failed: {e}")))?;
    Ok(!output.stdout.is_empty())
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Project detection and harness compilation.
//!
//! Separates build-system concerns from config parsing.

use std::path::{Path, PathBuf};

use crate::project::config::TileConfig;

// ── Project detection ────────────────────────────────────────────────────

/// Check if a `Tile.toml` exists in CWD.
pub fn has_tile_toml() -> bool { Path::new("Tile.toml").exists() }

/// Get the active profile name from env var or default.
pub fn active_profile() -> String {
    std::env::var("TILE_PROFILE").unwrap_or_else(|_| "default".to_string())
}

/// Resolve the output directory from Tile.toml (or fallback).
pub fn resolve_out_dir() -> String {
    if has_tile_toml() {
        let cfg = TileConfig::load(Path::new(".")).ok().flatten();
        let profile_name = active_profile();
        cfg.as_ref()
            .map(|c| c.resolved_profile(&profile_name).out)
            .unwrap_or_else(|| "tile-out".to_string())
    } else {
        "tile-out".to_string()
    }
}

/// Resolve the air-file cache directory (used by `tile clean` and `tile build`).
pub fn air_cache_dir() -> PathBuf {
    std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new("target").join("tile-build-air"))
}

// ── Harness compilation ──────────────────────────────────────────────────

/// Compile a bench harness binary by running `cargo build` in the project.
///
/// Returns the path to the compiled binary.
///
/// # Arguments
/// * `project_dir` — path to the project root containing Cargo.toml
/// * `release` — if true, builds with `--release`; debug otherwise
pub fn compile_harness(project_dir: &Path, release: bool) -> Result<PathBuf, String> {
    let mut cargo_cmd = std::process::Command::new("cargo");
    cargo_cmd.args(["build", "-q"]);
    if release {
        cargo_cmd.arg("--release");
    }
    cargo_cmd.current_dir(project_dir);

    let output = cargo_cmd.output().map_err(|e| format!("cargo build failed: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("harness build failed:\n{stderr}"));
    }

    // Discover the binary name from Cargo.toml
    let cargo_toml_path = project_dir.join("Cargo.toml");
    let cargo_text = std::fs::read_to_string(&cargo_toml_path)
        .map_err(|e| format!("failed to read Cargo.toml: {e}"))?;
    let cargo_val: toml::Value =
        toml::from_str(&cargo_text).map_err(|e| format!("failed to parse Cargo.toml: {e}"))?;

    let bin_name = cargo_val
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .ok_or_else(|| "no package.name in Cargo.toml".to_string())?;

    // Binary lives in target/<profile>/<name>
    let profile_dir = if release { "release" } else { "debug" };
    let binary = project_dir.join("target").join(profile_dir).join(bin_name);
    if !binary.exists() {
        return Err(format!(
            "built binary not found at {} — did cargo build succeed?",
            binary.display()
        ));
    }
    Ok(binary)
}

/// A `CompileService` trait for testability.
pub trait CompileService: Send + Sync {
    fn compile_harness(&self, project_dir: &Path, release: bool) -> Result<PathBuf, String>;
}

/// Real compile service delegating to `cargo build`.
pub struct RealCompileService;

impl CompileService for RealCompileService {
    fn compile_harness(&self, project_dir: &Path, release: bool) -> Result<PathBuf, String> {
        compile_harness(project_dir, release)
    }
}

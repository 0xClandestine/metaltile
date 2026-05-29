//! `tile.toml` project manifest types.
//!
//! Every MetalTile project that uses `tile bench` / `tile test` has a
//! `tile.toml` at the project root.  This module defines the deserialisable
//! config types.

use serde::Deserialize;

/// Root-level `tile.toml` configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TileConfig {
    /// Project metadata.
    #[serde(default)]
    pub project: ProjectConfig,
    /// Runner subprocess configuration.
    #[serde(default)]
    pub runner: RunnerConfig,
    /// Benchmark defaults.
    #[serde(default)]
    pub bench: BenchConfig,
    /// Test defaults.
    #[serde(default)]
    pub test: TestConfig,
}

/// `[project]` section.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProjectConfig {
    /// Project name.
    #[serde(default)]
    pub name: String,
}

/// `[runner]` section — controls how the auto-generated harness is built.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RunnerConfig {
    /// Extra Cargo args forwarded when spawning the auto-generated runner.
    #[serde(default)]
    pub cargo_args: Vec<String>,
}

/// `[bench]` section — default benchmark parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchConfig {
    /// Number of warmup iterations before measurement.
    #[serde(default = "default_warmup_iters")]
    pub warmup_iters: u32,
    /// Number of measured iterations for timing.
    #[serde(default = "default_bench_iters")]
    pub bench_iters: u32,
    /// Directory containing reference `.metal` source files.
    ///
    /// When set, `#[bench]` functions that call `.with_reference()` can name
    /// a file relative to this directory (e.g. `"unary.metal"`) and the
    /// runner will compile and time that Metal kernel alongside the MetalTile
    /// kernel, reporting live `ref_gbps` / `mt_pct`.
    ///
    /// For MetalTile itself this points at the checked-out MLX source tree:
    ///
    /// ```toml
    /// [bench]
    /// reference_metal_path = ".cache/mlx/mlx/backend/metal/kernels"
    /// ```
    ///
    /// Any project with its own reference kernels can point here instead.
    /// Omit the field entirely if you have no reference kernels.
    #[serde(default)]
    pub reference_metal_path: Option<std::path::PathBuf>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            warmup_iters: default_warmup_iters(),
            bench_iters: default_bench_iters(),
            reference_metal_path: None,
        }
    }
}

/// `[test]` section — default test parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct TestConfig {
    /// Default tolerance applied globally unless overridden per-kernel.
    #[serde(default = "default_test_tol")]
    pub default_tol: f64,
}

impl Default for TestConfig {
    fn default() -> Self { TestConfig { default_tol: default_test_tol() } }
}

const fn default_warmup_iters() -> u32 { 5 }
const fn default_bench_iters() -> u32 { 20 }
const fn default_test_tol() -> f64 { 1e-4 }

impl TileConfig {
    /// Load `tile.toml` from the given directory, walking up to find one.
    ///
    /// Returns `None` if no `tile.toml` is found in the current directory
    /// or any parent.
    pub fn discover(start_dir: &std::path::Path) -> Result<Option<Self>, crate::Error> {
        let mut dir = Some(start_dir.to_path_buf());
        while let Some(current) = dir {
            let candidate = current.join("tile.toml");
            if candidate.exists() {
                let contents = std::fs::read_to_string(&candidate).map_err(|e| {
                    crate::Error::Internal(format!("failed to read {}: {e}", candidate.display()))
                })?;
                let config: TileConfig = toml::from_str(&contents).map_err(|e| {
                    crate::Error::Internal(format!("failed to parse {}: {e}", candidate.display()))
                })?;
                return Ok(Some(config));
            }
            dir = current.parent().map(|p| p.to_path_buf());
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.bench.warmup_iters, 5);
        assert_eq!(cfg.bench.bench_iters, 20);
        assert!((cfg.test.default_tol - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn parse_example_toml() {
        let toml_str = r#"
[project]
name = "metaltile-std"

[runner]
cargo_args = ["--release"]

[bench]
warmup_iters = 5
bench_iters = 20

[test]
default_tol = 1e-4
"#;
        let cfg: TileConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(cfg.project.name, "metaltile-std");
        assert_eq!(cfg.runner.cargo_args, vec!["--release"]);
        assert_eq!(cfg.bench.warmup_iters, 5);
        assert_eq!(cfg.bench.bench_iters, 20);
        assert!((cfg.test.default_tol - 1e-4).abs() < 1e-10);
    }
}

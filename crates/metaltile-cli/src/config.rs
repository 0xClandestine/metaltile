//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `TileConfig` — layered configuration for the `tile` CLI.
//!
//! # Load order (later layers override earlier ones)
//!
//! 1. Built-in defaults
//! 2. `extends` base file (resolved relative to the tile.toml that references it)
//! 3. `tile.toml` found by walking up from CWD (optional, ignored if absent)
//! 4. Profile-specific overrides from `[profiles.<name>]` section
//! 5. `TILE_*` environment variables
//! 6. CLI flags (applied at the harness level after extraction)
//!
//! # Profile selection
//!
//! Select a profile with `TILE_PROFILE=ci` or `--profile ci`.
//!
//! ```toml
//! [profiles.ci]
//! verbose = 0
//!
//! [profiles.ci.bench]
//! runs = 15
//! warmup_runs = 3
//! ```
//!
//! # Extends / inheritance
//!
//! ```toml
//! extends = "~/.metaltile/global.toml"
//! ```
//!
//! # Sub-tables
//!
//! ```toml
//! [bench]
//! runs = 3
//! warmup_runs = 1
//!
//! [build]
//! sdk = "macosx"
//! default_dtypes = ["f32", "f16", "bf16"]
//! time_passes = false
//!
//! [runner]
//! binary = "__tile_runner"
//! extra_args = []
//! ```
//!
//! Top-level flat fields (`runs`, `warmup_runs`, `runner_binary`) are kept for
//! backward compatibility and are overridden by their sub-table counterparts.

use std::path::{Path, PathBuf};

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};

// ── Sub-table structs ─────────────────────────────────────────────────────

/// Bench-specific configuration (`[bench]` table in tile.toml).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchConfig {
    /// Number of timed benchmark iterations per kernel (after warmup).
    pub runs: Option<usize>,
    /// Number of warmup dispatches before timing begins.
    pub warmup_runs: Option<usize>,
}

/// Build-specific configuration (`[build]` table in tile.toml).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Dtypes to compile by default when `--dtypes` is not passed.
    pub default_dtypes: Option<Vec<String>>,
    /// xcrun SDK to use for Metal compilation.
    pub sdk: Option<String>,
    /// Enable per-pass timing output.
    pub time_passes: Option<bool>,
}

/// Runner subprocess configuration (`[runner]` table in tile.toml).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Path to the `__tile_runner` subprocess binary.
    pub binary: Option<String>,
    /// Extra arguments forwarded verbatim to the runner subprocess.
    pub extra_args: Option<Vec<String>>,
}

// ── TileConfig ─────────────────────────────────────────────────────────────

/// Top-level configuration for the `tile` CLI.
///
/// Use the `effective_*()` accessors rather than reading fields directly —
/// they apply the sub-table-over-flat precedence rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileConfig {
    // ── Flat (backward-compat top-level fields) ────────────────────────
    /// Path to the `__tile_runner` binary.  **Deprecated:** use `[runner] binary`.
    pub runner_binary: String,
    /// Optional path to the MetalTile project root.
    pub project_path: Option<String>,
    /// Verbosity level: 0 = quiet, 1 = profile columns, 2 = timing columns.
    pub verbose: u8,
    /// Bench iterations.  **Deprecated:** use `[bench] runs`.
    pub runs: usize,
    /// Bench warmup dispatches.  **Deprecated:** use `[bench] warmup_runs`.
    pub warmup_runs: usize,

    // ── Sub-tables ─────────────────────────────────────────────────────
    #[serde(default)]
    pub bench: BenchConfig,
    #[serde(default)]
    pub build: BuildConfig,
    #[serde(default)]
    pub runner: RunnerConfig,

    // ── Internal (not from TOML) ───────────────────────────────────────
    /// Warnings collected during config load (unknown keys, deprecated fields,
    /// etc.). Printed at the end of each command when non-empty.
    #[serde(skip)]
    pub warnings: Vec<String>,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            runner_binary: "__tile_runner".to_string(),
            project_path: None,
            verbose: 0,
            runs: 3,
            warmup_runs: 1,
            bench: BenchConfig::default(),
            build: BuildConfig::default(),
            runner: RunnerConfig::default(),
            warnings: Vec::new(),
        }
    }
}

impl TileConfig {
    /// Effective bench iterations: `[bench] runs` overrides flat `runs`.
    pub fn effective_runs(&self) -> usize { self.bench.runs.unwrap_or(self.runs) }

    /// Effective warmup count: `[bench] warmup_runs` overrides flat `warmup_runs`.
    pub fn effective_warmup_runs(&self) -> usize {
        self.bench.warmup_runs.unwrap_or(self.warmup_runs)
    }

    /// Effective runner binary: `[runner] binary` overrides flat `runner_binary`.
    pub fn effective_runner_binary(&self) -> &str {
        self.runner.binary.as_deref().unwrap_or(&self.runner_binary)
    }

    /// xcrun SDK (default: `"macosx"`).
    pub fn effective_sdk(&self) -> &str { self.build.sdk.as_deref().unwrap_or("macosx") }

    /// Default dtypes (default: `["f32", "f16", "bf16"]`).
    pub fn effective_default_dtypes(&self) -> Vec<String> {
        self.build
            .default_dtypes
            .clone()
            .unwrap_or_else(|| vec!["f32".to_string(), "f16".to_string(), "bf16".to_string()])
    }

    /// Extra runner args from `[runner] extra_args` (default: empty).
    pub fn effective_extra_runner_args(&self) -> Vec<String> {
        self.runner.extra_args.clone().unwrap_or_default()
    }

    /// Serialize the effective config as pretty-printed TOML (`tile config`).
    pub fn to_string_pretty(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

// ── ConfigLoader ───────────────────────────────────────────────────────────

/// Loads [`TileConfig`] from the layered sources described in the module doc.
pub struct ConfigLoader;

impl ConfigLoader {
    /// Walk parent directories from CWD looking for `tile.toml`.
    ///
    /// Returns the first path found, or `None` if the file doesn't exist
    /// anywhere in the ancestor chain (stops at filesystem root).
    pub fn find_tile_toml() -> Option<PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let mut dir: &Path = cwd.as_path();
        loop {
            let candidate = dir.join("tile.toml");
            if candidate.exists() {
                return Some(candidate);
            }
            dir = dir.parent()?;
        }
    }

    /// Build and extract the merged configuration using the default profile.
    ///
    /// Returns `Err` only when figment itself fails (malformed TOML, type
    /// mismatch in an env var, etc.).  A missing `tile.toml` is silently
    /// ignored.
    pub fn load() -> Result<TileConfig, Box<figment::Error>> { Self::load_with_profile(None) }

    /// Build and extract the merged configuration, selecting an optional named
    /// profile.
    ///
    /// Profile resolution:
    ///   1. `profile` argument (from `--profile` CLI flag)
    ///   2. `TILE_PROFILE` environment variable
    ///   3. No profile (flat config only)
    pub fn load_with_profile(profile: Option<&str>) -> Result<TileConfig, Box<figment::Error>> {
        let toml_path = Self::find_tile_toml().unwrap_or_else(|| PathBuf::from("tile.toml"));

        // Determine effective profile (arg > env > none).
        let env_profile = std::env::var("TILE_PROFILE").ok().filter(|s| !s.is_empty());
        let selected = profile.or(env_profile.as_deref());

        // Resolve `extends = "..."` base file.
        let extends_base = Self::resolve_extends(&toml_path);

        let mut figment = Figment::from(Serialized::defaults(TileConfig::default()));

        // Layer 1: extends base file (lowest priority above defaults).
        if let Some(base) = &extends_base {
            figment = figment.merge(Toml::file(base));
        }

        // Layer 2: tile.toml flat keys.
        figment = figment.merge(Toml::file(&toml_path));

        // Layer 3: profile-specific overrides from `[profiles.<name>]`.
        if let Some(p) = selected
            && let Some(profile_toml) = Self::extract_profile_toml(&toml_path, p)
        {
            figment = figment.merge(Toml::string(&profile_toml));
        }

        // Layer 4: TILE_* env vars.
        figment = figment.merge(Env::prefixed("TILE_"));

        let mut config: TileConfig = figment.extract().map_err(Box::new)?;

        // Post-load passes.
        interpolate_config(&mut config);
        config.warnings = collect_warnings(&toml_path);

        Ok(config)
    }

    /// Look for `extends = "path"` at the top level of a tile.toml and
    /// return the resolved `PathBuf` if the file exists.
    fn resolve_extends(toml_path: &Path) -> Option<PathBuf> {
        let content = std::fs::read_to_string(toml_path).ok()?;
        let value: toml::Value = toml::from_str(&content).ok()?;
        let raw = value.get("extends")?.as_str()?;
        let expanded = shell_tilde_expand(raw);
        let base = if Path::new(&expanded).is_absolute() {
            PathBuf::from(expanded)
        } else {
            toml_path.parent().unwrap_or(Path::new(".")).join(expanded)
        };
        base.exists().then_some(base)
    }

    /// Extract a `[profiles.<name>]` section from a tile.toml and return it
    /// as a serialized TOML string so it can be fed to figment as a flat
    /// override layer (sub-table keys like `[profiles.ci.bench]` are
    /// serialized back to `[bench]` since the nesting is stripped).
    fn extract_profile_toml(toml_path: &Path, profile: &str) -> Option<String> {
        let content = std::fs::read_to_string(toml_path).ok()?;
        let value: toml::Value = toml::from_str(&content).ok()?;
        let section = value.get("profiles")?.get(profile)?;
        toml::to_string(section).ok()
    }
}

// ── Env-var interpolation ─────────────────────────────────────────────────

/// Expand `${VAR}` and `${VAR:-default}` placeholders in `s`.
///
/// - `${HOME}` → value of `$HOME`
/// - `${MISSING:-fallback}` → `"fallback"` when `$MISSING` is unset
/// - `${MISSING}` → `""` when `$MISSING` is unset
fn interpolate_str(s: &str) -> String {
    let re = regex::Regex::new(r"\$\{([^}:]+)(?::-([^}]*))?\}").expect("static regex");
    re.replace_all(s, |caps: &regex::Captures<'_>| {
        let var_name = &caps[1];
        let default = caps.get(2).map(|m| m.as_str());
        std::env::var(var_name).ok().or_else(|| default.map(str::to_owned)).unwrap_or_default()
    })
    .into_owned()
}

/// Expand a leading `~/` to `$HOME/` (shell-style tilde expansion).
fn shell_tilde_expand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    s.to_owned()
}

fn interpolate_config(cfg: &mut TileConfig) {
    cfg.runner_binary = interpolate_str(&cfg.runner_binary);
    if let Some(p) = &cfg.project_path.clone() {
        cfg.project_path = Some(interpolate_str(p));
    }
    if let Some(b) = &cfg.runner.binary.clone() {
        cfg.runner.binary = Some(interpolate_str(b));
    }
    if let Some(sdk) = &cfg.build.sdk.clone() {
        cfg.build.sdk = Some(interpolate_str(sdk));
    }
}

// ── Unknown-key warnings ───────────────────────────────────────────────────

const KNOWN_KEYS: &[&str] = &[
    "runner_binary",
    "project_path",
    "verbose",
    "runs",
    "warmup_runs",
    "bench",
    "build",
    "runner",
    "profiles",
    "extends",
];

fn collect_warnings(toml_path: &Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(toml_path) else {
        return Vec::new();
    };
    let Ok(value) = toml::from_str::<toml::Value>(&content) else {
        return Vec::new();
    };
    let Some(table) = value.as_table() else {
        return Vec::new();
    };

    let mut warnings = Vec::new();

    for key in table.keys() {
        if !KNOWN_KEYS.contains(&key.as_str()) {
            warnings.push(format!("unknown config key `{key}` in tile.toml"));
        }
    }

    // Warn when both flat deprecated and sub-table forms coexist.
    let has_bench = table.contains_key("bench");
    let has_runner = table.contains_key("runner");
    if has_bench && (table.contains_key("runs") || table.contains_key("warmup_runs")) {
        warnings.push(
            "tile.toml: top-level `runs`/`warmup_runs` are ignored when `[bench]` is present; \
             remove the top-level fields"
                .into(),
        );
    }
    if has_runner && table.contains_key("runner_binary") {
        warnings.push(
            "tile.toml: top-level `runner_binary` is ignored when `[runner]` is present; \
             remove the top-level field"
                .into(),
        );
    }

    warnings
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use figment::providers::Serialized;

    use super::*;

    #[test]
    fn defaults_are_sensible() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.runner_binary, "__tile_runner");
        assert_eq!(cfg.verbose, 0);
        assert!(cfg.project_path.is_none());
        assert_eq!(cfg.runs, 3);
        assert_eq!(cfg.warmup_runs, 1);
        assert_eq!(cfg.effective_runs(), 3);
        assert_eq!(cfg.effective_warmup_runs(), 1);
        assert_eq!(cfg.effective_runner_binary(), "__tile_runner");
        assert_eq!(cfg.effective_sdk(), "macosx");
    }

    #[test]
    fn sub_table_bench_overrides_flat() {
        let mut cfg = TileConfig::default();
        cfg.bench.runs = Some(10);
        cfg.bench.warmup_runs = Some(5);
        assert_eq!(cfg.effective_runs(), 10);
        assert_eq!(cfg.effective_warmup_runs(), 5);
        // Flat fields unchanged.
        assert_eq!(cfg.runs, 3);
    }

    #[test]
    fn sub_table_runner_overrides_flat() {
        let mut cfg = TileConfig::default();
        cfg.runner.binary = Some("/usr/local/bin/custom_runner".into());
        assert_eq!(cfg.effective_runner_binary(), "/usr/local/bin/custom_runner");
        assert_eq!(cfg.runner_binary, "__tile_runner"); // flat unchanged
    }

    #[test]
    fn interpolate_str_replaces_env_var() {
        // SAFETY: single-threaded test, no concurrent reads.
        unsafe { std::env::set_var("_TILE_TEST_INTERP", "hello") };
        assert_eq!(interpolate_str("${_TILE_TEST_INTERP}/world"), "hello/world");
        unsafe { std::env::remove_var("_TILE_TEST_INTERP") };
    }

    #[test]
    fn interpolate_str_uses_default_when_unset() {
        unsafe { std::env::remove_var("_TILE_DEFINITELY_MISSING") };
        assert_eq!(interpolate_str("${_TILE_DEFINITELY_MISSING:-fallback}"), "fallback");
    }

    #[test]
    fn interpolate_str_empty_when_unset_no_default() {
        unsafe { std::env::remove_var("_TILE_DEFINITELY_MISSING") };
        assert_eq!(interpolate_str("before_${_TILE_DEFINITELY_MISSING}_after"), "before__after");
    }

    #[test]
    fn shell_tilde_expand_uses_home() {
        if let Ok(home) = std::env::var("HOME") {
            assert_eq!(shell_tilde_expand("~/.config"), format!("{home}/.config"));
        }
    }

    #[test]
    fn load_returns_defaults_without_tile_toml() {
        let cfg: TileConfig = Figment::from(Serialized::defaults(TileConfig::default()))
            .merge(Toml::file("/nonexistent/tile.toml"))
            .extract()
            .expect("should succeed with all-defaults");
        assert_eq!(cfg.runner_binary, "__tile_runner");
        assert_eq!(cfg.verbose, 0);
        assert_eq!(cfg.runs, 3);
        assert_eq!(cfg.warmup_runs, 1);
    }

    #[test]
    fn find_tile_toml_returns_some_from_workspace_root() {
        let found = ConfigLoader::find_tile_toml();
        if let Some(p) = found {
            assert!(p.ends_with("tile.toml"), "unexpected path: {p:?}");
        }
    }

    #[test]
    fn to_string_pretty_is_valid_toml() {
        let cfg = TileConfig::default();
        let s = cfg.to_string_pretty().expect("serialize");
        let _: toml::Value = toml::from_str(&s).expect("round-trip");
    }

    #[test]
    fn collect_warnings_empty_for_known_keys() {
        // Build a valid tile.toml string in a temp dir and verify no warnings.
        let dir = std::env::temp_dir().join("tile_config_test_known");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tile.toml");
        std::fs::write(&path, "verbose = 1\nruns = 5\n").unwrap();
        let ws = collect_warnings(&path);
        assert!(ws.is_empty(), "unexpected warnings: {ws:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn collect_warnings_flags_unknown_key() {
        let dir = std::env::temp_dir().join("tile_config_test_unknown");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tile.toml");
        std::fs::write(&path, "unknown_key_xyz = true\n").unwrap();
        let ws = collect_warnings(&path);
        assert!(ws.iter().any(|w| w.contains("unknown_key_xyz")), "warning missing: {ws:?}");
        let _ = std::fs::remove_file(&path);
    }
}

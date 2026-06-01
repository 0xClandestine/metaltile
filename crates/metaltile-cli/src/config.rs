//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `TileConfig` — layered configuration for the `tile` CLI.
//!
//! Load order (later layers override earlier ones):
//!   1. Built-in defaults (`Serialized::defaults`)
//!   2. `tile.toml` in the current directory (optional, ignored if absent)
//!   3. `TILE_*` environment variables
//!      (e.g. `TILE_VERBOSE=1`, `TILE_RUNNER_BINARY=/usr/local/bin/__tile_runner`)
//!
//! Note: only a single underscore prefix is used (`TILE_`), not double (`TILE__`).

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};

/// Top-level configuration for the `tile` CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileConfig {
    /// Path to the `__tile_runner` subprocess binary.
    /// Defaults to `__tile_runner` (resolved via `$PATH`).
    pub runner_binary: String,

    /// Optional path to the MetalTile project root.
    /// When set, `tile bench/test/build` look for source here.
    pub project_path: Option<String>,

    /// Verbosity level: 0 = quiet, 1 = profile columns (`-v`), 2 = timing columns (`-vv`).
    pub verbose: u8,

    /// Number of timed benchmark iterations per kernel (after warmup).
    pub runs: usize,

    /// Number of warmup dispatches before timing begins.
    pub warmup_runs: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            runner_binary: "__tile_runner".to_string(),
            project_path: None,
            verbose: 0,
            runs: 3,
            warmup_runs: 1,
        }
    }
}

/// Loads [`TileConfig`] from the layered sources described in the module doc.
pub struct ConfigLoader;

impl ConfigLoader {
    /// Build and extract the merged configuration.
    ///
    /// Returns `Err` only when figment itself fails (malformed TOML, type
    /// mismatch in an env var, etc.).  A missing `tile.toml` is silently
    /// ignored.
    pub fn load() -> Result<TileConfig, Box<figment::Error>> {
        Figment::from(Serialized::defaults(TileConfig::default()))
            .merge(Toml::file("tile.toml"))
            .merge(Env::prefixed("TILE_"))
            .extract()
            .map_err(Box::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sensible() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.runner_binary, "__tile_runner");
        assert_eq!(cfg.verbose, 0);
        assert!(cfg.project_path.is_none());
        assert_eq!(cfg.runs, 3);
        assert_eq!(cfg.warmup_runs, 1);
    }

    #[test]
    fn load_returns_defaults_without_tile_toml() {
        // Build the figment with an explicitly-absent path so this test is
        // CWD-independent and safe to run from the workspace root (which has
        // a real tile.toml that would otherwise override defaults).
        let cfg: TileConfig = Figment::from(Serialized::defaults(TileConfig::default()))
            .merge(Toml::file("/nonexistent/tile.toml"))
            .extract()
            .expect("should succeed with all-defaults");
        assert_eq!(cfg.runner_binary, "__tile_runner");
        assert_eq!(cfg.verbose, 0);
        assert_eq!(cfg.runs, 3);
        assert_eq!(cfg.warmup_runs, 1);
    }
}

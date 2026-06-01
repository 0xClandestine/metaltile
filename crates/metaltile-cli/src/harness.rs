//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `Harness` — owns the `TileConfig` and provides the entry point for
//! dispatching CLI subcommand work.
//!
//! Phase 1 (current): routes directly to the in-process runner inside
//! `metaltile::runner`.  Phase 2 will replace the in-process call with a
//! `ProjectRunner` that spawns `__tile_runner` as a subprocess and parses
//! the JSON Lines it emits.

use crate::config::TileConfig;

/// Top-level harness for the `tile` CLI.  Constructed once in `main` and
/// passed down to each subcommand handler.
pub struct Harness {
    pub config: TileConfig,
}

impl Harness {
    /// Create a `Harness` from the layered config (defaults → `tile.toml` →
    /// `TILE__*` env vars).  Logs a warning and falls back to defaults when
    /// config loading fails.
    pub fn from_config() -> Self {
        let config = crate::config::ConfigLoader::load().unwrap_or_else(|e| {
            tracing::warn!("tile.toml config error: {e}; using defaults");
            TileConfig::default()
        });
        Self { config }
    }

    /// Return the path to the `__tile_runner` binary as configured.
    pub fn runner_binary(&self) -> &str { &self.config.runner_binary }
}

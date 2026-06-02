//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `Harness` — owns the `TileConfig` and global CLI flags; passed to every
//! subcommand handler so they share a single consistent view of configuration.

use crate::{GlobalArgs, config::TileConfig};

/// Top-level harness for the `tile` CLI.  Constructed once in `main` and
/// passed down to each subcommand handler.
pub struct Harness {
    pub config: TileConfig,
    /// Global flags parsed from the CLI (`--quiet`, `--json`, `--color`, `-j`, `-v`).
    pub global: GlobalArgs,
}

impl Harness {
    /// Create a `Harness` from the parsed global flags.
    ///
    /// Config is loaded via `ConfigLoader::load_with_profile`, which handles
    /// parent-dir walk, extends, profiles, env var interpolation, and warning
    /// collection.  On load failure, falls back to defaults and logs a warning.
    pub fn new(global: GlobalArgs) -> Self {
        let profile = global.profile.as_deref();
        let config = crate::config::ConfigLoader::load_with_profile(profile).unwrap_or_else(|e| {
            tracing::warn!("tile.toml config error: {e}; using defaults");
            TileConfig::default()
        });
        Self { config, global }
    }

    /// Path to the `__tile_runner` binary (sub-table overrides flat field).
    pub fn runner_binary(&self) -> &str { self.config.effective_runner_binary() }

    /// True when `--quiet` / `-q` is set.
    pub fn is_quiet(&self) -> bool { self.global.quiet }

    /// True when the global `--json` flag is set.
    pub fn json_output(&self) -> bool { self.global.json }

    /// Effective verbosity level: global `-v` count wins over `tile.toml verbose`.
    pub fn verbosity(&self) -> u8 {
        if self.global.verbose > 0 { self.global.verbose } else { self.config.verbose }
    }

    /// Print config-load warnings to stderr (called at end of each command).
    pub fn print_warnings(&self) {
        for w in &self.config.warnings {
            eprintln!("tile: warning: {w}");
        }
    }
}

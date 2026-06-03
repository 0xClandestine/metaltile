//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `ProjectRunner` — spawns `__tile_runner` as a subprocess and streams its
//! `ProtocolMessage` JSON-lines output to stdout.
//!
//! ```text
//!  tile CLI  ──spawn──►  __tile_runner  (metaltile-std linked → inventory populated)
//!             JSON Lines ◄── stdout
//! ```
//!
//! `__tile_runner` is a hidden binary scaffolded by `tile init` into the
//! project's `bin/` directory and installed there via
//! `cargo install --path . --root .`.

use crate::harness::Harness;

/// Describes one invocation of `__tile_runner`.
#[derive(Debug, Clone, Default)]
pub struct RunnerInvocation {
    /// Subcommand: "bench", "test", "build", or "inspect".
    pub command: String,
    /// Optional name filter (passed as `--filter`).
    pub filter: Option<String>,
    /// Optional dtype filter (passed as `--dtype`).
    pub dtype: Option<String>,
    /// Optional inspect kind (passed as `--kind`).
    pub inspect_kind: Option<String>,
    /// Enable profiling (passed as `--profile`).
    pub profile: bool,
    /// Override warmup dispatch count (passed as `--warmup-runs`).
    pub warmup_runs: Option<usize>,
    /// Override timed iteration count (passed as `--runs`).
    pub runs: Option<usize>,
}

/// Owns the `Harness` reference and exposes the subprocess dispatch method.
pub struct ProjectRunner<'a> {
    pub harness: &'a Harness,
}

impl<'a> ProjectRunner<'a> {
    pub fn new(harness: &'a Harness) -> Self { Self { harness } }

    /// Spawn `__tile_runner` as a subprocess with stdout/stderr inherited,
    /// so its `ProtocolMessage` JSON lines flow directly to the terminal.
    /// Returns `true` when the runner exits with status 0.
    pub fn run(&self, inv: &RunnerInvocation) -> bool {
        let binary = self.runner_binary();
        let argv = build_argv(inv);
        match std::process::Command::new(&binary).args(&argv).status() {
            Ok(s) => s.success(),
            Err(e) => {
                eprintln!("[tile] failed to spawn '{binary}': {e}");
                false
            },
        }
    }

    /// Resolve the `__tile_runner` binary path.
    ///
    /// Precedence:
    /// 1. `tile.toml [runner] binary` config override (default: `bin/__tile_runner`).
    /// 2. `./bin/__tile_runner` relative to CWD — the standard install location
    ///    populated by `cargo install --path . --root .` in a tile project.
    /// 3. Sibling of the current `tile` executable — works when both binaries
    ///    are installed to the same prefix (e.g. `~/.cargo/bin/`).
    /// 4. Bare `"__tile_runner"` resolved via `$PATH` as a last resort.
    fn runner_binary(&self) -> String {
        let configured = self.harness.runner_binary();
        if configured != "__tile_runner" {
            return configured.to_string();
        }
        let cwd = std::env::current_dir().unwrap_or_default();
        let local = cwd.join("bin").join("__tile_runner");
        if local.exists() {
            return local.to_string_lossy().into_owned();
        }
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            let sibling = dir.join("__tile_runner");
            if sibling.exists() {
                return sibling.to_string_lossy().into_owned();
            }
        }
        "__tile_runner".to_string()
    }
}

fn build_argv(inv: &RunnerInvocation) -> Vec<String> {
    let mut argv = vec![inv.command.clone()];
    if let Some(f) = &inv.filter {
        argv.push("--filter".to_string());
        argv.push(f.clone());
    }
    if let Some(d) = &inv.dtype {
        argv.push("--dtype".to_string());
        argv.push(d.clone());
    }
    if let Some(k) = &inv.inspect_kind {
        argv.push("--kind".to_string());
        argv.push(k.clone());
    }
    if inv.profile {
        argv.push("--profile".to_string());
    }
    if let Some(w) = inv.warmup_runs {
        argv.push("--warmup-runs".to_string());
        argv.push(w.to_string());
    }
    if let Some(r) = inv.runs {
        argv.push("--runs".to_string());
        argv.push(r.to_string());
    }
    argv
}

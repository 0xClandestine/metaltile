//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `ProjectRunner` — will spawn `__tile_runner` as a subprocess and stream
//! `ProtocolMessage` JSON lines back to the caller.
//!
//! # Architecture (Phase 2 target)
//!
//! ```text
//!  tile CLI  ──spawn──►  __tile_runner
//!             JSON Lines ◄── stdout
//! ```
//!
//! Each subcommand builds a `RunnerInvocation` describing the command,
//! filter, dtype, etc., then calls `ProjectRunner::run(invocation, callback)`.
//! The runner process writes one JSON-encoded `ProtocolMessage` per line; this
//! module parses each line and calls the callback in order.
//!
//! # Phase 1 (current)
//!
//! `run` delegates in-process to `metaltile::runner::RunnerHarness`, which
//! writes JSON lines directly to stdout.  The `on_message` callback is **not**
//! invoked in the Phase-1 path — callers that depend on it should wait for
//! Phase 2 subprocess wiring (Step 7 → Step 9 migration).

use metaltile::runner::{RunnerArgs, RunnerHarness};

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

/// Owns the `Harness` reference and exposes the subprocess-or-in-process
/// dispatch method.
pub struct ProjectRunner<'a> {
    pub harness: &'a Harness,
}

impl<'a> ProjectRunner<'a> {
    pub fn new(harness: &'a Harness) -> Self { Self { harness } }

    /// Run the invocation in-process (Phase 1).
    ///
    /// Emits `ProtocolMessage` JSON lines directly to stdout via
    /// `RunnerHarness::run`.  Returns `true` when the runner signals success.
    ///
    /// Phase 2 will replace this with a subprocess spawn that captures stdout
    /// and forwards parsed `ProtocolMessage`s to a caller-supplied callback.
    pub fn run(&self, inv: &RunnerInvocation) -> bool {
        let args = match RunnerArgs::parse(build_argv(inv)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[tile] runner args error: {e}");
                return false;
            },
        };
        RunnerHarness::run(&args)
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

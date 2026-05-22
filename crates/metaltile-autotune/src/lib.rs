//! Autotune engine — search, measurement, cost modeling, and training
//! data export for MetalTile kernels.
//!
//! Splits out of `metaltile-cli` so the engine is independently
//! testable and reusable from non-CLI surfaces (future TUI, Python
//! binding, batch driver). The CLI is a thin controller on top:
//! parses args, owns the `GpuRunner` lifecycle, renders progress.
//!
//! Public entry point: [`run_autotune`].
//!
//! Quick start:
//!
//! ```no_run
//! use metaltile_autotune::{AutotuneOptions, run_autotune};
//! let opts = AutotuneOptions { measure: false, quick: false, filter: None, shape_overrides: None };
//! let summary = run_autotune(&opts, /* runner= */ None, |result| {
//!     // render `result` however you like
//!     let _ = result;
//! })
//! .expect("autotune");
//! let _ = summary.tuned;
//! ```

use std::path::PathBuf;

use thiserror::Error;

mod budget;
mod cost;
mod engine;
mod export;
mod measurer;
mod search;
mod util;

pub use budget::TuneOutcome;
pub use engine::{KernelOk, run_autotune};
pub use export::{
    collect_training_rows, default_training_data_path, write_training_jsonl,
    write_training_jsonl_to_file,
};

// Re-export so callers don't need a transitive metaltile dep just to
// match on TrainingRow fields.
pub use metaltile::autotune::TrainingRow;

/// Knobs the CLI translates from `AutotuneArgs`.
///
/// - `measure`: real GPU timing (else static cost).
/// - `quick`: 3 warmup + 11 iters per candidate (else 20 + 100). No-op without `measure`.
/// - `filter`: case-insensitive substring against `kernel_name`.
/// - `shape_overrides`: when `Some`, tune each `N` independently and
///   write distinct cache entries per bucket. When `None`, fall back
///   to `spec.shapes[0]` (legacy behavior).
#[derive(Debug, Clone, Default)]
pub struct AutotuneOptions {
    pub measure: bool,
    pub quick: bool,
    pub filter: Option<String>,
    pub shape_overrides: Option<Vec<usize>>,
}

/// Streamed per-(kernel, dtype, n_override) result.
///
/// On success, `outcome` is `Ok(KernelOk)`. On failure (`Err(msg)`),
/// the kernel counts toward `skipped` in the summary; the CLI renders
/// a `skip` line.
#[derive(Debug, Clone)]
pub struct KernelTuneResult {
    pub kernel_name: &'static str,
    pub dtype_label: &'static str,
    pub n_override: Option<usize>,
    pub outcome: Result<KernelOk, String>,
}

/// Final tally + cache state.
#[derive(Debug, Clone)]
pub struct AutotuneSummary {
    pub tuned: usize,
    pub measured: usize,
    pub estimated: usize,
    pub skipped: usize,
    pub fallbacks: usize,
    pub cache_entries: usize,
    pub cache_dir: PathBuf,
}

/// Library error type. Maps cleanly onto CLI errors via thin From impls.
#[derive(Debug, Error)]
pub enum AutotuneError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    MetalTile(#[from] metaltile::MetalTileError),

    #[error("{0}")]
    Other(String),
}

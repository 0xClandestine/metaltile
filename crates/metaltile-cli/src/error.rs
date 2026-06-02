//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
use thiserror::Error;

/// Canonical exit codes for the `tile` binary.
///
/// Mirrors the pattern from `foundry_cli::ExitCode` so CI pipelines can
/// distinguish test failures from build failures from regressions.
#[repr(i32)]
pub enum TileExitCode {
    /// All kernels passed / command succeeded.
    Success = 0,
    /// One or more `#[test_kernel]` checks failed.
    TestFailure = 1,
    /// Compilation or build step failed.
    BuildFailure = 2,
    /// `tile diff` detected a performance regression beyond the threshold.
    Regression = 3,
    /// `tile.toml` parsing or configuration error.
    ConfigError = 10,
}

#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("metal compile failed: {0}")]
    MetalCompile(String),

    #[error("GPU runner initialization failed: {0}")]
    GpuInit(String),

    #[error("subprocess failed: {0}")]
    Subprocess(String),

    #[error("one or more tests failed")]
    TestFailure,

    #[error("build failed")]
    BuildFailure,

    #[error("performance regression detected")]
    Regression,

    #[error("{0}")]
    Other(String),
}

impl CliError {
    /// Map this error to its canonical process exit code.
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::TestFailure => TileExitCode::TestFailure as i32,
            CliError::BuildFailure | CliError::MetalCompile(_) => TileExitCode::BuildFailure as i32,
            CliError::Regression => TileExitCode::Regression as i32,
            _ => 1,
        }
    }
}

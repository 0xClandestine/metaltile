//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Typed CLI errors — no catch-all `Other` variant.
//!
//! Every error variant carries semantic meaning so callers (and the
//! top-level dispatch) can handle different failure modes differently.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    /// I/O errors from filesystem operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialisation / deserialisation errors.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Metal shader compilation (xcrun metal) failed.
    #[error("metal compile failed: {0}")]
    MetalCompile(String),

    /// GPU runner initialisation failed.
    #[error("GPU runner initialization failed: {0}")]
    GpuInit(String),

    #[error("GPU device error: {0}")]
    GpuDevice(String),

    /// Subprocess spawn / wait failure.
    #[error("subprocess failed: {0}")]
    Subprocess(String),

    /// Bench correctness check failures.
    #[error("correctness: {0} correctness check(s) failed")]
    Correctness(u32),

    /// Kernel compilation errors during `tile build`.
    #[error("build: {0} kernel(s) failed to compile")]
    BuildFailed(u32),

    /// Test failures.
    #[error("test: {0} test(s) failed")]
    TestFailed(u32),

    /// Regression(s) detected in `tile diff.
    #[error("diff: {0} regression(s) detected")]
    Regression(usize),

    /// Config / project-setup errors.
    #[error("config: {0}")]
    Config(String),

    /// Harness protocol errors (JSONL parsing).
    #[error("harness protocol: {0}")]
    HarnessProtocol(String),

    /// Git operations failed.
    #[error("git: {0}")]
    Git(String),

    /// Template / scaffolding errors.
    #[error("scaffold: {0}")]
    Scaffold(String),
}

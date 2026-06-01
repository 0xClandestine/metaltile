//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! JSON Lines protocol types for the MetalTile runner ↔ CLI communication.
//!
//! The `__tile_runner` subprocess writes newline-delimited JSON to stdout.
//! The `tile` CLI reads this stream and renders it. This module defines the
//! full set of serialisable message types that form the wire contract.
//!
//! # Wire format
//!
//! Every message is a single JSON object on one line, tagged by `"type"`:
//!
//! ```text
//! {"type":"start","runner_version":"0.1","command":"bench","total":42}
//! {"type":"bench","op":"unary/exp","dtype":"f32","mt_gbps":1234.5,...}
//! {"type":"done","ok":true,"bench_passed":42,"bench_failed":0,...}
//! ```
//!
//! # Protocol versioning
//!
//! The protocol is versioned via the `runner_version` field in the
//! [`ProtocolMessage::Start`] message. The CLI can gracefully degrade for
//! older runners that do not emit newer variants.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// The kind of compiled artifact produced by `tile build`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactKind {
    /// Metal Shading Language source (`.metal`).
    Msl,
    /// Compiled Metal library (`.metallib`).
    Metallib,
    /// Swift wrapper source.
    Swift,
    /// MetalTile IR dump.
    Ir,
}

/// The kind of content produced by `tile inspect`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InspectKind {
    /// Metal Shading Language source.
    Msl,
    /// MetalTile IR dump.
    Ir,
    /// Kernel statistics (register count, occupancy estimate, etc.).
    Stats,
    /// Annotated instruction listing.
    Listing,
}

/// A compile error for one dtype inside a `tile build` run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildError {
    /// The dtype that failed to compile (e.g. `"f16"`).
    pub dtype: String,
    /// Human-readable error message from the Metal compiler.
    pub message: String,
}

/// GPU occupancy and performance profile for a single bench run.
///
/// Populated when the runner is invoked with profiling enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInfo {
    /// Achieved occupancy as a percentage of theoretical maximum.
    pub occ_pct: f32,
    /// Registers allocated per thread.
    pub regs_per_thread: u32,
    /// Human-readable description of the primary performance bottleneck.
    pub bottleneck: String,
}

// ---------------------------------------------------------------------------
// ProtocolMessage
// ---------------------------------------------------------------------------

/// A single message in the runner ↔ CLI protocol (newline-delimited JSON).
///
/// The runner emits one `Start` at the beginning, a stream of per-item
/// events as work completes, and one `Done` as the final line.
/// The CLI parses each line and renders it without any GPU code.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProtocolMessage {
    // ── Lifecycle ────────────────────────────────────────────────────────────
    /// Emitted once as the very first line of a run.
    #[serde(rename = "start")]
    Start {
        /// Runner crate version string (e.g. `"0.1.0"`).
        runner_version: String,
        /// The subcommand being run (`"bench"`, `"test"`, `"build"`, `"inspect"`).
        command: String,
        /// Total number of items to be processed in this run.
        total: u32,
    },

    /// Emitted once as the very last line of a run.
    #[serde(rename = "done")]
    Done {
        /// Whether the run completed without any errors or failures.
        ok: bool,
        /// Number of bench items that passed correctness checks.
        bench_passed: u32,
        /// Number of bench items that failed correctness checks.
        bench_failed: u32,
        /// Number of test cases that passed.
        test_passed: u32,
        /// Number of test cases that failed.
        test_failed: u32,
    },

    // ── Per-item results ─────────────────────────────────────────────────────
    /// Result of a single benchmark (one kernel × one dtype).
    #[serde(rename = "bench")]
    BenchResult(BenchResult),

    /// Result of a single correctness test (one kernel × one dtype).
    #[serde(rename = "test")]
    TestResult(TestResult),

    /// Result of compiling a single kernel across all requested dtypes.
    #[serde(rename = "build")]
    BuildResult(BuildResult),

    /// A compiled artifact path emitted by `tile build --emit`.
    #[serde(rename = "artifact")]
    Artifact {
        /// The kind of artifact written to disk.
        kind: ArtifactKind,
        /// Absolute path to the written file.
        path: String,
    },

    /// Content produced by `tile inspect` for a single kernel.
    #[serde(rename = "inspect")]
    Inspect {
        /// Kernel name (e.g. `"unary/exp"`).
        name: String,
        /// What kind of content this is.
        kind: InspectKind,
        /// The content itself (MSL source, IR dump, stats text, etc.).
        content: String,
    },

    // ── Errors ───────────────────────────────────────────────────────────────
    /// A non-fatal error for one kernel/dtype combination.
    ///
    /// The run continues after this message; fatal errors exit immediately.
    #[serde(rename = "error")]
    ProtocolError {
        /// Kernel or bench name.
        name: String,
        /// Data type being processed when the error occurred (e.g. `"f16"`).
        dtype: String,
        /// Human-readable error message.
        message: String,
    },
}

// ---------------------------------------------------------------------------
// Per-item result structs
// ---------------------------------------------------------------------------

/// Result of a single benchmark (one kernel × one dtype).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Kernel/bench name (e.g. `"unary/exp"`).
    pub name: String,
    /// Data type (e.g. `"f16"`, `"f32"`).
    pub dtype: String,
    /// Throughput in GB/s for the MetalTile kernel.
    #[serde(default)]
    pub mt_gbps: f64,
    /// Throughput in GB/s for the reference kernel, if one was configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_gbps: Option<f64>,
    /// MetalTile speed relative to reference (%), if reference exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mt_pct: Option<f64>,
    /// Whether the kernel produced correct results.
    #[serde(default)]
    pub correct: bool,
    /// Minimum recorded latency in microseconds.
    #[serde(default)]
    pub min_us: f64,
    /// Mean latency in microseconds.
    #[serde(default)]
    pub mean_us: f64,
    /// GPU profiling data, present only when profiling was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProfileInfo>,
}

/// Result of a single correctness test (one kernel × one dtype).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Kernel/test name (e.g. `"unary/exp"`).
    pub name: String,
    /// Data type (e.g. `"f16"`, `"f32"`).
    pub dtype: String,
    /// Whether the test passed within tolerance.
    #[serde(default)]
    pub passed: bool,
    /// Maximum element-wise absolute error observed.
    #[serde(default)]
    pub max_err: f64,
}

/// Result of compiling a single kernel across all requested dtypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildResult {
    /// Kernel name (e.g. `"unary/exp"`).
    pub name: String,
    /// Dtypes that compiled successfully.
    pub dtypes_ok: Vec<String>,
    /// Dtypes that failed to compile, with error messages.
    pub dtypes_err: Vec<BuildError>,
}

// ---------------------------------------------------------------------------
// Serialisation helpers
// ---------------------------------------------------------------------------

impl ProtocolMessage {
    /// Serialise this message as a JSON line (with trailing newline).
    pub fn to_json_line(&self) -> Vec<u8> {
        let mut buf = serde_json::to_vec(self).expect("protocol message serialisation");
        buf.push(b'\n');
        buf
    }

    /// Parse a single JSON line (byte slice, with or without trailing newline).
    pub fn from_json_line(data: &[u8]) -> crate::Result<Self> {
        let trimmed = data.strip_suffix(b"\n").unwrap_or(data);
        serde_json::from_slice(trimmed).map_err(|e| crate::Error::Internal(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_result_roundtrip() {
        let msg = ProtocolMessage::BenchResult(BenchResult {
            name: "unary/exp".into(),
            dtype: "f16".into(),
            mt_gbps: 1234.5,
            ref_gbps: Some(1189.2),
            mt_pct: Some(103.8),
            correct: true,
            min_us: 12.3,
            mean_us: 12.8,
            profile: None,
        });
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::BenchResult(b) => {
                assert_eq!(b.name, "unary/exp");
                assert!((b.mt_gbps - 1234.5).abs() < 0.01);
                assert!((b.ref_gbps.unwrap() - 1189.2).abs() < 0.01);
                assert!(b.profile.is_none());
            },
            _ => panic!("expected BenchResult"),
        }
    }

    #[test]
    fn bench_result_with_profile_roundtrip() {
        let msg = ProtocolMessage::BenchResult(BenchResult {
            name: "unary/exp".into(),
            dtype: "f32".into(),
            mt_gbps: 900.0,
            ref_gbps: None,
            mt_pct: None,
            correct: true,
            min_us: 5.0,
            mean_us: 5.2,
            profile: Some(ProfileInfo {
                occ_pct: 87.5,
                regs_per_thread: 32,
                bottleneck: "memory bandwidth".into(),
            }),
        });
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::BenchResult(b) => {
                let p = b.profile.unwrap();
                assert!((p.occ_pct - 87.5).abs() < 0.01);
                assert_eq!(p.regs_per_thread, 32);
                assert_eq!(p.bottleneck, "memory bandwidth");
            },
            _ => panic!("expected BenchResult"),
        }
    }

    #[test]
    fn test_result_roundtrip() {
        let msg = ProtocolMessage::TestResult(TestResult {
            name: "unary/exp".into(),
            dtype: "f16".into(),
            passed: true,
            max_err: 3.2e-5,
        });
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        assert!(matches!(parsed, ProtocolMessage::TestResult(_)));
    }

    #[test]
    fn build_result_roundtrip() {
        let msg = ProtocolMessage::BuildResult(BuildResult {
            name: "unary/exp".into(),
            dtypes_ok: vec!["f32".into(), "f16".into()],
            dtypes_err: vec![BuildError {
                dtype: "bf16".into(),
                message: "unsupported operation".into(),
            }],
        });
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::BuildResult(b) => {
                assert_eq!(b.name, "unary/exp");
                assert_eq!(b.dtypes_ok.len(), 2);
                assert_eq!(b.dtypes_err.len(), 1);
                assert_eq!(b.dtypes_err[0].dtype, "bf16");
            },
            _ => panic!("expected BuildResult"),
        }
    }

    #[test]
    fn artifact_roundtrip() {
        let msg = ProtocolMessage::Artifact {
            kind: ArtifactKind::Msl,
            path: "/tmp/unary_exp.metal".into(),
        };
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::Artifact { kind, path } => {
                assert_eq!(kind, ArtifactKind::Msl);
                assert_eq!(path, "/tmp/unary_exp.metal");
            },
            _ => panic!("expected Artifact"),
        }
    }

    #[test]
    fn inspect_roundtrip() {
        let msg = ProtocolMessage::Inspect {
            name: "unary/exp".into(),
            kind: InspectKind::Msl,
            content: "kernel void mt_exp(...) {}".into(),
        };
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::Inspect { name, kind, content } => {
                assert_eq!(name, "unary/exp");
                assert_eq!(kind, InspectKind::Msl);
                assert!(content.contains("mt_exp"));
            },
            _ => panic!("expected Inspect"),
        }
    }

    #[test]
    fn start_roundtrip() {
        let msg = ProtocolMessage::Start {
            runner_version: "0.1.0".into(),
            command: "bench".into(),
            total: 42,
        };
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::Start { runner_version, command, total } => {
                assert_eq!(runner_version, "0.1.0");
                assert_eq!(command, "bench");
                assert_eq!(total, 42);
            },
            _ => panic!("expected Start"),
        }
    }

    #[test]
    fn done_roundtrip() {
        let msg = ProtocolMessage::Done {
            ok: true,
            bench_passed: 10,
            bench_failed: 1,
            test_passed: 5,
            test_failed: 0,
        };
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::Done { ok, bench_passed, bench_failed, .. } => {
                assert!(ok);
                assert_eq!(bench_passed, 10);
                assert_eq!(bench_failed, 1);
            },
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn from_json_line_strips_trailing_newline() {
        let msg = ProtocolMessage::Done {
            ok: true,
            bench_passed: 0,
            bench_failed: 0,
            test_passed: 0,
            test_failed: 0,
        };
        let mut json = msg.to_json_line(); // includes trailing \n
        // parse with the newline present (to_json_line adds it)
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        assert!(matches!(parsed, ProtocolMessage::Done { .. }));
        // parse without the newline
        json.pop();
        let parsed2 = ProtocolMessage::from_json_line(&json).unwrap();
        assert!(matches!(parsed2, ProtocolMessage::Done { .. }));
    }
}

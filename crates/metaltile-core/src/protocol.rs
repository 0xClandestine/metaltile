//! JSON Lines protocol types for the MetalTile runner ↔ CLI communication.
//!
//! The tile-runner binary writes newline-delimited JSON to stdout.  The CLI
//! reads this stream and renders it.  This module defines the serialisable
//! message types that form the contract between them.
//!
//! # Protocol versioning
//!
//! The protocol is versioned via the `runner_version` field in the
//! [`ProtocolMessage::Start`] message.  The CLI negotiates and gracefully
//! degrades for older runners.

use serde::{Deserialize, Serialize};

/// A single message in the runner protocol (newline-delimited JSON).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProtocolMessage {
    /// Announced at the beginning of a run.
    #[serde(rename = "start")]
    Start {
        /// Runner protocol version (e.g. "0.1").
        runner_version: String,
        /// Total number of benchmarks in this run.
        total_benches: u32,
        /// Total number of tests in this run.
        total_tests: u32,
    },
    /// Result of a single benchmark iteration.
    #[serde(rename = "bench")]
    BenchResult(BenchResult),
    /// Result of a single correctness test.
    #[serde(rename = "test")]
    TestResult(TestResult),
    /// A non-fatal error during a bench or test.
    #[serde(rename = "error")]
    ProtocolError {
        /// Kernel or bench name.
        name: String,
        /// Data type being tested when the error occurred.
        dtype: String,
        /// Human-readable error message.
        message: String,
    },
    /// Final summary sent at the end of a run.
    #[serde(rename = "done")]
    Done {
        /// Number of passed benchmarks.
        bench_passed: u32,
        /// Number of failed benchmarks.
        bench_failed: u32,
        /// Number of passed tests.
        test_passed: u32,
        /// Number of failed tests.
        test_failed: u32,
    },
}

/// Result of a single benchmark (one kernel × one dtype).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Kernel/bench name (e.g. "unary/exp").
    pub name: String,
    /// Data type (e.g. "f16", "f32").
    pub dtype: String,
    /// Throughput in GB/s for the MetalTile kernel.
    #[serde(default)]
    pub mt_gbps: f64,
    /// Throughput in GB/s for the reference kernel, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_gbps: Option<f64>,
    /// MetalTile speed relative to reference (%), if reference exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mt_pct: Option<f64>,
    /// Whether the kernel produced correct results (compared to reference
    /// or within self-consistency checks).
    #[serde(default)]
    pub correct: bool,
    /// Minimum recorded latency in microseconds.
    #[serde(default)]
    pub min_us: f64,
    /// Mean latency in microseconds.
    #[serde(default)]
    pub mean_us: f64,
}

/// Result of a single correctness test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Kernel/test name (e.g. "unary/exp").
    pub name: String,
    /// Data type (e.g. "f16", "f32").
    pub dtype: String,
    /// Whether the test passed within tolerance.
    #[serde(default)]
    pub passed: bool,
    /// Maximum element-wise error observed.
    #[serde(default)]
    pub max_err: f64,
}

impl ProtocolMessage {
    /// Serialise this message as a JSON line (with trailing newline).
    pub fn to_json_line(&self) -> Vec<u8> {
        let mut buf = serde_json::to_vec(self).expect("protocol message serialisation");
        buf.push(b'\n');
        buf
    }

    /// Parse a JSON line from a byte slice.
    pub fn from_json_line(data: &[u8]) -> crate::Result<Self> {
        serde_json::from_slice(data).map_err(|e| crate::Error::Internal(e.to_string()))
    }
}

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
        });
        let json = msg.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::BenchResult(b) => {
                assert_eq!(b.name, "unary/exp");
                assert!((b.mt_gbps - 1234.5).abs() < 0.01);
                assert!((b.ref_gbps.unwrap() - 1189.2).abs() < 0.01);
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
    fn start_done_roundtrip() {
        let start = ProtocolMessage::Start {
            runner_version: "0.1".into(),
            total_benches: 42,
            total_tests: 10,
        };
        let json = start.to_json_line();
        let parsed = ProtocolMessage::from_json_line(&json).unwrap();
        match parsed {
            ProtocolMessage::Start { ref runner_version, total_benches, .. } => {
                assert_eq!(runner_version, "0.1");
                assert_eq!(total_benches, 42);
            },
            _ => panic!("expected Start"),
        }
    }
}

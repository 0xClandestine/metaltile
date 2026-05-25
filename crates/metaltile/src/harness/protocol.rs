//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! JSONL subprocess protocol types.
//!
//! The harness binary communicates with `tile` CLI via newline-delimited JSON (JSONL)
//! over stdout, one record per line. The protocol is defined in §10 of the spec.
//!
//! ## Protocol
//!
//! ### `--action=bench`
//! ```jsonc
//! {"type":"result","op":"vector_add","dtype":"f32","shape":"N=64M","ref_gbps":850.2,"mt_gbps":912.1,"passed":true}
//! {"type":"done","ok":3,"errors":0}
//! ```
//!
//! ### `--action=test`
//! ```jsonc
//! {"type":"test_result","op":"vector_add","subop":"","dtype":"f32","passed":true}
//! {"type":"done","ok":1,"errors":0}
//! ```
//!
//! ### `--action=build`
//! ```jsonc
//! {"type":"result","op":"vector_add","dtype":"f32","ok":true}
//! {"type":"done","ok":1,"errors":0}
//! ```

use serde::{Deserialize, Serialize, de};

/// A generic JSONL protocol record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProtocolRecord {
    /// Benchmark result.
    #[serde(rename = "result")]
    Result {
        op: String,
        #[serde(default)]
        dtype: String,
        #[serde(default)]
        shape: String,
        #[serde(default)]
        ref_gbps: Option<f64>,
        #[serde(default)]
        mt_gbps: Option<f64>,
        #[serde(default)]
        passed: bool,
        #[serde(default)]
        ok: Option<bool>,
        #[serde(default)]
        error: Option<String>,
    },
    /// Test result.
    #[serde(rename = "test_result")]
    TestResult {
        op: String,
        #[serde(default)]
        subop: String,
        #[serde(default)]
        dtype: String,
        passed: bool,
    },
    /// Completion record.
    #[serde(rename = "done")]
    Done { ok: u32, errors: u32 },
}

/// Parse a single JSONL record from a line.
pub fn parse_record(line: &str) -> Result<ProtocolRecord, serde_json::Error> {
    serde_json::from_str(line)
}

/// Parse all JSONL records from a byte stream.
pub fn parse_all_records(data: &[u8]) -> Result<Vec<ProtocolRecord>, serde_json::Error> {
    let text =
        std::str::from_utf8(data).map_err(|e| de::Error::custom(format!("invalid UTF-8: {e}")))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect()
}

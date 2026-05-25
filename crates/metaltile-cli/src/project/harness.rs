//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! JSON-line protocol reader for external-project harness communication.
//!
//! Extracted to eliminate the copy-paste JSONL parsing loop that appeared
//! in `bench.rs`, `build.rs`, and `test.rs`.

use std::{io::BufRead, process::Child};

use serde_json::Value;

use crate::{
    error::CliError,
    term::{Color, Style, paint_stderr},
};

/// A parsed message from the harness JSONL protocol.
#[derive(Debug)]
pub enum HarnessMessage {
    /// A bench result row.
    BenchResult(Value),
    /// A build result row.
    BuildResult(Value),
    /// A test result row.
    TestResult(Value),
    /// The "done" signal (sent once at the end of the stream).
    Done { device: Option<String> },
    /// An unparseable line.
    Unknown(String),
}

/// Read JSONL messages from a child process's stdout until EOF.
///
/// Returns all parsed messages. I/O errors and JSON parse failures are
/// collected into `CliError::HarnessProtocol`.
pub fn read_harness_output(child: &mut Child) -> Result<Vec<HarnessMessage>, CliError> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CliError::HarnessProtocol("child had no stdout".into()))?;
    let reader = std::io::BufReader::new(stdout);
    let mut messages = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| CliError::HarnessProtocol(format!("read error: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(val) => {
                let msg = match val.get("type").and_then(|t| t.as_str()) {
                    Some("result") => HarnessMessage::BenchResult(val),
                    Some("build_result") => HarnessMessage::BuildResult(val),
                    Some("test_result") => HarnessMessage::TestResult(val),
                    Some("done") => {
                        let device = val.get("device").and_then(|v| v.as_str()).map(String::from);
                        HarnessMessage::Done { device }
                    },
                    _ => HarnessMessage::Unknown(line.clone()),
                };
                messages.push(msg);
            },
            Err(e) => {
                eprintln!(
                    "  {} JSON parse: {}",
                    paint_stderr("warn:", Style::new().fg(Color::Yellow).bold()),
                    e,
                );
                messages.push(HarnessMessage::Unknown(line));
            },
        }
    }

    Ok(messages)
}

/// Convenience: spawn a harness subprocess, read JSONL, and wait.
///
/// The caller provides a `Command` builder that is fully configured
/// (binary, args, env).  This function pipes stdout, reads all messages,
/// and waits for the child to exit.
pub fn run_harness(cmd: &mut std::process::Command) -> Result<Vec<HarnessMessage>, CliError> {
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .map_err(|e| CliError::Subprocess(format!("spawn: {e}")))?;

    let messages = read_harness_output(&mut child)?;

    let status = child.wait().map_err(|e| CliError::Subprocess(format!("wait: {e}")))?;
    if !status.success() {
        return Err(CliError::Subprocess(format!("harness exited with {status}")));
    }

    Ok(messages)
}

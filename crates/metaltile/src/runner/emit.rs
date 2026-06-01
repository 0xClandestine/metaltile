//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Protocol-message emitter for the `__tile_runner` subprocess.
//!
//! Each emitted message is one JSON line terminated by `\n`, written to
//! any `std::io::Write` sink (typically `std::io::stdout()`).

use std::io::{self, Write};

use metaltile_core::protocol::ProtocolMessage;

/// Write a single [`ProtocolMessage`] as a JSON line to `sink`.
///
/// Flushes after every write so the CLI process receives messages
/// immediately rather than buffered.
pub fn emit(sink: &mut impl Write, msg: &ProtocolMessage) -> io::Result<()> {
    sink.write_all(&msg.to_json_line())?;
    sink.flush()
}

/// Convenience wrapper that emits to locked stdout.
pub fn emit_stdout(msg: &ProtocolMessage) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = emit(&mut out, msg);
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Terminal styling backed by `anstyle` + `anstream`.
//!
//! Provides `Style` builder and `paint_stdout` / `paint_stderr` helpers.
//! For injectable output see `OutputWriter` which wraps any `Write` and
//! handles TTY detection, styles, and reset sequences.

use std::sync::OnceLock;

// ── Public types ─────────────────────────────────────────────────────────
/// Re-export of `anstyle::AnsiColor`.
pub use anstyle::AnsiColor as Color;

/// ANSI text style backed by `anstyle::Style`.
#[derive(Clone, Copy, Default)]
pub struct Style(pub(crate) anstyle::Style);

impl Style {
    pub fn new() -> Self { Self::default() }
    pub fn fg(mut self, color: Color) -> Self {
        self.0 = self.0.fg_color(Some(anstyle::Color::Ansi(color)));
        self
    }
    pub fn bold(mut self) -> Self {
        self.0 = self.0.bold();
        self
    }
    pub fn dim(mut self) -> Self {
        self.0 = self.0.dimmed();
        self
    }
}

impl From<Style> for anstyle::Style {
    fn from(s: Style) -> Self { s.0 }
}

// ── Stream identifier ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

// ── TTY detection ───────────────────────────────────────────────────────

fn is_term(stream: Stream) -> bool {
    static STDOUT_TERM: OnceLock<bool> = OnceLock::new();
    static STDERR_TERM: OnceLock<bool> = OnceLock::new();

    let cell = match stream {
        Stream::Stdout => &STDOUT_TERM,
        Stream::Stderr => &STDERR_TERM,
    };
    *cell.get_or_init(|| {
        let choice = match stream {
            Stream::Stdout => anstream::AutoStream::choice(&std::io::stdout()),
            Stream::Stderr => anstream::AutoStream::choice(&std::io::stderr()),
        };
        choice != anstream::ColorChoice::Never
    })
}

// ── Paint helpers ───────────────────────────────────────────────────────

pub fn paint_stdout(text: impl AsRef<str>, style: Style) -> String {
    paint(Stream::Stdout, text.as_ref(), style)
}

pub fn paint_stderr(text: impl AsRef<str>, style: Style) -> String {
    paint(Stream::Stderr, text.as_ref(), style)
}

fn paint(stream: Stream, text: &str, style: Style) -> String {
    if text.is_empty() || !is_term(stream) {
        return text.to_owned();
    }
    format!("{}{text}{}", style.0, anstyle::Reset)
}

// ── OutputWriter: injectable output with ANSI support ────────────────────

/// A `dyn Write` wrapper that handles ANSI styling and TTY detection.
///
/// Use this instead of raw `println!` / `eprintln!` when you want to
/// inject output destinations for testability.
pub struct OutputWriter {
    stream: Stream,
    inner: Box<dyn std::io::Write + Send>,
}

impl OutputWriter {
    /// Create a writer that writes to a `Write` implementor.
    pub fn new(stream: Stream, inner: Box<dyn std::io::Write + Send>) -> Self {
        Self { stream, inner }
    }

    /// Create a writer that writes to stdout with full TTY detection.
    pub fn stdout() -> Self { Self { stream: Stream::Stdout, inner: Box::new(std::io::stdout()) } }

    /// Create a writer that writes to stderr with full TTY detection.
    pub fn stderr() -> Self { Self { stream: Stream::Stderr, inner: Box::new(std::io::stderr()) } }

    /// Paint text with a style and write it, flushing the inner writer.
    pub fn paint(&mut self, text: impl AsRef<str>, style: Style) -> std::io::Result<()> {
        let styled = paint(self.stream, text.as_ref(), style);
        self.inner.write_all(styled.as_bytes())?;
        Ok(())
    }

    /// Write a line of plain text.
    pub fn line(&mut self, text: impl AsRef<str>) -> std::io::Result<()> {
        self.inner.write_all(text.as_ref().as_bytes())?;
        self.inner.write_all(b"\n")?;
        Ok(())
    }

    /// Write a styled line (styled text + newline).
    pub fn paint_line(&mut self, text: impl AsRef<str>, style: Style) -> std::io::Result<()> {
        self.paint(text, style)?;
        self.inner.write_all(b"\n")?;
        Ok(())
    }

    /// Flush the inner writer.
    pub fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }

    /// Access the inner `Write` reference for use with `write!` macro.
    pub fn inner(&mut self) -> &mut dyn std::io::Write { &mut *self.inner }
}

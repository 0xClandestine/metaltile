//! Terminal styling backed by `anstyle` + `anstream`.
//!
//! Thin wrapper that provides the same builder API (`Style::new().fg(Color::Cyan).bold()`)
//! while delegating color management and TTY detection to the clap ecosystem crates
//! already in the dependency tree.

use std::{
    sync::OnceLock,
};

// ── Public types ─────────────────────────────────────────────────────────

/// Re-export of `anstyle::AnsiColor` — drop-in replacement for our old
/// `Color` enum.  Variant names are identical: `Red`, `Green`, etc.
pub use anstyle::AnsiColor as Color;

/// ANSI text style backed by `anstyle::Style`.
#[derive(Clone, Copy, Default)]
pub struct Style(anstyle::Style);

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

// ── TTY detection (delegated to anstream) ───────────────────────────────

fn is_term(stream: Stream) -> bool {
    static STDOUT_TERM: OnceLock<bool> = OnceLock::new();
    static STDERR_TERM: OnceLock<bool> = OnceLock::new();

    let cell = match stream {
        Stream::Stdout => &STDOUT_TERM,
        Stream::Stderr => &STDERR_TERM,
    };
    *cell.get_or_init(|| {
        // anstream::AutoStream::choice checks NO_COLOR, CLICOLOR_FORCE,
        // CLICOLOR, TERM=dumb, and IsTerminal — same logic we had by hand.
        let choice = match stream {
            Stream::Stdout => anstream::AutoStream::choice(&std::io::stdout()),
            Stream::Stderr => anstream::AutoStream::choice(&std::io::stderr()),
        };
        choice != anstream::ColorChoice::Never
    })
}

#[derive(Clone, Copy)]
enum Stream {
    Stdout,
    Stderr,
}

// ── Paint helpers ────────────────────────────────────────────────────────

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

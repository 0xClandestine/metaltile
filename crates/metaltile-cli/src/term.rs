//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Terminal styling backed by `anstyle` + `anstream`.
//!
//! Thin wrapper that provides the same builder API (`Style::new().fg(Color::Cyan).bold()`)
//! while delegating color management and TTY detection to the clap ecosystem crates
//! already in the dependency tree.

use std::sync::{
    Arc,
    OnceLock,
    atomic::{AtomicBool, Ordering},
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

// ── Spinner ──────────────────────────────────────────────────────────────

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Animated terminal spinner — no-op when stderr is not a TTY.
///
/// ```text
/// [⠋] Compiling...
/// ```
///
/// Drop or call [`Spinner::stop`] to clear the line.
#[must_use]
pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    is_tty: bool,
}

impl Spinner {
    /// Spawn a spinner with the given message.
    pub fn new(msg: impl Into<String>) -> Self {
        let is_tty = is_term(Stream::Stderr);
        let stop = Arc::new(AtomicBool::new(false));

        if !is_tty {
            return Self { stop, handle: None, is_tty: false };
        }

        let stop2 = stop.clone();
        let msg = msg.into();
        let handle = std::thread::spawn(move || {
            for (i, frame) in FRAMES.iter().cycle().enumerate() {
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                eprint!("\r\x1b[K[{frame}] {msg}");
                if i % 10 == 0 {
                    // flush stderr periodically so the animation is visible
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                }
                std::thread::sleep(std::time::Duration::from_millis(80));
            }
        });

        Self { stop, handle: Some(handle), is_tty: true }
    }

    /// Stop the spinner and erase the line.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        if self.is_tty {
            eprint!("\r\x1b[K");
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) { self.stop(); }
}

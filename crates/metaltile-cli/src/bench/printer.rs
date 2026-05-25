//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Suite printer: formats BenchRow batches as terminal tables.
//!
//! Uses injectable [`OutputWriter`] for testability.  Table layout
//! computation is separated from rendering so the printer doesn't
//! need to track mutable state.

use std::collections::HashMap;

use metaltile::harness::BenchRow;
use crate::bench::profile::ProfileRow;
use metaltile_core::bench::types::CorrectnessStatus;

use crate::term::{Color, OutputWriter, Style, paint_stdout};

// ── SuitePrinter ──────────────────────────────────────────────────────────

/// Formats bench rows as terminal tables.
///
/// Writes to an injectable [`OutputWriter`] so tests can capture output
/// without redirecting stdout globally.
pub struct SuitePrinter {
    show_correctness: bool,
    writer: OutputWriter,
}

impl SuitePrinter {
    /// Create a new printer that writes to stdout.
    pub fn new(show_correctness: bool) -> Self {
        Self { show_correctness, writer: OutputWriter::stdout() }
    }

    /// Create a printer with an injectable output writer.
    pub fn with_writer(show_correctness: bool, writer: OutputWriter) -> Self {
        Self { show_correctness, writer }
    }

    /// Print a batch of results.
    pub fn print_batch<'a>(
        &mut self,
        results: &[BenchRow],
        profile_map: Option<&HashMap<(String, String), ProfileRow>>,
        verbose: u8,
    ) {
        if results.is_empty() {
            return;
        }

        let term_width = term_width();

        for chunk in Self::chunk_by_op(results) {
            // Print op header
            let m = chunk[0].metric();
            let (shape_w, ref_w, mt_w, pct_w, ck_w) =
                table_widths(term_width, &m, self.show_correctness);

            self.print_op_header(
                &chunk[0],
                &table_widths(term_width, &m, self.show_correctness),
                verbose,
            );

            // Print each data row
            for result in chunk {
                self.print_data_row(
                    result,
                    &(shape_w, ref_w, mt_w, pct_w, ck_w),
                    profile_map,
                    verbose,
                );
            }
        }

        self.flush();
    }

    /// Print summary footer — call once after all batches.
    pub fn finish(&mut self) { let _ = self.writer.line(""); }

    /// Flush the output writer.
    pub fn flush(&self) {
        let mut w = OutputWriter::stdout();
        let _ = w.flush();
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Group results by op_display, preserving order.
    fn chunk_by_op<'a>(results: &'a [BenchRow]) -> Vec<&'a [BenchRow]> {
        let mut chunks: Vec<&[BenchRow]> = Vec::new();
        let mut start = 0;
        for i in 1..=results.len() {
            if i == results.len() || results[i].op_display() != results[start].op_display() {
                chunks.push(&results[start..i]);
                start = i;
            }
        }
        chunks
    }

    fn print_op_header(
        &mut self,
        result: &BenchRow,
        &(shape_w, ref_w, mt_w, pct_w, ck_w): &TableWidths,
        verbose: u8,
    ) {
        let metric = result.metric();
        let sep = col_sep();
        let bold = Style::new().fg(Color::BrightWhite).bold();

        let mut hdr = format!(
            "  {} {} {} {} {} {} {}",
            paint_stdout(pad_left("Shape", shape_w), bold),
            sep,
            paint_stdout(pad_right(&format!("Ref({})", metric), ref_w), bold),
            sep,
            paint_stdout(pad_right(&format!("MT({})", metric), mt_w), bold),
            sep,
            paint_stdout(pad_right("MT%", pct_w), bold),
        );
        if self.show_correctness {
            hdr.push_str(&format!(" {} {}", sep, paint_stdout(pad_right("ok", ck_w), bold)));
        }
        if verbose >= 2 {
            hdr.push_str(&format!(
                " {} {} {} {} {} {}",
                sep,
                paint_stdout(pad_right("p95", 5), bold),
                sep,
                paint_stdout(pad_right("p99", 5), bold),
                sep,
                paint_stdout(pad_right("cv%", 5), bold),
            ));
        }
        if verbose >= 1 {
            hdr.push_str(&format!(
                " {} {} {} {} {} {}",
                sep,
                paint_stdout(pad_right("occ%", 5), bold),
                sep,
                paint_stdout(pad_right("regs", 4), bold),
                sep,
                paint_stdout(pad_right("bottleneck", 17), bold),
            ));
        }

        let op = paint_stdout(result.op_display(), Style::new().fg(Color::Cyan).bold());
        let _ = self.writer.line(&format!("  {op}"));
        let _ = self.writer.line(&hdr);

        let n_cols: usize = if self.show_correctness { 5 } else { 4 };
        let gaps = (n_cols.saturating_sub(1)) * 3;
        let timing_cols = if verbose >= 2 { 5 + 3 + 5 + 3 + 5 + 3 } else { 0 };
        let profile_cols = if verbose >= 1 { 5 + 3 + 4 + 3 + 17 + 3 } else { 0 };
        let total_w = 4
            + shape_w
            + gaps
            + ref_w
            + mt_w
            + pct_w
            + if self.show_correctness { ck_w } else { 0 }
            + timing_cols
            + profile_cols;
        let sep_line = paint_stdout("─".repeat(total_w), Style::new().fg(Color::BrightBlack).dim());
        let _ = self.writer.line(&format!("  {sep_line}"));
    }

    fn print_data_row(
        &mut self,
        result: &BenchRow,
        &(shape_w, ref_w, mt_w, pct_w, ck_w): &TableWidths,
        profile_map: Option<&HashMap<(String, String), ProfileRow>>,
        verbose: u8,
    ) {
        let _metric = result.metric();
        let sep = col_sep();

        let shape =
            paint_stdout(pad_left(result.shape(), shape_w), Style::new().fg(Color::BrightWhite));
        let ref_s = fmt_perf(result.ref_perf(), "—");
        let mt_s = fmt_perf(result.mt_perf(), "NYI");
        let pct_s = result.pct().map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());

        let ref_cell = style_reference(&pad_right(&ref_s, ref_w), result.ref_perf());
        let mt_cell = style_metaltile(&pad_right(&mt_s, mt_w), result);
        let pct_cell = style_pct(&pad_right(&pct_s, pct_w), result);

        let mut row = format!("  {shape} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell}");
        if self.show_correctness {
            let ck_icon = correctness_icon(result.correctness_status());
            let ck_cell =
                style_correctness(&pad_right(&ck_icon, ck_w), result.correctness_status());
            row.push_str(&format!(" {sep} {ck_cell}"));
        }
        if verbose >= 2 {
            let t = fmt_timing(result);
            row.push_str(&format!(" {sep} {t}"));
        }
        if verbose >= 1 {
            let p = fmt_profile(result, profile_map);
            row.push_str(&format!(" {sep} {p}"));
        }
        let _ = self.writer.line(&row);
    }
}

// ── Table layout types ──────────────────────────────────────────────────

type TableWidths = (usize, usize, usize, usize, usize);

fn term_width() -> usize {
    std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok()).unwrap_or(80).clamp(60, 200)
}

fn table_widths(term_width: usize, metric: &str, show_ck: bool) -> TableWidths {
    let avail = term_width.saturating_sub(2);
    let ref_w = 4 + metric.len() + 2;
    let mt_w = 3 + metric.len() + 2;
    let pct_w = 5;
    let ck_w = if show_ck { 3 } else { 0 };
    let rhs = ref_w + mt_w + pct_w + ck_w;
    let n_cols: usize = if show_ck { 5 } else { 4 };
    let gaps = (n_cols.saturating_sub(1)) * 3;
    let shape_w = avail.saturating_sub(rhs + gaps + 2);
    let shape_w = shape_w.clamp(8, 42);
    (shape_w, ref_w, mt_w, pct_w, ck_w)
}

// ── Cell styling ─────────────────────────────────────────────────────────

fn correctness_icon(status: CorrectnessStatus) -> String {
    match status {
        CorrectnessStatus::Passed { .. } => "✓".into(),
        CorrectnessStatus::Failed { .. } => "✗".into(),
        CorrectnessStatus::Unchecked => "!".into(),
        CorrectnessStatus::Unavailable => "—".into(),
    }
}

fn fmt_perf(v: Option<f64>, fallback: &str) -> String {
    match v {
        None => fallback.into(),
        Some(x) => format!("{x:.1}"),
    }
}

pub(crate) fn col_sep() -> String { paint_stdout("│", Style::new().fg(Color::BrightBlack).dim()) }

pub(crate) fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }

pub(crate) fn pad_right(text: &str, width: usize) -> String { format!("{text:>width$}") }

fn style_reference(text: &str, value: Option<f64>) -> String {
    let style = if value.is_some() {
        Style::new().fg(Color::BrightWhite)
    } else {
        Style::new().fg(Color::Red).bold()
    };
    paint_stdout(text, style)
}

fn style_metaltile(text: &str, result: &BenchRow) -> String {
    let style = match (result.mt_perf(), result.correctness_status()) {
        (Some(_), CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(_), _) => Style::new().fg(Color::BrightWhite).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(text, style)
}

fn style_pct(text: &str, result: &BenchRow) -> String {
    let style = match (result.pct(), result.correctness_status()) {
        (_, CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(p), _) if p >= 90.0 => Style::new().fg(Color::Green).bold(),
        (Some(p), _) if p >= 60.0 => Style::new().fg(Color::Yellow).bold(),
        (Some(_), _) => Style::new().fg(Color::Red).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(text, style)
}

fn style_correctness(text: &str, status: CorrectnessStatus) -> String {
    let style = match status {
        CorrectnessStatus::Passed { .. } => Style::new().fg(Color::Green).bold(),
        CorrectnessStatus::Failed { .. } => Style::new().fg(Color::Red).bold(),
        CorrectnessStatus::Unchecked => Style::new().fg(Color::Yellow).bold(),
        CorrectnessStatus::Unavailable => Style::new().fg(Color::BrightBlack).dim(),
    };
    paint_stdout(text, style)
}

fn fmt_timing(row: &BenchRow) -> String {
    let sep = col_sep();
    let dim = Style::new().fg(Color::BrightBlack).dim();
    match (row.timing_p95_us, row.timing_p99_us, row.timing_cv_pct) {
        (Some(p95), Some(p99), Some(cv)) => {
            let p95_s = paint_stdout(format!("{:>5.1}", p95), Style::new().fg(Color::BrightWhite));
            let p99_s = paint_stdout(format!("{:>5.1}", p99), Style::new().fg(Color::BrightWhite));
            let cv_str = if cv > 5.0 {
                paint_stdout(format!("{:>4.1}%", cv), Style::new().fg(Color::Yellow).bold())
            } else {
                paint_stdout(format!("{:>4.1}%", cv), Style::new().fg(Color::Green))
            };
            format!("{p95_s} {sep} {p99_s} {sep} {cv_str}")
        },
        _ => {
            let dash = paint_stdout("   —", dim);
            let dash2 = paint_stdout("   —", dim);
            let dash3 = paint_stdout("   —", dim);
            format!("{dash} {sep} {dash2} {sep} {dash3}")
        },
    }
}

fn fmt_profile(
    result: &BenchRow,
    profile_map: Option<&HashMap<(String, String), ProfileRow>>,
) -> String {
    let sep = col_sep();
    let dim = Style::new().fg(Color::BrightBlack).dim();
    let dash_occ = paint_stdout("   —", dim);
    let dash_regs = paint_stdout("   —", dim);
    let dash_bn = paint_stdout(" —", dim);
    let not_available = format!("{dash_occ} {sep} {dash_regs} {sep} {dash_bn}");

    let map = match profile_map {
        Some(m) => m,
        None => return not_available,
    };

    let dtype_label = result.shape().rsplit_once(' ').map(|(_, last)| last).unwrap_or("f32");
    let key = (result.op_display(), dtype_label.to_string());
    let p = match map.get(&key) {
        Some(p) => p,
        None => return not_available,
    };

    let occ_color = if p.occ_pct >= 100.0 {
        Color::Green
    } else if p.occ_pct >= 60.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let occ = paint_stdout(format!("{:>4.0}%", p.occ_pct), Style::new().fg(occ_color).bold());
    let regs =
        paint_stdout(format!("{:>3}r", p.regs_per_thread), Style::new().fg(Color::BrightWhite));
    let bn =
        paint_stdout(format!("{:>17}", p.bottleneck), Style::new().fg(Color::BrightBlack).dim());
    format!("{occ} {sep} {regs} {sep} {bn}")
}

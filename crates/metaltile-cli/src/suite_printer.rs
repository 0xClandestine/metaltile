//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Suite printer: formats OpResult batches and `ProtocolMessage::BenchResult`
//! JSON-stream entries as terminal tables.
//!
//! Two input paths:
//!
//! * **Phase 1 (current)**: `print_batch(&[OpResult])` — in-process bench
//!   results from `metaltile-std::bench_types`.
//! * **Phase 2 (subprocess)**: `print_bench_result(&BenchResult)` — parsed
//!   JSON lines from the `__tile_runner` subprocess.  The data type lives in
//!   `metaltile-core::protocol` so the CLI does not need `metaltile-std`.

use std::io::Write;

use metaltile_core::protocol::BenchResult as ProtoBenchResult;
use metaltile_std::bench_types::{CorrectnessStatus, OpResult};

use crate::term::{Color, Style, paint_stdout};

// ── SuitePrinter ──────────────────────────────────────────────────────────

pub struct SuitePrinter {
    show_correctness: bool,
    started: bool,
    last_op_display: Option<String>,
    term_width: usize,
    cur_metric: Option<&'static str>,
    /// Shows profile info (occ%, regs, bottleneck) in op headers when Some.
    profile_map: Option<std::collections::HashMap<(String, String), ProfileRow>>,
    /// Shows timing columns (p95, p99, cv%) when > 0 and results carry timing.
    verbose: u8,
}

/// Compile-time profile snippet for one kernel (used by bench -v).
#[derive(Clone)]
pub struct ProfileRow {
    pub occ_pct: f64,
    pub regs_per_thread: usize,
    pub bottleneck: &'static str,
}

impl SuitePrinter {
    pub fn new(show_correctness: bool) -> Self {
        Self {
            show_correctness,
            started: false,
            last_op_display: None,
            term_width: term_width(),
            cur_metric: None,
            profile_map: None,
            verbose: 0,
        }
    }

    /// Construct a printer whose baseline verbosity comes from the shared
    /// `Harness` config.  The caller may still override with `set_verbose`.
    pub fn from_harness(harness: &crate::harness::Harness) -> Self {
        let mut p = Self::new(true);
        p.verbose = harness.config.verbose;
        p
    }

    pub fn set_verbose(&mut self, v: u8) { self.verbose = v; }

    pub fn set_profile_map(&mut self, m: std::collections::HashMap<(String, String), ProfileRow>) {
        self.profile_map = Some(m);
    }

    pub fn set_term_width(&mut self, w: usize) { self.term_width = w.clamp(60, 200); }

    pub fn print_batch(&mut self, results: &[OpResult]) {
        if results.is_empty() {
            return;
        }
        if !self.started {
            self.started = true;
        }

        for result in results {
            let new_group = self.last_op_display.as_deref() != Some(&result.op_display());
            if new_group {
                if self.cur_metric != Some(result.metric()) {
                    self.cur_metric = Some(result.metric());
                }
                if self.last_op_display.is_some() {
                    println!();
                }
                self.print_op_header(result);
            }
            self.last_op_display = Some(result.op_display());
            self.print_data_row(result);
        }
        self.flush();
    }

    /// Print one `ProtocolMessage::BenchResult` line (Phase 2 subprocess path).
    ///
    /// Emits a single compact row: `  ✓/✗  <name> [<dtype>]  <mt> GB/s  (ref: <ref>)  <pct>%`.
    /// This intentionally does not try to match the full `print_batch` table
    /// layout — that refactor (grouping by kernel name, shared column widths)
    /// is a follow-up once all commands are subprocess-based.
    pub fn print_bench_result(&mut self, r: &ProtoBenchResult) {
        self.started = true;
        let ok_sym = if r.correct {
            paint_stdout("✓", Style::new().fg(Color::Green).bold())
        } else {
            paint_stdout("✗", Style::new().fg(Color::Red).bold())
        };
        let label =
            paint_stdout(format!("{} [{}]", r.name, r.dtype), Style::new().fg(Color::BrightWhite));
        let mt = paint_stdout(
            format!("{:.1} GB/s", r.mt_gbps),
            Style::new().fg(Color::BrightWhite).bold(),
        );
        let ref_part = match r.ref_gbps {
            Some(rg) => paint_stdout(format!("ref {rg:.1}"), Style::new().fg(Color::BrightBlack)),
            None => paint_stdout("no ref", Style::new().fg(Color::BrightBlack).dim()),
        };
        let pct_part = match r.mt_pct {
            Some(p) => {
                let style = if p >= 90.0 {
                    Style::new().fg(Color::Green).bold()
                } else if p >= 60.0 {
                    Style::new().fg(Color::Yellow).bold()
                } else {
                    Style::new().fg(Color::Red).bold()
                };
                paint_stdout(format!("{p:.0}%"), style)
            },
            None => paint_stdout("—", Style::new().fg(Color::BrightBlack).dim()),
        };
        println!("  {ok_sym}  {label}  {mt}  {ref_part}  {pct_part}");
        self.flush();
    }

    pub fn finish(&mut self) {
        if !self.started {
            return;
        }
        println!();
        self.flush();
    }

    fn print_op_header(&self, result: &OpResult) {
        let metric = self.cur_metric.unwrap_or("perf");
        let (shape_w, ref_w, mt_w, pct_w, ck_w) =
            sub_table_widths(self.term_width, metric, self.show_correctness);

        let sep = col_sep();
        let bold = Style::new().fg(Color::BrightWhite).bold();

        let mut hdr = format!(
            "  {}  {} {} {} {} {} {}",
            paint_stdout(pad_left("Shape", shape_w), bold),
            sep,
            paint_stdout(pad_right(&format!("Ref({})", metric), ref_w), bold),
            sep,
            paint_stdout(pad_right(&format!("MT({})", metric), mt_w), bold),
            sep,
            paint_stdout(pad_right("MT%", pct_w), bold),
        );
        if self.show_correctness {
            hdr.push_str(&format!(" {} {}", sep, paint_stdout(pad_right("ok", ck_w), bold),));
        }
        // -vv: scaling distribution columns
        if self.verbose >= 2 {
            let pw = 5;
            let qw = 5;
            let cw = 5;
            hdr.push_str(&format!(
                " {} {} {} {} {} {}",
                sep,
                paint_stdout(pad_right("p95", pw), bold),
                sep,
                paint_stdout(pad_right("p99", qw), bold),
                sep,
                paint_stdout(pad_right("cv%", cw), bold),
            ));
        }
        // -v/-vv: profile columns
        if self.verbose >= 1 {
            let ow = 5;
            let rw = 4;
            let bw = 17;
            hdr.push_str(&format!(
                " {} {} {} {} {} {}",
                sep,
                paint_stdout(pad_right("occ%", ow), bold),
                sep,
                paint_stdout(pad_right("regs", rw), bold),
                sep,
                paint_stdout(pad_right("bottleneck", bw), bold),
            ));
        }

        // Op line — just the name, no profile
        let op = paint_stdout(result.op_display(), Style::new().fg(Color::Cyan).bold());
        println!("  {op}");
        println!("{hdr}");

        let n_cols: usize = if self.show_correctness { 5 } else { 4 };
        let gaps = (n_cols.saturating_sub(1)) * 3;
        let timing_cols = if self.verbose >= 2 { 5 + 3 + 5 + 3 + 5 + 3 } else { 0 };
        let profile_cols = if self.verbose >= 1 { 5 + 3 + 4 + 3 + 17 + 3 } else { 0 };
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
        println!("  {sep_line}");
    }

    fn print_data_row(&self, result: &OpResult) {
        let metric = self.cur_metric.unwrap_or("perf");
        let (shape_w, ref_w, mt_w, pct_w, ck_w) =
            sub_table_widths(self.term_width, metric, self.show_correctness);

        let shape =
            paint_stdout(pad_left(result.shape(), shape_w), Style::new().fg(Color::BrightWhite));
        let ref_s = fmt_perf(result.ref_perf(), metric, "—");
        let mt_s = fmt_perf(result.mt_perf(), metric, "NYI");
        let pct_s = result.pct().map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());

        let ref_cell = style_reference(&pad_right(&ref_s, ref_w), result.ref_perf());
        let mt_cell = style_metaltile(&pad_right(&mt_s, mt_w), result);
        let pct_cell = style_pct(&pad_right(&pct_s, pct_w), result);
        let sep = col_sep();

        let mut row = format!("  {shape} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell}");
        if self.show_correctness {
            let ck_icon = correctness_icon(&result.correctness_status());
            let ck_cell =
                style_correctness(&pad_right(&ck_icon, ck_w), result.correctness_status());
            row.push_str(&format!(" {sep} {ck_cell}"));
        }
        if self.verbose >= 2 {
            let t = fmt_timing(result);
            row.push_str(&format!(" {sep} {t}"));
        }
        if self.verbose >= 1 {
            let p = fmt_profile(result, &self.profile_map);
            row.push_str(&format!(" {sep} {p}"));
        }
        println!("{row}");
    }

    fn flush(&self) { let _ = std::io::stdout().flush(); }
}

// ── Table layout helpers ──────────────────────────────────────────────────

fn term_width() -> usize {
    std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok()).unwrap_or(80).clamp(60, 200)
}

fn sub_table_widths(
    term_width: usize,
    metric: &str,
    show_ck: bool,
) -> (usize, usize, usize, usize, usize) {
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

fn correctness_icon(status: &CorrectnessStatus) -> String {
    match status {
        CorrectnessStatus::Passed { .. } => "✓".into(),
        CorrectnessStatus::Failed { .. } => "✗".into(),
        CorrectnessStatus::Unchecked => "!".into(),
        CorrectnessStatus::Unavailable => "—".into(),
    }
}

fn fmt_perf(v: Option<f64>, _metric: &str, fallback: &str) -> String {
    match v {
        None => fallback.into(),
        Some(x) => format!("{x:.1}"),
    }
}

fn col_sep() -> String { paint_stdout("│", Style::new().fg(Color::BrightBlack).dim()) }

fn pad_left(text: &str, width: usize) -> String { format!("{text:<width$}") }

fn pad_right(text: &str, width: usize) -> String { format!("{text:>width$}") }

fn style_reference(text: &str, value: Option<f64>) -> String {
    let style = if value.is_some() {
        Style::new().fg(Color::BrightWhite)
    } else {
        Style::new().fg(Color::Red).bold()
    };
    paint_stdout(text, style)
}

fn style_metaltile(text: &str, result: &OpResult) -> String {
    let style = match (result.mt_perf(), result.correctness_status()) {
        (Some(_), CorrectnessStatus::Failed { .. }) => Style::new().fg(Color::Red).bold(),
        (Some(_), _) => Style::new().fg(Color::BrightWhite).bold(),
        (None, _) => Style::new().fg(Color::Yellow).bold(),
    };
    paint_stdout(text, style)
}

fn style_pct(text: &str, result: &OpResult) -> String {
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

/// Format timing columns for a result row. Returns "p95 │ p99 │ cv%" or "   — │   — │   —".
fn fmt_timing(result: &OpResult) -> String {
    let sep = col_sep();
    let dim = Style::new().fg(Color::BrightBlack).dim();
    match result.mt_timing {
        Some(ref t) if t.is_valid() => {
            let p95 =
                paint_stdout(format!("{:>5.1}", t.p95_us), Style::new().fg(Color::BrightWhite));
            let p99 =
                paint_stdout(format!("{:>5.1}", t.p99_us), Style::new().fg(Color::BrightWhite));
            let cv_str = if t.cv_pct > 5.0 {
                paint_stdout(format!("{:>4.1}%", t.cv_pct), Style::new().fg(Color::Yellow).bold())
            } else {
                paint_stdout(format!("{:>4.1}%", t.cv_pct), Style::new().fg(Color::Green))
            };
            format!("{p95} {} {p99} {} {cv_str}", sep, sep)
        },
        _ => {
            let dash = paint_stdout("   —", dim);
            let dash2 = paint_stdout("   —", dim);
            let dash3 = paint_stdout("   —", dim);
            format!("{dash} {} {dash2} {} {dash3}", sep, sep)
        },
    }
}

/// Format profile columns for a result row. Returns "occ% │ regs │ bottleneck" or "   — │   — │ —".
fn fmt_profile(
    result: &OpResult,
    profile_map: &Option<std::collections::HashMap<(String, String), ProfileRow>>,
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

    // Parse dtype label from shape string (last word).
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

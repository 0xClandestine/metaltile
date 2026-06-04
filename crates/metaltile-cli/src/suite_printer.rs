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

use crate::{
    bench_types::{CorrectnessStatus, OpResult},
    term::{Color, Style, paint_stdout},
};

// ── SuitePrinter ──────────────────────────────────────────────────────────

pub struct SuitePrinter {
    show_correctness: bool,
    started: bool,
    last_op_display: Option<String>,
    term_width: usize,
    cur_metric: Option<&'static str>,
    /// Shows timing columns (p95, p99, cv%) when > 0 and results carry timing.
    verbose: u8,
}

impl SuitePrinter {
    pub fn new(show_correctness: bool) -> Self {
        Self {
            show_correctness,
            started: false,
            last_op_display: None,
            term_width: term_width(),
            cur_metric: None,
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
        // Compact profile suffix (Phase-2 wire path): GFLOP/s + bottleneck when
        // the runner attached a profile. The full `-v` roofline table is the
        // in-process `print_batch` path; this keeps the streamed line readable
        // while still surfacing the headline derived metrics off the wire.
        let profile_part = match &r.profile {
            Some(p) => {
                let mut bits = Vec::new();
                if let Some(g) = p.gflops {
                    bits.push(format!("{g:.1} GFLOP/s"));
                }
                if let Some(b) = &p.bottleneck {
                    bits.push(b.clone());
                }
                if bits.is_empty() {
                    String::new()
                } else {
                    format!(
                        "  {}",
                        paint_stdout(bits.join(" · "), Style::new().fg(Color::BrightBlack))
                    )
                }
            },
            None => String::new(),
        };
        println!("  {ok_sym}  {label}  {mt}  {ref_part}  {pct_part}{profile_part}");
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
        let w = sub_table_widths(self.term_width, metric, self.show_correctness);

        let sep = col_sep();
        // Column headers use the same muted bold as the rest of the CLI's labels
        // (cf. `tile device`): BrightBlack-bold headers recede so the BrightWhite
        // data values stand out. (Was BrightWhite-bold, which clashed with #255.)
        let bold = Style::new().fg(Color::BrightBlack).bold();

        // Default columns: Shape │ MT(µs) │ Ref(perf) │ MT(perf) │ MT% │ GFLOP/s [│ ok].
        let mut hdr = format!(
            "  {}  {} {} {} {} {} {} {} {} {} {}",
            paint_stdout(pad_left("Shape", w.shape), bold),
            sep,
            paint_stdout(pad_right("MT(µs)", w.mt_us), bold),
            sep,
            paint_stdout(pad_right(&format!("Ref({metric})"), w.ref_perf), bold),
            sep,
            paint_stdout(pad_right(&format!("MT({metric})"), w.mt_perf), bold),
            sep,
            paint_stdout(pad_right("MT%", w.pct), bold),
            sep,
            paint_stdout(pad_right("GFLOP/s", w.gflops), bold),
        );
        if self.show_correctness {
            hdr.push_str(&format!(" {} {}", sep, paint_stdout(pad_right("ok", w.ck), bold)));
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
        // -v/-vv: reference latency + roofline (%peak, AI) + occupancy/register profile
        if self.verbose >= 1 {
            hdr.push_str(&format!(
                " {} {} {} {} {} {} {} {}",
                sep,
                paint_stdout(pad_right("Ref(µs)", US_COL_W), bold),
                sep,
                paint_stdout(pad_right("%BW", PCT_COL_W), bold),
                sep,
                paint_stdout(pad_right("%FLOP", PCT_COL_W), bold),
                sep,
                paint_stdout(pad_right("AI", AI_COL_W), bold),
            ));
            hdr.push_str(&format!(
                " {} {} {} {} {} {}",
                sep,
                paint_stdout(pad_right("occ%", OCC_COL_W), bold),
                sep,
                paint_stdout(pad_right("regs", REGS_COL_W), bold),
                sep,
                paint_stdout(pad_right("bottleneck", BN_COL_W), bold),
            ));
        }

        // Op line — just the name, no profile
        let op = paint_stdout(result.op_display(), Style::new().fg(Color::Cyan).bold());
        println!("  {op}");
        println!("{hdr}");

        // Columns sharing a " │ " gap: Shape, MT(µs), Ref, MT, MT%, GFLOP/s (+ok).
        let n_cols: usize = if self.show_correctness { 7 } else { 6 };
        let gaps = (n_cols.saturating_sub(1)) * 3;
        let timing_cols = if self.verbose >= 2 { 5 + 3 + 5 + 3 + 5 + 3 } else { 0 };
        // -v adds Ref(µs), %BW, %FLOP, AI, then occ/regs/bottleneck — each with a
        // leading " │ " (3 chars).
        let profile_cols = if self.verbose >= 1 {
            (US_COL_W + 3)
                + (PCT_COL_W + 3)
                + (PCT_COL_W + 3)
                + (AI_COL_W + 3)
                + (OCC_COL_W + 3)
                + (REGS_COL_W + 3)
                + (BN_COL_W + 3)
        } else {
            0
        };
        let total_w = 4
            + w.shape
            + w.mt_us
            + gaps
            + w.ref_perf
            + w.mt_perf
            + w.pct
            + w.gflops
            + if self.show_correctness { w.ck } else { 0 }
            + timing_cols
            + profile_cols;
        let sep_line = paint_stdout("─".repeat(total_w), Style::new().fg(Color::BrightBlack).dim());
        println!("  {sep_line}");
    }

    fn print_data_row(&self, result: &OpResult) {
        let metric = self.cur_metric.unwrap_or("perf");
        let w = sub_table_widths(self.term_width, metric, self.show_correctness);

        let shape =
            paint_stdout(pad_left(result.shape(), w.shape), Style::new().fg(Color::BrightWhite));
        let mt_us_cell = fmt_latency(result.mt_us(), w.mt_us, true);
        let ref_s = fmt_perf(result.ref_perf(), metric, "—");
        let mt_s = fmt_perf(result.mt_perf(), metric, "NYI");
        let pct_s = result.pct().map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());

        let ref_cell = style_reference(&pad_right(&ref_s, w.ref_perf), result.ref_perf());
        let mt_cell = style_metaltile(&pad_right(&mt_s, w.mt_perf), result);
        let pct_cell = style_pct(&pad_right(&pct_s, w.pct), result);
        let gflops_cell = fmt_gflops(result.metrics().and_then(|m| m.gflops), w.gflops);
        let sep = col_sep();

        let mut row = format!(
            "  {shape} {sep} {mt_us_cell} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell} {sep} \
             {gflops_cell}"
        );
        if self.show_correctness {
            let ck_icon = correctness_icon(&result.correctness_status());
            let ck_cell =
                style_correctness(&pad_right(&ck_icon, w.ck), result.correctness_status());
            row.push_str(&format!(" {sep} {ck_cell}"));
        }
        if self.verbose >= 2 {
            let t = fmt_timing(result);
            row.push_str(&format!(" {sep} {t}"));
        }
        if self.verbose >= 1 {
            let ref_us_cell = fmt_latency(result.ref_us(), US_COL_W, false);
            let m = result.metrics();
            let bw_cell = fmt_pct(m.and_then(|x| x.pct_peak_bw), PCT_COL_W);
            let flop_cell = fmt_pct(m.and_then(|x| x.pct_peak_flops), PCT_COL_W);
            let ai_cell = fmt_ai(m.and_then(|x| x.arith_intensity), AI_COL_W);
            let occ_cell = fmt_occ(m.and_then(|x| x.occ_pct), OCC_COL_W);
            let regs_cell = fmt_regs(m.and_then(|x| x.regs_per_thread), REGS_COL_W);
            let bn_cell = fmt_bottleneck(m.and_then(|x| x.bottleneck), BN_COL_W);
            row.push_str(&format!(
                " {sep} {ref_us_cell} {sep} {bw_cell} {sep} {flop_cell} {sep} {ai_cell} {sep} \
                 {occ_cell} {sep} {regs_cell} {sep} {bn_cell}"
            ));
        }
        println!("{row}");
    }

    fn flush(&self) { let _ = std::io::stdout().flush(); }
}

// ── Table layout helpers ──────────────────────────────────────────────────

fn term_width() -> usize {
    std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok()).unwrap_or(80).clamp(60, 200)
}

/// Fixed width of the `MT(µs)` / `Ref(µs)` latency columns — holds the header
/// `Ref(µs)` (7 chars) and a value up to ~5 digits plus `.1`.
const US_COL_W: usize = 8;

/// Fixed width of the `GFLOP/s` column (header is 7 chars; values like `12345.6`).
const GFLOPS_COL_W: usize = 8;

/// Fixed width of the roofline `%BW` / `%FLOP` percentage columns (`-v`).
const PCT_COL_W: usize = 6;

/// Fixed width of the arithmetic-intensity `AI` column (`-v`); FLOPs/byte.
const AI_COL_W: usize = 7;

/// Fixed widths of the occupancy / registers / bottleneck profile columns (`-v`).
const OCC_COL_W: usize = 5;
const REGS_COL_W: usize = 4;
const BN_COL_W: usize = 17;

/// Per-(sub)table column widths for the always-on (default) data columns. The
/// verbose `-v`/`-vv` columns use their own fixed widths at their render sites.
#[derive(Clone, Copy)]
struct ColWidths {
    shape: usize,
    mt_us: usize,
    ref_perf: usize,
    mt_perf: usize,
    pct: usize,
    gflops: usize,
    ck: usize,
}

fn sub_table_widths(term_width: usize, metric: &str, show_ck: bool) -> ColWidths {
    let avail = term_width.saturating_sub(2);
    let ref_w = 4 + metric.len() + 2;
    let mt_w = 3 + metric.len() + 2;
    let pct_w = 5;
    let ck_w = if show_ck { 3 } else { 0 };
    // Default data columns: MT(µs), Ref(perf), MT(perf), MT%, GFLOP/s, [ok].
    let rhs = US_COL_W + ref_w + mt_w + pct_w + GFLOPS_COL_W + ck_w;
    // Columns sharing a " │ " gap: Shape, MT(µs), Ref, MT, MT%, GFLOP/s (+ok).
    let n_cols: usize = if show_ck { 7 } else { 6 };
    let gaps = (n_cols.saturating_sub(1)) * 3;
    let shape_w = avail.saturating_sub(rhs + gaps + 2).clamp(8, 42);
    ColWidths {
        shape: shape_w,
        mt_us: US_COL_W,
        ref_perf: ref_w,
        mt_perf: mt_w,
        pct: pct_w,
        gflops: GFLOPS_COL_W,
        ck: ck_w,
    }
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

/// Render a latency cell (microseconds) right-padded to `width`. `primary` (the
/// MT(µs) column) is bold; the secondary Ref(µs) column is plain. A missing
/// value (off-GPU / no reference) renders as a dim "—" rather than `0.0`.
fn fmt_latency(us: Option<f64>, width: usize, primary: bool) -> String {
    match us {
        Some(x) => {
            let style = if primary {
                Style::new().fg(Color::BrightWhite).bold()
            } else {
                Style::new().fg(Color::BrightWhite)
            };
            paint_stdout(pad_right(&format!("{x:.1}"), width), style)
        },
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
    }
}

/// Render a compute-throughput cell (GFLOP/s) right-padded to `width`. Blank
/// (dim "—") for memory-bound kernels that declared no FLOP count. Rendered in
/// the standard BrightWhite value colour (Cyan is reserved for the op/title row,
/// per the #255 CLI palette).
fn fmt_gflops(gflops: Option<f64>, width: usize) -> String {
    match gflops {
        Some(x) =>
            paint_stdout(pad_right(&format!("{x:.1}"), width), Style::new().fg(Color::BrightWhite)),
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
    }
}

/// Render a %-of-peak cell (`-v` roofline). Green ≥80, yellow ≥40, else red;
/// dim "—" when the device specs are unknown. Capped display at one decimal.
fn fmt_pct(pct: Option<f64>, width: usize) -> String {
    match pct {
        Some(x) => {
            let color = if x >= 80.0 {
                Color::Green
            } else if x >= 40.0 {
                Color::Yellow
            } else {
                Color::Red
            };
            paint_stdout(pad_right(&format!("{x:.0}%"), width), Style::new().fg(color))
        },
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
    }
}

/// Render an arithmetic-intensity cell (FLOPs/byte; `-v` roofline). Blank for
/// memory-bound kernels that declared no FLOP count.
fn fmt_ai(ai: Option<f64>, width: usize) -> String {
    match ai {
        Some(x) =>
            paint_stdout(pad_right(&format!("{x:.1}"), width), Style::new().fg(Color::BrightWhite)),
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
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

/// Render the occupancy cell (`-v`): green ≥100, yellow ≥60, else red; dim "—"
/// when no profile was estimated.
fn fmt_occ(occ_pct: Option<f64>, width: usize) -> String {
    match occ_pct {
        Some(p) => {
            let color = if p >= 100.0 {
                Color::Green
            } else if p >= 60.0 {
                Color::Yellow
            } else {
                Color::Red
            };
            paint_stdout(pad_right(&format!("{p:.0}%"), width), Style::new().fg(color).bold())
        },
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
    }
}

/// Render the registers-per-thread cell (`-v`), e.g. `489r`. Dim "—" when no
/// profile was estimated.
fn fmt_regs(regs: Option<usize>, width: usize) -> String {
    match regs {
        Some(r) =>
            paint_stdout(pad_right(&format!("{r}r"), width), Style::new().fg(Color::BrightWhite)),
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
    }
}

/// Render the bottleneck-verdict cell (`-v`): the combined roofline + occupancy
/// classification. Dim "—" when it couldn't be determined.
fn fmt_bottleneck(bottleneck: Option<&str>, width: usize) -> String {
    match bottleneck {
        Some(b) => paint_stdout(pad_right(b, width), Style::new().fg(Color::BrightBlack).dim()),
        None => paint_stdout(pad_right("—", width), Style::new().fg(Color::BrightBlack).dim()),
    }
}

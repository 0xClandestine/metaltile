use std::{cell::RefCell, io::Write, ptr::NonNull};

use metaltile_codegen::msl::MslGenerator;
pub use metaltile_core::dtype::DType;
use metaltile_core::ir::{Kernel, KernelMode};

use crate::stats::BenchStats;
use crate::term::{Color, Style, paint_stdout};

// ── Dtype variant helpers ─────────────────────────────────────────────────────

/// All floating-point dtypes to iterate over in multi-variant benches.
pub const FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];
/// Short names for the three floating-point dtypes, matching MLX convention.
pub const FLOAT_DTYPE_STRS: &[&str] = &["f32", "f16", "bf16"];
/// Integer dtypes supported by MLX elementwise and copy kernels.
pub const INTEGER_DTYPES: &[DType] = &[DType::I32, DType::U32, DType::I8, DType::U8];

pub fn dtype_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::Bool => "bool",
        _ => "?",
    }
}

/// MLX template-name suffix used in kernel instantiation strings.
pub fn mlx_tname(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I32 => "int32",
        DType::U32 => "uint32",
        DType::I8 => "int8",
        DType::U8 => "uint8",
        DType::Bool => "bool_",
        _ => "float32",
    }
}

/// Bytes per element.
pub fn elem_bytes(dt: DType) -> usize {
    match dt {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U8 | DType::Bool | DType::I8 => 1,
        _ => 4,
    }
}

/// Absolute-error tolerance for elementwise op correctness checks.
pub fn dtype_tol(dt: DType) -> f32 {
    match dt {
        DType::F32 => 1e-4,
        // f16 ULP at magnitude ~20 (e.g. exp(3)) is ~0.016, so 1.5e-2 covers one ULP.
        DType::F16 => 1.5e-2,
        // bf16 ULP at magnitude ~17 (e.g. pow(3,2.5)) is ~0.125, so 1.3e-1 covers 1 ULP.
        DType::BF16 => 1.3e-1,
        // Integers are exact — zero tolerance.
        _ => 0.0,
    }
}

/// Absolute-error tolerance for reduction ops (accumulated rounding over many elements).
pub fn dtype_tol_reduce(dt: DType) -> f32 {
    match dt {
        DType::F32 => 1e-3,
        // f16 accumulation of ~512 elements summing to ~224 can have 1 ULP ≈ 0.25 error
        // vs an f32-accumulated reference.
        DType::F16 => 0.5,
        // MT accumulates in float32 (accurate), MLX accumulates in bfloat (lossy).
        // For 16 384 elements summing to ~9 000, BF16 accumulated error ≈ sum * 2^-7 ≈ 70.
        DType::BF16 => 128.0,
        _ => 1e-3,
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 31) as u16) << 15;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant32 = x & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    // Round-to-nearest-even
    let mant16 = mant32 >> 13;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xFFF;
    let round_up = round_bit == 1 && (sticky != 0 || (mant16 & 1) == 1);
    let mant16 = (mant16 + u32::from(round_up)) as u16;
    // Mantissa overflow bumps exponent
    if mant16 > 0x3FF {
        sign | (((exp + 1) as u16) << 10)
    } else {
        sign | ((exp as u16) << 10) | mant16
    }
}

fn f32_to_bf16(v: f32) -> u16 {
    let x = v.to_bits();
    let rounded = x.wrapping_add(0x7FFF).wrapping_add((x >> 16) & 1);
    (rounded >> 16) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp5 = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    if exp5 == 0 {
        return f32::from_bits(sign);
    }
    if exp5 == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exp5 + 112) << 23) | (mantissa << 13))
}

fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

/// Quantize `vals` through `dt` and back to f32 so the cpu_ref uses the same
/// representable values that the GPU will actually receive.
pub fn quantize_roundtrip(vals: &[f32], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => vals.to_vec(),
        DType::F16 => vals.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect(),
        DType::BF16 => vals.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect(),
        DType::I32 => vals.iter().map(|&v| v as i32 as f32).collect(),
        DType::U32 => vals.iter().map(|&v| v as u32 as f32).collect(),
        DType::I8 => vals.iter().map(|&v| v as i8 as f32).collect(),
        DType::U8 => vals.iter().map(|&v| v as u8 as f32).collect(),
        _ => vals.to_vec(),
    }
}

type ResultReporterFn = NonNull<dyn FnMut(&OpResult)>;

thread_local! {
    static RESULT_REPORTER: RefCell<Option<ResultReporterFn>> = RefCell::new(None);
}

pub const DEFAULT_MIN_COSINE_SIM: f32 = 0.999;

/// Result of a numerical equivalence check between the reference and MT kernels.
#[derive(Debug, Clone, Copy)]
pub struct EquivResult {
    /// Number of elements compared.
    pub n_checked: usize,
    /// Maximum absolute element-wise error.
    pub max_abs_err: f32,
    /// Cosine similarity across the compared vectors.
    pub cosine_sim: f32,
    /// True iff all correctness thresholds were satisfied.
    pub passed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EquivTolerance {
    pub max_abs_err: f32,
    pub min_cosine_sim: f32,
}

impl EquivTolerance {
    pub const fn new(max_abs_err: f32, min_cosine_sim: f32) -> Self {
        Self { max_abs_err, min_cosine_sim }
    }
}

/// Compare reference and MT output arrays element-wise.
/// Uses the provided absolute error tolerance plus a cosine-similarity floor.
pub fn check_equiv_with(
    ref_vals: &[f32],
    mt_vals: &[f32],
    tolerance: EquivTolerance,
) -> EquivResult {
    let n = ref_vals.len().min(mt_vals.len());
    let mut max_err = 0.0f32;
    let mut dot = 0.0f64;
    let mut ref_norm_sq = 0.0f64;
    let mut mt_norm_sq = 0.0f64;
    for (&r, &m) in ref_vals[..n].iter().zip(&mt_vals[..n]) {
        let err = (r - m).abs();
        if err > max_err {
            max_err = err;
        }
        let r = r as f64;
        let m = m as f64;
        dot += r * m;
        ref_norm_sq += r * r;
        mt_norm_sq += m * m;
    }

    let cosine_sim = match (ref_norm_sq > 0.0, mt_norm_sq > 0.0) {
        (false, false) => 1.0,
        (false, true) | (true, false) => 0.0,
        (true, true) => {
            let denom = ref_norm_sq.sqrt() * mt_norm_sq.sqrt();
            (dot / denom) as f32
        },
    }
    .clamp(-1.0, 1.0);

    let same_len = ref_vals.len() == mt_vals.len();
    EquivResult {
        n_checked: n,
        max_abs_err: max_err,
        cosine_sim,
        passed: same_len
            && max_err.is_finite()
            && cosine_sim.is_finite()
            && max_err <= tolerance.max_abs_err
            && cosine_sim >= tolerance.min_cosine_sim,
    }
}

/// Compare reference and MT output arrays element-wise.
/// `max_abs_err` is the maximum allowed absolute error; cosine similarity uses
/// the shared default floor to catch gross directional mismatches.
pub fn check_equiv(ref_vals: &[f32], mt_vals: &[f32], max_abs_err: f32) -> EquivResult {
    check_equiv_with(ref_vals, mt_vals, EquivTolerance::new(max_abs_err, DEFAULT_MIN_COSINE_SIM))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrectnessStatus {
    Passed { max_abs_err: f32, cosine_sim: f32 },
    Failed { max_abs_err: f32, cosine_sim: f32 },
    Unchecked,
    Unavailable,
}

#[derive(Debug, Clone, Copy)]
pub struct OpBench {
    op: &'static str,
    metric: &'static str,
}

impl OpBench {
    pub const fn new(op: &'static str, metric: &'static str) -> Self { Self { op, metric } }
    pub const fn op(&self) -> &'static str { self.op }

    pub fn result(
        &self,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
    ) -> OpResult {
        self.result_sub(None::<&str>, shape, ref_perf, mt_perf, equiv)
    }

    /// Like `result()` but with a sub-operation label displayed as "op (subop)".
    pub fn result_sub(
        &self,
        subop: Option<impl Into<String>>,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
    ) -> OpResult {
        self.result_sub_timed(subop, shape, ref_perf, mt_perf, equiv, None, None)
    }

    /// Like `result_sub()` but with optional GPU timing stats for -vv output.
    #[allow(clippy::too_many_arguments)]
    pub fn result_sub_timed(
        &self,
        subop: Option<impl Into<String>>,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
        mt_timing: Option<BenchStats>,
        ref_timing: Option<BenchStats>,
    ) -> OpResult {
        let shape = shape.into();
        if mt_perf.is_some() && equiv.is_none() {
            panic!("implemented benchmark '{}' [{}] is missing correctness", self.op, shape);
        }
        let result = OpResult {
            op: self.op,
            subop: subop.map(|s| s.into()),
            shape,
            metric: self.metric,
            ref_perf,
            mt_perf,
            equiv,
            mt_timing,
            ref_timing,
        };
        report_result(&result);
        result
    }

    pub fn implemented(
        &self,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: f64,
        equiv: EquivResult,
    ) -> OpResult {
        self.result(shape, ref_perf, Some(mt_perf), Some(equiv))
    }

    pub fn nyi(&self, shape: impl Into<String>, ref_perf: Option<f64>) -> OpResult {
        self.result(shape, ref_perf, None, None)
    }
}

pub struct OpResult {
    op: &'static str,
    /// Optional sub-operation displayed as "op (subop)" in the Op column.
    /// Does not affect blank-line grouping — that still uses `op`.
    subop: Option<String>,
    shape: String,
    /// "GFLOPS" or "GB/s"
    metric: &'static str,
    /// Performance of the MLX Metal reference kernel.
    ref_perf: Option<f64>,
    /// Performance of MetalTile-generated kernel; None = not yet implemented.
    mt_perf: Option<f64>,
    /// Numerical equivalence check result.
    equiv: Option<EquivResult>,
    /// GPU timing stats for MetalTile (-vv mode only).
    pub mt_timing: Option<BenchStats>,
    /// GPU timing stats for reference (-vv mode only).
    pub ref_timing: Option<BenchStats>,
}

impl OpResult {
    pub fn op(&self) -> &'static str { self.op }

    /// Rendered op name: "op (subop)" if subop is set, else "op".
    pub fn op_display(&self) -> String {
        match &self.subop {
            Some(s) => format!("{} ({})", self.op, s),
            None => self.op.to_string(),
        }
    }

    pub fn shape(&self) -> &str { &self.shape }

    pub fn metric(&self) -> &'static str { self.metric }

    pub fn ref_perf(&self) -> Option<f64> { self.ref_perf }

    pub fn mt_perf(&self) -> Option<f64> { self.mt_perf }

    pub fn equiv(&self) -> Option<&EquivResult> { self.equiv.as_ref() }

    pub fn pct(&self) -> Option<f64> {
        match (self.ref_perf, self.mt_perf) {
            (Some(r), Some(m)) if r > 0.0 => Some(m / r * 100.0),
            _ => None,
        }
    }

    pub fn correctness_status(&self) -> CorrectnessStatus {
        match (&self.equiv, self.mt_perf) {
            (Some(e), _) if e.passed =>
                CorrectnessStatus::Passed { max_abs_err: e.max_abs_err, cosine_sim: e.cosine_sim },
            (Some(e), _) =>
                CorrectnessStatus::Failed { max_abs_err: e.max_abs_err, cosine_sim: e.cosine_sim },
            (None, Some(_)) => CorrectnessStatus::Unchecked,
            (None, None) => CorrectnessStatus::Unavailable,
        }
    }

    pub fn correctness_cell(&self) -> String {
        match self.correctness_status() {
            CorrectnessStatus::Passed { max_abs_err, .. } =>
                if max_abs_err < 1e-5 {
                    "✓".into()
                } else {
                    format!("✓ {max_abs_err:.2e}")
                },
            CorrectnessStatus::Failed { max_abs_err, cosine_sim } =>
                if cosine_sim < 0.999 {
                    format!("✗ {max_abs_err:.2e} cos={cosine_sim:.3}")
                } else {
                    format!("✗ {max_abs_err:.2e}")
                },
            CorrectnessStatus::Unchecked => "! missing-check".into(),
            CorrectnessStatus::Unavailable => "—".into(),
        }
    }

    pub fn is_unchecked(&self) -> bool {
        matches!(self.correctness_status(), CorrectnessStatus::Unchecked)
    }
}

pub fn validate_results(results: &[OpResult]) -> Result<(), String> {
    let unchecked: Vec<String> = results
        .iter()
        .filter(|r| r.is_unchecked())
        .map(|r| format!("{} [{}]", r.op(), r.shape()))
        .collect();
    if unchecked.is_empty() {
        Ok(())
    } else {
        Err(format!("implemented benchmarks missing correctness checks: {}", unchecked.join(", ")))
    }
}

pub fn print_suite(results: &[OpResult]) {
    validate_results(results).unwrap_or_else(|err| panic!("{err}"));

    let mut printer = SuitePrinter::new(
        results.iter().any(|r| !matches!(r.correctness_status(), CorrectnessStatus::Unavailable)),
    );
    printer.print_batch(results);
    printer.finish();
}

fn report_result(result: &OpResult) {
    RESULT_REPORTER.with(|slot| {
        if let Some(mut reporter) = *slot.borrow() {
            // Safety: the pointer is installed by `set_result_reporter` and restored by its
            // guard before the captured closure can go out of scope.
            unsafe {
                reporter.as_mut()(result);
            }
        }
    });
}

pub struct ResultReporterGuard {
    previous: Option<ResultReporterFn>,
}

impl Drop for ResultReporterGuard {
    fn drop(&mut self) {
        RESULT_REPORTER.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

pub fn set_result_reporter(reporter: &mut dyn FnMut(&OpResult)) -> ResultReporterGuard {
    // SAFETY: The guard restores the previous reporter on drop. The caller's &mut
    // borrow ensures the closure outlives the guard (Rust's borrow checker enforces this
    // at the call site). We erase the lifetime here to satisfy the 'static bound of
    // the thread-local, which is safe because the guard guarantees restoration before
    // the reference could become dangling.
    let reporter: NonNull<dyn FnMut(&OpResult)> =
        unsafe { std::mem::transmute(NonNull::from(reporter)) };
    let previous = RESULT_REPORTER.with(|slot| (*slot.borrow_mut()).replace(reporter));
    ResultReporterGuard { previous }
}


pub struct SuitePrinter {
    show_correctness: bool,
    started: bool,
    last_op_display: Option<String>,
    term_width: usize,
    cur_metric: Option<&'static str>,
    /// Shows profile info (occ%, regs, bottleneck) in op headers when Some.
    profile_map: Option<std::collections::HashMap<String, ProfileRow>>,
    /// Shows timing columns (p95μs, cv%) when > 0 and results carry timing.
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

    pub fn set_verbose(&mut self, v: u8) { self.verbose = v; }

    pub fn set_profile_map(&mut self, m: std::collections::HashMap<String, ProfileRow>) {
        self.profile_map = Some(m);
    }

    pub fn set_term_width(&mut self, w: usize) { self.term_width = w.clamp(60, 200); }

    pub fn print_batch(&mut self, results: &[OpResult]) {
        if results.is_empty() { return; }
        if !self.started { self.started = true; }

        for result in results {
            let new_group = self.last_op_display.as_deref() != Some(&result.op_display());
            if new_group {
                if self.cur_metric != Some(result.metric()) {
                    self.cur_metric = Some(result.metric());
                }
                if self.last_op_display.is_some() { println!(); }
                self.print_op_header(result);
            }
            self.last_op_display = Some(result.op_display());
            self.print_data_row(result);
        }
        self.flush();
    }

    pub fn finish(&mut self) {
        if !self.started { return; }
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
            paint_stdout(&pad_left("Shape", shape_w), bold),
            sep,
            paint_stdout(&pad_right(&format!("Ref({})", metric), ref_w), bold),
            sep,
            paint_stdout(&pad_right(&format!("MT({})", metric), mt_w), bold),
            sep,
            paint_stdout(&pad_right("MT%", pct_w), bold),
        );
        if self.show_correctness {
            hdr.push_str(&format!(
                " {} {}",
                sep,
                paint_stdout(&pad_right("ok", ck_w), bold),
            ));
        }
        if self.verbose >= 2 {
            let pw = 5;  // p95
            let qw = 5;  // p99
            let cw = 5;  // cv%
            hdr.push_str(&format!(
                " {} {} {} {} {} {}",
                sep,
                paint_stdout(&pad_right("p95", pw), bold),
                sep,
                paint_stdout(&pad_right("p99", qw), bold),
                sep,
                paint_stdout(&pad_right("cv%", cw), bold),
            ));
        }

        // Op line: name [profile if available]
        let op = paint_stdout(&result.op_display(), Style::new().fg(Color::Cyan).bold());
        if let Some(ref map) = self.profile_map {
            if let Some(p) = map.get(&result.op_display()) {
                let occ_color = if p.occ_pct >= 100.0 { Color::Green }
                    else if p.occ_pct >= 60.0 { Color::Yellow }
                    else { Color::Red };
                let occ = paint_stdout(format!("{:.1}%", p.occ_pct), Style::new().fg(occ_color).bold());
                let regs = paint_stdout(format!("{}r", p.regs_per_thread), Style::new().fg(Color::BrightWhite));
                let bn = paint_stdout(p.bottleneck, Style::new().fg(Color::BrightBlack).dim());
                println!("  {op} · occ={occ} · regs={regs} · {bn}");
            } else {
                println!("  {op}");
            }
        } else {
            println!("  {op}");
        }
        println!("{hdr}");

        let n_cols: usize = if self.show_correctness { 5 } else { 4 };
        let gaps = (n_cols.saturating_sub(1)) * 3;
        let extra_cols = if self.verbose >= 2 { 5 + 3 + 5 + 3 + 5 + 3 } else { 0 }; // p95 + gap + p99 + gap + cv% + gap
        let total_w = 4 + shape_w + gaps + ref_w + mt_w + pct_w
            + if self.show_correctness { ck_w } else { 0 }
            + extra_cols;
        let sep_line = paint_stdout("─".repeat(total_w), Style::new().fg(Color::BrightBlack).dim());
        println!("  {sep_line}");
    }

    fn print_data_row(&self, result: &OpResult) {
        let metric = self.cur_metric.unwrap_or("perf");
        let (shape_w, ref_w, mt_w, pct_w, ck_w) =
            sub_table_widths(self.term_width, metric, self.show_correctness);

        let shape = paint_stdout(&pad_left(result.shape(), shape_w), Style::new().fg(Color::BrightWhite));
        let ref_s = fmt_perf(result.ref_perf(), metric, "—");
        let mt_s = fmt_perf(result.mt_perf(), metric, "NYI");
        let pct_s = result.pct().map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into());

        let ref_cell = style_reference(&pad_right(&ref_s, ref_w), result.ref_perf());
        let mt_cell = style_metaltile(&pad_right(&mt_s, mt_w), result);
        let pct_cell = style_pct(&pad_right(&pct_s, pct_w), result);
        let sep = col_sep();

        if self.show_correctness {
            let ck_icon = correctness_icon(&result.correctness_status());
            let ck_cell = style_correctness(&pad_right(&ck_icon, ck_w), result.correctness_status());
            if self.verbose >= 2 {
                let t = fmt_timing(result);
                println!("  {shape} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell} {sep} {ck_cell} {sep} {t}");
            } else {
                println!("  {shape} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell} {sep} {ck_cell}");
            }
        } else {
            if self.verbose >= 2 {
                let t = fmt_timing(result);
                println!("  {shape} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell} {sep} {t}");
            } else {
                println!("  {shape} {sep} {ref_cell} {sep} {mt_cell} {sep} {pct_cell}");
            }
        }
    }

    fn flush(&self) { let _ = std::io::stdout().flush(); }
}

fn term_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80)
        .clamp(60, 200)
}

fn sub_table_widths(term_width: usize, metric: &str, show_ck: bool) -> (usize, usize, usize, usize, usize) {
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

fn col_sep() -> String {
    paint_stdout("│", Style::new().fg(Color::BrightBlack).dim())
}

fn pad_left(text: &str, width: usize) -> String {
    format!("{text:<width$}")
}

fn pad_right(text: &str, width: usize) -> String {
    format!("{text:>width$}")
}

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
            let p95 = paint_stdout(format!("{:>5.1}", t.p95_us), Style::new().fg(Color::BrightWhite));
            let p99 = paint_stdout(format!("{:>5.1}", t.p99_us), Style::new().fg(Color::BrightWhite));
            let cv_str = if t.cv_pct > 5.0 {
                paint_stdout(format!("{:>4.1}%", t.cv_pct), Style::new().fg(Color::Yellow).bold())
            } else {
                paint_stdout(format!("{:>4.1}%", t.cv_pct), Style::new().fg(Color::Green))
            };
            format!("{p95} {} {p99} {} {cv_str}", sep, sep)
        }
        _ => {
            let dash = paint_stdout("   —", dim);
            let dash2 = paint_stdout("   —", dim);
            let dash3 = paint_stdout("   —", dim);
            format!("{dash} {} {dash2} {} {dash3}", sep, sep)
        }
    }
}

// ── Shared bench abstractions ─────────────────────────────────────────────────

/// Generate MSL for an elementwise kernel IR produced by `make_ir`.
///
/// Uses default `KernelMode::Elementwise`. `label` is used only in the error message.
pub fn generate_elementwise_msl<F>(make_ir: F, label: &str) -> String
where F: Fn() -> Kernel {
    MslGenerator::default().generate(&make_ir()).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

/// Generate MSL for a reduction kernel IR produced by `make_ir`, setting `Reduction` mode.
///
/// `label` is used only in the error message when code generation fails.
pub fn generate_reduction_msl<F>(make_ir: F, label: &str) -> String
where F: Fn() -> Kernel {
    let mut k = make_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

/// Per-dtype context bundled at the top of every bench function.
pub struct DtypeCtx {
    pub dt: DType,
    /// MLX template-name suffix (e.g. `"float32"`).
    pub tn: &'static str,
    /// Short label used in shape strings (e.g. `"f32"`).
    pub label: &'static str,
    /// Bytes per element.
    pub eb: usize,
    /// Absolute-error tolerance for correctness checks.
    pub tol: f32,
}

impl DtypeCtx {
    /// Context for reduction ops — uses `dtype_tol_reduce`.
    pub fn reduce(dt: DType) -> Self {
        Self {
            dt,
            tn: mlx_tname(dt),
            label: dtype_label(dt),
            eb: elem_bytes(dt),
            tol: dtype_tol_reduce(dt),
        }
    }

    /// Context for elementwise ops — uses `dtype_tol`.
    pub fn elementwise(dt: DType) -> Self {
        Self {
            dt,
            tn: mlx_tname(dt),
            label: dtype_label(dt),
            eb: elem_bytes(dt),
            tol: dtype_tol(dt),
        }
    }
}

/// Emit the standard two-test block for a reduction op.
///
/// Generates:
/// - `msl_generates_for_all_dtypes` — calls `$msl_fn(dt)` for each float dtype
/// - `kernels_compile` (macos only) — compiles the generated MSL
///
/// Usage:
/// ```ignore
/// bench_tests!(msl_fn: layer_norm_msl_for, kernel_name: "mt_layer_norm");
/// ```
#[macro_export]
macro_rules! bench_tests {
    (msl_fn: $msl_fn:ident, kernel_name: $name:expr) => {
        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn msl_generates_for_all_dtypes() {
                for &dt in $crate::bench_types::FLOAT_DTYPES {
                    let msl = $msl_fn(dt);
                    assert!(!msl.trim().is_empty(), "MSL empty for {dt:?}");
                }
            }

            #[cfg(target_os = "macos")]
            #[test]
            fn kernels_compile() {
                // NOTE: GpuRunner is not available in metaltile-std.
                // This test is only meaningful in metaltile-bench or metaltile-cli.
                // The MSL generation test above covers the pure path.
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::{CorrectnessStatus, EquivResult, OpBench, OpResult, check_equiv, validate_results};

    fn sample_result(mt_perf: Option<f64>, equiv: Option<EquivResult>) -> OpResult {
        OpBench::new("sample", "GB/s").result("shape", Some(1.0), mt_perf, equiv)
    }

    #[test]
    fn correctness_status_distinguishes_unchecked_from_unavailable() {
        let unchecked = OpResult {
            op: "sample",
            subop: None,
            shape: "shape".into(),
            metric: "GB/s",
            ref_perf: Some(1.0),
            mt_perf: Some(2.0),
            equiv: None,
            mt_timing: None,
            ref_timing: None,
        };
        let unavailable = sample_result(None, None);
        assert_eq!(unchecked.correctness_status(), CorrectnessStatus::Unchecked);
        assert_eq!(unchecked.correctness_cell(), "! missing-check");
        assert!(unchecked.is_unchecked());
        assert_eq!(unavailable.correctness_status(), CorrectnessStatus::Unavailable);
        assert_eq!(unavailable.correctness_cell(), "—");
    }

    #[test]
    fn check_equiv_reports_cosine_similarity() {
        let equiv = check_equiv(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.001], 1e-2);
        assert_eq!(equiv.n_checked, 3);
        assert!(equiv.passed);
        assert!(equiv.cosine_sim > 0.999_999);
        assert!(equiv.max_abs_err > 0.0);
    }

    #[test]
    fn correctness_status_formats_checked_results() {
        let passed = sample_result(
            Some(2.0),
            Some(EquivResult { n_checked: 16, max_abs_err: 0.0, cosine_sim: 1.0, passed: true }),
        );
        let failed = sample_result(
            Some(2.0),
            Some(EquivResult { n_checked: 16, max_abs_err: 1.5, cosine_sim: 0.5, passed: false }),
        );

        assert_eq!(passed.correctness_status(), CorrectnessStatus::Passed {
            max_abs_err: 0.0,
            cosine_sim: 1.0
        });
        assert_eq!(passed.correctness_cell(), "✓");
        assert_eq!(failed.correctness_status(), CorrectnessStatus::Failed {
            max_abs_err: 1.5,
            cosine_sim: 0.5
        });
        assert_eq!(failed.correctness_cell(), "✗ 1.50e0 cos=0.500");
    }

    #[test]
    #[should_panic(expected = "missing correctness")]
    fn op_bench_rejects_implemented_row_without_correctness() {
        let _ = OpBench::new("sample", "GB/s").result("shape", Some(1.0), Some(2.0), None);
    }

    #[test]
    fn validation_reports_unchecked_rows() {
        let unchecked = OpResult {
            op: "sample",
            subop: None,
            shape: "shape".into(),
            metric: "GB/s",
            ref_perf: Some(1.0),
            mt_perf: Some(2.0),
            equiv: None,
            mt_timing: None,
            ref_timing: None,
        };
        let err = validate_results(&[unchecked]).expect_err("unchecked rows should fail");
        assert!(err.contains("sample [shape]"));
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! In-process bench result types for the Phase-1 CLI bench runner.
//!
//! These types exist solely to support `cmd/bench.rs`'s in-process runner and
//! the `SuitePrinter::print_batch` display path. They have no place in
//! `metaltile-std` — the kernel stdlib should know nothing about how the CLI
//! renders results.
//!
//! Phase-2 target: delete this module entirely once `cmd/bench.rs` is replaced
//! by driving `RunnerHarness` and rendering `protocol::BenchResult` lines.

use std::{cell::RefCell, ptr::NonNull};

use metaltile::runner::BenchStats;
use metaltile_core::dtype::DType;

// ── Dtype helper ──────────────────────────────────────────────────────────────

/// Short label for a dtype, e.g. `"f32"`, `"bf16"`.
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

// ── Correctness ───────────────────────────────────────────────────────────────

/// Result of a numerical equivalence check between the reference and MT kernels.
#[derive(Debug, Clone, Copy)]
pub struct EquivResult {
    pub n_checked: usize,
    pub max_abs_err: f32,
    pub cosine_sim: f32,
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

pub const DEFAULT_MIN_COSINE_SIM: f32 = 0.999;

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

// ── Derived metrics ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DerivedMetrics {
    pub gflops: Option<f64>,
    pub pct_peak_bw: Option<f64>,
    pub pct_peak_flops: Option<f64>,
    pub arith_intensity: Option<f64>,
    pub occ_pct: Option<f64>,
    pub regs_per_thread: Option<usize>,
    pub bottleneck: Option<&'static str>,
}

// ── OpResult / OpBench ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct OpResultExtras {
    pub mt_timing: Option<BenchStats>,
    pub ref_timing: Option<BenchStats>,
    pub metrics: Option<DerivedMetrics>,
}

#[derive(Debug, Clone, Copy)]
pub struct OpBench {
    op: &'static str,
    metric: &'static str,
    legacy: bool,
}

impl OpBench {
    pub const fn new(op: &'static str, metric: &'static str) -> Self {
        Self { op, metric, legacy: false }
    }

    pub const fn legacy(mut self, legacy: bool) -> Self {
        self.legacy = legacy;
        self
    }

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
        self.result_with_extras(subop, shape, ref_perf, mt_perf, equiv, OpResultExtras {
            mt_timing,
            ref_timing,
            metrics: None,
        })
    }

    pub fn result_with_extras(
        &self,
        subop: Option<impl Into<String>>,
        shape: impl Into<String>,
        ref_perf: Option<f64>,
        mt_perf: Option<f64>,
        equiv: Option<EquivResult>,
        extras: OpResultExtras,
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
            mt_timing: extras.mt_timing,
            ref_timing: extras.ref_timing,
            metrics: extras.metrics,
            legacy: self.legacy,
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
    subop: Option<String>,
    shape: String,
    metric: &'static str,
    ref_perf: Option<f64>,
    mt_perf: Option<f64>,
    equiv: Option<EquivResult>,
    pub mt_timing: Option<BenchStats>,
    pub ref_timing: Option<BenchStats>,
    metrics: Option<DerivedMetrics>,
    legacy: bool,
}

impl OpResult {
    pub fn op(&self) -> &'static str { self.op }
    pub fn subop(&self) -> Option<&str> { self.subop.as_deref() }

    pub fn op_display(&self) -> String {
        let base = match &self.subop {
            Some(s) => format!("{} ({})", self.op, s),
            None => self.op.to_string(),
        };
        if self.legacy { format!("{base} (legacy)") } else { base }
    }

    pub fn is_legacy(&self) -> bool { self.legacy }
    pub fn shape(&self) -> &str { &self.shape }
    pub fn metric(&self) -> &'static str { self.metric }
    pub fn ref_perf(&self) -> Option<f64> { self.ref_perf }
    pub fn mt_perf(&self) -> Option<f64> { self.mt_perf }
    pub fn equiv(&self) -> Option<&EquivResult> { self.equiv.as_ref() }

    pub fn mt_us(&self) -> Option<f64> {
        self.mt_timing.as_ref().filter(|t| t.is_valid()).map(|t| t.min_us)
    }

    pub fn ref_us(&self) -> Option<f64> {
        self.ref_timing.as_ref().filter(|t| t.is_valid()).map(|t| t.min_us)
    }

    pub fn metrics(&self) -> Option<&DerivedMetrics> { self.metrics.as_ref() }

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

// ── Result reporter (thread-local sink for the bench runner) ──────────────────

type ResultReporterFn = NonNull<dyn FnMut(&OpResult)>;

thread_local! {
    static RESULT_REPORTER: RefCell<Option<ResultReporterFn>> = RefCell::new(None);
}

fn report_result(result: &OpResult) {
    RESULT_REPORTER.with(|slot| {
        if let Some(mut reporter) = *slot.borrow() {
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
    let reporter: NonNull<dyn FnMut(&OpResult)> =
        unsafe { std::mem::transmute(NonNull::from(reporter)) };
    let previous = RESULT_REPORTER.with(|slot| (*slot.borrow_mut()).replace(reporter));
    ResultReporterGuard { previous }
}

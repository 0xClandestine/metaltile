//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
use std::{cell::RefCell, ptr::NonNull};

use metaltile::{harness::bench::BenchBuffer, runner::BenchStats};
use metaltile_codegen::msl::MslGenerator;
pub use metaltile_core::dtype::DType;
use metaltile_core::ir::{Kernel, KernelMode};

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

/// Deterministic, range-controlled input distributions for A/B reference
/// comparisons.
///
/// Random bytes reinterpreted as floats overflow transcendentals (exp, sinh) to
/// inf/nan and fall outside restricted domains (log needs > 0, asin needs
/// `|x| ≤ 1`), which would make every MT-vs-reference comparison spuriously
/// fail. So a reference-compared bench seeds its input from a small repeating
/// pattern inside the op's valid domain (mirrors the legacy `BufInit`).
/// Throughput is data-independent, so this does not perturb the GB/s figures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputDomain {
    /// Mixed signs around zero: `[-3, -1.5, -0.5, 0, 0.25, 0.75, 1.5, 3]`.
    Signed,
    /// Strictly positive `0.25..=4.0` — for `log`/`sqrt`/`rsqrt`/`acosh` domains.
    Positive,
    /// Inside the unit interval: `[-0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9]` — for
    /// `asin`/`acos`/`atanh`/`erfinv`.
    Unit,
    /// Small positive `1e-4..=1.6e-3` — for long reductions (`sum`/`prod` over
    /// millions of elements) so the accumulated result stays finite in f16
    /// (a `sum` of millions of `Positive` values overflows; a `prod` blows up to
    /// inf). `sum` lands in the tens of thousands; `prod` underflows cleanly to
    /// 0 on both kernels.
    Tiny,
}

impl InputDomain {
    /// The deterministic value at flat index `i`.
    pub fn value(self, i: usize) -> f32 {
        match self {
            InputDomain::Signed => [-3.0, -1.5, -0.5, 0.0, 0.25, 0.75, 1.5, 3.0][i % 8],
            InputDomain::Positive => 0.25 + (i % 16) as f32 * 0.25,
            InputDomain::Unit => [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9][i % 7],
            InputDomain::Tiny => 1e-4 + (i % 16) as f32 * 1e-4,
        }
    }
}

/// Build a `BenchBuffer` of `n` elements filled with `domain`'s deterministic
/// pattern, packed for `dt`.
///
/// Use for the **input** of a reference-compared bench so MetalTile and the
/// reference kernel see identical, in-domain data (the runner shares this exact
/// buffer with the reference by name).
pub fn input_buffer(name: &str, n: usize, dt: DType, domain: InputDomain) -> BenchBuffer {
    // Generate lazily: a `BenchSetup` is often built only to read its kernel IR
    // (codegen-consistency tests, `tile build`), and an eager 64M-element fill
    // would materialise ~256 MB per call for nothing. The bytes are produced
    // only when the bench actually runs (`initial_bytes`).
    BenchBuffer::lazy(
        name,
        n,
        dt,
        std::sync::Arc::new(move || {
            let vals: Vec<f32> = (0..n).map(|i| domain.value(i)).collect();
            crate::utils::pack_f32(&vals, dt)
        }),
    )
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

/// Derived performance metrics computed by the runner from the raw timing,
/// bytes-moved, FLOP count, and device peak specs. Everything is `Option` so a
/// kernel that supplies no FLOP count (memory-bound) or runs on an unknown device
/// simply leaves the corresponding cells blank — the columns/JSON fields are
/// **additive**, never replacing the existing GB/s figures.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DerivedMetrics {
    /// Compute throughput in GFLOP/s (`flops ÷ min latency`). `None` unless the
    /// bench annotated a FLOP count via [`BenchSetup::flops`](metaltile_core::bench::BenchSetup::flops).
    pub gflops: Option<f64>,
    /// Achieved bandwidth as a percentage of the device's peak DRAM bandwidth.
    pub pct_peak_bw: Option<f64>,
    /// Achieved compute as a percentage of the device's peak compute (for the
    /// kernel's dtype/engine). `None` unless both `gflops` and device specs exist.
    pub pct_peak_flops: Option<f64>,
    /// Arithmetic intensity in FLOPs/byte (`flops ÷ bytes_moved`) — places the
    /// kernel on the roofline (left of the ridge ⇒ memory-bound, right ⇒ compute).
    pub arith_intensity: Option<f64>,
    /// Estimated SIMD occupancy (%), from the CPU-side occupancy pass.
    pub occ_pct: Option<f64>,
    /// Estimated registers per thread, from the CPU-side register pass.
    pub regs_per_thread: Option<usize>,
    /// One-word bottleneck verdict (`memory-bound`, `compute-bound`,
    /// `occupancy-limited`, `register-limited`, `latency-bound`), combining the
    /// roofline position with the occupancy/register signals.
    pub bottleneck: Option<&'static str>,
}

/// Build the wire-format [`ProfileInfo`](metaltile_core::protocol::ProfileInfo)
/// from the in-process [`DerivedMetrics`]. This is the single source of truth
/// for the metric schema: the future `__tile_runner` subprocess emits a
/// `ProtocolMessage::BenchResult` whose `profile` is produced via this
/// conversion, so the CLI renders the same numbers whether it computed them
/// in-process (Phase 1) or parsed them off the wire (Phase 2).
impl From<&DerivedMetrics> for metaltile_core::protocol::ProfileInfo {
    fn from(m: &DerivedMetrics) -> Self {
        Self {
            gflops: m.gflops,
            pct_peak_bw: m.pct_peak_bw,
            pct_peak_flops: m.pct_peak_flops,
            arith_intensity: m.arith_intensity,
            occ_pct: m.occ_pct,
            regs_per_thread: m.regs_per_thread.map(|r| r as u32),
            bottleneck: m.bottleneck.map(str::to_string),
        }
    }
}

/// Optional per-run extras threaded into an [`OpResult`] alongside the headline
/// perf figures: GPU timing distributions and the derived roofline metrics. Kept
/// in one struct so [`OpBench::result_with_extras`] doesn't grow an unwieldy
/// argument list as new metrics are added.
#[derive(Debug, Clone, Default)]
pub struct OpResultExtras {
    /// GPU timing stats for the MetalTile kernel (latency, percentiles, cv%).
    pub mt_timing: Option<BenchStats>,
    /// GPU timing stats for the reference kernel, when an A/B ran.
    pub ref_timing: Option<BenchStats>,
    /// Derived roofline / throughput metrics.
    pub metrics: Option<DerivedMetrics>,
}

#[derive(Debug, Clone, Copy)]
pub struct OpBench {
    op: &'static str,
    metric: &'static str,
    /// When true, rows are tagged "(legacy)" — used during migration to mark a
    /// `#[kernel(bench(...))]` whose kernel also has a new `#[bench]` registration.
    legacy: bool,
}

impl OpBench {
    pub const fn new(op: &'static str, metric: &'static str) -> Self {
        Self { op, metric, legacy: false }
    }

    /// Mark rows produced by this `OpBench` as legacy (tagged "(legacy)").
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
        self.result_with_extras(subop, shape, ref_perf, mt_perf, equiv, OpResultExtras {
            mt_timing,
            ref_timing,
            metrics: None,
        })
    }

    /// The fullest constructor: like `result_sub_timed` but also carries the
    /// derived roofline/throughput metrics (and bundles the timing stats) in an
    /// [`OpResultExtras`]. All other `result*` helpers funnel through here, so the
    /// single `report_result` call always sees a fully-populated row.
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
    /// Derived roofline / throughput metrics (GFLOP/s, %-of-peak, arithmetic
    /// intensity, occupancy, bottleneck). `None` until the runner computes them.
    metrics: Option<DerivedMetrics>,
    /// When true, the op label renders with a "(legacy)" suffix. Set for a
    /// `#[kernel(bench(...))]` whose kernel also has a new `#[bench]`, so the
    /// old and new rows are visually distinct during migration.
    legacy: bool,
}

impl OpResult {
    pub fn op(&self) -> &'static str { self.op }

    /// Raw sub-operation label, if set. Consumers that need the combined
    /// "op (subop)" display string should call [`op_display`] instead.
    pub fn subop(&self) -> Option<&str> { self.subop.as_deref() }

    /// Rendered op name: "op (subop)" if subop is set, else "op", with a
    /// trailing " (legacy)" when this row is a superseded legacy registration.
    pub fn op_display(&self) -> String {
        let base = match &self.subop {
            Some(s) => format!("{} ({})", self.op, s),
            None => self.op.to_string(),
        };
        if self.legacy { format!("{base} (legacy)") } else { base }
    }

    /// Whether this row is a superseded legacy registration.
    pub fn is_legacy(&self) -> bool { self.legacy }

    pub fn shape(&self) -> &str { &self.shape }

    pub fn metric(&self) -> &'static str { self.metric }

    pub fn ref_perf(&self) -> Option<f64> { self.ref_perf }

    pub fn mt_perf(&self) -> Option<f64> { self.mt_perf }

    pub fn equiv(&self) -> Option<&EquivResult> { self.equiv.as_ref() }

    /// MetalTile wall-clock latency in microseconds (the `min` sample — the
    /// steady-state runtime, matching the GB/s convention). Derived from the
    /// threaded `mt_timing`, so there is no separately-stored latency to drift.
    pub fn mt_us(&self) -> Option<f64> {
        self.mt_timing.as_ref().filter(|t| t.is_valid()).map(|t| t.min_us)
    }

    /// Reference kernel wall-clock latency in microseconds, when an A/B ran.
    pub fn ref_us(&self) -> Option<f64> {
        self.ref_timing.as_ref().filter(|t| t.is_valid()).map(|t| t.min_us)
    }

    /// Derived roofline / throughput metrics, if the runner computed them.
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
    use super::{
        CorrectnessStatus,
        DType,
        EquivResult,
        InputDomain,
        OpBench,
        OpResult,
        check_equiv,
        input_buffer,
        validate_results,
    };

    fn sample_result(mt_perf: Option<f64>, equiv: Option<EquivResult>) -> OpResult {
        OpBench::new("sample", "GB/s").result("shape", Some(1.0), mt_perf, equiv)
    }

    #[test]
    fn input_domains_stay_in_their_valid_range() {
        // Signed straddles zero; magnitudes bounded so exp/sinh don't overflow.
        for i in 0..64 {
            let v = InputDomain::Signed.value(i);
            assert!((-3.0..=3.0).contains(&v));
        }
        // Positive is strictly > 0 (safe for log/sqrt/division denominators).
        for i in 0..64 {
            assert!(InputDomain::Positive.value(i) > 0.0);
        }
        // Unit stays inside [-1, 1] (asin/acos/atanh domain).
        for i in 0..64 {
            assert!(InputDomain::Unit.value(i).abs() <= 1.0);
        }
        // Tiny is small and positive so long reductions stay finite.
        for i in 0..64 {
            let v = InputDomain::Tiny.value(i);
            assert!(v > 0.0 && v < 1e-2);
        }
    }

    #[test]
    fn input_domains_are_deterministic_and_periodic() {
        // The same index always yields the same value (so the MT input and the
        // reference input — generated independently — are byte-identical), and
        // the pattern repeats so a bounded compare prefix exercises every value.
        for d in [InputDomain::Signed, InputDomain::Positive, InputDomain::Unit, InputDomain::Tiny]
        {
            assert_eq!(d.value(0), d.value(0));
            assert_eq!(d.value(3), d.value(3 + 16 * 7)); // 112 is a common multiple of 8/16/7
        }
    }

    #[test]
    fn input_buffer_packs_expected_width_and_round_trips_f32() {
        let buf = input_buffer("a", 8, DType::F32, InputDomain::Signed);
        assert_eq!(buf.name(), "a");
        assert_eq!(buf.len(), 8);
        // f32 bytes round-trip to the domain pattern.
        let bytes = buf.initial_bytes();
        assert_eq!(bytes.len(), 8 * 4);
        let v0 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(v0, InputDomain::Signed.value(0));
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
            metrics: None,
            legacy: false,
        };
        let unavailable = sample_result(None, None);
        assert_eq!(unchecked.correctness_status(), CorrectnessStatus::Unchecked);
        assert_eq!(unchecked.correctness_cell(), "! missing-check");
        assert!(unchecked.is_unchecked());
        assert_eq!(unavailable.correctness_status(), CorrectnessStatus::Unavailable);
        assert_eq!(unavailable.correctness_cell(), "—");
    }

    #[test]
    fn mt_us_and_ref_us_surface_the_min_sample() {
        use metaltile::runner::BenchStats;
        let mt = BenchStats::from_samples(vec![120.0, 130.0, 250.0]);
        let reference = BenchStats::from_samples(vec![80.0, 90.0, 100.0]);
        let r = OpBench::new("sample", "GB/s").result_with_extras(
            None::<&str>,
            "shape",
            Some(1.0),
            Some(2.0),
            Some(EquivResult { n_checked: 1, max_abs_err: 0.0, cosine_sim: 1.0, passed: true }),
            super::OpResultExtras {
                mt_timing: Some(mt),
                ref_timing: Some(reference),
                metrics: None,
            },
        );
        // Latency is the min sample (steady state), matching the GB/s convention.
        assert_eq!(r.mt_us(), Some(120.0));
        assert_eq!(r.ref_us(), Some(80.0));
    }

    #[test]
    fn mt_us_is_none_without_timing_or_when_invalid() {
        use metaltile::runner::BenchStats;
        // No timing at all → None.
        let no_timing = sample_result(
            Some(2.0),
            Some(EquivResult { n_checked: 1, max_abs_err: 0.0, cosine_sim: 1.0, passed: true }),
        );
        assert_eq!(no_timing.mt_us(), None);
        assert_eq!(no_timing.ref_us(), None);
        // Off-GPU stub timing (all-zero samples ⇒ !is_valid) → None, not 0.0.
        let stub = BenchStats::from_samples(vec![0.0, 0.0]);
        assert!(!stub.is_valid());
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
            metrics: None,
            legacy: false,
        };
        let err = validate_results(&[unchecked]).expect_err("unchecked rows should fail");
        assert!(err.contains("sample [shape]"));
    }
}

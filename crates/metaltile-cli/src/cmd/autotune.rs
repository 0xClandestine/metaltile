//! `tile autotune` — cache-warming pipeline.
//!
//! Walks the `BenchSpec` inventory, runs `Autotuner::tune` per (kernel,
//! dtype), and flushes the resulting cache to `~/.cache/metaltile/<chip>/`.
//!
//! Two cost models:
//!
//! - **default** — static, occupancy-based. Cheap, populates the cache
//!   sensibly without any GPU dispatch.
//! - **`--measure`** — real GPU timing. For `BenchDispatch::Generic`
//!   kernels (the bulk of the elementwise/reduction inventory),
//!   `BenchDispatch::SdpaVector` (`mt_sdpa_vector`), and
//!   `BenchDispatch::SdpaPrefill` (`mt_sdpa_prefill*`) we compile +
//!   dispatch each candidate via `GpuRunner::measure` and record
//!   median elapsed_us. Other non-Generic dispatch variants fall
//!   back to the static cost model with a per-kernel log note;
//!   wiring the remaining arms is incremental follow-up work.
//!
//! ## Bucket extrapolation
//!
//! Each cache entry's key is `(kernel, dtype, shape_bucket)`, but the
//! winner is selected from a *single* measurement at `spec.shapes[0]`
//! — the rest of the bucket inherits that winner. This is by design:
//! the bucket breaks ([`BUCKET_BREAKS`] in metaltile-runtime) are sized
//! so same-bucket shapes mostly want the same schedule. Multi-shape
//! per cache key is Phase 2.

use std::{
    io::Write,
    path::{Path, PathBuf},
};

use metaltile::{
    MetalTileError,
    autotune::{Autotuner, TrainingRow, TuneConfig},
};
use metaltile_codegen::{
    msl::{MslConfig, MslGenerator},
    passes::{
        self,
        Pass,
        occupancy::{self, Bottleneck},
        schedule::ScheduleConfig,
    },
};
use metaltile_core::{constexpr::ConstExprValues, dtype::DType, ir::KernelMode};
use metaltile_std::{
    runner::{GpuBuffer, GpuRunner, buffer_typed, zeros_typed},
    spec::{BenchDispatch, BenchSpec, ScalarBufSpec, ShapeSpec},
};

use crate::{
    AutotuneArgs,
    matches_filter,
    term::{Color, Style, paint_stderr, paint_stdout},
};

pub fn run(args: &AutotuneArgs) -> Result<(), crate::CliError> {
    let _span = tracing::info_span!(
        "autotune",
        filter = ?args.filter,
        measure = args.measure,
        quick = args.quick,
        shapes = ?args.shapes,
    )
    .entered();

    if args.clear {
        return clear_cache();
    }
    if args.dump {
        return dump_cache();
    }
    if args.list {
        return list_kernels(args.filter.as_deref());
    }
    if let Some(dest) = args.export_training_data.as_deref() {
        return export_training_data(dest);
    }

    // `None` → legacy single-shape behavior (use spec.shapes[0]).
    // `Some(vec)` → tune each N independently; cache key incorporates
    // the N bucket so different Ns produce distinct entries.
    let shape_overrides = parse_shape_overrides(&args.shapes)?;

    // Bring up a GpuRunner only when we actually need one — `--measure`
    // is the only mode that dispatches kernels. Static cost runs
    // headless and works fine on machines without Metal (CI).
    let runner =
        if args.measure { Some(GpuRunner::new().map_err(crate::CliError::GpuInit)?) } else { None };

    let mut tuner = Autotuner::new(Autotuner::default_cache_dir(), /* enabled= */ true);

    let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
    specs.sort_unstable_by_key(|s| (s.op, s.subop));

    println!(
        "{} {}",
        paint_stdout("tile autotune", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(
            format!("· cache: {}", Autotuner::default_cache_dir().display()),
            Style::new().fg(Color::BrightBlack),
        ),
    );
    let mode_note = match (args.measure, args.quick) {
        (true, true) =>
            "cost = GPU-measured median μs (Generic + SdpaVector + SdpaPrefill; --quick: 3 warmup + 11 iters).",
        (true, false) =>
            "cost = GPU-measured median μs (Generic + SdpaVector + SdpaPrefill; 20 warmup + 100 iters); other → static.",
        (false, _) => "cost = static occupancy estimate. Pass --measure for real GPU timing.",
    };
    println!(
        "  {} {}",
        paint_stdout("[mode]", Style::new().fg(Color::BrightBlack).bold()),
        paint_stdout(mode_note, Style::new().fg(Color::BrightBlack)),
    );
    if let Some(list) = shape_overrides.as_deref() {
        let ns = list.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        println!(
            "  {} {}",
            paint_stdout("[shapes]", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(
                format!("tuning N ∈ {{{ns}}} — {} entries per (kernel, dtype)", list.len()),
                Style::new().fg(Color::BrightBlack),
            ),
        );
    }

    let bench_budget = if args.quick { BenchBudget::Quick } else { BenchBudget::Standard };

    let mut tuned = 0usize;
    let mut measured = 0usize;
    let mut estimated = 0usize;
    let mut skipped = 0usize;
    let mut fallbacks = 0usize;
    for spec in specs {
        if !matches_filter(args.filter.as_deref(), spec.kernel_name) {
            continue;
        }
        for &dt in spec.dtypes {
            // Each n_override is one cache entry. None → use the
            // spec's own shape; Some(n) → override per --shapes.
            let overrides: Vec<Option<usize>> = match shape_overrides.as_deref() {
                Some(list) => list.iter().map(|&n| Some(n)).collect(),
                None => vec![None],
            };
            for n_override in overrides {
                match tune_one(&mut tuner, spec, dt, runner.as_ref(), bench_budget, n_override) {
                    Ok(TuneReport { outcome: TuneOutcome::Measured, fallback_configs }) => {
                        tuned += 1;
                        measured += 1;
                        fallbacks += fallback_configs;
                    },
                    Ok(TuneReport { outcome: TuneOutcome::Estimated, fallback_configs }) => {
                        tuned += 1;
                        estimated += 1;
                        fallbacks += fallback_configs;
                    },
                    Err(e) => {
                        skipped += 1;
                        let n_tag = n_override.map(|n| format!(" (N={n})")).unwrap_or_default();
                        eprintln!(
                            "  {} {}{}: {}",
                            paint_stderr("skip", Style::new().fg(Color::Yellow).bold()),
                            paint_stderr(spec.kernel_name, Style::new().fg(Color::BrightWhite)),
                            paint_stderr(n_tag, Style::new().fg(Color::BrightBlack)),
                            paint_stderr(e.to_string(), Style::new().fg(Color::BrightBlack)),
                        );
                    },
                }
            }
        }
    }

    let sep = format!("  {}  ", paint_stdout("·", Style::new().fg(Color::BrightBlack).dim()));
    let fallback_segment = if args.measure {
        format!(
            "{sep}{}",
            paint_stdout(
                format!("{fallbacks} config fallbacks"),
                if fallbacks > 0 {
                    Style::new().fg(Color::Yellow)
                } else {
                    Style::new().fg(Color::BrightBlack)
                },
            ),
        )
    } else {
        String::new()
    };
    println!(
        "\n  {tuned}{sep}{measured}{sep}{estimated}{sep}{skipped}{fallback_segment}{sep}{disk}",
        tuned = paint_stdout(format!("{tuned} tuned"), Style::new().fg(Color::Green).bold(),),
        measured = paint_stdout(
            format!("{measured} measured"),
            if measured > 0 {
                Style::new().fg(Color::Green)
            } else {
                Style::new().fg(Color::BrightBlack)
            },
        ),
        estimated =
            paint_stdout(format!("{estimated} estimated"), Style::new().fg(Color::BrightBlack),),
        skipped = paint_stdout(
            format!("{skipped} skipped"),
            if skipped > 0 {
                Style::new().fg(Color::Yellow).bold()
            } else {
                Style::new().fg(Color::BrightBlack)
            },
        ),
        disk = paint_stdout(
            format!("{} entries on disk", tuner.cache().len()),
            Style::new().fg(Color::Cyan).bold(),
        ),
    );

    Ok(())
}

/// How many warmup + measure iterations `--measure` runs per candidate.
///
/// `Standard` (20 warmup + 100 iters) is the Playbook minimum for
/// JIT-cache-warm medians. `Quick` (3 + 11) is a triage mode opted into
/// via `--quick`; it gets the search done in seconds but the resulting
/// medians are noisy and shouldn't be persisted to a long-lived cache.
#[derive(Debug, Clone, Copy)]
enum BenchBudget {
    Standard,
    Quick,
}

impl BenchBudget {
    fn iters(self) -> (usize, usize) {
        match self {
            BenchBudget::Standard => (20, 100),
            BenchBudget::Quick => (3, 11),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TuneOutcome {
    Measured,
    Estimated,
}

#[derive(Debug, Clone, Copy)]
struct TuneReport {
    outcome: TuneOutcome,
    /// How many config candidates in this sweep fell back to
    /// `static_cost` because their measure call errored. Surfaced in
    /// the run-end summary so the user knows when `--measure` quietly
    /// degraded.
    fallback_configs: usize,
}

/// Strategy for measuring one candidate config. The real impl is
/// [`GpuMeasurer`]; tests inject a fake to cover the orchestration
/// (outcome flipping, fallback counting, no-measurer path) without
/// needing a Metal device.
trait CandidateMeasurer {
    fn measure(&self, cfg: &TuneConfig, budget: BenchBudget) -> Result<f64, String>;
}

struct GpuMeasurer<'a> {
    runner: &'a GpuRunner,
    spec: &'a BenchSpec,
    dt: DType,
    mode: KernelMode,
    shape: &'a ShapeSpec,
    n: usize,
    b: usize,
    bufs: Vec<GpuBuffer>,
}

impl<'a> GpuMeasurer<'a> {
    /// Build the buffer set and freeze the dispatch (`n`, `b`) for this
    /// shape. Allocates once — the candidate sweep reuses it across
    /// every config. `n_override` lets `--shapes` retarget the same
    /// spec at multiple cache buckets in one run; when `None`, falls
    /// back to `spec.shapes[0].n`.
    fn new(
        runner: &'a GpuRunner,
        spec: &'a BenchSpec,
        dt: DType,
        mode: KernelMode,
        n_override: Option<usize>,
    ) -> Self {
        let shape = &spec.shapes[0];
        let n = n_override.unwrap_or(shape.n);
        let b = shape.b.max(1);
        let bufs = build_generic_buffers(runner, shape, n, b, dt);
        Self { runner, spec, dt, mode, shape, n, b, bufs }
    }
}

impl CandidateMeasurer for GpuMeasurer<'_> {
    fn measure(&self, cfg: &TuneConfig, budget: BenchBudget) -> Result<f64, String> {
        measure_generic(
            self.runner,
            self.spec,
            self.dt,
            self.mode,
            self.shape,
            self.n,
            self.b,
            cfg,
            &self.bufs,
            budget,
        )
    }
}

/// Measurer for `BenchDispatch::SdpaVector` (`mt_sdpa_vector`).
/// Mirrors `run_spec::run_sdpa_vector`'s buffer recipe + dispatch
/// geometry: 8 buffers (Q, K, V, out, head_dim, n_kv, gqa, scale) and
/// one threadgroup per Q head at the kernel's hardcoded TPG. The
/// candidate config's `threads.0` is ignored — the `SdpaVector`
/// family clamps it to the hardcoded value, so the measurer trusts
/// the dispatch params and not the sweep.
struct SdpaVectorMeasurer<'a> {
    runner: &'a GpuRunner,
    spec: &'a BenchSpec,
    dt: DType,
    n_q_heads: usize,
    tpg: usize,
    bufs: Vec<GpuBuffer>,
}

impl<'a> SdpaVectorMeasurer<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        runner: &'a GpuRunner,
        spec: &'a BenchSpec,
        dt: DType,
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        tpg: usize,
    ) -> Result<Self, String> {
        if !n_q_heads.is_multiple_of(gqa_factor) {
            return Err(format!(
                "SdpaVector: n_q_heads ({n_q_heads}) must be divisible by gqa_factor ({gqa_factor})",
            ));
        }
        let n_kv_heads = n_q_heads / gqa_factor;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Same value pattern run_sdpa_vector uses — modest range, no
        // zero shortcut, no NaN landmines. Sized to the largest tensor
        // (K/V at head_dim × n_kv × n_kv_heads) and aliased into the
        // smaller tensors via prefix slicing.
        let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
        let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
        let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
        let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
        let out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
        let hd_buf = runner.buffer_u32(head_dim as u32);
        let n_buf = runner.buffer_u32(n_kv as u32);
        let gqa_buf = runner.buffer_u32(gqa_factor as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);

        Ok(Self {
            runner,
            spec,
            dt,
            n_q_heads,
            tpg,
            bufs: vec![q_buf, k_buf, v_buf, out_buf, hd_buf, n_buf, gqa_buf, sc_buf],
        })
    }
}

impl CandidateMeasurer for SdpaVectorMeasurer<'_> {
    fn measure(&self, cfg: &TuneConfig, budget: BenchBudget) -> Result<f64, String> {
        let (warmup, iters) = budget.iters();

        let mut k = (self.spec.kernel_ir)(self.dt);
        k.mode = KernelMode::Reduction;

        // Pin `expected_tpg` to the kernel's hardcoded TPG, NOT
        // `cfg.threads.0`. The SdpaVector family already keeps
        // `threads = (1024, 1, 1)`, but be defensive: a future config
        // generator drift mustn't silently miscompile us.
        let msl_cfg = MslConfig { expected_tpg: Some(self.tpg as u32), ..MslConfig::default() };
        let generator = MslGenerator::new(msl_cfg).with_schedule_override(tune_to_schedule(cfg));
        let msl = generator.generate(&k).map_err(|e| format!("MSL gen: {e}"))?;
        let compiled = self
            .runner
            .compile(&msl, self.spec.kernel_name)
            .map_err(|e| format!("compile: {e}"))?;

        let buf_refs: Vec<&GpuBuffer> = self.bufs.iter().collect();
        let tgs = [self.n_q_heads, 1, 1];
        let tpg_arr = [self.tpg, 1, 1];

        self.runner.flush_slc();
        let samples = self.runner.measure(&compiled, &buf_refs, tgs, tpg_arr, warmup, iters);
        median_us(samples)
    }
}

/// Measurer for `BenchDispatch::SdpaPrefill` (`mt_sdpa_prefill*`).
/// Mirrors `run_spec::run_sdpa_prefill`'s 10-buffer recipe + dispatch
/// geometry (`[q_tiles, n_q_heads, batch]` TGs at the kernel's
/// per-spec TPG). The kernel's `bfloat_reinterpret_cast` flag is set
/// the same way run_spec does — keeps the f32→bf16 MFA reinterpret
/// active for the MMA kernels so their compile + dispatch match
/// production behavior.
struct SdpaPrefillMeasurer<'a> {
    runner: &'a GpuRunner,
    spec: &'a BenchSpec,
    dt: DType,
    q_tiles: usize,
    n_q_heads: usize,
    batch: usize,
    tpg: usize,
    bufs: Vec<GpuBuffer>,
}

impl<'a> SdpaPrefillMeasurer<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        runner: &'a GpuRunner,
        spec: &'a BenchSpec,
        dt: DType,
        head_dim: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        batch: usize,
        q_len: usize,
        k_len: usize,
        bq: usize,
        tpg: usize,
    ) -> Result<Self, String> {
        if !n_q_heads.is_multiple_of(gqa_factor) {
            return Err(format!(
                "SdpaPrefill: n_q_heads ({n_q_heads}) must be divisible by gqa_factor ({gqa_factor})",
            ));
        }
        if !q_len.is_multiple_of(bq) {
            return Err(format!("SdpaPrefill: q_len ({q_len}) must be a multiple of bq ({bq})",));
        }
        let n_kv_heads = n_q_heads / gqa_factor;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let qsz = batch * n_q_heads * q_len * head_dim;
        let kvsz = batch * n_kv_heads * k_len * head_dim;
        let vals: Vec<f32> = (0..qsz.max(kvsz)).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        let q_buf = buffer_typed(runner, &vals[..qsz], dt);
        let k_buf = buffer_typed(runner, &vals[..kvsz], dt);
        let v_buf = buffer_typed(runner, &vals[..kvsz], dt);
        let out_buf = zeros_typed(runner, qsz, dt);
        let q_len_buf = runner.buffer_u32(q_len as u32);
        let k_len_buf = runner.buffer_u32(k_len as u32);
        let gqa_buf = runner.buffer_u32(gqa_factor as u32);
        let n_q_heads_buf = runner.buffer_u32(n_q_heads as u32);
        let n_kv_heads_buf = runner.buffer_u32(n_kv_heads as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);

        Ok(Self {
            runner,
            spec,
            dt,
            q_tiles: q_len / bq,
            n_q_heads,
            batch,
            tpg,
            bufs: vec![
                q_buf,
                k_buf,
                v_buf,
                out_buf,
                q_len_buf,
                k_len_buf,
                gqa_buf,
                n_q_heads_buf,
                n_kv_heads_buf,
                sc_buf,
            ],
        })
    }
}

impl CandidateMeasurer for SdpaPrefillMeasurer<'_> {
    fn measure(&self, cfg: &TuneConfig, budget: BenchBudget) -> Result<f64, String> {
        let (warmup, iters) = budget.iters();

        let mut k = (self.spec.kernel_ir)(self.dt);
        // SimdGroup2D mode: the kernel reads tgid_{x,y,z} directly.
        // Match the dispatch geometry the run_spec path uses.
        k.mode = KernelMode::SimdGroup2D;
        // run_spec::run_sdpa_prefill sets this unconditionally for
        // every SdpaPrefill spec — the MMA kernels need it for the
        // f32→bf16 narrowing cast, and the non-MMA variant tolerates
        // it as a no-op. Keep the autotune path bit-identical.
        k.bfloat_reinterpret_cast = true;

        // Pin `expected_tpg` to the dispatch's TPG, not the family's
        // placeholder `cfg.threads.0` — different prefill kernels
        // use different per-tile TPGs, and the kernel constexprs are
        // baked around the dispatch value.
        let msl_cfg = MslConfig { expected_tpg: Some(self.tpg as u32), ..MslConfig::default() };
        let generator = MslGenerator::new(msl_cfg).with_schedule_override(tune_to_schedule(cfg));
        let msl = generator.generate(&k).map_err(|e| format!("MSL gen: {e}"))?;
        let compiled = self
            .runner
            .compile(&msl, self.spec.kernel_name)
            .map_err(|e| format!("compile: {e}"))?;

        let buf_refs: Vec<&GpuBuffer> = self.bufs.iter().collect();
        let tgs = [self.q_tiles, self.n_q_heads, self.batch];
        let tpg_arr = [self.tpg, 1, 1];

        self.runner.flush_slc();
        let samples = self.runner.measure(&compiled, &buf_refs, tgs, tpg_arr, warmup, iters);
        median_us(samples)
    }
}

/// Either measurer impl, picked per dispatch type at construction.
/// Holds the buffers + dispatch geometry the measurer needs. Add a
/// new arm here when wiring more `BenchDispatch` kinds under
/// `--measure`.
enum AnyMeasurer<'a> {
    Generic(GpuMeasurer<'a>),
    SdpaVector(SdpaVectorMeasurer<'a>),
    SdpaPrefill(SdpaPrefillMeasurer<'a>),
}

impl AnyMeasurer<'_> {
    fn as_dyn(&self) -> &dyn CandidateMeasurer {
        match self {
            AnyMeasurer::Generic(m) => m,
            AnyMeasurer::SdpaVector(m) => m,
            AnyMeasurer::SdpaPrefill(m) => m,
        }
    }
}

/// Per-(kernel, dtype) sweep state: drives one candidate through
/// `measure → fallback?` and accumulates the outcome counters that
/// land in the run-end summary. Lifted out of the bench closure so the
/// orchestration can be unit-tested directly (see the `tests` module
/// below).
struct CandidateSearch<'a> {
    measurer: Option<&'a dyn CandidateMeasurer>,
    budget: BenchBudget,
    log_ctx: LogCtx<'a>,
    outcome: TuneOutcome,
    fallback_configs: usize,
}

#[derive(Clone, Copy)]
struct LogCtx<'a> {
    kernel: &'a str,
    dtype: &'a str,
}

impl<'a> CandidateSearch<'a> {
    fn new(
        measurer: Option<&'a dyn CandidateMeasurer>,
        budget: BenchBudget,
        log_ctx: LogCtx<'a>,
    ) -> Self {
        Self { measurer, budget, log_ctx, outcome: TuneOutcome::Estimated, fallback_configs: 0 }
    }

    /// Try `measurer.measure`; on success flip outcome to `Measured`.
    /// On error log at info! (the user passed --measure to *see* why
    /// candidates don't measure), bump the fallback counter, and call
    /// `static_fallback` so the candidate still gets scored.
    fn step(
        &mut self,
        cfg: &TuneConfig,
        static_fallback: impl FnOnce(&TuneConfig) -> Result<f64, MetalTileError>,
    ) -> Result<f64, MetalTileError> {
        if let Some(m) = self.measurer {
            match m.measure(cfg, self.budget) {
                Ok(us) => {
                    self.outcome = TuneOutcome::Measured;
                    return Ok(us);
                },
                Err(e) => {
                    self.fallback_configs += 1;
                    tracing::info!(
                        kernel = self.log_ctx.kernel,
                        dtype = %self.log_ctx.dtype,
                        config = ?cfg,
                        error = %e,
                        "measure failed; falling back to static",
                    );
                },
            }
        }
        static_fallback(cfg)
    }

    fn into_report(self) -> TuneReport {
        TuneReport { outcome: self.outcome, fallback_configs: self.fallback_configs }
    }
}

/// Build a single (kernel, dtype, shape_bucket) → cache entry. Tries
/// GPU measurement first if `runner` is available + the spec uses
/// Generic dispatch; else falls back to the static occupancy estimate.
///
/// `n_override`: when `Some(n)`, retargets `spec.shapes[0]` at the
/// given `n` so one `BenchSpec` can populate multiple cache buckets
/// across a single CLI run (the cache key incorporates the N bucket
/// via `synth_constexprs_for`). When `None`, behaves like the legacy
/// single-shape path.
fn tune_one(
    tuner: &mut Autotuner,
    spec: &BenchSpec,
    dt: DType,
    runner: Option<&GpuRunner>,
    budget: BenchBudget,
    n_override: Option<usize>,
) -> Result<TuneReport, crate::CliError> {
    let family = Autotuner::infer_family(spec.kernel_name);
    // For Generic dispatch the bucket-N comes from `--shapes` (if any)
    // or `spec.shapes[0].n`. For SdpaVector it comes from the
    // dispatch variant's `n_kv` (the sequence length the kernel
    // reduces over) — same idea, different source.
    let effective_n = effective_bucket_n(spec, n_override);
    let constexprs = synth_constexprs_for(spec, effective_n);
    let dtype_label = metaltile_std::bench_types::dtype_label(dt);
    let entry_name = format!("{}@{}", spec.kernel_name, dtype_label);

    let kernel_template = (spec.kernel_ir)(dt);
    let mode = metaltile_std::spec::effective_mode(spec);

    // Hoist the per-shape buffer set out of the bench closure: the
    // candidate sweep walks ~6 configs and the buffer contents don't
    // depend on the config, so allocating once per (kernel, dtype,
    // shape) is both faster and gives a cleaner bench number.
    let measurer = build_measurer(runner, spec, dt, mode, n_override)?;

    let log_ctx = LogCtx { kernel: spec.kernel_name, dtype: dtype_label };
    let mut search = CandidateSearch::new(measurer.as_ref().map(|m| m.as_dyn()), budget, log_ctx);

    let mut bench = |cfg: &TuneConfig| -> Result<f64, MetalTileError> {
        search.step(cfg, |cfg| static_cost(&kernel_template, mode, cfg))
    };

    tuner
        .tune(&entry_name, family, &constexprs, &mut bench)
        .map_err(|e| crate::CliError::Other(format!("{}: {e}", entry_name)))?;

    Ok(search.into_report())
}

/// Build the right `AnyMeasurer` for the dispatch type, or `None` if
/// the spec isn't `--measure`-capable today. `n_override` only
/// applies to `Generic` dispatch (the other arms get their shape
/// from the dispatch variant itself).
fn build_measurer<'a>(
    runner: Option<&'a GpuRunner>,
    spec: &'a BenchSpec,
    dt: DType,
    mode: KernelMode,
    n_override: Option<usize>,
) -> Result<Option<AnyMeasurer<'a>>, crate::CliError> {
    let Some(runner) = runner else { return Ok(None) };
    match spec.dispatch {
        BenchDispatch::Generic if !spec.shapes.is_empty() =>
            Ok(Some(AnyMeasurer::Generic(GpuMeasurer::new(runner, spec, dt, mode, n_override)))),
        BenchDispatch::SdpaVector { head_dim, n_kv, n_q_heads, gqa_factor, batch: _, tpg } => {
            let m = SdpaVectorMeasurer::new(
                runner, spec, dt, head_dim, n_kv, n_q_heads, gqa_factor, tpg,
            )
            .map_err(crate::CliError::Other)?;
            Ok(Some(AnyMeasurer::SdpaVector(m)))
        },
        BenchDispatch::SdpaPrefill {
            head_dim,
            n_q_heads,
            gqa_factor,
            batch,
            q_len,
            k_len,
            bq,
            bk: _,
            wm: _,
            wn: _,
            tpg,
        } => {
            let m = SdpaPrefillMeasurer::new(
                runner, spec, dt, head_dim, n_q_heads, gqa_factor, batch, q_len, k_len, bq, tpg,
            )
            .map_err(crate::CliError::Other)?;
            Ok(Some(AnyMeasurer::SdpaPrefill(m)))
        },
        _ => Ok(None),
    }
}

/// What value should the cache key's `N` bucket reflect for this
/// `(spec, n_override)` pair? Generic dispatch uses `--shapes`
/// override or `spec.shapes[0].n`; SdpaVector uses the dispatch's
/// hardcoded `n_kv`; other arms fall back to `n_override` (which is
/// `None` outside `--shapes`, so the legacy default kicks in).
fn effective_bucket_n(spec: &BenchSpec, n_override: Option<usize>) -> Option<usize> {
    match spec.dispatch {
        BenchDispatch::SdpaVector { n_kv, .. } => Some(n_kv),
        // For prefill, the bucketing dimension that matters most is
        // `k_len` (the KV sequence the tile walks). q_len is usually
        // smaller and more uniform across deployments.
        BenchDispatch::SdpaPrefill { k_len, .. } => Some(k_len),
        BenchDispatch::Generic => n_override,
        _ => n_override,
    }
}

/// Static, GPU-free cost: lower occupancy → higher cost. Stable and
/// deterministic; useful as a fallback when --measure isn't requested
/// (or when the candidate config rejects the kernel at compile time).
fn static_cost(
    kernel_template: &metaltile_core::ir::Kernel,
    mode: KernelMode,
    cfg: &TuneConfig,
) -> Result<f64, MetalTileError> {
    let mut k = kernel_template.clone();
    k.mode = mode;
    passes::run_passes(&mut k, &passes::standard_pipeline())
        .map_err(|e| MetalTileError::Autotune(format!("pipeline failed: {e}")))?;
    let sched: ScheduleConfig = tune_to_schedule(cfg);
    passes::schedule::SchedulePass::new(sched)
        .run(&mut k)
        .map_err(|e| MetalTileError::Autotune(format!("schedule failed: {e}")))?;

    let est = occupancy::estimate_occupancy(&k, cfg.threads.0, None);
    let mut us = 100.0 - est.occupancy_pct;
    if matches!(est.bottleneck, Bottleneck::RegisterLimited) {
        us += 20.0;
    }
    Ok(us)
}

/// Compile + dispatch the kernel under `cfg` and return median
/// elapsed_us across the budget's iters after its warmup. Only handles
/// `BenchDispatch::Generic` shape recipes; other dispatch arms have
/// per-kernel buffer setup and need follow-up wiring.
///
/// `bufs` is the pre-allocated buffer set for `shape` — held across
/// every config in the sweep so we don't pay the round-trip-to-VRAM
/// cost between candidates. See `tune_one`.
#[allow(clippy::too_many_arguments)]
fn measure_generic(
    runner: &GpuRunner,
    spec: &BenchSpec,
    dt: DType,
    mode: KernelMode,
    shape: &ShapeSpec,
    n: usize,
    b: usize,
    cfg: &TuneConfig,
    bufs: &[GpuBuffer],
    budget: BenchBudget,
) -> Result<f64, String> {
    let (warmup, iters) = budget.iters();

    let mut k = (spec.kernel_ir)(dt);
    k.mode = mode;

    let msl_cfg = MslConfig { expected_tpg: Some(cfg.threads.0), ..MslConfig::default() };
    let generator = MslGenerator::new(msl_cfg).with_schedule_override(tune_to_schedule(cfg));
    let msl = generator.generate(&k).map_err(|e| format!("MSL gen: {e}"))?;
    let compiled = runner.compile(&msl, spec.kernel_name).map_err(|e| format!("compile: {e}"))?;

    let buf_refs: Vec<&GpuBuffer> = bufs.iter().collect();

    let tpg_x = cfg.threads.0 as usize;
    let tgs = shape.grid.eval(n, b, tpg_x);
    let tpg = [tpg_x, cfg.threads.1 as usize, cfg.threads.2 as usize];

    // Flush SLC before each kernel's bench burst so DVFS stays at peak
    // — same hygiene `tile bench` uses.
    runner.flush_slc();
    let samples = runner.measure(&compiled, &buf_refs, tgs, tpg, warmup, iters);
    median_us(samples)
}

fn build_generic_buffers(
    runner: &GpuRunner,
    shape: &ShapeSpec,
    n: usize,
    b: usize,
    dt: DType,
) -> Vec<GpuBuffer> {
    let mut out: Vec<GpuBuffer> =
        Vec::with_capacity(shape.tensor_bufs.len() + shape.scalar_bufs.len());
    for buf_spec in shape.tensor_bufs {
        let count = buf_spec.count.resolve(n, b);
        let param_dt = buf_spec.dtype_override.unwrap_or(dt);
        // Honor the spec's BufInit — the authors picked these patterns
        // because zero-paths legitimately hit short-circuits (fmul-by-0,
        // softmax(0), NaN-guards-never-trigger) that bias the bench
        // away from what the kernel actually does in production.
        // `BufInit::Zeros` still flows through this path cheaply.
        let init_data = buf_spec.init.generate(count);
        out.push(buffer_typed(runner, &init_data, param_dt));
    }
    for &sb in shape.scalar_bufs {
        out.push(match sb {
            ScalarBufSpec::U32N => runner.buffer_u32(n as u32),
            ScalarBufSpec::U32B => runner.buffer_u32(b as u32),
            ScalarBufSpec::U64N => runner.buffer_u64(n as u64),
            ScalarBufSpec::U64B => runner.buffer_u64(b as u64),
            ScalarBufSpec::I64B => runner.buffer_i64(b as i64),
        });
    }
    let _ = zeros_typed; // re-exported for symmetry with run_spec; not needed here
    out
}

/// Filter out non-finite (NaN/±∞) and negative samples before sorting.
///
/// `GPUEndTime - GPUStartTime` can come back negative under DVFS edge
/// cases, and a negative-timing run would otherwise win every
/// comparison. NaNs sort as `Equal` under `partial_cmp`, which silently
/// corrupts the median. Caller gets an `Err` if no valid sample
/// remains so the candidate is treated as a measurement failure (and
/// falls through to static_cost), rather than a fake winner.
fn median_us(samples: Vec<f64>) -> Result<f64, String> {
    let mut valid: Vec<f64> = samples.into_iter().filter(|s| s.is_finite() && *s >= 0.0).collect();
    if valid.is_empty() {
        return Err("measure returned no finite, non-negative samples".into());
    }
    valid.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite values above"));
    let n = valid.len();
    Ok(if n % 2 == 1 { valid[n / 2] } else { (valid[n / 2 - 1] + valid[n / 2]) / 2.0 })
}

/// Validate the `--shapes` CLI flag (clap has already parsed it into a
/// `Vec<usize>`). Returns `Ok(None)` when the flag was absent, so the
/// caller can fall back to the legacy single-shape path. Returns
/// `Err` if any N is zero — usize parse already rejects negatives.
fn parse_shape_overrides(shapes: &[usize]) -> Result<Option<Vec<usize>>, crate::CliError> {
    if shapes.is_empty() {
        return Ok(None);
    }
    if let Some(&bad) = shapes.iter().find(|&&n| n == 0) {
        return Err(crate::CliError::Other(format!("--shapes: N must be > 0 (got {bad})")));
    }
    Ok(Some(shapes.to_vec()))
}

/// Build a synthetic `ConstExprValues` for `spec` good enough to bucket.
///
/// `n_override` retargets the N constexpr — `--shapes` uses this to
/// land the same `BenchSpec` in different cache buckets. `B` always
/// comes from `spec.shapes[0]` (multi-shape today is 1-D over `N`).
fn synth_constexprs_for(spec: &BenchSpec, n_override: Option<usize>) -> ConstExprValues {
    let mut ce = ConstExprValues::new();
    let (n, b) = match (spec.shapes.first(), n_override) {
        (Some(first), Some(n)) => (n, first.b.max(1)),
        (Some(first), None) => (first.n, first.b.max(1)),
        (None, Some(n)) => (n, 1),
        (None, None) => (1024, 1),
    };
    ce.insert("N", n);
    ce.insert("B", b);
    ce
}

fn tune_to_schedule(cfg: &TuneConfig) -> ScheduleConfig {
    let tile = if cfg.tile_dims.len() == 3 {
        (cfg.tile_dims[0] as u32, cfg.tile_dims[1] as u32, cfg.tile_dims[2] as u32)
    } else {
        (32, 32, 16)
    };
    ScheduleConfig {
        threads_per_threadgroup: cfg.threads,
        threadgroups_per_grid: (1, 1, 1),
        tile_dims: tile,
        simd_size: 32,
    }
}

/// Emit `Autotuner::export_training_data()` as JSONL — one row per
/// line, in cache-key order so diffs across runs are reviewable.
///
/// `dest` semantics, following the CLI surface:
/// - `""`  → default file `~/.cache/metaltile/training_data.jsonl`
///   (clap's `default_missing_value` fires when the user passes
///   `--export-training-data` with no value).
/// - `"-"` → stdout (good for piping into `jq`, pandas, etc.).
/// - any other value → that path.
fn export_training_data(dest: &str) -> Result<(), crate::CliError> {
    let tuner = Autotuner::new(Autotuner::default_cache_dir(), /* enabled= */ true);
    let rows = tuner.export_training_data();

    if dest == "-" {
        write_training_jsonl(std::io::stdout().lock(), &rows)?;
        return Ok(());
    }

    let path = if dest.is_empty() {
        Autotuner::default_cache_dir().join("training_data.jsonl")
    } else {
        PathBuf::from(dest)
    };
    write_training_jsonl_to_file(&path, &rows)?;
    println!(
        "  {} {} {}",
        paint_stdout("exported", Style::new().fg(Color::Green).bold()),
        paint_stdout(format!("{} rows →", rows.len()), Style::new().fg(Color::BrightBlack),),
        paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightWhite)),
    );
    Ok(())
}

fn write_training_jsonl_to_file(path: &Path, rows: &[TrainingRow]) -> Result<(), crate::CliError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    write_training_jsonl(std::io::BufWriter::new(file), rows)
}

fn write_training_jsonl<W: Write>(mut w: W, rows: &[TrainingRow]) -> Result<(), crate::CliError> {
    for row in rows {
        let line = serde_json::to_string(row)
            .map_err(|e| crate::CliError::Other(format!("serialize training row: {e}")))?;
        writeln!(w, "{line}")?;
    }
    Ok(())
}

fn dump_cache() -> Result<(), crate::CliError> {
    let path = Autotuner::default_cache_dir().join("tuning_cache.json");
    if !path.exists() {
        eprintln!(
            "  {} no cache file at {}",
            paint_stderr("[info]", Style::new().fg(Color::BrightBlack).bold()),
            paint_stderr(path.display().to_string(), Style::new().fg(Color::BrightBlack)),
        );
        return Ok(());
    }
    let s = std::fs::read_to_string(&path)?;
    println!("{s}");
    Ok(())
}

fn clear_cache() -> Result<(), crate::CliError> {
    let path = Autotuner::default_cache_dir().join("tuning_cache.json");
    if path.exists() {
        std::fs::remove_file(&path)?;
        println!(
            "  {} {}",
            paint_stdout("cleared", Style::new().fg(Color::Green).bold()),
            paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightWhite)),
        );
    } else {
        println!(
            "  {} no cache file at {}",
            paint_stdout("[info]", Style::new().fg(Color::BrightBlack).bold()),
            paint_stdout(path.display().to_string(), Style::new().fg(Color::BrightBlack)),
        );
    }
    Ok(())
}

fn list_kernels(filter: Option<&str>) -> Result<(), crate::CliError> {
    let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
    specs.sort_unstable_by_key(|s| (s.kernel_name,));
    let mut count = 0;
    for spec in specs {
        if !matches_filter(filter, spec.kernel_name) {
            continue;
        }
        let family = Autotuner::infer_family(spec.kernel_name);
        let n_configs = family.config_space().len();
        let dispatch_tag = match spec.dispatch {
            BenchDispatch::Generic => "Generic",
            _ => "non-Generic",
        };
        println!(
            "  {}  →  family={:?}  configs={}  dispatch={}",
            paint_stdout(spec.kernel_name, Style::new().fg(Color::Cyan).bold()),
            family,
            n_configs,
            dispatch_tag,
        );
        count += 1;
    }
    if count == 0 {
        eprintln!(
            "  {} no kernels matched",
            paint_stderr("[warn]", Style::new().fg(Color::Yellow).bold()),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_to_schedule_picks_up_3d_tile_dims() {
        let cfg = TuneConfig {
            tile_dims: vec![64, 32, 16],
            threads: (256, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: true,
            use_async_copy: false,
        };
        let s = tune_to_schedule(&cfg);
        assert_eq!(s.tile_dims, (64, 32, 16));
        assert_eq!(s.threads_per_threadgroup, (256, 1, 1));
    }

    #[test]
    fn tune_to_schedule_falls_back_when_tile_dims_missing() {
        let cfg = TuneConfig {
            tile_dims: vec![],
            threads: (512, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: false,
            use_async_copy: false,
        };
        let s = tune_to_schedule(&cfg);
        assert_eq!(s.tile_dims, (32, 32, 16));
        assert_eq!(s.threads_per_threadgroup, (512, 1, 1));
    }

    #[test]
    fn tune_to_schedule_uses_default_when_tile_dims_too_short() {
        let cfg = TuneConfig {
            tile_dims: vec![16],
            threads: (1024, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: false,
            use_async_copy: false,
        };
        let s = tune_to_schedule(&cfg);
        assert_eq!(s.tile_dims, (32, 32, 16));
    }

    #[test]
    fn median_us_returns_middle_for_odd_n() {
        assert_eq!(median_us(vec![3.0, 1.0, 2.0]).unwrap(), 2.0);
    }

    #[test]
    fn median_us_averages_middle_pair_for_even_n() {
        assert_eq!(median_us(vec![4.0, 1.0, 2.0, 3.0]).unwrap(), 2.5);
    }

    #[test]
    fn median_us_filters_nan_samples_before_sorting() {
        // Without filtering, NaN sorts as Equal and corrupts the median.
        let m = median_us(vec![3.0, f64::NAN, 1.0, 2.0]).unwrap();
        assert_eq!(m, 2.0);
    }

    #[test]
    fn median_us_filters_negative_and_infinite_samples() {
        // GPUEndTime - GPUStartTime can come back negative under DVFS;
        // such a sample would otherwise win every comparison.
        let m = median_us(vec![-5.0, f64::INFINITY, 4.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m, 2.5);
    }

    #[test]
    fn median_us_errors_when_all_samples_invalid() {
        let err = median_us(vec![f64::NAN, -1.0, f64::NEG_INFINITY]).unwrap_err();
        assert!(err.contains("finite"), "got: {err}");
    }

    #[test]
    fn median_us_errors_when_input_empty() {
        let err = median_us(vec![]).unwrap_err();
        assert!(err.contains("finite") || err.contains("no"), "got: {err}");
    }

    #[test]
    fn bench_budget_iters_match_documented_values() {
        assert_eq!(BenchBudget::Standard.iters(), (20, 100));
        assert_eq!(BenchBudget::Quick.iters(), (3, 11));
    }

    // ── CandidateSearch orchestration ────────────────────────────────
    //
    // The reviewer's bonus: the bench-closure orchestration (outcome
    // flipping, fallback counting, no-measurer fallthrough) had no
    // unit coverage because `tune_one` needed a real GpuRunner. The
    // CandidateMeasurer seam lets us drive it with canned timings.

    use std::cell::RefCell;

    /// Programmable measurer. Each script entry is the result for the
    /// next `measure` call, in order. `Some(Ok(us))` → success;
    /// `Some(Err(msg))` → failure (caller will fall back); `None` →
    /// the test asked for more calls than scripted, which is a bug.
    struct ScriptedMeasurer {
        script: RefCell<std::collections::VecDeque<Result<f64, String>>>,
        last_budget: std::cell::Cell<Option<BenchBudget>>,
    }
    impl ScriptedMeasurer {
        fn new(script: impl IntoIterator<Item = Result<f64, String>>) -> Self {
            Self {
                script: RefCell::new(script.into_iter().collect()),
                last_budget: std::cell::Cell::new(None),
            }
        }
    }
    impl CandidateMeasurer for ScriptedMeasurer {
        fn measure(&self, _cfg: &TuneConfig, budget: BenchBudget) -> Result<f64, String> {
            self.last_budget.set(Some(budget));
            self.script
                .borrow_mut()
                .pop_front()
                .expect("ScriptedMeasurer: more measure() calls than scripted")
        }
    }

    fn synth_cfg() -> TuneConfig { TuneConfig::default() }
    fn synth_log_ctx() -> LogCtx<'static> { LogCtx { kernel: "test_kernel", dtype: "f32" } }

    #[test]
    fn candidate_search_measure_success_flips_outcome_to_measured() {
        let m = ScriptedMeasurer::new([Ok(5.0)]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Quick, synth_log_ctx());
        let got = s.step(&synth_cfg(), |_| panic!("static_fallback should not run")).unwrap();
        assert_eq!(got, 5.0);
        let r = s.into_report();
        assert_eq!(r.outcome, TuneOutcome::Measured);
        assert_eq!(r.fallback_configs, 0);
    }

    #[test]
    fn candidate_search_measure_failure_falls_through_and_counts_it() {
        let m = ScriptedMeasurer::new([Err("compile failed".into())]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Quick, synth_log_ctx());
        let got = s.step(&synth_cfg(), |_| Ok(42.0)).unwrap();
        assert_eq!(got, 42.0, "fallback value flows back to caller");
        let r = s.into_report();
        // Outcome stays Estimated when no candidate measured.
        assert_eq!(r.outcome, TuneOutcome::Estimated);
        assert_eq!(r.fallback_configs, 1);
    }

    #[test]
    fn candidate_search_no_measurer_calls_static_directly() {
        let mut s = CandidateSearch::new(None, BenchBudget::Standard, synth_log_ctx());
        let got = s.step(&synth_cfg(), |_| Ok(7.5)).unwrap();
        assert_eq!(got, 7.5);
        let r = s.into_report();
        assert_eq!(r.outcome, TuneOutcome::Estimated);
        assert_eq!(r.fallback_configs, 0);
    }

    #[test]
    fn candidate_search_mixed_sweep_keeps_measured_once_any_candidate_succeeds() {
        // 4 configs: fail, succeed, fail, succeed → outcome=Measured,
        // fallback_configs=2.
        let m = ScriptedMeasurer::new([
            Err("config 0 bad".into()),
            Ok(3.0),
            Err("config 2 bad".into()),
            Ok(9.0),
        ]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Standard, synth_log_ctx());
        let mut static_calls = 0;
        for _ in 0..4 {
            let _ = s
                .step(&synth_cfg(), |_| {
                    static_calls += 1;
                    Ok(100.0)
                })
                .unwrap();
        }
        assert_eq!(static_calls, 2, "static_fallback runs only when measure fails");
        let r = s.into_report();
        assert_eq!(r.outcome, TuneOutcome::Measured);
        assert_eq!(r.fallback_configs, 2);
    }

    #[test]
    fn candidate_search_propagates_static_fallback_error() {
        let m = ScriptedMeasurer::new([Err("compile".into())]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Quick, synth_log_ctx());
        let err = s
            .step(&synth_cfg(), |_| Err(MetalTileError::Autotune("static blew up".into())))
            .unwrap_err();
        assert!(err.to_string().contains("static blew up"));
        let r = s.into_report();
        assert_eq!(r.fallback_configs, 1, "fallback still counted even when static errors");
    }

    #[test]
    fn candidate_search_forwards_budget_to_measurer() {
        let m = ScriptedMeasurer::new([Ok(1.0)]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Standard, synth_log_ctx());
        let _ = s.step(&synth_cfg(), |_| unreachable!()).unwrap();
        assert!(matches!(m.last_budget.get(), Some(BenchBudget::Standard)));
    }

    // ── --shapes parsing + per-shape constexprs ──────────────────────

    #[test]
    fn parse_shape_overrides_empty_is_none() {
        assert!(parse_shape_overrides(&[]).unwrap().is_none());
    }

    #[test]
    fn parse_shape_overrides_passes_through_valid_list() {
        let got = parse_shape_overrides(&[64, 256, 1024, 16384]).unwrap().unwrap();
        assert_eq!(got, vec![64, 256, 1024, 16384]);
    }

    #[test]
    fn parse_shape_overrides_rejects_zero() {
        let err = parse_shape_overrides(&[64, 0, 1024]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("N must be > 0"), "got: {msg}");
    }

    #[test]
    fn synth_constexprs_uses_override_when_present() {
        // Two overrides for the same spec land in different N buckets,
        // so cache_key (which composes the bucketed N) differs.
        let spec = mock_spec_with_n(100);
        let ce_low = synth_constexprs_for(&spec, Some(64));
        let ce_high = synth_constexprs_for(&spec, Some(16_384));
        let k_low = metaltile::autotune::cache_key("k", &ce_low);
        let k_high = metaltile::autotune::cache_key("k", &ce_high);
        assert_ne!(
            k_low, k_high,
            "different N overrides must produce different cache keys (got both {k_low})",
        );
    }

    #[test]
    fn synth_constexprs_falls_back_to_spec_shape_when_no_override() {
        // None override + spec.shapes[0].n=100 must match an explicit Some(100).
        let spec = mock_spec_with_n(100);
        let ce_default = synth_constexprs_for(&spec, None);
        let ce_explicit = synth_constexprs_for(&spec, Some(100));
        assert_eq!(
            metaltile::autotune::cache_key("k", &ce_default),
            metaltile::autotune::cache_key("k", &ce_explicit),
        );
    }

    #[test]
    fn synth_constexprs_with_no_shape_and_no_override_uses_defaults() {
        // No spec shape, no override → should still produce a stable key
        // (the static-cost-only fallback path).
        let spec = mock_spec_empty_shapes();
        let ce = synth_constexprs_for(&spec, None);
        let k = metaltile::autotune::cache_key("k", &ce);
        // 1024 lands in the 1024..4096 bucket.
        assert!(k.contains("N=1024..4096"), "got: {k}");
    }

    /// Build a `BenchSpec` literal good enough for `synth_constexprs_for`
    /// — only `shapes` is read by that helper, so most fields are
    /// placeholder constants. Stays in tests; not a public helper.
    fn mock_spec_with_n(n: usize) -> BenchSpec {
        static TENSOR_BUFS: &[metaltile_std::spec::TensorBufSpec] = &[];
        static SCALAR_BUFS: &[metaltile_std::spec::ScalarBufSpec] = &[];
        static CEXPRS: &[(&str, metaltile_std::spec::Dim)] = &[];
        // Shapes can't easily be made `&'static` from a runtime n, so
        // construct a leaked slice for the test's lifetime.
        let shapes: &'static [ShapeSpec] = Box::leak(Box::new([ShapeSpec {
            label: "test",
            n,
            b: 1,
            check_n: n,
            check_b: 1,
            mode: metaltile_core::ir::KernelMode::Elementwise,
            tpg: 256,
            grid: metaltile_std::spec::DispatchGrid::DivCeilN,
            tensor_bufs: TENSOR_BUFS,
            scalar_bufs: SCALAR_BUFS,
            cexprs: CEXPRS,
            out_elems: metaltile_std::spec::Dim::N,
            reads: 1,
            bytes_fn: metaltile_std::spec::bytes_elementwise,
            mlx_args: None,
            mlx_grid: None,
            mlx_tpg: 256,
        }]));
        mock_spec(shapes)
    }

    fn mock_spec_empty_shapes() -> BenchSpec { mock_spec(&[]) }

    // ── --export-training-data writer ────────────────────────────────

    #[test]
    fn write_training_jsonl_emits_one_object_per_line() {
        use metaltile::autotune::TuneConfig;
        let rows = vec![
            TrainingRow {
                kernel: "mt_a".into(),
                dtype: "f16".into(),
                family: "Elementwise".into(),
                bucket: [("N".to_string(), (0usize, 256usize))].into_iter().collect(),
                best_config: TuneConfig::default(),
                perf_us: 1.5,
                timestamp: 1,
            },
            TrainingRow {
                kernel: "mt_b".into(),
                dtype: "f32".into(),
                family: "Matmul".into(),
                bucket: [("N".to_string(), (256usize, 1024usize))].into_iter().collect(),
                best_config: TuneConfig::default(),
                perf_us: 2.5,
                timestamp: 2,
            },
        ];
        let mut buf: Vec<u8> = Vec::new();
        write_training_jsonl(&mut buf, &rows).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        // Each line is a complete JSON object that round-trips.
        for line in &lines {
            let _: TrainingRow = serde_json::from_str(line).expect("each line is valid JSON");
        }
        assert!(lines[0].contains("\"kernel\":\"mt_a\""));
        assert!(lines[1].contains("\"kernel\":\"mt_b\""));
    }

    #[test]
    fn write_training_jsonl_empty_rows_writes_nothing() {
        let mut buf: Vec<u8> = Vec::new();
        write_training_jsonl(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }

    fn mock_spec(shapes: &'static [ShapeSpec]) -> BenchSpec {
        fn ir(_dt: DType) -> metaltile_core::ir::Kernel {
            // Never actually invoked by synth_constexprs_for — only the
            // shapes are read.
            metaltile_core::ir::Kernel::new("mock")
        }
        BenchSpec {
            op: "test",
            subop: "test",
            kernel_name: "k",
            kernel_ir: ir,
            dtypes: &[],
            tol: 0.0,
            mlx_src: None,
            mlx_pattern: None,
            shapes,
            dispatch: BenchDispatch::Generic,
            kernel_mode: None,
        }
    }
}

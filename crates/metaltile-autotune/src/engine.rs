//! Public autotune entry point + per-(kernel, dtype, n_override) driver.

use metaltile::{
    MetalTileError,
    autotune::{Autotuner, PsoReflection, TuneConfig},
};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_std::{
    runner::GpuRunner,
    spec::{BenchDispatch, BenchSpec},
};

use crate::{
    AutotuneError, AutotuneOptions, AutotuneSummary, KernelTuneResult,
    budget::{BenchBudget, TuneOutcome, TuneReport},
    cost::static_cost,
    measurer::{AnyMeasurer, GpuMeasurer, SdpaPrefillMeasurer, SdpaVectorMeasurer},
    search::{CandidateSearch, LogCtx},
    util::{effective_bucket_n, synth_constexprs_for},
};

/// Run the autotune pipeline. Streams per-kernel results through
/// `on_kernel_done`; returns a final summary the caller can use to
/// render a closing block.
///
/// No `println!` / `eprintln!` here — the CLI owns presentation.
pub fn run_autotune<F>(
    options: &AutotuneOptions,
    runner: Option<&GpuRunner>,
    mut on_kernel_done: F,
) -> Result<AutotuneSummary, AutotuneError>
where
    F: FnMut(&KernelTuneResult),
{
    let _span = tracing::info_span!(
        "autotune",
        filter = ?options.filter,
        measure = options.measure,
        quick = options.quick,
        shapes = ?options.shape_overrides,
    )
    .entered();

    let mut tuner = Autotuner::new(Autotuner::default_cache_dir(), /* enabled= */ true);

    let mut specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>.into_iter().collect();
    specs.sort_unstable_by_key(|s| (s.op, s.subop));

    let bench_budget =
        if options.quick { BenchBudget::Quick } else { BenchBudget::Standard };

    let mut tuned = 0usize;
    let mut measured = 0usize;
    let mut estimated = 0usize;
    let mut skipped = 0usize;
    let mut fallbacks = 0usize;
    for spec in specs {
        if !matches_filter(options.filter.as_deref(), spec.kernel_name) {
            continue;
        }
        for &dt in spec.dtypes {
            // Each n_override is one cache entry. None → use the
            // spec's own shape; Some(n) → override per --shapes.
            let overrides: Vec<Option<usize>> = match options.shape_overrides.as_deref() {
                Some(list) => list.iter().map(|&n| Some(n)).collect(),
                None => vec![None],
            };
            for n_override in overrides {
                let dtype_label = metaltile_std::bench_types::dtype_label(dt);
                let result = tune_one(&mut tuner, spec, dt, runner, bench_budget, n_override);
                let kernel_result = match &result {
                    Ok(TuneReport { outcome: TuneOutcome::Measured, fallback_configs }) => {
                        tuned += 1;
                        measured += 1;
                        fallbacks += fallback_configs;
                        KernelTuneResult {
                            kernel_name: spec.kernel_name,
                            dtype_label,
                            n_override,
                            outcome: Ok(KernelOk {
                                outcome: TuneOutcome::Measured,
                                fallback_configs: *fallback_configs,
                            }),
                        }
                    },
                    Ok(TuneReport { outcome: TuneOutcome::Estimated, fallback_configs }) => {
                        tuned += 1;
                        estimated += 1;
                        fallbacks += fallback_configs;
                        KernelTuneResult {
                            kernel_name: spec.kernel_name,
                            dtype_label,
                            n_override,
                            outcome: Ok(KernelOk {
                                outcome: TuneOutcome::Estimated,
                                fallback_configs: *fallback_configs,
                            }),
                        }
                    },
                    Err(e) => {
                        skipped += 1;
                        KernelTuneResult {
                            kernel_name: spec.kernel_name,
                            dtype_label,
                            n_override,
                            outcome: Err(e.to_string()),
                        }
                    },
                };
                on_kernel_done(&kernel_result);
            }
        }
    }

    let cache_entries = tuner.cache().len();
    Ok(AutotuneSummary {
        tuned,
        measured,
        estimated,
        skipped,
        fallbacks,
        cache_entries,
        cache_dir: Autotuner::default_cache_dir(),
    })
}

/// Per-kernel sweep success payload.
#[derive(Debug, Clone, Copy)]
pub struct KernelOk {
    pub outcome: TuneOutcome,
    pub fallback_configs: usize,
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
) -> Result<TuneReport, AutotuneError> {
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

    let mut bench = |cfg: &TuneConfig| -> Result<(f64, Option<PsoReflection>), MetalTileError> {
        search.step(cfg, |cfg| static_cost(&kernel_template, mode, cfg))
    };

    tuner
        .tune(&entry_name, family, &constexprs, &mut bench)
        .map_err(|e| AutotuneError::Other(format!("{}: {e}", entry_name)))?;

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
) -> Result<Option<AnyMeasurer<'a>>, AutotuneError> {
    let Some(runner) = runner else { return Ok(None) };
    match spec.dispatch {
        BenchDispatch::Generic if !spec.shapes.is_empty() =>
            Ok(Some(AnyMeasurer::Generic(GpuMeasurer::new(runner, spec, dt, mode, n_override)))),
        BenchDispatch::SdpaVector { head_dim, n_kv, n_q_heads, gqa_factor, batch: _, tpg } => {
            let m = SdpaVectorMeasurer::new(
                runner, spec, dt, head_dim, n_kv, n_q_heads, gqa_factor, tpg,
            )
            .map_err(AutotuneError::Other)?;
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
            .map_err(AutotuneError::Other)?;
            Ok(Some(AnyMeasurer::SdpaPrefill(m)))
        },
        _ => Ok(None),
    }
}

/// Filter helper: case-insensitive substring match. Duplicated from
/// the CLI's private helper rather than imported, to keep the library
/// callable without depending on `metaltile-cli`.
fn matches_filter(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else { return true };
    label.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())
}

//! Per-(kernel, dtype) candidate sweep orchestration.

use metaltile::{
    MetalTileError,
    autotune::{PsoReflection, TuneConfig},
};

use crate::{
    budget::{BenchBudget, TuneOutcome, TuneReport},
    measurer::CandidateMeasurer,
};

/// Per-(kernel, dtype) sweep state: drives one candidate through
/// `measure → fallback?` and accumulates the outcome counters that
/// land in the run-end summary. Lifted out of the bench closure so the
/// orchestration can be unit-tested directly (see the `tests` module
/// below).
pub(crate) struct CandidateSearch<'a> {
    measurer: Option<&'a dyn CandidateMeasurer>,
    budget: BenchBudget,
    log_ctx: LogCtx<'a>,
    outcome: TuneOutcome,
    fallback_configs: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct LogCtx<'a> {
    pub kernel: &'a str,
    pub dtype: &'a str,
}

impl<'a> CandidateSearch<'a> {
    pub(crate) fn new(
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
    pub(crate) fn step(
        &mut self,
        cfg: &TuneConfig,
        static_fallback: impl FnOnce(&TuneConfig) -> Result<f64, MetalTileError>,
    ) -> Result<(f64, Option<PsoReflection>), MetalTileError> {
        if let Some(m) = self.measurer {
            match m.measure(cfg, self.budget) {
                Ok((us, refl)) => {
                    self.outcome = TuneOutcome::Measured;
                    return Ok((us, refl));
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
        // Static fallback path has no PSO; no reflection to report.
        static_fallback(cfg).map(|us| (us, None))
    }

    pub(crate) fn into_report(self) -> TuneReport {
        TuneReport { outcome: self.outcome, fallback_configs: self.fallback_configs }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

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
        fn measure(
            &self,
            _cfg: &TuneConfig,
            budget: BenchBudget,
        ) -> Result<(f64, Option<PsoReflection>), String> {
            self.last_budget.set(Some(budget));
            self.script
                .borrow_mut()
                .pop_front()
                .expect("ScriptedMeasurer: more measure() calls than scripted")
                .map(|us| (us, None))
        }
    }

    fn synth_cfg() -> TuneConfig { TuneConfig::default() }
    fn synth_log_ctx() -> LogCtx<'static> { LogCtx { kernel: "test_kernel", dtype: "f32" } }

    #[test]
    fn candidate_search_measure_success_flips_outcome_to_measured() {
        let m = ScriptedMeasurer::new([Ok(5.0)]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Quick, synth_log_ctx());
        let got = s.step(&synth_cfg(), |_| panic!("static_fallback should not run")).unwrap();
        assert_eq!(got, (5.0, None));
        let r = s.into_report();
        assert_eq!(r.outcome, TuneOutcome::Measured);
        assert_eq!(r.fallback_configs, 0);
    }

    #[test]
    fn candidate_search_measure_failure_falls_through_and_counts_it() {
        let m = ScriptedMeasurer::new([Err("compile failed".into())]);
        let mut s = CandidateSearch::new(Some(&m), BenchBudget::Quick, synth_log_ctx());
        let got = s.step(&synth_cfg(), |_| Ok(42.0)).unwrap();
        assert_eq!(got, (42.0, None), "fallback value flows back to caller");
        let r = s.into_report();
        // Outcome stays Estimated when no candidate measured.
        assert_eq!(r.outcome, TuneOutcome::Estimated);
        assert_eq!(r.fallback_configs, 1);
    }

    #[test]
    fn candidate_search_no_measurer_calls_static_directly() {
        let mut s = CandidateSearch::new(None, BenchBudget::Standard, synth_log_ctx());
        let got = s.step(&synth_cfg(), |_| Ok(7.5)).unwrap();
        assert_eq!(got, (7.5, None));
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
}

//! Bench iteration budget + per-kernel outcome tracking.

/// How many warmup + measure iterations `--measure` runs per candidate.
///
/// `Standard` (20 warmup + 100 iters) is the Playbook minimum for
/// JIT-cache-warm medians. `Quick` (3 + 11) is a triage mode opted into
/// via `--quick`; it gets the search done in seconds but the resulting
/// medians are noisy and shouldn't be persisted to a long-lived cache.
#[derive(Debug, Clone, Copy)]
pub enum BenchBudget {
    Standard,
    Quick,
}

impl BenchBudget {
    pub fn iters(self) -> (usize, usize) {
        match self {
            BenchBudget::Standard => (20, 100),
            BenchBudget::Quick => (3, 11),
        }
    }
}

/// Per-kernel sweep outcome. Surfaced in the public API as
/// [`crate::TuneOutcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneOutcome {
    Measured,
    Estimated,
}

/// Result of one (kernel, dtype, n_override) sweep. `fallback_configs`
/// counts how many candidates in this sweep fell back to `static_cost`
/// because their measure call errored.
#[derive(Debug, Clone, Copy)]
pub struct TuneReport {
    pub outcome: TuneOutcome,
    pub fallback_configs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_budget_iters_match_documented_values() {
        assert_eq!(BenchBudget::Standard.iters(), (20, 100));
        assert_eq!(BenchBudget::Quick.iters(), (3, 11));
    }
}

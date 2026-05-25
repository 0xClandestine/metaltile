//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Shared result row types produced by both in-tree and external code paths.

use metaltile_core::bench::types::CorrectnessStatus;

// ── BenchRow ──────────────────────────────────────────────────────────────

/// One bench measurement: kernel × shape × dtype.
#[derive(Clone)]
pub struct BenchRow {
    pub op: String,
    pub subop: Option<String>,
    pub shape: String,
    pub metric: String,
    pub ref_perf: Option<f64>,
    pub mt_perf: Option<f64>,
    pub correctness: CorrectnessStatus,
    /// `-vv` GPU timing. Always `None` for external-project results.
    pub timing_p95_us: Option<f64>,
    pub timing_p99_us: Option<f64>,
    pub timing_cv_pct: Option<f64>,
}

impl BenchRow {
    /// Parse a `"result"` JSONL record from an external harness into a `BenchRow`.
    pub fn from_json(val: &serde_json::Value) -> Option<Self> {
        let op = val.get("op")?.as_str()?.to_string();
        let subop =
            val.get("subop").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from);
        let shape = val.get("shape").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let metric = val.get("metric").and_then(|v| v.as_str()).unwrap_or("GB/s").to_string();
        let ref_perf = val.get("ref_gbps").and_then(|v| v.as_f64());
        let mt_perf = val.get("mt_gbps").and_then(|v| v.as_f64());
        let passed = val.get("passed").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_err = val.get("max_err").and_then(|v| v.as_f64()).map(|v| v as f32);
        let cosine_sim =
            val.get("cosine_sim").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(1.0);

        let correctness = match (mt_perf.is_some(), passed, max_err) {
            (false, ..) => CorrectnessStatus::Unavailable,
            (true, true, Some(e)) => CorrectnessStatus::Passed { max_abs_err: e, cosine_sim },
            (true, true, None) => CorrectnessStatus::Passed { max_abs_err: 0.0, cosine_sim: 1.0 },
            (true, false, Some(e)) => CorrectnessStatus::Failed { max_abs_err: e, cosine_sim },
            (true, false, None) =>
                CorrectnessStatus::Failed { max_abs_err: f32::MAX, cosine_sim: 0.0 },
        };

        Some(BenchRow {
            op,
            subop,
            shape,
            metric,
            ref_perf,
            mt_perf,
            correctness,
            timing_p95_us: None,
            timing_p99_us: None,
            timing_cv_pct: None,
        })
    }

    pub fn op_display(&self) -> String {
        match &self.subop {
            Some(s) if !s.is_empty() => format!("{} ({})", self.op, s),
            _ => self.op.clone(),
        }
    }

    pub fn shape(&self) -> &str { &self.shape }

    pub fn metric(&self) -> &str { &self.metric }

    pub fn ref_perf(&self) -> Option<f64> { self.ref_perf }

    pub fn mt_perf(&self) -> Option<f64> { self.mt_perf }

    pub fn pct(&self) -> Option<f64> {
        match (self.ref_perf, self.mt_perf) {
            (Some(r), Some(m)) if r > 0.0 => Some(m / r * 100.0),
            _ => None,
        }
    }

    pub fn correctness_status(&self) -> CorrectnessStatus { self.correctness }

    pub fn is_unchecked(&self) -> bool { matches!(self.correctness, CorrectnessStatus::Unchecked) }
}

impl From<&metaltile_core::bench::types::OpResult> for BenchRow {
    fn from(r: &metaltile_core::bench::types::OpResult) -> Self {
        let (p95, p99, cv) = match &r.mt_timing {
            Some(t) if t.is_valid() => (Some(t.p95_us), Some(t.p99_us), Some(t.cv_pct)),
            _ => (None, None, None),
        };
        BenchRow {
            op: r.op().to_string(),
            subop: r.subop().map(String::from),
            shape: r.shape().to_string(),
            metric: r.metric().to_string(),
            ref_perf: r.ref_perf(),
            mt_perf: r.mt_perf(),
            correctness: r.correctness_status(),
            timing_p95_us: p95,
            timing_p99_us: p99,
            timing_cv_pct: cv,
        }
    }
}

// ── TestRow ───────────────────────────────────────────────────────────────

/// One test result: kernel × dtype.
pub struct TestRow {
    pub kernel_name: String,
    pub dtype: String,
    pub passed: bool,
    pub max_err: Option<f32>,
}

// ── BuildRow ──────────────────────────────────────────────────────────────

/// One build result: kernel × dtype.
pub struct BuildRow {
    pub kernel_name: String,
    pub dtype: String,
    pub ok: bool,
    pub error: Option<String>,
}

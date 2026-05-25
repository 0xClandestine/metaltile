//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! BenchReport — a completed bench run as a queryable, serialisable value.

use metaltile_core::bench::types::CorrectnessStatus;
use serde_json::Value;

use metaltile::harness::BenchRow;

// ── BenchSummary ──────────────────────────────────────────────────────────

/// Aggregate counts over a completed bench run.
pub struct BenchSummary {
    pub total: usize,
    pub implemented: usize,
    pub correct: usize,
    pub unchecked: usize,
}

// ── BenchReport ───────────────────────────────────────────────────────────

/// A completed bench run: an ordered slice of measurements with
/// validation, summary, and serialisation methods.
pub struct BenchReport {
    rows: Vec<BenchRow>,
}

impl BenchReport {
    pub fn new(rows: Vec<BenchRow>) -> Self { Self { rows } }

    pub fn rows(&self) -> &[BenchRow] { &self.rows }

    /// Validate that every implemented row has a correctness check.
    /// Returns a description of any unchecked rows on failure.
    pub fn validate(&self) -> Result<(), String> {
        let unchecked: Vec<String> = self
            .rows
            .iter()
            .filter(|r| r.is_unchecked())
            .map(|r| format!("{} [{}]", r.op_display(), r.shape))
            .collect();
        if unchecked.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "implemented benchmarks missing correctness checks: {}",
                unchecked.join(", ")
            ))
        }
    }

    /// Aggregate counts over all rows.
    pub fn summary(&self) -> BenchSummary {
        BenchSummary {
            total: self.rows.len(),
            implemented: self.rows.iter().filter(|r| r.mt_perf().is_some()).count(),
            correct: self
                .rows
                .iter()
                .filter(|r| matches!(r.correctness_status(), CorrectnessStatus::Passed { .. }))
                .count(),
            unchecked: self.rows.iter().filter(|r| r.is_unchecked()).count(),
        }
    }

    /// Serialise all rows as a device-tagged JSON snapshot.
    ///
    /// Returns the formatted JSON string. The caller is responsible for
    /// writing it to disk and reporting success/failure to the user.
    pub fn to_json(&self, device: &str) -> String {
        let s = self.summary();
        let mut out = String::new();
        out.push_str(&format!(
            "{{\"device\":{:?},\"summary\":{{\"total\":{},\"implemented\":{},\"correct\":{},\"unchecked\":{}}},\"results\":[\n",
            device, s.total, s.implemented, s.correct, s.unchecked,
        ));
        for (i, r) in self.rows.iter().enumerate() {
            let comma = if i + 1 < self.rows.len() { "," } else { "" };
            out.push_str(&format!("  {}{}\n", Self::format_row_json(r), comma));
        }
        out.push_str("]}");
        out
    }

    /// Convert each row to a `serde_json::Value` for diff comparisons.
    pub fn to_json_values(&self) -> Vec<Value> {
        self.rows.iter().map(Self::row_to_value).collect()
    }

    /// Derive a file-system safe chip slug from a device name.
    ///
    /// `"Apple M5 Max"` → `"apple-m5-max"` — matches the naming convention
    /// used by `baselines/<chip>.json`.
    pub fn chip_slug(device: &str) -> String {
        let mut out = String::with_capacity(device.len());
        let mut prev_dash = false;
        for ch in device.chars() {
            let lowered = ch.to_ascii_lowercase();
            if lowered.is_ascii_alphanumeric() {
                out.push(lowered);
                prev_dash = false;
            } else if !prev_dash && !out.is_empty() {
                out.push('-');
                prev_dash = true;
            }
        }
        while out.ends_with('-') {
            out.pop();
        }
        out
    }

    fn row_to_value(r: &BenchRow) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("op".into(), Value::from(r.op.as_str()));
        if let Some(sub) = &r.subop {
            obj.insert("subop".into(), Value::from(sub.as_str()));
        }
        obj.insert("shape".into(), Value::from(r.shape.as_str()));
        obj.insert("metric".into(), Value::from(r.metric.as_str()));
        obj.insert("ref".into(), r.ref_perf.map(Value::from).unwrap_or(Value::Null));
        obj.insert("mt".into(), r.mt_perf.map(Value::from).unwrap_or(Value::Null));
        Value::Object(obj)
    }

    fn format_row_json(r: &BenchRow) -> String {
        match r.subop.as_deref() {
            Some(s) => format!(
                "{{\"op\":{:?},\"subop\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}",
                r.op,
                s,
                r.shape,
                r.metric,
                fmt_f(r.ref_perf),
                fmt_f(r.mt_perf),
            ),
            None => format!(
                "{{\"op\":{:?},\"shape\":{:?},\"metric\":{:?},\"ref\":{},\"mt\":{}}}",
                r.op,
                r.shape,
                r.metric,
                fmt_f(r.ref_perf),
                fmt_f(r.mt_perf),
            ),
        }
    }
}

fn fmt_f(v: Option<f64>) -> String { v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "null".into()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_slug_normalises_apple_device_names() {
        assert_eq!(BenchReport::chip_slug("Apple M5 Max"), "apple-m5-max");
        assert_eq!(BenchReport::chip_slug("  Apple  --M1 (Pro)  "), "apple-m1-pro");
        assert_eq!(BenchReport::chip_slug("Apple_M2_Max!"), "apple-m2-max");
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Compile-time occupancy and register profiles for bench -v/-vv display.

use std::collections::HashMap;

use metaltile_codegen::passes::{
    self,
    occupancy::{self, Bottleneck},
};

use metaltile::harness::Filters;

// ── ProfileRow ────────────────────────────────────────────────────────────

/// Occupancy and register profile for a single kernel × dtype pair.
#[derive(Clone)]
pub struct ProfileRow {
    pub occ_pct: f64,
    pub regs_per_thread: usize,
    pub bottleneck: &'static str,
}

// ── ProfileMap ────────────────────────────────────────────────────────────

/// Map of `(op_display, dtype_label)` → `ProfileRow`, computed at
/// command startup when `-v` is requested.
pub struct ProfileMap(HashMap<(String, String), ProfileRow>);

impl ProfileMap {
    /// Run the standard pass pipeline + occupancy estimator over the
    /// filtered spec corpus. CPU-only and fast — typically < 100 ms.
    pub fn compute(filters: &Filters) -> Self {
        let mut map = HashMap::new();
        let specs = metaltile::bench::bench_specs();
        for spec in specs {
            if !filters.matches_kernel(spec.kernel_name, spec.op) {
                continue;
            }
            let op_display = if spec.subop.is_empty() {
                spec.op.to_string()
            } else {
                format!("{} ({})", spec.op, spec.subop)
            };
            for &dt in spec.dtypes {
                let mut k = (spec.kernel_ir)(dt);
                k.mode = spec.dispatch.default_mode(spec.shapes);
                if passes::run_passes(&mut k, &passes::standard_pipeline()).is_err() {
                    continue;
                }
                let reg_est = passes::register_estimate::estimate_registers(&k);
                let candidates: Vec<(u32, Option<u32>)> =
                    [64u32, 128, 256, 512, 1024].iter().map(|&s| (s, None)).collect();
                let Some((_tg, est)) = occupancy::best_threadgroup_size(&k, &candidates) else {
                    continue;
                };
                let bottleneck = match est.bottleneck {
                    Bottleneck::ThreadLimited => "thread-limited",
                    Bottleneck::RegisterLimited => "register-limited",
                    Bottleneck::MemoryLimited => "tgmem-limited",
                    _ => "unknown",
                };
                let dtype_label = metaltile_core::bench::types::dtype_label(dt).to_string();
                map.insert((op_display.clone(), dtype_label), ProfileRow {
                    occ_pct: est.occupancy_pct,
                    regs_per_thread: reg_est.regs_per_thread,
                    bottleneck,
                });
            }
        }
        ProfileMap(map)
    }

    /// Look up the profile for an `(op_display, dtype_label)` pair.
    pub fn get(&self, op_display: &str, dtype: &str) -> Option<&ProfileRow> {
        self.0.get(&(op_display.to_string(), dtype.to_string()))
    }

    /// Consume the map into its inner `HashMap` for use with `SuitePrinter`.
    pub fn into_inner(self) -> HashMap<(String, String), ProfileRow> { self.0 }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! CPU-side kernel profiling + bottleneck classification for `tile bench`.
//!
//! [`estimate_profile`] runs the standard optimization pipeline on a clone of a
//! kernel and reports its estimated occupancy, register pressure, and the raw
//! occupancy bottleneck — the same analysis the `-v` profile columns used to
//! compute in the CLI, now shared so `run_kernel_bench` can attach it to every
//! row (and feed the bottleneck verdict).
//!
//! [`classify_bottleneck`] folds the roofline position (arithmetic intensity vs
//! the device ridge point) together with those occupancy/register signals into a
//! single human-readable verdict.

use metaltile_codegen::passes::{
    self,
    occupancy::{self, Bottleneck},
};
use metaltile_core::ir::Kernel;

/// Threadgroup sizes the occupancy estimator sweeps to find the best occupancy.
/// Matches the CLI's historical `-v` profile sweep.
const TG_CANDIDATES: [u32; 5] = [64, 128, 256, 512, 1024];

/// CPU-estimated execution profile for one kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelProfile {
    /// Best estimated SIMD occupancy across the threadgroup-size sweep (%).
    pub occ_pct: f64,
    /// Estimated registers per thread (independent of threadgroup size).
    pub regs_per_thread: usize,
    /// The occupancy estimator's raw bottleneck for the best threadgroup size.
    pub raw_bottleneck: Bottleneck,
}

/// Run the standard pipeline on a clone of `kernel` and estimate its occupancy +
/// register pressure. Returns `None` if the pipeline fails or no threadgroup
/// size yields an estimate (so the caller leaves the profile columns blank).
pub fn estimate_profile(kernel: &Kernel) -> Option<KernelProfile> {
    let mut k = kernel.clone();
    passes::run_passes(&mut k, &passes::standard_pipeline()).ok()?;
    let regs_per_thread = passes::register_estimate::estimate_registers(&k).regs_per_thread;
    let candidates: Vec<(u32, Option<u32>)> = TG_CANDIDATES.iter().map(|&s| (s, None)).collect();
    let (_, est) = occupancy::best_threadgroup_size(&k, &candidates)?;
    Some(KernelProfile {
        occ_pct: est.occupancy_pct,
        regs_per_thread,
        raw_bottleneck: est.bottleneck,
    })
}

// ── Bottleneck verdict ─────────────────────────────────────────────────────

/// Occupancy below this (%) leaves too few resident threads to hide memory
/// latency — the kernel is occupancy-starved rather than truly memory/compute
/// bound.
const LOW_OCCUPANCY_PCT: f64 = 50.0;
/// At/above this %-of-peak bandwidth, a memory-region kernel is saturating DRAM.
const BW_SATURATED_PCT: f64 = 60.0;
/// At/above this %-of-peak compute, a compute-region kernel is saturating the ALUs.
const FLOPS_SATURATED_PCT: f64 = 50.0;

/// Combine the roofline position with the occupancy/register signals into one
/// bottleneck verdict: `memory-bound` / `compute-bound` / `occupancy-limited` /
/// `register-limited` / `latency-bound`. Returns `None` when there isn't enough
/// information to classify (no device specs and no profile).
///
/// `ridge_point` is the device's ridge in FLOPs/byte (`peak_compute ÷ peak_bw`):
/// a kernel left of it (lower arithmetic intensity) is bandwidth-limited, right
/// of it compute-limited.
pub fn classify_bottleneck(
    arith_intensity: Option<f64>,
    ridge_point: Option<f64>,
    pct_peak_bw: Option<f64>,
    pct_peak_flops: Option<f64>,
    profile: Option<&KernelProfile>,
) -> Option<&'static str> {
    let occ = profile.map(|p| p.occ_pct);
    let reg_limited = profile.is_some_and(|p| p.raw_bottleneck == Bottleneck::RegisterLimited);
    let low_occ = occ.is_some_and(|o| o < LOW_OCCUPANCY_PCT);

    // Resource-starvation verdicts shared by both roofline regions: a kernel that
    // can't fill the machine is limited by that before bandwidth/compute.
    let starvation = || {
        if reg_limited {
            Some("register-limited")
        } else if low_occ {
            Some("occupancy-limited")
        } else {
            None
        }
    };

    // Which side of the ridge are we on (when AI + ridge are both known)?
    let compute_region = arith_intensity.zip(ridge_point).map(|(ai, ridge)| ai >= ridge);

    match compute_region {
        // Right of the ridge: compute is the ceiling.
        Some(true) =>
            if pct_peak_flops.is_some_and(|p| p >= FLOPS_SATURATED_PCT) {
                Some("compute-bound")
            } else {
                // Not saturating compute → blame starvation if any, else it's
                // still compute-region (the ceiling it must climb is compute).
                starvation().or(Some("compute-bound"))
            },
        // Left of the ridge: bandwidth is the ceiling.
        Some(false) =>
            if pct_peak_bw.is_some_and(|p| p >= BW_SATURATED_PCT) {
                Some("memory-bound")
            } else {
                // Not saturating BW and occupancy is fine ⇒ launch/latency bound.
                starvation().or(Some("latency-bound"))
            },
        // No FLOP count (memory-bound kernel) or no device specs: lean on the
        // bandwidth + occupancy signals alone.
        None =>
            if pct_peak_bw.is_some_and(|p| p >= BW_SATURATED_PCT) {
                Some("memory-bound")
            } else {
                starvation().or_else(|| pct_peak_bw.map(|_| "latency-bound"))
            },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A memory-region kernel (low AI) that saturates DRAM reads `memory-bound`.
    #[test]
    fn low_ai_saturating_bandwidth_is_memory_bound() {
        let v = classify_bottleneck(Some(0.5), Some(26.0), Some(92.0), Some(3.0), None);
        assert_eq!(v, Some("memory-bound"));
    }

    // A compute-region kernel (high AI) that saturates the ALUs reads `compute-bound`.
    #[test]
    fn high_ai_saturating_flops_is_compute_bound() {
        let v = classify_bottleneck(Some(120.0), Some(26.0), Some(20.0), Some(85.0), None);
        assert_eq!(v, Some("compute-bound"));
    }

    // Low occupancy dominates: even in the memory region, an occupancy-starved
    // kernel that isn't saturating BW reads `occupancy-limited`.
    #[test]
    fn low_occupancy_not_saturating_reads_occupancy_limited() {
        let prof = KernelProfile {
            occ_pct: 10.0,
            regs_per_thread: 32,
            raw_bottleneck: Bottleneck::ThreadLimited,
        };
        let v = classify_bottleneck(Some(0.5), Some(26.0), Some(20.0), Some(1.0), Some(&prof));
        assert_eq!(v, Some("occupancy-limited"));
    }

    // Register pressure is called out specifically when it's the raw bottleneck.
    #[test]
    fn register_pressure_reads_register_limited() {
        let prof = KernelProfile {
            occ_pct: 12.0,
            regs_per_thread: 256,
            raw_bottleneck: Bottleneck::RegisterLimited,
        };
        let v = classify_bottleneck(Some(0.5), Some(26.0), Some(15.0), Some(1.0), Some(&prof));
        assert_eq!(v, Some("register-limited"));
    }

    // Not saturating bandwidth, occupancy fine ⇒ latency/launch-overhead bound.
    #[test]
    fn underutilized_with_good_occupancy_reads_latency_bound() {
        let prof = KernelProfile {
            occ_pct: 100.0,
            regs_per_thread: 16,
            raw_bottleneck: Bottleneck::ThreadLimited,
        };
        let v = classify_bottleneck(Some(0.5), Some(26.0), Some(20.0), Some(1.0), Some(&prof));
        assert_eq!(v, Some("latency-bound"));
    }

    // No specs and no profile ⇒ cannot classify.
    #[test]
    fn no_signals_is_unclassified() {
        assert_eq!(classify_bottleneck(None, None, None, None, None), None);
    }
}

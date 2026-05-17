//! Occupancy estimation for Metal GPU kernels.
//!
//! Computes an estimated occupancy percentage based on register pressure,
//! threadgroup memory usage, and threadgroup size.
//!
//! ## Apple GPU Context
//!
//! | Family | Max Threads/TG | TG Memory | Max Regs/Thread | Notes |
//! |---|---|---|---|---|
//! | Apple7 (M1) | 1024 | ~32KB | 128 | Fixed allocation |
//! | Apple8 (M2) | 1024 | ~32KB | 128 | Similar to M1 |
//! | Apple9 (M3) | 1024 | ~32KB | Dynamic | OMU-managed |
//! | Apple10 (M4) | 1024 | ~32KB | Dynamic | Improved OMU |
//! | Apple11 (M5) | 1024 | ~32KB | Dynamic | Smarter OMU |
//!
//! For M3+, register allocation is dynamic; we use 128 as a conservative
//! heuristic since the OMU is opaque to us.
//!
//! ## Usage
//!
//! This module is not a Pass — it runs as post-pipeline analysis that
//! feeds into the autotuner.

use metaltile_core::ir::Kernel;

use super::register_estimate;

/// Per-GPU-family resource limits.
#[derive(Debug, Clone, Copy)]
pub struct GpuLimits {
    /// Maximum threads per threadgroup.
    pub max_threads_per_tg: u32,
    /// Threadgroup memory in bytes.
    pub tg_memory_bytes: u32,
    /// Maximum registers per thread (conservative for M3+).
    pub max_regs_per_thread: u32,
}

impl Default for GpuLimits {
    fn default() -> Self {
        GpuLimits { max_threads_per_tg: 1024, tg_memory_bytes: 32 * 1024, max_regs_per_thread: 128 }
    }
}

/// Bottleneck preventing higher occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// Register file is the limiting factor.
    RegisterLimited,
    /// Threadgroup memory is the limiting factor.
    MemoryLimited,
    /// Thread count is the limiting factor.
    ThreadLimited,
}

/// Occupancy estimate for a kernel with a given threadgroup size.
#[derive(Debug, Clone)]
pub struct OccupancyEstimate {
    /// Estimated occupancy as a percentage (0.0–100.0).
    pub occupancy_pct: f64,
    /// The primary bottleneck.
    pub bottleneck: Bottleneck,
    /// Maximum simultaneous threadgroups per compute unit, if estimable.
    pub max_tgs_per_cu: Option<u32>,
}

/// Compute an occupancy estimate for `kernel` with the given `threadgroup_size`.
///
/// `tg_mem_usage_bytes` is an optional estimate of threadgroup memory usage.
/// If None, memory is assumed not to be the bottleneck.
pub fn estimate_occupancy(
    kernel: &Kernel,
    threadgroup_size: u32,
    tg_mem_usage_bytes: Option<u32>,
) -> OccupancyEstimate {
    let limits = GpuLimits::default();
    let reg_est = register_estimate::estimate_registers(kernel);

    // Register-limited occupancy.
    let reg_occ = limits.max_regs_per_thread as f64 / reg_est.regs_per_thread as f64;
    let reg_occ = reg_occ.min(1.0); // can't exceed 100%

    // Thread-limited occupancy.
    let thr_occ = limits.max_threads_per_tg as f64 / threadgroup_size as f64;
    let thr_occ = thr_occ.min(1.0);

    // Memory-limited occupancy.
    let mem_occ = if let Some(mem_used) = tg_mem_usage_bytes {
        if mem_used == 0 { 1.0 } else { (limits.tg_memory_bytes as f64 / mem_used as f64).min(1.0) }
    } else {
        1.0
    };

    // Find the minimum → that's the occupancy.
    let mut occ = reg_occ.min(thr_occ).min(mem_occ);
    // Round to avoid floating-point noise.
    occ = (occ * 1000.0).round() / 1000.0;

    let bottleneck = if occ >= 0.999 {
        // At or near 100%, the thread count is the hard ceiling.
        Bottleneck::ThreadLimited
    } else if (occ - reg_occ).abs() < 0.001 {
        Bottleneck::RegisterLimited
    } else if (occ - mem_occ).abs() < 0.001 {
        Bottleneck::MemoryLimited
    } else {
        Bottleneck::ThreadLimited
    };

    // Max TGs per CU: if occupancy is 25%, 4 TGs can fit.
    let max_tgs = if occ > 0.0 { Some((1.0 / occ).round() as u32) } else { None };

    OccupancyEstimate { occupancy_pct: occ * 100.0, bottleneck, max_tgs_per_cu: max_tgs }
}

/// Convenience: estimate occupancy for common threadgroup sizes and return the best.
///
/// `candidates` is a list of (threadgroup_size, tg_mem_bytes) to evaluate.
/// Returns the candidate with the highest estimated occupancy.
pub fn best_threadgroup_size(
    kernel: &Kernel,
    candidates: &[(u32, Option<u32>)],
) -> Option<(u32, OccupancyEstimate)> {
    let mut best: Option<(u32, OccupancyEstimate)> = None;

    for &(tg_size, mem) in candidates {
        let est = estimate_occupancy(kernel, tg_size, mem);
        match &best {
            None => best = Some((tg_size, est)),
            Some((_, prev)) if est.occupancy_pct > prev.occupancy_pct => {
                best = Some((tg_size, est));
            },
            _ => {},
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{Op, ValueId};

    use super::*;

    #[test]
    fn empty_kernel_full_occupancy() {
        let k = Kernel::new("empty");
        let est = estimate_occupancy(&k, 256, None);
        assert!((est.occupancy_pct - 100.0).abs() < 0.1);
        assert_eq!(est.bottleneck, Bottleneck::ThreadLimited);
    }

    #[test]
    fn register_heavy_kernel_reduced_occupancy() {
        let mut k = Kernel::new("regheavy");
        // Push 100 const ops → ~150 regs/thread → occupancy ~85%
        for i in 0..100u32 {
            k.body.push_op(Op::Const { value: i as i64 }, ValueId::new(i));
        }

        let est = estimate_occupancy(&k, 256, None);
        // regs_per_thread = 100 * 1.5 = 150, which exceeds 128 → occupancy < 100%
        assert!(est.occupancy_pct < 100.0);
        assert_eq!(est.bottleneck, Bottleneck::RegisterLimited);
    }

    #[test]
    fn threadgroup_size_limits_occupancy() {
        let k = Kernel::new("bigtg");
        // 2048 threads/tg → capped at 1024.
        let est = estimate_occupancy(&k, 2048, None);
        // 1024/2048 = 0.5
        assert!((est.occupancy_pct - 50.0).abs() < 1.0);
    }

    #[test]
    fn best_threadgroup_size_picks_highest() {
        let k = Kernel::new("best");
        let candidates = &[(64, None), (128, None), (256, None), (512, None), (1024, None)];
        let best = best_threadgroup_size(&k, candidates).unwrap();
        // Empty kernel: all threadgroup sizes give 100%, tie breaks to first (64).
        assert_eq!(best.0, 64);
    }
}

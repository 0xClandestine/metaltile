//! Occupancy Estimation — predict GPU thread occupancy from register pressure.
//!
//! Computes an estimated occupancy percentage based on register pressure,
//! threadgroup memory usage, and threadgroup size.  This analysis feeds the
//! autotuner, not the compilation pipeline directly.
//!
//! ## Apple GPU Context
//!
//! | Family | Max Threads/TG | TG Memory | Reg Guide | Notes |
//! |---|---|---|---|---|
//! | Apple7 (M1) | 1024 | ~32KB | 128 | Fixed allocation |
//! | Apple8 (M2) | 1024 | ~32KB | 128 | Similar to M1 |
//! | Apple9 (M3) | 1024 | ~32KB | Dynamic | OMU-managed |
//! | Apple10 (M4) | 1024 | ~32KB | Dynamic | Improved OMU |
//! | Apple11 (M5) | 1024 | ~32KB | Dynamic | Smarter OMU |
//!
//! For M3+, register allocation is dynamically managed by the Occupancy
//! Management Unit (OMU).  Our 128-register guide is a soft heuristic —
//! the OMU may run shaders above or below this threshold depending on
//! cache pressure and available L1.  We model register pressure as a
//! gradual degradation, not a hard ceiling.
//!
//! ## Usage
//!
//! This module is not a [`Pass`] — it runs as post-pipeline analysis that
//! feeds into the autotuner.
//!
//! ## References
//! - Apple (2023), "Explore GPU advancements in M3 and A17 Pro",
//!   WWDC Tech Talks.  Introduced the OMU and dynamic register allocation.
//!   https://developer.apple.com/videos/play/tech-talks/111375
//! - Apple (2021), "Create image processing apps powered by Apple silicon",
//!   WWDC21.  Register pressure and occupancy trade-offs on Apple GPUs.
//!   https://developer.apple.com/videos/play/wwdc2021/10153/
//! - Apple, "Finding your Metal app's GPU occupancy", Xcode documentation.
//!   https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy
//! - Rosenzweig (2021), "Dissecting the Apple M1 GPU, part III",
//!   Asahi Linux blog.  Reverse-engineered register file and occupancy details.
//!   https://alyssarosenzweig.ca/blog/asahi-gpu-part-3/
//! - Poletto & Sarkar (1999), "Linear scan register allocation",
//!   ACM TOPLAS 21(5):895–913.  Foundation for the linear-scan liveness
//!   model used in [`register_estimate`].

use metaltile_core::ir::Kernel;

use super::register_estimate;

/// Per-GPU-family resource limits.
#[derive(Debug, Clone, Copy)]
pub struct GpuLimits {
    /// Maximum threads per threadgroup.
    pub max_threads_per_tg: u32,
    /// Threadgroup memory in bytes.
    pub tg_memory_bytes: u32,
    /// Soft register guide (not a hard ceiling on M3+ where the OMU
    /// dynamically allocates registers).
    pub regs_per_thread_guide: u32,
}

impl Default for GpuLimits {
    fn default() -> Self {
        GpuLimits {
            max_threads_per_tg: 1024,
            tg_memory_bytes: 32 * 1024,
            regs_per_thread_guide: 128,
        }
    }
}

/// Bottleneck preventing higher occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// Register pressure is degrading occupancy.
    RegisterLimited,
    /// Threadgroup memory size is the limiting factor.
    MemoryLimited,
    /// Thread count is the limiting factor.
    ThreadLimited,
    /// Tile working set exceeds likely on-chip cache.
    /// The OMU may throttle occupancy to prevent L1 thrashing.
    /// (Set by the autotuner when tile dims are known; not computed in estimate_occupancy.)
    CachePressure,
}

/// Occupancy estimate for a kernel with a given threadgroup size.
#[derive(Debug, Clone)]
pub struct OccupancyEstimate {
    /// Estimated occupancy as a percentage (0.0–100.0).
    pub occupancy_pct: f64,
    /// The primary bottleneck.
    pub bottleneck: Bottleneck,
    /// Upper bound on simultaneous threadgroups per shader core.
    ///
    /// Computed from the simple resource model. The actual count is decided
    /// by the OMU at runtime and may be lower due to cache pressure.
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

    // --- Register pressure (soft degradation, not a hard ceiling) ---
    //
    // On M3+, the OMU dynamically allocates registers. High register usage
    // degrades occupancy gradually rather than hitting a hard cliff at 128.
    // At ≤ the guide value, no penalty. Beyond that, linear degradation
    // to 10% at double the guide (i.e., at 256 regs/thr).
    let reg_occ = if reg_est.regs_per_thread <= limits.regs_per_thread_guide as usize {
        1.0
    } else {
        let excess = reg_est.regs_per_thread as f64 - limits.regs_per_thread_guide as f64;
        (1.0 - excess / limits.regs_per_thread_guide as f64).max(0.1)
    };

    // --- Thread-limited occupancy ---
    //
    // Hard ceiling: max 1024 threads per threadgroup on Apple GPUs.
    let thr_occ = limits.max_threads_per_tg as f64 / threadgroup_size as f64;
    let thr_occ = thr_occ.min(1.0);

    // --- Threadgroup memory ---
    //
    // Hard ceiling: max 32 KB per threadgroup on Apple GPUs.
    let mem_occ = if let Some(mem_used) = tg_mem_usage_bytes {
        if mem_used == 0 { 1.0 } else { (limits.tg_memory_bytes as f64 / mem_used as f64).min(1.0) }
    } else {
        1.0
    };

    // Occupancy is the minimum across all dimensions.
    let mut occ = reg_occ.min(thr_occ).min(mem_occ);
    occ = (occ * 1000.0).round() / 1000.0;

    // --- Bottleneck identification ---
    //
    // Pick the strictest limiter. When multiple limiters are within rounding
    // tolerance of each other, we report the most actionable one (register
    // pressure > memory > thread count).
    let bottleneck = if occ >= 0.999 {
        Bottleneck::ThreadLimited
    } else if reg_occ <= mem_occ && reg_occ <= thr_occ || (reg_occ - occ).abs() < 0.002 {
        Bottleneck::RegisterLimited
    } else if mem_occ <= thr_occ || (mem_occ - occ).abs() < 0.002 {
        Bottleneck::MemoryLimited
    } else {
        Bottleneck::ThreadLimited
    };

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

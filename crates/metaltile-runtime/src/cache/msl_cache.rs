//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! MSL source‑generation cache.
//!
//! Codegen passes (vectorize, fusion, …) cost tens of microseconds per
//! kernel.  At short context (`n_kv ≤ 1 K`) total iteration time is
//! ~40 µs, so re‑running passes every dispatch is a significant
//! fraction.  This cache stores generated MSL strings keyed by the
//! FNV‑1a `pso_cache_key` (kernel name + first‑param dtype + sorted
//! fn_consts) and returns the cached string on hit.

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::ir::Kernel;
#[cfg(target_os = "macos")]
use parking_lot::Mutex;
#[cfg(target_os = "macos")]
use rustc_hash::FxHashMap;

use crate::error::MetalTileError;

// ---------------------------------------------------------------------------
// MSL cache type
// ---------------------------------------------------------------------------

/// Thread‑safe MSL source cache.
///
/// Keys are the same FNV‑1a hashes used by
/// [`PsoCache`](super::pso_cache::PsoCache) so callers compute the key
/// once and check both caches with it.
pub(crate) struct MslCache {
    #[cfg(target_os = "macos")]
    cache: Mutex<FxHashMap<u64, String>>,
    #[cfg(not(target_os = "macos"))]
    _private: (),
}

impl MslCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            MslCache { cache: Mutex::new(FxHashMap::default()) }
        }
        #[cfg(not(target_os = "macos"))]
        {
            MslCache { _private: () }
        }
    }

    /// Return the MSL source for `kernel`, generating it on miss.
    ///
    /// `key` is the FNV‑1a hash produced by
    /// [`pso_cache_key`](super::pso_cache::pso_cache_key).  The
    /// caller should compute it once and pass it here and to
    /// `PsoCache::get_or_compile`.
    ///
    /// Releases the lock around the codegen call to keep critical
    /// sections short.  Double‑computing on a concurrent miss is
    /// acceptable — `MslGenerator::default().generate(kernel)` is
    /// pure, the second writer overwrites with an identical string.
    #[cfg(target_os = "macos")]
    pub(crate) fn get_or_generate(
        &self,
        kernel: &Kernel,
        key: u64,
    ) -> Result<String, MetalTileError> {
        if let Some(cached) = self.cache.lock().get(&key).cloned() {
            return Ok(cached);
        }
        let generated = MslGenerator::default().generate(kernel)?;
        self.cache.lock().insert(key, generated.clone());
        Ok(generated)
    }
}

#[cfg(all(target_os = "macos", test))]
mod perf {
    //! `#[ignore]`'d microbench for the lock + map cost — runs under
    //!
    //! ```text
    //! cargo test -p metaltile-runtime --release perf_cache_lock_throughput \
    //!     -- --ignored --nocapture
    //! ```
    //!
    //! per the playbook §"Measurement infrastructure".  Times raw
    //! lock/get cycles against a pre-populated cache so the swap from
    //! `std::sync::Mutex` → `parking_lot::Mutex` is independently
    //! measurable.  All ops hit the same key range so no codegen
    //! happens on the timed path — we're measuring pure lock + hash +
    //! lookup overhead.

    use std::{hint::black_box, time::Instant};

    use super::*;

    #[test]
    #[ignore]
    fn perf_cache_lock_throughput() {
        const N: u64 = 5_000_000;

        // ── parking_lot::Mutex (the new state — what shipped) ──
        let pl_cache = MslCache::new();
        {
            let mut guard = pl_cache.cache.lock();
            for key in 0u64..256 {
                guard.insert(key, format!("kernel_{key}"));
            }
        }
        for _ in 0..1_000 {
            black_box(pl_cache.cache.lock().get(&0u64));
        }
        let t0 = Instant::now();
        for i in 0..N {
            black_box(pl_cache.cache.lock().get(&(i & 0xff)));
        }
        let pl_elapsed = t0.elapsed();
        let pl_ns = pl_elapsed.as_nanos() as f64 / N as f64;

        // ── std::sync::Mutex (the old state — for direct comparison) ──
        let std_cache: std::sync::Mutex<FxHashMap<u64, String>> =
            std::sync::Mutex::new(FxHashMap::default());
        {
            let mut guard = std_cache.lock().unwrap();
            for key in 0u64..256 {
                guard.insert(key, format!("kernel_{key}"));
            }
        }
        for _ in 0..1_000 {
            black_box(std_cache.lock().unwrap().get(&0u64));
        }
        let t0 = Instant::now();
        for i in 0..N {
            black_box(std_cache.lock().unwrap().get(&(i & 0xff)));
        }
        let std_elapsed = t0.elapsed();
        let std_ns = std_elapsed.as_nanos() as f64 / N as f64;

        let speedup = std_ns / pl_ns;
        let delta_pct = (1.0 - pl_ns / std_ns) * 100.0;
        println!();
        println!("=== cache lock+get throughput ({N} iters each) ===");
        println!("  std::sync::Mutex      : {std_elapsed:?}  ({std_ns:.2} ns/op)");
        println!("  parking_lot::Mutex    : {pl_elapsed:?}  ({pl_ns:.2} ns/op)");
        println!("  speedup               : {speedup:.2}× ({delta_pct:+.1}%)");

        // Defense-in-depth: the swap should never be a regression. If
        // parking_lot is slower than std::sync on this hardware, the
        // PR's perf claim is invalid.
        assert!(
            pl_ns <= std_ns * 1.05,
            "parking_lot::Mutex regressed vs std::sync::Mutex \
             (pl={pl_ns:.2} ns, std={std_ns:.2} ns) — investigate before merging"
        );
    }
}

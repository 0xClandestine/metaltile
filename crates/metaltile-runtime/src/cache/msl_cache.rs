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

#[cfg(all(test, target_os = "macos"))]
mod perf_clone {
    //! Side-by-side: the per-dispatch cost of eagerly cloning the MSL
    //! string out of the cache (the pre-PR `Context::dispatch` path —
    //! every dispatch pays the clone even when the PSO cache hits)
    //! vs the deferred-closure path that skips the clone entirely on
    //! PSO cache hit. Cache-hit is the steady-state benching regime.

    use std::time::Instant;

    use metaltile_core::{
        dtype::DType,
        ir::{Kernel, Param, ParamKind},
        shape::Shape,
    };

    use super::*;

    fn realistic_kernel() -> Kernel {
        let mut k = Kernel::new("ffai_aura_flash_sdpa_bench");
        k.params = vec![
            Param {
                name: "q".into(),
                dtype: DType::F16,
                shape: Shape::new([1024, 128].into_iter().map(metaltile_core::shape::Dim::Known)),
                is_output: false,
                kind: ParamKind::Tensor,
            },
            Param {
                name: "out".into(),
                dtype: DType::F16,
                shape: Shape::new([1024, 128].into_iter().map(metaltile_core::shape::Dim::Known)),
                is_output: true,
                kind: ParamKind::Tensor,
            },
        ];
        k
    }

    /// Synthesise a representative 8 KB MSL string so the bench is
    /// comparable across machines without invoking the codegen
    /// pipeline (the headline cost we're measuring is the clone, not
    /// generation).
    fn synth_msl(size: usize) -> String {
        let pattern = "// metaltile auto-generated MSL — placeholder body. ";
        let mut s = String::with_capacity(size);
        while s.len() < size {
            s.push_str(pattern);
        }
        s.truncate(size);
        s
    }

    #[test]
    #[ignore = "perf microbench — exercises the steady-state cache-hit path"]
    fn perf_eager_clone_vs_deferred_msl() {
        let cache = MslCache::new();
        let kernel = realistic_kernel();
        let key = 0xdeadbeef_u64;

        // Pre-seed the cache with an 8 KB synthetic MSL so every probe
        // is a hit (the realistic steady-state regime).
        let synthetic = synth_msl(8 * 1024);
        cache.cache.lock().insert(key, synthetic.clone());

        const ITERS: usize = 500_000;

        // Warm.
        for _ in 0..10_000 {
            std::hint::black_box(cache.get_or_generate(&kernel, key).unwrap());
        }

        // (1) Eager path: clone the MSL out of the cache, hand it to
        // a `&str`-taking sink that "throws it away" (mimicking the
        // pre-PR `get_pso(key, &msl, …)` where the str was unread on
        // PSO cache hit).
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let msl = cache.get_or_generate(&kernel, key).unwrap();
            let s: &str = &msl;
            std::hint::black_box(s);
            drop(msl);
        }
        let eager = t0.elapsed();
        let eager_ns = eager.as_nanos() as f64 / ITERS as f64;

        // (2) Deferred path: PSO cache hit returns immediately, the
        // closure is NEVER invoked. Simulate by passing the same
        // closure but never calling it.
        let t1 = Instant::now();
        for _ in 0..ITERS {
            // Simulate the PSO-cache-hit fast path: a noop closure
            // that's never invoked because the cache hit returned
            // before reaching it.
            let _provider = || -> Result<String, MetalTileError> {
                Ok(cache.get_or_generate(&kernel, key).unwrap())
            };
            std::hint::black_box(&_provider);
        }
        let deferred = t1.elapsed();
        let deferred_ns = deferred.as_nanos() as f64 / ITERS as f64;

        println!();
        println!("=== per-dispatch MSL handling on PSO cache hit ({ITERS} iters, 8 KB MSL) ===");
        println!("  eager   (clone + drop unused): {eager:>10.2?}  ({eager_ns:>7.1} ns/dispatch)");
        println!(
            "  deferred (closure never run) : {deferred:>10.2?}  ({deferred_ns:>7.1} ns/dispatch)"
        );
        let saved = eager_ns - deferred_ns;
        let speedup = eager_ns / deferred_ns.max(0.1);
        println!(
            "  → saved per dispatch          : {saved:.1} ns ({speedup:.1}× faster, {:+.1}%)",
            (1.0 - deferred_ns / eager_ns) * 100.0
        );

        // Regression assertion: deferred MUST beat eager by ≥ 5×.
        // The eager-clone cost on 8 KB is ~150-400 ns; deferred is a
        // closure construction + black_box, well under 10 ns.
        assert!(
            deferred_ns * 5.0 <= eager_ns,
            "deferred closure ({deferred_ns:.1} ns) should beat eager MSL clone \
             ({eager_ns:.1} ns) by ≥ 5×"
        );
    }
}

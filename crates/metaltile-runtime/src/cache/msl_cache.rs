//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! MSL source‑generation cache.
//!
//! Codegen passes (vectorize, fusion, …) cost tens of microseconds per
//! kernel.  At short context (`n_kv ≤ 1 K`) total iteration time is
//! ~40 µs, so re‑running passes every dispatch is a significant
//! fraction.  This cache stores generated MSL strings keyed by the
//! FNV‑1a `pso_cache_key` (kernel name + first‑param dtype + sorted
//! fn_consts) and returns the cached string on hit.

#[cfg(target_os = "macos")]
use std::sync::Mutex;

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::ir::Kernel;
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
    #[cfg(target_os = "macos")]
    pub(crate) fn get_or_generate(
        &self,
        kernel: &Kernel,
        key: u64,
    ) -> Result<String, MetalTileError> {
        // Drop the guard BEFORE the match — Mutex isn't reentrant,
        // and temporaries in a match scrutinee live until the end
        // of the match body (RFC 66), so writing back inside `None`
        // would deadlock against the still‑held guard.
        let cached = self
            .cache
            .lock()
            .map_err(|_| MetalTileError::LockPoisoned("MSL cache".into()))?
            .get(&key)
            .cloned();

        match cached {
            Some(msl) => Ok(msl),
            None => {
                let generated = MslGenerator::default().generate(kernel)?;
                self.cache
                    .lock()
                    .map_err(|_| MetalTileError::LockPoisoned("MSL cache".into()))?
                    .insert(key, generated.clone());
                Ok(generated)
            },
        }
    }
}

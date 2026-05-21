//! Autotuner: persistent tuning cache for kernel schedules.
//!
//! The autotuner stores the best schedule configuration for each
//! (kernel, chip, shape_bucket) combination. Configs are persisted
//! to `~/.cache/metaltile/<chip>/tuning_cache.json`.
//!
//! ## Phase 1 (current)
//!
//! - [`TuneCache::lookup`] buckets the supplied [`ConstExprValues`] and
//!   does an exact lookup by stable string key.
//! - [`Autotuner::tune`] takes a caller-supplied bench closure, walks the
//!   [`config_space::KernelFamily`] config space, and caches the winner.
//! - [`Autotuner::get_or_tune`] still only does cache lookup. The search
//!   is invoked offline by `tile autotune` to warm the disk cache; the
//!   hot path stays branchless.
//!
//! Phase 2 will replace the search with a learned predictor (see
//! `metaltile-planning/`).

pub mod config_space;

use std::{collections::BTreeMap, path::PathBuf, time::Duration};

pub use config_space::KernelFamily;
use metaltile_core::constexpr::ConstExprValues;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// A single autotune configuration: tile sizes, thread layout, etc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneConfig {
    /// Tile dimensions (M, N, K for matmul-style ops).
    pub tile_dims: Vec<usize>,
    /// Threads per threadgroup (x, y, z).
    pub threads: (u32, u32, u32),
    /// Unroll factor for inner loops.
    pub unroll_factor: u32,
    /// Whether to use SIMD matrix multiply.
    pub use_simd_matrix: bool,
    /// Whether to use async copy for streaming.
    pub use_async_copy: bool,
}

impl Default for TuneConfig {
    fn default() -> Self {
        TuneConfig {
            tile_dims: vec![32, 32, 32],
            threads: (256, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: true,
            use_async_copy: false,
        }
    }
}

/// A shape bucket: ranges of dimension values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShapeBucket {
    /// Which constexpr dimension this bucket covers (by name).
    pub dim_name: String,
    /// Lower bound (inclusive).
    pub lo: usize,
    /// Upper bound (exclusive).
    pub hi: usize,
}

/// Coarse power-of-four-ish breakpoints. Picked so a 100-element shape
/// and a 200-element shape land in the same bucket — autotune cost
/// scales with bucket count, and same-bucket shapes mostly want the
/// same schedule.
const BUCKET_BREAKS: &[usize] = &[0, 256, 1024, 4096, 16384, 65_536, 262_144];

/// Round `value` to the half-open `[lo, hi)` bucket it falls in.
pub fn bucket_value(value: usize) -> (usize, usize) {
    for w in BUCKET_BREAKS.windows(2) {
        if value < w[1] {
            return (w[0], w[1]);
        }
    }
    (*BUCKET_BREAKS.last().unwrap(), usize::MAX)
}

/// Bucket every constexpr value, sorted by name (deterministic via
/// [`ConstExprValues`]'s `BTreeMap` backing).
pub fn bucket_constexprs(ce: &ConstExprValues) -> Vec<ShapeBucket> {
    ce.iter()
        .map(|(name, &v)| {
            let (lo, hi) = bucket_value(v);
            ShapeBucket { dim_name: name.clone(), lo, hi }
        })
        .collect()
}

/// Stable cache key: `"{kernel}#{name1}={lo1}..{hi1}#..."`. Used as the
/// `entries` map key in [`TuneCache`] and as the disk key.
pub fn cache_key(kernel_name: &str, ce: &ConstExprValues) -> String {
    let mut s = String::with_capacity(kernel_name.len() + 16);
    s.push_str(kernel_name);
    for b in bucket_constexprs(ce) {
        // hi==usize::MAX prints as `..`, the open right edge of the
        // last bucket. Stays serde-stable across machines.
        if b.hi == usize::MAX {
            s.push_str(&format!("#{}={}..", b.dim_name, b.lo));
        } else {
            s.push_str(&format!("#{}={}..{}", b.dim_name, b.lo, b.hi));
        }
    }
    s
}

/// A single tuning entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneEntry {
    /// Shape bucket this config is for.
    pub bucket: Vec<ShapeBucket>,
    /// The best configuration found.
    pub best_config: TuneConfig,
    /// Achieved performance (μs — lower is better).
    pub perf: f64,
    /// When this entry was last updated (unix seconds).
    pub timestamp: u64,
}

/// Persistent autotune cache.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TuneCache {
    /// entries[cache_key] = best config
    entries: BTreeMap<String, TuneEntry>,
}

impl TuneCache {
    /// Load from disk, or create empty.
    pub fn load(path: &PathBuf) -> Self {
        if path.exists() {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            TuneCache::default()
        }
    }

    /// Save to disk.
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Look up the best config for `(kernel_name, bucketed constexprs)`.
    /// Returns `Some` only on exact bucket-key match.
    pub fn lookup(&self, kernel_name: &str, constexprs: &ConstExprValues) -> Option<&TuneEntry> {
        let key = cache_key(kernel_name, constexprs);
        self.entries.get(&key)
    }

    /// Insert or update a tuning entry.
    pub fn insert(&mut self, key: impl Into<String>, entry: TuneEntry) {
        self.entries.insert(key.into(), entry);
    }

    /// Number of entries (for diagnostics / tests).
    pub fn len(&self) -> usize { self.entries.len() }

    /// True iff the cache holds no entries.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Iterate over `(key, entry)` pairs — used by the CLI's
    /// `tile autotune --dump` and by future training-data exporters.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TuneEntry)> { self.entries.iter() }
}

/// Inputs needed to benchmark a kernel candidate during tuning. Mirrors
/// the args [`crate::Context::dispatch_with_options`] takes so callers
/// can build it once per shape and feed it to many configs.
#[derive(Debug, Default, Clone)]
pub struct BenchInput {
    pub buffers: BTreeMap<String, Vec<u8>>,
    pub fn_consts: BTreeMap<String, u32>,
}

/// The autotuner: coordinates tuning across kernel launches.
pub struct Autotuner {
    /// Disk cache path.
    cache_path: PathBuf,
    /// In-memory cache.
    cache: TuneCache,
    /// Whether autotune is enabled.
    enabled: bool,
}

impl Autotuner {
    /// Create a new autotuner with a cache directory.
    pub fn new(cache_dir: PathBuf, enabled: bool) -> Self {
        let cache_path = cache_dir.join("tuning_cache.json");
        let cache = TuneCache::load(&cache_path);

        Autotuner { cache_path, cache, enabled }
    }

    /// Default cache directory: `~/.cache/metaltile/`.
    pub fn default_cache_dir() -> PathBuf {
        dirs_next().unwrap_or_else(|| PathBuf::from(".cache")).join("metaltile")
    }

    /// Enable or disable autotuning.
    pub fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }

    /// Whether the autotuner is currently enabled.
    pub fn enabled(&self) -> bool { self.enabled }

    /// Borrow the underlying cache (e.g. for the CLI to dump entries).
    pub fn cache(&self) -> &TuneCache { &self.cache }

    /// Map a kernel name to a [`KernelFamily`] via prefix matching. Used
    /// by the CLI to pick a config space when the caller doesn't specify
    /// one explicitly.
    pub fn infer_family(kernel_name: &str) -> KernelFamily {
        config_space::infer_family(kernel_name)
    }

    /// Get the best known config, or `None` if the cache misses.
    ///
    /// Phase 1: when autotune is disabled we return [`TuneConfig::default`];
    /// when enabled, we serve hits from disk. Searches are run offline
    /// via `tile autotune` (see [`Autotuner::tune`]), not lazily here —
    /// keeping the hot path free of GPU dispatches and factory lookups.
    #[tracing::instrument(skip(self, constexprs), fields(key = %kernel_name))]
    pub fn get_or_tune(
        &mut self,
        kernel_name: &str,
        constexprs: &ConstExprValues,
    ) -> Option<TuneConfig> {
        if !self.enabled {
            return Some(TuneConfig::default());
        }

        if let Some(entry) = self.cache.lookup(kernel_name, constexprs) {
            debug!(
                bucket = ?entry.bucket,
                perf_us = entry.perf,
                "autotune cache hit",
            );
            return Some(entry.best_config.clone());
        }

        debug!("autotune cache miss — run `tile autotune` to warm");
        None
    }

    /// Run a search over `family.config_space()`, benchmark each
    /// candidate with `bench_fn`, cache the winner, and flush the cache
    /// to disk.
    ///
    /// `bench_fn(cfg) -> elapsed_us`. The caller is responsible for
    /// building a `Kernel` from `cfg`, applying any `SchedulePass`, and
    /// dispatching it on a `Context`. Decoupling avoids a cyclic
    /// borrow with `Context::tuner`, and makes the search trivially
    /// testable with mocked timings.
    ///
    /// On bench errors we log + skip the candidate. If every candidate
    /// errors we surface the last error.
    pub fn tune(
        &mut self,
        kernel_name: &str,
        family: KernelFamily,
        constexprs: &ConstExprValues,
        bench_fn: &mut dyn FnMut(&TuneConfig) -> Result<f64, crate::error::MetalTileError>,
    ) -> Result<TuneConfig, crate::error::MetalTileError> {
        let space = family.config_space();
        let key = cache_key(kernel_name, constexprs);
        info!(
            kernel = kernel_name,
            family = ?family,
            n_configs = space.len(),
            bucket_key = %key,
            "autotune search start",
        );

        let mut best: Option<(TuneConfig, f64)> = None;
        let mut last_err: Option<crate::error::MetalTileError> = None;
        for cfg in space.iter() {
            match bench_fn(cfg) {
                Ok(us) => {
                    info!(config = ?cfg, elapsed_us = us, "autotune candidate");
                    let take = best.as_ref().is_none_or(|(_, b)| us < *b);
                    if take {
                        best = Some((cfg.clone(), us));
                    }
                },
                Err(e) => {
                    warn!(config = ?cfg, error = %e, "autotune candidate failed");
                    last_err = Some(e);
                },
            }
        }

        let (winner, perf) = best.ok_or_else(|| {
            last_err.unwrap_or_else(|| {
                crate::error::MetalTileError::Autotune("config space was empty".into())
            })
        })?;

        info!(winner = ?winner, perf_us = perf, bucket_key = %key, "autotune winner");

        let entry = TuneEntry {
            bucket: bucket_constexprs(constexprs),
            best_config: winner.clone(),
            perf,
            timestamp: unix_secs_now(),
        };
        self.cache.insert(key, entry);
        self.flush()?;

        Ok(winner)
    }

    /// Persist the cache to disk.
    pub fn flush(&self) -> Result<(), crate::error::MetalTileError> {
        Ok(self.cache.save(&self.cache_path)?)
    }

    /// Export cache entries as predictor-training rows.
    ///
    /// Phase 2 will train a model that picks a `TuneConfig` from
    /// `(kernel, dtype, shape_bucket)` features. The cache already
    /// holds one row per winner; this method denormalizes the key —
    /// splitting `"mt_acos@f16#B=0..256#N=1024..4096"` into separate
    /// `kernel`, `dtype`, and `bucket` fields — and tags each row with
    /// the inferred `family`. Rows are emitted in cache key order so
    /// the JSONL output is stable across runs.
    pub fn export_training_data(&self) -> Vec<TrainingRow> {
        self.cache.iter().map(|(key, entry)| TrainingRow::from_entry(key, entry)).collect()
    }
}

/// One JSONL row of training data — a denormalized view of a single
/// `TuneEntry` for downstream consumption by the Phase 2 predictor
/// trainer. Schema is forward-compatible: new fields can be added
/// without breaking existing readers (serde uses field names).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingRow {
    /// Kernel name, e.g. `"mt_acos"`. Empty when the cache key has no
    /// `@dtype` suffix (legacy entries).
    pub kernel: String,
    /// Dtype label, e.g. `"f16"`. Empty for legacy entries.
    pub dtype: String,
    /// Inferred [`KernelFamily`], serialized as `"Matmul"` / etc.
    pub family: String,
    /// Bucket dimensions as `{name: [lo, hi]}`. `hi == usize::MAX` is
    /// preserved as-is — readers should treat it as `+∞`.
    pub bucket: BTreeMap<String, (usize, usize)>,
    pub best_config: TuneConfig,
    pub perf_us: f64,
    pub timestamp: u64,
}

impl TrainingRow {
    pub fn from_entry(key: &str, entry: &TuneEntry) -> Self {
        let (kernel, dtype) = split_entry_name(key);
        let family = format!("{:?}", config_space::infer_family(&kernel));
        let bucket = entry.bucket.iter().map(|b| (b.dim_name.clone(), (b.lo, b.hi))).collect();
        TrainingRow {
            kernel,
            dtype,
            family,
            bucket,
            best_config: entry.best_config.clone(),
            perf_us: entry.perf,
            timestamp: entry.timestamp,
        }
    }
}

/// Split a cache key like `"mt_acos@f16#B=0..256#N=1024..4096"` back
/// into `(kernel, dtype)`. Strips the trailing `#…` bucket fragment so
/// it's robust to future dim additions. Legacy keys without `@` get
/// `dtype = ""` and the whole prefix as `kernel`.
fn split_entry_name(key: &str) -> (String, String) {
    let prefix = key.split_once('#').map(|(p, _)| p).unwrap_or(key);
    match prefix.split_once('@') {
        Some((k, d)) => (k.to_string(), d.to_string()),
        None => (prefix.to_string(), String::new()),
    }
}

fn dirs_next() -> Option<PathBuf> { std::env::var("HOME").ok().map(PathBuf::from) }

fn unix_secs_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use metaltile_core::constexpr::ConstExprValues;

    use super::*;

    /// Unique scratch dir per test, rooted in `std::env::temp_dir()` so we
    /// don't trample the real `~/.cache/metaltile/` cache when running
    /// tests in parallel.
    fn scratch_dir() -> PathBuf {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "metaltile-autotune-test-{}-{}",
            std::process::id(),
            n,
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ce_with(name: &str, value: usize) -> ConstExprValues {
        let mut v = ConstExprValues::new();
        v.insert(name, value);
        v
    }

    fn sample_config() -> TuneConfig {
        TuneConfig {
            tile_dims: vec![32, 32, 32],
            threads: (256, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: true,
            use_async_copy: false,
        }
    }

    fn sample_entry() -> TuneEntry {
        TuneEntry {
            bucket: vec![ShapeBucket { dim_name: "N".into(), lo: 0, hi: 256 }],
            best_config: sample_config(),
            perf: 12.34,
            timestamp: 0,
        }
    }

    // ── bucket_value / bucket_constexprs / cache_key ──────────────────

    #[test]
    fn bucket_value_lands_small_shapes_in_first_bucket() {
        assert_eq!(bucket_value(0), (0, 256));
        assert_eq!(bucket_value(100), (0, 256));
        assert_eq!(bucket_value(255), (0, 256));
    }

    #[test]
    fn bucket_value_handles_breakpoints() {
        // 256 is the right edge of bucket 0 (exclusive) and the left
        // edge of bucket 1 (inclusive).
        assert_eq!(bucket_value(256), (256, 1024));
        assert_eq!(bucket_value(1023), (256, 1024));
        assert_eq!(bucket_value(1024), (1024, 4096));
    }

    #[test]
    fn bucket_value_open_right_edge_for_huge_shapes() {
        let (lo, hi) = bucket_value(10_000_000);
        assert_eq!(lo, 262_144);
        assert_eq!(hi, usize::MAX);
    }

    #[test]
    fn cache_key_is_stable_across_sorted_names() {
        // BTreeMap iteration sorts by name → A then B then K.
        let mut ce = ConstExprValues::new();
        ce.insert("K", 32);
        ce.insert("B", 4);
        ce.insert("A", 1);
        let k = cache_key("mt_kernel", &ce);
        assert_eq!(k, "mt_kernel#A=0..256#B=0..256#K=0..256");
    }

    // ── TuneCache ────────────────────────────────────────────────────

    #[test]
    fn cache_load_nonexistent_returns_default() {
        let c = TuneCache::load(&PathBuf::from("/definitely/nonexistent/path.json"));
        assert!(c.is_empty());
    }

    #[test]
    fn cache_insert_save_load_roundtrip() {
        let dir = scratch_dir();
        let path = dir.join("tuning_cache.json");
        let mut c = TuneCache::default();
        c.insert("kernel_a#N=0..256", sample_entry());
        c.save(&path).unwrap();
        assert!(path.exists());

        let loaded = TuneCache::load(&path);
        assert_eq!(loaded.len(), 1);
        let e = loaded.entries.get("kernel_a#N=0..256").expect("entry survived round-trip");
        assert_eq!(e.bucket[0].dim_name, "N");
        assert_eq!(e.best_config.tile_dims, vec![32, 32, 32]);
        assert_eq!(e.perf, 12.34);
    }

    #[test]
    fn cache_lookup_hits_after_insert() {
        // The motivating Phase 1 test from the plan: insert with N=100
        // (bucket 0..256), query with N=150 → still 0..256, must hit.
        let mut c = TuneCache::default();
        let ce_insert = ce_with("N", 100);
        c.insert(cache_key("mt_kernel", &ce_insert), sample_entry());

        let ce_query = ce_with("N", 150);
        let got = c.lookup("mt_kernel", &ce_query);
        assert!(got.is_some(), "150 and 100 share the 0..256 bucket → hit");
        assert_eq!(got.unwrap().perf, 12.34);
    }

    #[test]
    fn cache_lookup_misses_across_buckets() {
        let mut c = TuneCache::default();
        c.insert(cache_key("mt_kernel", &ce_with("N", 100)), sample_entry());
        // 500 sits in 256..1024, a different bucket → miss.
        assert!(c.lookup("mt_kernel", &ce_with("N", 500)).is_none());
    }

    #[test]
    fn cache_lookup_misses_across_kernels() {
        let mut c = TuneCache::default();
        c.insert(cache_key("mt_kernel_a", &ce_with("N", 100)), sample_entry());
        // Same shape, different kernel → cache key differs → miss.
        assert!(c.lookup("mt_kernel_b", &ce_with("N", 100)).is_none());
    }

    #[test]
    fn cache_save_creates_parent_dirs() {
        let dir = scratch_dir();
        let nested = dir.join("a").join("b").join("c").join("cache.json");
        let c = TuneCache::default();
        c.save(&nested).expect("save should mkdir -p the parents");
        assert!(nested.exists());
    }

    // ── Autotuner ────────────────────────────────────────────────────

    #[test]
    fn autotuner_get_or_tune_disabled_returns_default_config() {
        let dir = scratch_dir();
        let mut tuner = Autotuner::new(dir, false);
        let ce = ConstExprValues::new();
        let cfg = tuner.get_or_tune("any_kernel", &ce).expect("disabled tuner returns default");
        assert_eq!(cfg.tile_dims, vec![32, 32, 32]);
        assert_eq!(cfg.threads, (256, 1, 1));
        assert!(cfg.use_simd_matrix);
    }

    #[test]
    fn autotuner_get_or_tune_enabled_with_empty_cache_returns_none() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, true);
        let ce = ConstExprValues::new();
        assert!(t.get_or_tune("any_kernel", &ce).is_none());
    }

    #[test]
    fn autotuner_get_or_tune_enabled_hits_warmed_cache() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, true);
        let ce_warm = ce_with("N", 100);

        // Mocked bench: pick a deterministic winner inside the
        // Elementwise config space so we can assert against it below.
        let mut bench = |cfg: &TuneConfig| -> Result<f64, crate::error::MetalTileError> {
            Ok(if cfg.threads.0 == 1024 { 4.0 } else { 9.0 })
        };
        let warm_winner = t
            .tune("warm_kernel", KernelFamily::Elementwise, &ce_warm, &mut bench)
            .expect("tune should succeed");
        assert_eq!(warm_winner.threads.0, 1024);

        // Same bucket as warm shape → cache hit, returns the winner
        // without running bench.
        let hit = t.get_or_tune("warm_kernel", &ce_with("N", 200)).expect("cache hits");
        assert_eq!(hit.threads.0, 1024);
    }

    #[test]
    fn autotuner_set_enabled_flips_state() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, false);
        let ce = ConstExprValues::new();
        assert!(t.get_or_tune("k", &ce).is_some()); // disabled → default
        t.set_enabled(true);
        assert!(t.get_or_tune("k", &ce).is_none()); // enabled, empty cache → None
    }

    #[test]
    fn autotuner_flush_writes_cache_file() {
        let dir = scratch_dir();
        let t = Autotuner::new(dir.clone(), false);
        t.flush().expect("flush succeeds even on empty cache");
        assert!(dir.join("tuning_cache.json").exists());
    }

    #[test]
    fn default_cache_dir_uses_home_or_falls_back() {
        let p = Autotuner::default_cache_dir();
        assert!(p.ends_with("metaltile"));
    }

    // ── tune() search loop ───────────────────────────────────────────

    #[test]
    fn tune_picks_faster_config_and_caches_it() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir.clone(), true);
        let ce = ce_with("N", 256);

        // Mocked bench: synthesize a perf landscape where the
        // 512-thread / SIMD-on config is fastest.
        let mut bench = |cfg: &TuneConfig| -> Result<f64, crate::error::MetalTileError> {
            let mut us = 100.0;
            if cfg.threads.0 == 512 {
                us -= 50.0;
            }
            if cfg.use_simd_matrix {
                us -= 20.0;
            }
            Ok(us)
        };

        let winner = t
            .tune("mock_matmul", KernelFamily::Matmul, &ce, &mut bench)
            .expect("tune should pick a winner");
        assert_eq!(winner.threads.0, 512);
        assert!(winner.use_simd_matrix);

        // Cache was flushed: a fresh Autotuner reading the same dir
        // sees the winner.
        let mut t2 = Autotuner::new(dir, true);
        let got = t2.get_or_tune("mock_matmul", &ce).expect("disk cache hit");
        assert_eq!(got.threads.0, 512);
        assert!(got.use_simd_matrix);
    }

    #[test]
    fn tune_skips_failing_candidates_and_returns_a_winner() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, true);
        let ce = ce_with("M", 128);

        let mut bench = |cfg: &TuneConfig| -> Result<f64, crate::error::MetalTileError> {
            // Fail anything that asks for async copy; pick a non-zero
            // winner among the rest.
            if cfg.use_async_copy {
                return Err(crate::error::MetalTileError::Autotune("simulated".into()));
            }
            Ok(if cfg.threads.0 == 256 { 9.0 } else { 12.0 })
        };

        let winner = t.tune("mock_red", KernelFamily::Reduction, &ce, &mut bench).expect("winner");
        assert!(!winner.use_async_copy);
        assert_eq!(winner.threads.0, 256);
    }

    #[test]
    fn tune_surfaces_last_error_when_every_candidate_fails() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, true);
        let ce = ce_with("N", 256);

        let mut bench = |_cfg: &TuneConfig| -> Result<f64, crate::error::MetalTileError> {
            Err(crate::error::MetalTileError::Autotune("everything broken".into()))
        };

        let err = t.tune("mock_doomed", KernelFamily::Matmul, &ce, &mut bench).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("everything broken"), "got: {msg}");
    }

    // ── infer_family ─────────────────────────────────────────────────

    #[test]
    fn infer_family_routes_by_kernel_name_prefix() {
        assert!(matches!(Autotuner::infer_family("mt_gemv_f32"), KernelFamily::Matmul));
        // mt_sdpa_prefill_* moved to its own SdpaPrefill family
        // in the same commit that wired SdpaPrefillMeasurer; sdpa
        // kernels without the prefill prefix (e.g. mt_sdpa_attention)
        // still land in Matmul via the `sdpa` token below.
        assert!(matches!(
            Autotuner::infer_family("mt_sdpa_prefill_mma"),
            KernelFamily::SdpaPrefill
        ));
        assert!(matches!(Autotuner::infer_family("mt_sdpa_decode_2pass"), KernelFamily::Decode));
        assert!(matches!(Autotuner::infer_family("mt_rms_norm_f16"), KernelFamily::Reduction));
        assert!(matches!(Autotuner::infer_family("mt_unary_acos_f32"), KernelFamily::Elementwise));
        // Unknown → Elementwise (smallest config space, safest fallback).
        assert!(matches!(Autotuner::infer_family("never_seen_kernel"), KernelFamily::Elementwise));
    }

    // ── ShapeBucket / TuneConfig / TuneEntry serde ───────────────────

    #[test]
    fn shape_bucket_serde_roundtrip() {
        let b = ShapeBucket { dim_name: "M".into(), lo: 0, hi: 128 };
        let s = serde_json::to_string(&b).unwrap();
        let b2: ShapeBucket = serde_json::from_str(&s).unwrap();
        assert_eq!(b, b2);
    }

    #[test]
    fn tune_config_serde_roundtrip() {
        let c = sample_config();
        let s = serde_json::to_string(&c).unwrap();
        let c2: TuneConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(c2.tile_dims, c.tile_dims);
        assert_eq!(c2.threads, c.threads);
        assert_eq!(c2.unroll_factor, c.unroll_factor);
        assert_eq!(c2.use_simd_matrix, c.use_simd_matrix);
        assert_eq!(c2.use_async_copy, c.use_async_copy);
    }

    #[test]
    fn tune_config_default_matches_disabled_fallback() {
        let d = TuneConfig::default();
        assert_eq!(d.tile_dims, vec![32, 32, 32]);
        assert_eq!(d.threads, (256, 1, 1));
        assert!(d.use_simd_matrix);
        assert!(!d.use_async_copy);
    }

    // ── TrainingRow / export_training_data ───────────────────────────

    #[test]
    fn split_entry_name_handles_kernel_at_dtype_with_bucket() {
        let (k, d) = split_entry_name("mt_acos@f16#B=0..256#N=1024..4096");
        assert_eq!(k, "mt_acos");
        assert_eq!(d, "f16");
    }

    #[test]
    fn split_entry_name_handles_no_bucket_fragment() {
        let (k, d) = split_entry_name("mt_acos@f32");
        assert_eq!(k, "mt_acos");
        assert_eq!(d, "f32");
    }

    #[test]
    fn split_entry_name_handles_legacy_key_without_dtype() {
        let (k, d) = split_entry_name("legacy_kernel#N=0..256");
        assert_eq!(k, "legacy_kernel");
        assert_eq!(d, "");
    }

    #[test]
    fn training_row_from_entry_populates_all_fields() {
        let entry = TuneEntry {
            bucket: vec![ShapeBucket { dim_name: "B".into(), lo: 0, hi: 256 }, ShapeBucket {
                dim_name: "N".into(),
                lo: 1024,
                hi: 4096,
            }],
            best_config: sample_config(),
            perf: 12.5,
            timestamp: 999,
        };
        let row = TrainingRow::from_entry("mt_unary_acos@f16#B=0..256#N=1024..4096", &entry);
        assert_eq!(row.kernel, "mt_unary_acos");
        assert_eq!(row.dtype, "f16");
        // mt_unary_* → Elementwise by infer_family.
        assert_eq!(row.family, "Elementwise");
        assert_eq!(row.bucket.get("N"), Some(&(1024usize, 4096usize)));
        assert_eq!(row.bucket.get("B"), Some(&(0usize, 256usize)));
        assert_eq!(row.perf_us, 12.5);
        assert_eq!(row.timestamp, 999);
        assert_eq!(row.best_config.threads, sample_config().threads);
    }

    #[test]
    fn export_training_data_empty_cache_returns_empty_vec() {
        let dir = scratch_dir();
        let t = Autotuner::new(dir, true);
        assert!(t.export_training_data().is_empty());
    }

    #[test]
    fn export_training_data_emits_one_row_per_cache_entry_sorted_by_key() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, true);
        // Insert deliberately out of order to confirm the output is
        // sorted (cache iter is BTreeMap-backed).
        let mk = |bucket_lo: usize| TuneEntry {
            bucket: vec![ShapeBucket { dim_name: "N".into(), lo: bucket_lo, hi: bucket_lo + 256 }],
            best_config: sample_config(),
            perf: bucket_lo as f64,
            timestamp: 1,
        };
        t.cache.insert("mt_kernel@f16#N=1024..1280", mk(1024));
        t.cache.insert("mt_kernel@f16#N=0..256", mk(0));
        t.cache.insert("mt_kernel@bf16#N=0..256", mk(0));

        let rows = t.export_training_data();
        assert_eq!(rows.len(), 3);
        // BTreeMap key order: bf16 < f16, then N ascending.
        assert_eq!(rows[0].dtype, "bf16");
        assert_eq!(rows[1].dtype, "f16");
        assert_eq!(rows[1].bucket.get("N"), Some(&(0usize, 256usize)));
        assert_eq!(rows[2].bucket.get("N"), Some(&(1024usize, 1280usize)));
    }

    #[test]
    fn training_row_serde_roundtrip_preserves_fields() {
        let row = TrainingRow {
            kernel: "mt_acos".into(),
            dtype: "f32".into(),
            family: "Elementwise".into(),
            bucket: [("N".to_string(), (0usize, 256usize))].into_iter().collect(),
            best_config: sample_config(),
            perf_us: 7.5,
            timestamp: 42,
        };
        let json = serde_json::to_string(&row).unwrap();
        let parsed: TrainingRow = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, row);
    }
}

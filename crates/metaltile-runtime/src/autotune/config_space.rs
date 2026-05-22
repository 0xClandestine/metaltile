//! Per-family config space generators.
//!
//! Not every kernel benefits from the same set of knobs — matmul
//! variants care about (tile_dims, threads, SIMD-matrix); reductions
//! care mostly about threads + unroll; elementwise kernels just need
//! threads. Keeping the spaces small (≤6 configs each) keeps Phase 1
//! search budget bounded; the search is precisely an offline grid scan,
//! and Phase 2's predictor replaces it with a single function call.

use super::TuneConfig;

/// Coarse family bucket used to pick a config space + (later) train
/// a per-family predictor. We deliberately collapse "tensor-core
/// matmul" and "GEMV" into a single Matmul bucket: they share the
/// same knobs (tile_dims, SIMD-matrix on/off, threads) and the
/// per-shape winner is a function of the bucket, not the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFamily {
    Matmul,
    Reduction,
    Elementwise,
    Decode,
    /// Single-pass vector SDPA (`mt_sdpa_vector`). Hardcodes
    /// `BN × BD = 32 × 32 = 1024` threadgroup, so the search must
    /// keep `threads.0 = 1024` and vary only the orthogonal knobs.
    SdpaVector,
    /// Flash-Attention-2 prefill tile (`mt_sdpa_prefill`,
    /// `mt_sdpa_prefill_mma`, `mt_sdpa_prefill_mma_bf16`). Each
    /// kernel hardcodes its own (bq, bk, wm, wn) geometry and
    /// carries its own `tpg` on the dispatch — the measurer pins
    /// `expected_tpg` to that per-dispatch value at compile time.
    /// The family's `threads` field is therefore advisory; only the
    /// orthogonal knobs (simd_matrix, async_copy) drive winners.
    SdpaPrefill,
}

impl KernelFamily {
    /// Candidate configs for this family. Order is incidental — the
    /// search compares perf, not position.
    pub fn config_space(self) -> Vec<TuneConfig> {
        match self {
            KernelFamily::Matmul => matmul_space(),
            KernelFamily::Reduction => reduction_space(),
            KernelFamily::Elementwise => elementwise_space(),
            KernelFamily::Decode => decode_space(),
            KernelFamily::SdpaVector => sdpa_vector_space(),
            KernelFamily::SdpaPrefill => sdpa_prefill_space(),
        }
    }
}

fn matmul_space() -> Vec<TuneConfig> {
    let mut out = Vec::with_capacity(6);
    for &(tm, tn, tk) in &[(32usize, 32, 32), (64, 64, 16), (64, 64, 32)] {
        for &threads in &[(256u32, 1u32, 1u32), (512, 1, 1)] {
            out.push(TuneConfig {
                tile_dims: vec![tm, tn, tk],
                threads,
                unroll_factor: 4,
                use_simd_matrix: true,
                use_async_copy: false,
            });
        }
    }
    // One non-SIMD variant for cases where the SIMD matrix path is
    // off the critical path (e.g. small K). Important for the test
    // `tune_picks_faster_config_and_caches_it` to have a SIMD off vs
    // on contrast in the space.
    out.push(TuneConfig {
        tile_dims: vec![32, 32, 32],
        threads: (256, 1, 1),
        unroll_factor: 4,
        use_simd_matrix: false,
        use_async_copy: false,
    });
    out
}

fn reduction_space() -> Vec<TuneConfig> {
    let mut out = Vec::with_capacity(4);
    for &threads in &[(128u32, 1u32, 1u32), (256, 1, 1), (512, 1, 1), (1024, 1, 1)] {
        out.push(TuneConfig {
            tile_dims: vec![],
            threads,
            unroll_factor: 4,
            use_simd_matrix: false,
            use_async_copy: false,
        });
    }
    out
}

fn elementwise_space() -> Vec<TuneConfig> {
    let mut out = Vec::with_capacity(3);
    for &threads in &[(256u32, 1u32, 1u32), (512, 1, 1), (1024, 1, 1)] {
        out.push(TuneConfig {
            tile_dims: vec![],
            threads,
            unroll_factor: 4,
            use_simd_matrix: false,
            use_async_copy: false,
        });
    }
    out
}

fn decode_space() -> Vec<TuneConfig> {
    let mut out = Vec::with_capacity(4);
    for &simd_on in &[true, false] {
        for &async_on in &[true, false] {
            out.push(TuneConfig {
                tile_dims: vec![32, 32, 32],
                threads: (256, 1, 1),
                unroll_factor: 4,
                use_simd_matrix: simd_on,
                use_async_copy: async_on,
            });
        }
    }
    out
}

/// `mt_sdpa_vector` hardcodes the threadgroup at 1024 (32 × 32
/// simdgroup matrix), so varying `threads` would break the kernel —
/// it would either fail to compile (function-constant mismatch) or
/// silently produce zeros (per the register-pressure threadgroup-cap
/// gotcha). The search varies only the orthogonal knobs.
fn sdpa_vector_space() -> Vec<TuneConfig> {
    let mut out = Vec::with_capacity(4);
    for &simd_on in &[true, false] {
        for &async_on in &[true, false] {
            out.push(TuneConfig {
                tile_dims: vec![],
                threads: (1024, 1, 1),
                unroll_factor: 4,
                use_simd_matrix: simd_on,
                use_async_copy: async_on,
            });
        }
    }
    out
}

/// Flash-Attention-2 prefill tile (`mt_sdpa_prefill*`). Each kernel
/// carries its own `tpg` on the dispatch — the measurer pins
/// `expected_tpg` to that per-dispatch value, so the family's
/// `threads` field is just a placeholder for the schedule pass.
/// Same `(simd × async)` quadrant shape as SdpaVector.
fn sdpa_prefill_space() -> Vec<TuneConfig> {
    let mut out = Vec::with_capacity(4);
    for &simd_on in &[true, false] {
        for &async_on in &[true, false] {
            out.push(TuneConfig {
                tile_dims: vec![],
                threads: (256, 1, 1),
                unroll_factor: 4,
                use_simd_matrix: simd_on,
                use_async_copy: async_on,
            });
        }
    }
    out
}

/// Best-effort kernel-name → family routing. Prefix matching only;
/// unknown kernels fall back to [`KernelFamily::Elementwise`] (smallest
/// config space, safest for a kernel we know nothing about).
///
/// The prefixes match the metaltile-std naming convention
/// (`mt_<op>_<variant>_<dtype>` for fused kernels, `mt_<op>` for plain).
pub fn infer_family(kernel_name: &str) -> KernelFamily {
    // Order matters: decode must match before plain "sdpa" because
    // `mt_sdpa_decode_*` is also "sdpa".
    if has_token(kernel_name, "decode") {
        return KernelFamily::Decode;
    }
    // `mt_sdpa_vector` is the single-pass vector SDPA kernel; its
    // hardcoded TPG=1024 needs a config space that *doesn't* sweep
    // threads (which the Matmul space does). The 2-pass form lives
    // under a different dispatch and routes elsewhere.
    if kernel_name == "mt_sdpa_vector" {
        return KernelFamily::SdpaVector;
    }
    // Flash-Attention-2 prefill tiles: each kernel hardcodes its own
    // (bq, bk, wm, wn) geometry and a fixed TPG, so the Matmul space
    // (which sweeps threads) would produce broken or duplicate
    // configs. Prefix-match catches `mt_sdpa_prefill`,
    // `mt_sdpa_prefill_mma`, and `mt_sdpa_prefill_mma_bf16`.
    if kernel_name.starts_with("mt_sdpa_prefill") {
        return KernelFamily::SdpaPrefill;
    }
    if has_token(kernel_name, "sdpa")
        || has_token(kernel_name, "gemm")
        || has_token(kernel_name, "gemv")
        || has_token(kernel_name, "matmul")
        || has_token(kernel_name, "qmm")
    {
        return KernelFamily::Matmul;
    }
    if has_token(kernel_name, "rms_norm")
        || has_token(kernel_name, "layer_norm")
        || has_token(kernel_name, "softmax")
        || has_token(kernel_name, "logsumexp")
        || has_token(kernel_name, "reduce")
        || has_token(kernel_name, "scan")
        || has_token(kernel_name, "arg_reduce")
        || has_token(kernel_name, "sort")
        // Flash-attention pass kernels run online-softmax reductions
        // (per-block m/l/o accumulators) — same shape as `softmax`
        // even though their name doesn't say so. `aura_flash_p1_*`
        // (Grid3D dispatch) and `aura_flash_pass2_*` (declared
        // `KernelMode::Reduction`) both want the Reduction config
        // space. The `aura_flash_sdpa_*` / `flash_quantized_sdpa_*`
        // forms stay in Matmul via the `sdpa` token below.
        || has_token(kernel_name, "flash_p1")
        || has_token(kernel_name, "flash_pass2")
    {
        return KernelFamily::Reduction;
    }
    KernelFamily::Elementwise
}

/// `name` contains `token` as either a substring after `_` or at the start.
/// Cheaper than a full regex and avoids `gemv` matching `genvecsum`.
fn has_token(name: &str, token: &str) -> bool {
    if name == token {
        return true;
    }
    if let Some(rest) = name.strip_prefix(token)
        && (rest.is_empty() || rest.starts_with('_'))
    {
        return true;
    }
    name.split('_').any(|seg| seg == token)
        || name.contains(&format!("_{token}_"))
        || name.contains(&format!("_{token}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_space_is_nonempty_and_includes_simd_off_variant() {
        let s = KernelFamily::Matmul.config_space();
        assert!(!s.is_empty());
        assert!(s.iter().any(|c| !c.use_simd_matrix), "should include simd-off variant");
        assert!(s.iter().any(|c| c.use_simd_matrix), "should include simd-on variant");
    }

    #[test]
    fn reduction_space_varies_thread_count() {
        let s = KernelFamily::Reduction.config_space();
        let threads: std::collections::BTreeSet<_> = s.iter().map(|c| c.threads.0).collect();
        // At least 3 distinct thread counts → meaningful search.
        assert!(threads.len() >= 3, "got: {threads:?}");
    }

    #[test]
    fn decode_space_covers_simd_async_quadrants() {
        let s = KernelFamily::Decode.config_space();
        assert_eq!(s.len(), 4);
        let quadrants: std::collections::BTreeSet<_> =
            s.iter().map(|c| (c.use_simd_matrix, c.use_async_copy)).collect();
        assert_eq!(quadrants.len(), 4);
    }

    #[test]
    fn infer_family_handles_known_prefixes() {
        assert_eq!(infer_family("mt_gemv_f32"), KernelFamily::Matmul);
        assert_eq!(infer_family("mt_qmm_int4_f16"), KernelFamily::Matmul);
        // Prefill kernels moved to their own family in the same
        // commit that wired SdpaPrefillMeasurer; a generic sdpa
        // kernel like mt_sdpa_attention would still hit Matmul.
        assert_eq!(infer_family("mt_sdpa_attention"), KernelFamily::Matmul);
        assert_eq!(infer_family("mt_sdpa_decode_2pass"), KernelFamily::Decode);
        assert_eq!(infer_family("mt_rms_norm_f16"), KernelFamily::Reduction);
        assert_eq!(infer_family("mt_logsumexp_f32"), KernelFamily::Reduction);
        assert_eq!(infer_family("mt_softmax"), KernelFamily::Reduction);
        assert_eq!(infer_family("mt_unary_acos_f32"), KernelFamily::Elementwise);
    }

    #[test]
    fn infer_family_decode_beats_sdpa_matmul() {
        // `mt_sdpa_decode_*` contains both "decode" and "sdpa" — the
        // decode bucket has tighter constraints (memory-bound, KV walk),
        // so route it there, not into Matmul.
        assert_eq!(infer_family("mt_sdpa_decode_2pass"), KernelFamily::Decode);
        assert_eq!(infer_family("mt_decode_only"), KernelFamily::Decode);
    }

    #[test]
    fn infer_family_unknown_falls_back_to_elementwise() {
        assert_eq!(infer_family("totally_made_up"), KernelFamily::Elementwise);
        assert_eq!(infer_family(""), KernelFamily::Elementwise);
    }

    #[test]
    fn sdpa_vector_space_pins_threads_at_1024_and_varies_simd_async() {
        // The kernel hardcodes BN×BD=32×32=1024, so every config in
        // this space MUST keep threads=(1024,1,1). Any drift would
        // either fail to compile or — worse — silently produce zeros.
        let s = KernelFamily::SdpaVector.config_space();
        assert!(!s.is_empty());
        for c in &s {
            assert_eq!(c.threads, (1024, 1, 1), "SdpaVector TPG must stay at 1024: {c:?}");
        }
        // All four (simd × async) quadrants present, like Decode.
        let quadrants: std::collections::BTreeSet<_> =
            s.iter().map(|c| (c.use_simd_matrix, c.use_async_copy)).collect();
        assert_eq!(quadrants.len(), 4, "expected all four simd×async quadrants");
    }

    #[test]
    fn infer_family_routes_mt_sdpa_vector_to_sdpa_vector() {
        assert_eq!(infer_family("mt_sdpa_vector"), KernelFamily::SdpaVector);
    }

    #[test]
    fn infer_family_does_not_route_other_sdpa_kernels_to_sdpa_vector() {
        // Adjacent names: prefill → SdpaPrefill; decode → Decode;
        // a hypothetical mt_sdpa_vector_something stays Matmul (the
        // mt_sdpa_vector kernel-name match is exact, not prefix).
        assert_eq!(infer_family("mt_sdpa_prefill_mma"), KernelFamily::SdpaPrefill);
        assert_eq!(infer_family("mt_sdpa_decode_2pass"), KernelFamily::Decode);
        assert_eq!(infer_family("mt_sdpa_vector_2pass"), KernelFamily::Matmul);
    }

    #[test]
    fn infer_family_routes_aura_flash_p1_and_pass2_to_reduction() {
        // Online-softmax pass kernels: classification was wrong
        // (Elementwise) — the inner loop is a reduction, and the
        // varying-threads space is what they want to tune.
        assert_eq!(infer_family("aura_flash_p1_kb4_vb2_d128"), KernelFamily::Reduction);
        assert_eq!(infer_family("aura_flash_p1_kb4_vb4_d64"), KernelFamily::Reduction);
        assert_eq!(infer_family("aura_flash_pass2_d128"), KernelFamily::Reduction);
        assert_eq!(infer_family("aura_flash_pass2_d512"), KernelFamily::Reduction);
    }

    #[test]
    fn sdpa_prefill_space_covers_simd_async_quadrants() {
        let s = KernelFamily::SdpaPrefill.config_space();
        assert!(!s.is_empty());
        let quadrants: std::collections::BTreeSet<_> =
            s.iter().map(|c| (c.use_simd_matrix, c.use_async_copy)).collect();
        assert_eq!(quadrants.len(), 4, "expected all four simd×async quadrants");
        // tile_dims stays empty — prefill geometry is baked into the
        // kernel constexprs, not pushed via the schedule.
        for c in &s {
            assert!(c.tile_dims.is_empty(), "prefill tile_dims should be empty: {c:?}");
        }
    }

    #[test]
    fn infer_family_routes_mt_sdpa_prefill_variants_to_sdpa_prefill() {
        assert_eq!(infer_family("mt_sdpa_prefill"), KernelFamily::SdpaPrefill);
        assert_eq!(infer_family("mt_sdpa_prefill_mma"), KernelFamily::SdpaPrefill);
        assert_eq!(infer_family("mt_sdpa_prefill_mma_bf16"), KernelFamily::SdpaPrefill);
    }

    #[test]
    fn infer_family_keeps_aura_flash_sdpa_in_matmul() {
        // The sdpa-variant flash kernels are the per-tile MMA path —
        // they belong in Matmul, not Reduction. The `sdpa` rule runs
        // before the `flash_*` block, so kernels with both tokens go
        // to Matmul as intended.
        assert_eq!(infer_family("aura_flash_sdpa_kb4_vb2_d128"), KernelFamily::Matmul);
        assert_eq!(infer_family("flash_quantized_sdpa_b4_d128"), KernelFamily::Matmul);
    }
}

//! Pure helpers: median + bucket-N + synthetic constexprs.

use metaltile_core::constexpr::ConstExprValues;
use metaltile_std::spec::{BenchDispatch, BenchSpec};

/// What value should the cache key's `N` bucket reflect for this
/// `(spec, n_override)` pair? Generic dispatch uses `--shapes`
/// override or `spec.shapes[0].n`; SdpaVector uses the dispatch's
/// hardcoded `n_kv`; other arms fall back to `n_override` (which is
/// `None` outside `--shapes`, so the legacy default kicks in).
pub(crate) fn effective_bucket_n(spec: &BenchSpec, n_override: Option<usize>) -> Option<usize> {
    match spec.dispatch {
        BenchDispatch::SdpaVector { n_kv, .. } => Some(n_kv),
        // For prefill, the bucketing dimension that matters most is
        // `k_len` (the KV sequence the tile walks). q_len is usually
        // smaller and more uniform across deployments.
        BenchDispatch::SdpaPrefill { k_len, .. } => Some(k_len),
        BenchDispatch::Generic => n_override,
        _ => n_override,
    }
}

/// Build a synthetic `ConstExprValues` for `spec` good enough to bucket.
///
/// `n_override` retargets the N constexpr — `--shapes` uses this to
/// land the same `BenchSpec` in different cache buckets. `B` always
/// comes from `spec.shapes[0]` (multi-shape today is 1-D over `N`).
pub(crate) fn synth_constexprs_for(spec: &BenchSpec, n_override: Option<usize>) -> ConstExprValues {
    let mut ce = ConstExprValues::new();
    let (n, b) = match (spec.shapes.first(), n_override) {
        (Some(first), Some(n)) => (n, first.b.max(1)),
        (Some(first), None) => (first.n, first.b.max(1)),
        (None, Some(n)) => (n, 1),
        (None, None) => (1024, 1),
    };
    ce.insert("N", n);
    ce.insert("B", b);
    ce
}

/// Filter out non-finite (NaN/±∞) and negative samples before sorting.
///
/// `GPUEndTime - GPUStartTime` can come back negative under DVFS edge
/// cases, and a negative-timing run would otherwise win every
/// comparison. NaNs sort as `Equal` under `partial_cmp`, which silently
/// corrupts the median. Caller gets an `Err` if no valid sample
/// remains so the candidate is treated as a measurement failure (and
/// falls through to static_cost), rather than a fake winner.
pub(crate) fn median_us(samples: Vec<f64>) -> Result<f64, String> {
    let mut valid: Vec<f64> = samples.into_iter().filter(|s| s.is_finite() && *s >= 0.0).collect();
    if valid.is_empty() {
        return Err("measure returned no finite, non-negative samples".into());
    }
    valid.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite values above"));
    let n = valid.len();
    Ok(if n % 2 == 1 { valid[n / 2] } else { (valid[n / 2 - 1] + valid[n / 2]) / 2.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use metaltile_core::dtype::DType;
    use metaltile_std::spec::{BenchDispatch, BenchSpec, ScalarBufSpec, ShapeSpec};

    #[test]
    fn median_us_returns_middle_for_odd_n() {
        assert_eq!(median_us(vec![3.0, 1.0, 2.0]).unwrap(), 2.0);
    }

    #[test]
    fn median_us_averages_middle_pair_for_even_n() {
        assert_eq!(median_us(vec![4.0, 1.0, 2.0, 3.0]).unwrap(), 2.5);
    }

    #[test]
    fn median_us_filters_nan_samples_before_sorting() {
        // Without filtering, NaN sorts as Equal and corrupts the median.
        let m = median_us(vec![3.0, f64::NAN, 1.0, 2.0]).unwrap();
        assert_eq!(m, 2.0);
    }

    #[test]
    fn median_us_filters_negative_and_infinite_samples() {
        // GPUEndTime - GPUStartTime can come back negative under DVFS;
        // such a sample would otherwise win every comparison.
        let m = median_us(vec![-5.0, f64::INFINITY, 4.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m, 2.5);
    }

    #[test]
    fn median_us_errors_when_all_samples_invalid() {
        let err = median_us(vec![f64::NAN, -1.0, f64::NEG_INFINITY]).unwrap_err();
        assert!(err.contains("finite"), "got: {err}");
    }

    #[test]
    fn median_us_errors_when_input_empty() {
        let err = median_us(vec![]).unwrap_err();
        assert!(err.contains("finite") || err.contains("no"), "got: {err}");
    }

    #[test]
    fn synth_constexprs_uses_override_when_present() {
        // Two overrides for the same spec land in different N buckets,
        // so cache_key (which composes the bucketed N) differs.
        let spec = mock_spec_with_n(100);
        let ce_low = synth_constexprs_for(&spec, Some(64));
        let ce_high = synth_constexprs_for(&spec, Some(16_384));
        let k_low = metaltile_runtime::autotune::cache_key("k", &ce_low);
        let k_high = metaltile_runtime::autotune::cache_key("k", &ce_high);
        assert_ne!(
            k_low, k_high,
            "different N overrides must produce different cache keys (got both {k_low})",
        );
    }

    #[test]
    fn synth_constexprs_falls_back_to_spec_shape_when_no_override() {
        // None override + spec.shapes[0].n=100 must match an explicit Some(100).
        let spec = mock_spec_with_n(100);
        let ce_default = synth_constexprs_for(&spec, None);
        let ce_explicit = synth_constexprs_for(&spec, Some(100));
        assert_eq!(
            metaltile_runtime::autotune::cache_key("k", &ce_default),
            metaltile_runtime::autotune::cache_key("k", &ce_explicit),
        );
    }

    #[test]
    fn synth_constexprs_with_no_shape_and_no_override_uses_defaults() {
        // No spec shape, no override → should still produce a stable key
        // (the static-cost-only fallback path).
        let spec = mock_spec_empty_shapes();
        let ce = synth_constexprs_for(&spec, None);
        let k = metaltile_runtime::autotune::cache_key("k", &ce);
        // 1024 lands in the 1024..4096 bucket.
        assert!(k.contains("N=1024..4096"), "got: {k}");
    }

    /// Build a `BenchSpec` literal good enough for `synth_constexprs_for`
    /// — only `shapes` is read by that helper, so most fields are
    /// placeholder constants.
    fn mock_spec_with_n(n: usize) -> BenchSpec {
        static TENSOR_BUFS: &[metaltile_std::spec::TensorBufSpec] = &[];
        static SCALAR_BUFS: &[ScalarBufSpec] = &[];
        static CEXPRS: &[(&str, metaltile_std::spec::Dim)] = &[];
        // Shapes can't easily be made `&'static` from a runtime n, so
        // construct a leaked slice for the test's lifetime.
        let shapes: &'static [ShapeSpec] = Box::leak(Box::new([ShapeSpec {
            label: "test",
            n,
            b: 1,
            check_n: n,
            check_b: 1,
            mode: metaltile_core::ir::KernelMode::Elementwise,
            tpg: 256,
            grid: metaltile_std::spec::DispatchGrid::DivCeilN,
            tensor_bufs: TENSOR_BUFS,
            scalar_bufs: SCALAR_BUFS,
            cexprs: CEXPRS,
            out_elems: metaltile_std::spec::Dim::N,
            reads: 1,
            bytes_fn: metaltile_std::spec::bytes_elementwise,
            mlx_args: None,
            mlx_grid: None,
            mlx_tpg: 256,
        }]));
        mock_spec(shapes)
    }

    fn mock_spec_empty_shapes() -> BenchSpec { mock_spec(&[]) }

    fn mock_spec(shapes: &'static [ShapeSpec]) -> BenchSpec {
        fn ir(_dt: DType) -> metaltile_core::ir::Kernel {
            // Never invoked by synth_constexprs_for — only the shapes are read.
            metaltile_core::ir::Kernel::new("mock")
        }
        BenchSpec {
            op: "test",
            subop: "test",
            kernel_name: "k",
            kernel_ir: ir,
            dtypes: &[],
            tol: 0.0,
            mlx_src: None,
            mlx_pattern: None,
            shapes,
            dispatch: BenchDispatch::Generic,
            kernel_mode: None,
        }
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Compute pipeline‑state cache.
//!
//! Metal PSO compilation is expensive (tens to hundreds of
//! microseconds).  This cache maps a FNV‑1a hash of `(kernel_name,
//! first_param_dtype, fn_consts)` → `MTLComputePipelineState` so that
//! identical kernel configurations compile once and are reused across
//! dispatches.
//!
//! # Cache key
//!
//! The key is produced by [`pso_cache_key`].  It uses `dtype.label()`
//! (not `size_bytes()`) because `f16` and `bf16` share a 2‑byte
//! footprint but compile to different MSL bodies (`half` vs `bfloat`).
//! Hashing by `size_bytes` would collide the two dtypes and the cached
//! PSO would produce wrong results for whichever dtype dispatched
//! second.

#[cfg(any(target_os = "macos", test))]
use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

#[cfg(any(target_os = "macos", test))]
use metaltile_core::ir::Kernel;
#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_foundation::NSString;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLComputePipelineDescriptor,
    MTLComputePipelineState,
    MTLDevice,
    MTLLibrary,
    MTLPipelineOption,
};
#[cfg(target_os = "macos")]
use rustc_hash::FxHashMap;

#[cfg(target_os = "macos")]
use crate::error::MetalTileError;

// ---------------------------------------------------------------------------
// Hashing helpers (shared with msl_cache)
// ---------------------------------------------------------------------------

/// FNV‑1a 64‑bit offset basis.
#[cfg(any(target_os = "macos", test))]
pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Extend an FNV‑1a hash with `bytes`.  Threads the accumulator so
/// callers can fold separate fields into one key without
/// intermediate allocations.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn fnv1a_extend(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// Create a stable cache key for a kernel + function‑constants pair.
///
/// The key folds the kernel name, a literal `":"` separator, the
/// string label of the first parameter's dtype, and the sorted
/// `fn_consts`.  All of these are cheap to hash (tens of bytes)
/// compared to the full MSL source (5–50 KB), so per‑pass cache
/// lookups drop from ~10–30 µs to ~16 ns.
///
/// See the module doc for why `dtype.label()` is used instead of
/// `size_bytes`.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn pso_cache_key(kernel: &Kernel, fn_consts: &BTreeMap<String, u32>) -> u64 {
    let mut h = FNV_OFFSET;
    fnv1a_extend(&mut h, kernel.name.as_bytes());
    fnv1a_extend(&mut h, b":");
    // Fold EVERY param's dtype label, not just the first. Quantized
    // kernels (mt_qmm, dequant_gemv_int*, …) take a packed `Tensor<u32>`
    // weight as their first parameter, so `params[0].dtype` is identical
    // (u32) across the f32 / f16 / bf16 monomorphizations — keying on
    // only the first param collided all three onto one PSO, and whichever
    // dtype compiled first served the rest (garbage output for the
    // others). The full param-dtype signature is the real differentiator,
    // since a kernel's monomorphizations all share one `kernel.name`.
    for p in &kernel.params {
        fnv1a_extend(&mut h, p.dtype.label().as_bytes());
        fnv1a_extend(&mut h, b",");
    }
    for (n, v) in fn_consts {
        fnv1a_extend(&mut h, n.as_bytes());
        fnv1a_extend(&mut h, &v.to_le_bytes());
    }
    h
}

// ---------------------------------------------------------------------------
// PSO cache type
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Thread‑safe compute pipeline‑state cache.
///
/// All public methods require `&self` (the inner `Mutex` provides
/// interior mutability), so the cache can be shared freely.
#[cfg(target_os = "macos")]
pub(crate) struct PsoCache {
    cache: Mutex<FxHashMap<u64, Pso>>,
}

#[cfg(target_os = "macos")]
impl PsoCache {
    /// Create an empty cache.
    pub fn new() -> Self { PsoCache { cache: Mutex::new(FxHashMap::default()) } }

    /// Return a cached pipeline state for `key`, or compile it from
    /// `msl_source` on miss.
    ///
    /// `dev` is the Metal device used for compilation.  `kernel_name`
    /// and `fn_consts` are only needed on a cache miss; on hit the
    /// method just clones the `Retained` handle (cheap).
    #[cfg(target_os = "macos")]
    pub(crate) fn get_or_compile(
        &self,
        dev: &ProtocolObject<dyn MTLDevice>,
        key: u64,
        msl_source: &str,
        kernel_name: &str,
        fn_consts: &BTreeMap<String, u32>,
    ) -> Result<Pso, MetalTileError> {
        use std::ptr::NonNull;

        let mut lock =
            self.cache.lock().map_err(|_| MetalTileError::LockPoisoned("PSO cache".into()))?;

        if let Some(cached) = lock.get(&key) {
            return Ok(cached.clone());
        }

        // --- cache miss: compile MSL ---
        let lib = dev
            .newLibraryWithSource_options_error(&NSString::from_str(msl_source), None)
            .map_err(|e| MetalTileError::MslCompilation(format!("{e:?}")))?;

        let fun = if fn_consts.is_empty() {
            lib.newFunctionWithName(&NSString::from_str(kernel_name))
                .ok_or_else(|| MetalTileError::FunctionNotFound { name: kernel_name.to_string() })?
        } else {
            use objc2_metal::{MTLDataType, MTLFunctionConstantValues};
            let fcv = MTLFunctionConstantValues::new();
            for (name, val) in fn_consts {
                // SAFETY: MTLFunctionConstantValues documents that
                // `setConstantValue_type_withName` copies the value
                // during the call and does not retain the pointer.
                // The reference is valid for the duration of this loop
                // iteration.
                let val_ref: &u32 = val;
                unsafe {
                    fcv.setConstantValue_type_withName(
                        NonNull::new(val_ref as *const u32 as *mut _)
                            .expect("reference always non-null"),
                        MTLDataType::UInt,
                        &NSString::from_str(name),
                    );
                }
            }
            lib.newFunctionWithName_constantValues_error(&NSString::from_str(kernel_name), &fcv)
                .map_err(|e| MetalTileError::FunctionNotFound {
                    name: format!("{kernel_name} (with constants): {e:?}"),
                })?
        };

        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(&fun));
        let pso = dev
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption(0),
                None,
            )
            .map_err(|e| MetalTileError::PipelineCreation {
                name: kernel_name.to_string(),
                reason: format!("{e:?}"),
            })?;

        lock.insert(key, pso.clone());
        Ok(pso)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use metaltile_core::{
        dtype::DType,
        ir::{Kernel, Param, ParamKind},
        shape::Shape,
    };

    use super::*;

    fn tensor_param(
        name: &str,
        dtype: DType,
        dims: &[usize],
        is_output: bool,
        kind: ParamKind,
    ) -> Param {
        Param {
            name: name.into(),
            dtype,
            shape: Shape::new(dims.iter().copied().map(metaltile_core::shape::Dim::Known)),
            is_output,
            kind,
        }
    }

    fn key_kernel(name: &str, dtype: DType) -> Kernel {
        let mut k = Kernel::new(name);
        k.params = vec![tensor_param("input", dtype, &[4], false, ParamKind::Tensor)];
        k
    }

    #[test]
    fn fnv1a_extend_empty_input_is_identity() {
        let mut h = FNV_OFFSET;
        fnv1a_extend(&mut h, &[]);
        assert_eq!(h, FNV_OFFSET);

        let mut h2 = 0xdead_beef_dead_beef_u64;
        fnv1a_extend(&mut h2, &[]);
        assert_eq!(h2, 0xdead_beef_dead_beef_u64);
    }

    #[test]
    fn fnv1a_extend_matches_canonical_fnv1a_64() {
        let cases: &[(&[u8], u64)] = &[
            (b"", 0xcbf2_9ce4_8422_2325),
            (b"a", 0xaf63_dc4c_8601_ec8c),
            (b"foobar", 0x8594_4171_f739_67e8),
        ];
        for (input, want) in cases {
            let mut h = FNV_OFFSET;
            fnv1a_extend(&mut h, input);
            assert_eq!(h, *want, "input={input:?}");
        }
    }

    #[test]
    fn fnv1a_extend_is_byte_by_byte_associative() {
        let bytes = b"sdpa_decode_2pass_pass1";
        let mut whole = FNV_OFFSET;
        fnv1a_extend(&mut whole, bytes);

        for split in 0..=bytes.len() {
            let (a, b) = bytes.split_at(split);
            let mut piecewise = FNV_OFFSET;
            fnv1a_extend(&mut piecewise, a);
            fnv1a_extend(&mut piecewise, b);
            assert_eq!(piecewise, whole, "split at {split}");
        }
    }

    #[test]
    fn fnv1a_extend_handles_large_input_without_overflow_panic() {
        let blob: Vec<u8> = (0..10_240).map(|i| (i as u8).wrapping_add(0x42)).collect();
        let mut a = FNV_OFFSET;
        let mut b = FNV_OFFSET;
        fnv1a_extend(&mut a, &blob);
        fnv1a_extend(&mut b, &blob);
        assert_eq!(a, b);
        assert_ne!(a, FNV_OFFSET);
    }

    #[test]
    fn pso_cache_key_is_stable_and_discriminates_inputs() {
        let k = key_kernel("sdpa_decode", DType::F32);
        let mut consts = BTreeMap::new();
        consts.insert("gqa".to_string(), 4u32);
        consts.insert("head_dim".to_string(), 128u32);
        let baseline = pso_cache_key(&k, &consts);
        assert_eq!(baseline, pso_cache_key(&k, &consts), "key must be deterministic");

        let k_other_name = key_kernel("sdpa_prefill", DType::F32);
        assert_ne!(baseline, pso_cache_key(&k_other_name, &consts));

        let k_other_dtype = key_kernel("sdpa_decode", DType::F16);
        assert_ne!(baseline, pso_cache_key(&k_other_dtype, &consts));

        let mut consts_other_val = consts.clone();
        consts_other_val.insert("gqa".to_string(), 8u32);
        assert_ne!(baseline, pso_cache_key(&k, &consts_other_val));

        let mut consts_other_name = BTreeMap::new();
        consts_other_name.insert("kv_heads".to_string(), 4u32);
        consts_other_name.insert("head_dim".to_string(), 128u32);
        assert_ne!(baseline, pso_cache_key(&k, &consts_other_name));

        let empty_consts = BTreeMap::new();
        assert_ne!(baseline, pso_cache_key(&k, &empty_consts));
        let mut k_empty = Kernel::new("noop");
        k_empty.params = vec![];
        let _ = pso_cache_key(&k_empty, &empty_consts);
    }

    #[test]
    fn pso_cache_key_reorder_does_not_change_key() {
        let k = key_kernel("sdpa_decode", DType::F32);
        let mut a = BTreeMap::new();
        a.insert("gqa".to_string(), 4u32);
        a.insert("head_dim".to_string(), 128u32);
        a.insert("n_kv_heads".to_string(), 8u32);

        let mut b = BTreeMap::new();
        b.insert("n_kv_heads".to_string(), 8u32);
        b.insert("head_dim".to_string(), 128u32);
        b.insert("gqa".to_string(), 4u32);

        assert_eq!(pso_cache_key(&k, &a), pso_cache_key(&k, &b));
    }

    #[test]
    fn pso_cache_key_distinguishes_same_size_dtypes() {
        let consts = BTreeMap::new();
        let k_f32 = key_kernel("noop", DType::F32);
        let k_i32 = key_kernel("noop", DType::I32);
        let k_f16 = key_kernel("noop", DType::F16);
        let k_bf16 = key_kernel("noop", DType::BF16);

        assert_ne!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_i32, &consts));
        assert_ne!(pso_cache_key(&k_f16, &consts), pso_cache_key(&k_bf16, &consts));
        assert_ne!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_f16, &consts));
    }

    // Regression test for the quantized-kernel PSO collision: keying on
    // only the first param's dtype made monomorphizations that share a
    // first param (e.g. quantized kernels with a packed `Tensor<u32>`
    // weight) but differ in a *later* value dtype collide onto one PSO.
    // Every param's dtype must now participate in the key.
    #[test]
    fn pso_cache_key_folds_all_param_dtypes() {
        let consts = BTreeMap::new();
        // Two "kernels" with the same name + identical first param (u32
        // packed weight) but f32 vs f16 value dtypes — the exact shape of
        // mt_qmm / dequant_gemv monomorphizations. They must NOT collide.
        let mut k_f32 = Kernel::new("mt_qmm");
        k_f32.params = vec![
            tensor_param("w", DType::U32, &[4], false, ParamKind::Tensor),
            tensor_param("scales", DType::F32, &[4], false, ParamKind::Tensor),
            tensor_param("out", DType::F32, &[4], true, ParamKind::Tensor),
        ];
        let mut k_f16 = Kernel::new("mt_qmm");
        k_f16.params = vec![
            tensor_param("w", DType::U32, &[4], false, ParamKind::Tensor),
            tensor_param("scales", DType::F16, &[4], false, ParamKind::Tensor),
            tensor_param("out", DType::F16, &[4], true, ParamKind::Tensor),
        ];
        assert_ne!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_f16, &consts));

        // And f16 vs bf16 (same byte width, different label) must differ too.
        let mut k_bf16 = Kernel::new("mt_qmm");
        k_bf16.params = vec![
            tensor_param("w", DType::U32, &[4], false, ParamKind::Tensor),
            tensor_param("scales", DType::BF16, &[4], false, ParamKind::Tensor),
            tensor_param("out", DType::BF16, &[4], true, ParamKind::Tensor),
        ];
        assert_ne!(pso_cache_key(&k_f16, &consts), pso_cache_key(&k_bf16, &consts));
    }

    #[test]
    fn pso_cache_key_no_params_still_well_defined() {
        let mut k = Kernel::new("zero_param_kernel");
        k.params = vec![];
        let mut consts = BTreeMap::new();
        consts.insert("foo".to_string(), 1u32);
        let key = pso_cache_key(&k, &consts);
        assert_ne!(key, FNV_OFFSET);
        assert_eq!(key, pso_cache_key(&k, &consts));

        let mut k_other = Kernel::new("other_zero_param_kernel");
        k_other.params = vec![];
        assert_ne!(key, pso_cache_key(&k_other, &consts));
    }

    #[test]
    fn pso_cache_key_separator_prevents_name_dtype_smudge() {
        let mut k_a = Kernel::new("foo");
        k_a.params = vec![tensor_param("x", DType::F32, &[1], false, ParamKind::Tensor)];

        let mut k_b = Kernel::new("foo:");
        k_b.params = vec![];
        let consts = BTreeMap::new();
        assert_ne!(pso_cache_key(&k_a, &consts), pso_cache_key(&k_b, &consts));
    }

    #[test]
    fn pso_cache_key_const_value_endianness_pinned() {
        let k = key_kernel("k", DType::F32);
        let mut a = BTreeMap::new();
        a.insert("c".to_string(), 0x0000_0001_u32);
        let mut b = BTreeMap::new();
        b.insert("c".to_string(), 0x0100_0000_u32);
        assert_ne!(pso_cache_key(&k, &a), pso_cache_key(&k, &b));
    }

    #[test]
    fn pso_cache_key_distinct_for_realistic_kernel_matrix() {
        let kernels = ["sdpa_decode", "sdpa_prefill", "rmsnorm", "matmul_4bit"];
        let dtypes = [DType::F32, DType::F16, DType::BF16];
        let gqas = [1u32, 4, 8];
        let head_dims = [64u32, 128];

        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut total = 0usize;
        for kname in kernels {
            for dt in dtypes {
                for gqa in gqas {
                    for hd in head_dims {
                        let k = key_kernel(kname, dt);
                        let mut consts = BTreeMap::new();
                        consts.insert("gqa".to_string(), gqa);
                        consts.insert("head_dim".to_string(), hd);
                        let key = pso_cache_key(&k, &consts);
                        assert!(
                            seen.insert(key),
                            "duplicate key for ({kname}, {dt:?}, gqa={gqa}, hd={hd})"
                        );
                        total += 1;
                    }
                }
            }
        }
        assert_eq!(total, seen.len());
        assert_eq!(total, kernels.len() * dtypes.len() * gqas.len() * head_dims.len());
    }

    fn perf_bench_msl_blob() -> Vec<u8> {
        (0..10_240).map(|i| (i as u8).wrapping_add(0x42)).collect()
    }

    fn perf_kernel_for_bench() -> Kernel { key_kernel("sdpa_decode_2pass_pass1", DType::F32) }

    fn perf_bench_consts() -> BTreeMap<String, u32> {
        let mut consts = BTreeMap::new();
        consts.insert("gqa".to_string(), 4u32);
        consts
    }

    fn pre_fix_pass_key(msl_bytes: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        fnv1a_extend(&mut h, std::hint::black_box(msl_bytes));
        std::hint::black_box(h)
    }

    fn post_fix_pass_key(kernel: &Kernel, consts: &BTreeMap<String, u32>) -> u64 {
        std::hint::black_box(pso_cache_key(
            std::hint::black_box(kernel),
            std::hint::black_box(consts),
        ))
    }

    #[test]
    fn perf_pass_keys_run_and_discriminate() {
        let msl_bytes = perf_bench_msl_blob();
        let kernel = perf_kernel_for_bench();
        let consts = perf_bench_consts();

        let pre = pre_fix_pass_key(&msl_bytes);
        assert_eq!(pre, pre_fix_pass_key(&msl_bytes), "pre-fix key is deterministic");
        assert_ne!(pre, 0);

        let post = post_fix_pass_key(&kernel, &consts);
        assert_eq!(post, post_fix_pass_key(&kernel, &consts), "post-fix key is deterministic");
        assert_ne!(post, 0);

        assert_ne!(pre, post, "pre/post must hash distinct inputs");
    }

    #[test]
    #[ignore = "perf microbench"]
    fn perf_dispatch_chain_pso_key() {
        let msl_bytes = perf_bench_msl_blob();
        let kernel = perf_kernel_for_bench();
        let consts = perf_bench_consts();

        const ITERS: usize = 1_000_000;
        let mut warm: u64 = 0;
        for _ in 0..10_000 {
            warm ^= pre_fix_pass_key(&msl_bytes);
        }
        std::hint::black_box(warm);

        let start = std::time::Instant::now();
        let mut pre_acc: u64 = 0;
        for _ in 0..ITERS {
            pre_acc ^= pre_fix_pass_key(&msl_bytes);
        }
        let pre = start.elapsed();
        std::hint::black_box(pre_acc);

        let start = std::time::Instant::now();
        let mut post_acc: u64 = 0;
        for _ in 0..ITERS {
            post_acc ^= post_fix_pass_key(&kernel, &consts);
        }
        let post = start.elapsed();
        std::hint::black_box(post_acc);

        let pre_ns = pre.as_nanos() as f64 / ITERS as f64;
        let post_ns = post.as_nanos() as f64 / ITERS as f64;
        let saved_ns = pre_ns - post_ns;
        println!(
            "pre-fix  (FNV over 10 KB MSL):  {pre:?}  ({pre_ns:.0} ns/call)\n\
             post-fix (pso_cache_key call):  {post:?}  ({post_ns:.0} ns/call)\n\
             saved per dispatch_chain pass:  {saved_ns:.0} ns",
        );
    }
}

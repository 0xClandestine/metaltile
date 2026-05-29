//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Runtime context — public entry point for Metal dispatch.
//!
//! [`Context`] owns the Metal device adapter and the GPU family
//! classifier.  All dispatch operations (single‑kernel, chained,
//! resident uploads) delegate to the specialised modules:
//!
//! | Module              | Responsibility                           |
//! |---------------------|------------------------------------------|
//! | [`gpu_family`]      | Hardware probe + name‑based detection    |
//! | [`metal_device`]    | Device, queue, buffer pool, caches       |
//! | [`single_dispatch`] | One kernel → one command buffer          |
//! | [`chain_dispatch`]  | N kernels → one command buffer (fused)   |
//! | [`buffer_pool`]     | Thread‑local MTLBuffer recycling         |
//! | [`pso_cache`]       | PSO compilation with FNV‑1a keys         |
//! | [`msl_cache`]       | MSL source‑generation cache              |

use std::collections::BTreeMap;

use metaltile_core::ir::Kernel;

#[cfg(target_os = "macos")]
use crate::{
    cache::pso_cache::pso_cache_key,
    device::metal_device::MetalDevice,
    dispatch::{chain_dispatch::ChainDispatch, single_dispatch::SingleDispatch},
};
use crate::{device::gpu_family::GpuFamily, error::MetalTileError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Grid sizing specification for Metal dispatch.
#[derive(Debug, Clone)]
pub enum GridSpec {
    /// 1D elementwise: N total threads.
    Elementwise { n: usize },
    /// Reduction: B threadgroups × T threads.
    Reduction { num_rows: usize, threads_per_group: usize },
    /// 3D grid with explicit dimensions.
    Grid3D { x: usize, y: usize, z: usize, threads_per_group: usize },
}

/// Output from a kernel dispatch.
#[derive(Debug)]
pub struct DispatchResult {
    /// Wall‑clock time on the GPU, in microseconds.
    pub elapsed_us: f64,
    /// Theoretical GFLOPS (always 0.0 — not yet wired).
    pub gflops: f64,
    /// Output buffer contents keyed by parameter name.
    pub outputs: BTreeMap<String, Vec<u8>>,
}

impl DispatchResult {
    /// Borrow the raw bytes of an output buffer by parameter name.
    ///
    /// Returns `None` if no output with that name was produced. Prefer this
    /// over indexing `outputs` directly so callers don't panic on a typo'd
    /// or absent name.
    pub fn output(&self, name: &str) -> Option<&[u8]> { self.outputs.get(name).map(Vec::as_slice) }

    /// Read an output buffer as little-endian `f32` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the named output is absent. Interprets the bytes as
    /// 4-byte `f32`; for half-precision outputs use [`output`](Self::output)
    /// plus a dtype-aware unpack helper instead.
    pub fn output_f32(&self, name: &str) -> Result<Vec<f32>, MetalTileError> {
        let bytes = self
            .output(name)
            .ok_or_else(|| MetalTileError::Dispatch(format!("output '{name}' not found")))?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Read an output buffer as little-endian `u32` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the named output is absent.
    pub fn output_u32(&self, name: &str) -> Result<Vec<u32>, MetalTileError> {
        let bytes = self
            .output(name)
            .ok_or_else(|| MetalTileError::Dispatch(format!("output '{name}' not found")))?;
        Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// One pass in a fused dispatch chain.  See [`Context::dispatch_chain`].
///
/// Buffers reused across consecutive passes (output of pass `i` →
/// input of pass `j>i`) are auto‑aliased to a single Metal allocation
/// in `MTLStorageModePrivate` — they never leave GPU memory.
///
/// Pass [`Context::upload_resident`]-produced handles in `resident`
/// to bind a pre‑uploaded Metal buffer for a parameter by name; the
/// dispatch then skips the per‑call alloc + `memcpy` for that input.
/// Holding the [`ResidentBuffer`] across iterations keeps the bytes
/// GPU‑resident.
pub struct DispatchSpec<'a> {
    /// The kernel to dispatch.
    pub kernel: &'a Kernel,
    /// Host‑side input buffers (keyed by parameter name).
    pub buffers: &'a BTreeMap<String, Vec<u8>>,
    /// Function constants for `[[function_constant(N)]]` annotations.
    pub fn_consts: &'a BTreeMap<String, u32>,
    /// Number of threadgroups along each axis `[x, y, z]`.
    pub grid_groups: [usize; 3],
    /// Threads per threadgroup `[x, y, z]`.
    pub threads_per_group: [usize; 3],
    /// Pre‑uploaded resident buffers keyed by parameter name.
    pub resident: &'a BTreeMap<String, ResidentBuffer>,
}

/// Opaque handle to a GPU‑resident input buffer.
///
/// Produced by [`Context::upload_resident`]; pass via
/// [`DispatchSpec::resident`] to bind without per‑call allocation +
/// host `memcpy`.
///
/// Cloning is cheap (`Rc::clone`) and shares the underlying Metal
/// buffer.  The buffer returns to the dispatch buffer pool when the
/// last clone drops.
#[derive(Clone)]
pub struct ResidentBuffer {
    #[cfg(target_os = "macos")]
    pub(crate) inner: std::rc::Rc<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    >,
    #[cfg(not(target_os = "macos"))]
    _stub: (),
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Metal runtime context — the public entry point for GPU dispatch.
///
/// # Lifecycle
///
/// Create with [`Context::new`].  The constructor probes for a Metal
/// device and detects the GPU family.  On non‑macOS platforms, or
/// when no GPU is found, [`Context::has_gpu`] returns `false` and
/// dispatches are no‑ops (returning empty results).
///
/// # Example
///
/// ```ignore
/// let ctx = Context::new()?;
/// if ctx.has_gpu() {
///     let result = ctx.dispatch(&my_kernel)?;
///     println!("{:.2} µs", result.elapsed_us);
/// }
/// ```
pub struct Context {
    /// The Metal device adapter (None when no Metal available).
    #[cfg(target_os = "macos")]
    device: Option<MetalDevice>,
    /// GPU family classifier.
    gpu_family: GpuFamily,
}

impl Context {
    /// Initialise the runtime context.
    ///
    /// Probes the default Metal device on macOS.  On other platforms
    /// `has_gpu()` will return `false` but the context is still valid
    /// (dispatches are no‑ops).
    pub fn new() -> Result<Self, MetalTileError> {
        #[cfg(target_os = "macos")]
        let device = MetalDevice::create()?;
        let gpu_family = GpuFamily::detect();

        tracing::info!(
            has_metal = cfg!(target_os = "macos"),
            gpu_family = %gpu_family,
            "runtime context initialized"
        );

        Ok(Context {
            #[cfg(target_os = "macos")]
            device,
            gpu_family,
        })
    }

    // ── capability queries ──────────────────────────────────────────

    /// True when a Metal GPU is available.
    pub fn has_gpu(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.device.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// The GPU family classifier.
    ///
    /// On macOS this comes from the hardware probe; otherwise it
    /// returns [`GpuFamily::Unknown`].  Use
    /// [`GpuFamily::from_device_name`] for a cross‑platform heuristic
    /// when a device name string is available.
    pub fn gpu_family(&self) -> GpuFamily { self.gpu_family }

    /// Apple GPU family level detected at construction time, or
    /// `None` off macOS / when no Metal device is available.
    ///
    /// Shortcut for `self.gpu_family().family_level()`.  Kept for
    /// backward compatibility with callers that expect `Option<u32>`.
    pub fn chip_family(&self) -> Option<u32> { self.gpu_family.family_level() }

    // ── single‑kernel dispatch ──────────────────────────────────────

    /// Dispatch a kernel with no input buffers and no function
    /// constants.
    pub fn dispatch(&self, kernel: &Kernel) -> Result<DispatchResult, MetalTileError> {
        self.dispatch_with_buffers(kernel, &BTreeMap::new())
    }

    /// Dispatch a kernel with caller‑supplied input buffers.
    pub fn dispatch_with_buffers(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
    ) -> Result<DispatchResult, MetalTileError> {
        self.dispatch_with_options(kernel, buffers, &BTreeMap::new())
    }

    /// Dispatch a kernel with buffers and function constants.
    ///
    /// `fn_consts` maps constant name → `u32` value for kernels that
    /// use `[[function_constant(N)]]` annotations (e.g. ROPE).
    #[tracing::instrument(
        skip(self, kernel, buffers, fn_consts),
        fields(kernel = %kernel.name)
    )]
    pub fn dispatch_with_options(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        fn_consts: &BTreeMap<String, u32>,
    ) -> Result<DispatchResult, MetalTileError> {
        self.dispatch_with_grid_opt(kernel, buffers, fn_consts, None)
    }

    /// Dispatch a kernel with an explicit grid override.
    ///
    /// Use when the auto‑derived grid (from output buffer size +
    /// `kernel.mode`) doesn't fit — e.g. a reduction kernel that
    /// needs one threadgroup per Q head with a fixed thread count
    /// rather than per output‑element.
    ///
    /// `grid_groups` is the number of threadgroups per axis;
    /// `threads_per_group` is the size of each threadgroup.  Both are
    /// `[x, y, z]`.
    #[tracing::instrument(
        skip(self, kernel, buffers, fn_consts),
        fields(kernel = %kernel.name)
    )]
    pub fn dispatch_with_grid(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        fn_consts: &BTreeMap<String, u32>,
        grid_groups: [usize; 3],
        threads_per_group: [usize; 3],
    ) -> Result<DispatchResult, MetalTileError> {
        self.dispatch_with_grid_opt(
            kernel,
            buffers,
            fn_consts,
            Some((grid_groups, threads_per_group)),
        )
    }

    /// Internal dispatch method shared by `dispatch_with_options` and
    /// `dispatch_with_grid`.
    fn dispatch_with_grid_opt(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        fn_consts: &BTreeMap<String, u32>,
        grid_override: Option<([usize; 3], [usize; 3])>,
    ) -> Result<DispatchResult, MetalTileError> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (kernel, buffers, fn_consts, grid_override);
            Ok(DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
        }

        #[cfg(target_os = "macos")]
        {
            let Some(dev) = self.device.as_ref() else {
                return Ok(DispatchResult {
                    elapsed_us: 0.0,
                    gflops: 0.0,
                    outputs: BTreeMap::new(),
                });
            };

            let key = pso_cache_key(kernel, fn_consts);
            let msl = dev.get_msl(kernel, key)?;
            let pso = dev.get_pso(key, &msl, &kernel.name, fn_consts)?;

            let dispatch = SingleDispatch::new(dev, kernel, &pso, buffers, grid_override);
            dispatch.execute()
        }
    }

    // ── resident buffer upload ──────────────────────────────────────

    /// Upload host bytes into a pool‑managed Metal buffer and return
    /// an opaque handle.
    ///
    /// Pass the handle via [`DispatchSpec::resident`] to bind without
    /// per‑call allocation + `memcpy`.  The buffer stays GPU‑resident
    /// as long as any clone of the [`ResidentBuffer`] exists; on the
    /// last drop it returns to the pool.
    #[tracing::instrument(skip(self, bytes), fields(bytes = bytes.len()))]
    pub fn upload_resident(&self, bytes: &[u8]) -> Result<ResidentBuffer, MetalTileError> {
        #[cfg(target_os = "macos")]
        {
            let dev = self.device.as_ref().ok_or(MetalTileError::NoDevice)?;

            let buf = dev.acquire_shared(Some(bytes), bytes.len())?;
            Ok(ResidentBuffer { inner: buf })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bytes;
            Ok(ResidentBuffer { _stub: () })
        }
    }

    // ── chained dispatch ────────────────────────────────────────────

    /// Dispatch N kernel passes through a single Metal command buffer.
    ///
    /// Buffers referenced as outputs of pass `i` and as inputs of any
    /// later pass are allocated **once** as `MTLStorageModePrivate`
    /// and shared across passes — they never round‑trip through host
    /// RAM.  Pass‑to‑pass ordering is enforced with a
    /// `memoryBarrierWithScope` between consecutive encoders.  Only
    /// buffers consumed by no later pass are read back at the end.
    ///
    /// For a 2‑pass SDPA decode this replaces two separate cmd‑buffer
    /// commits + a ~MB‑sized host `memcpy` of `partial_o/m/l` with
    /// one commit and zero host traffic between passes.
    #[tracing::instrument(
        skip(self, specs),
        fields(spec_count = specs.len())
    )]
    pub fn dispatch_chain(
        &self,
        specs: &[DispatchSpec<'_>],
    ) -> Result<Vec<DispatchResult>, MetalTileError> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = specs;
            Ok(specs
                .iter()
                .map(|_| DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
                .collect())
        }

        #[cfg(target_os = "macos")]
        {
            let Some(dev) = self.device.as_ref() else {
                return Ok(specs
                    .iter()
                    .map(|_| DispatchResult {
                        elapsed_us: 0.0,
                        gflops: 0.0,
                        outputs: BTreeMap::new(),
                    })
                    .collect());
            };

            ChainDispatch::new(dev, specs).execute()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use metaltile_core::{DType, Dim, Param, Shape, ir::ParamKind};

    use super::*;
    use crate::{
        cache::pso_cache::{FNV_OFFSET, fnv1a_extend, pso_cache_key},
        dispatch::buffer_plan::{
            ParamBufferPlan,
            build_param_buffer_plans,
            encode_u32s,
            resolve_strided_metadata,
        },
    };

    fn sample_result() -> DispatchResult {
        let mut outputs = BTreeMap::new();
        // 1.0f32 then 2.0f32, little-endian.
        outputs.insert("out".to_string(), vec![0, 0, 128, 63, 0, 0, 0, 64]);
        DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs }
    }

    #[test]
    fn dispatch_result_output_borrows_raw_bytes() {
        let r = sample_result();
        assert_eq!(r.output("out").unwrap().len(), 8);
        assert!(r.output("missing").is_none());
    }

    #[test]
    fn dispatch_result_output_f32_decodes_le() {
        let r = sample_result();
        assert_eq!(r.output_f32("out").unwrap(), vec![1.0f32, 2.0]);
    }

    #[test]
    fn dispatch_result_output_u32_decodes_le() {
        let mut outputs = BTreeMap::new();
        outputs.insert("c".to_string(), vec![1, 0, 0, 0, 255, 255, 255, 255]);
        let r = DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs };
        assert_eq!(r.output_u32("c").unwrap(), vec![1u32, u32::MAX]);
    }

    #[test]
    fn dispatch_result_accessors_error_on_missing_name() {
        let r = sample_result();
        assert!(r.output_f32("nope").is_err());
        assert!(r.output_u32("nope").is_err());
    }

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
            shape: Shape::new(dims.iter().copied().map(Dim::Known)),
            is_output,
            kind,
        }
    }

    #[test]
    fn buffer_plans_follow_binding_indices_and_static_output_sizes() {
        let mut kernel = Kernel::new("binding_plan_test");
        kernel.params = vec![
            tensor_param("input", DType::F32, &[2, 2], false, ParamKind::Strided),
            tensor_param("out_a", DType::F32, &[4], true, ParamKind::Tensor),
            tensor_param("out_b", DType::F32, &[4], true, ParamKind::Tensor),
        ];

        let mut buffers = BTreeMap::new();
        buffers.insert("input".into(), vec![0u8; 16]);
        buffers.insert("out_b".into(), vec![0u8; 16]);

        let plans = build_param_buffer_plans(&kernel, &buffers).expect("buffer plans");
        assert_eq!(plans, vec![
            ParamBufferPlan { data_binding_index: 0, data_len: 16 },
            ParamBufferPlan { data_binding_index: 3, data_len: 16 },
            ParamBufferPlan { data_binding_index: 4, data_len: 16 },
        ]);
    }

    #[test]
    fn strided_metadata_defaults_to_row_major_shape_and_strides() {
        let param = tensor_param("input", DType::F32, &[2, 3, 4], false, ParamKind::Strided);
        let buffers = BTreeMap::new();

        let (shape_data, stride_data) =
            resolve_strided_metadata(&param, &buffers).expect("default strided metadata");

        assert_eq!(shape_data.as_ref(), encode_u32s(&[2, 3, 4]).as_slice());
        assert_eq!(stride_data.as_ref(), encode_u32s(&[12, 4, 1]).as_slice());
    }

    // Perf microbench — run with:
    //   cargo test --release -p metaltile-runtime perf_resolve_strided_metadata \
    //     -- --ignored --nocapture
    //
    // Times `resolve_strided_metadata` in a hot loop over a realistic param
    // name. Pre-change (`format!("{}_shape", …)` + `format!("{}_strides", …)`)
    // allocates two Strings per call; post-change reuses a single pre-sized
    // String via in-place suffix rewrite.
    fn key_kernel(name: &str, dtype: DType) -> Kernel {
        let mut k = Kernel::new(name);
        k.params = vec![tensor_param("input", dtype, &[4], false, ParamKind::Tensor)];
        k
    }

    #[test]
    fn pso_cache_key_is_stable_and_discriminates_inputs() {
        let k = key_kernel("sdpa_decode", DType::F32);
        let mut consts = BTreeMap::new();
        consts.insert("gqa".to_string(), 4u32);
        consts.insert("head_dim".to_string(), 128u32);
        let baseline = pso_cache_key(&k, &consts);
        assert_eq!(baseline, pso_cache_key(&k, &consts), "key must be deterministic");

        // Kernel name change → different key.
        let k_other_name = key_kernel("sdpa_prefill", DType::F32);
        assert_ne!(baseline, pso_cache_key(&k_other_name, &consts));

        // First-param dtype change → different key (f32 vs f16 specializations).
        let k_other_dtype = key_kernel("sdpa_decode", DType::F16);
        assert_ne!(baseline, pso_cache_key(&k_other_dtype, &consts));

        // fn_const value change → different key.
        let mut consts_other_val = consts.clone();
        consts_other_val.insert("gqa".to_string(), 8u32);
        assert_ne!(baseline, pso_cache_key(&k, &consts_other_val));

        // fn_const name change → different key.
        let mut consts_other_name = BTreeMap::new();
        consts_other_name.insert("kv_heads".to_string(), 4u32);
        consts_other_name.insert("head_dim".to_string(), 128u32);
        assert_ne!(baseline, pso_cache_key(&k, &consts_other_name));

        // Empty fn_consts and empty params are both well-defined.
        let empty_consts = BTreeMap::new();
        assert_ne!(baseline, pso_cache_key(&k, &empty_consts));
        let mut k_empty = Kernel::new("noop");
        k_empty.params = vec![];
        let _ = pso_cache_key(&k_empty, &empty_consts);
    }

    #[test]
    fn fnv1a_extend_empty_input_is_identity() {
        // Hashing no bytes must not perturb the accumulator. Guarantees the
        // empty-fn_consts path in `pso_cache_key` doesn't depend on a separator.
        let mut h = FNV_OFFSET;
        fnv1a_extend(&mut h, &[]);
        assert_eq!(h, FNV_OFFSET);

        let mut h2 = 0xdead_beef_dead_beef_u64;
        fnv1a_extend(&mut h2, &[]);
        assert_eq!(h2, 0xdead_beef_dead_beef_u64);
    }

    #[test]
    fn fnv1a_extend_matches_canonical_fnv1a_64() {
        // Spot-check against canonical FNV-1a 64-bit vectors. "" → FNV_OFFSET,
        // "a" → 0xaf63dc4c8601ec8c, "foobar" → 0x85944171f73967e8 (well-known
        // FNV reference values). Pins the constants in case anyone "optimises"
        // the prime/offset.
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
        // Folding bytes in one shot must equal folding the same bytes split
        // across two calls. This is the property `pso_cache_key` relies on
        // when it threads the accumulator through name → ":" → dtype → consts.
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
        // 10 KB blob matches the perf-microbench size and exercises the inner
        // wrapping_mul on a long stream. Result must be deterministic.
        let blob: Vec<u8> = (0..10_240).map(|i| (i as u8).wrapping_add(0x42)).collect();
        let mut a = FNV_OFFSET;
        let mut b = FNV_OFFSET;
        fnv1a_extend(&mut a, &blob);
        fnv1a_extend(&mut b, &blob);
        assert_eq!(a, b);
        assert_ne!(a, FNV_OFFSET);
    }

    #[test]
    fn pso_cache_key_reorder_does_not_change_key_because_btreemap_is_sorted() {
        // `pso_cache_key` iterates `fn_consts: &BTreeMap`, which yields keys
        // in sorted order regardless of insertion order. Two maps built in
        // different orders must hash identically — otherwise a caller that
        // inserted in a different order would miss the PSO cache.
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
        // Regression: the cache key used to fold in `dtype.size_bytes()`,
        // which made f16 and bf16 (both 2 bytes) — and f32 and i32 (both 4)
        // — hash to the same slot. That collision swapped the bf16 PSO into
        // the f16 dispatch on the second test to run in
        // `sdpa_decode_2pass_gpu::matches_cpu_reference_*_chained_resident_gqa`,
        // producing max |diff| in the hundreds of millis at f16 and a
        // codecov-only CI flake. The fix folds in `dtype.label()` instead
        // — each dtype must hash to a distinct slot.
        let consts = BTreeMap::new();
        let k_f32 = key_kernel("noop", DType::F32);
        let k_i32 = key_kernel("noop", DType::I32);
        let k_f16 = key_kernel("noop", DType::F16);
        let k_bf16 = key_kernel("noop", DType::BF16);

        assert_ne!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_i32, &consts));
        assert_ne!(pso_cache_key(&k_f16, &consts), pso_cache_key(&k_bf16, &consts));
        assert_ne!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_f16, &consts));
    }

    #[test]
    fn pso_cache_key_folds_all_param_dtypes() {
        // Regression: `pso_cache_key` used to fold only `params.first()`.
        // Quantized kernels (mt_qmm, dequant_gemv_int*) take a packed
        // `Tensor<u32>` weight as their first param, so the f32 / f16 /
        // bf16 monomorphizations shared one key and collided onto a single
        // PSO — whichever dtype compiled first served the others, reading
        // their narrower buffers through the wrong pipeline (garbage). The
        // full param-dtype signature now participates: kernels that differ
        // only in a *later* value dtype must hash to distinct keys.
        let consts = BTreeMap::new();
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
    }

    #[test]
    fn pso_cache_key_no_params_still_well_defined() {
        // A kernel with zero params skips the dtype-size fold but must still
        // produce a stable, non-trivial key (the kernel name + ":" + sorted
        // fn_consts get folded in). Pins that the `if let Some(...)` branch
        // is the *only* effect of param absence — separator and consts still
        // fold.
        let mut k = Kernel::new("zero_param_kernel");
        k.params = vec![];
        let mut consts = BTreeMap::new();
        consts.insert("foo".to_string(), 1u32);
        let key = pso_cache_key(&k, &consts);
        assert_ne!(key, FNV_OFFSET);
        assert_eq!(key, pso_cache_key(&k, &consts));

        // A different kernel name still discriminates with no params present.
        let mut k_other = Kernel::new("other_zero_param_kernel");
        k_other.params = vec![];
        assert_ne!(key, pso_cache_key(&k_other, &consts));
    }

    /// Representative MSL source size for a real metaltile kernel:
    /// sdpa_decode_2pass_pass1 generates ~12 KB; sdpa_vector ~6 KB. A
    /// mid-size 10 KB blob is the pre-fix per-pass cost we're comparing
    /// against. Shared by the always-on coverage test and the perf bench.
    fn perf_bench_msl_blob() -> Vec<u8> {
        (0..10_240).map(|i| (i as u8).wrapping_add(0x42)).collect()
    }

    fn perf_kernel_for_bench() -> Kernel { key_kernel("sdpa_decode_2pass_pass1", DType::F32) }

    fn perf_bench_consts() -> BTreeMap<String, u32> {
        let mut consts = BTreeMap::new();
        consts.insert("gqa".to_string(), 4u32);
        consts
    }

    /// Pre-fix per-pass cost: FNV-1a over the full MSL source.
    fn pre_fix_pass_key(msl_bytes: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        fnv1a_extend(&mut h, std::hint::black_box(msl_bytes));
        std::hint::black_box(h)
    }

    /// Post-fix per-pass cost: structured `pso_cache_key` over the small
    /// (name, first-param dtype, sorted fn_consts) tuple.
    fn post_fix_pass_key(kernel: &Kernel, consts: &BTreeMap<String, u32>) -> u64 {
        std::hint::black_box(pso_cache_key(
            std::hint::black_box(kernel),
            std::hint::black_box(consts),
        ))
    }

    #[test]
    fn pso_cache_key_separator_prevents_name_dtype_smudge() {
        // `pso_cache_key` folds `kernel.name` then `b":"` then the dtype-size
        // bytes. Without the separator, a kernel named `"foo"` with dtype
        // size N could collide with one named `"foo<sep_bytes>"` with no
        // params. The literal `":"` byte is what stops that.
        let mut k_a = Kernel::new("foo");
        k_a.params = vec![tensor_param("x", DType::F32, &[1], false, ParamKind::Tensor)];

        // A kernel named "foo:" with no params: if the separator weren't
        // there, the post-name accumulator state would equal `"foo" + ":"`
        // from `k_a`, and a no-params kernel skips the dtype fold — so
        // without the separator these *could* collide. With the separator,
        // they must differ because `k_a` folds in 8 dtype bytes after ":"
        // and `k_b` folds in nothing.
        let mut k_b = Kernel::new("foo:");
        k_b.params = vec![];
        let consts = BTreeMap::new();
        assert_ne!(pso_cache_key(&k_a, &consts), pso_cache_key(&k_b, &consts));
    }

    #[test]
    fn pso_cache_key_const_value_endianness_pinned() {
        // fn_const values are folded as little-endian u32 bytes. Pin that
        // by checking that two values which are byte-swapped versions of
        // each other produce different keys (catches an accidental switch
        // to `to_be_bytes`).
        let k = key_kernel("k", DType::F32);
        let mut a = BTreeMap::new();
        a.insert("c".to_string(), 0x0000_0001_u32);
        let mut b = BTreeMap::new();
        b.insert("c".to_string(), 0x0100_0000_u32);
        assert_ne!(pso_cache_key(&k, &a), pso_cache_key(&k, &b));
    }

    #[test]
    fn pso_cache_key_distinct_for_realistic_kernel_matrix() {
        // Sanity-sweep across a realistic matrix of (kernel, dtype, gqa,
        // head_dim) tuples and assert all keys are pairwise distinct. Any
        // accidental aliasing (e.g. dropping the const name in favour of
        // value-only hashing) would surface here as a duplicate.
        let kernels = ["sdpa_decode", "sdpa_prefill", "rmsnorm", "matmul_4bit"];
        // All dtypes the runtime monomorphises over — f16 and bf16 must
        // be distinct slots (the cache key folds in `dtype.label()`, not
        // `size_bytes()`; see `pso_cache_key_distinguishes_same_size_dtypes`).
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

    #[test]
    fn perf_pass_keys_run_and_discriminate() {
        // Always-on coverage for the per-pass key helpers used by the
        // `#[ignore]`'d perf bench below. Both paths must produce stable,
        // non-trivial keys, and the two paths must NOT collide — the whole
        // point of the PR is that they're different discriminators for the
        // same kernel.
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

    // Perf microbench — run with:
    //   cargo test --release -p metaltile-runtime perf_dispatch_chain_pso_key \
    //     -- --ignored --nocapture
    //
    // Times the per-pass PSO cache key computation in `dispatch_chain_metal`.
    // Pre-fix: FNV-1a over the full MSL source string (5–50 KB per pass).
    // Post-fix: `pso_cache_key` over the kernel name + first-param dtype size
    // + sorted fn_consts (~30–60 bytes). Demonstrates the savings without a
    // GPU; the fix itself is the helper call inside `dispatch_chain_metal`.
    #[test]
    #[ignore = "perf microbench"]
    fn perf_dispatch_chain_pso_key() {
        let msl_bytes = perf_bench_msl_blob();
        let kernel = perf_kernel_for_bench();
        let consts = perf_bench_consts();

        const ITERS: usize = 1_000_000;
        // Warmup.
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

    #[test]
    #[ignore = "perf microbench"]
    fn perf_resolve_strided_metadata() {
        let param = tensor_param(
            "long_strided_kv_cache",
            DType::F32,
            &[8, 4096, 128],
            false,
            ParamKind::Strided,
        );
        let buffers = BTreeMap::new();
        const ITERS: usize = 5_000_000;
        for _ in 0..50_000 {
            std::hint::black_box(
                resolve_strided_metadata(&param, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(
                resolve_strided_metadata(&param, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        println!("resolve_strided_metadata × {ITERS}: {elapsed:?} ({ns_per_call:.1} ns/call)");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn context_on_macos_has_gpu_and_family() {
        let ctx = Context::new().expect("Context::new should succeed on macOS");
        assert!(ctx.has_gpu());

        // GitHub Actions' hosted macOS runners report no Apple GPU
        // family (virtualised / non‑Apple GPU), so `chip_family()`
        // is `None` there.  Treat that as a pass — real hardware
        // still gets the strict check.
        let Some(level) = ctx.chip_family() else {
            assert!(
                std::env::var_os("CI").is_some(),
                "macOS Context must report a family on real hardware",
            );
            return;
        };
        assert!(level >= 7, "Apple GPU family level should be ≥7 on M‑series, got {level}");
        assert!(level <= 20, "level looks unreasonably high ({level})");
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn context_off_macos_has_no_gpu() {
        let ctx = Context::new().expect("Context::new should succeed off macOS too");
        assert!(!ctx.has_gpu());
        assert!(ctx.chip_family().is_none());
        assert_eq!(ctx.gpu_family(), GpuFamily::Unknown);
    }

    #[test]
    fn empty_chain_returns_empty_vec() {
        let ctx = Context::new().expect("Context::new should succeed everywhere");
        let results = ctx.dispatch_chain(&[]).expect("empty chain should succeed");
        assert!(results.is_empty());
    }
}

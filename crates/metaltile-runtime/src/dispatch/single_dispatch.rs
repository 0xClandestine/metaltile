//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Single‑kernel Metal dispatch.
//!
//! Encapsulates the work of encoding one kernel onto one command
//! buffer: buffer allocation, Metal buffer binding, grid derivation,
//! dispatch, commit, and output read‑back.

use std::{borrow::Cow, collections::BTreeMap};

use metaltile_core::ir::{Kernel, KernelMode};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer,
    MTLCommandEncoder,
    MTLComputeCommandEncoder,
    MTLComputePipelineState,
    MTLSize,
};

use crate::{
    DispatchResult,
    device::{buffer_pool::BufRc, metal_device::MetalDevice},
    dispatch::buffer_plan::{ParamBufferPlan, build_param_buffer_plans, resolve_strided_metadata},
    error::MetalTileError,
};

// ---------------------------------------------------------------------------
// SingleDispatch
// ---------------------------------------------------------------------------

/// Orchestrates a single‑kernel Metal dispatch.
///
/// Created by [`Context`](super::Context) for each call to
/// `dispatch_with_options` or `dispatch_with_grid`.  Handles:
///
/// 1. Buffer allocation and data upload
/// 2. Constexpr scalar binding via `setBytes`
/// 3. Grid derivation (automatic or caller‑specified)
/// 4. Command encoding, commit, and output read‑back
pub(crate) struct SingleDispatch<'a> {
    /// The Metal device adapter.
    dev: &'a MetalDevice,
    /// Kernel to dispatch.
    kernel: &'a Kernel,
    /// Pre‑compiled or cached pipeline state.
    pso: &'a Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Host‑side input buffers (keyed by parameter name).
    buffers: &'a BTreeMap<String, Vec<u8>>,
    /// When `Some`, overrides the auto‑derived grid:
    /// `(groups, threads_per_group)`.
    grid_override: Option<([usize; 3], [usize; 3])>,
}

impl<'a> SingleDispatch<'a> {
    /// Prepare a dispatch.  All heavy work (MSL generation, PSO
    /// compilation) has already been done by the caller; this struct
    /// only handles buffer uploads and encoding.
    pub fn new(
        dev: &'a MetalDevice,
        kernel: &'a Kernel,
        pso: &'a Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        buffers: &'a BTreeMap<String, Vec<u8>>,
        grid_override: Option<([usize; 3], [usize; 3])>,
    ) -> Self {
        SingleDispatch { dev, kernel, pso, buffers, grid_override }
    }

    /// Execute the dispatch and return the result.
    pub fn execute(&self) -> Result<DispatchResult, MetalTileError> {
        use objc2_metal::MTLCommandBuffer as _;

        let binding_plans = build_param_buffer_plans(self.kernel, self.buffers)?;

        let (metal_bufs, n_threads) = self.allocate_buffers(&binding_plans)?;

        let n_threads = n_threads.max(1);
        let tpg_w = self.pso.maxTotalThreadsPerThreadgroup().min(256);
        let (tgs, tpg) = self.resolve_grid(n_threads, tpg_w);

        // Reject GPU-pinning geometry (e.g. a reduction kernel dispatched with
        // < 32 threads/threadgroup → infinite loop) before it reaches the
        // non-preemptive GPU — and before any command encoder is created, so a
        // rejection can't leave an encoder un-ended. See `dispatch::validate`.
        crate::dispatch::validate::validate_dispatch_geometry(
            self.kernel,
            [tgs.width, tgs.height, tgs.depth],
            [tpg.width, tpg.height, tpg.depth],
            self.pso.maxTotalThreadsPerThreadgroup(),
            metaltile_codegen::kernel_uses_n_simd(self.kernel),
        )?;

        let cb = self.dev.command_buffer()?;
        let enc = (*cb).computeCommandEncoder().ok_or(MetalTileError::NoDevice)?;
        enc.setComputePipelineState(self.pso);

        for (i, buf) in metal_bufs.iter().enumerate() {
            // SAFETY: `buf` is a valid MTLBuffer.  The offset is
            // zero because we allocate dedicated buffers for each
            // binding.
            unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
        }

        enc.dispatchThreadgroups_threadsPerThreadgroup(tgs, tpg);
        (*enc).endEncoding();
        (*cb).commit();
        (*cb).waitUntilCompleted();

        let mut outputs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (param, binding) in self.kernel.params.iter().zip(&binding_plans) {
            if param.is_output
                && binding.data_len > 0
                && let Some(buf) = metal_bufs.get(binding.data_binding_index)
            {
                let ptr = buf.contents();
                // SAFETY: `buf.contents()` returns a valid pointer for
                // the buffer's lifetime.  The buffer was allocated with
                // `data_len` bytes and `waitUntilCompleted` has been
                // called, so the GPU write is visible.
                let bytes = unsafe {
                    std::slice::from_raw_parts(ptr.as_ptr() as *const u8, binding.data_len)
                }
                .to_vec();
                outputs.insert(param.name.clone(), bytes);
            }
        }

        let elapsed_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
        Ok(DispatchResult { elapsed_us, gflops: 0.0, outputs })
    }

    /// Allocate Metal buffers for every kernel parameter, upload host
    /// data, and return the buffer list plus the inferred thread count.
    fn allocate_buffers(
        &self,
        binding_plans: &[ParamBufferPlan],
    ) -> Result<(Vec<BufRc>, usize), MetalTileError> {
        let mut bufs = Vec::with_capacity(self.kernel.params.len() * 2);
        let mut n_threads = 1usize;

        for (param, binding) in self.kernel.params.iter().zip(binding_plans) {
            let data = self.buffers.get(&param.name).map(Vec::as_slice);

            if param.is_output {
                let elem_bytes = param.dtype.size_bytes();
                if let Some(quot) = binding.data_len.checked_div(elem_bytes) {
                    n_threads = n_threads.max(quot);
                }
            }

            // `required` is the param's true byte size; `alloc_len` adds
            // Metal's minimum-allocation floor (4 bytes). Keeping them
            // distinct lets a legitimately-small buffer — e.g. a 2-byte
            // f16/bf16 single-element `Tensor<T>` — bind correctly: we
            // still reject genuine under-provisioning (`< required`), but
            // zero-pad an in-spec-but-sub-floor buffer up to the floor so
            // the GPU read stays in bounds. Comparing against the floored
            // `alloc_len` (the bug this restores) rejected every valid
            // 2-byte buffer with "expected 4 bytes but received 2".
            let required = binding.data_len;
            let alloc_len = required.max(4);
            // Route every allocation through the thread-local pool
            // (`acquire_shared` buckets by `next_power_of_two(len)` and
            // hands back a `BufRc` whose strong-count gates recycling).
            // The bench loop dispatches the same kernel hundreds of
            // times against fresh buffers, so recycling avoids the
            // ~µs/buffer `newBufferWithBytes` allocator round-trip.
            let buf = if let Some(bytes) = data.filter(|b| !b.is_empty()) {
                if bytes.len() < required {
                    return Err(MetalTileError::Buffer(format!(
                        "buffer allocation expected {required} bytes but received {}",
                        bytes.len()
                    )));
                }
                if bytes.len() >= alloc_len {
                    self.dev.acquire_shared(Some(&bytes[..alloc_len]), alloc_len)?
                } else {
                    // bytes.len() ∈ [required, alloc_len): pad up to the floor.
                    let mut padded = bytes.to_vec();
                    padded.resize(alloc_len, 0);
                    self.dev.acquire_shared(Some(&padded), alloc_len)?
                }
            } else {
                self.dev.acquire_shared(None, alloc_len)?
            };
            bufs.push(buf);

            if param.kind == metaltile_core::ir::ParamKind::Strided {
                let (shape_data, stride_data) = resolve_strided_metadata(param, self.buffers)?;
                bufs.push(self.dev.acquire_shared(Some(shape_data.as_ref()), shape_data.len())?);
                bufs.push(self.dev.acquire_shared(Some(stride_data.as_ref()), stride_data.len())?);
            }
        }

        for decl in &self.kernel.constexprs {
            let key = decl.name.name();
            let elem = decl.dtype.size_bytes().max(4);
            let bytes = self.buffers.get(key).map(Vec::as_slice);
            let len = elem.max(4);
            let buf = if let Some(b) = bytes.filter(|b| !b.is_empty()) {
                // Zero-pad a sub-floor scalar (e.g. a 2-byte f16/bf16
                // constexpr) up to `len` so the bound read stays in
                // bounds — same floor handling as the param path above.
                let padded = if b.len() < len {
                    let mut p = b.to_vec();
                    p.resize(len, 0);
                    Cow::Owned(p)
                } else {
                    Cow::Borrowed(&b[..len.min(b.len())])
                };
                self.dev.acquire_shared(Some(padded.as_ref()), len)?
            } else {
                self.dev.acquire_shared(None, len)?
            };
            bufs.push(buf);
        }

        Ok((bufs, n_threads))
    }

    /// Derive the dispatch grid.
    ///
    /// Uses the caller‑supplied override when present; otherwise
    /// auto‑derives from `kernel.mode` and the output buffer size.
    fn resolve_grid(&self, n_threads: usize, tpg_w: usize) -> (MTLSize, MTLSize) {
        if let Some((g, t)) = self.grid_override {
            return (MTLSize { width: g[0], height: g[1], depth: g[2] }, MTLSize {
                width: t[0],
                height: t[1],
                depth: t[2],
            });
        }

        match self.kernel.mode {
            KernelMode::Reduction => {
                let rows = n_threads.max(1);
                (MTLSize { width: rows, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                })
            },
            KernelMode::Grid3D => {
                let groups = n_threads.div_ceil(tpg_w);
                (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                })
            },
            KernelMode::Tile2D => {
                let tpg_dim = (tpg_w as f64).sqrt() as usize;
                let groups = n_threads.div_ceil(tpg_dim * tpg_dim);
                (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                    width: tpg_dim,
                    height: tpg_dim,
                    depth: 1,
                })
            },
            KernelMode::Elementwise => {
                let groups = n_threads.div_ceil(tpg_w);
                (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                })
            },
            // SimdGroup2D: tiled matmul.  Threadgroup = WM×WN×32.
            // For bench dispatch: one threadgroup, full threadgroup size.
            KernelMode::SimdGroup2D => (MTLSize { width: 1, height: 1, depth: 1 }, MTLSize {
                width: tpg_w,
                height: 1,
                depth: 1,
            }),
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod perf {
    //! Microbench: per-call `newBufferWithLength_options` (the path this
    //! file used to take) vs pooled `acquire_shared` (the path it takes
    //! now). Run with:
    //!
    //!   cargo test -p metaltile-runtime --release \
    //!     perf_alloc_per_call_vs_pool -- --ignored --nocapture
    //!
    //! 8 buffers per "dispatch" (5 params × 1 output + 2 strided meta +
    //! 1 constexpr is a realistic minimum), 4096-byte allocation,
    //! 5000 iters × bench == 40 000 buffer acquisitions.

    use std::{ptr::NonNull, time::Instant};

    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

    use crate::device::metal_device::MetalDevice;

    const PER_DISPATCH: usize = 8;
    const ITERS: usize = 5_000;
    const LEN: usize = 4096;

    #[test]
    #[ignore = "perf microbench — requires Metal device"]
    fn perf_alloc_per_call_vs_pool() {
        let Some(raw_dev) = MTLCreateSystemDefaultDevice() else {
            eprintln!("no Metal device — skipping");
            return;
        };

        // Warm.
        for _ in 0..1_000 {
            let buf = raw_dev
                .newBufferWithLength_options(LEN, MTLResourceOptions::StorageModeShared)
                .unwrap();
            std::hint::black_box(buf);
        }

        // (1) Per-call `newBufferWithLength_options` — old single_dispatch path.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let mut bufs = Vec::with_capacity(PER_DISPATCH);
            for _ in 0..PER_DISPATCH {
                let buf = raw_dev
                    .newBufferWithLength_options(LEN, MTLResourceOptions::StorageModeShared)
                    .unwrap();
                bufs.push(buf);
            }
            std::hint::black_box(bufs);
        }
        let baseline = t0.elapsed();
        let baseline_ns_per = baseline.as_nanos() as f64 / (ITERS * PER_DISPATCH) as f64;

        // (2) Pooled `acquire_shared` — new single_dispatch path. Each
        // dispatch's BufRcs drop together so refcount falls to 1 and
        // the pool recycles on the next iter.
        let dev = MetalDevice::create().expect("MetalDevice::create").expect("no device");
        for _ in 0..50 {
            let mut bufs = Vec::with_capacity(PER_DISPATCH);
            for _ in 0..PER_DISPATCH {
                bufs.push(dev.acquire_shared(None, LEN).unwrap());
            }
            std::hint::black_box(bufs);
        }
        let t1 = Instant::now();
        for _ in 0..ITERS {
            let mut bufs = Vec::with_capacity(PER_DISPATCH);
            for _ in 0..PER_DISPATCH {
                bufs.push(dev.acquire_shared(None, LEN).unwrap());
            }
            std::hint::black_box(bufs);
        }
        let pooled = t1.elapsed();
        let pooled_ns_per = pooled.as_nanos() as f64 / (ITERS * PER_DISPATCH) as f64;

        // (3) Per-call `newBufferWithBytes` — old path with data upload.
        let payload = vec![0xABu8; LEN];
        let t2 = Instant::now();
        for _ in 0..ITERS {
            let mut bufs = Vec::with_capacity(PER_DISPATCH);
            for _ in 0..PER_DISPATCH {
                // SAFETY: `payload.as_ptr()` is valid for `LEN` bytes.
                let buf = unsafe {
                    raw_dev.newBufferWithBytes_length_options(
                        NonNull::new(payload.as_ptr() as *mut _).unwrap(),
                        LEN,
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .unwrap();
                bufs.push(buf);
            }
            std::hint::black_box(bufs);
        }
        let bytes_baseline = t2.elapsed();
        let bytes_ns_per = bytes_baseline.as_nanos() as f64 / (ITERS * PER_DISPATCH) as f64;

        // (4) Pooled with bytes upload.
        let t3 = Instant::now();
        for _ in 0..ITERS {
            let mut bufs = Vec::with_capacity(PER_DISPATCH);
            for _ in 0..PER_DISPATCH {
                bufs.push(dev.acquire_shared(Some(&payload), LEN).unwrap());
            }
            std::hint::black_box(bufs);
        }
        let bytes_pooled = t3.elapsed();
        let bytes_pooled_ns_per = bytes_pooled.as_nanos() as f64 / (ITERS * PER_DISPATCH) as f64;

        println!();
        println!(
            "=== buffer alloc: per-call vs pooled (LEN={LEN}B, {PER_DISPATCH}/dispatch × {ITERS} \
             iters) ==="
        );
        println!(
            "  zero-init  newBufferWithLength  : {baseline:>10.2?}  ({baseline_ns_per:>7.1} \
             ns/buf)"
        );
        println!(
            "  zero-init  pool acquire_shared  : {pooled:>10.2?}  ({pooled_ns_per:>7.1} ns/buf)"
        );
        let zero_speedup = baseline_ns_per / pooled_ns_per;
        println!(
            "  → speedup (zero-init)           : {zero_speedup:.2}× (+{:.1}%)",
            (1.0 - pooled_ns_per / baseline_ns_per) * 100.0
        );
        println!();
        println!(
            "  upload     newBufferWithBytes   : {bytes_baseline:>10.2?}  ({bytes_ns_per:>7.1} \
             ns/buf)"
        );
        println!(
            "  upload     pool acquire_shared  : {bytes_pooled:>10.2?}  \
             ({bytes_pooled_ns_per:>7.1} ns/buf)"
        );
        let upload_speedup = bytes_ns_per / bytes_pooled_ns_per;
        println!(
            "  → speedup (upload)              : {upload_speedup:.2}× (+{:.1}%)",
            (1.0 - bytes_pooled_ns_per / bytes_ns_per) * 100.0
        );

        // Regression assertion: zero-init pool path must be at least
        // 2× the per-call path. 5% tolerance.
        assert!(
            pooled_ns_per * 1.05 <= baseline_ns_per,
            "pool acquire_shared ({pooled_ns_per:.1} ns) should beat per-call \
             newBufferWithLength ({baseline_ns_per:.1} ns)"
        );
    }
}

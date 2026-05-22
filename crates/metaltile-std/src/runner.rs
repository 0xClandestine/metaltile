//! GPU runner: compile Metal source, allocate buffers, dispatch kernels, measure GPU time.
//!
//! All Metal-specific code is gated with `#[cfg(target_os = "macos")]`.
//! On other platforms every method returns `Err` or a zero-filled stub.

use crate::{
    bench_types::{DType, OpResult, elem_bytes},
    stats::BenchStats,
};

/// Convert IEEE 754 half-float bits to f32.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp5 = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    if exp5 == 0 {
        return f32::from_bits(sign); // denormal → zero (flush)
    }
    if exp5 == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)); // inf/nan
    }
    let exp8 = (exp5 as i32 - 15 + 127) as u32;
    f32::from_bits(sign | (exp8 << 23) | (mantissa << 13))
}

// ── Public types ─────────────────────────────────────────────────────────────

pub struct GpuRunner {
    pub device_name: String,
    #[cfg(target_os = "macos")]
    inner: MacosRunner,
    /// Pre-compiled kernel and scratch buffer for SLC cache-flush.
    /// The 128 MB scratch comfortably exceeds the SLC of every current
    /// Apple Silicon variant, so a single write evicts any cached benchmark
    /// data; when dispatched repeatedly the same kernel also serves as the
    /// full-occupancy workload that keeps DVFS pinned at peak clock through
    /// the upcoming bench window.
    #[cfg(target_os = "macos")]
    slc_kernel: CompiledKernel,
    #[cfg(target_os = "macos")]
    slc_buf: GpuBuffer,
}

#[allow(clippy::manual_non_exhaustive)]
pub struct CompiledKernel {
    #[cfg(target_os = "macos")]
    inner: MacosPipeline,
    #[cfg(not(target_os = "macos"))]
    _priv: (),
}

#[allow(clippy::manual_non_exhaustive)]
pub struct GpuBuffer {
    pub size_bytes: usize,
    #[cfg(target_os = "macos")]
    inner: MacosBuffer,
    #[cfg(not(target_os = "macos"))]
    _priv: (),
}

// ── macOS implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod metal_impl {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::{NSRange, NSString};
    use objc2_metal::{
        MTLBuffer,
        MTLCommandBuffer,
        MTLCommandEncoder,
        MTLCommandQueue,
        MTLCommonCounterSetTimestamp,
        MTLComputeCommandEncoder,
        MTLComputePassDescriptor,
        MTLComputePipelineDescriptor,
        MTLComputePipelineState,
        MTLCounterResultTimestamp,
        MTLCounterSampleBuffer,
        MTLCounterSampleBufferDescriptor,
        MTLCounterSet,
        MTLDataType,
        MTLDevice,
        MTLFunctionConstantValues,
        MTLLibrary,
        MTLResourceOptions,
        MTLStorageMode,
    };

    /// GPU ticks elapsed across one encoder, from the
    /// `MTLCommonCounterSetTimestamp` counter set. This is the only
    /// counter set Apple Silicon exposes through the public
    /// `MTLCounterSampleBuffer` API — the cycles/utilization data Xcode's
    /// GPU debugger displays comes from a private path Apple does not
    /// surface here.
    ///
    /// Convert ticks → nanoseconds via [`MacosRunner::gpu_tick_period_ns`]:
    /// `ns = ticks * tick_period_ns`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct TimestampCounters {
        /// `end - start` GPU timestamp delta, in device ticks.
        pub gpu_ticks: u64,
    }

    /// One iteration's measurement when counter sampling is enabled.
    #[derive(Debug, Clone, Copy)]
    pub struct CounterSample {
        /// Wallclock GPU time, μs — same value [`MacosRunner::measure`] returns.
        pub gpu_us: f64,
        /// Per-encoder timestamp delta from `MTLCommonCounterSetTimestamp`.
        pub ts: TimestampCounters,
    }

    pub struct MacosRunner {
        pub device: Retained<ProtocolObject<dyn MTLDevice>>,
        pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    }

    pub struct MacosPipeline {
        pub pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    pub struct MacosBuffer {
        pub buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    impl MacosRunner {
        pub fn new() -> Result<(String, Self), String> {
            let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
            let name = device.name().to_string();
            let queue = device.newCommandQueue().ok_or("newCommandQueue failed")?;
            Ok((name, MacosRunner { device, queue }))
        }

        pub fn compile(&self, source: &str, fn_name: &str) -> Result<MacosPipeline, String> {
            let opts = objc2_metal::MTLCompileOptions::new();
            let src = NSString::from_str(source);
            let lib: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map_err(|e| format!("compile '{fn_name}': {e}"))?;
            let fname = NSString::from_str(fn_name);
            let func = lib
                .newFunctionWithName(&fname)
                .ok_or_else(|| format!("no function '{fn_name}'"))?;
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&func));
            let pso = self
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    objc2_metal::MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|e| format!("pipeline '{fn_name}': {e}"))?;
            Ok(MacosPipeline { pso })
        }

        /// Compile a kernel with boolean function constants (index → value).
        pub fn compile_with_bool_constants(
            &self,
            source: &str,
            fn_name: &str,
            bool_constants: &[(usize, bool)],
        ) -> Result<MacosPipeline, String> {
            let opts = objc2_metal::MTLCompileOptions::new();
            let src = NSString::from_str(source);
            let lib: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map_err(|e| format!("compile '{fn_name}': {e}"))?;
            let cv = MTLFunctionConstantValues::new();
            for &(idx, val) in bool_constants {
                let val_ptr =
                    std::ptr::NonNull::new(&val as *const bool as *mut std::ffi::c_void).unwrap();
                unsafe {
                    cv.setConstantValue_type_atIndex(val_ptr, MTLDataType::Bool, idx);
                }
            }
            let fname = NSString::from_str(fn_name);
            let func = lib
                .newFunctionWithName_constantValues_error(&fname, &cv)
                .map_err(|e| format!("specialize '{fn_name}': {e}"))?;
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&func));
            let pso = self
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    objc2_metal::MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|e| format!("pipeline '{fn_name}': {e}"))?;
            Ok(MacosPipeline { pso })
        }

        pub fn alloc_bytes(&self, data: &[u8]) -> MacosBuffer {
            use std::ptr::NonNull;
            let len = data.len().max(4);
            let buf = unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(data.as_ptr() as *mut _).unwrap(),
                        len,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("newBufferWithBytes failed")
            };
            MacosBuffer { buf }
        }

        pub fn alloc_zeros(&self, n_bytes: usize) -> MacosBuffer {
            let len = n_bytes.max(4);
            let buf = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .expect("newBufferWithLength failed");
            MacosBuffer { buf }
        }

        pub fn read_bytes(buf: &MacosBuffer, n_bytes: usize) -> Vec<u8> {
            use objc2_metal::MTLBuffer;
            let ptr = buf.buf.contents();
            unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u8, n_bytes) }.to_vec()
        }

        /// Find the `MTLCommonCounterSetTimestamp` counter set on the
        /// device, or `None` if the device doesn't expose any counter sets
        /// at all. Apple Silicon devices observed in the field expose only
        /// this set through the public `MTLCounterSampleBuffer` API.
        pub fn timestamp_counter_set(&self) -> Option<Retained<ProtocolObject<dyn MTLCounterSet>>> {
            let sets = self.device.counterSets()?;
            let target: &NSString = unsafe { MTLCommonCounterSetTimestamp };
            for i in 0..sets.count() {
                let set = sets.objectAtIndex(i);
                if set.name().isEqualToString(target) {
                    return Some(set);
                }
            }
            None
        }

        /// Nanoseconds per GPU tick on this device, derived from a single
        /// paired `(cpu_ns, gpu_ticks)` snapshot via
        /// `sampleTimestamps:gpuTimestamp:`. One snapshot is enough only
        /// because Apple Silicon's GPU timestamp domain *is* mach absolute
        /// time (= nanoseconds); on systems where the two domains diverge
        /// the caller should sample twice and divide the deltas. Returned
        /// as f64 so caller can do `ticks as f64 * period_ns` directly.
        pub fn gpu_tick_period_ns(&self) -> f64 {
            use objc2_metal::MTLTimestamp;
            let mut cpu: MTLTimestamp = 0;
            let mut gpu: MTLTimestamp = 0;
            unsafe {
                self.device.sampleTimestamps_gpuTimestamp(
                    core::ptr::NonNull::new_unchecked(&mut cpu as *mut _),
                    core::ptr::NonNull::new_unchecked(&mut gpu as *mut _),
                );
            }
            if gpu == 0 { 1.0 } else { cpu as f64 / gpu as f64 }
        }

        /// Diagnostic: every counter set name the device exposes. Used by the
        /// `counter_probe` example to surface what's actually available on
        /// non-StageUtilization devices (most Apple Silicon).
        pub fn counter_set_names(&self) -> Vec<String> {
            let Some(sets) = self.device.counterSets() else {
                return vec![];
            };
            (0..sets.count()).map(|i| sets.objectAtIndex(i).name().to_string()).collect()
        }

        /// Diagnostic: which `MTLCounterSamplingPoint` values the device
        /// supports. Returned as `(name, supported)` pairs.
        pub fn counter_sampling_support(&self) -> Vec<(&'static str, bool)> {
            use objc2_metal::MTLCounterSamplingPoint;
            let pts: &[(&'static str, MTLCounterSamplingPoint)] = &[
                ("AtStageBoundary", MTLCounterSamplingPoint::AtStageBoundary),
                ("AtDrawBoundary", MTLCounterSamplingPoint::AtDrawBoundary),
                ("AtDispatchBoundary", MTLCounterSamplingPoint::AtDispatchBoundary),
                ("AtTileDispatchBoundary", MTLCounterSamplingPoint::AtTileDispatchBoundary),
                ("AtBlitBoundary", MTLCounterSamplingPoint::AtBlitBoundary),
            ];
            pts.iter().map(|(name, p)| (*name, self.device.supportsCounterSampling(*p))).collect()
        }

        /// Mirror of [`measure`] that also samples
        /// `MTLCommonCounterSetTimestamp` at start- and end-of-encoder
        /// boundaries for every iteration.
        ///
        /// Returns one [`CounterSample`] per non-warmup iter. The function
        /// allocates a single `MTLCounterSampleBuffer` sized for the full
        /// `warmup + iters` run (kernel-mode alloc is non-trivial; sharing
        /// it across iters keeps the autotune inner loop cheap).
        ///
        /// Fails when the device exposes no timestamp counter set or
        /// sample-buffer allocation fails; in both cases the caller can
        /// fall back to the timing-only [`measure`] path.
        pub fn measure_with_counters(
            &self,
            pso: &MacosPipeline,
            buffers: &[&MacosBuffer],
            tgs: [usize; 3],
            tpg: [usize; 3],
            warmup: usize,
            iters: usize,
        ) -> Result<Vec<CounterSample>, String> {
            use objc2_metal::MTLSize;

            let counter_set = self
                .timestamp_counter_set()
                .ok_or_else(|| "device exposes no timestamp counter set".to_string())?;

            let total_passes = warmup + iters;
            let sample_count = 2 * total_passes; // start + end per pass

            // Allocate the sample buffer once, reuse across iters.
            let desc = MTLCounterSampleBufferDescriptor::new();
            desc.setCounterSet(Some(&counter_set));
            desc.setStorageMode(MTLStorageMode::Shared);
            unsafe { desc.setSampleCount(sample_count) };
            let sample_buf = self
                .device
                .newCounterSampleBufferWithDescriptor_error(&desc)
                .map_err(|e| format!("newCounterSampleBufferWithDescriptor: {:?}", e))?;

            let mut results = Vec::with_capacity(iters);
            for pass in 0..total_passes {
                let start_idx = 2 * pass;
                let end_idx = 2 * pass + 1;

                let pass_desc = MTLComputePassDescriptor::new();
                let attach =
                    unsafe { pass_desc.sampleBufferAttachments().objectAtIndexedSubscript(0) };
                attach.setSampleBuffer(Some(&sample_buf));
                unsafe {
                    attach.setStartOfEncoderSampleIndex(start_idx);
                    attach.setEndOfEncoderSampleIndex(end_idx);
                }

                unsafe {
                    let cb = self.queue.commandBuffer().expect("commandBuffer");
                    let enc = cb
                        .computeCommandEncoderWithDescriptor(&pass_desc)
                        .expect("computeCommandEncoderWithDescriptor");
                    enc.setComputePipelineState(&pso.pso);
                    for (i, b) in buffers.iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.buf), 0, i);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize { width: tgs[0], height: tgs[1], depth: tgs[2] },
                        MTLSize { width: tpg[0], height: tpg[1], depth: tpg[2] },
                    );
                    enc.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();

                    if pass >= warmup {
                        let gpu_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
                        let (start, end) = resolve_timestamp_pair(&sample_buf, start_idx, end_idx)?;
                        results.push(CounterSample {
                            gpu_us,
                            ts: TimestampCounters {
                                gpu_ticks: end.timestamp.wrapping_sub(start.timestamp),
                            },
                        });
                    }
                }
            }
            Ok(results)
        }
    }

    /// Resolve indices `[start_idx, end_idx]` from `sample_buf` as a pair
    /// of `MTLCounterResultTimestamp` structs. The resolved `NSData`
    /// covers the requested `NSRange`; we copy out the two 8-byte structs
    /// at offsets 0 and `size_of::<...>()`.
    fn resolve_timestamp_pair(
        sample_buf: &ProtocolObject<dyn MTLCounterSampleBuffer>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(MTLCounterResultTimestamp, MTLCounterResultTimestamp), String> {
        debug_assert_eq!(end_idx, start_idx + 1, "expected adjacent sample indices");
        let range = NSRange::new(start_idx, 2);
        let data = unsafe { sample_buf.resolveCounterRange(range) }
            .ok_or_else(|| "resolveCounterRange returned nil".to_string())?;
        let bytes = data.to_vec();
        let stride = core::mem::size_of::<MTLCounterResultTimestamp>();
        if bytes.len() < 2 * stride {
            return Err(format!(
                "resolved data too short: got {} bytes, need {}",
                bytes.len(),
                2 * stride,
            ));
        }
        // `MTLCounterResultTimestamp` is `#[repr(C)]` { u64 }, so the
        // resolved buffer layout matches the struct.
        let start = unsafe {
            core::ptr::read_unaligned(bytes.as_ptr() as *const MTLCounterResultTimestamp)
        };
        let end = unsafe {
            core::ptr::read_unaligned(bytes.as_ptr().add(stride) as *const MTLCounterResultTimestamp)
        };
        Ok((start, end))
    }

    impl MacosRunner {
        pub fn measure(
            &self,
            pso: &MacosPipeline,
            buffers: &[&MacosBuffer],
            tgs: [usize; 3],
            tpg: [usize; 3],
            warmup: usize,
            iters: usize,
        ) -> Vec<f64> {
            use objc2_metal::MTLSize;
            let mut results = Vec::with_capacity(iters);
            for pass in 0..(warmup + iters) {
                unsafe {
                    let cb = self.queue.commandBuffer().expect("commandBuffer");
                    let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
                    enc.setComputePipelineState(&pso.pso);
                    for (i, b) in buffers.iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.buf), 0, i);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize { width: tgs[0], height: tgs[1], depth: tgs[2] },
                        MTLSize { width: tpg[0], height: tpg[1], depth: tpg[2] },
                    );
                    enc.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();
                    if pass >= warmup {
                        let gpu_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
                        results.push(gpu_us);
                    }
                }
            }
            results
        }
    }
}

#[cfg(target_os = "macos")]
pub use metal_impl::{CounterSample, TimestampCounters};
#[cfg(target_os = "macos")]
use metal_impl::{MacosBuffer, MacosPipeline, MacosRunner};

/// Non-macOS stub of [`CounterSample`] so callers can write platform-agnostic
/// signatures. All-zero on non-macOS.
#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CounterSample {
    pub gpu_us: f64,
    pub ts: TimestampCounters,
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TimestampCounters {
    pub gpu_ticks: u64,
}

// ── GpuRunner ────────────────────────────────────────────────────────────────

/// Number of SLC-kernel dispatches done by `flush_slc` before each bench.
/// One dispatch (~0.6 ms on M-series) is enough to evict the SLC; the rest
/// keep the GPU clock at peak so the kernel under test doesn't measure
/// part of the DVFS ramp.
#[cfg(target_os = "macos")]
const SLC_FLUSH_DISPATCHES: usize = 16;

/// One-shot dispatch count at process start; sized to comfortably exceed the
/// DVFS ramp regardless of system idle state.
#[cfg(target_os = "macos")]
const WAKE_DISPATCHES: usize = 64;

impl GpuRunner {
    pub fn new() -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            const SLC_FLUSH_MSL: &str = concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n",
                "kernel void _mt_slc_flush(",
                "device uint* buf [[buffer(0)]],",
                "uint gid [[thread_position_in_grid]]",
                ") { buf[gid] = buf[gid] + gid; }"
            );
            const SLC_BYTES: usize = 128 * 1024 * 1024; // > SLC of every current Apple Silicon variant

            let (name, inner) = MacosRunner::new()?;
            let slc_pso = inner
                .compile(SLC_FLUSH_MSL, "_mt_slc_flush")
                .map_err(|e| format!("SLC flush compile: {e}"))?;
            let slc_kernel = CompiledKernel { inner: slc_pso };
            let slc_buf = GpuBuffer { size_bytes: SLC_BYTES, inner: inner.alloc_zeros(SLC_BYTES) };

            let runner = GpuRunner { device_name: name, inner, slc_kernel, slc_buf };
            runner.wake_dvfs();
            Ok(runner)
        }
        #[cfg(not(target_os = "macos"))]
        Err("Metal not available on this platform".into())
    }

    /// One-shot at process start: pull DVFS up to peak so the first bench
    /// doesn't measure a partially-ramped GPU. Subsequent benches are kept
    /// hot by [`flush_slc`].
    #[cfg(target_os = "macos")]
    fn wake_dvfs(&self) { self.run_slc(WAKE_DISPATCHES); }

    #[allow(unused_variables)]
    pub fn compile(&self, source: &str, fn_name: &str) -> Result<CompiledKernel, String> {
        #[cfg(target_os = "macos")]
        {
            Ok(CompiledKernel { inner: self.inner.compile(source, fn_name)? })
        }
        #[cfg(not(target_os = "macos"))]
        Err("not macOS".into())
    }

    /// Compile a kernel with boolean function constants. `bool_constants` is a list of
    /// (function_constant_index, value) pairs.
    #[allow(unused_variables)]
    pub fn compile_with_bool_constants(
        &self,
        source: &str,
        fn_name: &str,
        bool_constants: &[(usize, bool)],
    ) -> Result<CompiledKernel, String> {
        #[cfg(target_os = "macos")]
        {
            Ok(CompiledKernel {
                inner: self.inner.compile_with_bool_constants(source, fn_name, bool_constants)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        Err("not macOS".into())
    }

    // ── Buffer constructors ──────────────────────────────────────────────────

    pub fn buffer_bytes(&self, data: &[u8]) -> GpuBuffer {
        #[cfg(target_os = "macos")]
        return GpuBuffer { size_bytes: data.len(), inner: self.inner.alloc_bytes(data) };
        #[cfg(not(target_os = "macos"))]
        GpuBuffer { size_bytes: data.len(), _priv: () }
    }

    pub fn buffer_zeros(&self, n_bytes: usize) -> GpuBuffer {
        #[cfg(target_os = "macos")]
        return GpuBuffer { size_bytes: n_bytes, inner: self.inner.alloc_zeros(n_bytes) };
        #[cfg(not(target_os = "macos"))]
        GpuBuffer { size_bytes: n_bytes, _priv: () }
    }

    pub fn buffer_f32(&self, data: &[f32]) -> GpuBuffer {
        // Zero-copy view as bytes via bytemuck::Pod. Apple Silicon is LE
        // and f32/u16 are POD, so this is byte-identical to the previous
        // `.flat_map(.to_le_bytes()).collect()` path without the
        // intermediate Vec<u8> alloc.
        self.buffer_bytes(bytemuck::cast_slice(data))
    }

    /// `data` is raw fp16 bits (e.g. `0x3C00` = 1.0).
    pub fn buffer_f16(&self, data: &[u16]) -> GpuBuffer {
        self.buffer_bytes(bytemuck::cast_slice(data))
    }

    pub fn buffer_u32(&self, v: u32) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_i32(&self, v: i32) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_u64(&self, v: u64) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_i64(&self, v: i64) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_f32_scalar(&self, v: f32) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }

    // ── Readback ─────────────────────────────────────────────────────────────

    /// Read `n_bytes` raw bytes back from a GPU buffer.
    #[allow(unused_variables)]
    pub fn read_bytes(&self, buf: &GpuBuffer, n_bytes: usize) -> Vec<u8> {
        #[cfg(target_os = "macos")]
        {
            MacosRunner::read_bytes(&buf.inner, n_bytes)
        }
        #[cfg(not(target_os = "macos"))]
        vec![0u8; n_bytes]
    }

    /// Read `n` f32 values back from a GPU buffer allocated with buffer_zeros / buffer_f32.
    /// The buffer must use StorageModeShared (all buffers created by GpuRunner do).
    #[allow(unused_variables)]
    pub fn read_f32_slice(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let bytes = MacosRunner::read_bytes(&buf.inner, n * 4);
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0f32; n]
    }

    /// Read `n` bfloat16 values back from a GPU buffer, returned as f32.
    /// BF16 is just the top 16 bits of a float32 representation.
    #[allow(unused_variables)]
    pub fn read_bf16_slice(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let bytes = MacosRunner::read_bytes(&buf.inner, n * 2);
            bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0f32; n]
    }

    /// Read `n` f16 values back from a GPU buffer, returned as f32.
    #[allow(unused_variables)]
    pub fn read_f16_slice(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let bytes = MacosRunner::read_bytes(&buf.inner, n * 2);
            bytes
                .chunks_exact(2)
                .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0f32; n]
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────

    #[allow(unused_variables)]
    pub fn measure(
        &self,
        kernel: &CompiledKernel,
        buffers: &[&GpuBuffer],
        tgs: [usize; 3],
        tpg: [usize; 3],
        warmup: usize,
        iters: usize,
    ) -> Vec<f64> {
        #[cfg(target_os = "macos")]
        {
            let raw: Vec<&MacosBuffer> = buffers.iter().map(|b| &b.inner).collect();
            self.inner.measure(&kernel.inner, &raw, tgs, tpg, warmup, iters)
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0; iters]
    }

    /// Nanoseconds per GPU tick reported by [`Self::measure_with_counters`].
    /// 1.0 on non-macOS (no counter data available).
    pub fn gpu_tick_period_ns(&self) -> f64 {
        #[cfg(target_os = "macos")]
        {
            self.inner.gpu_tick_period_ns()
        }
        #[cfg(not(target_os = "macos"))]
        1.0
    }

    /// Read PSO reflection (`max_total_threads_per_threadgroup`,
    /// `static_threadgroup_memory_length`, `thread_execution_width`)
    /// from the compiled kernel. These are derived from the PSO at
    /// creation time and reflect register pressure, threadgroup-memory
    /// usage, and SIMD width respectively.
    ///
    /// On non-macOS targets returns `PsoReflection::default()`.
    #[allow(unused_variables)]
    pub fn pso_reflection(
        &self,
        kernel: &CompiledKernel,
    ) -> metaltile_runtime::autotune::PsoReflection {
        #[cfg(target_os = "macos")]
        {
            use objc2_metal::MTLComputePipelineState;
            let pso = &kernel.inner.pso;
            metaltile_runtime::autotune::PsoReflection {
                max_total_threads_per_threadgroup: pso.maxTotalThreadsPerThreadgroup() as u64,
                static_threadgroup_memory_length: pso.staticThreadgroupMemoryLength() as u64,
                thread_execution_width: pso.threadExecutionWidth() as u64,
            }
        }
        #[cfg(not(target_os = "macos"))]
        metaltile_runtime::autotune::PsoReflection::default()
    }

    /// Diagnostic: counter-set names exposed by this device.
    /// Empty on non-macOS. See [`Self::measure_with_counters`].
    pub fn counter_set_names(&self) -> Vec<String> {
        #[cfg(target_os = "macos")]
        {
            self.inner.counter_set_names()
        }
        #[cfg(not(target_os = "macos"))]
        vec![]
    }

    /// Diagnostic: `(sampling_point_name, supported)` for every
    /// `MTLCounterSamplingPoint` variant. Empty on non-macOS.
    pub fn counter_sampling_support(&self) -> Vec<(&'static str, bool)> {
        #[cfg(target_os = "macos")]
        {
            self.inner.counter_sampling_support()
        }
        #[cfg(not(target_os = "macos"))]
        vec![]
    }

    /// Like [`Self::measure`], but also samples
    /// `MTLCommonCounterSetStageUtilization` counters at the start and end
    /// of every iteration's encoder. Returns one [`CounterSample`] per
    /// non-warmup iter.
    ///
    /// Returns `Err` when the device exposes no stage-utilization counter
    /// set or sample-buffer allocation fails. On Apple Silicon (M1+) the
    /// happy path holds; on older Intel macOS hardware callers should fall
    /// back to [`Self::measure`].
    #[allow(unused_variables)]
    pub fn measure_with_counters(
        &self,
        kernel: &CompiledKernel,
        buffers: &[&GpuBuffer],
        tgs: [usize; 3],
        tpg: [usize; 3],
        warmup: usize,
        iters: usize,
    ) -> Result<Vec<CounterSample>, String> {
        #[cfg(target_os = "macos")]
        {
            let raw: Vec<&MacosBuffer> = buffers.iter().map(|b| &b.inner).collect();
            self.inner.measure_with_counters(&kernel.inner, &raw, tgs, tpg, warmup, iters)
        }
        #[cfg(not(target_os = "macos"))]
        Err("Metal counter sampling is only available on macOS".into())
    }

    #[allow(unused_variables)]
    pub fn bench(
        &self,
        kernel: &CompiledKernel,
        buffers: &[&GpuBuffer],
        tgs: [usize; 3],
        tpg: [usize; 3],
        warmup: usize,
        iters: usize,
    ) -> BenchStats {
        BenchStats::from_samples(self.measure(kernel, buffers, tgs, tpg, warmup, iters))
    }

    /// Write 128 MB to a scratch buffer to evict the System Level Cache, and
    /// repeat the dispatch enough times to leave the GPU clock pinned at peak
    /// when the next bench dispatch starts.
    ///
    /// One dispatch alone (~0.6 ms on M-series) evicts the SLC but is too
    /// short to keep DVFS at peak: between flush_slc returning and the bench
    /// kernel's first warmup iteration there's enough CPU-side setup that the
    /// clock can fall back to idle, and small kernels (low core occupancy)
    /// can then stay stuck in the slow regime for the whole timed window.
    /// Repeating the dispatch turns flush_slc into the sustained, full-grid
    /// workload that DVFS treats as "stay at peak".
    pub fn flush_slc(&self) {
        #[cfg(target_os = "macos")]
        self.run_slc(SLC_FLUSH_DISPATCHES);
    }

    /// Internal: dispatch the SLC kernel `n` times back-to-back.
    #[cfg(target_os = "macos")]
    fn run_slc(&self, n: usize) {
        const N_ELEM: usize = 128 * 1024 * 1024 / 4; // 32 M uint32 elements
        const TPG: usize = 256;
        for _ in 0..n {
            self.inner.measure(
                &self.slc_kernel.inner,
                &[&self.slc_buf.inner],
                [N_ELEM / TPG, 1, 1],
                [TPG, 1, 1],
                0,
                1,
            );
        }
    }

    /// Returns true if the device supports simdgroup matrix operations (M1+ / Apple GPU family 7+).
    pub fn supports_simd_matrix(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            use objc2_metal::{MTLDevice, MTLGPUFamily};
            let dev = &self.inner.device;
            // Apple GPU families are cumulative — Apple10 (M5) implies Apple9/8/7 —
            // so any of these returning true is sufficient. Listed newest-first to
            // short-circuit on modern hardware.
            dev.supportsFamily(MTLGPUFamily::Apple10)
                || dev.supportsFamily(MTLGPUFamily::Apple9)
                || dev.supportsFamily(MTLGPUFamily::Apple8)
                || dev.supportsFamily(MTLGPUFamily::Apple7)
        }
        #[cfg(not(target_os = "macos"))]
        false
    }
}

// ── Dtype ↔ GPU buffer helpers ────────────────────────────────────────────────

fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 31) as u16) << 15;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant32 = x & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mant16 = mant32 >> 13;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xFFF;
    let round_up = round_bit == 1 && (sticky != 0 || (mant16 & 1) == 1);
    let mant16 = (mant16 + u32::from(round_up)) as u16;
    if mant16 > 0x3FF {
        sign | (((exp + 1) as u16) << 10)
    } else {
        sign | ((exp as u16) << 10) | mant16
    }
}

fn f32_to_bf16(v: f32) -> u16 {
    let x = v.to_bits();
    let rounded = x.wrapping_add(0x7FFF).wrapping_add((x >> 16) & 1);
    (rounded >> 16) as u16
}

pub fn buffer_typed(runner: &GpuRunner, vals: &[f32], dt: DType) -> GpuBuffer {
    match dt {
        DType::F32 => runner.buffer_f32(vals),
        DType::F16 => runner.buffer_f16(&vals.iter().map(|&v| f32_to_f16(v)).collect::<Vec<_>>()),
        DType::BF16 => runner.buffer_f16(&vals.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>()),
        DType::I32 => runner
            .buffer_bytes(&vals.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect::<Vec<_>>()),
        DType::U32 => runner
            .buffer_bytes(&vals.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect::<Vec<_>>()),
        DType::I8 => runner.buffer_bytes(&vals.iter().map(|&v| v as i8 as u8).collect::<Vec<_>>()),
        DType::U8 => runner.buffer_bytes(&vals.iter().map(|&v| v as u8).collect::<Vec<_>>()),
        DType::Bool => runner.buffer_f32(vals),
        DType::I4 | DType::U64 | DType::I64 =>
            unimplemented!("buffer_typed: unsupported dtype {dt:?}"),
    }
}

pub fn zeros_typed(runner: &GpuRunner, n: usize, dt: DType) -> GpuBuffer {
    runner.buffer_zeros(n * elem_bytes(dt))
}

pub fn read_typed(runner: &GpuRunner, buf: &GpuBuffer, n: usize, dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => runner.read_f32_slice(buf, n),
        DType::F16 => runner.read_f16_slice(buf, n),
        DType::BF16 => runner.read_bf16_slice(buf, n),
        DType::I32 => {
            let bytes = runner.read_bytes(buf, n * 4);
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        },
        DType::U32 => {
            let bytes = runner.read_bytes(buf, n * 4);
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        },
        DType::I8 => {
            let bytes = runner.read_bytes(buf, n);
            bytes.iter().map(|&b| b as i8 as f32).collect()
        },
        DType::U8 => {
            let bytes = runner.read_bytes(buf, n);
            bytes.iter().map(|&b| b as f32).collect()
        },
        DType::Bool => runner.read_f32_slice(buf, n),
        DType::I4 | DType::U64 | DType::I64 =>
            unimplemented!("read_typed: unsupported dtype {dt:?}"),
    }
}

// ── Single-run dispatch ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn run_typed_once(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
    dt: DType,
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    read_typed(runner, out, n, dt)
}

pub fn run_f16_once_as_f32(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    runner.read_f16_slice(out, n)
}

// ── Throughput ───────────────────────────────────────────────────────────────

pub fn to_gflops(st: &BenchStats, flops: f64) -> Option<f64> {
    st.is_valid().then(|| flops / (st.min_us * 1e-6) / 1e9)
}

pub fn to_gbps(st: &BenchStats, bytes: f64) -> Option<f64> {
    // Use min to report steady-state throughput. Slow tail samples are usually
    // leftover DVFS ramp or scheduler noise rather than the kernel's true
    // wall-clock cost; min is also stable across bimodal warmup-boundary splits
    // where median can flip wildly on a 1-sample shift. The p95/p99/cv% columns
    // surface variance separately when warmup wasn't enough.
    st.is_valid().then(|| bytes / (st.min_us * 1e-6) / 1e9)
}

/// Default warmup / timed iteration counts for the bench helpers.
///
/// DVFS ramp is handled by [`GpuRunner::flush_slc`] (which dispatches the SLC
/// kernel repeatedly at full grid occupancy before each bench), so warmup here
/// only has to absorb per-kernel first-touch effects (page residency, JIT
/// dispatch state). 15 iterations is comfortably past that boundary for every
/// kernel currently in the suite.
const BENCH_WARMUP: usize = 15;
const BENCH_ITERS: usize = 10;

pub fn bench_gbps(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    grid: [usize; 3],
    tpg: [usize; 3],
    bytes: f64,
) -> Option<(f64, BenchStats)> {
    runner.flush_slc();
    let stats = runner.bench(kernel, buffers, grid, tpg, BENCH_WARMUP, BENCH_ITERS);
    to_gbps(&stats, bytes).map(|x| (x, stats))
}

/// Like bench_gbps but discards timing stats (used when not needed).
pub fn bench_gbps_only(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    grid: [usize; 3],
    tpg: [usize; 3],
    bytes: f64,
) -> Option<f64> {
    runner.flush_slc();
    to_gbps(&runner.bench(kernel, buffers, grid, tpg, BENCH_WARMUP, BENCH_ITERS), bytes)
}

pub fn bench_all_dtypes<F>(runner: &GpuRunner, f: F) -> Vec<OpResult>
where F: Fn(&GpuRunner, DType) -> Vec<OpResult> {
    crate::bench_types::FLOAT_DTYPES.iter().flat_map(|&dt| f(runner, dt)).collect()
}

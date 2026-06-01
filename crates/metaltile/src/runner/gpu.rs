//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU runner: compile Metal source, allocate buffers, dispatch kernels, measure GPU time.
//!
//! All Metal-specific code is gated with `#[cfg(target_os = "macos")]`.
//! On other platforms every method returns `Err` or a zero-filled stub.

use metaltile_core::DType;

// ── Timing statistics ─────────────────────────────────────────────────────────

/// Summary statistics for a set of GPU timing measurements.
#[derive(Debug, Clone)]
pub struct BenchStats {
    /// Minimum (best) GPU execution time in microseconds.
    pub min_us: f64,
    /// Mean GPU execution time in microseconds.
    pub mean_us: f64,
    /// Median (p50) GPU execution time in microseconds.
    pub median_us: f64,
    /// 95th-percentile GPU execution time in microseconds.
    pub p95_us: f64,
    /// 99th-percentile GPU execution time in microseconds.
    pub p99_us: f64,
    /// Standard deviation in microseconds.
    pub stddev_us: f64,
    /// Coefficient of variation (stddev/mean × 100). >5% suggests instability.
    pub cv_pct: f64,
}

impl BenchStats {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        assert!(!samples.is_empty());
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();
        let min = samples[0];
        let mean = samples.iter().sum::<f64>() / n as f64;
        let median = samples[n / 2];
        let p95 = samples[(n * 95 / 100).min(n - 1)];
        let p99 = samples[(n * 99 / 100).min(n - 1)];
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        let cv_pct = if mean > 0.0 { stddev / mean * 100.0 } else { 0.0 };
        BenchStats {
            min_us: min,
            mean_us: mean,
            median_us: median,
            p95_us: p95,
            p99_us: p99,
            stddev_us: stddev,
            cv_pct,
        }
    }

    /// True if timing data came from a real GPU dispatch (non-macOS always returns false).
    pub fn is_valid(&self) -> bool { self.mean_us > 0.0 }
}

// ── Convert IEEE 754 half-float bits to f32 ───────────────────────────────────

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp5 = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    if exp5 == 0 {
        return f32::from_bits(sign);
    }
    if exp5 == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    let exp8 = (exp5 as i32 - 15 + 127) as u32;
    f32::from_bits(sign | (exp8 << 23) | (mantissa << 13))
}

// ── Public types ─────────────────────────────────────────────────────────────

pub struct GpuRunner {
    pub device_name: String,
    #[cfg(target_os = "macos")]
    inner: MacosRunner,
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
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer,
        MTLCommandBuffer,
        MTLCommandEncoder,
        MTLCommandQueue,
        MTLComputeCommandEncoder,
        MTLComputePipelineDescriptor,
        MTLComputePipelineState,
        MTLDataType,
        MTLDevice,
        MTLFunctionConstantValues,
        MTLLibrary,
        MTLResourceOptions,
    };

    pub struct MacosRunner {
        pub device: Retained<ProtocolObject<dyn MTLDevice>>,
        pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        library_cache: std::sync::Mutex<
            std::collections::HashMap<u64, Retained<ProtocolObject<dyn MTLLibrary>>>,
        >,
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
            Ok((name, MacosRunner {
                device,
                queue,
                library_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            }))
        }

        fn library(
            &self,
            source: &str,
        ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut hasher);
            let key = hasher.finish();
            if let Some(lib) = self.library_cache.lock().unwrap().get(&key) {
                return Ok(lib.clone());
            }
            let opts = objc2_metal::MTLCompileOptions::new();
            let src = NSString::from_str(source);
            let lib: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map_err(|e| format!("compile source: {e}"))?;
            self.library_cache.lock().unwrap().insert(key, lib.clone());
            Ok(lib)
        }

        pub fn compile(&self, source: &str, fn_name: &str) -> Result<MacosPipeline, String> {
            let lib = self.library(source)?;
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

        pub fn compile_with_bool_constants(
            &self,
            source: &str,
            fn_name: &str,
            bool_constants: &[(usize, bool)],
        ) -> Result<MacosPipeline, String> {
            let lib = self.library(source)?;
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
use metal_impl::{MacosBuffer, MacosPipeline, MacosRunner};

// ── GpuRunner ────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
const SLC_FLUSH_DISPATCHES: usize = 16;
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
            const SLC_BYTES: usize = 128 * 1024 * 1024;

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

    #[cfg(target_os = "macos")]
    fn wake_dvfs(&self) { self.run_slc(WAKE_DISPATCHES); }

    #[allow(unused_variables)]
    pub fn compile(&self, source: &str, fn_name: &str) -> Result<CompiledKernel, String> {
        #[cfg(target_os = "macos")]
        return Ok(CompiledKernel { inner: self.inner.compile(source, fn_name)? });
        #[cfg(not(target_os = "macos"))]
        Err("not macOS".into())
    }

    #[allow(unused_variables)]
    pub fn compile_with_bool_constants(
        &self,
        source: &str,
        fn_name: &str,
        bool_constants: &[(usize, bool)],
    ) -> Result<CompiledKernel, String> {
        #[cfg(target_os = "macos")]
        return Ok(CompiledKernel {
            inner: self.inner.compile_with_bool_constants(source, fn_name, bool_constants)?,
        });
        #[cfg(not(target_os = "macos"))]
        Err("not macOS".into())
    }

    // ── Buffer constructors ───────────────────────────────────────────────────

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

    // ── Readback ──────────────────────────────────────────────────────────────

    #[allow(unused_variables)]
    pub fn read_bytes(&self, buf: &GpuBuffer, n_bytes: usize) -> Vec<u8> {
        #[cfg(target_os = "macos")]
        {
            MacosRunner::read_bytes(&buf.inner, n_bytes)
        }
        #[cfg(not(target_os = "macos"))]
        vec![0u8; n_bytes]
    }

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

    // ── Dispatch ──────────────────────────────────────────────────────────────

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

    pub fn flush_slc(&self) {
        #[cfg(target_os = "macos")]
        self.run_slc(SLC_FLUSH_DISPATCHES);
    }

    #[cfg(target_os = "macos")]
    fn run_slc(&self, n: usize) {
        const N_ELEM: usize = 128 * 1024 * 1024 / 4;
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

    pub fn supports_simd_matrix(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            use objc2_metal::{MTLDevice, MTLGPUFamily};
            let dev = &self.inner.device;
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

/// Number of bytes per element for common dtypes.
pub fn elem_bytes(dt: DType) -> usize {
    match dt {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U8 | DType::Bool | DType::I8 => 1,
        _ => 4,
    }
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
        DType::I4 | DType::U64 | DType::I64 => {
            unimplemented!("buffer_typed: unsupported dtype {dt:?}")
        },
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
        DType::I4 | DType::U64 | DType::I64 => {
            unimplemented!("read_typed: unsupported dtype {dt:?}")
        },
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
    st.is_valid().then(|| bytes / (st.min_us * 1e-6) / 1e9)
}

pub const BENCH_WARMUP: usize = 15;
pub const BENCH_ITERS: usize = 10;

pub fn bench_gbps(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    grid: [usize; 3],
    tpg: [usize; 3],
    bytes: f64,
) -> Option<(f64, BenchStats)> {
    bench_gbps_with(runner, kernel, buffers, grid, tpg, bytes, BENCH_WARMUP, BENCH_ITERS)
}

/// Like [`bench_gbps`] but with explicit warmup / iteration counts, allowing
/// `tile.toml` `warmup_runs` / `runs` to override the compile-time defaults.
#[allow(clippy::too_many_arguments)]
pub fn bench_gbps_with(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    grid: [usize; 3],
    tpg: [usize; 3],
    bytes: f64,
    warmup: usize,
    iters: usize,
) -> Option<(f64, BenchStats)> {
    runner.flush_slc();
    let stats = runner.bench(kernel, buffers, grid, tpg, warmup, iters);
    to_gbps(&stats, bytes).map(|x| (x, stats))
}

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

#[cfg(test)]
mod throughput_tests {
    use super::{BenchStats, to_gbps, to_gflops};

    /// Both throughput conversions use the `min` sample (steady state) and are
    /// `None` for the off-GPU all-zero stub (`is_valid() == false`).
    #[test]
    fn throughput_conversions_use_min_and_guard_invalid() {
        // 1e9 FLOPs in 100 µs = 1e13 FLOP/s = 10_000 GFLOP/s; min sample is 100.
        let st = BenchStats::from_samples(vec![100.0, 150.0, 300.0]);
        assert_eq!(to_gflops(&st, 1e9), Some(10_000.0));
        // 1e9 bytes in 100 µs = 1e13 B/s = 10_000 GB/s.
        assert_eq!(to_gbps(&st, 1e9), Some(10_000.0));
        // Off-GPU stub: all-zero samples ⇒ invalid ⇒ no throughput.
        let stub = BenchStats::from_samples(vec![0.0, 0.0]);
        assert_eq!(to_gflops(&stub, 1e9), None);
        assert_eq!(to_gbps(&stub, 1e9), None);
    }
}

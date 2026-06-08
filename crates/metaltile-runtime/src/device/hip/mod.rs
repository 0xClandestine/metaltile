//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! HIP / ROCm runtime backend (`AMD_BACKEND_SPEC.md §4-§5` — Phase 1).
//!
//! `HipDevice` is the AMD analog of `CudaDevice`: it owns a HIP context
//! and provides compile (hipRTC: HIP C++ → AMDGPU code-object → module),
//! allocate, upload, launch, and read-back. Phase 1 = elementwise smoke.
//!
//! Feature-gated (`hip`); designed for Linux **and** Windows ROCm.
//! The HIP host API is a near-1:1 rename of the CUDA Driver API, so the
//! module structure mirrors `device/cuda/mod.rs` closely.
//!
//! ## hipRTC compile target
//!
//! hipRTC compiles for a specific `gfx*` architecture (the AMDGPU LLVM
//! target name). We query the device name via `hipDeviceGetName` and pass
//! `--offload-arch=<gfx>` to the compiler. For the user's RX 9070 XT
//! that's `gfx1201` (RDNA 4, wave32).

mod ffi;

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use metaltile_codegen::{CodegenBackend, HipGenerator};
use metaltile_core::ir::Kernel;

use crate::error::MetalTileError;

use ffi::*;

fn synth_strided_meta(shape: &metaltile_core::shape::Shape, strides: bool) -> Vec<u8> {
    use metaltile_core::shape::Dim;
    let dims: Vec<u32> = (0..shape.rank())
        .map(|i| match shape.dim(i) {
            Some(Dim::Known(n)) => *n as u32,
            _ => 1,
        })
        .collect();
    let vals: Vec<u32> = if strides {
        let mut s = vec![1u32; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[i + 1];
        }
        s
    } else {
        dims
    };
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn hip_check(res: hipError_t, what: &str) -> Result<(), MetalTileError> {
    if res == HIP_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let s = hipGetErrorString(res);
        if s.is_null() {
            format!("HIP error code {res}")
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        }
    };
    Err(MetalTileError::Dispatch(format!("{what}: {msg}")))
}

fn hiprtc_check(res: hiprtcResult, what: &str) -> Result<(), MetalTileError> {
    if res == HIPRTC_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let s = hiprtcGetErrorString(res);
        if s.is_null() {
            format!("hiprtc error code {res}")
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        }
    };
    Err(MetalTileError::Compilation(format!("{what}: {msg}")))
}

/// A device-side allocation. Frees on drop.
pub struct HipBuffer<'d> {
    ptr: hipDeviceptr_t,
    len: usize,
    _dev: &'d HipDevice,
}

impl HipBuffer<'_> {
    pub fn device_ptr(&self) -> hipDeviceptr_t { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Drop for HipBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { hipFree(self.ptr) };
        }
    }
}

/// A compiled, loaded HIP module. Unloads on drop.
pub struct HipModuleHandle {
    module: hipModule_t,
}

impl HipModuleHandle {
    pub fn function(&self, name: &str) -> Result<HipKernel, MetalTileError> {
        let cname = CString::new(name).map_err(|e| MetalTileError::Dispatch(e.to_string()))?;
        let mut func: hipFunction_t = ptr::null_mut();
        hip_check(
            unsafe { hipModuleGetFunction(&mut func, self.module, cname.as_ptr()) },
            &format!("hipModuleGetFunction({name})"),
        )?;
        Ok(HipKernel { func })
    }
}

impl Drop for HipModuleHandle {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { hipModuleUnload(self.module) };
        }
    }
}

/// Handle to a `__global__` function inside a [`HipModuleHandle`].
#[derive(Clone, Copy)]
pub struct HipKernel {
    func: hipFunction_t,
}

struct Prepared<'d> {
    _module: HipModuleHandle,
    func: HipKernel,
    dev_bufs: Vec<HipBuffer<'d>>,
    dev_ptrs: Vec<hipDeviceptr_t>,
    scalars: Vec<Vec<u8>>,
    out_meta: Vec<Option<(String, usize)>>,
    shared_bytes: u32,
}

impl Prepared<'_> {
    fn args(&self) -> Vec<*mut c_void> {
        let mut args: Vec<*mut c_void> =
            Vec::with_capacity(self.dev_ptrs.len() + self.scalars.len());
        for p in &self.dev_ptrs {
            args.push(p as *const hipDeviceptr_t as *mut c_void);
        }
        for s in &self.scalars {
            args.push(s.as_ptr() as *mut c_void);
        }
        args
    }
}

/// HIP device + context. AMD analog of `CudaDevice` / NVIDIA Driver-API
/// context.
pub struct HipDevice {
    ctx: hipCtx_t,
    name: String,
    gfx: String,
    warp_size: i32,
    /// `hipDeviceAttributeMaxSharedMemoryPerBlockOptin` — the largest
    /// dynamic shared-memory size a kernel can request via
    /// `hipFuncSetAttribute`. Drives the dispatch error path for
    /// cooperative-MMA kernels that exceed the limit.
    max_shared_per_block_optin: i32,
}

unsafe impl Send for HipDevice {}
unsafe impl Sync for HipDevice {}

impl HipDevice {
    /// Initialize HIP, grab device 0, create a context. Returns `Ok(None)`
    /// if no HIP device is present (mirrors `CudaDevice::create`).
    pub fn create() -> Result<Option<Self>, MetalTileError> {
        unsafe {
            if hipInit(0) != HIP_SUCCESS {
                return Ok(None);
            }
            let mut dev: hipDevice_t = 0;
            if hipDeviceGet(&mut dev, 0) != HIP_SUCCESS {
                return Ok(None);
            }
            // Device marketing name (e.g. "AMD Radeon RX 9070 XT"). Used in
            // diagnostics; the gfx target is parsed separately below.
            let name = {
                let mut buf = [0u8; 256];
                hip_check(
                    hipDeviceGetName(buf.as_mut_ptr() as *mut c_char, buf.len() as i32, dev),
                    "hipDeviceGetName",
                )?;
                let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
                String::from_utf8_lossy(&buf[..end]).into_owned()
            };
            // gfx target detection. `hipDeviceGetName` on Windows returns the
            // marketing name, not the gfx code; the most reliable source is
            // hipDeviceProp_t.gcnArchName, but that struct is large and varies
            // by ROCm version. Phase 1 takes the simple route: read
            // `METALTILE_HIP_GFX` if set, else default to gfx1201 (RDNA 4 /
            // RX 9070 XT — the user's primary target). Override for anything
            // else: gfx1100 RDNA 3, gfx942 MI300, gfx950 MI350.
            let gfx = std::env::var("METALTILE_HIP_GFX")
                .unwrap_or_else(|_| "gfx1201".to_string());
            // Derive wave size from gfx family. The CUDA-style attribute
            // query (HIP_DEVICE_ATTRIBUTE_WARP_SIZE) is unreliable on Windows
            // ROCm 7.x — the enum index drifts between releases — so we use
            // the canonical mapping instead. gfx9xx (CDNA) is wave64;
            // gfx10/11/12 (RDNA) is wave32.
            let warp_size: i32 = if gfx.starts_with("gfx9") { 64 } else { 32 };

            // Query the largest dynamic LDS a kernel can opt into. Used
            // both for diagnostics and to validate kernel shared-memory
            // requests before launching (avoids a generic "invalid
            // argument" failure on over-budget MPP kernels).
            let mut max_shared_optin: i32 = 0;
            if hipDeviceGetAttribute(
                &mut max_shared_optin,
                HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                dev,
            ) != HIP_SUCCESS
            {
                // Older ROCm or attribute drift — fall back to the
                // RDNA-4 default cap so per-launch checks aren't gated.
                max_shared_optin = 65536;
            }

            let mut ctx: hipCtx_t = ptr::null_mut();
            hip_check(hipCtxCreate(&mut ctx, 0, dev), "hipCtxCreate")?;
            Ok(Some(HipDevice {
                ctx,
                name,
                gfx,
                warp_size,
                max_shared_per_block_optin: max_shared_optin,
            }))
        }
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn gfx_arch(&self) -> &str { &self.gfx }
    pub fn warp_size(&self) -> i32 { self.warp_size }
    pub fn max_shared_per_block_optin(&self) -> i32 { self.max_shared_per_block_optin }

    /// Compile HIP C++ source → AMDGPU code-object → loaded module via
    /// hipRTC + `hipModuleLoadData`.
    pub fn compile(
        &self,
        src: &str,
        prog_name: &str,
    ) -> Result<HipModuleHandle, MetalTileError> {
        let csrc =
            CString::new(src).map_err(|e| MetalTileError::Compilation(e.to_string()))?;
        let cname = CString::new(prog_name)
            .map_err(|e| MetalTileError::Compilation(e.to_string()))?;

        let mut prog: hiprtcProgram = ptr::null_mut();
        hiprtc_check(
            unsafe {
                hiprtcCreateProgram(
                    &mut prog,
                    csrc.as_ptr(),
                    cname.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "hiprtcCreateProgram",
        )?;

        // Compile options. `--offload-arch=<gfx>` is the HIP analog of
        // `--gpu-architecture=compute_<cc>`. `-ffp-contract=off` is the
        // CUDA `--fmad=false` equivalent — match IEEE oracle rounding.
        //
        // hipRTC bundles `hip/hip_runtime.h` but NOT the type headers
        // (`hip_fp16.h`, `hip_bf16.h`); they live under `<HIP_PATH>/include`.
        // We add the include path so the emitted preamble's
        // `#include <hip/hip_fp16.h>` resolves.
        let arch = CString::new(format!("--offload-arch={}", self.gfx)).unwrap();
        let no_fma = CString::new("-ffp-contract=off").unwrap();
        // Force correctly-rounded f32 divide and sqrt at the IR level.
        // Without this, AMDGPU's backend lowers `a / b` to
        // `V_RCP_F32 + V_MUL_F32` (1 ULP cheaper, 1 ULP less accurate).
        // The flag matches IEEE-754 rounding for every divide that
        // didn't already go through `mt_fdiv`.
        let prec_div = CString::new("-fno-fast-math").unwrap();
        let hip_root = std::env::var("HIP_PATH")
            .or_else(|_| std::env::var("ROCM_PATH"))
            .unwrap_or_else(|_| {
                if cfg!(windows) {
                    r"C:\Program Files\AMD\ROCm\7.1".to_string()
                } else {
                    "/opt/rocm".to_string()
                }
            });
        let inc = CString::new(format!("-I{hip_root}/include")).unwrap();
        let opts: [*const c_char; 4] = [arch.as_ptr(), no_fma.as_ptr(), prec_div.as_ptr(), inc.as_ptr()];
        let compile_res = unsafe {
            hiprtcCompileProgram(prog, opts.len() as _, opts.as_ptr())
        };

        let log = unsafe {
            let mut log_size: usize = 0;
            hiprtcGetProgramLogSize(prog, &mut log_size);
            if log_size > 1 {
                let mut buf = vec![0u8; log_size];
                hiprtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char);
                String::from_utf8_lossy(&buf[..log_size.saturating_sub(1)])
                    .into_owned()
            } else {
                String::new()
            }
        };

        if compile_res != HIPRTC_SUCCESS {
            unsafe { hiprtcDestroyProgram(&mut prog) };
            let msg = unsafe {
                CStr::from_ptr(hiprtcGetErrorString(compile_res))
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(MetalTileError::Compilation(format!(
                "hiprtcCompileProgram failed: {msg}\n--- log ---\n{log}"
            )));
        }

        // Fetch the compiled code object (ELF for AMDGPU).
        let code = unsafe {
            let mut sz: usize = 0;
            hiprtc_check(hiprtcGetCodeSize(prog, &mut sz), "hiprtcGetCodeSize")?;
            let mut buf = vec![0u8; sz];
            hiprtc_check(
                hiprtcGetCode(prog, buf.as_mut_ptr() as *mut c_char),
                "hiprtcGetCode",
            )?;
            buf
        };
        unsafe { hiprtcDestroyProgram(&mut prog) };

        let mut module: hipModule_t = ptr::null_mut();
        hip_check(
            unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) },
            "hipModuleLoadData",
        )?;
        Ok(HipModuleHandle { module })
    }

    pub fn alloc(&self, len: usize) -> Result<HipBuffer<'_>, MetalTileError> {
        if len == 0 {
            return Ok(HipBuffer { ptr: 0, len: 0, _dev: self });
        }
        let mut ptr: hipDeviceptr_t = 0;
        hip_check(unsafe { hipMalloc(&mut ptr, len) }, "hipMalloc")?;
        Ok(HipBuffer { ptr, len, _dev: self })
    }

    pub fn upload(&self, data: &[u8]) -> Result<HipBuffer<'_>, MetalTileError> {
        let buf = self.alloc(data.len())?;
        if !data.is_empty() {
            hip_check(
                unsafe {
                    hipMemcpyHtoD(buf.ptr, data.as_ptr() as *const c_void, data.len())
                },
                "hipMemcpyHtoD",
            )?;
        }
        Ok(buf)
    }

    pub fn download(
        &self,
        buf: &HipBuffer,
        out: &mut [u8],
    ) -> Result<(), MetalTileError> {
        let n = out.len().min(buf.len);
        if n == 0 {
            return Ok(());
        }
        hip_check(
            unsafe { hipMemcpyDtoH(out.as_mut_ptr() as *mut c_void, buf.ptr, n) },
            "hipMemcpyDtoH",
        )
    }

    pub fn launch_1d(
        &self,
        func: HipKernel,
        grid_blocks: u32,
        block_threads: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        self.launch(func, [grid_blocks, 1, 1], [block_threads, 1, 1], 0, args)
    }

    pub fn launch(
        &self,
        func: HipKernel,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        // Pre-check: dynamic shared memory exceeding the device's opt-in
        // cap turns into a generic "invalid argument" at launch time and
        // obscures the real cause. Surface it as a clear error with the
        // numbers so the corpus harness can bucket it as a *device-limit*
        // failure (not a codegen ERROR).
        if shared_bytes as i32 > self.max_shared_per_block_optin {
            return Err(MetalTileError::Dispatch(format!(
                "hipModuleLaunchKernel: shared memory requested ({shared_bytes} bytes) \
                 exceeds device max ({max} bytes) — kernel needs Phase-5 MPP retune \
                 for this hardware",
                max = self.max_shared_per_block_optin
            )));
        }
        if shared_bytes > 0 {
            let attr_res = unsafe {
                hipFuncSetAttribute(
                    func.func,
                    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_bytes as c_int,
                )
            };
            if attr_res != HIP_SUCCESS {
                // The attribute set itself failed — likely the request is
                // still over an implementation-specific cap. Report it
                // explicitly so the harness can bucket it.
                let msg = unsafe {
                    CStr::from_ptr(hipGetErrorString(attr_res))
                        .to_string_lossy()
                        .into_owned()
                };
                return Err(MetalTileError::Dispatch(format!(
                    "hipFuncSetAttribute(MaxDynamicSharedMemorySize={shared_bytes}): {msg}"
                )));
            }
        }
        hip_check(
            unsafe {
                hipModuleLaunchKernel(
                    func.func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_bytes,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )?;
        self.synchronize()
    }

    fn prepare<'d>(
        &'d self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        block: [u32; 3],
    ) -> Result<Prepared<'d>, MetalTileError> {
        let cg = HipGenerator::new();
        let src = cg.generate(kernel).map_err(MetalTileError::Codegen)?;
        let module = self.compile(&src, &format!("{}.hip", kernel.name))?;
        let func = module.function(&kernel.name)?;
        let shared_bytes = cg.shared_bytes(kernel, block[0]) as u32;

        let mut dev_bufs: Vec<HipBuffer> = Vec::new();
        let mut dev_ptrs: Vec<hipDeviceptr_t> = Vec::new();
        let mut out_meta: Vec<Option<(String, usize)>> = Vec::new();
        for p in &kernel.params {
            let bytes = buffers.get(&p.name).ok_or_else(|| {
                MetalTileError::Dispatch(format!(
                    "missing buffer for param '{}'",
                    p.name
                ))
            })?;
            let buf = self.upload(bytes)?;
            dev_ptrs.push(buf.device_ptr());
            out_meta.push(if p.is_output { Some((p.name.clone(), bytes.len())) } else { None });
            dev_bufs.push(buf);

            if p.kind == metaltile_core::ir::ParamKind::Strided {
                for suffix in ["_shape", "_strides"] {
                    let key = format!("{}{}", p.name, suffix);
                    let meta = match buffers.get(&key) {
                        Some(b) => b.clone(),
                        None => synth_strided_meta(&p.shape, suffix == "_strides"),
                    };
                    let mb = self.upload(&meta)?;
                    dev_ptrs.push(mb.device_ptr());
                    out_meta.push(None);
                    dev_bufs.push(mb);
                }
            }
        }

        let mut scalars: Vec<Vec<u8>> = Vec::new();
        for ce in &kernel.constexprs {
            let name = ce.name.name();
            let bytes = buffers.get(name).ok_or_else(|| {
                MetalTileError::Dispatch(format!("missing constexpr '{name}'"))
            })?;
            scalars.push(bytes.clone());
        }
        if kernel.mode == metaltile_core::ir::KernelMode::Elementwise {
            let n_elems = kernel
                .params
                .iter()
                .position(|p| p.is_output)
                .and_then(|i| {
                    let p = &kernel.params[i];
                    buffers
                        .get(&p.name)
                        .map(|b| (b.len() / p.dtype.size_bytes().max(1)) as u32)
                })
                .unwrap_or(0);
            scalars.push(n_elems.to_le_bytes().to_vec());
        }

        Ok(Prepared {
            _module: module,
            func,
            dev_bufs,
            dev_ptrs,
            scalars,
            out_meta,
            shared_bytes,
        })
    }

    /// End-to-end generic dispatch (CUDA `run_kernel` analog).
    pub fn run_kernel(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<BTreeMap<String, Vec<u8>>, MetalTileError> {
        let prep = self.prepare(kernel, buffers, block)?;

        let mut args = prep.args();
        self.launch(prep.func, grid, block, prep.shared_bytes, &mut args)?;
        drop(args);

        let mut out = BTreeMap::new();
        for (buf, meta) in prep.dev_bufs.iter().zip(&prep.out_meta) {
            if let Some((name, len)) = meta {
                let mut host = vec![0u8; *len];
                self.download(buf, &mut host)?;
                out.insert(name.clone(), host);
            }
        }
        Ok(out)
    }

    pub fn synchronize(&self) -> Result<(), MetalTileError> {
        hip_check(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }
}

impl Drop for HipDevice {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { hipCtxDestroy(self.ctx) };
        }
    }
}

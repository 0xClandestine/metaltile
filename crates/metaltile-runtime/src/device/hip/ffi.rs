//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Hand-rolled FFI to the AMD HIP runtime (`amdhip64`) and hipRTC
//! (`hiprtc`) — only the symbols the Phase-1 hipRTC-compile + launch path
//! needs (`AMD_BACKEND_SPEC.md §3-§4`). The HIP host API is intentionally
//! a near-1:1 rename of the CUDA Driver API (HIPIFY exists precisely for
//! this), so this surface mirrors the CUDA `ffi.rs` shape exactly.
//!
//! Windows note: AMD ships the import lib as `amdhip64.lib` (versioned
//! DLL `amdhip64_7.dll` for ROCm 7.x). hipRTC is `hiprtc0701.dll` with
//! import lib `hiprtc.lib`. The build script handles both.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type hipError_t = c_int;
pub type hipDevice_t = c_int;
pub type hipCtx_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;
/// Device pointer — `unsigned long long` on every supported HIP target.
pub type hipDeviceptr_t = u64;

pub type hiprtcResult = c_int;
pub type hiprtcProgram = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;
pub const HIPRTC_SUCCESS: hiprtcResult = 0;

// `hipDeviceAttribute_t` values used here. Mirrored from
// `<hip/hip_runtime_api.h>` — they're stable across ROCm 5/6/7.
pub const HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 23;
pub const HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 24;
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 30;
/// `hipDeviceAttributeMaxSharedMemoryPerBlockOptin` — the largest dynamic
/// LDS a kernel can request via `hipFuncSetAttribute`. On RDNA 4 the
/// per-workgroup default is 64KB; the opt-in can extend into the per-WGP
/// LDS (up to 160KB on gfx1201). On CDNA it's lower per the silicon.
pub const HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 45;

// Function-attribute index for the dynamic shared-memory opt-in.
// HIP's `hipFuncAttributeMaxDynamicSharedMemorySize == 8`; identical to CUDA.
pub const HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

pub const HIP_EVENT_DEFAULT: c_uint = 0;

// On Windows the DLL versioned-name pattern means the simplest portable thing
// is to name the import library; rustc forwards to the linker which resolves
// against `amdhip64.lib`/`hiprtc.lib` (in `lib/`) → loader picks the matching
// DLL at runtime.
#[link(name = "amdhip64")]
unsafe extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipDeviceGet(device: *mut hipDevice_t, ordinal: c_int) -> hipError_t;
    pub fn hipDeviceGetAttribute(
        pi: *mut c_int,
        attrib: c_int,
        dev: hipDevice_t,
    ) -> hipError_t;
    pub fn hipCtxCreate(pctx: *mut hipCtx_t, flags: c_uint, dev: hipDevice_t) -> hipError_t;
    pub fn hipCtxDestroy(ctx: hipCtx_t) -> hipError_t;
    pub fn hipCtxSynchronize() -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    pub fn hipFuncSetAttribute(
        func: hipFunction_t,
        attrib: c_int,
        value: c_int,
    ) -> hipError_t;
    pub fn hipMalloc(dptr: *mut hipDeviceptr_t, bytesize: usize) -> hipError_t;
    pub fn hipFree(dptr: hipDeviceptr_t) -> hipError_t;
    pub fn hipMemcpyHtoD(
        dst: hipDeviceptr_t,
        src: *const c_void,
        byte_count: usize,
    ) -> hipError_t;
    pub fn hipMemcpyDtoH(
        dst: *mut c_void,
        src: hipDeviceptr_t,
        byte_count: usize,
    ) -> hipError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;

    pub fn hipEventCreate(p_event: *mut hipEvent_t) -> hipError_t;
    pub fn hipEventRecord(h_event: hipEvent_t, h_stream: hipStream_t) -> hipError_t;
    pub fn hipEventSynchronize(h_event: hipEvent_t) -> hipError_t;
    pub fn hipEventElapsedTime(
        p_milliseconds: *mut f32,
        h_start: hipEvent_t,
        h_end: hipEvent_t,
    ) -> hipError_t;
    pub fn hipEventDestroy(h_event: hipEvent_t) -> hipError_t;

    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, dev: hipDevice_t) -> hipError_t;
}

#[link(name = "hiprtc")]
unsafe extern "C" {
    pub fn hiprtcCreateProgram(
        prog: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> hiprtcResult;
    pub fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> hiprtcResult;
    pub fn hiprtcCompileProgram(
        prog: hiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> hiprtcResult;
    /// "code" here is the compiled AMDGPU code-object (ELF), the analog of
    /// NVRTC's PTX. `hipModuleLoadData` accepts it directly.
    pub fn hiprtcGetCodeSize(prog: hiprtcProgram, code_size_ret: *mut usize) -> hiprtcResult;
    pub fn hiprtcGetCode(prog: hiprtcProgram, code: *mut c_char) -> hiprtcResult;
    pub fn hiprtcGetProgramLogSize(prog: hiprtcProgram, log_size_ret: *mut usize) -> hiprtcResult;
    pub fn hiprtcGetProgramLog(prog: hiprtcProgram, log: *mut c_char) -> hiprtcResult;
    pub fn hiprtcGetErrorString(result: hiprtcResult) -> *const c_char;
}

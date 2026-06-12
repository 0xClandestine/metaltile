//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! CUDA runtime backend (CUDA_BACKEND_SPEC §4.1 / §4.4 — SCOPE Phase 1).
//!
//! `CudaDevice` is the NVIDIA analog of `MetalDevice`: it owns a CUDA
//! context and provides compile (NVRTC: CUDA C++ → PTX → module),
//! allocate, upload, launch, and read-back. Phase 1 is the smoke path —
//! enough to compile a generated elementwise kernel, run it on the GX10
//! (sm_121), and read results back for the CPU oracle. The `Device`-trait
//! abstraction + `Context` integration is Phase 6 (CLI/harness wiring);
//! this module stands on its own so the end-to-end pipeline is provable
//! now.
//!
//! Feature-gated (`cuda`) and Linux-targeted; the macOS Metal path is
//! untouched and stays the zero-config default.

mod ffi;

use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_void},
    ptr,
    sync::Mutex,
};

use ffi::*;
use metaltile_codegen::{CodegenBackend, CudaGenerator};
use metaltile_core::ir::Kernel;

use crate::error::MetalTileError;

/// Synthesize a Strided param's `_shape` or `_strides` (row-major) buffer
/// from the static shape, as little-endian u32s. Unknown dims default to 1.
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

fn cu_check(res: CUresult, what: &str) -> Result<(), MetalTileError> {
    if res == CUDA_SUCCESS {
        return Ok(());
    }
    let mut s: *const c_char = ptr::null();
    let msg = unsafe {
        if cuGetErrorString(res, &mut s) == CUDA_SUCCESS && !s.is_null() {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        } else {
            format!("CUDA error code {res}")
        }
    };
    Err(MetalTileError::Dispatch(format!("{what}: {msg}")))
}

/// Opt a function into a >48KB dynamic-shared-memory launch.
///
/// On Volta+ (sm_70+) `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`
/// raises the per-block dynamic-smem cap toward the device max. On pre-Volta
/// (sm_5x/6x, e.g. GTX 10-series Pascal) the cap is a hard 48KB and this
/// opt-in is rejected — historically that rejection was ignored and the
/// subsequent `cuLaunchKernel` surfaced a cryptic `invalid argument`. We now
/// check the return and fail *before* launch with a clear, typed reason.
fn ensure_dynamic_smem(func: CUfunction, shared_bytes: u32) -> Result<(), MetalTileError> {
    if shared_bytes == 0 {
        return Ok(());
    }
    let res = unsafe {
        cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_bytes as c_int,
        )
    };
    if res == CUDA_SUCCESS {
        return Ok(());
    }
    // Pre-Volta hard cap: dynamic shared memory cannot exceed 48KB and the
    // opt-in is unavailable, so this kernel cannot run on this arch.
    Err(MetalTileError::DeviceCapability(format!(
        "unsupported on this device: kernel needs {shared_bytes} bytes (>48KB) of dynamic shared memory, but this GPU architecture caps dynamic shared memory at 48KB (cuFuncSetAttribute opt-in rejected, code {res})"
    )))
}

fn nvrtc_check(res: nvrtcResult, what: &str) -> Result<(), MetalTileError> {
    if res == NVRTC_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let s = nvrtcGetErrorString(res);
        if s.is_null() {
            format!("nvrtc error code {res}")
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        }
    };
    Err(MetalTileError::Compilation(format!("{what}: {msg}")))
}

/// A device-side allocation. Frees on drop.
pub struct DeviceBuffer<'d> {
    ptr: CUdeviceptr,
    len: usize,
    _dev: &'d CudaDevice,
}

impl DeviceBuffer<'_> {
    pub fn device_ptr(&self) -> CUdeviceptr { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // Park back in the caching pool (raw cuMemFree when pool disabled).
            self._dev.free_raw_pooled(self.ptr, self.len);
        }
    }
}

/// A compiled, loaded CUDA module. Unloads on drop.
pub struct CudaModule {
    module: CUmodule,
}

impl CudaModule {
    /// Look up a kernel function by its `extern "C"` symbol name.
    pub fn function(&self, name: &str) -> Result<CudaFunction, MetalTileError> {
        let cname = CString::new(name).map_err(|e| MetalTileError::Dispatch(e.to_string()))?;
        let mut func: CUfunction = ptr::null_mut();
        cu_check(
            unsafe { cuModuleGetFunction(&mut func, self.module, cname.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(CudaFunction { func })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { cuModuleUnload(self.module) };
        }
    }
}

/// A handle to a `__global__` function inside a [`CudaModule`].
#[derive(Clone, Copy)]
pub struct CudaFunction {
    func: CUfunction,
}

/// One `cuLaunchKernel` kernel-param slot, in emitted-signature order:
/// either a pointer param (index into `Prepared::dev_ptrs`) or a by-value
/// scalar (index into `Prepared::scalars` — the driver copies the value
/// from the host pointer at launch).
enum ArgSlot {
    Buf(usize),
    Scalar(usize),
}

/// A compiled + resident kernel ready to launch repeatedly. See
/// [`CudaDevice::prepare`]. Holds its device resources alive; `args()`
/// hands out raw kernel-param pointers into `dev_ptrs`/`scalars`, so those
/// vecs must not be mutated after the first `args()` call.
struct Prepared<'d> {
    _module: CudaModule,
    func: CudaFunction,
    dev_bufs: Vec<DeviceBuffer<'d>>,
    dev_ptrs: Vec<CUdeviceptr>,
    scalars: Vec<Vec<u8>>,
    arg_slots: Vec<ArgSlot>,
    out_meta: Vec<Option<(String, usize)>>,
    shared_bytes: u32,
}

impl Prepared<'_> {
    /// Build the `cuLaunchKernel` param array in emitted-signature order,
    /// each entry a raw pointer into our stable `dev_ptrs`/`scalars`.
    fn args(&self) -> Vec<*mut c_void> {
        self.arg_slots
            .iter()
            .map(|slot| match slot {
                ArgSlot::Buf(i) => &self.dev_ptrs[*i] as *const CUdeviceptr as *mut c_void,
                ArgSlot::Scalar(i) => self.scalars[*i].as_ptr() as *mut c_void,
            })
            .collect()
    }
}

/// Soft cap on total bytes parked in the caching pool. Beyond this we stop
/// retaining freed buffers (and evict on demand) so the pool can't hoard VRAM.
/// The 4 GiB default gives a forward's transients plenty of headroom while
/// leaving model weights + KV resident on a unified-memory box (e.g. the
/// 120 GB GB10); `METALTILE_POOL_CAP_MB=N` overrides it per host/workload.
#[inline]
fn pool_cap_bytes() -> usize {
    use std::sync::OnceLock;
    static CAP: OnceLock<usize> = OnceLock::new();
    const DEFAULT: usize = 4 * 1024 * 1024 * 1024;
    *CAP.get_or_init(|| {
        std::env::var("METALTILE_POOL_CAP_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map_or(DEFAULT, |mb| mb * 1024 * 1024)
    })
}

/// Round a requested allocation length up to a pool BUCKET size. A forward's
/// transient outputs (per-layer/per-expert GEMM/attn buffers) have ever-varying
/// exact sizes, so an exact-size free-list almost never hits; rounding coalesces
/// "close enough" sizes into shared buckets so the cache actually reuses memory.
///
/// Scheme (mirrors PyTorch's CUDACachingAllocator block rounding):
/// - `<= 1 MiB`: round up to the next 512 B (small, tight — avoids huge waste).
/// - `> 1 MiB`:  round up to the next power-of-two-ish 1/8 step (≤ ~12% slack).
///
/// MUST be deterministic: `free` re-derives the same bucket from the requested
/// len that `alloc` did, so a buffer always returns to the bucket it came from.
#[inline]
fn size_bucket(len: usize) -> usize {
    debug_assert!(len > 0);
    const MIN_BLOCK: usize = 512;
    const SMALL_LIMIT: usize = 1024 * 1024; // 1 MiB
    if len <= SMALL_LIMIT {
        // Round up to a multiple of 512 B.
        (len + MIN_BLOCK - 1) & !(MIN_BLOCK - 1)
    } else {
        // Round up to the next 1/8-of-a-power-of-two step: take the leading
        // power of two, then snap to the next multiple of (pow2 / 8). Bounds
        // worst-case slack to ~12.5% while keeping the bucket count small.
        let pow2 = len.next_power_of_two();
        let step = (pow2 / 8).max(MIN_BLOCK);
        len.div_ceil(step) * step
    }
}

/// CUDA device + context. NVIDIA analog of `MetalDevice`.
pub struct CudaDevice {
    ctx: CUcontext,
    dev: CUdevice,
    cc_major: i32,
    cc_minor: i32,
    /// Caching device allocator (PyTorch-style). `cuMemAlloc`/`cuMemFree` are
    /// each a DEVICE-WIDE SYNC on the driver, so churning thousands per forward
    /// (one per transient GEMM/attn/Mamba output) serializes the whole pipeline
    /// and starves the compute kernels (nsys: ~2958 alloc/free/forward, GPU 34%
    /// busy). The pool keys a free-list by a ROUNDED size bucket (see
    /// [`size_bucket`]) so the ever-varying output shapes of a forward still
    /// hit the cache. On `free`, the (bucket-sized) buffer is returned to the
    /// list instead of `cuMemFree`d; on `alloc`, a same-bucket buffer is popped
    /// if present. All GPU work rides one ordered stream, so reusing a freed
    /// buffer for a later op is safe — stream order guarantees the prior kernel
    /// finished consuming it.
    ///
    /// Key = bucket size in bytes (NOT the requested len). `pooled_bytes`
    /// tracks the total parked so we can cap the pool ([`pool_cap_bytes`]) and
    /// evict beyond it rather than hoard VRAM. Default-on; set
    /// `METALTILE_POOL_ALLOC_OFF=1` ([`pool_enabled`]) for direct
    /// `cuMemAlloc`/`cuMemFree` (clean A/B).
    pool: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
    /// Total bytes currently parked in `pool` (sum of bucket_size × count).
    pooled_bytes: Mutex<usize>,
    /// Caching allocator active unless `METALTILE_POOL_ALLOC_OFF` is set.
    /// Cached once at create().
    pool_enabled: bool,
    /// Pinned host-staging buffers for async H2D: free-list by size, plus the
    /// in-flight list whose copies haven't been synced yet (reclaimed on the next
    /// `synchronize`/`download`). `usize` holds the host pointer.
    pinned_free: Mutex<HashMap<usize, Vec<usize>>>,
    pinned_inflight: Mutex<Vec<(usize, usize)>>,
    /// Dedicated non-blocking stream that ALL kernel launches + async H2D ride on.
    /// Replacing the null stream is what makes a whole decode token CUDA-graph
    /// CAPTURABLE (phase-1 of the megakernel: replay ~390 launches as one graph,
    /// eliminating per-launch host enqueue + inter-kernel bubbles → higher
    /// sustained bandwidth). `synchronize`/`dtoh` sync THIS stream.
    stream: CUstream,
    /// When true, launches/copies are being recorded into a CUDA graph (no real
    /// execution yet) — `dtoh`/`synchronize` must NOT be called mid-capture.
    capturing: std::sync::atomic::AtomicBool,
    /// Lazily-created cuBLAS handle (tensor-core GEMM escape hatch, Path A). The
    /// coop_tile MMA lowers to software emulation on CUDA (~0.1% of peak); this
    /// routes the prefill projection/MoE GEMMs through cuBLAS's tensor-core path
    /// (`cublasGemmEx`, bf16/f16 × f32-accumulate). `usize` holds the handle ptr
    /// (so the field is plain-Send); bound to `self.stream` on first use.
    cublas: Mutex<usize>,
    /// Lazily-created cublasLt handle + a fixed device workspace, used by the
    /// DETERMINISTIC GEMM path (`gemm_cublaslt`). cublasLt lets us forbid split-K
    /// reductions (REDUCTION_SCHEME_NONE), which the legacy `cublasGemmEx`
    /// heuristic / atomics-mode could not on sm_121. `(handle_ptr, workspace_ptr)`
    /// as `usize` to stay plain-Send; workspace is a single 32 MiB slab.
    cublaslt: Mutex<(usize, usize)>,
}

/// Fixed cublasLt workspace size (32 MiB). Large enough for the heuristic to
/// pick efficient non-split-K tensor-op kernels for the Nemotron prefill GEMMs.
const CUBLASLT_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

// SAFETY / THREADING CONTRACT: `cuCtxCreate` makes the context current on
// the CREATING thread only, so every driver call in this module assumes it
// runs on the thread that called [`CudaDevice::create`]. Calls from any
// other thread see no current context and fail with a clean
// CUDA_ERROR_INVALID_CONTEXT (interior state is Mutex-guarded, so there is
// no data race — just an error). Send/Sync are asserted so a higher layer
// can hold the device in an `Arc`/static to keep the context (and thus
// persistent device buffers) alive; all GPU submission must stay on the
// The device holds the refcounted PRIMARY context; `ensure_current` binds it
// to the calling thread (cuCtxSetCurrent, cached per thread) at every public
// entry point, so cross-thread use is sound.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

thread_local! {
    /// The ctx this thread last bound via `ensure_current` — skip the driver
    /// call when it's already current (the per-call cost would otherwise be
    /// paid ~400×/token on the decode path).
    static CURRENT_CTX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl CudaDevice {
    /// Bind this device's primary context to the calling thread (no-op when
    /// already bound). Must run at the top of every public entry point that
    /// touches the driver.
    fn ensure_current(&self) {
        CURRENT_CTX.with(|c| {
            if c.get() != self.ctx as usize {
                unsafe { cuCtxSetCurrent(self.ctx) };
                c.set(self.ctx as usize);
            }
        });
    }

    /// Initialize CUDA, grab device 0, create a context. Returns `Ok(None)`
    /// if no CUDA device is present (mirrors `MetalDevice::create`).
    pub fn create() -> Result<Option<Self>, MetalTileError> {
        unsafe {
            if cuInit(0) != CUDA_SUCCESS {
                return Ok(None);
            }
            let mut dev: CUdevice = 0;
            if cuDeviceGet(&mut dev, 0) != CUDA_SUCCESS {
                return Ok(None);
            }
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            cu_check(
                cuDeviceGetAttribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
                "cuDeviceGetAttribute(cc_major)",
            )?;
            cu_check(
                cuDeviceGetAttribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
                "cuDeviceGetAttribute(cc_minor)",
            )?;
            // Primary context (shared, refcounted) instead of cuCtxCreate:
            // a created context is current ONLY on the creating thread,
            // which made the Send/Sync impls a documented lie. The primary
            // ctx + a per-thread cuCtxSetCurrent guard (ensure_current)
            // makes cross-thread use sound.
            let mut ctx: CUcontext = ptr::null_mut();
            cu_check(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut stream: CUstream = ptr::null_mut();
            let sres = cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING);
            if sres != CUDA_SUCCESS {
                cuDevicePrimaryCtxRelease_v2(dev);
                cu_check(sres, "cuStreamCreate")?;
            }
            // Caching allocator DEFAULT-ON (alloc()/Drop route through it): nsys
            // showed raw cuMemAlloc/cuMemFree at 62% of CUDA API time. METALTILE_POOL_ALLOC_OFF=1
            // restores direct driver alloc/free (clean A/B). Pool cap defaults
            // to 4GB; METALTILE_POOL_CAP_MB overrides (pool_cap_bytes()).
            let pool_enabled = std::env::var("METALTILE_POOL_ALLOC_OFF").is_err();
            Ok(Some(CudaDevice {
                ctx,
                dev,
                cc_major: major,
                cc_minor: minor,
                pool: Mutex::new(HashMap::new()),
                pooled_bytes: Mutex::new(0),
                pool_enabled,
                pinned_free: Mutex::new(HashMap::new()),
                pinned_inflight: Mutex::new(Vec::new()),
                stream,
                capturing: std::sync::atomic::AtomicBool::new(false),
                cublas: Mutex::new(0),
                cublaslt: Mutex::new((0, 0)),
            }))
        }
    }

    /// Compute capability as `(major, minor)` — e.g. `(12, 1)` on GB10.
    pub fn compute_capability(&self) -> (i32, i32) { (self.cc_major, self.cc_minor) }

    /// Lazily create + cache the cuBLAS handle, bound to `self.stream` with the
    /// tensor-op math mode. Returns the raw handle ptr.
    fn cublas_handle(&self) -> Result<cublasHandle_t, MetalTileError> {
        let mut guard = self.cublas.lock().unwrap();
        if *guard == 0 {
            let mut h: cublasHandle_t = ptr::null_mut();
            let st = unsafe { cublasCreate_v2(&mut h) };
            if st != CUBLAS_STATUS_SUCCESS {
                return Err(MetalTileError::Dispatch(format!("cublasCreate failed: {st}")));
            }
            unsafe {
                cublasSetStream_v2(h, self.stream);
                cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
                // Determinism: forbid atomic-accumulation (split-K) kernels so
                // identical inputs give bit-exact identical outputs run-to-run.
                // The DEFAULT_TENSOR_OP heuristic otherwise selects split-K
                // variants for some MoE-prefill shapes that reduce via atomics
                // → logit jitter + argmax flips. Tensor cores stay active.
                // METALTILE_GEMM_ATOMICS=1 opts back into the nondeterministic
                // heuristic (for A/B perf measurement only).
                let atomics = std::env::var("METALTILE_GEMM_ATOMICS").ok().as_deref() == Some("1");
                cublasSetAtomicsMode(
                    h,
                    if atomics { CUBLAS_ATOMICS_ALLOWED } else { CUBLAS_ATOMICS_NOT_ALLOWED },
                );
            }
            // Persistent 32MB cuBLAS workspace → GEMMs are CUDA-graph capture/replay
            // safe (default workspace is a transient per-call alloc that the graph
            // captures by pointer then reads after free → "unspecified launch failure"
            // / null read on replay). Allocated once, never freed (device-lifetime).
            let ws_bytes = 32usize * 1024 * 1024;
            if let Ok(ws) = self.alloc_raw(ws_bytes) {
                unsafe {
                    cublasSetWorkspace_v2(h, ws as *mut std::ffi::c_void, ws_bytes);
                }
            }
            *guard = h as usize;
        }
        Ok(*guard as cublasHandle_t)
    }

    /// cuBLAS GEMM algorithm selector for the tensor-core GEMM paths.
    ///
    /// Determinism is enforced handle-wide via `cublasSetAtomicsMode(..NOT_ALLOWED)`
    /// (see `cublas_handle`), which forbids the split-K kernels that reduce via
    /// atomics. With atomics forbidden, the `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (99)
    /// heuristic is free to pick the best *deterministic* tensor-op kernel, so we
    /// keep DEFAULT here. `METALTILE_GEMM_ALGO=N` (0..=15) lets you pin a specific
    /// `CUBLAS_GEMM_ALGO{N}_TENSOR_OP` for experimentation.
    #[inline]
    fn gemm_algo() -> c_int {
        use std::sync::OnceLock;
        static ALGO: OnceLock<c_int> = OnceLock::new();
        *ALGO.get_or_init(|| {
            match std::env::var("METALTILE_GEMM_ALGO")
                .ok()
                .and_then(|s| s.trim().parse::<i32>().ok())
            {
                Some(n) if (0..=15).contains(&n) => CUBLAS_GEMM_ALGO0_TENSOR_OP + n as c_int,
                _ => CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            }
        })
    }

    /// Tensor-core GEMM via cuBLAS (Path A escape hatch). Computes the ROW-MAJOR
    /// product `C[m,n] = X[m,k] · W[n,k]ᵀ` — i.e. the standard projection
    /// `out[r,o] = Σ_k W[o,k]·x[r,k]` where weight is `[out=n, k]` and activation
    /// is `[rows=m, k]`. Inputs/output are device pointers of `dtype` (f16 or
    /// bf16); accumulation is f32 (CUBLAS_COMPUTE_32F) on the Tensor Cores.
    ///
    /// cuBLAS is column-major: a row-major `C[m,n]` is a col-major `[n,m]`. The
    /// identity `C = X·Wᵀ` (row-major) ⇒ in col-major call
    /// `op(A)=Wᵀ? ` — we issue `transa=T, transb=N, m=n, n=m, k=k`,
    /// `A=W (lda=k), B=X (ldb=k), C=out (ldc=n)`, which yields exactly C[m,n].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublas(
        &self,
        x: CUdeviceptr,   // [m, k] row-major activation
        w: CUdeviceptr,   // [n, k] row-major weight
        out: CUdeviceptr, // [m, n] row-major result
        m: usize,
        n: usize,
        k: usize,
        dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        // DETERMINISTIC by default: route through cublasLt with split-K reductions
        // forbidden (REDUCTION_SCHEME_NONE). The legacy cublasGemmEx heuristic
        // picks split-K atomic-accumulate kernels for some MoE-prefill shapes →
        // run-to-run logit jitter + argmax flips. `cublasSetAtomicsMode` and the
        // legacy algo selector do NOT suppress this on sm_121; cublasLt does.
        // METALTILE_GEMM_NONDET=1 opts back into the legacy path (A/B perf only).
        if std::env::var("METALTILE_GEMM_NONDET").ok().as_deref() != Some("1") {
            return self.gemm_cublaslt(x, w, out, m, n, k, dtype, dtype);
        }
        let h = self.cublas_handle()?;
        let cdt = match dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublas: unsupported dtype {other:?} (need f16/bf16)"
                ))),
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Col-major: out_cm[n,m] = Wᵀ_op · X. A=W (transposed), B=X (no trans).
        //   transa=T (A is [n,k] row-major = [k,n] col-major, opᵀ → [n,k])
        //   transb=N (B is [m,k] row-major = [k,m] col-major)
        //   result [n,m] col-major == [m,n] row-major (our C).
        let st = unsafe {
            cublasGemmEx(
                h,
                CUBLAS_OP_T, // op(A) = Aᵀ
                CUBLAS_OP_N, // op(B) = B
                n as c_int,  // rows of op(A) and C  (col-major)
                m as c_int,  // cols of op(B) and C
                k as c_int,  // shared dim
                &alpha as *const f32 as *const c_void,
                w,
                cdt,
                k as c_int, // A = W, lda = k
                x,
                cdt,
                k as c_int, // B = X, ldb = k
                &beta as *const f32 as *const c_void,
                out,
                cdt,
                n as c_int, // C = out, ldc = n
                CUBLAS_COMPUTE_32F,
                Self::gemm_algo(), // deterministic explicit algo (see gemm_algo)
            )
        };
        if st != CUBLAS_STATUS_SUCCESS {
            return Err(MetalTileError::Dispatch(format!(
                "cublasGemmEx failed: status {st} (m={m} n={n} k={k})"
            )));
        }
        Ok(())
    }

    /// Lazily create the cublasLt handle + fixed device workspace. Returns
    /// `(handle, workspace_ptr)`.
    fn cublaslt_ctx(&self) -> Result<(cublasLtHandle_t, CUdeviceptr), MetalTileError> {
        let mut guard = self.cublaslt.lock().unwrap();
        if guard.0 == 0 {
            let mut h: cublasLtHandle_t = ptr::null_mut();
            let st = unsafe { cublasLtCreate(&mut h) };
            if st != CUBLAS_STATUS_SUCCESS {
                return Err(MetalTileError::Dispatch(format!("cublasLtCreate failed: {st}")));
            }
            let ws = self.alloc_raw(CUBLASLT_WORKSPACE_BYTES)?;
            *guard = (h as usize, ws as usize);
        }
        Ok((guard.0 as cublasLtHandle_t, guard.1 as CUdeviceptr))
    }

    /// CUTLASS grouped MoE GEMM (AOT-linked, `cfg(have_cutlass)` via build.rs +
    /// CUTLASS_DIR). `a` = sorted-token f16 `[mt,K]`, `w` = contiguous f16 expert
    /// slab `[n_exp,N,K]`, `c` = out f16 `[mt,N]`. `group_rows[g]`/`expert_ids[g]`
    /// (HOST slices) describe each contiguous token group + its weight slab.
    /// `out[t,n] = Σ_k a[t,k]·w[eid][n,k]`. Beats cuBLAS per-expert on the skinny
    /// MoE shape. Errors if the runtime was built without CUTLASS.
    #[allow(clippy::too_many_arguments)] // mirrors the C entry point's signature
    pub fn moe_grouped_cutlass(
        &self,
        a: CUdeviceptr,
        w: CUdeviceptr,
        c: CUdeviceptr,
        group_rows: &[i32],
        expert_ids: &[i32],
        n: usize,
        k: usize,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass(
                    a: *const c_void,
                    w: *const c_void,
                    c: *mut c_void,
                    group_rows: *const c_int,
                    expert_ids: *const c_int,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                ) -> c_int;
            }
            if group_rows.len() != expert_ids.len() {
                return Err(MetalTileError::Dispatch(
                    "moe_grouped_cutlass: group_rows/expert_ids len mismatch".into(),
                ));
            }
            let r = unsafe {
                moe_grouped_gemm_cutlass(
                    a as *const c_void,
                    w as *const c_void,
                    c as *mut c_void,
                    group_rows.as_ptr(),
                    expert_ids.as_ptr(),
                    group_rows.len() as c_int,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(MetalTileError::Dispatch(format!(
                    "moe_grouped_gemm_cutlass failed: code {r}"
                )));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (a, w, c, group_rows, expert_ids, n, k);
            Err(MetalTileError::Dispatch(
                "moe_grouped_cutlass: runtime built without CUTLASS (set CUTLASS_DIR)".into(),
            ))
        }
    }

    /// f16/bf16 inputs, **f32 output** — convenience wrapper over [`gemm_cublaslt`]
    /// that keeps the A/B (weight/activation) dtype but writes the result as f32.
    /// cuBLAS already accumulates in f32 (`CUBLAS_COMPUTE_32F`); only the D-layout
    /// type changes, so this FUSES the post-GEMM `f16→f32` cast into the matmul at
    /// zero extra cost (no separate cast kernel, MFU preserved). Used by Nemotron
    /// prefill so the residual stream stays f32 without a trailing cast pass.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublas_f32out(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        out: CUdeviceptr,
        m: usize,
        n: usize,
        k: usize,
        ab_dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        self.gemm_cublaslt(x, w, out, m, n, k, ab_dtype, metaltile_core::DType::F32)
    }

    /// DETERMINISTIC tensor-core GEMM via cublasLt. Same row-major contract as
    /// [`gemm_cublas`] (`C[m,n] = X[m,k] · W[n,k]ᵀ`) but forbids split-K
    /// reductions (`REDUCTION_SCHEME_NONE`) so the heuristic only returns algos
    /// whose result is bit-exact reproducible run-to-run. This is the fix for the
    /// MoE-prefill logit jitter that `cublasSetAtomicsMode`/legacy-algo selection
    /// could not suppress on sm_121. Falls back to an error (caller can retry the
    /// nondeterministic path) if no deterministic algo is found.
    ///
    /// `out_dtype` is the D-layout type (f16/bf16 = same as inputs, or f32 to fuse
    /// the post-GEMM widening cast). Accumulation is always f32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublaslt(
        &self,
        x: CUdeviceptr,   // [m, k] row-major activation
        w: CUdeviceptr,   // [n, k] row-major weight
        out: CUdeviceptr, // [m, n] row-major result
        m: usize,
        n: usize,
        k: usize,
        dtype: metaltile_core::DType,
        out_dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        let (lt, workspace) = self.cublaslt_ctx()?;
        let cdt = match dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublaslt: unsupported dtype {other:?} (need f16/bf16)"
                ))),
        };
        let ddt = match out_dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            metaltile_core::DType::F32 => CUDA_R_32F,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublaslt: unsupported out_dtype {other:?}"
                ))),
        };
        // Col-major identity (same as gemm_cublas): out_cm[n,m] = (W^T)·X.
        //   op(A)=T, op(B)=N; A=W (rows=k, cols=n, ld=k), B=X (rows=k, cols=m, ld=k),
        //   D=out (rows=n, cols=m, ld=n).
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            let mut desc: cublasLtMatmulDesc_t = ptr::null_mut();
            let s = cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            if s != CUBLAS_STATUS_SUCCESS {
                return Err(MetalTileError::Dispatch(format!("cublasLtMatmulDescCreate: {s}")));
            }
            let opt = CUBLAS_OP_T;
            let opn = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                &opt as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &opn as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );

            // Layouts describe the PHYSICAL (untransposed) storage, col-major.
            // A=W is [n,k] row-major == [k,n] col-major → rows=k, cols=n, ld=k.
            // B=X is [m,k] row-major == [k,m] col-major → rows=k, cols=m, ld=k.
            // D=out is [m,n] row-major == [n,m] col-major → rows=n, cols=m, ld=n.
            let mut a_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut b_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut d_l: cublasLtMatrixLayout_t = ptr::null_mut();
            cublasLtMatrixLayoutCreate(&mut a_l, cdt, k as u64, n as u64, k as i64);
            cublasLtMatrixLayoutCreate(&mut b_l, cdt, k as u64, m as u64, k as i64);
            cublasLtMatrixLayoutCreate(&mut d_l, ddt, n as u64, m as u64, n as i64);

            let mut pref: cublasLtMatmulPreference_t = ptr::null_mut();
            cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_bytes: usize = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            );
            // THE determinism lever: restrict allowed reduction schemes to NONE,
            // so the heuristic never returns a split-K (atomic-accumulate) algo.
            let red_mask: u32 = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                &red_mask as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            );

            let mut result = cublasLtMatmulHeuristicResult_t::default();
            let mut returned: c_int = 0;
            let hs = cublasLtMatmulAlgoGetHeuristic(
                lt,
                desc,
                a_l,
                b_l,
                d_l,
                d_l,
                pref,
                1,
                &mut result,
                &mut returned,
            );
            if hs != CUBLAS_STATUS_SUCCESS || returned < 1 {
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(a_l);
                cublasLtMatrixLayoutDestroy(b_l);
                cublasLtMatrixLayoutDestroy(d_l);
                cublasLtMatmulDescDestroy(desc);
                return Err(MetalTileError::Dispatch(format!(
                    "cublasLt: no deterministic algo (m={m} n={n} k={k} status={hs} returned={returned})"
                )));
            }
            let mm = cublasLtMatmul(
                lt,
                desc,
                &alpha as *const f32 as *const c_void,
                w,
                a_l,
                x,
                b_l,
                &beta as *const f32 as *const c_void,
                out,
                d_l,
                out,
                d_l,
                result.algo.as_ptr(),
                workspace,
                ws_bytes,
                self.stream,
            );
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(a_l);
            cublasLtMatrixLayoutDestroy(b_l);
            cublasLtMatrixLayoutDestroy(d_l);
            cublasLtMatmulDescDestroy(desc);
            if mm != CUBLAS_STATUS_SUCCESS {
                return Err(MetalTileError::Dispatch(format!(
                    "cublasLtMatmul failed: status {mm} (m={m} n={n} k={k})"
                )));
            }
        }
        Ok(())
    }

    /// Strided-batched tensor-core GEMM via cuBLAS.
    /// Computes `batch_count` independent GEMMs in one call:
    ///   `C_i[m,n] = X_i[m,k] · W_i[n,k]^T`
    /// where each matrix starts at `x_base + i*stride_x`, `w_base + i*stride_w`,
    /// `out_base + i*stride_out` (all in BYTES, f16 inputs/output, f32 accumulate).
    /// Strides must be multiples of 2 (f16 element size).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublas_strided_batched(
        &self,
        x_base: CUdeviceptr,
        stride_x: i64,
        w_base: CUdeviceptr,
        stride_w: i64,
        out_base: CUdeviceptr,
        stride_out: i64,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        // DETERMINISTIC by default via cublasLt (split-K reductions forbidden);
        // see gemm_cublas. METALTILE_GEMM_NONDET=1 → legacy nondeterministic path.
        if std::env::var("METALTILE_GEMM_NONDET").ok().as_deref() != Some("1") {
            return self.gemm_cublaslt_strided_batched(
                x_base,
                stride_x,
                w_base,
                stride_w,
                out_base,
                stride_out,
                m,
                n,
                k,
                batch_count,
                dtype,
            );
        }
        let h = self.cublas_handle()?;
        let cdt = match dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublas_strided_batched: unsupported dtype {other:?}"
                ))),
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Col-major: same transpose logic as gemm_cublas (transa=T, transb=N, args reordered).
        // stride_x / stride_w / stride_out are in ELEMENTS (divide by 2 since f16).
        let el = 2i64; // bytes per f16 / bf16
        let st = unsafe {
            cublasGemmStridedBatchedEx(
                h,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const f32 as *const c_void,
                w_base,
                cdt,
                k as c_int,
                stride_w / el,
                x_base,
                cdt,
                k as c_int,
                stride_x / el,
                &beta as *const f32 as *const c_void,
                out_base,
                cdt,
                n as c_int,
                stride_out / el,
                batch_count as c_int,
                CUBLAS_COMPUTE_32F,
                Self::gemm_algo(), // deterministic explicit algo (see gemm_algo)
            )
        };
        if st != CUBLAS_STATUS_SUCCESS {
            return Err(MetalTileError::Dispatch(format!(
                "cublasGemmStridedBatchedEx failed: {st} (m={m} n={n} k={k} batch={batch_count})"
            )));
        }
        Ok(())
    }

    /// DETERMINISTIC strided-batched GEMM via cublasLt (split-K reductions
    /// forbidden). Same contract/strides (BYTES) as [`gemm_cublas_strided_batched`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublaslt_strided_batched(
        &self,
        x_base: CUdeviceptr,
        stride_x: i64,
        w_base: CUdeviceptr,
        stride_w: i64,
        out_base: CUdeviceptr,
        stride_out: i64,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        let (lt, workspace) = self.cublaslt_ctx()?;
        let cdt = match dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublaslt_strided_batched: unsupported dtype {other:?}"
                ))),
        };
        let el = 2i64; // bytes per f16/bf16 element
        let bc = batch_count as i32;
        // Strided-batch offsets are in ELEMENTS (same convention as ld).
        let off_a = stride_w / el; // A = W
        let off_b = stride_x / el; // B = X
        let off_d = stride_out / el; // D = out
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            let mut desc: cublasLtMatmulDesc_t = ptr::null_mut();
            let s = cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            if s != CUBLAS_STATUS_SUCCESS {
                return Err(MetalTileError::Dispatch(format!(
                    "cublasLtMatmulDescCreate(strided): {s}"
                )));
            }
            let opt = CUBLAS_OP_T;
            let opn = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                &opt as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &opn as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );

            // Same per-matrix layouts as gemm_cublaslt, plus batch count + stride.
            let mut a_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut b_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut d_l: cublasLtMatrixLayout_t = ptr::null_mut();
            cublasLtMatrixLayoutCreate(&mut a_l, cdt, k as u64, n as u64, k as i64);
            cublasLtMatrixLayoutCreate(&mut b_l, cdt, k as u64, m as u64, k as i64);
            cublasLtMatrixLayoutCreate(&mut d_l, cdt, n as u64, m as u64, n as i64);
            let set_batch = |layout: cublasLtMatrixLayout_t, stride: i64| {
                cublasLtMatrixLayoutSetAttribute(
                    layout,
                    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                    &bc as *const i32 as *const c_void,
                    std::mem::size_of::<i32>(),
                );
                cublasLtMatrixLayoutSetAttribute(
                    layout,
                    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &stride as *const i64 as *const c_void,
                    std::mem::size_of::<i64>(),
                );
            };
            set_batch(a_l, off_a);
            set_batch(b_l, off_b);
            set_batch(d_l, off_d);

            let mut pref: cublasLtMatmulPreference_t = ptr::null_mut();
            cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_bytes: usize = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            );
            let red_mask: u32 = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                &red_mask as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            );

            let mut result = cublasLtMatmulHeuristicResult_t::default();
            let mut returned: c_int = 0;
            let hs = cublasLtMatmulAlgoGetHeuristic(
                lt,
                desc,
                a_l,
                b_l,
                d_l,
                d_l,
                pref,
                1,
                &mut result,
                &mut returned,
            );
            if hs != CUBLAS_STATUS_SUCCESS || returned < 1 {
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(a_l);
                cublasLtMatrixLayoutDestroy(b_l);
                cublasLtMatrixLayoutDestroy(d_l);
                cublasLtMatmulDescDestroy(desc);
                return Err(MetalTileError::Dispatch(format!(
                    "cublasLt(strided): no deterministic algo (m={m} n={n} k={k} batch={batch_count} status={hs} returned={returned})"
                )));
            }
            let mm = cublasLtMatmul(
                lt,
                desc,
                &alpha as *const f32 as *const c_void,
                w_base,
                a_l,
                x_base,
                b_l,
                &beta as *const f32 as *const c_void,
                out_base,
                d_l,
                out_base,
                d_l,
                result.algo.as_ptr(),
                workspace,
                ws_bytes,
                self.stream,
            );
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(a_l);
            cublasLtMatrixLayoutDestroy(b_l);
            cublasLtMatrixLayoutDestroy(d_l);
            cublasLtMatmulDescDestroy(desc);
            if mm != CUBLAS_STATUS_SUCCESS {
                return Err(MetalTileError::Dispatch(format!(
                    "cublasLtMatmul(strided) failed: status {mm} (m={m} n={n} k={k} batch={batch_count})"
                )));
            }
        }
        Ok(())
    }

    /// Grouped-batched tensor-core GEMM (CUDA 13+). One cuBLAS call over
    /// `group_count` independent GEMMs, each with its own `(m_i, n, k)` and
    /// pointer pair `(x_i, w_i, out_i)`. Used for fused MoE prefill: each
    /// group is one active expert (variable number of tokens, fixed n/k).
    ///
    /// All A/B are `[m_i, k]` / `[n, k]` row-major f16; C is `[m_i, n]` f16.
    /// `x_ptrs[i]`, `w_ptrs[i]`, `out_ptrs[i]` are raw device pointers (u64)
    /// into already-allocated buffers. `m_per_group[i]` is the token count for
    /// group i; n and k are shared across all groups.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublas_grouped(
        &self,
        x_ptrs: &[u64],      // [group_count] device pointers for X
        w_ptrs: &[u64],      // [group_count] device pointers for W
        out_ptrs: &[u64],    // [group_count] device pointers for Out
        m_per_group: &[i32], // [group_count] rows per group (token count)
        n: usize,            // shared output dim
        k: usize,            // shared input dim
        dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        let group_count = x_ptrs.len();
        assert_eq!(w_ptrs.len(), group_count);
        assert_eq!(out_ptrs.len(), group_count);
        assert_eq!(m_per_group.len(), group_count);
        if group_count == 0 {
            return Ok(());
        }

        let h = self.cublas_handle()?;
        let cdt = match dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublas_grouped: unsupported dtype {other:?}"
                ))),
        };
        // cublasGemmGroupedBatchedEx: col-major same as GemmEx.
        // Row-major C[m_i,n] = X[m_i,k] · W[n,k]^T
        // Col-major: transa=T (A=W, lda=k), transb=N (B=X, ldb=k), m_cm=n, n_cm=m_i
        // Each "group" has group_size[i]=1 (one GEMM with its own m).
        let n_i = n as i32;
        let k_i = k as i32;
        // alpha/beta: the grouped API takes HOST ARRAYS with one entry per
        // group (unlike GemmEx's single scalar) — a lone scalar ref would be
        // read group_count entries deep (OOB host read for group_count > 1).
        let alphas: Vec<f32> = vec![1.0f32; group_count];
        let betas: Vec<f32> = vec![0.0f32; group_count];
        let transas: Vec<c_int> = vec![CUBLAS_OP_T; group_count];
        let transbs: Vec<c_int> = vec![CUBLAS_OP_N; group_count];
        // m_array[i] = n (cm rows = output cols), n_array[i] = m_per_group[i] (cm cols = token rows)
        let m_arr: Vec<c_int> = vec![n_i; group_count];
        let n_arr: Vec<c_int> = m_per_group.to_vec();
        let k_arr: Vec<c_int> = vec![k_i; group_count];
        let lda: Vec<c_int> = vec![k_i; group_count]; // A=W, lda=k
        let ldb: Vec<c_int> = vec![k_i; group_count]; // B=X, ldb=k
        let ldc: Vec<c_int> = vec![n_i; group_count]; // C=out, ldc=n
        let group_sizes: Vec<c_int> = vec![1i32; group_count];

        // Build void* pointer arrays (W, X, Out) for cuBLAS.
        // IMPORTANT: cublasGemmGroupedBatchedEx (like cublasGemmBatchedEx) requires
        // Aarray[], Barray[], Carray[] to be DEVICE-SIDE arrays of device pointers —
        // NOT host arrays. Pack all three into ONE contiguous host staging Vec and
        // upload with a SINGLE H2D copy into ONE pooled device buffer (was: 3×
        // cuMemAlloc + 3× cuMemcpyHtoD + 3× cuMemFree per call — each driver
        // alloc/free is a device-wide sync that serialized the stream per grouped
        // GEMM). `alloc_raw`/`free_raw_pooled` recycle the buffer across calls when
        // the pool is enabled (default; the raw driver alloc here otherwise
        // bypassed the caching allocator entirely).
        let ptr_bytes = group_count * 8; // each pointer is 8 bytes (u64)
        let triple_bytes = ptr_bytes * 3;
        // Layout: [A(W) | B(X) | C(Out)] contiguous.
        let mut staging: Vec<u64> = Vec::with_capacity(group_count * 3);
        staging.extend_from_slice(w_ptrs); // A = W device pointers
        staging.extend_from_slice(x_ptrs); // B = X device pointers
        staging.extend_from_slice(out_ptrs); // C = Out device pointers

        let base = self.alloc_raw(triple_bytes)?;
        let a_dev: CUdeviceptr = base;
        let b_dev: CUdeviceptr = base + ptr_bytes as CUdeviceptr;
        let c_dev: CUdeviceptr = base + (2 * ptr_bytes) as CUdeviceptr;
        let all_bytes =
            unsafe { std::slice::from_raw_parts(staging.as_ptr() as *const u8, triple_bytes) };
        // Single synchronous H2D: `staging` is pageable host memory, so a sync copy
        // guarantees it's fully consumed before this Vec drops (correctness over the
        // marginal async win for a ≤few-KB pointer array). cuBLAS reads a_dev/b_dev/
        // c_dev during the enqueued GEMM; the copy is complete before we enqueue.
        // H2D on self.stream (NOT null stream): orders this pooled-buffer write
        // after any prior GEMM's stream read of a recycled copy (see the same
        // fix + rationale in gemm_cublas_batched). Pageable src ⇒ host-sync copy,
        // so `staging` stays valid through the call.
        cu_check(
            unsafe {
                cuMemcpyHtoDAsync_v2(
                    a_dev,
                    all_bytes.as_ptr() as *const c_void,
                    triple_bytes,
                    self.stream,
                )
            },
            "cuMemcpyHtoDAsync(grouped_ptrs)",
        )?;

        let st = unsafe {
            cublasGemmGroupedBatchedEx(
                h,
                transas.as_ptr(),
                transbs.as_ptr(),
                m_arr.as_ptr(),
                n_arr.as_ptr(),
                k_arr.as_ptr(),
                alphas.as_ptr() as *const c_void,
                a_dev as *const *const c_void,
                cdt,
                lda.as_ptr(),
                b_dev as *const *const c_void,
                cdt,
                ldb.as_ptr(),
                betas.as_ptr() as *const c_void,
                c_dev as *mut *mut c_void,
                cdt,
                ldc.as_ptr(),
                group_count as c_int,
                group_sizes.as_ptr(),
                CUBLAS_COMPUTE_32F,
            )
        };
        // Return the single pointer-array buffer to the pool. The GEMM was
        // ENQUEUED on the ordered stream and reads a_dev/b_dev/c_dev there; any
        // later op that reuses this buffer also enqueues on the same stream
        // AFTER the GEMM, so stream order guarantees the read completes before a
        // reuse overwrites it (same invariant the caching allocator relies on).
        self.free_raw_pooled(base, triple_bytes);
        if st != CUBLAS_STATUS_SUCCESS {
            return Err(MetalTileError::Dispatch(format!(
                "cublasGemmGroupedBatchedEx failed: status {st} (groups={group_count} n={n} k={k})"
            )));
        }
        Ok(())
    }

    /// Pointer-array batched GEMM: `batch_count` independent GEMMs sharing
    /// (m,n,k), reading A/B/C from per-batch DEVICE pointers. Row-major
    /// `C_i[m,n] = X_i[m,k] · W_i[n,k]^T` (same contract/transpose as
    /// `gemm_cublas_strided_batched`, col-major transa=T/transb=N reorder).
    /// Unlike strided, each operand's per-batch pointer is arbitrary, so one
    /// operand can BROADCAST (multiple batches reuse the same slice) — the SSD
    /// per-group B/C fan-out. `x_ptrs`/`w_ptrs`/`out_ptrs` are device pointers
    /// (one per batch). Scales fine with large batch_count (unlike grouped).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublas_batched(
        &self,
        x_ptrs: &[u64],   // [batch] device ptr per X (rows operand)
        w_ptrs: &[u64],   // [batch] device ptr per W (weight operand)
        out_ptrs: &[u64], // [batch] device ptr per Out
        m: usize,
        n: usize,
        k: usize,
        dtype: metaltile_core::DType,
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        let batch_count = x_ptrs.len();
        assert_eq!(w_ptrs.len(), batch_count);
        assert_eq!(out_ptrs.len(), batch_count);
        if batch_count == 0 {
            return Ok(());
        }
        let h = self.cublas_handle()?;
        let cdt = match dtype {
            metaltile_core::DType::F16 => CUDA_R_16F,
            metaltile_core::DType::BF16 => CUDA_R_16BF,
            other =>
                return Err(MetalTileError::Dispatch(format!(
                    "gemm_cublas_batched: unsupported dtype {other:?}"
                ))),
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Col-major transpose reorder (same as gemm_cublas_strided_batched):
        // out_cm[n,m] = A(=W, transa=T, lda=k) · B(=X, transb=N, ldb=k); ldc=n.
        // cublasGemmBatchedEx requires Aarray/Barray/Carray to be DEVICE arrays
        // of device pointers — pack [A(W)|B(X)|C(Out)] into ONE host staging Vec,
        // one H2D into ONE pooled device buffer (same recipe as the grouped path).
        let ptr_bytes = batch_count * 8;
        let triple_bytes = ptr_bytes * 3;
        let mut staging: Vec<u64> = Vec::with_capacity(batch_count * 3);
        staging.extend_from_slice(w_ptrs); // A = W
        staging.extend_from_slice(x_ptrs); // B = X
        staging.extend_from_slice(out_ptrs); // C = Out
        let base = self.alloc_raw(triple_bytes)?;
        let a_dev: CUdeviceptr = base;
        let b_dev: CUdeviceptr = base + ptr_bytes as CUdeviceptr;
        let c_dev: CUdeviceptr = base + (2 * ptr_bytes) as CUdeviceptr;
        let all_bytes =
            unsafe { std::slice::from_raw_parts(staging.as_ptr() as *const u8, triple_bytes) };
        // H2D on self.stream (NOT the null stream): the pooled ptr-array buffer is
        // recycled across calls, and the PRIOR batched/grouped GEMM reading its
        // copy of the array runs on self.stream. A synchronous null-stream copy
        // here does NOT order against that pending stream read → it can overwrite
        // the array before the prior GEMM consumes it (observed: only the last
        // chunk of a 4-chunk SSD scan corrupted). Issuing the copy on self.stream
        // makes stream order serialize prior-read → this-write. (Pageable src ⇒
        // the async copy is host-synchronous, so `staging` stays valid.)
        cu_check(
            unsafe {
                cuMemcpyHtoDAsync_v2(
                    a_dev,
                    all_bytes.as_ptr() as *const c_void,
                    triple_bytes,
                    self.stream,
                )
            },
            "cuMemcpyHtoDAsync(batched_ptrs)",
        )?;
        let st = unsafe {
            cublasGemmBatchedEx(
                h,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const f32 as *const c_void,
                a_dev as *const *const c_void,
                cdt,
                k as c_int,
                b_dev as *const *const c_void,
                cdt,
                k as c_int,
                &beta as *const f32 as *const c_void,
                c_dev as *const *mut c_void,
                cdt,
                n as c_int,
                batch_count as c_int,
                CUBLAS_COMPUTE_32F,
                Self::gemm_algo(),
            )
        };
        self.free_raw_pooled(base, triple_bytes);
        if st != CUBLAS_STATUS_SUCCESS {
            return Err(MetalTileError::Dispatch(format!(
                "cublasGemmBatchedEx failed: status {st} (m={m} n={n} k={k} batch={batch_count})"
            )));
        }
        Ok(())
    }

    /// Compile CUDA C++ source → PTX → loaded module via NVRTC + the
    /// driver JIT. Targets the device's own virtual arch
    /// (`compute_<major><minor>`); the driver JITs PTX to the live GPU.
    pub fn compile(&self, src: &str, prog_name: &str) -> Result<CudaModule, MetalTileError> {
        self.ensure_current();
        let csrc = CString::new(src).map_err(|e| MetalTileError::Compilation(e.to_string()))?;
        let cname =
            CString::new(prog_name).map_err(|e| MetalTileError::Compilation(e.to_string()))?;

        let mut prog: nvrtcProgram = ptr::null_mut();
        nvrtc_check(
            unsafe {
                nvrtcCreateProgram(
                    &mut prog,
                    csrc.as_ptr(),
                    cname.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "nvrtcCreateProgram",
        )?;

        // Compile to the device's virtual architecture.
        let arch =
            CString::new(format!("--gpu-architecture=compute_{}{}", self.cc_major, self.cc_minor))
                .unwrap();
        // NVRTC does not auto-include the toolkit headers (cuda_fp16.h,
        // cuda_bf16.h) — point it at <toolkit>/include.
        let cuda_root = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let inc = CString::new(format!("--include-path={cuda_root}/include")).unwrap();
        // CCCL (CUDA C++ Core Libraries) include — needed for cooperative_groups.h
        // which includes <cuda/std/type_traits> on CUDA 12+. CCCL lives at either
        // {cuda_root}/include (older installs) or in the targets/<arch>/include/cccl
        // sub-directory on distro packages. We probe for the sub-dir and add it if found.
        // CCCL headers may live in a versioned targets sub-dir (distro CUDA packages
        // on CUDA 12+). Check common paths; fall back to just the main include dir.
        let cccl_path = {
            let fixed1 = format!("{cuda_root}/targets/sbsa-linux/include/cccl");
            let fixed2 = format!("{cuda_root}/targets/x86_64-linux/include/cccl");
            // Versioned TOOLKIT installs sit as `<root>-<toolkit-version>`
            // siblings of the unversioned root (e.g. /usr/local/cuda-13.0);
            // probe those, newest first. (The toolkit version is unrelated to
            // the device's compute capability — don't derive paths from it.)
            let by_toolkit = || -> Option<String> {
                let root = std::path::Path::new(&cuda_root);
                let stem = root.file_name()?.to_str()?.to_string();
                let mut versioned: Vec<std::path::PathBuf> = std::fs::read_dir(root.parent()?)
                    .ok()?
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with(&format!("{stem}-")))
                    })
                    .collect();
                versioned.sort();
                for dir in versioned.into_iter().rev() {
                    if let Ok(rd) = std::fs::read_dir(dir.join("targets")) {
                        for t in rd.flatten() {
                            let cccl = t.path().join("include/cccl");
                            if cccl.is_dir() {
                                return Some(cccl.display().to_string());
                            }
                        }
                    }
                }
                None
            };
            if std::path::Path::new(&fixed1).exists() {
                Some(fixed1)
            } else if std::path::Path::new(&fixed2).exists() {
                Some(fixed2)
            } else {
                by_toolkit()
            }
        };
        let cccl_inc =
            cccl_path.as_ref().map(|p| CString::new(format!("--include-path={p}")).unwrap());
        // Disable contraction of `a*b+c` into FMA by default: the CPU oracle
        // uses non-fused IEEE arithmetic, so default FMA fusion drifts in
        // accumulation-heavy kernels (conv, attention, recurrence). Matching
        // the oracle's rounding tightens bit-accuracy for the correctness
        // suite. For inference/perf runs FMA is strictly better (one fused op
        // per mul-add ⇒ half the FP issue + shorter dep chain in the hot GEMV/
        // gather/sdpa loops); opt in with `MT_FMAD=1`.
        // FMAD: default ON for inference (FMA fusion halves mul+add latency in
        // GEMV/SDPA dot-products). Opt out with MT_FMAD=0 for exact CPU-oracle
        // comparison. The correctness test (argmax 1234) passes with FMAD=true.
        let fmad_on = std::env::var("MT_FMAD").map(|v| v != "0" && v != "false").unwrap_or(true);
        let fmad = CString::new(if fmad_on { "--fmad=true" } else { "--fmad=false" }).unwrap();
        // MT_FAST_MATH=1: enable --use_fast_math (implies --fmad=true + fast
        // intrinsics: __expf, __sinf, etc.). Trades ~1-2 ULP precision for
        // ~2-4x faster transcendentals. Safe for inference softmax.
        let fast_math_on =
            std::env::var("MT_FAST_MATH").map(|v| v == "1" || v == "true").unwrap_or(false);
        let compile_res = match (fast_math_on, cccl_inc.as_ref()) {
            (true, Some(cccl)) => {
                let fast = CString::new("--use_fast_math").unwrap();
                let opts: [*const c_char; 4] =
                    [arch.as_ptr(), inc.as_ptr(), cccl.as_ptr(), fast.as_ptr()];
                unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) }
            },
            (false, Some(cccl)) => {
                let opts: [*const c_char; 4] =
                    [arch.as_ptr(), inc.as_ptr(), cccl.as_ptr(), fmad.as_ptr()];
                unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) }
            },
            (true, None) => {
                let fast = CString::new("--use_fast_math").unwrap();
                let opts: [*const c_char; 3] = [arch.as_ptr(), inc.as_ptr(), fast.as_ptr()];
                unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) }
            },
            (false, None) => {
                let opts: [*const c_char; 3] = [arch.as_ptr(), inc.as_ptr(), fmad.as_ptr()];
                unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) }
            },
        };

        // Always fetch the log — it carries the actual compiler diagnostics.
        let log = unsafe {
            let mut log_size: usize = 0;
            nvrtcGetProgramLogSize(prog, &mut log_size);
            if log_size > 1 {
                let mut buf = vec![0u8; log_size];
                nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char);
                String::from_utf8_lossy(&buf[..log_size.saturating_sub(1)]).into_owned()
            } else {
                String::new()
            }
        };

        if compile_res != NVRTC_SUCCESS {
            unsafe { nvrtcDestroyProgram(&mut prog) };
            return Err(MetalTileError::Compilation(format!(
                "nvrtcCompileProgram failed: {}\n--- log ---\n{log}",
                unsafe { CStr::from_ptr(nvrtcGetErrorString(compile_res)).to_string_lossy() }
            )));
        }

        // Fetch PTX.
        let ptx = unsafe {
            let mut ptx_size: usize = 0;
            nvrtc_check(nvrtcGetPTXSize(prog, &mut ptx_size), "nvrtcGetPTXSize")?;
            let mut buf = vec![0u8; ptx_size];
            nvrtc_check(nvrtcGetPTX(prog, buf.as_mut_ptr() as *mut c_char), "nvrtcGetPTX")?;
            buf
        };
        unsafe { nvrtcDestroyProgram(&mut prog) };

        // Load module (driver JITs PTX → cubin for the live arch).
        let mut module: CUmodule = ptr::null_mut();
        cu_check(
            unsafe { cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(CudaModule { module })
    }

    /// Allocate `len` bytes of device memory.
    pub fn alloc(&self, len: usize) -> Result<DeviceBuffer<'_>, MetalTileError> {
        self.ensure_current();
        if len == 0 {
            return Ok(DeviceBuffer { ptr: 0, len: 0, _dev: self });
        }
        // Route through the caching allocator (alloc_raw) so the main Tensor path
        // is pooled too — not just alloc_raw callers. Falls back to a raw cuMemAlloc
        // when the pool is disabled, so behavior is unchanged in that mode. nsys
        // showed cuMemAlloc/cuMemFree = 62% of CUDA API time (6110 driver allocs,
        // each ~0.5ms + device-synchronizing → the prefill GPU idle).
        let ptr = self.alloc_raw(len)?;
        Ok(DeviceBuffer { ptr, len, _dev: self })
    }

    /// Allocate + upload host bytes in one shot.
    pub fn upload(&self, data: &[u8]) -> Result<DeviceBuffer<'_>, MetalTileError> {
        self.ensure_current();
        let buf = self.alloc(data.len())?;
        if !data.is_empty() {
            // Enqueue on self.stream (NOT the null stream). Kernels ride self.stream
            // (CU_STREAM_NON_BLOCKING), which does NOT order against null-stream copies —
            // so a null-stream HtoD lets the driver run a dependent kernel BEFORE the copy
            // lands and silently DROP the kernel's 32-bit stores (the kernel no-ops; only
            // visible when it sparse-writes a pre-seeded buffer, e.g. partial_rope). Observed
            // on Pascal/WDDM; it is undefined cross-stream ordering on ANY arch (latent on
            // GB10). Sync after to keep the `data` borrow valid until the copy completes.
            cu_check(
                unsafe {
                    cuMemcpyHtoDAsync_v2(
                        buf.ptr,
                        data.as_ptr() as *const c_void,
                        data.len(),
                        self.stream,
                    )
                },
                "cuMemcpyHtoDAsync(upload)",
            )?;
            cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize(upload)")?;
        }
        Ok(buf)
    }

    /// Copy device memory back into a host buffer.
    pub fn download(&self, buf: &DeviceBuffer, out: &mut [u8]) -> Result<(), MetalTileError> {
        self.ensure_current();
        let n = out.len().min(buf.len);
        if n == 0 {
            return Ok(());
        }
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize(download)")?;
        cu_check(
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, buf.ptr, n) },
            "cuMemcpyDtoH",
        )?;
        self.reclaim_pinned();
        Ok(())
    }

    /// Allocate `len` bytes, returning the raw device pointer. The caller
    /// owns the allocation and must free it via [`free_raw`]. Unlike
    /// [`alloc`], this returns a plain pointer with no borrow of the device,
    /// so a higher layer (e.g. the ffai engine) can build persistent device
    /// tensors that outlive any single call — it just has to keep the
    /// `CudaDevice` (hence its context) alive while the pointers are live.
    ///
    /// [`free_raw`]: CudaDevice::free_raw
    pub fn alloc_raw(&self, len: usize) -> Result<CUdeviceptr, MetalTileError> {
        self.ensure_current();
        if len == 0 {
            return Ok(0);
        }
        if self.pool_enabled {
            // Caching path: serve from the bucket free-list if possible, else
            // allocate the FULL BUCKET size (so the buffer can be reused by any
            // later request that rounds to the same bucket). The caller only
            // touches `len` bytes; the extra slack is the rounding overhead.
            let bucket = size_bucket(len);
            if let Some(ptr) = self.pool.lock().unwrap().get_mut(&bucket).and_then(|v| v.pop()) {
                *self.pooled_bytes.lock().unwrap() -= bucket;
                return Ok(ptr);
            }
            let mut ptr: CUdeviceptr = 0;
            cu_check(unsafe { cuMemAlloc_v2(&mut ptr, bucket) }, "cuMemAlloc")?;
            return Ok(ptr);
        }
        // Pool off: direct driver alloc. The exact-size pop only ever serves
        // buffers parked mid-capture by `free_raw_pooled` (see there).
        if let Some(ptr) = self.pool.lock().unwrap().get_mut(&len).and_then(|v| v.pop()) {
            return Ok(ptr);
        }
        let mut ptr: CUdeviceptr = 0;
        cu_check(unsafe { cuMemAlloc_v2(&mut ptr, len) }, "cuMemAlloc")?;
        Ok(ptr)
    }

    /// Free a pointer returned by [`alloc_raw`]. No-op on a null pointer.
    /// Eagerly releases to the driver (use [`free_raw_pooled`] on the hot path).
    pub fn free_raw(&self, ptr: CUdeviceptr) {
        self.ensure_current();
        // Never cuMemFree while a CUDA graph is capturing (the graph captured this
        // pointer → replay would read freed memory). Retain it (leaks for the
        // transient capture; acceptable vs a corrupt graph).
        if ptr != 0 && !self.is_capturing() {
            unsafe { cuMemFree_v2(ptr) };
        }
    }

    /// Return a `len`-byte allocation to the size-bucketed pool for reuse
    /// instead of releasing it to the driver — avoids a synchronous
    /// `cuMemFree`/`cuMemAlloc` round-trip (each a device-wide sync) on the next
    /// same-bucket request. `len` MUST be the value passed to [`alloc_raw`] so
    /// the bucket re-derives to the one the buffer was allocated as. With the
    /// pool disabled (`METALTILE_POOL_ALLOC_OFF`) this releases to the driver
    /// directly (except mid-graph-capture, where the pointer must stay live).
    pub fn free_raw_pooled(&self, ptr: CUdeviceptr, len: usize) {
        self.ensure_current();
        if ptr == 0 {
            return;
        }
        if self.pool_enabled {
            let bucket = size_bucket(len);
            // Cap the pool: if parking this buffer would exceed the soft cap,
            // release it to the driver instead of hoarding VRAM. (One cuMemFree
            // under memory pressure is acceptable vs. unbounded growth.)
            let mut parked = self.pooled_bytes.lock().unwrap();
            // NEVER cuMemFree while a CUDA graph is capturing: the graph records
            // buffer pointers, so releasing one to the driver makes replay read
            // freed memory ("unspecified launch failure"). Retain past the cap for
            // the (transient) capture; the pool drains normally afterward.
            if *parked + bucket > pool_cap_bytes() && !self.is_capturing() {
                drop(parked);
                unsafe { cuMemFree_v2(ptr) };
                return;
            }
            *parked += bucket;
            drop(parked);
            self.pool.lock().unwrap().entry(bucket).or_default().push(ptr);
            return;
        }
        // Pool off: release to the driver — EXCEPT mid-capture (the graph
        // recorded this pointer; freeing it would make replay read freed
        // memory). Capture-parked buffers are reused by alloc_raw or freed
        // on Drop.
        if self.is_capturing() {
            self.pool.lock().unwrap().entry(len).or_default().push(ptr);
            return;
        }
        unsafe { cuMemFree_v2(ptr) };
    }

    /// Copy host bytes into an existing device allocation (host→device), ASYNC
    /// via a pinned staging buffer — enqueues on the (ordered) default stream
    /// without a host-blocking GPU drain. The pinned buffer is reclaimed on the
    /// next `synchronize`/`download` (by then the copy has completed).
    pub fn htod(&self, ptr: CUdeviceptr, bytes: &[u8]) -> Result<(), MetalTileError> {
        self.ensure_current();
        if bytes.is_empty() {
            return Ok(());
        }
        let len = bytes.len();
        // Large uploads skip the pinned staging path (pinning them would balloon
        // host pinned memory), but MUST still be enqueued on `self.stream` — NOT
        // the null stream. `self.stream` is CU_STREAM_NON_BLOCKING, so a null-
        // stream `cuMemcpyHtoD_v2` does NOT order against the kernels riding
        // `self.stream`: a later kernel that reads this buffer (or this very copy
        // landing in a pool-recycled buffer mid-GEMM) races the unordered copy →
        // run-to-run nondeterminism. `cuMemcpyHtoDAsync_v2` from pageable host
        // memory is still host-synchronous (CUDA falls back to a blocking copy,
        // so no pinning needed) but is correctly stream-ordered. Fixes the MoE
        // prefill FEWER_SYNCS logit jitter (per-expert >256KB activation uploads
        // interleaved with the on-stream expert GEMMs).
        if len > 262_144 {
            return cu_check(
                unsafe {
                    cuMemcpyHtoDAsync_v2(ptr, bytes.as_ptr() as *const c_void, len, self.stream)
                },
                "cuMemcpyHtoDAsync(large)",
            );
        }
        let pinned = match self.pinned_free.lock().unwrap().get_mut(&len).and_then(|v| v.pop()) {
            Some(p) => p,
            None => {
                let mut p: *mut c_void = ptr::null_mut();
                cu_check(unsafe { cuMemAllocHost_v2(&mut p, len) }, "cuMemAllocHost")?;
                p as usize
            },
        };
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), pinned as *mut u8, len) };
        cu_check(
            unsafe { cuMemcpyHtoDAsync_v2(ptr, pinned as *const c_void, len, self.stream) },
            "cuMemcpyHtoDAsync",
        )?;
        self.pinned_inflight.lock().unwrap().push((pinned, len));
        Ok(())
    }

    /// Reclaim in-flight pinned staging buffers after a sync point (their copies
    /// have completed) back to the free-list for reuse.
    fn reclaim_pinned(&self) {
        let mut inflight = self.pinned_inflight.lock().unwrap();
        if inflight.is_empty() {
            return;
        }
        let mut free = self.pinned_free.lock().unwrap();
        for (p, len) in inflight.drain(..) {
            free.entry(len).or_default().push(p);
        }
    }

    /// Copy device memory back to a host slice (device→host).
    pub fn dtoh(&self, ptr: CUdeviceptr, out: &mut [u8]) -> Result<(), MetalTileError> {
        self.ensure_current();
        if out.is_empty() {
            return Ok(());
        }
        // All kernels run on `self.stream` (non-blocking); a default-stream sync
        // copy would NOT order against it, so drain the stream first.
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize(dtoh)")?;
        cu_check(
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, ptr, out.len()) },
            "cuMemcpyDtoH",
        )?;
        self.reclaim_pinned();
        Ok(())
    }

    /// Launch `func` over a 1-D grid. `args` are raw pointers to each
    /// kernel argument value, in signature order (CUDA `kernelParams`).
    pub fn launch_1d(
        &self,
        func: CudaFunction,
        grid_blocks: u32,
        block_threads: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        self.launch(func, [grid_blocks, 1, 1], [block_threads, 1, 1], 0, args)
    }

    /// Launch `func` over a 3-D grid (blocks × threads-per-block) with
    /// `shared_bytes` of dynamic shared memory. For >48KB the function must
    /// opt in via `cuFuncSetAttribute` first (GB10 allows up to ~99KB/block).
    pub fn launch(
        &self,
        func: CudaFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        // Opt into the requested dynamic shared size (no-op below the default
        // cap, required above 48KB). Fails *before* launch on archs that cap
        // dynamic smem at 48KB (pre-Volta) instead of a cryptic launch error.
        ensure_dynamic_smem(func.func, shared_bytes)?;
        cu_check(
            unsafe {
                cuLaunchKernel(
                    func.func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        self.synchronize()
    }

    /// Like [`launch`] but does NOT synchronize after the launch — the kernel is
    /// enqueued async on the (ordered) default stream and the caller syncs only
    /// when it actually reads results back (`dtoh`/`synchronize`). The per-launch
    /// `cuCtxSynchronize` in `launch` serialized ~390 dispatches/decode-token and
    /// was the dominant decode overhead (~4ms/token of GPU-idle host stalls).
    pub fn launch_async(
        &self,
        func: CudaFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        ensure_dynamic_smem(func.func, shared_bytes)?;
        cu_check(
            unsafe {
                cuLaunchKernel(
                    func.func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// Cooperative kernel launch — uses `cuLaunchCooperativeKernel` so the
    /// kernel can call `cg::this_grid().sync()` for a global grid barrier.
    /// Required for two-phase fused kernels (e.g. MoE up+down in one launch).
    /// Note: cooperative launch is NOT capturable in a CUDA graph. Fallback to
    /// eager mode when CUDA-graph capture is active.
    pub fn launch_async_coop(
        &self,
        func: CudaFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        self.ensure_current();
        ensure_dynamic_smem(func.func, shared_bytes)?;
        cu_check(
            unsafe {
                cuLaunchCooperativeKernel(
                    func.func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    self.stream,
                    args.as_mut_ptr(),
                )
            },
            "cuLaunchCooperativeKernel",
        )
    }

    /// Returns `true` if a CUDA-graph capture is currently in progress on this
    /// device's stream. Callers that use non-capturable launches (e.g. cooperative
    /// kernel) must fall back to a capturable alternative during capture.
    pub fn is_capturing(&self) -> bool { self.capturing.load(std::sync::atomic::Ordering::SeqCst) }

    /// Begin recording all subsequent stream work into a CUDA graph (phase-1
    /// megakernel). Caller must run a NO-host-sync (all-device) sequence, then
    /// `end_capture`. THREAD_LOCAL mode scopes capture to this thread's stream.
    pub fn begin_capture(&self) -> Result<(), MetalTileError> {
        self.ensure_current();
        self.capturing.store(true, std::sync::atomic::Ordering::SeqCst);
        cu_check(
            unsafe { cuStreamBeginCapture_v2(self.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )
    }

    /// Finish capture → instantiate an executable graph. Replay with `graph_launch`.
    pub fn end_capture(&self) -> Result<CUgraphExec, MetalTileError> {
        self.ensure_current();
        let mut graph: CUgraph = ptr::null_mut();
        let res =
            cu_check(unsafe { cuStreamEndCapture(self.stream, &mut graph) }, "cuStreamEndCapture");
        // Clear the flag even on failure — a stuck `capturing` would make
        // every later free_raw/free_raw_pooled leak "for the capture" forever.
        self.capturing.store(false, std::sync::atomic::Ordering::SeqCst);
        res?;
        let mut exec: CUgraphExec = ptr::null_mut();
        let inst = cu_check(
            unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) },
            "cuGraphInstantiate",
        );
        unsafe { cuGraphDestroy(graph) };
        inst?;
        Ok(exec)
    }

    /// Replay a captured decode token: ONE host launch replaces ~390 — no
    /// per-kernel enqueue, no inter-kernel host bubbles. Syncs the stream after.
    ///
    /// `exec` must be a live handle returned by [`Self::end_capture`].
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // opaque driver handle, not dereferenced host-side
    pub fn graph_launch(&self, exec: CUgraphExec) -> Result<(), MetalTileError> {
        self.ensure_current();
        cu_check(unsafe { cuGraphLaunch(exec, self.stream) }, "cuGraphLaunch")?;
        self.synchronize()
    }

    /// Issue `n` sequential graph launches WITHOUT syncing between them, then
    /// sync once at the end. The GPU stream is FIFO-ordered so graphs execute
    /// sequentially despite the async enqueues — no data race on intermediate
    /// buffers. This eliminates the per-token host-GPU handoff overhead that
    /// `graph_launch` (sync-per-token) incurs, giving the maximum throughput
    /// for the captured graph. Use only for throughput benchmarking — state
    /// (KV cache, SSM state) is overwritten sequentially and not meaningful.
    /// `exec` must be a live handle returned by [`Self::end_capture`].
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // opaque driver handle, not dereferenced host-side
    pub fn graph_launch_batch(&self, exec: CUgraphExec, n: usize) -> Result<(), MetalTileError> {
        self.ensure_current();
        for _ in 0..n {
            cu_check(unsafe { cuGraphLaunch(exec, self.stream) }, "cuGraphLaunch(batch)")?;
        }
        self.synchronize()
    }

    /// Dispatch-path upload with ZEROED SLACK after the payload. Grid3D
    /// kernels carry no bounds guard (Metal parity), so a ceil-div launch
    /// runs tail threads whose loads/stores walk past a buffer's logical
    /// end. The Metal harness survives that because MTLBuffers are
    /// page-granular and zero-filled; a recycled CUDA pool bucket instead
    /// hands those threads garbage (e.g. a gather kernel loading a garbage
    /// row index → wild address → sticky illegal-memory-access poisoning
    /// the whole context). Mirror the Metal environment: over-allocate and
    /// zero-fill so tail threads read benign zeros and their stores land in
    /// owned slack. Dispatch/test path only — the engine's `alloc_raw`/
    /// `htod` hot path is unaffected.
    fn upload_padded(&self, data: &[u8]) -> Result<DeviceBuffer<'_>, MetalTileError> {
        const DISPATCH_PAD_BYTES: usize = 4096;
        let buf = self.alloc(data.len() + DISPATCH_PAD_BYTES)?;
        cu_check(
            unsafe { cuMemsetD8Async(buf.ptr, 0, buf.len, self.stream) },
            "cuMemsetD8Async(dispatch pad)",
        )?;
        if !data.is_empty() {
            cu_check(
                unsafe {
                    cuMemcpyHtoDAsync_v2(
                        buf.ptr,
                        data.as_ptr() as *const c_void,
                        data.len(),
                        self.stream,
                    )
                },
                "cuMemcpyHtoDAsync(upload_padded)",
            )?;
        }
        // Sync so the pageable `data` borrow can end (same contract as upload).
        cu_check(
            unsafe { cuStreamSynchronize(self.stream) },
            "cuStreamSynchronize(upload_padded)",
        )?;
        Ok(buf)
    }

    /// A compiled+resident kernel: module, function, device buffers, and the
    /// marshalled scalar/pointer args, ready to launch repeatedly without
    /// re-compiling or re-uploading. Produced by [`CudaDevice::prepare`] and
    /// consumed by both [`CudaDevice::run_kernel`] (one launch + read-back)
    /// and [`CudaDevice::bench_kernel`] (timed launch loop).
    ///
    /// `dev_bufs` and the module own their device resources (freed on drop);
    /// `dev_ptrs` and `scalars` back the raw arg pointers, so neither vec is
    /// reallocated after [`Prepared::args`] hands out pointers into them.
    fn prepare<'d>(
        &'d self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        block: [u32; 3],
    ) -> Result<Prepared<'d>, MetalTileError> {
        // 1. IR → CUDA C++ → module.
        let cg = CudaGenerator::new();
        let src = cg.generate(kernel).map_err(MetalTileError::Codegen)?;
        // MT_DUMP_CUDA_SRC=<path>: write generated CUDA C++ for every kernel.
        if let Ok(dir) = std::env::var("MT_DUMP_CUDA_SRC") {
            let path = format!("{}/{}.cu", dir, kernel.name);
            let _ = std::fs::write(&path, &src);
        }
        let module = self.compile(&src, &format!("{}.cu", kernel.name))?;
        let func = module.function(&kernel.name)?;
        // Dynamic shared-memory size for this launch geometry (per-warp
        // arrays scale with the TOTAL thread count, x·y·z).
        let shared_bytes = cg.shared_bytes(kernel, block[0] * block[1] * block[2]) as u32;

        // 2. Marshal each param (in kernel.params order) into arg_slots.
        //    Tensor/Strided params upload to a device buffer; Scalar params
        //    stay HOST bytes passed by value through kernelParams — the
        //    emitted signature takes them by value (`float x`), so a device
        //    pointer here would be reinterpreted as the scalar's bytes.
        // dev_bufs / dev_ptrs / out_meta are kept index-aligned. Strided
        // params add 2 extra companion buffers, so an output's buffer is NOT
        // at its kernel.params index — out_meta carries the (name,len) for
        // read-back so the alignment is by buffer, not by param.
        let mut dev_bufs: Vec<DeviceBuffer> = Vec::new();
        let mut dev_ptrs: Vec<CUdeviceptr> = Vec::new();
        let mut scalars: Vec<Vec<u8>> = Vec::new();
        let mut arg_slots: Vec<ArgSlot> = Vec::new();
        let mut out_meta: Vec<Option<(String, usize)>> = Vec::new();
        for p in &kernel.params {
            let bytes = buffers.get(&p.name).ok_or_else(|| {
                MetalTileError::Dispatch(format!("missing buffer for param '{}'", p.name))
            })?;
            if p.kind == metaltile_core::ir::ParamKind::Scalar {
                scalars.push(bytes.clone());
                arg_slots.push(ArgSlot::Scalar(scalars.len() - 1));
                continue;
            }
            let buf = self.upload_padded(bytes)?;
            dev_ptrs.push(buf.device_ptr());
            arg_slots.push(ArgSlot::Buf(dev_ptrs.len() - 1));
            out_meta.push(if p.is_output { Some((p.name.clone(), bytes.len())) } else { None });
            dev_bufs.push(buf);

            // Strided params carry two companion buffers (shape, strides) in
            // signature order. The harness provides them by name; if absent,
            // synthesize a row-major layout from the static shape.
            if p.kind == metaltile_core::ir::ParamKind::Strided {
                for suffix in ["_shape", "_strides"] {
                    let key = format!("{}{}", p.name, suffix);
                    let meta = match buffers.get(&key) {
                        Some(b) => b.clone(),
                        None => synth_strided_meta(&p.shape, suffix == "_strides"),
                    };
                    let mb = self.upload_padded(&meta)?;
                    dev_ptrs.push(mb.device_ptr());
                    arg_slots.push(ArgSlot::Buf(dev_ptrs.len() - 1));
                    out_meta.push(None);
                    dev_bufs.push(mb);
                }
            }
        }

        // 3. Trailing by-value args: constexprs (signature order) then the
        //    synthetic _n_elems for Elementwise.
        for ce in &kernel.constexprs {
            let name = ce.name.name();
            // No binding = the constexpr is dead in this variant (e.g. a
            // shared signature whose pruned body never reads it) — bind a
            // zero of the declared width, mirroring the Metal dispatch path
            // which binds a dummy buffer instead of erroring.
            let bytes = match buffers.get(name) {
                Some(b) => b.clone(),
                None => vec![0u8; ce.dtype.size_bytes().max(4)],
            };
            scalars.push(bytes);
            arg_slots.push(ArgSlot::Scalar(scalars.len() - 1));
        }
        if kernel.mode == metaltile_core::ir::KernelMode::Elementwise {
            // Bounds = element count of the first output param.
            let n_elems = kernel
                .params
                .iter()
                .position(|p| p.is_output)
                .and_then(|i| {
                    let p = &kernel.params[i];
                    buffers.get(&p.name).map(|b| (b.len() / p.dtype.size_bytes().max(1)) as u32)
                })
                .unwrap_or(0);
            scalars.push(n_elems.to_le_bytes().to_vec());
            arg_slots.push(ArgSlot::Scalar(scalars.len() - 1));
        }

        Ok(Prepared {
            _module: module,
            func,
            dev_bufs,
            dev_ptrs,
            scalars,
            arg_slots,
            out_meta,
            shared_bytes,
        })
    }

    /// End-to-end generic dispatch: generate CUDA for `kernel`, compile,
    /// allocate + upload every param (by name from `buffers`), pack kernel
    /// args in signature order, launch over (`grid`×`block`), and read back
    /// the output params. The CUDA analog of `Context::dispatch_with_grid`
    /// plus `SingleDispatch`, used to run the registered kernel-test corpus
    /// on CUDA. `buffers` must contain every param's bytes (inputs AND
    /// pre-sized outputs) plus each constexpr's name→LE-bytes.
    pub fn run_kernel(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<BTreeMap<String, Vec<u8>>, MetalTileError> {
        self.ensure_current();
        let prep = self.prepare(kernel, buffers, block)?;

        // Launch (with dynamic shared memory).
        let mut args = prep.args();
        self.launch(prep.func, grid, block, prep.shared_bytes, &mut args)?;
        drop(args);

        // Read back outputs (aligned to dev_bufs, not kernel.params).
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

    /// Time `kernel` on the GPU: compile + upload once, run `warmup`
    /// untimed launches (NVRTC/JIT, caches, clocks settle), then `iters`
    /// launches each bracketed by CUDA events. Returns the per-iter GPU
    /// elapsed times in **microseconds** — feed to `BenchStats::from_samples`
    /// for min/median/p95. Event timing measures device wall-clock only, so
    /// host scheduling jitter does not pollute the samples.
    ///
    /// No read-back: throughput is data-independent, and skipping the DtoH
    /// copy keeps the timed region pure kernel execution. Outputs stay
    /// resident (overwritten each iter — fine, we only time).
    pub fn bench_kernel(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        grid: [u32; 3],
        block: [u32; 3],
        warmup: u32,
        iters: u32,
    ) -> Result<Vec<f64>, MetalTileError> {
        self.ensure_current();
        let prep = self.prepare(kernel, buffers, block)?;
        let mut args = prep.args();

        // Warmup — first launch pays JIT/cache/clock-ramp costs.
        for _ in 0..warmup {
            self.launch(prep.func, grid, block, prep.shared_bytes, &mut args)?;
        }

        // Two events bracket each timed launch.
        let (mut start, mut stop): (CUevent, CUevent) = (ptr::null_mut(), ptr::null_mut());
        cu_check(unsafe { cuEventCreate(&mut start, CU_EVENT_DEFAULT) }, "cuEventCreate(start)")?;
        cu_check(unsafe { cuEventCreate(&mut stop, CU_EVENT_DEFAULT) }, "cuEventCreate(stop)")?;

        // Closure owns its sample vec (returned on success) so no outer
        // borrow outlives it — lets us unconditionally destroy events after.
        let mut timed = || -> Result<Vec<f64>, MetalTileError> {
            let mut samples = Vec::with_capacity(iters as usize);
            for _ in 0..iters {
                ensure_dynamic_smem(prep.func.func, prep.shared_bytes)?;
                // Events + launch all ride self.stream — recording on the null
                // stream would not order against the non-blocking work stream.
                cu_check(unsafe { cuEventRecord(start, self.stream) }, "cuEventRecord(start)")?;
                cu_check(
                    unsafe {
                        cuLaunchKernel(
                            prep.func.func,
                            grid[0],
                            grid[1],
                            grid[2],
                            block[0],
                            block[1],
                            block[2],
                            prep.shared_bytes,
                            self.stream,
                            args.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    },
                    "cuLaunchKernel",
                )?;
                cu_check(unsafe { cuEventRecord(stop, self.stream) }, "cuEventRecord(stop)")?;
                cu_check(unsafe { cuEventSynchronize(stop) }, "cuEventSynchronize")?;
                let mut ms: f32 = 0.0;
                cu_check(
                    unsafe { cuEventElapsedTime(&mut ms, start, stop) },
                    "cuEventElapsedTime",
                )?;
                samples.push(ms as f64 * 1000.0); // ms → µs
            }
            Ok(samples)
        };
        let res = timed();

        // Always destroy events, even on error.
        unsafe {
            cuEventDestroy_v2(start);
            cuEventDestroy_v2(stop);
        }
        res
    }

    pub fn synchronize(&self) -> Result<(), MetalTileError> {
        self.ensure_current();
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize")?;
        self.reclaim_pinned();
        Ok(())
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // Drain the stream before anything is freed — an async H2D copy
        // still in flight would otherwise DMA from pinned host memory we
        // are about to cuMemFreeHost.
        unsafe { cuStreamSynchronize(self.stream) };
        // Library handles + pinned host memory first: cuCtxDestroy does not
        // release host pinned allocations, and destroying handles after their
        // context is gone is undefined.
        if let Ok(mut h) = self.cublas.lock()
            && *h != 0
        {
            unsafe { cublasDestroy_v2(*h as cublasHandle_t) };
            *h = 0;
        }
        if let Ok(mut lt) = self.cublaslt.lock()
            && lt.0 != 0
        {
            unsafe {
                cublasLtDestroy(lt.0 as cublasLtHandle_t);
                if lt.1 != 0 {
                    cuMemFree_v2(lt.1 as CUdeviceptr);
                }
            }
            *lt = (0, 0);
        }
        if let Ok(mut free) = self.pinned_free.lock() {
            for (_, ptrs) in free.drain() {
                for p in ptrs {
                    unsafe { cuMemFreeHost(p as *mut c_void) };
                }
            }
        }
        if let Ok(mut inflight) = self.pinned_inflight.lock() {
            for (p, _) in inflight.drain(..) {
                unsafe { cuMemFreeHost(p as *mut c_void) };
            }
        }
        if let Ok(mut pool) = self.pool.lock() {
            for (_, ptrs) in pool.drain() {
                for p in ptrs {
                    unsafe { cuMemFree_v2(p) };
                }
            }
        }
        if let Ok(mut parked) = self.pooled_bytes.lock() {
            *parked = 0;
        }
        if !self.stream.is_null() {
            unsafe { cuStreamDestroy_v2(self.stream) };
        }
        if !self.ctx.is_null() {
            // Primary context: release the refcount taken in create() —
            // destroying it would tear the context down under any other
            // holder in the process.
            unsafe { cuDevicePrimaryCtxRelease_v2(self.dev) };
        }
    }
}

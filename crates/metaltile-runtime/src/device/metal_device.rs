//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Metal device adapter.
//!
//! Owns the Metal device, command queue, buffer pool, and both caches
//! (PSO + MSL).  A single `MetalDevice` replaces the three pairs of
//! `static OnceLock<…>` that were previously duplicated inside
//! `dispatch_metal` and `dispatch_chain_metal`.
//!
//! The type is `pub(crate)` — public API consumers go through
//! [`super::Context`].

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandQueue,
    MTLComputePipelineState,
    MTLCreateSystemDefaultDevice,
    MTLDevice,
    MTLResourceOptions,
};

use crate::{
    cache::{msl_cache::MslCache, pso_cache::PsoCache},
    device::buffer_pool::{BufRc, BufferPool},
    error::MetalTileError,
};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Metal device protocol object.
pub(crate) type Dev = ProtocolObject<dyn MTLDevice>;
/// Compute pipeline state.
pub(crate) type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
/// Command queue.
pub(crate) type Queue = ProtocolObject<dyn MTLCommandQueue>;

// ---------------------------------------------------------------------------
// MetalDevice
// ---------------------------------------------------------------------------

/// A connected Metal GPU, its command queue, and supporting caches.
///
/// # Lifetime
///
/// Creating a `MetalDevice` probes the default system device.  If no
/// Metal‑capable GPU is found, [`MetalDevice::create`] returns
/// `Err(MetalTileError::NoDevice)`.
///
/// `MetalDevice` is **not** `Send` because the `objc2` types it holds
/// are `!Send`.  Share it within one thread or wrap in an `Arc<Mutex<…>>`
/// when cross‑thread access is needed.
pub(crate) struct MetalDevice {
    /// The Metal GPU device (e.g. `"Apple M4 Max"`).
    device: Retained<Dev>,
    /// Persistent command queue.  Apple's Best Practices Guide flags
    /// `newCommandQueue` as expensive ("should not be repeatedly
    /// created and destroyed"); we create one at startup and reuse it.
    queue: Retained<Queue>,
    /// Thread‑safe compute pipeline‑state cache.
    pso_cache: PsoCache,
    /// Thread‑safe MSL source‑generation cache.
    msl_cache: MslCache,
    /// Thread‑local buffer pool.
    buffer_pool: BufferPool,
}

impl MetalDevice {
    /// Probe the default Metal device and initialize all subsystems.
    ///
    /// Returns `Ok(None)` on non‑macOS platforms (no Metal available).
    /// Returns `Err(…)` on macOS when no GPU is found.
    #[cfg(target_os = "macos")]
    pub fn create() -> Result<Option<Self>, MetalTileError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalTileError::DeviceCreation)?;

        let queue = device.newCommandQueue().ok_or(MetalTileError::QueueCreation)?;

        tracing::debug!("metal device created");

        Ok(Some(MetalDevice {
            device,
            queue,
            pso_cache: PsoCache::new(),
            msl_cache: MslCache::new(),
            buffer_pool: BufferPool::new(),
        }))
    }

    /// Stub for non‑macOS platforms.
    #[cfg(not(target_os = "macos"))]
    pub fn create() -> Result<Option<Self>, MetalTileError> { Ok(None) }

    // ── accessors ──────────────────────────────────────────────────

    /// Borrow the Metal device.
    #[allow(dead_code)]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> { &self.device }

    /// Borrow the command queue.
    #[allow(dead_code)]
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> { &self.queue }

    /// Convenience: get a command buffer from the queue.
    pub fn command_buffer(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>, MetalTileError> {
        self.queue.commandBuffer().ok_or(MetalTileError::NoDevice)
    }

    // ── pipeline compilation ───────────────────────────────────────

    /// Get or compile a compute pipeline state.
    ///
    /// `key` is produced by [`pso_cache_key`](crate::pso_cache::pso_cache_key).
    ///
    /// MSL generation is deferred — the closure is only invoked on PSO
    /// cache miss. In steady-state benching the cache is always warm,
    /// so the 5-12 KB MSL string never gets materialised at all.
    pub fn get_pso(
        &self,
        key: u64,
        kernel: &metaltile_core::ir::Kernel,
        kernel_name: &str,
        fn_consts: &std::collections::BTreeMap<String, u32>,
    ) -> Result<Pso, MetalTileError> {
        self.pso_cache.get_or_compile(
            &self.device,
            key,
            || self.msl_cache.get_or_generate(kernel, key),
            kernel_name,
            fn_consts,
        )
    }

    /// Get or generate MSL source for a kernel.
    ///
    /// `key` is produced by [`pso_cache_key`](crate::pso_cache::pso_cache_key).
    /// Most callers should prefer [`Self::get_pso`] which materialises
    /// the MSL lazily; this entry-point stays for snapshot tests and
    /// the MSL-cache perf bench.
    #[allow(dead_code)]
    pub fn get_msl(
        &self,
        kernel: &metaltile_core::ir::Kernel,
        key: u64,
    ) -> Result<String, MetalTileError> {
        self.msl_cache.get_or_generate(kernel, key)
    }

    // ── buffer pool ────────────────────────────────────────────────

    /// Acquire a buffer from the pool (shared storage, with copy).
    pub fn acquire_shared(
        &self,
        bytes: Option<&[u8]>,
        len: usize,
    ) -> Result<BufRc, MetalTileError> {
        use objc2_metal::MTLBuffer as _;

        let opts =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        let buf = self.buffer_pool.acquire(&self.device, len, opts)?;

        if let Some(data) = bytes.filter(|b| !b.is_empty()) {
            if data.len() < len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer expected {len} bytes, got {}",
                    data.len()
                )));
            }
            // SAFETY: `buf.contents()` returns a valid `NonNull<c_void>`
            // for the lifetime of the buffer.  We checked `data.len() ≥ len`
            // above, so the memcpy is within bounds.
            let dst = buf.contents();
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst.as_ptr() as *mut u8, data.len());
            }
        }

        Ok(buf)
    }

    /// Acquire a buffer from the pool (private GPU storage, no copy).
    ///
    /// Private‑storage buffers are **not** host‑readable.  Use this
    /// only for intermediate results that stay on the GPU across
    /// chained passes.
    pub fn acquire_private(&self, len: usize) -> Result<BufRc, MetalTileError> {
        let opts = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        self.buffer_pool.acquire(&self.device, len, opts)
    }
}

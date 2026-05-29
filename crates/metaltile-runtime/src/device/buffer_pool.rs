//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Thread-local Metal buffer pool.
//!
//! Bucketed by `(size, storage_mode)` so allocations of the same class
//! are recycled.  A buffer is returned to the caller only when the pool
//! holds the sole `Rc` reference; otherwise a fresh allocation is made.
//! Thread‑local because `Retained<MTLBuffer>` is `!Send`.
//!
//! # Example
//!
//! ```ignore
//! let buf: BufRc = pool.acquire(dev, 1024, MTLResourceOptions::StorageModeShared)?;
//! // ... dispatch work that keeps a clone alive ...
//! // When the last clone drops, the Rc returns to the pool.
//! ```

#[cfg(target_os = "macos")]
use std::cell::RefCell;

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLDevice, MTLResourceOptions};
#[cfg(target_os = "macos")]
use rustc_hash::FxHashMap;

use crate::error::MetalTileError;

// ---------------------------------------------------------------------------
// Platform types
// ---------------------------------------------------------------------------

/// Pool-bucketing key: `(next_power_of_two(size), storage_mode_bits)`.
#[cfg(target_os = "macos")]
pub(crate) type PoolKey = (usize, u64);

/// Reference-counted Metal buffer used by the pool and dispatchers.
#[cfg(target_os = "macos")]
pub(crate) type BufRc =
    std::rc::Rc<objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>;

// ---------------------------------------------------------------------------
// Thread-local storage
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
std::thread_local! {
    /// FxHashMap over HashMap because PoolKey is `(usize, u64)` — already
    /// densely numeric; SipHash would just shuffle bits that don't need it.
    static BUF_POOL: RefCell<FxHashMap<PoolKey, Vec<BufRc>>>
        = RefCell::new(FxHashMap::default());
}

// ---------------------------------------------------------------------------
// Pool handle
// ---------------------------------------------------------------------------

/// Handle to the thread-local buffer pool.
///
/// Cheap to construct (zero‑sized).  All real work happens inside
/// [`BufferPool::acquire`] which hits the `thread_local!` storage.
pub(crate) struct BufferPool {
    #[cfg(target_os = "macos")]
    _private: (),
    #[cfg(not(target_os = "macos"))]
    _private: (),
}

impl BufferPool {
    /// Create a handle to the thread-local pool.
    pub fn new() -> Self { BufferPool { _private: () } }

    /// Acquire a buffer of at least `len` bytes from the pool, or
    /// allocate a fresh one.  The buffer uses `opts` for storage mode
    /// and hazard tracking.
    #[cfg(target_os = "macos")]
    pub(crate) fn acquire(
        &self,
        dev: &ProtocolObject<dyn MTLDevice>,
        len: usize,
        opts: MTLResourceOptions,
    ) -> Result<BufRc, MetalTileError> {
        use objc2_metal::MTLDevice as _;

        let bucket = len.max(4).next_power_of_two();
        let key: PoolKey = (bucket, opts.0 as u64);

        BUF_POOL.with(|cell| {
            let mut pool = cell.borrow_mut();
            let slot = pool.entry(key).or_default();

            // Recycle a buffer whose strong count is 1 (only the pool
            // holds a reference).  Callers that are still using the
            // buffer keep at least one clone alive, bumping the count
            // above 1.
            for buf in slot.iter() {
                if std::rc::Rc::strong_count(buf) == 1 {
                    return Ok(buf.clone());
                }
            }

            let new = std::rc::Rc::new(
                dev.newBufferWithLength_options(bucket, opts).ok_or(MetalTileError::NoDevice)?,
            );
            slot.push(new.clone());
            Ok(new)
        })
    }
}

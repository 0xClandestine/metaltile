//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Dispatch-geometry validation — reject GPU-pinning dispatches before they
//! reach the (non-preemptive) Metal GPU.
//!
//! Metal compute dispatches run to completion with no preemption: an infinite
//! loop inside a kernel never yields, the WindowServer starves of GPU time, and
//! a hard power-cycle is the only recovery (see docs/developing.md, "A wrong
//! dispatch can freeze the machine"). The concrete trap: reduction / simdgroup
//! kernels derive `n_simd = threads_per_threadgroup / 32`; a reduction loop
//! strided by `n_simd` becomes an **infinite GPU loop** when `n_simd == 0`,
//! i.e. when the kernel is dispatched with fewer than 32 threads/threadgroup.
//!
//! This module is the runtime line of defense: it inspects the threadgroup
//! geometry against the kernel's [`KernelMode`] and returns a proper error
//! instead of letting the dispatch pin the device. It is pure (no Metal types),
//! so it is host-testable on any platform; the codegen also emits an in-kernel
//! `if (n_simd == 0u) return;` escape hatch as defense-in-depth.

use metaltile_core::ir::Kernel;

use crate::error::MetalTileError;

/// The Apple-Silicon simdgroup width. `simd_*` intrinsics reduce across exactly
/// this many lanes, so any kernel that strides a loop by `n_simd = tpg / 32`
/// needs full simdgroups.
const SIMD_WIDTH: usize = 32;

/// Validate a dispatch's `grid` (threadgroup counts) and `tpg`
/// (threads-per-threadgroup).
///
/// `uses_n_simd` is the freeze-prone signal from
/// [`metaltile_codegen::kernel_uses_n_simd`] — `true` iff the kernel derives
/// `n_simd = tpg / 32` and strides a reduction by it. (Mode alone is too coarse:
/// the per-thread `mt_hadamard_m*` matvecs are Reduction-mode but dispatch
/// safely at TPG < 32 because they never compute `n_simd`.)
///
/// Rejects, with a [`MetalTileError::Dispatch`] describing the violation:
/// - any zero grid/threadgroup dimension (dispatches nothing, or is malformed);
/// - a threadgroup larger than the device's `max_threads_per_threadgroup`;
/// - when `uses_n_simd`, a threadgroup that is **not a positive multiple of 32**
///   — fewer than 32 threads makes `n_simd == 0` (a zero-stride loop → infinite
///   GPU loop → device pin), and a non-multiple silently drops the tail
///   simdgroup.
pub fn validate_dispatch_geometry(
    kernel: &Kernel,
    grid: [usize; 3],
    tpg: [usize; 3],
    max_threads_per_threadgroup: usize,
    uses_n_simd: bool,
) -> Result<(), MetalTileError> {
    if grid.iter().chain(tpg.iter()).any(|&d| d == 0) {
        return Err(MetalTileError::Dispatch(format!(
            "kernel '{}': degenerate dispatch — a grid/threadgroup dimension is 0 \
             (grid={grid:?}, tpg={tpg:?})",
            kernel.name
        )));
    }

    let total = tpg[0].saturating_mul(tpg[1]).saturating_mul(tpg[2]);
    if total > max_threads_per_threadgroup {
        return Err(MetalTileError::Dispatch(format!(
            "kernel '{}': {total} threads/threadgroup exceeds the device maximum of \
             {max_threads_per_threadgroup} (tpg={tpg:?})",
            kernel.name
        )));
    }

    if uses_n_simd && (total < SIMD_WIDTH || !total.is_multiple_of(SIMD_WIDTH)) {
        return Err(MetalTileError::Dispatch(format!(
            "kernel '{}' strides a reduction by n_simd = tpg / {SIMD_WIDTH}: \
             threads/threadgroup must be a positive multiple of {SIMD_WIDTH}, got {total} \
             (tpg={tpg:?}) — fewer than {SIMD_WIDTH}, or a non-multiple, would pin the GPU \
             (n_simd = 0 → infinite loop)",
            kernel.name
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::Kernel;

    use super::*;

    fn kernel() -> Kernel { Kernel::new("test_kernel") }

    const MAX_TPG: usize = 1024;

    // A kernel that derives `n_simd` (uses_n_simd = true).
    #[test]
    fn accepts_valid_n_simd_geometry() {
        // 1024, 512, 256, 32 — all positive multiples of 32.
        for t in [32usize, 256, 512, 1024] {
            assert!(
                validate_dispatch_geometry(&kernel(), [4, 1, 1], [t, 1, 1], MAX_TPG, true).is_ok(),
                "tpg {t} should be valid"
            );
        }
        // Multiple of 32 spread across dims (16·2 = 32).
        assert!(
            validate_dispatch_geometry(&kernel(), [4, 1, 1], [16, 2, 1], MAX_TPG, true).is_ok()
        );
    }

    #[test]
    fn rejects_n_simd_kernel_below_one_simdgroup() {
        // The documented freeze: tpg < 32 → n_simd == 0 → infinite GPU loop.
        for t in [1usize, 4, 16, 31] {
            let err = validate_dispatch_geometry(&kernel(), [4, 1, 1], [t, 1, 1], MAX_TPG, true)
                .unwrap_err();
            assert!(matches!(err, MetalTileError::Dispatch(_)), "tpg {t} must be rejected");
        }
    }

    #[test]
    fn rejects_n_simd_kernel_non_multiple_of_32() {
        // 48 = 32 + 16: the tail 16 threads can't form a full simdgroup.
        assert!(
            validate_dispatch_geometry(&kernel(), [4, 1, 1], [48, 1, 1], MAX_TPG, true).is_err()
        );
    }

    #[test]
    fn non_n_simd_kernel_allows_small_threadgroups() {
        // The per-thread `mt_hadamard_m*` case: Reduction-mode but never derives
        // n_simd, so a TPG of 12 / 28 / 1 must be allowed.
        for t in [1usize, 12, 28] {
            assert!(
                validate_dispatch_geometry(&kernel(), [4, 1, 1], [t, 1, 1], MAX_TPG, false).is_ok(),
                "non-n_simd kernel with tpg {t} should be valid"
            );
        }
    }

    #[test]
    fn rejects_zero_dimension_regardless_of_n_simd() {
        for uses_n_simd in [true, false] {
            assert!(
                validate_dispatch_geometry(&kernel(), [0, 1, 1], [256, 1, 1], MAX_TPG, uses_n_simd)
                    .is_err()
            );
            assert!(
                validate_dispatch_geometry(&kernel(), [4, 1, 1], [256, 0, 1], MAX_TPG, uses_n_simd)
                    .is_err()
            );
        }
    }

    #[test]
    fn rejects_threadgroup_over_device_cap() {
        // 2048 > 1024 device max — independent of the n_simd rule.
        assert!(
            validate_dispatch_geometry(&kernel(), [4, 1, 1], [2048, 1, 1], MAX_TPG, false).is_err()
        );
    }
}

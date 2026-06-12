//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Name-dispatchable wiring + ergonomics for the grouped CUTLASS MoE GEMM
//! family (the f16 grouped GEMM and its block-scaled NVFP4 sibling).
//!
//! Issue #285: the two grouped CUTLASS GEMMs (`moe_grouped_cutlass`, #274, and
//! `moe_grouped_cutlass_fp4`, #282) shipped as raw FFI methods on
//! [`CudaDevice`] with ~11 positional, mixed-type arguments and a manually
//! managed prepared-handle pointer. Nothing could reach them by name from the
//! planner/executor, the positional argument lists invited silent pointer
//! transposition, and the prepared handle leaked (the `_release` C entry was
//! never bound on the Rust side).
//!
//! This module adds the family-consistent fixes:
//!
//!   * [`MoeGroupedGemmDesc`] / [`MoeGroupedFp4Desc`] — `repr(C)` descriptors
//!     that group the device pointers and scalar dimensions for each kernel.
//!     The fields are named, so a transposed pointer is a compile error rather
//!     than a silent wrong-buffer dispatch. Both kernels get one so the family
//!     stays consistent.
//!   * [`MoePreparedHandle`] — an owning newtype over the persistent NVFP4
//!     handle whose [`Drop`] calls the C `_release` entry, fixing the leak.
//!   * [`MoeGroupedKernel`] + [`CudaDevice::dispatch_grouped_moe`] — a small
//!     by-NAME dispatch seam so the planner/executor can invoke either kernel
//!     through one entry point keyed on the registered kernel name, instead of
//!     calling the raw `CudaDevice` method directly. This is the bounded
//!     increment toward planner integration.
//!
//! DEFERRED to #285: a first-class `Op::InlineCuda` IR variant (the CUDA analog
//! of `Op::InlineMsl`) so the grouped GEMMs can appear as IR nodes inside a
//! fused kernel graph and flow through codegen/`type_check`/`remap` the way
//! MSL inline ops do. That is a large IR change touching `metaltile-core`,
//! `metaltile-codegen`, and every IR pass; it is intentionally NOT half-built
//! here. The by-name registry below makes the kernels reachable from the
//! planner today; the IR-node representation is the remaining #285 work.

use super::ffi::CUdeviceptr;
use super::CudaDevice;
use crate::error::MetalTileError;

// ---------------------------------------------------------------------------
// Registered kernel names
// ---------------------------------------------------------------------------

/// Registered name for the f16 grouped CUTLASS MoE GEMM (#274).
pub const MOE_GROUPED_CUTLASS: &str = "moe_grouped_cutlass";
/// Registered name for the block-scaled NVFP4 grouped CUTLASS MoE GEMM (#282).
pub const MOE_GROUPED_CUTLASS_FP4: &str = "moe_grouped_cutlass_fp4";

/// Every grouped-MoE GEMM reachable through [`CudaDevice::dispatch_grouped_moe`].
///
/// Resolved from a `&str` name via [`MoeGroupedKernel::from_name`] so the
/// planner/executor can pick a kernel by its registered name without a direct
/// reference to the concrete `CudaDevice` method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeGroupedKernel {
    /// f16 grouped GEMM (#274).
    F16,
    /// block-scaled NVFP4 grouped GEMM (#282).
    Fp4,
}

impl MoeGroupedKernel {
    /// Resolve a registered kernel name to its variant, or `None` if unknown.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            MOE_GROUPED_CUTLASS => Some(Self::F16),
            MOE_GROUPED_CUTLASS_FP4 => Some(Self::Fp4),
            _ => None,
        }
    }

    /// The registered name for this variant.
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => MOE_GROUPED_CUTLASS,
            Self::Fp4 => MOE_GROUPED_CUTLASS_FP4,
        }
    }
}

// ---------------------------------------------------------------------------
// repr(C) argument descriptors
// ---------------------------------------------------------------------------

/// Device-pointer + scalar arguments for the f16 grouped CUTLASS MoE GEMM
/// (#274). Replaces the positional `(a, w, c, n, k)` device/scalar arguments;
/// the per-group host slices stay borrowed (see [`CudaDevice::moe_grouped_cutlass_desc`]).
///
/// `a` = sorted-token f16 `[mt, K]`, `w` = contiguous f16 expert slab
/// `[n_exp, N, K]`, `c` = f16 out `[mt, N]`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoeGroupedGemmDesc {
    /// sorted-token f16 activations `[mt, K]`.
    pub a: CUdeviceptr,
    /// contiguous f16 expert weight slab `[n_exp, N, K]`.
    pub w: CUdeviceptr,
    /// f16 output `[mt, N]`.
    pub c: CUdeviceptr,
    /// GEMM N (output feature dim).
    pub n: usize,
    /// GEMM K (contraction dim).
    pub k: usize,
}

/// Device-pointer + scalar arguments for the block-scaled NVFP4 grouped CUTLASS
/// MoE GEMM (#282). Replaces the positional device/scalar arguments; the
/// per-group host slices stay borrowed (see
/// [`CudaDevice::moe_grouped_cutlass_fp4_desc`]).
///
/// `a` = packed e2m1 sorted-token activations `[mt, K/2]`, `sfa` = per-group
/// ue4m3 scale pool, `w`/`sfw` = packed e2m1 + scale expert slabs, `c` = f16 out
/// `[mt, N]`. `alpha_vec` = device `f32[n_groups]` per-group scales (`0` = none).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoeGroupedFp4Desc {
    /// packed e2m1 sorted-token activations `[mt, K/2]`.
    pub a: CUdeviceptr,
    /// per-group ue4m3 activation scale pool.
    pub sfa: CUdeviceptr,
    /// packed e2m1 expert weight slab.
    pub w: CUdeviceptr,
    /// ue4m3 expert weight scale slab.
    pub sfw: CUdeviceptr,
    /// f16 output `[mt, N]`.
    pub c: CUdeviceptr,
    /// device `f32[n_groups]` per-group scale, `0` = none (alpha 1).
    pub alpha_vec: CUdeviceptr,
    /// GEMM N (output feature dim).
    pub n: usize,
    /// GEMM K (contraction dim).
    pub k: usize,
}

// ---------------------------------------------------------------------------
// owning prepared-handle newtype
// ---------------------------------------------------------------------------

/// Owning handle for a prepared persistent NVFP4 grouped GEMM
/// (`moe_grouped_cutlass_fp4_prepare`). Releasing the C-side resources is
/// automatic on [`Drop`] — it calls the `_release` FFI entry — so callers no
/// longer manage the raw `u64` pointer (which previously leaked).
pub struct MoePreparedHandle {
    handle: u64,
}

impl MoePreparedHandle {
    /// Wrap a raw prepared-handle pointer returned by the `_prepare` FFI.
    ///
    /// # Safety
    /// `handle` must be a non-null handle returned by
    /// `moe_grouped_gemm_cutlass_fp4_prepare` and not otherwise released.
    pub(crate) unsafe fn from_raw(handle: u64) -> Self {
        MoePreparedHandle { handle }
    }

    /// The raw handle pointer, for passing to the `_run` FFI.
    pub fn as_raw(&self) -> u64 {
        self.handle
    }
}

impl Drop for MoePreparedHandle {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        #[cfg(have_cutlass)]
        unsafe {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_release(handle: *mut core::ffi::c_void);
            }
            moe_grouped_gemm_cutlass_fp4_release(self.handle as *mut core::ffi::c_void);
        }
    }
}

// ---------------------------------------------------------------------------
// by-name dispatch + descriptor-based entry points
// ---------------------------------------------------------------------------

impl CudaDevice {
    /// Dispatch a grouped CUTLASS MoE GEMM by its registered name.
    ///
    /// This is the planner/executor-facing seam: a name (`"moe_grouped_cutlass"`
    /// or `"moe_grouped_cutlass_fp4"`) selects the kernel, and the matching
    /// descriptor + per-group host slices drive it. The non-fp4 path ignores
    /// `sfa`/`sfw`/`alpha_vec`/`sfa_off`; the fp4 path requires them.
    ///
    /// Prefer this over calling the concrete methods directly when the kernel is
    /// chosen at plan time by name.
    pub fn dispatch_grouped_moe(
        &self,
        kernel: MoeGroupedKernel,
        f16_desc: Option<&MoeGroupedGemmDesc>,
        fp4_desc: Option<&MoeGroupedFp4Desc>,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
    ) -> Result<(), MetalTileError> {
        match kernel {
            MoeGroupedKernel::F16 => {
                let d = f16_desc.ok_or_else(|| {
                    MetalTileError::Dispatch(
                        "dispatch_grouped_moe: F16 kernel needs a MoeGroupedGemmDesc".into(),
                    )
                })?;
                self.moe_grouped_cutlass_desc(d, group_rows, expert_ids)
            }
            MoeGroupedKernel::Fp4 => {
                let d = fp4_desc.ok_or_else(|| {
                    MetalTileError::Dispatch(
                        "dispatch_grouped_moe: Fp4 kernel needs a MoeGroupedFp4Desc".into(),
                    )
                })?;
                self.moe_grouped_cutlass_fp4_desc(d, group_rows, expert_ids, sfa_off)
            }
        }
    }

    /// Descriptor form of [`CudaDevice::moe_grouped_cutlass`]: the device
    /// pointers + dims travel in a named [`MoeGroupedGemmDesc`], the per-group
    /// host slices stay borrowed.
    pub fn moe_grouped_cutlass_desc(
        &self,
        desc: &MoeGroupedGemmDesc,
        group_rows: &[i32],
        expert_ids: &[i32],
    ) -> Result<(), MetalTileError> {
        self.moe_grouped_cutlass(desc.a, desc.w, desc.c, group_rows, expert_ids, desc.n, desc.k)
    }

    /// Descriptor form of [`CudaDevice::moe_grouped_cutlass_fp4`]: the device
    /// pointers + dims travel in a named [`MoeGroupedFp4Desc`], the per-group
    /// host slices stay borrowed.
    pub fn moe_grouped_cutlass_fp4_desc(
        &self,
        desc: &MoeGroupedFp4Desc,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
    ) -> Result<(), MetalTileError> {
        self.moe_grouped_cutlass_fp4(
            desc.a,
            desc.sfa,
            desc.w,
            desc.sfw,
            desc.c,
            group_rows,
            expert_ids,
            sfa_off,
            desc.alpha_vec,
            desc.n,
            desc.k,
        )
    }
}

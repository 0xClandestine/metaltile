//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! HIP / ROCm codegen backend (`AMD_BACKEND_SPEC.md`).
//!
//! HIP is a CUDA-portable C++ dialect: at the **kernel** level, `__global__`,
//! `blockIdx`/`threadIdx`/`blockDim`, `__shared__`, `__syncthreads`,
//! `__syncwarp`, `__shfl_*_sync`, atomics, and the precise single-precision
//! math intrinsics (`expf`, `exp2f`, `rsqrtf`, `__frcp_rn`, `fmaxf`/`fminf`,
//! `fmaf`) are **bit-identical** to the CUDA emitter's output. The only deltas
//! are:
//!
//! 1. Header includes — `cuda_fp16.h` / `cuda_bf16.h` → `hip/hip_fp16.h` /
//!    `hip/hip_bf16.h`.
//! 2. The bf16 type name — `__nv_bfloat16` → `__hip_bfloat16`.
//! 3. The `MT_INF` literal — `__int_as_float` is in HIP, but reusing the same
//!    expression keeps the preamble identical.
//!
//! So this generator **does not fork the CUDA op-walker**. It delegates to
//! [`crate::cuda::CudaGenerator`] for the full kernel emission, then runs a
//! small textual transform (`cuda_to_hip`) over the result. That keeps the
//! invariants that took the CUDA backend 4164/4164 passes to establish.
//!
//! **Wavefront width:** the default profile is wave32 (RDNA1+; gfx10/11/12 —
//! including the RX 9070 XT / gfx1201). CDNA wave64 needs the broader 64-bit
//! shuffle masks the existing CUDA emitter doesn't yet produce; that is the
//! Phase-2 hazard called out in `AMD_BACKEND_SPEC.md §4.1` and is gated by
//! [`TargetProfile::hip_wave64`].

use metaltile_core::ir::Kernel;

use crate::{
    Result,
    backend::{CodegenBackend, Target, TargetProfile},
    cuda::CudaGenerator,
};

/// HIP code generator. Thin wrapper around [`CudaGenerator`] that post-
/// processes the emitted CUDA source into HIP-flavored C++.
#[derive(Debug, Clone)]
pub struct HipGenerator {
    profile: TargetProfile,
    /// We construct a CUDA-profiled inner generator (lane_width 32, shared
    /// idioms) so the **op-walker** uses CUDA spellings the transform
    /// recognises; the *outer* `profile` is what callers see via the
    /// `CodegenBackend` trait.
    inner: CudaGenerator,
}

impl Default for HipGenerator {
    fn default() -> Self { Self::new() }
}

impl HipGenerator {
    pub fn new() -> Self {
        Self::with_profile(TargetProfile::hip())
    }

    /// Pin to a specific HIP profile. For [`TargetProfile::hip_wave64`]
    /// (CDNA — MI200/MI300/MI350), we *also* swap the inner CUDA
    /// generator's profile to a lane_width=64 CUDA variant so its
    /// reduction tree sizes the shared scratch to 64 warps and emits the
    /// right per-warp boundaries. The textual transform then widens the
    /// shuffle masks to 64 bits.
    ///
    /// The inner CudaGenerator profile inherits `mma` and `lane_width`
    /// from the HIP profile — that's how `SoftwareLocalC` reaches the
    /// CoopTile op-walker.
    pub fn with_profile(profile: TargetProfile) -> Self {
        debug_assert_eq!(profile.target, Target::Hip);
        let mut inner_profile = TargetProfile::cuda();
        inner_profile.lane_width = profile.lane_width;
        inner_profile.mma = profile.mma;
        // Propagate the linear-order subgroup-sum opt-in so the inner
        // CUDA emitter lowers f32 Sum/Mean to the shuffle-loop path. This
        // is what makes HIP match the CPU oracle's `iter().sum()` rounding
        // bit-exactly on the gated-DeltaNet recurrence.
        inner_profile.precise_simd_sum = profile.precise_simd_sum;
        Self { profile, inner: CudaGenerator::with_profile(inner_profile) }
    }

    /// Dynamic shared-memory bytes for a launch, forwarded to the inner
    /// CUDA generator (the layout is identical — `__shared__` decls are
    /// pure C and survive `cuda_to_hip` unchanged).
    pub fn shared_bytes(&self, kernel: &Kernel, block_x: u32) -> usize {
        self.inner.shared_bytes(kernel, block_x)
    }
}

impl CodegenBackend for HipGenerator {
    fn target(&self) -> Target { Target::Hip }

    fn profile(&self) -> &TargetProfile { &self.profile }

    fn generate(&self, kernel: &Kernel) -> Result<String> {
        let cuda_src = self.inner.generate(kernel)?;
        let src = cuda_to_hip(&cuda_src);
        // Wave64 (CDNA) — widen any 32-bit shuffle masks to 64-bit
        // ALL-LANES-ACTIVE. The wave32 transform already swapped
        // `0xffffffffu` to `0xffffffffull` (the low-32 mask in a 64-bit
        // container), but on wave64 the upper 32 lanes are real, so the
        // mask needs all-1s.
        if self.profile.lane_width == 64 {
            Ok(src.replace("0xffffffffull", "0xffffffffffffffffull"))
        } else {
            Ok(src)
        }
    }
}

/// Textual CUDA → HIP transform. Surgical: only touches the lines that
/// genuinely differ between the dialects. The kernel body itself is
/// untouched because the CUDA syntax HIP recognises 1:1.
///
/// The transforms are **conservative** — each only matches text the CUDA
/// emitter actually produces, so unrelated code (e.g. user-supplied
/// preambles in future inline kernels) is not silently rewritten.
pub fn cuda_to_hip(cuda_src: &str) -> String {
    let mut s = cuda_src.to_string();

    // Header includes. hipRTC auto-includes `hip/hip_runtime.h`, so the
    // `__global__`/`__shared__` keywords resolve without an explicit include
    // — we only need the fp16 / bf16 type headers.
    s = s.replace("#include <cuda_fp16.h>", "#include <hip/hip_fp16.h>");
    s = s.replace("#include <cuda_bf16.h>", "#include <hip/hip_bf16.h>");

    // bf16 type name. NVIDIA's is `__nv_bfloat16`; AMD's is `__hip_bfloat16`.
    s = s.replace("__nv_bfloat16", "__hip_bfloat16");

    // Warp-shuffle mask width — the wave32/wave64 hazard from
    // `AMD_BACKEND_SPEC.md §4.1`. HIP 7.x ships a `static_assert
    // (sizeof(MaskT) == 8)` on `__shfl_*_sync`, so the CUDA-style
    // `0xffffffffu` (32-bit `unsigned int`) literal fails to compile even on
    // wave32 (RDNA). Rewrite to `0xffffffffull` (64-bit `unsigned long long`,
    // low-32 mask) which is correct on wave32 AND a valid subset on wave64
    // (wave64 users wanting all-lanes-active can later swap to the broader
    // `0xffffffffffffffffull` via a wave64-specific transform).
    //
    // We don't touch `__shfl_*_sync` itself — HIP exposes the same API as
    // CUDA — only the mask literal width changes.
    s = s.replace("0xffffffffu", "0xffffffffull");

    // Header preamble comment.
    s = s.replace(
        "// Generated by MetalTile (CUDA backend). DO NOT EDIT.",
        "// Generated by MetalTile (HIP backend). DO NOT EDIT.",
    );

    s
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ParamKind, ValueId},
        shape::Shape,
    };

    use super::*;

    fn vector_add_ir() -> Kernel {
        let mut k = Kernel::new("vector_add");
        for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
            k.params.push(Param {
                name: name.into(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: is_out,
                kind: ParamKind::Tensor,
            });
        }
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.name_value(ValueId::new(0), "idx");
        k.body.push_op(
            Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
            ValueId::new(1),
        );
        k.body.name_value(ValueId::new(1), "x");
        k.body.push_op(
            Op::Load { src: "b".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
            ValueId::new(2),
        );
        k.body.name_value(ValueId::new(2), "y");
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(3),
        );
        k.body.name_value(ValueId::new(3), "sum");
        k.body.push_op_no_result(Op::Store {
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(3),
            mask: None,
        });
        k
    }

    #[test]
    fn hip_generator_emits_vector_add() {
        let src = HipGenerator::new().generate(&vector_add_ir()).unwrap();
        // Same kernel body as CUDA — no transform touches the op walker.
        assert!(src.contains("extern \"C\" __global__ void vector_add("));
        assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        // HIP-flavored headers.
        assert!(src.contains("#include <hip/hip_fp16.h>"));
        assert!(src.contains("#include <hip/hip_bf16.h>"));
        // No CUDA header should survive.
        assert!(!src.contains("cuda_fp16.h"));
        assert!(!src.contains("cuda_bf16.h"));
        assert!(!src.contains("__nv_bfloat16"));
        // Banner is updated.
        assert!(src.contains("(HIP backend)"));
    }

    #[test]
    fn hip_generator_reports_hip_target() {
        let g = HipGenerator::new();
        assert_eq!(g.target(), Target::Hip);
        assert_eq!(g.profile().shared_mem_kw, "__shared__");
        assert_eq!(g.profile().lane_width, 32);
    }

    #[test]
    fn cuda_to_hip_is_idempotent() {
        // Running the transform twice produces the same output (each rule
        // only fires against the CUDA spellings, not the HIP ones).
        let src = HipGenerator::new().generate(&vector_add_ir()).unwrap();
        let twice = cuda_to_hip(&src);
        assert_eq!(src, twice);
    }

    /// Build a simple reduction-mode kernel so the emitted source contains
    /// a `__shfl_down_sync` mask — the only place wave32 vs wave64 differ
    /// textually. We don't run it (no CDNA hardware available); we just
    /// inspect the source.
    fn row_reduce_sum_ir() -> Kernel {
        use metaltile_core::constexpr::ConstExpr;
        use metaltile_core::ir::{ConstExprDecl, KernelMode, ReduceKind};
        let mut k = Kernel::new("row_reduce_sum");
        k.mode = KernelMode::Reduction;
        k.params.push(Param {
            name: "inp".into(), dtype: DType::F32, shape: Shape::scalar(),
            is_output: false, kind: ParamKind::Tensor,
        });
        k.params.push(Param {
            name: "out".into(), dtype: DType::F32, shape: Shape::scalar(),
            is_output: true, kind: ParamKind::Tensor,
        });
        k.constexprs.push(ConstExprDecl {
            name: ConstExpr::new("n"),
            dtype: DType::U32,
            value: None,
        });
        let (row, nv, rs, re, acc, res) = (
            ValueId::new(0), ValueId::new(1), ValueId::new(2),
            ValueId::new(3), ValueId::new(4), ValueId::new(5),
        );
        k.body.push_op(Op::ProgramId { axis: 0 }, row);
        k.body.name_value(row, "row");
        k.body.push_op(Op::Load { src: "n".into(), indices: vec![], mask: None, other: None }, nv);
        k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: row, rhs: nv }, rs);
        k.body.name_value(rs, "rs");
        k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: rs, rhs: nv }, re);
        k.body.name_value(re, "re");
        k.body.push_op(
            Op::StrideReduce {
                src: "inp".into(), offset: rs, stride: nv, end: re,
                op: ReduceKind::Sum, dtype: DType::F32,
                transform: None, secondary_src: None, secondary_base: None,
            },
            acc,
        );
        k.body.name_value(acc, "acc");
        k.body.push_op(Op::Reduce { value: acc, axis: 0, op: ReduceKind::Sum }, res);
        k.body.name_value(res, "result");
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(), indices: vec![IndexExpr::Value(row)], value: res, mask: None,
        });
        k
    }

    #[test]
    fn wave32_emits_32bit_mask() {
        let g = HipGenerator::with_profile(TargetProfile::hip());
        let src = g.generate(&row_reduce_sum_ir()).unwrap();
        // Wave32: low-32 mask wrapped in a 64-bit container to satisfy
        // HIP 7.x's `static_assert(sizeof(MaskT) == 8)`.
        assert!(src.contains("__shfl_down_sync(0xffffffffull"));
        assert!(!src.contains("0xffffffffffffffffull"));
    }

    #[test]
    fn wave64_emits_64bit_mask() {
        let g = HipGenerator::with_profile(TargetProfile::hip_wave64());
        let src = g.generate(&row_reduce_sum_ir()).unwrap();
        // Wave64: full all-lanes-active 64-bit mask. Inner CUDA generator
        // also sizes the warp-reduce loop to lane_width=64.
        assert!(src.contains("__shfl_down_sync(0xffffffffffffffffull"));
        // No bare 32-bit-in-64-bit leftover anywhere.
        assert!(!src.contains("0xffffffffull,"));
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Vulkan / SPIR-V codegen backend (`VULKAN_BACKEND_SPEC.md`).
//!
//! Emits GLSL compute shader text (compiled to SPIR-V at runtime by shaderc
//! in the Vulkan runtime). Text-emit-then-compile mirrors the CUDA / HIP
//! path; we go through GLSL rather than direct `rspirv` because GLSL is
//! debuggable and shaderc is mature. A `rspirv` direct emitter
//! (`VULKAN_BACKEND_SPEC.md §5`) is the planned upgrade once the op
//! surface stabilises.
//!
//! ## Phase 2 coverage
//!
//! - **`KernelMode::Elementwise`** — 1-D bounds-guarded gid;
//!   `pc._n_elems` push constant.
//! - **`KernelMode::Grid3D`** — per-axis `gl_GlobalInvocationID.{x,y,z}`.
//! - **`KernelMode::Reduction`** — block-per-output-row; per-thread
//!   grid-stride accumulation (`Op::StrideReduce`) → workgroup-shared
//!   barrier-tree reduction (`Op::Reduce`). The reduction is
//!   **subgroup-width agnostic** (the portable path called out in
//!   `VULKAN_BACKEND_SPEC.md §4.1`) — it depends only on `local_size_x`
//!   and `barrier()`, so it runs correctly on any Vulkan subgroup size.
//!   The subgroup-op fast path is Phase 3.
//!
//! Ops covered: `ProgramId`, `Const`, `Load`, `Store`, `BinOp`, `UnaryOp`,
//! `Fma`, `Cast`, `Select`, `Activation`, `DeclareLocal`, `SetLocal`,
//! `StrideReduce`, `Reduce`, `ThreadgroupAlloc`/`Load`/`Store`,
//! `StackAlloc`/`Load`/`Store`, `Barrier`, `Loop`, `If`.
//!
//! ## Param layout
//!
//! Every tensor param becomes a Storage Buffer Object (SSBO) bound at its
//! position in `kernel.params`. Constexprs and the synthetic `_n_elems`
//! bounds guard share one **push constant block** (`PushConsts`).
//!
//! ## Phase 2 deliberate simplifications (tracked for Phase 3)
//!
//! - Only `f32` / `i32` / `u32` SSBOs. `f16`/`bf16`/`i8` need
//!   `VK_KHR_shader_float16_int8` + matching feature declarations (`§4.3`).
//! - No subgroup ops (`SimdReduce`/`SimdScan`/`SimdShuffle*`). The
//!   portable workgroup reduction covers `Op::Reduce`; subgroup-level ops
//!   error as `UnsupportedOp` and will land behind a
//!   `VK_EXT_subgroup_size_control` feature query in Phase 3.
//! - No `SimdgroupMatMul` / `CoopTile*` — the MMA family is Phase 4
//!   (`VK_KHR_cooperative_matrix` with the runtime shape-query ladder).
//! - No atomics yet (`Op::Atomic`). GLSL `atomicAdd` exists; wiring it is
//!   a small Phase-3 chore.
//! - `Strided` params still unsupported.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use metaltile_core::{
    dtype::DType,
    ir::{
        ActKind, BinOpKind, Block, IndexExpr, Kernel, KernelMode, Op, Param, ParamKind,
        ReduceKind, UnaryOpKind, ValueId,
    },
};

use crate::{
    Result,
    backend::{CodegenBackend, Target, TargetProfile},
    error::Error,
};

/// Default compute-shader local size (`local_size_x`).
pub const DEFAULT_LOCAL_SIZE: u32 = 256;

/// SSA value-name override map (loop vars + parent-block names) threaded
/// into nested Loop/If body emission, mirroring the CUDA walker.
type Names = BTreeMap<ValueId, String>;

/// Per-SSA glsl type — one of "uint" / "int" / "float". Threaded through
/// op emission so `BinOp::{Add,Sub,Mul,Div}` can promote to the operand
/// type when both lhs/rhs are integer (the systemic `idx / N`
/// float-division bug fix — `pos = rest / head_dim` for u32 operands
/// MUST stay integer division or the dequant-by-byte indexing falls
/// apart).
type Types = BTreeMap<ValueId, &'static str>;

/// Per-mutable-local glsl type — keyed by the IR's local name (eg
/// `rem`, `src_off`, `acc`). Threaded the same way so `DeclareLocal`
/// picks the right declared type and `SetLocal`/`Load(__ml_X)`
/// preserve it. The strided_copy_nd `let mut rem = p` was the bug —
/// declaring `float mt_loc_rem` turned `rem / extent` into float
/// division and every thread ended up reading `src[0]`.
type LocalTypes = BTreeMap<String, &'static str>;

#[derive(Debug, Clone)]
pub struct GlslGenerator {
    profile: TargetProfile,
    local_size: [u32; 3],
}

impl Default for GlslGenerator {
    fn default() -> Self { Self::new() }
}

impl GlslGenerator {
    pub fn new() -> Self {
        Self { profile: TargetProfile::vulkan(), local_size: [DEFAULT_LOCAL_SIZE, 1, 1] }
    }

    pub fn with_profile(profile: TargetProfile) -> Self {
        debug_assert_eq!(profile.target, Target::Spirv);
        Self { profile, local_size: [DEFAULT_LOCAL_SIZE, 1, 1] }
    }

    /// Override the workgroup x-dimension only (1-D dispatch). Convenience
    /// for kernels that don't need y/z.
    pub fn with_local_size(mut self, local_size: u32) -> Self {
        self.local_size = [local_size, 1, 1];
        self
    }

    /// Override the full 3-D workgroup size — `local_size_x/y/z`. Required
    /// by `KernelMode::Grid3D` kernels that take a 3-D tpg from the harness.
    pub fn with_local_size_3d(mut self, local_size: [u32; 3]) -> Self {
        self.local_size = local_size;
        self
    }

    pub fn local_size(&self) -> u32 { self.local_size[0] }
    pub fn local_size_3d(&self) -> [u32; 3] { self.local_size }
    /// Total threads per workgroup (product of all 3 axes). Drives the
    /// shared-memory reduction scratch sizing.
    pub fn local_size_total(&self) -> u32 {
        self.local_size[0] * self.local_size[1] * self.local_size[2]
    }

    /// True when the gated `VK_KHR_cooperative_matrix` codegen path is
    /// active (set via `MT_VK_COOPMAT=1`, see `TargetProfile::vulkan`).
    /// All coopmat-specific emission (extensions, fp16 staging tiles,
    /// fragment MMA) branches on this.
    fn coopmat_on(&self) -> bool {
        matches!(self.profile.mma, crate::backend::MmaStrategy::VkCooperativeMatrix)
    }

    fn vname(&self, vid: Option<ValueId>, block: &Block, ov: &Names) -> String {
        match vid {
            Some(v) => {
                if let Some(n) = ov.get(&v) {
                    return n.clone();
                }
                match block.names.get(&v) {
                    Some(hint) => format!("v_{hint}_{}", v.as_u32()),
                    None => format!("v{}", v.as_u32()),
                }
            }
            None => "/*<no-value>*/".to_string(),
        }
    }

    fn idx_term(&self, ix: &IndexExpr, block: &Block, ov: &Names) -> String {
        match ix {
            IndexExpr::Value(v) => self.vname(Some(*v), block, ov),
            IndexExpr::Const(c) => format!("uint({c})"),
            IndexExpr::Range(v, off) => {
                format!("({} + uint({off}))", self.vname(Some(*v), block, ov))
            }
        }
    }

    fn emit_idx(
        &self,
        indices: &[IndexExpr],
        block: &Block,
        ov: &Names,
        kernel: &Kernel,
        src: &str,
    ) -> Result<String> {
        match indices {
            [] => Ok(String::new()),
            [one] => Ok(self.idx_term(one, block, ov)),
            many => {
                let param = kernel.params.iter().find(|p| p.name == src);
                if matches!(param.map(|p| &p.kind), Some(ParamKind::Strided)) {
                    // Strided: use the companion `_strides` SSBO so the
                    // index respects the host's per-dim stride layout.
                    let arr = safe_glsl_ident(src);
                    let terms: Vec<String> = many
                        .iter()
                        .enumerate()
                        .map(|(d, ix)| {
                            format!("({}) * {arr}_strides[{d}]", self.idx_term(ix, block, ov))
                        })
                        .collect();
                    return Ok(terms.join(" + "));
                }
                let shape = param.map(|p| &p.shape).ok_or_else(|| {
                    Error::UnsupportedOp(format!("spirv: multi-dim index on unknown param `{src}`"))
                })?;
                let rank = shape.rank();
                if rank < many.len() {
                    return Err(Error::UnsupportedOp(format!(
                        "spirv: multi-dim index rank mismatch on `{src}`"
                    )));
                }
                let mut terms = Vec::new();
                for (d, ix) in many.iter().enumerate() {
                    let mut stride: u64 = 1;
                    for s in (d + 1)..rank {
                        match shape.dim(s) {
                            Some(metaltile_core::shape::Dim::Known(n)) => stride *= *n as u64,
                            _ => {
                                return Err(Error::UnsupportedOp(format!(
                                    "spirv: multi-dim index needs static dims on `{src}`"
                                )));
                            }
                        }
                    }
                    terms.push(format!(
                        "({}) * uint({stride})",
                        self.idx_term(ix, block, ov)
                    ));
                }
                Ok(terms.join(" + "))
            }
        }
    }

    fn emit_preamble(&self, out: &mut String) {
        writeln!(out, "#version 460").ok();
        writeln!(out, "// Generated by MetalTile (Vulkan/SPIR-V backend). DO NOT EDIT.").ok();
        // GLSL extensions required for Phase-3 dtype coverage.
        // The driver/device must support the matching Vulkan features
        // (shaderFloat16, shaderInt8, storageBuffer16BitAccess,
        // storageBuffer8BitAccess, scalarBlockLayout) — VulkanDevice
        // requests them at creation. shaderc errors at compile time if a
        // kernel uses an extension the target Vulkan version doesn't
        // know — but we target 1.2, where all of these are core/promoted.
        writeln!(out, "#extension GL_EXT_shader_explicit_arithmetic_types : enable").ok();
        writeln!(out, "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable").ok();
        writeln!(out, "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable").ok();
        writeln!(out, "#extension GL_EXT_shader_16bit_storage : enable").ok();
        writeln!(out, "#extension GL_EXT_shader_8bit_storage : enable").ok();
        writeln!(out, "#extension GL_EXT_scalar_block_layout : enable").ok();
        // Subgroup ops (Vulkan 1.1 core via KHR_shader_subgroup family).
        // Required by Op::SimdReduce / SimdScan / SimdBroadcast /
        // SimdShuffleXor / SimdgroupBarrier. The extensions are
        // additive — the smaller ones (basic, vote) are guaranteed by
        // Vulkan 1.1; the larger sets (arithmetic, shuffle, ballot) are
        // optional but present on every desktop GPU we care about.
        writeln!(out, "#extension GL_KHR_shader_subgroup_basic : enable").ok();
        writeln!(out, "#extension GL_KHR_shader_subgroup_vote : enable").ok();
        writeln!(out, "#extension GL_KHR_shader_subgroup_arithmetic : enable").ok();
        writeln!(out, "#extension GL_KHR_shader_subgroup_ballot : enable").ok();
        writeln!(out, "#extension GL_KHR_shader_subgroup_shuffle : enable").ok();
        writeln!(out, "#extension GL_KHR_shader_subgroup_shuffle_relative : enable").ok();
        // bfloat16 is much newer (VK_KHR_shader_bfloat16, 2024). When the
        // device doesn't support it, we fall back to a software shim that
        // promotes bf16 through f32 (see `mt_bf16_to_f32` helpers).
        writeln!(out, "#extension GL_EXT_bfloat16 : enable").ok();
        // Real cooperative-matrix path (gated by MmaStrategy::VkCooperativeMatrix
        // via `MT_VK_COOPMAT=1`). Emits coopMatLoad/coopMatMulAdd/coopMatStore
        // 16×16×16 fp16→fp32 fragment ops on RDNA4. Requires the Vulkan memory
        // model + explicit fp16 storage; the runtime requests the matching
        // device features and pins subgroupSize=64 for these kernels.
        if self.coopmat_on() {
            writeln!(out, "#extension GL_KHR_cooperative_matrix : require").ok();
            writeln!(out, "#extension GL_KHR_memory_scope_semantics : require").ok();
            writeln!(out, "#extension GL_EXT_control_flow_attributes : require").ok();
        }
        // GLSL has no `INFINITY` macro (MSL / CUDA both do — and the IR
        // emits the literal directly for mask values like `-INFINITY`).
        // `1.0/0.0` is well-defined as `+inf` under IEEE-754 and is what
        // glslang lowers to OpConstant with the inf bit set.
        writeln!(out, "#define INFINITY (1.0/0.0)").ok();
        writeln!(out).ok();
        // bfloat16 conversion helpers. GLSL has no native bfloat16 type in
        // Vulkan 1.2 (the `GL_EXT_bfloat16` extension is too new for most
        // drivers), so we store bf16 as `uint16_t` and bit-reinterpret on
        // load / store. The conversion shifts the bf16 16-bit pattern into
        // the high half of a uint32 (which IS an IEEE-754 f32 sharing the
        // bf16 sign + exponent + 7 mantissa bits) and uses
        // `uintBitsToFloat`. Round-to-nearest-even on the store side.
        writeln!(out, "float mt_bf16_to_f32(uint16_t b) {{").ok();
        writeln!(out, "    return uintBitsToFloat(uint(b) << 16);").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "uint16_t mt_f32_to_bf16(float f) {{").ok();
        writeln!(out, "    uint u = floatBitsToUint(f);").ok();
        // RNE rounding bias = 0x7fff + (mantissa-bit-16 ? 1 : 0).
        writeln!(out, "    uint rne = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;").ok();
        writeln!(out, "    return uint16_t(rne & 0xffffu);").ok();
        writeln!(out, "}}").ok();
        writeln!(out).ok();
        // Block-scaled quant decode helpers — ports of the CUDA preamble.
        // Pure arithmetic, so they transcribe directly to GLSL. The corpus
        // hit ~30 dequant kernels through the `/*TODO-decode-*/` stub in
        // Phase 2.1; landing the real decode unlocks all of them.
        writeln!(out, "float mt_decode_e2m1(uint code) {{").ok();
        writeln!(out, "    uint m = code & 7u;").ok();
        writeln!(out, "    float mag = (m < 1u) ? 0.0 : (m < 2u) ? 0.5 : (m < 3u) ? 1.0 : (m < 4u) ? 1.5").ok();
        writeln!(out, "              : (m < 5u) ? 2.0 : (m < 6u) ? 3.0 : (m < 7u) ? 4.0 : 6.0;").ok();
        writeln!(out, "    return ((code & 8u) != 0u) ? -mag : mag;").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "float mt_decode_e4m3(uint bits) {{").ok();
        writeln!(out, "    uint e = (bits >> 3u) & 15u;").ok();
        writeln!(out, "    uint m = bits & 7u;").ok();
        writeln!(out, "    float mag = (e < 1u) ? (float(m) * 0.001953125)").ok();
        writeln!(out, "              : ((1.0 + float(m) * 0.125) * exp2(float(int(e)) - 7.0));").ok();
        writeln!(out, "    return (((bits >> 7u) & 1u) != 0u) ? -mag : mag;").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "float mt_decode_e5m2(uint bits) {{").ok();
        writeln!(out, "    uint e = (bits >> 2u) & 31u;").ok();
        writeln!(out, "    uint m = bits & 3u;").ok();
        writeln!(out, "    float mag = (e < 1u) ? (float(m) * 0.0000152587890625)").ok();
        writeln!(out, "              : ((1.0 + float(m) * 0.25) * exp2(float(int(e)) - 15.0));").ok();
        writeln!(out, "    return (((bits >> 7u) & 1u) != 0u) ? -mag : mag;").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "float mt_decode_int8(uint bits) {{").ok();
        // Sign-extend low 8 bits — GLSL doesn't have signed-byte casts so
        // we do the bit-shift trick: shift up to MSB then arithmetic-shift
        // back down. `int` arith shift is GLSL-defined as signed >>.
        writeln!(out, "    int s = int(bits << 24u) >> 24;").ok();
        writeln!(out, "    return float(s);").ok();
        writeln!(out, "}}").ok();
        writeln!(out).ok();
        // GLSL.std.450 lacks erf / erfinv / expm1 / log10. Provide
        // software stand-ins so the unary intrinsic table can reference
        // them. mt_erfinv is a coarse Winitzki approximation; tune later.
        writeln!(out, "float mt_expm1(float x) {{ return exp(x) - 1.0; }}").ok();
        writeln!(out, "float mt_log10(float x) {{ return log(x) / log(10.0); }}").ok();
        writeln!(out, "float mt_erf(float x) {{").ok();
        writeln!(out, "    float t = 1.0 / (1.0 + 0.3275911 * abs(x));").ok();
        writeln!(out, "    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)").ok();
        writeln!(out, "        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x*x);").ok();
        writeln!(out, "    return sign(x) * y;").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "float mt_erfinv(float x) {{").ok();
        writeln!(out, "    float ln = log(1.0 - x*x);").ok();
        writeln!(out, "    float a = 0.147;").ok();
        writeln!(out, "    float t = 2.0/(3.14159265 * a) + ln*0.5;").ok();
        writeln!(out, "    return sign(x) * sqrt(sqrt(t*t - ln/a) - t);").ok();
        writeln!(out, "}}").ok();
        // Manual atan2 — AMD driver's GLSL `atan(y, x)` two-arg
        // overload sometimes returns the wrong quadrant (off by π on
        // the corpus test data). Compute via the standard piecewise
        // identity so the result matches CPU `f32::atan2` to ULP.
        // Correctly-rounded f32 divide via Markstein's algorithm. Vulkan
        // does not require `OpFDiv` to be correctly rounded, and AMD's
        // driver substitutes `a / b` → `a * (1/b)`, which costs 1 ULP on
        // values like `8.0 / 1.5`. The GLSL `precise` qualifier doesn't
        // block the substitution (it only blocks fp contraction). One
        // Newton-Raphson refinement step on top of the hardware reciprocal
        // gives a correctly-rounded result, assuming hardware FMA is
        // correctly rounded — which it is on every AMDGPU since GCN.
        writeln!(out, "float mt_fdiv(float a, float b) {{").ok();
        writeln!(out, "    float r = 1.0 / b;").ok();
        writeln!(out, "    float q = a * r;").ok();
        writeln!(out, "    float err = fma(-b, q, a);").ok();
        writeln!(out, "    return fma(err, r, q);").ok();
        writeln!(out, "}}").ok();
        // Refined reciprocal square root. AMD's `V_RSQ_F32` (under
        // GLSL `inversesqrt`) is ~1.4 ULP — fine for graphics, but
        // accumulation-heavy recurrences (gated DeltaNet) compound
        // each ULP through the state. One Newton-Raphson step:
        // r' = r * (1.5 - 0.5 * x * r * r), brings rsqrt to ≤1 ULP.
        writeln!(out, "float mt_rsqrt(float x) {{").ok();
        writeln!(out, "    float r = inversesqrt(x);").ok();
        writeln!(out, "    return r * fma(-0.5 * x * r, r, 1.5);").ok();
        writeln!(out, "}}").ok();
        // Linear-order subgroup sum. `subgroupAdd` uses a butterfly
        // tree, which is a different rounding path than the CPU
        // oracle's left-to-right accumulation. For most kernels this
        // costs at most 1-2 ULPs and both rounds are valid f32, but
        // in accumulation-heavy recurrences (gated DeltaNet over 4
        // tokens with state amplified to magnitude 24256) the drift
        // compounds to ~3 ULPs and exceeds tight absolute tolerances.
        // `mt_subgroup_add` linearly broadcasts and accumulates so the
        // result matches Rust's `iter().sum()` bit-exactly.
        writeln!(out, "float mt_subgroup_add(float v) {{").ok();
        writeln!(out, "    float s = 0.0;").ok();
        writeln!(out, "    for (uint i = 0u; i < 32u; i++) {{").ok();
        writeln!(out, "        s += subgroupShuffle(v, i);").ok();
        writeln!(out, "    }}").ok();
        writeln!(out, "    return s;").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "float mt_atan2(float y, float x) {{").ok();
        writeln!(out, "    const float PI = 3.14159265358979;").ok();
        writeln!(out, "    if (x > 0.0) return atan(y / x);").ok();
        writeln!(out, "    if (x < 0.0) return atan(y / x) + (y >= 0.0 ? PI : -PI);").ok();
        writeln!(out, "    if (y > 0.0) return PI * 0.5;").ok();
        writeln!(out, "    if (y < 0.0) return -PI * 0.5;").ok();
        writeln!(out, "    return 0.0;").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "float mt_silu(float x) {{ return x / (1.0 + exp(-x)); }}").ok();
        writeln!(out, "float mt_relu(float x) {{ return max(0.0, x); }}").ok();
        writeln!(out, "float mt_sigmoid(float x) {{ return 1.0 / (1.0 + exp(-x)); }}").ok();
        writeln!(out, "float mt_gelu(float x) {{").ok();
        writeln!(out, "    const float k = 0.7978845608;").ok();
        writeln!(out, "    float arg = clamp(k * (x + 0.044715 * x*x*x), -15.0, 15.0);").ok();
        writeln!(out, "    return 0.5 * x * (1.0 + tanh(arg));").ok();
        writeln!(out, "}}").ok();
        writeln!(out).ok();
    }

    fn glsl_scalar_type(dt: DType) -> Result<&'static str> {
        Ok(match dt {
            DType::F32 => "float",
            DType::I32 => "int",
            DType::U32 => "uint",
            DType::Bool => "bool",
            // 16-bit / 8-bit types require the matching Vulkan features
            // + GLSL extensions emitted in emit_preamble. The device
            // requests these at create() time; on hardware that lacks them
            // the pipeline create fails and we fall back through
            // run_kernel's error path.
            DType::F16 => "float16_t",
            DType::BF16 => "uint16_t", // bf16 in storage; reinterpreted on use
            DType::U16 => "uint16_t",
            DType::I8 => "int8_t",
            DType::U8 => "uint8_t",
            DType::I4 => "int8_t", // packed; sub-byte handled by decode helpers
            DType::I64 => "int64_t",
            DType::U64 => "uint64_t",
        })
    }

    fn emit_bindings(&self, kernel: &Kernel, out: &mut String) -> Result<()> {
        let mut binding: u32 = 0;
        for p in &kernel.params {
            let ty = Self::glsl_scalar_type(p.dtype)?;
            // For inputs we keep `readonly` as an optimisation hint;
            // outputs go with NO qualifier so kernels that do an
            // in-place read-modify-write on the output (axpy-style,
            // logits-mask, repetition penalty) compile. Pure-write
            // outputs are unaffected by the missing hint.
            let access = if p.is_output { "" } else { "readonly " };
            let arr = safe_glsl_ident(&p.name);
            // `scalar` block layout (Vulkan 1.2 `scalarBlockLayout`) packs
            // every type at its natural alignment — required for 16-bit /
            // 8-bit SSBO arrays to match host-side byte layout. For pure
            // 32-bit kernels it matches std430 exactly, so the change is
            // safe to apply uniformly.
            writeln!(
                out,
                "layout(set = 0, binding = {binding}, scalar) {access}buffer Buf_{name} {{",
                name = p.name,
            )
            .ok();
            writeln!(out, "    {ty} {arr}[];").ok();
            writeln!(out, "}};").ok();
            binding += 1;
            // Strided params carry two companion SSBOs (shape, strides)
            // in signature order — the runtime allocates them adjacent
            // to the data buffer.
            if matches!(p.kind, ParamKind::Strided) {
                writeln!(
                    out,
                    "layout(set = 0, binding = {binding}, scalar) readonly buffer Buf_{name}_shape {{",
                    name = p.name,
                )
                .ok();
                writeln!(out, "    uint {arr}_shape[];").ok();
                writeln!(out, "}};").ok();
                binding += 1;
                writeln!(
                    out,
                    "layout(set = 0, binding = {binding}, scalar) readonly buffer Buf_{name}_strides {{",
                    name = p.name,
                )
                .ok();
                writeln!(out, "    uint {arr}_strides[];").ok();
                writeln!(out, "}};").ok();
                binding += 1;
            }
        }
        writeln!(out).ok();
        Ok(())
    }

    fn emit_push_constants(&self, kernel: &Kernel, out: &mut String) -> Result<()> {
        if kernel.constexprs.is_empty() && kernel.mode != KernelMode::Elementwise {
            return Ok(());
        }
        // `scalar` layout on the push-constant block ensures 16/8-bit
        // constexprs pack at their natural alignment to match the
        // host-side byte stream (we use `to_le_bytes` per constexpr +
        // a trailing u32 `_n_elems`).
        writeln!(out, "layout(push_constant, scalar) uniform PushConsts {{").ok();
        for ce in &kernel.constexprs {
            let ty = Self::glsl_scalar_type(ce.dtype)?;
            // Constexpr names like `length` are GLSL built-in method names
            // and parse as method calls when accessed via `pc.length`.
            // Map to a safe identifier here AND on the read side
            // (Op::Load with empty indices).
            let name = safe_glsl_ident(ce.name.name());
            writeln!(out, "    {ty} {name};").ok();
        }
        if kernel.mode == KernelMode::Elementwise {
            writeln!(out, "    uint _n_elems;").ok();
        }
        writeln!(out, "}} pc;").ok();
        writeln!(out).ok();
        Ok(())
    }

    /// Hoist `ThreadgroupAlloc` (→ `shared` arrays) and `StackAlloc` (→
    /// per-thread arrays declared as local vars in `main`) to the top of
    /// `main`. We declare shared arrays at *file scope* — GLSL requires it.
    fn emit_shared_arrays(&self, kernel: &Kernel, out: &mut String) {
        // Collect shared arrays from every block.
        for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
            for op in &blk.ops {
                if let Op::ThreadgroupAlloc { dtype, size, name } = op {
                    // GLSL `shared` decls must live at file scope. Pass
                    // the name through `safe_glsl_ident` so the corpus's
                    // `shared` / `length` / `output` array names don't
                    // collide with the GLSL keyword we're about to emit.
                    let ty = match dtype {
                        DType::F32 => "float",
                        DType::I32 => "int",
                        DType::U32 => "uint",
                        _ => "float", // fallback for fp16/bf16 (Phase 3)
                    };
                    let n = safe_glsl_ident(name);
                    writeln!(out, "shared {ty} {n}[{size}];").ok();
                }
            }
        }
        // Implicit reduction scratch buffer — one per `Op::Reduce`. We
        // pre-declare them by walking every block. Naming: `_red_<vid>`.
        // Each is sized to `local_size_x` (every thread contributes one
        // partial). For `lsize` = 256 (default), this is 1 KB per Reduce.
        for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
            for (i, op) in blk.ops.iter().enumerate() {
                if matches!(op, Op::Reduce { .. })
                    && let Some(Some(vid)) = blk.results.get(i)
                {
                    writeln!(
                        out,
                        "shared float _red_{}[{}];",
                        vid.as_u32(),
                        self.local_size_total()
                    )
                    .ok();
                }
            }
        }
        // simdgroup_matrix<f32,8,8> per-warp tiles. Each SimdgroupAlloc
        // gets a 64-float slot per warp, indexed by `simd_group * 64 +
        // _sg_fm * 8 + _sg_fnX` — the Apple lane→element map. Total size
        // = 64 * (lsize / lane_width) floats per tile.
        let n_warps = self.local_size_total().div_ceil(self.profile.lane_width).max(1);
        for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
            for (i, op) in blk.ops.iter().enumerate() {
                if matches!(op, Op::SimdgroupAlloc { .. })
                    && let Some(Some(vid)) = blk.results.get(i)
                {
                    let count = 64 * n_warps;
                    writeln!(
                        out,
                        "shared float _SGM_{}[{count}];",
                        vid.as_u32()
                    )
                    .ok();
                }
            }
        }
        // CoopTile shared A/B/C — one tile per Setup. SimdGroup-scope
        // tiles are sized `n_warps * m*k`; Threadgroup-scope tiles are
        // sized `m*k`. C is keyed by `coop_c_name` so `*_acc` setups
        // share a single accumulator (mirrors the CUDA layout).
        //
        // SoftwareLocalC: SimdGroup-scope C moves to a lane-local
        // private array (declared inside `main()`), saving
        // `n_warps * m*n * 4` bytes of shared. The decl itself is emitted
        // in `emit_body`.
        let local_c = matches!(
            self.profile.mma,
            crate::backend::MmaStrategy::SoftwareLocalC
        );
        let mut seen_c: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
            for op in &blk.ops {
                if let Op::CoopTileSetup { name, m, n, k, exec_scope, .. } = op {
                    use metaltile_core::ir::CoopTileScope;
                    let pw = matches!(exec_scope, CoopTileScope::SimdGroup);
                    let mul = if pw { n_warps } else { 1 };
                    let nm = ct_ident(name);
                    // Coopmat staging tiles must be fp16 so `coopMatLoad`
                    // can read them as `gl_MatrixUseA/B` fragments.
                    let st_ty = if self.coopmat_on() { "float16_t" } else { "float" };
                    writeln!(out, "shared {st_ty} _CTA_{nm}[{}];", mul * m * k).ok();
                    writeln!(out, "shared {st_ty} _CTB_{nm}[{}];", mul * k * n).ok();
                    let cnm = ct_ident(&coop_c_name(name));
                    if seen_c.insert(cnm.clone()) {
                        // Skip the shared C entry for SimdGroup-scope
                        // SoftwareLocalC — the accumulator lives in
                        // private memory instead. Coopmat keeps C in
                        // subgroup-distributed registers (declared in
                        // emit_body) so its shared slot is skipped too.
                        if !((local_c || self.coopmat_on()) && pw) {
                            writeln!(
                                out,
                                "shared float _CTC_{cnm}[{}];",
                                mul * m * n
                            )
                            .ok();
                        }
                    }
                }
            }
        }
        writeln!(out).ok();
    }

    fn emit_signature(&self, out: &mut String) {
        let [x, y, z] = self.local_size;
        writeln!(
            out,
            "layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;",
        )
        .ok();
        writeln!(out).ok();
        writeln!(out, "void main() {{").ok();
    }

    fn emit_body(&self, kernel: &Kernel, out: &mut String) -> Result<()> {
        match kernel.mode {
            KernelMode::Elementwise => {
                writeln!(out, "    uint _gtid = gl_GlobalInvocationID.x;").ok();
                writeln!(out, "    if (_gtid >= pc._n_elems) return;").ok();
                writeln!(out, "    uint tid = _gtid;").ok();
                self.emit_simd_aliases(out);
            }
            KernelMode::Grid3D => {
                writeln!(out, "    uint gid_x = gl_GlobalInvocationID.x;").ok();
                writeln!(out, "    uint gid_y = gl_GlobalInvocationID.y;").ok();
                writeln!(out, "    uint gid_z = gl_GlobalInvocationID.z;").ok();
                self.emit_simd_aliases(out);
            }
            KernelMode::SimdGroup2D => {
                // Apple `simdgroup`/`threadgroup_2d` mode — block IDs +
                // local thread IDs both accessible. Mirrors the CUDA
                // emitter's `blockIdx`/`threadIdx` emit.
                writeln!(out, "    uint tgid_x = gl_WorkGroupID.x;").ok();
                writeln!(out, "    uint tgid_y = gl_WorkGroupID.y;").ok();
                writeln!(out, "    uint tgid_z = gl_WorkGroupID.z;").ok();
                writeln!(out, "    uint lid_x = gl_LocalInvocationID.x;").ok();
                writeln!(out, "    uint lid_y = gl_LocalInvocationID.y;").ok();
                writeln!(out, "    uint lid_z = gl_LocalInvocationID.z;").ok();
                writeln!(out, "    uint tid = gl_LocalInvocationIndex;").ok();
                self.emit_simd_aliases(out);
            }
            KernelMode::Reduction => {
                // Block-per-output-row reduction. `tid` is the local
                // 1-D thread index inside the workgroup; the per-axis
                // tgids drive the ProgramId map for the row. `tid`
                // collapses the (y,z) axes — Reduction kernels rarely
                // touch them but we keep the shape consistent.
                writeln!(
                    out,
                    "    uint tid = gl_LocalInvocationID.z * (gl_WorkGroupSize.x * gl_WorkGroupSize.y) + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;"
                )
                .ok();
                writeln!(out, "    uint tgid_x = gl_WorkGroupID.x;").ok();
                writeln!(out, "    uint tgid_y = gl_WorkGroupID.y;").ok();
                writeln!(out, "    uint tgid_z = gl_WorkGroupID.z;").ok();
                writeln!(out, "    uint lsize = {}u;", self.local_size_total()).ok();
                let lw = self.profile.lane_width;
                writeln!(out, "    uint n_simd = lsize / {lw}u;").ok();
                // Linear lane numbering for simdgroup-matrix indexing
                // (see emit_simd_aliases).
                writeln!(out, "    uint simd_lane  = tid % {lw}u;").ok();
                writeln!(out, "    uint simd_group = tid / {lw}u;").ok();
            }
            other => {
                return Err(Error::UnsupportedOp(format!(
                    "spirv: KernelMode::{other:?} not yet supported"
                )));
            }
        }
        // SoftwareLocalC: declare per-warp lane-local C arrays inside
        // main(). Each lane holds `m*n / lane_width` floats; they
        // replace the per-warp shared `_CTC_<cnm>` slot.
        if matches!(self.profile.mma, crate::backend::MmaStrategy::SoftwareLocalC) {
            let lw = self.profile.lane_width;
            let mut seen_c: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
                for op in &blk.ops {
                    if let Op::CoopTileSetup { name, m, n, exec_scope, .. } = op
                        && matches!(exec_scope, metaltile_core::ir::CoopTileScope::SimdGroup)
                    {
                        let cnm = ct_ident(&coop_c_name(name));
                        if seen_c.insert(cnm.clone()) {
                            let per_lane = (m * n).div_ceil(lw).max(1);
                            writeln!(out, "    float _CTC_lane_{cnm}[{per_lane}];").ok();
                        }
                    }
                }
            }
        }
        // Coopmat: declare the register-blocked fragment accumulator
        // for each SimdGroup-scope CoopTile. `acc[MF][NF]` of
        // `coopmat<float,Subgroup,16,16,Accumulator>` persists across the
        // K-loop (CoopTileRun is called per kb step). One array per
        // distinct C name (mirrors `_CTC_lane_*`).
        if self.coopmat_on() {
            let mut seen_c: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
                for op in &blk.ops {
                    if let Op::CoopTileSetup { name, m, n, exec_scope, .. } = op
                        && matches!(exec_scope, metaltile_core::ir::CoopTileScope::SimdGroup)
                    {
                        let cnm = ct_ident(&coop_c_name(name));
                        if seen_c.insert(cnm.clone()) {
                            let mf = (m / 16).max(1);
                            let nf = (n / 16).max(1);
                            writeln!(
                                out,
                                "    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> _CMAcc_{cnm}[{mf}][{nf}];"
                            )
                            .ok();
                        }
                    }
                }
            }
        }
        // Apple simdgroup_matrix<f32,8,8> lane→element coords (from
        // mma_layout_probe). Mirrors the CUDA emitter. Only needed when
        // the kernel actually uses a simdgroup-matrix op; harmless
        // otherwise but kept gated to keep the preamble small.
        if uses_simdgroup(kernel) {
            writeln!(out, "    uint _sg_qid = simd_lane / 4u;").ok();
            writeln!(out, "    uint _sg_fm  = (_sg_qid & 4u) + ((simd_lane / 2u) % 4u);").ok();
            writeln!(out, "    uint _sg_fn0 = (_sg_qid & 2u) * 2u + (simd_lane % 2u) * 2u;").ok();
            writeln!(out, "    uint _sg_fn1 = _sg_fn0 + 1u;").ok();
        }
        let mut types: Types = BTreeMap::new();
        let mut local_types: LocalTypes = BTreeMap::new();
        self.emit_ops(&kernel.body, kernel, &Names::new(), &mut types, &mut local_types, out)?;
        writeln!(out, "}}").ok();
        Ok(())
    }

    fn emit_simd_aliases(&self, out: &mut String) {
        let lw = self.profile.lane_width;
        writeln!(out, "    uint lsize      = {}u;", self.local_size_total()).ok();
        writeln!(out, "    uint n_simd     = lsize / {lw}u;").ok();
        // Linear `tid % 32` / `tid / 32` for the simdgroup-matrix
        // software emulation: the Apple lane→element formula
        // (`_sg_qid = simd_lane / 4`, etc.) requires a consistent
        // linear lane numbering that's *internal* to the shader; the
        // subgroup-id is only used at op level for subgroup intrinsics
        // (`subgroupAdd`, etc., which operate on the actual hardware
        // subgroup regardless of what we call `simd_*`).
        writeln!(out, "    uint simd_lane  = gl_LocalInvocationID.x % {lw}u;").ok();
        writeln!(out, "    uint simd_group = gl_LocalInvocationID.x / {lw}u;").ok();
    }

    fn emit_ops(
        &self,
        blk: &Block,
        kernel: &Kernel,
        ov: &Names,
        types: &mut Types,
        local_types: &mut LocalTypes,
        out: &mut String,
    ) -> Result<()> {
        for (i, op) in blk.ops.iter().enumerate() {
            let vid = blk.results.get(i).and_then(|x| *x);
            self.emit_op(op, vid, blk, kernel, ov, types, local_types, out)?;
        }
        Ok(())
    }

    fn child_ov(&self, parent: &Block, ov: &Names) -> Names {
        let mut child: Names = parent
            .names
            .iter()
            .map(|(&k, v)| (k, format!("v_{v}_{}", k.as_u32())))
            .collect();
        for (&k, v) in ov {
            child.insert(k, v.clone());
        }
        child
    }

    fn emit_op(
        &self,
        op: &Op,
        vid: Option<ValueId>,
        block: &Block,
        kernel: &Kernel,
        ov: &Names,
        types: &mut Types,
        local_types: &mut LocalTypes,
        out: &mut String,
    ) -> Result<()> {
        let pad = "    ";
        match op {
            Op::ProgramId { axis } => {
                let v = self.vname(vid, block, ov);
                let src = match kernel.mode {
                    KernelMode::Elementwise => "_gtid".to_string(),
                    KernelMode::Grid3D => {
                        match axis { 0 => "gid_x", 1 => "gid_y", _ => "gid_z" }.to_string()
                    }
                    KernelMode::Reduction | KernelMode::SimdGroup2D => {
                        // SimdGroup2D: each workgroup processes one
                        // tile, so `program_id(axis)` returns the
                        // per-tile workgroup ID. Without this fix every
                        // TG saw tgid=0 and wrote on top of itself
                        // (the `fused/gather/masked/segmented/splitk_*`
                        // family cascade).
                        match axis { 0 => "tgid_x", 1 => "tgid_y", _ => "tgid_z" }.to_string()
                    }
                    _ => "0u".to_string(),
                };
                writeln!(out, "{pad}uint {v} = {src};").ok();
                if let Some(id) = vid {
                    types.insert(id, "uint");
                }
            }
            Op::Const { value } => {
                let v = self.vname(vid, block, ov);
                if *value >= 0 {
                    writeln!(out, "{pad}uint {v} = uint({value});").ok();
                    if let Some(id) = vid {
                        types.insert(id, "uint");
                    }
                } else {
                    writeln!(out, "{pad}int {v} = {value};").ok();
                    if let Some(id) = vid {
                        types.insert(id, "int");
                    }
                }
            }
            Op::Load { src, indices, .. } => {
                let v = self.vname(vid, block, ov);
                let constexpr = kernel.constexprs.iter().find(|c| c.name.name() == src);
                if indices.is_empty() {
                    // DSL built-in aliases (all uint) — keep them as
                    // `uint` SSA values so subsequent integer arithmetic
                    // (BinOp Div on row indices etc.) stays integer.
                    // Casting these to `float` was the systemic bug that
                    // turned every `idx / N` into float division.
                    if let Some(b) = match src.as_str() {
                        "tid" => Some("tid"),
                        "simd_id" => Some("simd_group"),
                        "simd_lane" => Some("simd_lane"),
                        "simd_group" => Some("simd_group"),
                        "tid_x" => Some("gl_LocalInvocationID.x"),
                        "tid_y" => Some("gl_LocalInvocationID.y"),
                        "tid_z" => Some("gl_LocalInvocationID.z"),
                        // Workgroup-grid built-ins. Mirrors the CUDA
                        // emitter's blockIdx.{x,y,z}. Must emit as uint
                        // so `tg * 8 + sg * 4` index math stays integer
                        // (the qgemv `sg` collapse bug — float `tgid_x`
                        // turned the per-warp row computation into
                        // identical values across SGs).
                        "tgid_x" => Some("tgid_x"),
                        "tgid_y" => Some("tgid_y"),
                        "tgid_z" => Some("tgid_z"),
                        // Grid3D mode globals; Apple's per-axis gid_*.
                        "gid_x" => Some("gid_x"),
                        "gid_y" => Some("gid_y"),
                        "gid_z" => Some("gid_z"),
                        // Local-invocation IDs in SimdGroup2D mode.
                        "lid_x" => Some("lid_x"),
                        "lid_y" => Some("lid_y"),
                        "lid_z" => Some("lid_z"),
                        "n_simd" => Some("n_simd"),
                        "lsize" => Some("lsize"),
                        _ => None,
                    } {
                        writeln!(out, "{pad}uint {v} = {b};").ok();
                        if let Some(id) = vid { types.insert(id, "uint"); }
                    } else if let Some(c) = constexpr {
                        // Constexpr at its declared dtype — integer
                        // constexprs MUST stay integer or every kernel
                        // that does `idx / N` returns float garbage.
                        let nm = safe_glsl_ident(src);
                        match c.dtype {
                            DType::F32 => {
                                writeln!(out, "{pad}float {v} = pc.{nm};").ok();
                                if let Some(id) = vid { types.insert(id, "float"); }
                            }
                            DType::U32 | DType::U16 | DType::U8 => {
                                writeln!(out, "{pad}uint {v} = pc.{nm};").ok();
                                if let Some(id) = vid { types.insert(id, "uint"); }
                            }
                            DType::I32 | DType::I8 | DType::I4 | DType::I64 => {
                                writeln!(out, "{pad}int {v} = pc.{nm};").ok();
                                if let Some(id) = vid { types.insert(id, "int"); }
                            }
                            _ => {
                                writeln!(out, "{pad}float {v} = float(pc.{nm});").ok();
                                if let Some(id) = vid { types.insert(id, "float"); }
                            }
                        }
                    } else if let Some(local_name) = src.strip_prefix("__ml_") {
                        // Mutable-local read. Use the declared type so
                        // `rem / extent` (uint local) stays uint.
                        let lt = local_types.get(local_name).copied().unwrap_or("float");
                        writeln!(out, "{pad}{lt} {v} = mt_loc_{local_name};").ok();
                        if let Some(id) = vid { types.insert(id, lt); }
                    } else {
                        // Other no-index loads (special const literals
                        // like `-INFINITY`). Default to float.
                        writeln!(out, "{pad}float {v} = float({});", safe_glsl_ident(src)).ok();
                        if let Some(id) = vid { types.insert(id, "float"); }
                    }
                } else {
                    let idx = self.emit_idx(indices, block, ov, kernel, src)?;
                    let arr = safe_glsl_ident(src);
                    let dtype = kernel
                        .params
                        .iter()
                        .find(|p| p.name == *src)
                        .map(|p| p.dtype);
                    // Match the SSBO's natural type so bitwise ops on
                    // u32 packs (`val >> 8 & 255`) don't go through a
                    // float round-trip that loses bits. bf16 needs the
                    // helper; other integer dtypes load as their native
                    // type. F32/F16 keep the `float` path so trig /
                    // arithmetic stays in floating point.
                    match dtype {
                        Some(DType::BF16) => {
                            writeln!(out, "{pad}float {v} = mt_bf16_to_f32({arr}[uint({idx})]);").ok();
                            if let Some(id) = vid { types.insert(id, "float"); }
                        }
                        Some(DType::U32) | Some(DType::U16) | Some(DType::U8) => {
                            writeln!(out, "{pad}uint {v} = uint({arr}[uint({idx})]);").ok();
                            if let Some(id) = vid { types.insert(id, "uint"); }
                        }
                        Some(DType::I32) | Some(DType::I8) | Some(DType::I4) | Some(DType::I64) => {
                            writeln!(out, "{pad}int {v} = int({arr}[uint({idx})]);").ok();
                            if let Some(id) = vid { types.insert(id, "int"); }
                        }
                        _ => {
                            writeln!(out, "{pad}float {v} = float({arr}[uint({idx})]);").ok();
                            if let Some(id) = vid { types.insert(id, "float"); }
                        }
                    }
                }
            }
            Op::Store { dst, indices, value, .. } => {
                let val = self.vname(Some(*value), block, ov);
                let idx = self.emit_idx(indices, block, ov, kernel, dst)?;
                let p = kernel.params.iter().find(|p| p.name == *dst);
                let arr = safe_glsl_ident(dst);
                // bf16 destination needs the rounding helper, not a
                // numerical (uint16_t)(float) cast.
                let store_expr = match p.map(|p| p.dtype) {
                    Some(DType::BF16) => format!("mt_f32_to_bf16({val})"),
                    Some(dt) => {
                        let ty = Self::glsl_scalar_type(dt)?;
                        format!("{ty}({val})")
                    }
                    None => format!("float({val})"),
                };
                writeln!(out, "{pad}{arr}[uint({idx})] = {store_expr};").ok();
            }
            Op::BinOp { op: bop, lhs, rhs } => {
                let v = self.vname(vid, block, ov);
                let l = self.vname(Some(*lhs), block, ov);
                let r = self.vname(Some(*rhs), block, ov);
                // Bitwise + integer-mod ops MUST emit a `uint`-typed
                // result, not `float`. Float intermediates lose precision
                // on u32 values > 2^24 (the affine_dequantize bug:
                // `(val >> 0) & 255` returned 0 for w[1]=0x9E3779B1
                // because float(0x9E3779B1) rounded to 0x9E377A00, then
                // `uint(rounded)` had the wrong low byte).
                let int_op = matches!(
                    bop,
                    BinOpKind::Shl
                        | BinOpKind::Shr
                        | BinOpKind::BitAnd
                        | BinOpKind::BitOr
                        | BinOpKind::BitXor
                        | BinOpKind::Mod
                );
                // Arithmetic (Add/Sub/Mul/Div/Min/Max) on two integer
                // operands SHOULD emit at the integer type so integer
                // division stays integer (`rest / head_dim` for u32
                // operands MUST be u32 division — the fp8/conv index
                // bug). Promote to `int` if either side is `int` (signed
                // wins), else `uint`. Cmp* always returns bool→float.
                let arith_op = matches!(
                    bop,
                    BinOpKind::Add
                        | BinOpKind::Sub
                        | BinOpKind::Mul
                        | BinOpKind::Div
                        | BinOpKind::Min
                        | BinOpKind::Max
                );
                let lt = types.get(lhs).copied().unwrap_or("float");
                let rt = types.get(rhs).copied().unwrap_or("float");
                let both_int = (lt == "uint" || lt == "int") && (rt == "uint" || rt == "int");
                let ty = if int_op {
                    "uint"
                } else if arith_op && both_int {
                    if lt == "int" || rt == "int" { "int" } else { "uint" }
                } else {
                    "float"
                };
                // Force IEEE-754 round-to-nearest-even on float divide. The
                // Vulkan spec does NOT require correctly-rounded OpFDiv,
                // so AMD's driver substitutes `v / k` → `v * (1/k)`, which
                // costs 1 ULP. The GLSL `precise` qualifier alone doesn't
                // prevent this substitution (it only blocks fp contraction).
                // Use the Markstein algorithm: one Newton-Raphson refinement
                // step on top of the hardware reciprocal gives correctly-
                // rounded f32 divide assuming hardware fma is correctly
                // rounded (true on AMD GFX1xxx V_FMAC_F32). Without this
                // `test_logits_repetition_penalty` misses bit-exact tol=0
                // by 1 ULP on 2 elements out of 8192.
                if matches!(bop, BinOpKind::Div) && ty == "float" {
                    writeln!(out, "{pad}{ty} {v} = mt_fdiv({l}, {r});").ok();
                } else {
                    writeln!(out, "{pad}{ty} {v} = {};", glsl_binop(*bop, &l, &r)).ok();
                }
                if let Some(id) = vid {
                    types.insert(id, ty);
                }
            }
            Op::Fma { a, b, c } => {
                let v = self.vname(vid, block, ov);
                let av = self.vname(Some(*a), block, ov);
                let bv = self.vname(Some(*b), block, ov);
                let cv = self.vname(Some(*c), block, ov);
                writeln!(out, "{pad}float {v} = fma({av}, {bv}, {cv});").ok();
            }
            Op::UnaryOp { op: uop, value } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                writeln!(out, "{pad}float {v} = {};", self.glsl_unary(*uop, &rv)).ok();
            }
            Op::Cast { value, dtype } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                // `Cast::<U32>()` / `Cast::<I32>()` need to preserve the
                // integer type so subsequent BinOp arithmetic stays
                // integer. Other casts (to bf16/f16/f32 of integer
                // sources) keep the value as `float` — the SSBO Store
                // path applies the right narrowing.
                match dtype {
                    DType::U32 | DType::U16 | DType::U8 => {
                        writeln!(out, "{pad}uint {v} = uint({rv});").ok();
                        if let Some(id) = vid { types.insert(id, "uint"); }
                    }
                    DType::I32 | DType::I8 | DType::I4 | DType::I64 => {
                        writeln!(out, "{pad}int {v} = int({rv});").ok();
                        if let Some(id) = vid { types.insert(id, "int"); }
                    }
                    _ => {
                        writeln!(out, "{pad}float {v} = float({rv});").ok();
                        if let Some(id) = vid { types.insert(id, "float"); }
                    }
                }
            }
            Op::Select { cond, on_true, on_false } => {
                let v = self.vname(vid, block, ov);
                let c = self.vname(Some(*cond), block, ov);
                let a = self.vname(Some(*on_true), block, ov);
                let b = self.vname(Some(*on_false), block, ov);
                // Pick integer type when both branches are integer —
                // forcing `float` here loses precision on u32 > 2^24
                // (the int5/int6 dequant `select(in0_X, u0, u1)` packs).
                let at = types.get(on_true).copied().unwrap_or("float");
                let bt = types.get(on_false).copied().unwrap_or("float");
                let both_int = (at == "uint" || at == "int") && (bt == "uint" || bt == "int");
                let ty = if both_int {
                    if at == "int" || bt == "int" { "int" } else { "uint" }
                } else {
                    "float"
                };
                writeln!(out, "{pad}{ty} {v} = ((bool({c})) ? ({a}) : ({b}));").ok();
                if let Some(id) = vid { types.insert(id, ty); }
            }
            Op::DeclareLocal { name, value } => {
                let rv = self.vname(Some(*value), block, ov);
                // Type-aware declaration. `let mut rem = p` for uint
                // `p` MUST declare `uint mt_loc_rem`, else `rem /
                // extent` is float division (the strided_copy_nd bug).
                let ty = types.get(value).copied().unwrap_or("float");
                writeln!(out, "{pad}{ty} mt_loc_{name} = {rv};").ok();
                local_types.insert(name.clone(), ty);
            }
            Op::SetLocal { name, value } => {
                let rv = self.vname(Some(*value), block, ov);
                let lt = local_types.get(name).copied().unwrap_or("float");
                let rt = types.get(value).copied().unwrap_or("float");
                if rt == lt {
                    writeln!(out, "{pad}mt_loc_{name} = {rv};").ok();
                } else {
                    writeln!(out, "{pad}mt_loc_{name} = {lt}({rv});").ok();
                }
            }
            Op::Activation { kind, value } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                let expr = match kind {
                    ActKind::Silu => format!("mt_silu({rv})"),
                    ActKind::Gelu => format!("mt_gelu({rv})"),
                    ActKind::Relu => format!("mt_relu({rv})"),
                    ActKind::Sigmoid => format!("mt_sigmoid({rv})"),
                    ActKind::Tanh => format!("tanh({rv})"),
                };
                writeln!(out, "{pad}float {v} = {expr};").ok();
            }
            // Per-thread grid-stride accumulation. For Reduction mode the
            // threadgroup cooperates (each thread strides by `lsize`); for
            // Grid3D each thread folds its own run (stride from the IR).
            Op::StrideReduce {
                src, offset, stride, end, op: rk, transform, secondary_src, secondary_base, ..
            } => {
                let v = self.vname(vid, block, ov);
                let off = self.vname(Some(*offset), block, ov);
                let en = self.vname(Some(*end), block, ov);
                let coop = matches!(kernel.mode, KernelMode::Reduction);
                // GLSL is strictly typed: the SSA values we propagate as
                // `float` (from Load/BinOp on indices) must be cast to
                // `uint` to drive the loop. The bound `end` likewise.
                let (start, step) = if coop {
                    (format!("uint({off}) + tid"), "lsize".to_string())
                } else {
                    let st = self.vname(Some(*stride), block, ov);
                    (format!("uint({off})"), format!("uint({st})"))
                };
                let src_arr = safe_glsl_ident(src);
                // Use dtype-aware reads so bf16 SSBOs go through
                // mt_bf16_to_f32 (the systematic `all_reduce_*[bf16]`
                // bug — `float(uint16_t)` is a numerical conversion,
                // not the bit-reinterpret bf16 needs).
                let src_dtype = kernel.params.iter().find(|p| p.name == *src).map(|p| p.dtype);
                let load_src = |ix: &str| -> String {
                    if matches!(src_dtype, Some(DType::BF16)) {
                        format!("mt_bf16_to_f32({src_arr}[{ix}])")
                    } else {
                        format!("float({src_arr}[{ix}])")
                    }
                };
                let base_elem = match secondary_src {
                    Some(sec) => {
                        let bv = self.vname(*secondary_base, block, ov);
                        let sec_arr = safe_glsl_ident(sec);
                        let sec_dtype = kernel
                            .params
                            .iter()
                            .find(|p| p.name == *sec)
                            .map(|p| p.dtype);
                        let load_sec = if matches!(sec_dtype, Some(DType::BF16)) {
                            format!("mt_bf16_to_f32({sec_arr}[uint(_i) - uint({bv})])")
                        } else {
                            format!("float({sec_arr}[uint(_i) - uint({bv})])")
                        };
                        format!("{} * {load_sec}", load_src("_i"))
                    }
                    None => load_src("_i"),
                };
                let elem_expr = match transform.as_ref().map(|t| t.as_slice()) {
                    None | Some([]) => base_elem,
                    Some(ops) => {
                        let mut e = base_elem;
                        for sub in ops {
                            e = match sub {
                                Op::UnaryOp { op, .. } => self.glsl_unary(*op, &e),
                                Op::Activation { kind, .. } => match kind {
                                    ActKind::Silu => format!("mt_silu({e})"),
                                    ActKind::Gelu => format!("mt_gelu({e})"),
                                    ActKind::Relu => format!("mt_relu({e})"),
                                    ActKind::Sigmoid => format!("mt_sigmoid({e})"),
                                    ActKind::Tanh => format!("tanh({e})"),
                                },
                                Op::Cast { dtype, .. } => {
                                    let ty = Self::glsl_scalar_type(*dtype)?;
                                    format!("{ty}({e})")
                                }
                                Op::BinOp { op, rhs, .. } => {
                                    let rv = self.vname(Some(*rhs), block, ov);
                                    match op {
                                        BinOpKind::Mul => format!("(({e}) * float({rv}))"),
                                        BinOpKind::Add => format!("(({e}) + float({rv}))"),
                                        BinOpKind::Sub => format!("(({e}) - float({rv}))"),
                                        BinOpKind::Div => format!("(({e}) / float({rv}))"),
                                        _ => e,
                                    }
                                }
                                _ => e,
                            };
                        }
                        e
                    }
                };
                writeln!(out, "{pad}float {v} = {};", reduce_init(*rk)).ok();
                writeln!(
                    out,
                    "{pad}for (uint _i = {start}; _i < uint({en}); _i += {step}) {{"
                )
                .ok();
                writeln!(out, "{pad}    float _e = {elem_expr};").ok();
                writeln!(out, "{pad}    {v} = {};", reduce_combine(*rk, &v, "_e")).ok();
                writeln!(out, "{pad}}}").ok();
            }
            // Workgroup-shared barrier-tree reduction — subgroup-width
            // agnostic (depends only on `local_size_x` + `barrier()`).
            // This is the portable path called out in
            // `VULKAN_BACKEND_SPEC.md §4.1`; the subgroup-op fast path is
            // Phase 3.
            Op::Reduce { value, axis, op: rk } => {
                if *axis != 0 {
                    return Err(Error::UnsupportedOp(
                        "spirv: Reduce axis != 0 not yet supported".into(),
                    ));
                }
                let v = self.vname(vid, block, ov);
                let input = self.vname(Some(*value), block, ov);
                // Use the pre-declared shared scratch for this Reduce.
                let red = match vid {
                    Some(id) => format!("_red_{}", id.as_u32()),
                    None => {
                        return Err(Error::UnsupportedOp(
                            "spirv: Reduce without a result value".into(),
                        ));
                    }
                };
                let ls = self.local_size_total();
                writeln!(out, "{pad}{red}[tid] = ({input});").ok();
                writeln!(out, "{pad}barrier();").ok();
                // The barrier-tree must start at next-power-of-two(ls)/2 with a
                // `tid + _s < ls` guard. A bare `ls/2` halving tree only sums
                // correctly when ls (local_size/TPG) is a power of two; a
                // non-pow2 TPG (e.g. 384 = 2^7*3 for an RMSNorm at hidden=1536)
                // drops upper-partial lanes and loses ~1/3 of the reduction.
                // No-op for power-of-two ls (tid+_s<ls is then always true).
                let np = (ls as u64).next_power_of_two();
                writeln!(
                    out,
                    "{pad}for (uint _s = {np}u / 2u; _s > 0u; _s >>= 1) {{"
                )
                .ok();
                writeln!(out, "{pad}    if (tid < _s && tid + _s < {ls}u) {{").ok();
                let combine =
                    reduce_combine(*rk, &format!("{red}[tid]"), &format!("{red}[tid + _s]"));
                writeln!(out, "{pad}        {red}[tid] = {combine};").ok();
                writeln!(out, "{pad}    }}").ok();
                writeln!(out, "{pad}    barrier();").ok();
                writeln!(out, "{pad}}}").ok();
                if *rk == ReduceKind::Mean {
                    writeln!(
                        out,
                        "{pad}float {v} = {red}[0] / float({ls});"
                    )
                    .ok();
                } else {
                    writeln!(out, "{pad}float {v} = {red}[0];").ok();
                }
            }
            // Threadgroup memory: declarations hoisted to file scope by
            // emit_shared_arrays; the ops themselves are loads/stores.
            Op::ThreadgroupAlloc { .. } => {} // no-op (hoisted)
            Op::ThreadgroupLoad { name, index } => {
                let v = self.vname(vid, block, ov);
                let iv = self.vname(Some(*index), block, ov);
                let n = safe_glsl_ident(name);
                let (ty, expr) = dtype_load(tg_alloc_dtype(kernel, name), &format!("{n}[uint({iv})]"));
                writeln!(out, "{pad}{ty} {v} = {expr};").ok();
                if let Some(id) = vid { types.insert(id, ty); }
            }
            Op::ThreadgroupStore { name, index, value } => {
                let iv = self.vname(Some(*index), block, ov);
                let rv = self.vname(Some(*value), block, ov);
                let n = safe_glsl_ident(name);
                // Cast RHS to the shared array's declared type so
                // float→uint/int stores compile under strict GLSL typing.
                let dt = tg_alloc_dtype(kernel, name);
                let val = match dt {
                    Some(DType::U32) | Some(DType::U16) | Some(DType::U8) => {
                        format!("uint({rv})")
                    }
                    Some(DType::I32) | Some(DType::I8) | Some(DType::I4) | Some(DType::I64) => {
                        format!("int({rv})")
                    }
                    _ => format!("{rv}"),
                };
                writeln!(out, "{pad}{n}[uint({iv})] = {val};").ok();
            }
            // Per-thread (local) array — GLSL declares it in main.
            Op::StackAlloc { dtype, size, name } => {
                let ty = match dtype {
                    DType::F32 => "float",
                    DType::I32 => "int",
                    DType::U32 => "uint",
                    _ => "float",
                };
                let n = safe_glsl_ident(name);
                writeln!(out, "{pad}{ty} {n}[{size}];").ok();
            }
            Op::StackLoad { name, index } => {
                let v = self.vname(vid, block, ov);
                let iv = self.vname(Some(*index), block, ov);
                let n = safe_glsl_ident(name);
                let (ty, expr) = dtype_load(stack_alloc_dtype(kernel, name), &format!("{n}[uint({iv})]"));
                writeln!(out, "{pad}{ty} {v} = {expr};").ok();
                if let Some(id) = vid { types.insert(id, ty); }
            }
            Op::StackStore { name, index, value } => {
                let iv = self.vname(Some(*index), block, ov);
                let rv = self.vname(Some(*value), block, ov);
                let n = safe_glsl_ident(name);
                writeln!(out, "{pad}{n}[uint({iv})] = {rv};").ok();
            }
            Op::Barrier => {
                writeln!(out, "{pad}barrier();").ok();
            }
            // ── Subgroup (simdgroup / warp) primitives ────────────────
            // Vulkan calls the warp-level group the "subgroup"; on RDNA
            // 4 wave32, subgroup size = 32 = profile.lane_width. We use
            // the KHR_shader_subgroup_* extension family so the same
            // op covers AMD/NVIDIA/Intel cleanly.
            Op::SimdReduce { value, op: rk } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                // Sum/Mean go through `mt_subgroup_add` (linear-order
                // broadcast loop) so the result matches the CPU oracle's
                // left-to-right `iter().sum()` rounding exactly. Other
                // reductions (Max/Min order-independent, Product rarely
                // accumulation-heavy) stay on the fast hardware subgroup
                // intrinsic.
                let expr = match rk {
                    ReduceKind::Sum | ReduceKind::Mean => format!("mt_subgroup_add({rv})"),
                    ReduceKind::Max => format!("subgroupMax({rv})"),
                    ReduceKind::Min => format!("subgroupMin({rv})"),
                    ReduceKind::Product => format!("subgroupMul({rv})"),
                };
                writeln!(out, "{pad}float {v} = {expr};").ok();
                if *rk == ReduceKind::Mean {
                    // Mean is "sum / size"; the linear-add helper returns
                    // the sum across exactly 32 lanes.
                    writeln!(out, "{pad}{v} = {v} / 32.0;").ok();
                }
            }
            Op::SimdScan { value, op: rk, exclusive } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                let prefix = if *exclusive { "Exclusive" } else { "Inclusive" };
                let intrin = match rk {
                    ReduceKind::Sum | ReduceKind::Mean => format!("subgroup{prefix}Add"),
                    ReduceKind::Max => format!("subgroup{prefix}Max"),
                    ReduceKind::Min => format!("subgroup{prefix}Min"),
                    ReduceKind::Product => format!("subgroup{prefix}Mul"),
                };
                writeln!(out, "{pad}float {v} = {intrin}({rv});").ok();
            }
            Op::SimdLaneId => {
                let v = self.vname(vid, block, ov);
                // Use the preamble's `simd_lane` (`tid % 32`) so this
                // matches the simdgroup-matrix lane→element mapping.
                writeln!(out, "{pad}uint {v} = simd_lane;").ok();
                if let Some(id) = vid { types.insert(id, "uint"); }
            }
            Op::SimdGroupId => {
                let v = self.vname(vid, block, ov);
                // Same — `simd_group` from the preamble (`tid / 32`)
                // matches the shared simdgroup-matrix tile offset.
                writeln!(out, "{pad}uint {v} = simd_group;").ok();
                if let Some(id) = vid { types.insert(id, "uint"); }
            }
            Op::SimdBroadcast { value, lane } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                let rl = self.vname(Some(*lane), block, ov);
                writeln!(out, "{pad}float {v} = subgroupBroadcast({rv}, uint({rl}));").ok();
            }
            Op::SimdShuffleXor { value, mask } => {
                let v = self.vname(vid, block, ov);
                let rv = self.vname(Some(*value), block, ov);
                writeln!(out, "{pad}float {v} = subgroupShuffleXor({rv}, {mask}u);").ok();
            }
            Op::SimdgroupBarrier => {
                writeln!(out, "{pad}subgroupMemoryBarrierShared(); subgroupBarrier();").ok();
            }
            // Atomic ops on storage buffers — GLSL has direct intrinsics.
            // `Op::Atomic` is the IR's portable atomic-RMW; for f32 sums
            // we need `atomicAdd` on `float` which requires
            // `VK_EXT_shader_atomic_float`. For integer paths the core
            // GLSL intrinsics suffice; we emit those unconditionally and
            // leave the f32 atomic to a Phase-4 follow-up.
            Op::Atomic { op: ak, dst, index, value, .. } => {
                let iv = self.vname(Some(*index), block, ov);
                let rv = self.vname(Some(*value), block, ov);
                let arr = safe_glsl_ident(dst);
                let f = match ak {
                    metaltile_core::ir::AtomicKind::Add => "atomicAdd",
                    metaltile_core::ir::AtomicKind::Max => "atomicMax",
                    metaltile_core::ir::AtomicKind::Min => "atomicMin",
                    metaltile_core::ir::AtomicKind::And => "atomicAnd",
                    metaltile_core::ir::AtomicKind::Or => "atomicOr",
                    metaltile_core::ir::AtomicKind::Xor => "atomicXor",
                };
                // GLSL's integer atomics require the operand to be the
                // SAME type as the SSBO element. Our SSA values flow as
                // `float`; cast to the SSBO element type by looking up
                // the kernel param's dtype.
                let dt = kernel
                    .params
                    .iter()
                    .find(|p| p.name == *dst)
                    .map(|p| p.dtype)
                    .unwrap_or(DType::U32);
                let val_cast = match dt {
                    DType::U32 | DType::U16 | DType::U8 => format!("uint({rv})"),
                    DType::I32 | DType::I8 | DType::I4 | DType::I64 => format!("int({rv})"),
                    _ => format!("{rv}"),
                };
                writeln!(out, "{pad}{f}({arr}[uint({iv})], {val_cast});").ok();
            }
            // ── simdgroup_matrix<f32,8,8> software emulation ─────────────
            // Apple's per-warp 8×8 float fragment. The CUDA emitter
            // already does the software emulation (each warp owns a
            // 64-element shared tile, indexed by an Apple-specific
            // lane→element map). For Vulkan we mirror it via workgroup
            // `shared` arrays + the same lane coordinate math.
            //
            // The shared declaration is hoisted by `emit_shared_arrays`
            // (added below); per-op handlers only emit the access.
            Op::SimdgroupAlloc { .. } => {} // declared at file scope
            Op::SimdgroupElemLoad { value, index } => {
                let v = self.vname(vid, block, ov);
                let m = sgm_name(*value);
                let fnk = if *index == 0 { "_sg_fn0" } else { "_sg_fn1" };
                writeln!(
                    out,
                    "{pad}float {v} = {m}[simd_group * 64u + _sg_fm * 8u + {fnk}];"
                )
                .ok();
            }
            Op::SimdgroupElemStore { value, index, data } => {
                let m = sgm_name(*value);
                let dv = self.vname(Some(*data), block, ov);
                let fnk = if *index == 0 { "_sg_fn0" } else { "_sg_fn1" };
                writeln!(
                    out,
                    "{pad}{m}[simd_group * 64u + _sg_fm * 8u + {fnk}] = {dv};"
                )
                .ok();
            }
            Op::SimdgroupLoad { dest, tg, offset, stride, transpose } => {
                let m = sgm_name(*dest);
                let off = self.vname(Some(*offset), block, ov);
                writeln!(out, "{pad}{{ uint _base = simd_group * 64u;").ok();
                for fnk in ["_sg_fn0", "_sg_fn1"] {
                    let idx = if *transpose {
                        format!("{fnk} * {stride}u + _sg_fm")
                    } else {
                        format!("_sg_fm * {stride}u + {fnk}")
                    };
                    writeln!(
                        out,
                        "{pad}    {m}[_base + _sg_fm * 8u + {fnk}] = float({tg}[uint({off}) + {idx}]);"
                    )
                    .ok();
                }
                writeln!(out, "{pad}}}").ok();
            }
            // ── CoopTile cooperative GEMM (software emulation) ──────────
            // Mirrors the CUDA emitter's `mpp::matmul2d` software path,
            // adapted to GLSL.  Per-warp A/B/C shared tiles are hoisted
            // by emit_shared_arrays; ops here are loads / runs / stores.
            Op::CoopTileSetup { .. } => {} // declared at file scope
            Op::CoopTileZero { name } => {
                let (m, n, _, _, _, _, _, simd) = self.coop_cfg(kernel, name).ok_or_else(|| {
                    Error::UnsupportedOp(format!("spirv: CoopTile `{name}` no Setup"))
                })?;
                let cnm = ct_ident(&coop_c_name(name));
                if self.coopmat_on() && simd {
                    // Zero the register-blocked coopmat fragment accumulator.
                    let mf = (m / 16).max(1);
                    let nf = (n / 16).max(1);
                    writeln!(out, "{pad}[[unroll]] for (uint _mi = 0u; _mi < {mf}u; _mi++)").ok();
                    writeln!(out, "{pad}[[unroll]] for (uint _ni = 0u; _ni < {nf}u; _ni++)").ok();
                    writeln!(out, "{pad}    _CMAcc_{cnm}[_mi][_ni] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);").ok();
                    return Ok(());
                }
                let local_c = matches!(
                    self.profile.mma,
                    crate::backend::MmaStrategy::SoftwareLocalC
                );
                if local_c && simd {
                    let lw = self.profile.lane_width;
                    let per_lane = (m * n).div_ceil(lw).max(1);
                    writeln!(
                        out,
                        "{pad}for (uint _i = 0u; _i < {per_lane}u; _i++) _CTC_lane_{cnm}[_i] = 0.0;"
                    )
                    .ok();
                } else {
                    let (sid, ssize, _) = coop_scope_glsl(simd, self.profile.lane_width);
                    let base = coop_base_glsl(simd, m * n);
                    writeln!(
                        out,
                        "{pad}for (uint _e = {sid}; _e < {}u; _e += {ssize}) _CTC_{cnm}[{base} + _e] = 0.0;",
                        m * n
                    )
                    .ok();
                }
            }
            Op::CoopTileLoadA { name, ptr_name, ptr_offset, dtype: _, ei, .. } => {
                let (m, _, k, ta, _, _, _, simd) =
                    self.coop_cfg(kernel, name).ok_or_else(|| {
                        Error::UnsupportedOp(format!("spirv: CoopTile `{name}` no Setup"))
                    })?;
                let nm = ct_ident(name);
                let (sid, ssize, _) = coop_scope_glsl(simd, self.profile.lane_width);
                let base = coop_base_glsl(simd, m * k);
                let off = ptr_offset
                    .map(|o| self.vname(Some(o), block, ov))
                    .unwrap_or_else(|| "0".to_string());
                let arr = safe_glsl_ident(ptr_name);
                let src = if ta {
                    format!("(_e % {k}u) * {ei}u + (_e / {k}u)")
                } else {
                    format!("(_e / {k}u) * {ei}u + (_e % {k}u)")
                };
                let cast = if self.coopmat_on() { "float16_t" } else { "float" };
                writeln!(out, "{pad}barrier();").ok();
                writeln!(
                    out,
                    "{pad}for (uint _e = {sid}; _e < {}u; _e += {ssize}) _CTA_{nm}[{base} + _e] = {cast}({arr}[uint({off}) + {src}]);",
                    m * k
                )
                .ok();
            }
            Op::CoopTileLoadB { name, ptr_name, ptr_offset, dtype: _, ei, .. } => {
                let (_, n, k, _, tb, _, _, simd) =
                    self.coop_cfg(kernel, name).ok_or_else(|| {
                        Error::UnsupportedOp(format!("spirv: CoopTile `{name}` no Setup"))
                    })?;
                let nm = ct_ident(name);
                let (sid, ssize, _) = coop_scope_glsl(simd, self.profile.lane_width);
                let base = coop_base_glsl(simd, k * n);
                let off = ptr_offset
                    .map(|o| self.vname(Some(o), block, ov))
                    .unwrap_or_else(|| "0".to_string());
                let arr = safe_glsl_ident(ptr_name);
                let src = if tb {
                    format!("(_e % {n}u) * {ei}u + (_e / {n}u)")
                } else {
                    format!("(_e / {n}u) * {ei}u + (_e % {n}u)")
                };
                let cast = if self.coopmat_on() { "float16_t" } else { "float" };
                writeln!(
                    out,
                    "{pad}for (uint _e = {sid}; _e < {}u; _e += {ssize}) _CTB_{nm}[{base} + _e] = {cast}({arr}[uint({off}) + {src}]);",
                    k * n
                )
                .ok();
            }
            Op::CoopTileRun { name, .. } => {
                let (m, n, k, _, _, _, accum, simd) =
                    self.coop_cfg(kernel, name).ok_or_else(|| {
                        Error::UnsupportedOp(format!("spirv: CoopTile `{name}` no Setup"))
                    })?;
                let nm = ct_ident(name);
                let cnm = ct_ident(&coop_c_name(name));
                if self.coopmat_on() && simd {
                    // Real VK_KHR_cooperative_matrix MMA. Decompose the
                    // m×n×k SimdGroup tile into 16×16×16 fragments. A-tile
                    // reuse: each subgroup loads `matA` once per (mi,kt)
                    // K-step and reuses it across the NF matB column tiles
                    // (the structure that hit ~70 TFLOP/s on RDNA4 vs the
                    // naive one-tile-per-subgroup form). `acc[MF][NF]` is
                    // the register-blocked accumulator declared in
                    // emit_body and persists across the K-loop.
                    let mf = (m / 16).max(1);
                    let nf = (n / 16).max(1);
                    let kf = (k / 16).max(1);
                    let ba = coop_base_glsl(true, m * k);
                    let bb = coop_base_glsl(true, k * n);
                    writeln!(out, "{pad}barrier();").ok();
                    writeln!(out, "{pad}[[unroll]] for (uint _kt = 0u; _kt < {kf}u; _kt++) {{").ok();
                    writeln!(out, "{pad}    [[unroll]] for (uint _mi = 0u; _mi < {mf}u; _mi++) {{").ok();
                    writeln!(out, "{pad}        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> _ca;").ok();
                    writeln!(out, "{pad}        coopMatLoad(_ca, _CTA_{nm}, {ba} + (_mi * 16u) * {k}u + _kt * 16u, {k}u, gl_CooperativeMatrixLayoutRowMajor);").ok();
                    writeln!(out, "{pad}        [[unroll]] for (uint _ni = 0u; _ni < {nf}u; _ni++) {{").ok();
                    writeln!(out, "{pad}            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> _cb;").ok();
                    writeln!(out, "{pad}            coopMatLoad(_cb, _CTB_{nm}, {bb} + (_kt * 16u) * {n}u + _ni * 16u, {n}u, gl_CooperativeMatrixLayoutRowMajor);").ok();
                    writeln!(out, "{pad}            _CMAcc_{cnm}[_mi][_ni] = coopMatMulAdd(_ca, _cb, _CMAcc_{cnm}[_mi][_ni]);").ok();
                    writeln!(out, "{pad}        }}").ok();
                    writeln!(out, "{pad}    }}").ok();
                    writeln!(out, "{pad}}}").ok();
                    writeln!(out, "{pad}barrier();").ok();
                    return Ok(());
                }
                let local_c = matches!(
                    self.profile.mma,
                    crate::backend::MmaStrategy::SoftwareLocalC
                );
                if local_c && simd {
                    let lw = self.profile.lane_width;
                    let per_lane = (m * n).div_ceil(lw).max(1);
                    let ba = coop_base_glsl(true, m * k);
                    let bb = coop_base_glsl(true, k * n);
                    writeln!(out, "{pad}barrier();").ok();
                    writeln!(
                        out,
                        "{pad}for (uint _li = 0u; _li < {per_lane}u; _li++) {{"
                    )
                    .ok();
                    writeln!(out, "{pad}    uint _e = simd_lane + _li * {lw}u;").ok();
                    writeln!(out, "{pad}    if (_e >= {}u) break;", m * n).ok();
                    writeln!(out, "{pad}    uint _i = _e / {n}u, _j = _e % {n}u;").ok();
                    let init = if accum {
                        format!("_CTC_lane_{cnm}[_li]")
                    } else {
                        "0.0".to_string()
                    };
                    writeln!(out, "{pad}    float _acc = {init};").ok();
                    writeln!(out, "{pad}    for (uint _l = 0u; _l < {k}u; _l++) _acc += _CTA_{nm}[{ba} + _i * {k}u + _l] * _CTB_{nm}[{bb} + _l * {n}u + _j];").ok();
                    writeln!(out, "{pad}    _CTC_lane_{cnm}[_li] = _acc;").ok();
                    writeln!(out, "{pad}}}").ok();
                    writeln!(out, "{pad}barrier();").ok();
                } else {
                    let (sid, ssize, _) = coop_scope_glsl(simd, self.profile.lane_width);
                    let (ba, bb, bc) = (
                        coop_base_glsl(simd, m * k),
                        coop_base_glsl(simd, k * n),
                        coop_base_glsl(simd, m * n),
                    );
                    writeln!(out, "{pad}barrier();").ok();
                    writeln!(
                        out,
                        "{pad}for (uint _e = {sid}; _e < {}u; _e += {ssize}) {{",
                        m * n
                    )
                    .ok();
                    writeln!(out, "{pad}    uint _i = _e / {n}u, _j = _e % {n}u;").ok();
                    let init = if accum { format!("_CTC_{cnm}[{bc} + _e]") } else { "0.0".into() };
                    writeln!(out, "{pad}    float _acc = {init};").ok();
                    writeln!(out, "{pad}    for (uint _l = 0u; _l < {k}u; _l++) _acc += _CTA_{nm}[{ba} + _i * {k}u + _l] * _CTB_{nm}[{bb} + _l * {n}u + _j];").ok();
                    writeln!(out, "{pad}    _CTC_{cnm}[{bc} + _e] = _acc;").ok();
                    writeln!(out, "{pad}}}").ok();
                    writeln!(out, "{pad}barrier();").ok();
                }
            }
            Op::CoopTileStoreC { name, ptr_name, ptr_offset, dtype: _, ei, .. } => {
                let (m, n, _, _, _, _, _, simd) =
                    self.coop_cfg(kernel, name).ok_or_else(|| {
                        Error::UnsupportedOp(format!("spirv: CoopTile `{name}` no Setup"))
                    })?;
                let cnm = ct_ident(&coop_c_name(name));
                let off = ptr_offset
                    .map(|o| self.vname(Some(o), block, ov))
                    .unwrap_or_else(|| "0".to_string());
                let arr = safe_glsl_ident(ptr_name);
                let dst = format!("(_e / {n}u) * {ei}u + (_e % {n}u)");
                if self.coopmat_on() && simd {
                    // coopMatStore each 16×16 fp32 fragment to the output
                    // tile at row-major `(mi*16)*ei + ni*16`, stride ei.
                    let mf = (m / 16).max(1);
                    let nf = (n / 16).max(1);
                    writeln!(out, "{pad}barrier();").ok();
                    writeln!(out, "{pad}[[unroll]] for (uint _mi = 0u; _mi < {mf}u; _mi++)").ok();
                    writeln!(out, "{pad}[[unroll]] for (uint _ni = 0u; _ni < {nf}u; _ni++)").ok();
                    writeln!(out, "{pad}    coopMatStore(_CMAcc_{cnm}[_mi][_ni], {arr}, uint({off}) + (_mi * 16u) * {ei}u + _ni * 16u, {ei}u, gl_CooperativeMatrixLayoutRowMajor);").ok();
                    return Ok(());
                }
                let local_c = matches!(
                    self.profile.mma,
                    crate::backend::MmaStrategy::SoftwareLocalC
                );
                if local_c && simd {
                    let lw = self.profile.lane_width;
                    let per_lane = (m * n).div_ceil(lw).max(1);
                    writeln!(out, "{pad}barrier();").ok();
                    writeln!(out, "{pad}for (uint _li = 0u; _li < {per_lane}u; _li++) {{").ok();
                    writeln!(out, "{pad}    uint _e = simd_lane + _li * {lw}u;").ok();
                    writeln!(out, "{pad}    if (_e >= {}u) break;", m * n).ok();
                    writeln!(out, "{pad}    {arr}[uint({off}) + {dst}] = float(_CTC_lane_{cnm}[_li]);").ok();
                    writeln!(out, "{pad}}}").ok();
                } else {
                    let (sid, ssize, _) = coop_scope_glsl(simd, self.profile.lane_width);
                    let base = coop_base_glsl(simd, m * n);
                    writeln!(out, "{pad}barrier();").ok();
                    writeln!(
                        out,
                        "{pad}for (uint _e = {sid}; _e < {}u; _e += {ssize}) {arr}[uint({off}) + {dst}] = float(_CTC_{cnm}[{base} + _e]);",
                        m * n
                    )
                    .ok();
                }
            }
            Op::SimdgroupMatMul { a, b, c } => {
                let (ma, mb, mc) = (sgm_name(*a), sgm_name(*b), sgm_name(*c));
                writeln!(out, "{pad}subgroupMemoryBarrierShared(); subgroupBarrier();").ok();
                writeln!(out, "{pad}{{ uint _bs = simd_group * 64u;").ok();
                writeln!(out, "{pad}    float _acc0 = {mc}[_bs + _sg_fm * 8u + _sg_fn0];").ok();
                writeln!(out, "{pad}    float _acc1 = {mc}[_bs + _sg_fm * 8u + _sg_fn1];").ok();
                writeln!(out, "{pad}    for (uint _k = 0u; _k < 8u; _k++) {{").ok();
                writeln!(out, "{pad}        float _av = {ma}[_bs + _sg_fm * 8u + _k];").ok();
                writeln!(out, "{pad}        _acc0 += _av * {mb}[_bs + _k * 8u + _sg_fn0];").ok();
                writeln!(out, "{pad}        _acc1 += _av * {mb}[_bs + _k * 8u + _sg_fn1];").ok();
                writeln!(out, "{pad}    }}").ok();
                writeln!(out, "{pad}    subgroupMemoryBarrierShared(); subgroupBarrier();").ok();
                writeln!(out, "{pad}    {mc}[_bs + _sg_fm * 8u + _sg_fn0] = _acc0;").ok();
                writeln!(out, "{pad}    {mc}[_bs + _sg_fm * 8u + _sg_fn1] = _acc1;").ok();
                writeln!(out, "{pad}}}").ok();
            }
            // Control flow — nested-block recursion mirroring the CUDA walker.
            Op::Loop { var, start, end, step, body } => {
                let s = self.vname(Some(*start), block, ov);
                let e = self.vname(Some(*end), block, ov);
                let st = self.vname(Some(*step), block, ov);
                let lv = format!("i_{}", var.as_u32());
                // Same uint-cast pattern as StrideReduce — BinOps emit
                // `float` results, so any computed `start`/`end`/`step`
                // is a float SSA. Casting at the loop site avoids needing
                // full type tracking through the IR.
                writeln!(
                    out,
                    "{pad}for (uint {lv} = uint({s}); {lv} < uint({e}); {lv} += uint({st})) {{"
                )
                .ok();
                if let Some(bb) = kernel.blocks.get(body) {
                    let mut child = self.child_ov(block, ov);
                    // The macro encodes the loop induction var under TWO
                    // magic ValueIds (see msl emit_block).
                    child.insert(ValueId::new(0xC000_0000 | var.as_u32()), lv.clone());
                    child.insert(ValueId::new(var.as_u32() + 0x4000_0000), lv.clone());
                    // Loop induction var is `uint`; record both magic
                    // ValueIds so BinOps using the iter var pick the
                    // right type.
                    types.insert(ValueId::new(0xC000_0000 | var.as_u32()), "uint");
                    types.insert(ValueId::new(var.as_u32() + 0x4000_0000), "uint");
                    self.emit_ops(bb, kernel, &child, types, local_types, out)?;
                }
                writeln!(out, "{pad}}}").ok();
            }
            Op::If { cond, then_block, else_block } => {
                let c = self.vname(Some(*cond), block, ov);
                writeln!(out, "{pad}if (bool({c})) {{").ok();
                if let Some(tb) = kernel.blocks.get(then_block) {
                    let child = self.child_ov(block, ov);
                    self.emit_ops(tb, kernel, &child, types, local_types, out)?;
                }
                if let Some(ebid) = else_block {
                    writeln!(out, "{pad}}} else {{").ok();
                    if let Some(eb) = kernel.blocks.get(ebid) {
                        let child = self.child_ov(block, ov);
                        self.emit_ops(eb, kernel, &child, types, local_types, out)?;
                    }
                }
                writeln!(out, "{pad}}}").ok();
            }
            // Scalar zero / splat.
            Op::Zeros { shape, .. } if shape.rank() == 0 => {
                let v = self.vname(vid, block, ov);
                writeln!(out, "{pad}float {v} = 0.0;").ok();
            }
            Op::Splat { value, shape, .. } if shape.rank() == 0 => {
                let v = self.vname(vid, block, ov);
                writeln!(out, "{pad}float {v} = float({value});").ok();
            }
            other => {
                return Err(Error::UnsupportedOp(format!(
                    "spirv: op {} not yet supported",
                    op_variant_name(other)
                )));
            }
        }
        Ok(())
    }

    /// Find the `CoopTileSetup` config for a named GEMM op:
    /// `(m, n, k, ta, tb, tc, accumulate, simdgroup_scope)`. Mirrors the
    /// CUDA emitter so the same op shape is consumed.
    #[allow(clippy::type_complexity)]
    fn coop_cfg(
        &self,
        kernel: &Kernel,
        name: &str,
    ) -> Option<(u32, u32, u32, bool, bool, bool, bool, bool)> {
        use metaltile_core::ir::{CoopTileAccMode, CoopTileScope};
        for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
            for op in &blk.ops {
                if let Op::CoopTileSetup {
                    name: nm,
                    m,
                    n,
                    k,
                    ta,
                    tb,
                    tc,
                    acc_mode,
                    exec_scope,
                    ..
                } = op
                {
                    if nm == name {
                        return Some((
                            *m,
                            *n,
                            *k,
                            *ta,
                            *tb,
                            *tc,
                            matches!(acc_mode, CoopTileAccMode::MultiplyAccumulate),
                            matches!(exec_scope, CoopTileScope::SimdGroup),
                        ));
                    }
                }
            }
        }
        None
    }

    fn glsl_unary(&self, op: UnaryOpKind, arg: &str) -> String {
        use UnaryOpKind::*;
        match op {
            Neg => format!("(-{arg})"),
            // Route reciprocal through the Markstein-corrected divide
            // helper so `1/x` is correctly rounded (matching `Op::BinOp::Div`).
            Recip => format!("mt_fdiv(1.0, {arg})"),
            // Rsqrt → mt_rsqrt with one Newton-Raphson refinement step
            // on top of `inversesqrt`. Improves AMD's 1.4-ULP hardware
            // estimate to ≤1 ULP, which the gated-delta recurrence needs.
            Rsqrt => format!("mt_rsqrt({arg})"),
            // Block-scaled quant decode — preamble helpers; pure arithmetic
            // ports of the CUDA preamble.
            _ => format!("{}({arg})", self.profile.unary_intrinsic(op)),
        }
    }
}

fn glsl_binop(op: BinOpKind, l: &str, r: &str) -> String {
    use BinOpKind::*;
    match op {
        Add => format!("{l} + {r}"),
        Sub => format!("{l} - {r}"),
        Mul => format!("{l} * {r}"),
        Div => format!("{l} / {r}"),
        Max => format!("max({l}, {r})"),
        Min => format!("min({l}, {r})"),
        Pow => format!("pow({l}, {r})"),
        // Use the explicit `mt_atan2` helper from the preamble — AMD's
        // GLSL 2-arg `atan(y, x)` returns the wrong quadrant for some
        // inputs (off by π).
        ATan2 => format!("mt_atan2(float({l}), float({r}))"),
        Rem => format!("mod({l}, {r})"),
        Mod => format!("(uint({l}) % uint({r}))"),
        And => format!("(bool({l}) && bool({r}))"),
        Or => format!("(bool({l}) || bool({r}))"),
        Xor => format!("(bool({l}) != bool({r}))"),
        BitAnd => format!("(uint({l}) & uint({r}))"),
        BitOr => format!("(uint({l}) | uint({r}))"),
        BitXor => format!("(uint({l}) ^ uint({r}))"),
        Shl => format!("(uint({l}) << uint({r}))"),
        Shr => format!("(uint({l}) >> uint({r}))"),
        CmpLt => format!("float({l} < {r})"),
        CmpGt => format!("float({l} > {r})"),
        CmpLe => format!("float({l} <= {r})"),
        CmpGe => format!("float({l} >= {r})"),
        CmpEq => format!("float({l} == {r})"),
        CmpNe => format!("float({l} != {r})"),
    }
}

fn reduce_init(kind: ReduceKind) -> &'static str {
    match kind {
        ReduceKind::Sum | ReduceKind::Mean => "0.0",
        ReduceKind::Max => "-1.0/0.0",
        ReduceKind::Min => "1.0/0.0",
        ReduceKind::Product => "1.0",
    }
}

fn reduce_combine(kind: ReduceKind, a: &str, b: &str) -> String {
    match kind {
        ReduceKind::Sum | ReduceKind::Mean => format!("{a} + {b}"),
        ReduceKind::Max => format!("max({a}, {b})"),
        ReduceKind::Min => format!("min({a}, {b})"),
        ReduceKind::Product => format!("{a} * {b}"),
    }
}

fn op_variant_name(op: &Op) -> String {
    let dbg = format!("{op:?}");
    dbg.split([' ', '{', '(']).next().unwrap_or("?").to_string()
}

/// Look up the dtype of a named `Op::ThreadgroupAlloc` across the kernel
/// body and every named block. Returns `None` if no Alloc with that name
/// is found (callers should fall through to `float`).
fn tg_alloc_dtype(kernel: &Kernel, name: &str) -> Option<DType> {
    for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
        for op in &blk.ops {
            if let Op::ThreadgroupAlloc { dtype, name: n, .. } = op
                && n == name
            {
                return Some(*dtype);
            }
        }
    }
    None
}

/// Same as `tg_alloc_dtype` for `Op::StackAlloc`. The hadamard_m28
/// kernel stores u32 sign patterns in a stack array; loading as
/// `float` truncates the high bits.
fn stack_alloc_dtype(kernel: &Kernel, name: &str) -> Option<DType> {
    for blk in std::iter::once(&kernel.body).chain(kernel.blocks.values()) {
        for op in &blk.ops {
            if let Op::StackAlloc { dtype, name: n, .. } = op
                && n == name
            {
                return Some(*dtype);
            }
        }
    }
    None
}

/// Pick the right GLSL load expression + result type from a DType — used
/// by ThreadgroupLoad / StackLoad. Returns `(glsl_type, expr_around_arr)`.
fn dtype_load(dtype: Option<DType>, arr_idx: &str) -> (&'static str, String) {
    match dtype {
        Some(DType::U32) | Some(DType::U16) | Some(DType::U8) => {
            ("uint", format!("uint({arr_idx})"))
        }
        Some(DType::I32) | Some(DType::I8) | Some(DType::I4) | Some(DType::I64) => {
            ("int", format!("int({arr_idx})"))
        }
        _ => ("float", format!("float({arr_idx})")),
    }
}

/// Per-warp shared-tile name for a `simdgroup_matrix<f32,8,8>` value.
/// Matches the CUDA emitter's naming so the runtime side-tables stay
/// identical in shape.
fn sgm_name(v: ValueId) -> String { format!("_SGM_{}", v.as_u32()) }

/// CoopTile name sanitization, matches the CUDA emitter so the shared
/// arrays line up by spelling.
fn ct_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn coop_c_name(name: &str) -> String {
    name.strip_suffix("_acc").unwrap_or(name).to_string()
}

/// `(thread-id, scope-size, barrier)` for a cooperative iteration in
/// GLSL — same shape as the CUDA helper but with `subgroupBarrier()`.
fn coop_scope_glsl(simd: bool, lane_width: u32) -> (&'static str, String, &'static str) {
    if simd {
        ("simd_lane", format!("{lane_width}u"), "subgroupMemoryBarrierShared(); subgroupBarrier();")
    } else {
        ("tid", "lsize".to_string(), "barrier();")
    }
}

fn coop_base_glsl(simd: bool, tile: u32) -> String {
    if simd {
        format!("simd_group * {tile}u")
    } else {
        "0u".to_string()
    }
}

/// Does the kernel use any simdgroup-matrix op (drives the lane-coord
/// preamble + shared tile allocation)?
fn uses_simdgroup(kernel: &Kernel) -> bool {
    std::iter::once(&kernel.body).chain(kernel.blocks.values()).any(|b| {
        b.ops.iter().any(|op| {
            matches!(
                op,
                Op::SimdgroupAlloc { .. }
                    | Op::SimdgroupLoad { .. }
                    | Op::SimdgroupMatMul { .. }
                    | Op::SimdgroupElemLoad { .. }
                    | Op::SimdgroupElemStore { .. }
            )
        })
    })
}

/// Map a metaltile param / DSL identifier to a GLSL-safe one.
///
/// Two classes of fix:
/// - Common metaltile param names (`out`, `in`, `inout`, `length`,
///   `input`, …) collide with GLSL keywords. We prefix `_b_` so the
///   emitted SSBO array stays distinct from the keyword.
/// - GLSL forbids identifiers containing `__` (reserved for the
///   implementation). The metaltile macro emits mutable-local reads as
///   `Op::Load { src: "__ml_<name>" }`; rewrite the prefix to `mt_loc_`
///   to match the declaration emitted by `Op::DeclareLocal` /
///   `Op::SetLocal` after the same prefix swap.
///
/// Plain names like `a`, `b`, `inp`, `weights`, `xs` pass through
/// unchanged so emitted GLSL stays readable.
pub fn safe_glsl_ident(name: &str) -> String {
    // Rewrite the macro-internal mutable-local prefix.
    if let Some(rest) = name.strip_prefix("__ml_") {
        return format!("mt_loc_{rest}");
    }
    // The set of GLSL reserved words a metaltile kernel param is at all
    // likely to hit. Not exhaustive — the compiler will catch anything we
    // miss, and we can grow this on demand.
    const RESERVED: &[&str] = &[
        "in", "out", "inout", "uniform", "buffer", "shared",
        "const", "void", "bool", "int", "uint", "float", "double",
        "vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
        "true", "false", "if", "else", "for", "while", "do", "switch", "case",
        "return", "break", "continue", "discard",
        "layout", "precision", "highp", "mediump", "lowp",
        "attribute", "varying",
        "centroid", "flat", "smooth", "noperspective",
        "coherent", "volatile", "restrict", "readonly", "writeonly",
        "sampler1D", "sampler2D", "sampler3D", "samplerCube",
        "image1D", "image2D", "image3D",
        // GLSL reserves these for future use, even though they aren't
        // currently used; shaderc rejects them. The corpus hit them on
        // conv* / depthwise_conv2d / aura_value_int4 / ffai_gemm kernels.
        "input", "output", "texture", "image", "sampler",
        "active", "partition", "common", "filter",
        "row_major", "column_major", "packed",
        "asm", "class", "template", "this", "namespace", "interface",
        "public", "static", "extern", "external",
        "long", "short", "half", "fixed", "unsigned",
        "hvec2", "hvec3", "hvec4", "fvec2", "fvec3", "fvec4",
        "dvec2", "dvec3", "dvec4", "ivec2", "ivec3", "ivec4",
        "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4",
        "snorm", "unorm", "snorm8", "unorm8",
        // GLSL built-in functions / common names that surface as params:
        "length", "distance", "dot", "cross", "normalize", "reflect",
        "refract", "transpose", "determinant", "inverse",
    ];
    if RESERVED.contains(&name) {
        format!("_b_{name}")
    } else {
        name.to_string()
    }
}

impl CodegenBackend for GlslGenerator {
    fn target(&self) -> Target { Target::Spirv }

    fn profile(&self) -> &TargetProfile { &self.profile }

    fn generate(&self, kernel: &Kernel) -> Result<String> {
        // Run the same backend-neutral kernel-inline pass the CUDA emitter
        // does, so cross-kernel calls resolve in this codegen too.
        let mut inlined = kernel.clone();
        let k: &Kernel = match crate::passes::run_passes(
            &mut inlined,
            &[Box::new(crate::passes::kernel_inline::KernelInlinePass)],
        ) {
            Ok(()) => &inlined,
            Err(_) => kernel,
        };
        let mut out = String::new();
        self.emit_preamble(&mut out);
        self.emit_bindings(k, &mut out)?;
        self.emit_push_constants(k, &mut out)?;
        // Shared arrays (Op::ThreadgroupAlloc) + the implicit `_red_<vid>`
        // scratch buffers must be declared at file scope in GLSL — before
        // `main`. Walk every block to find them.
        self.emit_shared_arrays(k, &mut out);
        self.emit_signature(&mut out);
        self.emit_body(k, &mut out)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct GlslBindingPlan {
    pub bindings: Vec<GlslBinding>,
    pub push_constant_bytes: u32,
    pub has_n_elems: bool,
}

#[derive(Debug, Clone)]
pub struct GlslBinding {
    pub name: String,
    pub binding: u32,
    pub dtype: DType,
    pub is_output: bool,
    pub kind: ParamKind,
}

impl GlslGenerator {
    pub fn binding_plan(&self, kernel: &Kernel) -> Result<GlslBindingPlan> {
        let mut bindings = Vec::with_capacity(kernel.params.len());
        let mut next_binding: u32 = 0;
        for p in kernel.params.iter() {
            bindings.push(GlslBinding {
                name: p.name.clone(),
                binding: next_binding,
                dtype: p.dtype,
                is_output: p.is_output,
                kind: p.kind.clone(),
            });
            next_binding += 1;
            if matches!(p.kind, ParamKind::Strided) {
                // Two companion SSBOs (shape + strides) follow the
                // strided data buffer at the next two bindings. The
                // runtime allocates and binds them.
                bindings.push(GlslBinding {
                    name: format!("{}_shape", p.name),
                    binding: next_binding,
                    dtype: DType::U32,
                    is_output: false,
                    kind: ParamKind::Tensor,
                });
                next_binding += 1;
                bindings.push(GlslBinding {
                    name: format!("{}_strides", p.name),
                    binding: next_binding,
                    dtype: DType::U32,
                    is_output: false,
                    kind: ParamKind::Tensor,
                });
                next_binding += 1;
            }
        }
        let mut push_bytes: u32 = 0;
        for ce in &kernel.constexprs {
            push_bytes += ce.dtype.size_bytes() as u32;
        }
        let has_n_elems = kernel.mode == KernelMode::Elementwise;
        if has_n_elems {
            push_bytes += 4;
        }
        Ok(GlslBindingPlan { bindings, push_constant_bytes: push_bytes, has_n_elems })
    }
}

#[allow(dead_code)]
fn _unused(_: &Param) {}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        constexpr::ConstExpr,
        dtype::DType,
        ir::{
            BinOpKind, ConstExprDecl, IndexExpr, Kernel, KernelMode, Op, Param, ParamKind,
            ReduceKind, ValueId,
        },
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

    fn row_reduce_ir() -> Kernel {
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
    fn emits_vector_add_glsl() {
        let g = GlslGenerator::new();
        let src = g.generate(&vector_add_ir()).unwrap();
        assert!(src.contains("#version 460"));
        // 3-D local_size layout (y, z = 1 by default).
        assert!(src.contains("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;"));
        assert!(src.contains("layout(set = 0, binding = 0, scalar) readonly buffer Buf_a"));
        assert!(src.contains("if (_gtid >= pc._n_elems) return;"));
        assert!(src.contains("float v_sum_3 = v_x_1 + v_y_2;"));
    }

    #[test]
    fn emits_row_reduce_glsl() {
        let g = GlslGenerator::new();
        let src = g.generate(&row_reduce_ir()).unwrap();
        // No _n_elems guard in Reduction mode.
        assert!(!src.contains("_n_elems"));
        // Reduction preamble.
        assert!(src.contains("uint tgid_x = gl_WorkGroupID.x;"));
        // tid is the collapsed 3-D local-invocation index.
        assert!(src.contains("uint tid = gl_LocalInvocationID.z"));
        // Per-thread grid-stride accumulator with uint-cast bounds.
        assert!(src.contains("for (uint _i = uint(v_rs_2) + tid; _i < uint(v_re_3); _i += lsize)"));
        // Workgroup-shared reduction scratch + tree, sized to local_size product.
        assert!(src.contains("shared float _red_5[256];"));
        assert!(src.contains("_red_5[tid] = (v_acc_4);"));
        assert!(src.contains("for (uint _s = 256u / 2u; _s > 0u; _s >>= 1)"));
        // npot-safe guard: tree must drop upper-partial lanes for non-pow2 TPG.
        assert!(src.contains("tid + _s < 256u"));
        assert!(src.contains("barrier();"));
        assert!(src.contains("_b_out[uint(v_row_0)] = float(v_result_5);"));
    }

    #[test]
    fn binding_plan_matches_emitted_source() {
        let g = GlslGenerator::new();
        let plan = g.binding_plan(&vector_add_ir()).unwrap();
        assert_eq!(plan.bindings.len(), 3);
        assert_eq!(plan.bindings[0].binding, 0);
        assert!(plan.has_n_elems);
        assert_eq!(plan.push_constant_bytes, 4);
    }

    #[test]
    fn binding_plan_for_reduction_has_no_n_elems() {
        let g = GlslGenerator::new();
        let plan = g.binding_plan(&row_reduce_ir()).unwrap();
        assert_eq!(plan.bindings.len(), 2);
        assert!(!plan.has_n_elems);
        // One u32 constexpr (`n`), 4 bytes.
        assert_eq!(plan.push_constant_bytes, 4);
    }

    #[test]
    fn glsl_generator_reports_spirv_target() {
        let g = GlslGenerator::new();
        assert_eq!(g.target(), Target::Spirv);
        assert_eq!(g.profile().shared_mem_kw, "shared");
    }

    #[test]
    fn fp16_kernel_emits_float16t() {
        // Phase 3: f16 is now a supported dtype (kernel must declare the
        // GL_EXT_shader_explicit_arithmetic_types_float16 extension; the
        // preamble does so automatically).
        let mut k = vector_add_ir();
        for p in &mut k.params {
            p.dtype = DType::F16;
        }
        let src = GlslGenerator::new().generate(&k).unwrap();
        assert!(src.contains("float16_t"));
        assert!(src.contains("#extension GL_EXT_shader_explicit_arithmetic_types_float16"));
    }
}

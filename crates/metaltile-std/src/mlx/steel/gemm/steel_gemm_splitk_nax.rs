//! `mt_steel_gemm_splitk_nax` — split-K GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the two-kernel split-K GEMM
//! `steel_gemm_splitk`. Gated behind the `nax` Cargo feature — the
//! kernel requires the Metal 4 `MetalPerformancePrimitives` framework
//! (macOS 26+); codegen emits the framework include when it detects the
//! `mpp::` marker in the `Op::InlineMsl` body.
//!
//! Split-K partitions the K dimension across the grid z-axis so a
//! skinny-M / skinny-N matmul with a very large K still saturates the
//! GPU. It is a **two-kernel** dispatch:
//!
//!   1. `mt_steel_gemm_splitk_nax_{T}` — each K-split computes a partial
//!      `[M, N]` product over its slice of K via cooperative `matmul2d`
//!      and writes it (fp32) to a `[n_splits, M, N]` partials buffer.
//!   2. `mt_steel_gemm_splitk_accum_nax` — reduces the `n_splits`
//!      partial `[M, N]` matrices into the final `[M, N]` output. One
//!      thread per output element, plain sum.
//!
//! The split-K kernel is exactly `mt_steel_gemm_fused_nax` with a 3-D
//! grid: `tgid_z` selects the K-split and the K-loop walks only this
//! split's `[k_start, k_end)` range. The accumulator is **fp32** so the
//! cross-split sum keeps full precision for f16 inputs — the partials
//! tensor is `T = f32` regardless of the operand dtype. The inter-kernel
//! handoff is an ordinary device buffer: kernel 1 writes partials,
//! kernel 2 reads them.
//!
//! Geometry mirrors `mt_steel_gemm_fused_nax`:
//!
//!   tpg = 128 = 4 SG × 32 lanes (WM = WN = 2)
//!   BM = BN = BK = 32 → 32×32 output tile (1024 outputs/TG)
//!   Grid: [N/32, M/32, n_splits]
//!   Per SG: one 16×16×32 `matmul2d` per K-block (acc-mode multiply_accumulate)
//!
//! ## DISPATCH INVARIANTS — split-K kernel
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM=WN=2). Fixed.
//! - **Grid: `[n/32, m/32, n_splits]`** — `tgid_x` = N-block,
//!   `tgid_y` = M-block, `tgid_z` = K-split index.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`** — all loads are
//!   unconditional; ragged shapes read out of bounds. Callers must pad.
//! - **`k_per_split % 32 == 0`, `n_splits * k_per_split == k`** — the
//!   K-loop is clamped to `k` so the last split may legally over-run.
//! - **`partials` is fp32, length `n_splits * m * n`**, laid out
//!   `[split, M, N]` row-major.
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index, not the global thread index.
//!
//! ## DISPATCH INVARIANTS — accum kernel
//!
//! - **One thread per `[M, N]` output element** — Grid3D / Elementwise.
//! - **`partials` length `n_splits * m * n` (fp32)**, `out` length `m * n`.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_splitk_nax_gpu_correctness.rs`.

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind},
    shape::{Dim, Shape},
};

/// Tile geometry — keep in lock-step with the inline MSL below.
pub const BM: u32 = 32;
pub const BN: u32 = 32;
pub const BK: u32 = 32;
/// Threads per group (4 SG × 32 lanes).
pub const TPG: u32 = 128;
/// Threadgroup-mem row skew. Stride = BK + 4 = 36.
pub const TG_SKEW: u32 = 4;
pub const TG_LD: u32 = BK + TG_SKEW; // 36

/// MSL source — split-K partial GEMM. Codegen emits the bindings as
/// `const device {T} *a/b` + `device float *partials` + `constant uint
/// &k/n/k_per_split` per the standard signature path. Templated on `T`.
const GEMM_SPLITK_NAX_SRC_TEMPLATE: &str = r#"// --- mt_steel_gemm_splitk_nax body (BM=BN=BK=32, TG=128, 4 SGs WM=WN=2) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 32;
constexpr uint BN = 32;
constexpr uint BK = 32;
constexpr uint TG_LD = 36;     // BK + 4 skew

// Threadgroup tiles — Xs holds A in (m, k) row-major; Ws holds B in
// (n, k) row-major (so MPP with tb=true reads it as the K×N operand).
threadgroup {T} Xs[BM * TG_LD];
threadgroup {T} Ws[BN * TG_LD];

// Per-TG output tile origin in (m, n); tgid_z = K-split index.
const uint m_tile = tgid_y;
const uint n_tile = tgid_x;
const uint split  = tgid_z;
const uint lane_in_tg = simd_group * 32u + simd_lane;
// 4 SGs in a 2×2 WM=WN=2 warp grid.
const uint sm = simd_group / 2u;
const uint sn = simd_group & 1u;

// ── A coop-load mapping (128 lanes × 8 contiguous K) ──
const uint x_m_row  = lane_in_tg / 4u;
const uint x_k_quad = lane_in_tg & 3u;
const uint x_k_base = x_k_quad * 8u;

// ── B coop-load mapping (transpose [K,N] → Ws[n_row, k]) ──
const uint w_n_row  = lane_in_tg / 4u;
const uint w_k_quad = lane_in_tg & 3u;
const uint w_k_base = w_k_quad * 8u;

const uint x_m_base = m_tile * 32u;
const uint w_n_base = n_tile * 32u;

// ── This split's K-range — [k_start, k_end), clamped to k ──
const uint k_start   = split * k_per_split;
const uint k_end_raw = k_start + k_per_split;
const uint k_end     = (k_end_raw < k) ? k_end_raw : k;

// ── Set up MPP matmul: (M=16, N=16, K=32), ta=false, tb=true, tc=false ──
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/16, /*K=*/32,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op.template get_left_input_cooperative_tensor<{T}, {T}, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<{T}, {T}, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

// Zero accumulator (mode = multiply_accumulate adds to dst on each run()).
for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    ct_c[i] = 0.0f;
}

// Per-SG sub-tile origin inside the 32×32 TG tile.
const uint sg_m_base = sm * 16u;
const uint sg_n_base = sn * 16u;

for (uint kb = k_start; kb < k_end; kb += BK) {
    // ── 1. Coop A load (128 lanes × 8 contiguous K) ──
    const uint a_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
    const uint x_ws_base = x_m_row * TG_LD + x_k_base;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        Xs[x_ws_base + i] = ({T})a[a_row_dev_base + i];
    }

    // ── 2. Coop B load — B[k, n] → Ws[n_row, k] (transpose into TG) ──
    const uint ws_base = w_n_row * TG_LD + w_k_base;
    const uint b_n = w_n_base + w_n_row;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        const uint b_k = kb + w_k_base + i;
        Ws[ws_base + i] = ({T})b[b_k * n + b_n];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. Build per-SG tensor views over the TG tiles ──
    threadgroup {T}* xs_sg = Xs + sg_m_base * TG_LD;
    threadgroup {T}* ws_sg = Ws + sg_n_base * TG_LD;
    metal::tensor<threadgroup {T}, metal::extents<int, TG_LD, 16>, metal::tensor_inline>
        tA(xs_sg, metal::extents<int, TG_LD, 16>{});
    metal::tensor<threadgroup {T}, metal::extents<int, TG_LD, 16>, metal::tensor_inline>
        tB(ws_sg, metal::extents<int, TG_LD, 16>{});

    ct_a.load(tA);
    ct_b.load(tB);

    // ── 4. Run the matmul; ct_c accumulates ──
    gemm_op.run(ct_a, ct_b, ct_c);

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ── 5. Store ct_c to this split's fp32 partial slab [split, M, N] ──
const uint out_m_base = m_tile * 32u + sg_m_base;
const uint out_n_base = n_tile * 32u + sg_n_base;
const uint part_base  = split * m * n;
// Per-SG fp32 scratch (4 SG × 256 floats = 4 KB).
threadgroup float OutScratch[4 * 16 * 16];
threadgroup float* sg_scratch = OutScratch + simd_group * (16 * 16);
metal::tensor<threadgroup float, metal::extents<int, 16, 16>, metal::tensor_inline>
    tC(sg_scratch, metal::extents<int, 16, 16>{});
ct_c.store(tC);
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint lane = simd_lane;
// Map lane → (row, col): 32 lanes × 8 elems = 256 outputs.
const uint o_row = lane / 2u;
const uint o_col_base = (lane & 1u) * 8u;
#pragma clang loop unroll(full)
for (uint i = 0u; i < 8u; ++i) {
    partials[part_base + (out_m_base + o_row) * n + (out_n_base + o_col_base + i)] =
        sg_scratch[o_row * 16u + o_col_base + i];
}
#else
// Pre-Metal-4 fallback — silence the bindings so the metallib still links.
// Correctness test on such targets is the intended failure signal.
if (simd_group == 0u && simd_lane == 0u) {
    const uint o = tgid_z * m * n + tgid_y * 32u * n + tgid_x * 32u;
    const uint _ks = k_per_split; // silence unused-var
    partials[o] = (float)a[0] * (float)b[0] * (float)(_ks + k);
}
#endif
"#;

/// MSL source — split-K accumulation pass. One thread per `[M, N]`
/// output element, plain sum across the `n_splits` partial slabs.
/// Codegen emits `const device float *partials` + `device {T} *out` +
/// `constant uint &m/n/n_splits`. Templated on `T`.
const GEMM_SPLITK_ACCUM_NAX_SRC_TEMPLATE: &str = r#"// --- mt_steel_gemm_splitk_accum_nax body ---
// One thread per [M, N] output element. `tgid_x` is the global flat
// index (Elementwise grid sized to m * n by the dispatch).
const uint idx = tgid_x;
const uint total = m * n;
if (idx < total) {
    float acc = 0.0f;
    for (uint s = 0u; s < n_splits; ++s) {
        acc += partials[s * total + idx];
    }
    out[idx] = ({T})acc;
}
"#;

/// Substitute the `{T}` placeholder for the per-dtype MSL source.
fn substitute_dtype(src: &str, dt: DType) -> String {
    let t = match dt {
        DType::F32 => "float",
        DType::F16 => "half",
        _ => unreachable!("kernel_ir_for asserts dtype before reaching here"),
    };
    src.replace("{T}", t)
}

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_splitk_nax_{T}` —
/// the split-K partial GEMM (pass 1).
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = a            const device {T}    *
///   buffer(1) = b            const device {T}    *
///   buffer(2) = partials     device       float  *
///   buffer(3) = k            constant     uint   &
///   buffer(4) = n            constant     uint   &
///   buffer(5) = k_per_split  constant     uint   &
///
/// Dispatch geometry: grid `[n/32, m/32, n_splits]`, threadgroup `[128, 1, 1]`.
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_steel_gemm_splitk_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_steel_gemm_splitk_nax");
    k.mode = KernelMode::Reduction;

    k.params.push(Param {
        name: "a".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "b".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    // Partials buffer is always fp32 — the accumulator dtype — so the
    // cross-split sum keeps full precision for f16 inputs.
    k.params.push(Param {
        name: "partials".into(),
        dtype: DType::F32,
        shape: Shape::new([Dim::Any, Dim::Any, Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    // `m` — the partials buffer is laid out [n_splits, m, n]; the MSL
    // computes `part_base = split * m * n`, so `m` must be a bound
    // constexpr (the pass-1 kernel previously omitted it).
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("m"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("k_per_split"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any, Dim::Any]));

    // Force `tgid_y` + `tgid_z` alias emission — InlineMsl source
    // mentions them but the body text isn't scanned for the alias
    // trigger; codegen only looks at IR ops. Use the
    // `Op::Load { src: "tgid_*" }` direct-identifier form (see
    // `steel_gemm_fused_nax`). Reduction mode emits `tgid_x`
    // unconditionally, so axis=0 needs no hint.
    use metaltile_core::ir::ValueId;
    let mut body = Block::new(BlockId::new(0));
    body.push_op(
        Op::Load { src: "tgid_y".to_string(), indices: Vec::new(), mask: None, other: None },
        ValueId::new(0),
    );
    body.push_op(
        Op::Load { src: "tgid_z".to_string(), indices: Vec::new(), mask: None, other: None },
        ValueId::new(1),
    );
    body.push_op_no_result(Op::InlineMsl {
        source: substitute_dtype(GEMM_SPLITK_NAX_SRC_TEMPLATE, dt),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });
    k.body = body;
    // #140 made `Kernel::blocks` an `FxHashMap`; `sync_entry_block` keeps
    // the entry-block entry in sync with `body` after a manual InlineMsl
    // body construction.
    k.sync_entry_block();

    k
}

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_splitk_accum_nax_{T}` —
/// the split-K partial-sum reduction (pass 2).
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = partials   const device float *
///   buffer(1) = out        device       {T}   *
///   buffer(2) = m          constant     uint  &
///   buffer(3) = n          constant     uint  &
///   buffer(4) = n_splits   constant     uint  &
///
/// Dispatch geometry: grid `[m * n, 1, 1]`, threadgroup `[1, 1, 1]` —
/// one thread per output element.
pub fn accum_kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_steel_gemm_splitk_accum_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_steel_gemm_splitk_accum_nax");
    k.mode = KernelMode::Reduction;

    k.params.push(Param {
        name: "partials".into(),
        dtype: DType::F32,
        shape: Shape::new([Dim::Any, Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("m"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n_splits"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // The accum body only uses `tgid_x` (the flat output index), which
    // Reduction mode emits unconditionally — no alias hint needed.
    let mut body = Block::new(BlockId::new(0));
    body.push_op_no_result(Op::InlineMsl {
        source: substitute_dtype(GEMM_SPLITK_ACCUM_NAX_SRC_TEMPLATE, dt),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });
    k.body = body;
    // #140 made `Kernel::blocks` an `FxHashMap`; `sync_entry_block` keeps
    // the entry-block entry in sync with `body` after a manual InlineMsl
    // body construction.
    k.sync_entry_block();

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_ir_constructs_for_f32_and_f16() {
        for dt in [DType::F32, DType::F16] {
            let k = kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_splitk_nax");
            assert_eq!(k.params.len(), 3);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[1].name, "b");
            assert_eq!(k.params[2].name, "partials");
            assert!(k.params[2].is_output);
            assert_eq!(k.params[2].dtype, DType::F32);
            assert_eq!(k.constexprs.len(), 4);
            assert_eq!(k.constexprs[0].name.name(), "m");
            assert_eq!(k.constexprs[1].name.name(), "k");
            assert_eq!(k.constexprs[2].name.name(), "n");
            assert_eq!(k.constexprs[3].name.name(), "k_per_split");
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(
                k.body.ops.iter().any(|op| matches!(op, Op::Load { src, .. } if src == "tgid_y"))
            );
            assert!(
                k.body.ops.iter().any(|op| matches!(op, Op::Load { src, .. } if src == "tgid_z"))
            );
        }
    }

    #[test]
    fn accum_kernel_ir_constructs_for_f32_and_f16() {
        for dt in [DType::F32, DType::F16] {
            let k = accum_kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_splitk_accum_nax");
            assert_eq!(k.params.len(), 2);
            assert_eq!(k.params[0].name, "partials");
            assert_eq!(k.params[0].dtype, DType::F32);
            assert_eq!(k.params[1].name, "out");
            assert!(k.params[1].is_output);
            assert_eq!(k.constexprs.len(), 3);
            assert_eq!(k.constexprs[0].name.name(), "m");
            assert_eq!(k.constexprs[1].name.name(), "n");
            assert_eq!(k.constexprs[2].name.name(), "n_splits");
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
        }
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half")] {
            let mut k = kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            k.name = format!("mt_steel_gemm_splitk_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_splitk_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
            assert!(msl.contains("tgid_z"), "tgid_z must be bound:\n{msl}");
        }
    }

    #[test]
    fn codegen_emits_accum_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for dt in [DType::F32, DType::F16] {
            let mut k = accum_kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            k.name = format!("mt_steel_gemm_splitk_accum_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_splitk_accum_nax_{suffix}")));
            assert!(msl.contains("n_splits"));
        }
    }
}

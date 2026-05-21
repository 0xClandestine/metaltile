//! `mt_steel_gemm_gather_nax` — gather GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the `nn` steel gather-GEMM
//! `C = A_gathered · B_gathered`:
//!
//!   - `lhs_indices[out_row]` — one `u32` per output row; redirects each
//!     output row to a (non-contiguous) `A` source row.
//!   - `rhs_indices[n_block]` — one `u32` per `BN`-wide N-block; selects
//!     which `[K, N]` `B` matrix this output block multiplies against.
//!     Selected matrix base = `rhs_indices[n_tile] * k * n`.
//!
//! This is the MLX `gather_mm` op — the dense-matmul half of a MoE FFN.
//! Gated behind the `nax` Cargo feature (Metal 4 / macOS 26+).
//!
//! Cooperative-tensor counterpart of `steel_gemm_gather`. It is exactly
//! `mt_steel_gemm_fused_nax` with two extra integer loads before the
//! address arithmetic — the gather index of an output row is a per-row
//! scalar, the B-matrix index a per-N-block scalar. No new codegen
//! primitive is required; the redirection is ordinary arithmetic.
//!
//! Geometry mirrors `mt_qmm_nax` / `mt_steel_gemm_fused_nax`:
//!
//!   tpg = 128 = 4 SG × 32 lanes (WM = WN = 2)
//!   BM = BN = BK = 32 → 32×32 output tile
//!   Grid: [N/32, M/32, 1]
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM=WN=2). Fixed.
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`** — all loads
//!   unconditional; callers must pad.
//! - **`lhs_indices` length `m`** (one gathered `A`-row per output row),
//!   `u32`, each `< n_a_rows`. **`rhs_indices` length `n/32`** (one
//!   selected `B`-matrix per N-block), `u32`, each `< n_b_mats`. No
//!   bounds-check — callers keep indices in range.
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_gather_nax_gpu_correctness.rs`.

use std::collections::BTreeMap;

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

/// MSL source. Codegen emits bindings `const device {T} *a/b`,
/// `const device uint *lhs_indices/rhs_indices`, `device {T} *out`,
/// `constant uint &k/n`. Templated on `T` via `{T}`.
const GEMM_GATHER_NAX_SRC_TEMPLATE: &str = r#"// --- mt_steel_gemm_gather_nax body (BM=BN=BK=32, TG=128, 4 SGs WM=WN=2) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 32;
constexpr uint BN = 32;
constexpr uint BK = 32;
constexpr uint TG_LD = 36;     // BK + 4 skew

// Threadgroup tiles — Xs holds gathered A in (m, k) row-major; Ws holds
// the gathered B matrix in (n, k) row-major (tb=true reads it as K×N).
threadgroup {T} Xs[BM * TG_LD];
threadgroup {T} Ws[BN * TG_LD];

const uint m_tile = tgid_y;
const uint n_tile = tgid_x;
const uint lane_in_tg = simd_group * 32u + simd_lane;
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

// ── Gather: redirect rows / select the B matrix ──
// lhs_indices[out_row] picks the A source row for each output row.
// rhs_indices[n_tile] picks the [K,N] B matrix for this N-block; base
// element offset into the flat `b` operand is index * k * n.
const uint a_src_row = lhs_indices[x_m_base + x_m_row];
const uint b_mat     = rhs_indices[n_tile];
const uint b_base    = b_mat * k * n;

// ── Set up MPP matmul: (M=16, N=16, K=32), ta=false, tb=true, tc=false ──
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/16, /*K=*/32,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op.template get_left_input_cooperative_tensor<{T}, {T}, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<{T}, {T}, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    ct_c[i] = 0.0f;
}

const uint sg_m_base = sm * 16u;
const uint sg_n_base = sn * 16u;

for (uint kb = 0u; kb < k; kb += BK) {
    // ── 1. Coop A load — gathered row a_src_row ──
    const uint a_row_dev_base = a_src_row * k + kb + x_k_base;
    const uint x_ws_base = x_m_row * TG_LD + x_k_base;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        Xs[x_ws_base + i] = ({T})a[a_row_dev_base + i];
    }

    // ── 2. Coop B load — gathered matrix b_base, [k,n] → Ws[n_row, k] ──
    const uint ws_base = w_n_row * TG_LD + w_k_base;
    const uint b_n = w_n_base + w_n_row;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        const uint b_k = kb + w_k_base + i;
        Ws[ws_base + i] = ({T})b[b_base + b_k * n + b_n];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup {T}* xs_sg = Xs + sg_m_base * TG_LD;
    threadgroup {T}* ws_sg = Ws + sg_n_base * TG_LD;
    metal::tensor<threadgroup {T}, metal::extents<int, TG_LD, 16>, metal::tensor_inline>
        tA(xs_sg, metal::extents<int, TG_LD, 16>{});
    metal::tensor<threadgroup {T}, metal::extents<int, TG_LD, 16>, metal::tensor_inline>
        tB(ws_sg, metal::extents<int, TG_LD, 16>{});

    ct_a.load(tA);
    ct_b.load(tB);

    gemm_op.run(ct_a, ct_b, ct_c);

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ── Store ct_c to global out — destination row is the *contiguous*
//    output row, not the gathered A row.
const uint out_m_base = m_tile * 32u + sg_m_base;
const uint out_n_base = n_tile * 32u + sg_n_base;
threadgroup float OutScratch[4 * 16 * 16];
threadgroup float* sg_scratch = OutScratch + simd_group * (16 * 16);
metal::tensor<threadgroup float, metal::extents<int, 16, 16>, metal::tensor_inline>
    tC(sg_scratch, metal::extents<int, 16, 16>{});
ct_c.store(tC);
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint lane = simd_lane;
const uint o_row = lane / 2u;
const uint o_col_base = (lane & 1u) * 8u;
#pragma clang loop unroll(full)
for (uint i = 0u; i < 8u; ++i) {
    out[(out_m_base + o_row) * n + (out_n_base + o_col_base + i)] =
        ({T})sg_scratch[o_row * 16u + o_col_base + i];
}
#else
// Pre-Metal-4 fallback — silence the bindings so the metallib still links.
if (simd_group == 0u && simd_lane == 0u) {
    const uint o = tgid_y * 32u * n + tgid_x * 32u;
    const uint _k = k + lhs_indices[0] + rhs_indices[0]; // silence unused-var
    out[o] = ({T})((float)a[0] * (float)b[0]) * ({T})_k;
}
#endif
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

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_gather_nax_{T}`.
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = a            const device {T}  *
///   buffer(1) = b            const device {T}  *
///   buffer(2) = lhs_indices  const device uint *
///   buffer(3) = rhs_indices  const device uint *
///   buffer(4) = out          device       {T}  *
///   buffer(5) = k            constant     uint &
///   buffer(6) = n            constant     uint &
///
/// Dispatch geometry: grid `[n/32, m/32, 1]`, threadgroup `[128, 1, 1]`.
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_steel_gemm_gather_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_steel_gemm_gather_nax");
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
    k.params.push(Param {
        name: "lhs_indices".into(),
        dtype: DType::U32,
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "rhs_indices".into(),
        dtype: DType::U32,
        shape: Shape::new([Dim::Any]),
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

    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("k"),
        dtype: DType::U32,
        value: None,
    });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // Force `tgid_y` alias emission — see `quantized_nax` / `steel_gemm_fused_nax`.
    use metaltile_core::ir::ValueId;
    let mut body = Block::new(BlockId::new(0));
    body.push_op(
        Op::Load { src: "tgid_y".to_string(), indices: Vec::new(), mask: None, other: None },
        ValueId::new(0),
    );
    body.push_op_no_result(Op::InlineMsl {
        source: substitute_dtype(GEMM_GATHER_NAX_SRC_TEMPLATE, dt),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });
    k.body = body.clone();
    let mut blocks = BTreeMap::new();
    blocks.insert(BlockId::new(0), body);
    k.blocks = blocks;

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_ir_constructs_for_f32_and_f16() {
        for dt in [DType::F32, DType::F16] {
            let k = kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_gather_nax");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[1].name, "b");
            assert_eq!(k.params[2].name, "lhs_indices");
            assert_eq!(k.params[3].name, "rhs_indices");
            assert_eq!(k.params[4].name, "out");
            assert!(k.params[4].is_output);
            assert_eq!(k.constexprs.len(), 2);
            assert_eq!(k.constexprs[0].name.name(), "k");
            assert_eq!(k.constexprs[1].name.name(), "n");
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::Load { src, .. } if src == "tgid_y")));
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
            k.name = format!("mt_steel_gemm_gather_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_gather_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
            assert!(msl.contains("lhs_indices"));
            assert!(msl.contains("rhs_indices"));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
        }
    }
}

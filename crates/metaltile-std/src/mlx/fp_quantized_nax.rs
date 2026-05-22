//! `mt_fp_qmm_nax` — fp4 (E2M1) quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the fp4 quantized matmul from MLX
//! `metal/kernels/fp_quantized_nax.metal`. Gated behind the `nax` Cargo
//! feature — the kernel requires the Metal 4 `MetalPerformancePrimitives`
//! framework (macOS 26+); codegen emits the framework include when it
//! detects the `mpp::` marker in the `Op::InlineMsl` body.
//!
//! This is the fp4 counterpart of `mt_qmm_nax`. It mirrors the same
//! algorithm exactly — packed weights dequantized into threadgroup
//! memory once per K-block, then one cooperative `matmul2d` per
//! simdgroup per K-block against the fp `T` X-tile — but swaps the int4
//! nibble-dequant for an **fp4 E2M1 codebook lookup**:
//!
//!   - Each 4-bit code is `[sign : 1][magnitude : 3]`.
//!   - The 3-bit magnitude indexes the E2M1 codebook
//!     `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (the nvfp4 levels — see
//!     MLX `fp4.h`).
//!   - The sign bit (`code & 8`) negates the magnitude.
//!   - The dequantized value is `scale * codebook[code & 7] *
//!     (code & 8 ? -1 : +1)`. fp4 quantization is **scale-only** — no
//!     per-group bias, unlike the affine int4 path.
//!
//! 8 fp4 codes pack into one `u32`; one `u32` covers a full BK=32 step
//! at 8 codes... no — 8 codes per pack, 4 packs per BK-block row. The
//! per-K-block scale layout uses `GROUP_SIZE = 32` so exactly one group
//! covers each BK-block — the simplest faithful fp4 group geometry.
//!
//! Built as an IR escape-hatch via `Op::InlineMsl` rather than the
//! `#[kernel]` macro because the macro front-end does not expose
//! `mpp::` types. Geometry mirrors `mt_qmm_nax`:
//!
//!   tpg = 128 = 4 SG × 32 lanes (WM = WN = 2)
//!   BM = BN = BK = 32 → 32×32 output tile (1024 outputs/TG)
//!   Grid: [N/32, M/32, 1]
//!   Per SG: one 16×16×32 `matmul2d` per K-block (acc-mode multiply_accumulate)
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes, WM=WN=2). Fixed.
//! - **Grid: `[n/32, m/32, 1]`** — `tgid_x` = N-block, `tgid_y` = M-block.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`** — all loads are
//!   unconditional; ragged shapes read out of bounds. Callers must pad.
//! - **`w` is `u32`, 8 fp4 codes per pack**, `n * k / 8` packs, laid out
//!   `[N, K/8]` row-major (qmm_t weight layout).
//! - **`scales` length `n * (k / 32)`** (`GROUP_SIZE = 32`), one fp `T`
//!   scale per `[N-row, K-group]`.
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index, not the global thread index.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/fp_quantized_nax_gpu_correctness.rs`.


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
/// fp4 quantization group size — one group per BK-block.
pub const GROUP_SIZE: u32 = 32;

/// MSL source. Codegen emits the bindings as `const device uint *w`
/// + `const device {T} *scales/x` + `device {T} *out` + `constant uint
/// &k/n/gs_per_row`. Templated on `T` via `{T}`.
const FP_QMM_NAX_SRC_TEMPLATE: &str = r#"// --- mt_fp_qmm_nax body (BM=BN=BK=32, TG=128, 4 SGs WM=WN=2) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BM = 32;
constexpr uint BN = 32;
constexpr uint BK = 32;
constexpr uint TG_LD = 36;     // BK + 4 skew
constexpr uint GROUP_SIZE = 32;

// fp4 E2M1 magnitude codebook — the nvfp4 levels (MLX `fp4.h`). The
// 3-bit magnitude (code & 7) indexes this; bit 3 is the sign.
constexpr float FP4_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Threadgroup tiles — Xs holds X in (m, k) row-major; Ws holds dequant W
// in (n, k) row-major (qmm_t layout). Skew of 4 elements past BK breaks
// the 32-bank conflict on the column-strided frag loads inside matmul2d.
threadgroup {T} Xs[BM * TG_LD];
threadgroup {T} Ws[BN * TG_LD];

// Per-TG output tile origin in (m, n).
const uint m_tile = tgid_y;
const uint n_tile = tgid_x;
const uint lane_in_tg = simd_group * 32u + simd_lane;
// 4 SGs in a 2×2 WM=WN=2 warp grid.
const uint sm = simd_group / 2u;
const uint sn = simd_group & 1u;

// ── X coop-load mapping (128 lanes × 8 contiguous K) ──
const uint x_m_row  = lane_in_tg / 4u;
const uint x_k_quad = lane_in_tg & 3u;
const uint x_k_base = x_k_quad * 8u;

// ── W coop-dequant mapping ──
// 128 packs / 128 lanes = 1 pack per lane. lane_in_tg = w_row*4 + pack_in_row.
const uint w_row       = lane_in_tg / 4u;
const uint pack_in_row = lane_in_tg & 3u;

const uint x_m_base = m_tile * 32u;
const uint w_n_base = n_tile * 32u;
const uint packs_per_row = k / 8u;
const uint sb_base = (w_n_base + w_row) * gs_per_row;
const uint w_pack_row_base = (w_n_base + w_row) * packs_per_row;

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

const uint sg_m_base = sm * 16u;
const uint sg_n_base = sn * 16u;

for (uint kb = 0u; kb < k; kb += BK) {
    // ── 1. Coop X load (128 lanes × 8 contiguous K) ──
    const uint x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
    const uint x_ws_base = x_m_row * TG_LD + x_k_base;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        Xs[x_ws_base + i] = ({T})x[x_row_dev_base + i];
    }

    // ── 2. Coop W fp4-dequant — 1 pack/lane → 8 fp {T} into Ws ──
    const uint pack_k_off = kb / 8u + pack_in_row;
    const uint pack = w[w_pack_row_base + pack_k_off];
    const uint k_off = kb + pack_in_row * 8u;
    const uint g = k_off / GROUP_SIZE;
    const float s = (float)scales[sb_base + g];
    const uint ws_base = w_row * TG_LD + pack_in_row * 8u;
    // 8 fp4 codes packed little-end-first into the u32.
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        const uint code = (pack >> (i * 4u)) & 15u;
        const float mag  = FP4_LUT[code & 7u];
        const float sgn  = (code & 8u) ? -1.0f : 1.0f;
        Ws[ws_base + i] = ({T})(s * sgn * mag);
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

// ── 5. Store ct_c to global out (cast fp32 → {T}) ──
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
// Correctness test on such targets is the intended failure signal.
if (simd_group == 0u && simd_lane == 0u) {
    const uint o = tgid_y * 32u * n + tgid_x * 32u;
    const uint _gs = gs_per_row; // silence unused-var
    out[o] = ({T})((float)x[0] * (float)scales[0]) * ({T})(w[0] & 7u) * ({T})_gs;
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

/// Build the per-dtype [`Kernel`] IR for `mt_fp_qmm_nax_{T}`.
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = w          const device uint  *
///   buffer(1) = scales     const device {T}   *
///   buffer(2) = x          const device {T}   *
///   buffer(3) = out        device       {T}   *
///   buffer(4) = k          constant     uint  &
///   buffer(5) = n          constant     uint  &
///   buffer(6) = gs_per_row constant     uint  &
///
/// Dispatch geometry: grid `[n/32, m/32, 1]`, threadgroup `[128, 1, 1]`.
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_fp_qmm_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_fp_qmm_nax");
    k.mode = KernelMode::Reduction;

    k.params.push(Param {
        name: "w".into(),
        dtype: DType::U32,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "scales".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "x".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
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
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("gs_per_row"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // Force `tgid_y` alias emission — InlineMsl source mentions `tgid_y`
    // but the body text isn't scanned for the alias trigger; codegen
    // only looks at IR ops. Use the `Op::Load { src: "tgid_y" }`
    // direct-identifier form (see `quantized_nax`).
    use metaltile_core::ir::ValueId;
    let mut body = Block::new(BlockId::new(0));
    body.push_op(
        Op::Load { src: "tgid_y".to_string(), indices: Vec::new(), mask: None, other: None },
        ValueId::new(0),
    );
    body.push_op_no_result(Op::InlineMsl {
        source: substitute_dtype(FP_QMM_NAX_SRC_TEMPLATE, dt),
        inputs: Vec::new(),
        outputs: Vec::new(),
    });
    k.body = body;
    // #140 made `Kernel::blocks` an `FxHashMap`; `sync_entry_block` is the
    // post-refactor idiom for keeping the entry-block entry in sync with
    // `body` after a manual `InlineMsl` body construction.
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
            assert_eq!(k.name, "mt_fp_qmm_nax");
            assert_eq!(k.params.len(), 4);
            assert_eq!(k.params[0].name, "w");
            assert_eq!(k.params[0].dtype, DType::U32);
            assert_eq!(k.params[1].name, "scales");
            assert_eq!(k.params[2].name, "x");
            assert_eq!(k.params[3].name, "out");
            assert!(k.params[3].is_output);
            assert_eq!(k.constexprs.len(), 3);
            assert_eq!(k.constexprs[0].name.name(), "k");
            assert_eq!(k.constexprs[1].name.name(), "n");
            assert_eq!(k.constexprs[2].name.name(), "gs_per_row");
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
            k.name = format!("mt_fp_qmm_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_fp_qmm_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
            assert!(msl.contains("FP4_LUT"));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
        }
    }
}

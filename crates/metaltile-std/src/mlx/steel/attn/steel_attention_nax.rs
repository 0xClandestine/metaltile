//! `mt_sdpa_prefill_nax` — flash attention via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the steel flash-attention prefill
//! kernel. Gated behind the `nax` Cargo feature — the kernel requires
//! the Metal 4 `MetalPerformancePrimitives` framework (macOS 26+);
//! codegen emits the framework include when it detects the `mpp::`
//! marker in the `Op::InlineMsl` body.
//!
//! This is the cooperative-tensor counterpart of `mt_sdpa_prefill_mma`.
//! It runs the standard FlashAttention-2 online-softmax loop, but the
//! two matmuls inside the loop — `S = Q · Kᵀ` and `O += P · V` — are
//! each one cooperative `matmul2d` instead of an 8×8 `simdgroup_matmul`
//! ladder.
//!
//! ## Tile geometry
//!
//!   - **BQ = 16** queries per threadgroup, **BK = 16** keys per block,
//!     **BD = 32** head dimension.
//!   - **tpg = 32** (one simdgroup). The 16×16 S tile and the 16×32 O
//!     tile are each covered by a single cooperative `matmul2d`.
//!   - Grid: `[q_len / 16, n_q_heads, batch]` — `tgid_x` = Q-tile,
//!     `tgid_y` = Q-head, `tgid_z` = batch.
//!
//! `BD = 32` is a deliberate constraint: it makes the QK descriptor's
//! K-dim exactly 32, satisfying Apple's "at least one of M/N/K = 32"
//! rule for two-operand cooperative tensors with no head-dim tiling.
//! Larger head dims are a follow-up (loop the QK contraction over
//! 32-wide D-chunks).
//!
//! ## Per K-block flash loop
//!
//!   1. Coop-load the 16×32 K tile and 16×32 V tile into TG memory.
//!   2. `S = Q · Kᵀ` — `matmul2d(M=16, N=16, K=32)`, `ta=false,
//!      tb=true` (Kᵀ via the transposed-B read). `ct_s` is mode
//!      `multiply` — each K-block recomputes S fresh.
//!   3. Store S to TG scratch; each lane owns one S row, applies the
//!      causal mask, runs the online-softmax max/sum rescale, and
//!      writes the exp-weights `P` back into the scratch.
//!   4. `O += P · V` — `matmul2d(M=16, N=32, K=16)`, `ta=false,
//!      tb=false` (V is `[BK, BD]` row-major = the K×N operand).
//!      `ct_o` is mode `multiply_accumulate`, but the per-block
//!      max-rescale means O must be rescaled by `exp(m_old - m_new)`
//!      between blocks — done by reading O out to scratch, scaling,
//!      reloading. ct_o therefore runs in `multiply` mode per block and
//!      the accumulation is explicit in scratch.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: 32 threads** (1 SG). Fixed.
//! - **Grid: `[q_len / 16, n_q_heads, batch]`.**
//! - **`q_len % 16 == 0`, `k_len % 16 == 0`, `head_dim == 32`.** Loads
//!   are unconditional; callers must pad.
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_attention_nax_gpu_correctness.rs`.

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind},
    shape::{Dim, Shape},
};

/// Tile geometry — keep in lock-step with the inline MSL below.
pub const BQ: u32 = 16;
pub const BK: u32 = 16;
pub const BD: u32 = 32;
/// Threads per group (1 SG × 32 lanes).
pub const TPG: u32 = 32;
/// Threadgroup-mem row skew — 4 elems past the inner extent to scatter
/// 32-bank conflicts on the column-strided frag loads inside `matmul2d`.
pub const TG_SKEW: u32 = 4;
/// Leading dim of the BQ/BK × BD tiles (BD + skew).
pub const TG_LD_D: u32 = BD + TG_SKEW; // 36
/// Leading dim of the BQ × BK S/P scratch (BK + skew).
pub const TG_LD_K: u32 = BK + TG_SKEW; // 20

/// MSL source. Codegen emits the bindings as `const device {T}
/// *q/k/v` + `device {T} *out` + `constant uint
/// &q_len/k_len/gqa_factor/n_q_heads/n_kv_heads` + `constant float
/// &scale`. Templated on `T` via `{T}`.
const ATTN_NAX_SRC_TEMPLATE: &str = r#"// --- mt_sdpa_prefill_nax body (BQ=16, BK=16, BD=32, TG=32, 1 SG) ---
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
constexpr uint BQ = 16;
constexpr uint BK = 16;
constexpr uint BD = 32;
constexpr uint TG_LD_D = 36;   // BD + 4 skew
constexpr uint TG_LD_K = 20;   // BK + 4 skew

// ── Threadgroup tiles ──
// Qs : 16×32 query tile (m, d) row-major, skewed.
// Ks : 16×32 key   tile (k, d) row-major, skewed.
// Vs : 16×32 value tile (k, d) row-major, skewed.
// Ss : 16×16 score scratch (m, k) row-major, skewed.
// Os : 16×32 output accumulator (m, d) row-major, skewed.
// Ps : 16×16 fp `T` exp-weight tile staged for the P·V matmul.
// Obk: 16×32 per-block P·V product, added into Os.
// All `threadgroup` arrays are declared at kernel scope — MSL forbids
// `threadgroup` declarations inside loop bodies.
threadgroup {T}    Qs[BQ * TG_LD_D];
threadgroup {T}    Ks[BK * TG_LD_D];
threadgroup {T}    Vs[BK * TG_LD_D];
threadgroup float  Ss[BQ * TG_LD_K];
threadgroup float  Os[BQ * TG_LD_D];
threadgroup {T}    Ps[BQ * TG_LD_K];
threadgroup float  Obk[BQ * TG_LD_D];

// Grid: tgid_x = Q-tile, tgid_y = Q-head, tgid_z = batch.
const uint q_tile = tgid_x;
const uint q_head = tgid_y;
const uint batch  = tgid_z;
const uint kv_head = q_head / gqa_factor;
const uint lane = simd_lane;

const uint head_dim = BD;
const uint q_len_off = k_len - q_len;
// FlashAttention-2 works in log2 space — fold the 1/ln2 factor into the
// query scale so the softmax can use exp2.
const float scale_log2 = scale * 1.4426950408889634f;

// Slab offsets (batched-prefill layout):
//   q, out : [batch, n_q_heads,  q_len, head_dim]
//   k, v   : [batch, n_kv_heads, k_len, head_dim]
const uint kv_row_base    = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
const uint q_head_row_off = batch * n_q_heads  * q_len * head_dim + q_head  * q_len * head_dim;
const uint q_tile_first   = q_tile * BQ;

// ── Coop-load the 16×32 Q tile (32 lanes × 16 elems) ──
// lane fills column `lane` of every Q row.
for (uint r = 0u; r < BQ; ++r) {
    const uint q_dev = q_head_row_off + (q_tile_first + r) * head_dim + lane;
    Qs[r * TG_LD_D + lane] = ({T})((float)q[q_dev] * scale_log2);
    Os[r * TG_LD_D + lane] = 0.0f;
}

// ── Per-row online-softmax state — lane `r` (r < BQ) owns row r ──
// 16 rows, 32 lanes: lanes 0..16 each own one query row.
float row_m = -INFINITY;
float row_s = 0.0f;
const bool owns_row = lane < BQ;
const uint my_row = lane;                       // valid only when owns_row
const uint q_abs  = q_tile_first + my_row + q_len_off; // absolute query pos

// ── QK descriptor: (M=16, N=16, K=32), ta=false, tb=true ──
// A = Qs [16, 32] row-major; B = Ks [16, 32] row-major, tb=true reads it
// as the 32×16 Kᵀ operand. mode = multiply (S recomputed each block).
constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/16, /*K=*/32,
    /*ta=*/false, /*tb=*/true, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;

// ── PV descriptor: (M=16, N=32, K=16), ta=false, tb=false ──
// A = P [16, 16] row-major; B = Vs [16, 32] row-major = the K×N operand.
// mode = multiply (per-block product; accumulation is explicit in Os).
constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
    /*M=*/16, /*N=*/32, /*K=*/16,
    /*ta=*/false, /*tb=*/false, /*tc=*/false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

// Causal trim — bound the K-block loop at the last query of the tile.
const uint q_tile_last_abs = q_tile_first + (BQ - 1u) + q_len_off;
const uint kb_lim = (q_tile_last_abs / BK) + 1u;

for (uint kb = 0u; kb < kb_lim; ++kb) {
    const uint kb_off = kb * BK;

    // ── 1. Coop-load the 16×32 K and V tiles ──
    for (uint r = 0u; r < BK; ++r) {
        const uint kv_dev = kv_row_base + (kb_off + r) * head_dim + lane;
        Ks[r * TG_LD_D + lane] = ({T})k[kv_dev];
        Vs[r * TG_LD_D + lane] = ({T})v[kv_dev];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. S = Q · Kᵀ via cooperative matmul2d ──
    auto ct_q  = qk_op.template get_left_input_cooperative_tensor<{T}, {T}, float>();
    auto ct_kt = qk_op.template get_right_input_cooperative_tensor<{T}, {T}, float>();
    auto ct_s  = qk_op.template get_destination_cooperative_tensor<decltype(ct_q), decltype(ct_kt), float>();
    {
        metal::tensor<threadgroup {T}, metal::extents<int, TG_LD_D, 16>, metal::tensor_inline>
            tQ(Qs, metal::extents<int, TG_LD_D, 16>{});
        metal::tensor<threadgroup {T}, metal::extents<int, TG_LD_D, 16>, metal::tensor_inline>
            tK(Ks, metal::extents<int, TG_LD_D, 16>{});
        ct_q.load(tQ);
        ct_kt.load(tK);
    }
    qk_op.run(ct_q, ct_kt, ct_s);

    // Spill S to TG scratch so the lane-owned softmax can read its row.
    {
        metal::tensor<threadgroup float, metal::extents<int, TG_LD_K, 16>, metal::tensor_inline>
            tS(Ss, metal::extents<int, TG_LD_K, 16>{});
        ct_s.store(tS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. Online-softmax — each owning lane processes its row ──
    if (owns_row) {
        // Row max over the BK scores, with causal masking.
        float blk_m = -INFINITY;
        for (uint c = 0u; c < BK; ++c) {
            const uint k_abs = kb_off + c;
            float sc = Ss[my_row * TG_LD_K + c];
            if (k_abs > q_abs) { sc = -INFINITY; }
            Ss[my_row * TG_LD_K + c] = sc;
            blk_m = (sc > blk_m) ? sc : blk_m;
        }
        const float new_m = (blk_m > row_m) ? blk_m : row_m;
        const float rescale = exp2(row_m - new_m);
        // Exponentiate the row in place → P; track the block sum.
        float blk_s = 0.0f;
        for (uint c = 0u; c < BK; ++c) {
            const float p = exp2(Ss[my_row * TG_LD_K + c] - new_m);
            Ss[my_row * TG_LD_K + c] = p;
            blk_s += p;
        }
        row_s = row_s * rescale + blk_s;
        // Rescale the running O accumulator by exp(m_old - m_new).
        for (uint d = 0u; d < BD; ++d) {
            Os[my_row * TG_LD_D + d] *= rescale;
        }
        row_m = new_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 4. O += P · V via cooperative matmul2d ──
    // P is fp32 in Ss; matmul2d's left input is typed {T}, so stage P
    // through the {T} `Ps` scratch tile.
    for (uint c = 0u; c < BK; ++c) {
        if (owns_row) {
            Ps[my_row * TG_LD_K + c] = ({T})Ss[my_row * TG_LD_K + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto ct_p = pv_op.template get_left_input_cooperative_tensor<{T}, {T}, float>();
    auto ct_v = pv_op.template get_right_input_cooperative_tensor<{T}, {T}, float>();
    auto ct_o = pv_op.template get_destination_cooperative_tensor<decltype(ct_p), decltype(ct_v), float>();
    {
        metal::tensor<threadgroup {T}, metal::extents<int, TG_LD_K, 16>, metal::tensor_inline>
            tP(Ps, metal::extents<int, TG_LD_K, 16>{});
        metal::tensor<threadgroup {T}, metal::extents<int, TG_LD_D, 16>, metal::tensor_inline>
            tV(Vs, metal::extents<int, TG_LD_D, 16>{});
        ct_p.load(tP);
        ct_v.load(tV);
    }
    pv_op.run(ct_p, ct_v, ct_o);

    // Add the per-block P·V product (in `Obk`) into the running Os
    // accumulator.
    {
        metal::tensor<threadgroup float, metal::extents<int, TG_LD_D, 16>, metal::tensor_inline>
            tOb(Obk, metal::extents<int, TG_LD_D, 16>{});
        ct_o.store(tOb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns_row) {
        for (uint d = 0u; d < BD; ++d) {
            Os[my_row * TG_LD_D + d] += Obk[my_row * TG_LD_D + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ── 5. Normalize and store O ──
if (owns_row) {
    const float inv_s = (row_s > 0.0f) ? (1.0f / row_s) : 0.0f;
    for (uint d = 0u; d < BD; ++d) {
        const uint o_dev = q_head_row_off + (q_tile_first + my_row) * head_dim + d;
        out[o_dev] = ({T})(Os[my_row * TG_LD_D + d] * inv_s);
    }
}
#else
// Pre-Metal-4 fallback — silence the bindings so the metallib still links.
// Correctness test on such targets is the intended failure signal.
if (simd_lane == 0u) {
    const uint o = (tgid_z * n_q_heads + tgid_y) * q_len * 32u + tgid_x * 16u * 32u;
    const float _g = (float)gqa_factor + (float)n_kv_heads;
    out[o] = ({T})((float)q[0] * (float)k[0] * (float)v[0] * scale * _g);
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

/// Build the per-dtype [`Kernel`] IR for `mt_sdpa_prefill_nax_{T}`.
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = q           const device {T}   *
///   buffer(1) = k           const device {T}   *
///   buffer(2) = v           const device {T}   *
///   buffer(3) = out         device       {T}   *
///   buffer(4) = q_len       constant     uint  &
///   buffer(5) = k_len       constant     uint  &
///   buffer(6) = gqa_factor  constant     uint  &
///   buffer(7) = n_q_heads   constant     uint  &
///   buffer(8) = n_kv_heads  constant     uint  &
///   buffer(9) = scale       constant     float &
///
/// Dispatch geometry: grid `[q_len/16, n_q_heads, batch]`,
/// threadgroup `[32, 1, 1]`.
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_sdpa_prefill_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_sdpa_prefill_nax");
    k.mode = KernelMode::Reduction;

    for name in ["q", "k", "v"] {
        k.params.push(Param {
            name: name.into(),
            dtype: dt,
            shape: Shape::new([Dim::Any, Dim::Any]),
            is_output: false,
            kind: ParamKind::Tensor,
        });
    }
    k.params.push(Param {
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    for name in ["q_len", "k_len", "gqa_factor", "n_q_heads", "n_kv_heads"] {
        k.constexprs.push(ConstExprDecl {
            name: ConstExpr::new(name),
            dtype: DType::U32,
            value: None,
        });
    }
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("scale"),
        dtype: DType::F32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // Force `tgid_y` + `tgid_z` alias emission — InlineMsl source
    // mentions them but the body text isn't scanned for the alias
    // trigger; codegen only looks at IR ops. Use the
    // `Op::Load { src: "tgid_*" }` direct-identifier form (see
    // `steel_gemm_splitk_nax`). Reduction mode emits `tgid_x`
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
        source: substitute_dtype(ATTN_NAX_SRC_TEMPLATE, dt),
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
            assert_eq!(k.name, "mt_sdpa_prefill_nax");
            assert_eq!(k.params.len(), 4);
            assert_eq!(k.params[0].name, "q");
            assert_eq!(k.params[1].name, "k");
            assert_eq!(k.params[2].name, "v");
            assert_eq!(k.params[3].name, "out");
            assert!(k.params[3].is_output);
            assert_eq!(k.constexprs.len(), 6);
            assert_eq!(k.constexprs[0].name.name(), "q_len");
            assert_eq!(k.constexprs[1].name.name(), "k_len");
            assert_eq!(k.constexprs[2].name.name(), "gqa_factor");
            assert_eq!(k.constexprs[3].name.name(), "n_q_heads");
            assert_eq!(k.constexprs[4].name.name(), "n_kv_heads");
            assert_eq!(k.constexprs[5].name.name(), "scale");
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
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half")] {
            let mut k = kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            k.name = format!("mt_sdpa_prefill_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_sdpa_prefill_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name}    Qs")));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
            assert!(msl.contains("tgid_z"), "tgid_z must be bound:\n{msl}");
        }
    }
}

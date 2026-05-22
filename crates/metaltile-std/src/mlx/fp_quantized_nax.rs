//! `mt_fp_qmm_nax` — fp4 (E2M1) quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the fp4 quantized matmul from MLX
//! `metal/kernels/fp_quantized_nax.metal`. Gated behind the `nax` Cargo
//! feature — the kernel requires the Metal 4 `MetalPerformancePrimitives`
//! framework (macOS 26+); codegen emits the framework include when
//! it detects `CoopTile*` ops in the body.
//!
//! Fp4 counterpart of `mt_qmm_nax`. Mirrors the same coop-load /
//! coop-dequant / `matmul2d` pattern — packed weights are dequantized
//! into threadgroup memory once per K-block, then per-simdgroup
//! `matmul2d` runs against the fp `T` X-tile — but swaps the int4
//! nibble-dequant for an **fp4 E2M1 codebook lookup**:
//!
//!   - Each 4-bit code is `[sign : 1][exp : 2][mantissa : 1]`.
//!   - The 3-bit magnitude indexes the E2M1 codebook
//!     `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (the nvfp4 levels — see
//!     MLX `fp4.h`). We compute it without a table via the
//!     identity:
//!     - subnormal (exp = 0): magnitude = mantissa · 0.5
//!     - normal    (exp ≥ 1): magnitude = (1 + mantissa·0.5) · 2^(exp − 1)
//!
//!     One `Select` between the two branches in IR.
//!   - The sign bit (`code & 8`) negates the magnitude.
//!   - The dequantized value is `scale · sign · magnitude`. Fp4
//!     quantization is **scale-only** — no per-group bias, unlike
//!     the affine int4 path — so the IR drops the bias param/load
//!     entirely.
//!
//! 8 fp4 codes pack into one `u32`; one `u32` covers `BK/4 = 8` of a
//! BK=32 row. Pack count per BK-block matches the int4 path (4 packs
//! per row × 32 rows = 128 lanes = TPG), so the coop-load mapping is
//! shared with `mt_qmm_nax` — only the dequant inner loop differs.
//! Per-K-block scale layout uses `GROUP_SIZE = 32` (one scale per
//! BK-block per N-row).
//!
//! Expressed entirely via `CoopTile*` IR ops — no `Op::InlineMsl`.
//! Geometry mirrors `mt_qmm_nax`:
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
    ir::{
        BinOpKind,
        Block,
        BlockId,
        ConstExprDecl,
        CoopTileAccMode,
        CoopTileScope,
        IndexExpr,
        Kernel,
        KernelMode,
        Op,
        Param,
        ParamKind,
        ValueId,
        VarId,
    },
    shape::{Dim, Shape},
};
use rustc_hash::FxHashMap;

/// Tile geometry.
pub const BM: u32 = 32;
pub const BN: u32 = 32;
pub const BK: u32 = 32;
/// Threads per group (4 SG × 32 lanes).
pub const TPG: u32 = 128;
/// Threadgroup-mem row stride = BK + 4 (bank-conflict skew).
pub const TG_SKEW: u32 = 4;
pub const TG_LD: u32 = BK + TG_SKEW; // 36
/// fp4 quantization group size — one group per BK-block.
pub const GROUP_SIZE: u32 = 32;

/// Build the per-dtype [`Kernel`] IR for `mt_fp_qmm_nax`.
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
#[allow(unused_assignments)] // final nv!() bumps vid past last read — by design
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
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "scales".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "x".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("gs_per_row"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // -----------------------------------------------------------------------
    // ValueId counter shared across blocks (same convention as quantized_nax).
    // -----------------------------------------------------------------------
    let mut vid: u32 = 0;
    macro_rules! nv {
        () => {{
            let id = ValueId::new(vid);
            vid += 1;
            id
        }};
    }
    macro_rules! load0 {
        ($src:expr) => {
            Op::Load { src: $src.into(), indices: vec![], mask: None, other: None }
        };
    }
    macro_rules! bop {
        ($op:ident, $l:expr, $r:expr) => {
            Op::BinOp { op: BinOpKind::$op, lhs: $l, rhs: $r }
        };
    }

    // -----------------------------------------------------------------------
    // Block 0 — preamble: program IDs, per-lane indices, TG allocs, setup.
    // -----------------------------------------------------------------------
    let mut b0 = Block::new(BlockId::new(0));

    // TG arrays.
    b0.push_op_no_result(Op::ThreadgroupAlloc { dtype: dt, size: BM * TG_LD, name: "Xs".into() });
    b0.push_op_no_result(Op::ThreadgroupAlloc { dtype: dt, size: BN * TG_LD, name: "Ws".into() });
    b0.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: DType::F32,
        size: 4 * 16 * 16,
        name: "OutScratch".into(),
    });

    // Program IDs.
    let v_tgid_y = nv!();
    b0.push_op(Op::ProgramId { axis: 1 }, v_tgid_y);
    let v_tgid_x = nv!();
    b0.push_op(Op::ProgramId { axis: 0 }, v_tgid_x);

    // Built-in lane/SG indices.
    let v_sg = nv!();
    b0.push_op(load0!("simd_id"), v_sg);
    let v_lane = nv!();
    b0.push_op(load0!("simd_lane"), v_lane);

    // Integer constants.
    let c0 = nv!();
    b0.push_op(Op::Const { value: 0 }, c0);
    let c1 = nv!();
    b0.push_op(Op::Const { value: 1 }, c1);
    let c2 = nv!();
    b0.push_op(Op::Const { value: 2 }, c2);
    let c3 = nv!();
    b0.push_op(Op::Const { value: 3 }, c3);
    let c4 = nv!();
    b0.push_op(Op::Const { value: 4 }, c4);
    let c7 = nv!();
    b0.push_op(Op::Const { value: 7 }, c7);
    let c8 = nv!();
    b0.push_op(Op::Const { value: 8 }, c8);
    let c15 = nv!();
    b0.push_op(Op::Const { value: 15 }, c15);
    let c16 = nv!();
    b0.push_op(Op::Const { value: 16 }, c16);
    let c32 = nv!();
    b0.push_op(Op::Const { value: 32 }, c32);
    let c36 = nv!();
    b0.push_op(Op::Const { value: 36 }, c36); // TG_LD
    let c256 = nv!();
    b0.push_op(Op::Const { value: 256 }, c256);

    // Constexpr loads.
    let v_k = nv!();
    b0.push_op(load0!("k"), v_k);
    let v_n = nv!();
    b0.push_op(load0!("n"), v_n);
    let v_gs = nv!();
    b0.push_op(load0!("gs_per_row"), v_gs);

    // lane_in_tg = simd_group * 32 + simd_lane.
    let v_sg32 = nv!();
    b0.push_op(bop!(Mul, v_sg, c32), v_sg32);
    let v_lane_in_tg = nv!();
    b0.push_op(bop!(Add, v_sg32, v_lane), v_lane_in_tg);

    // sm = simd_group / 2, sn = simd_group & 1.
    let v_sm = nv!();
    b0.push_op(bop!(Div, v_sg, c2), v_sm);
    let v_sn = nv!();
    b0.push_op(bop!(BitAnd, v_sg, c1), v_sn);

    // sg_m_base = sm * 16, sg_n_base = sn * 16.
    let v_sg_m_base = nv!();
    b0.push_op(bop!(Mul, v_sm, c16), v_sg_m_base);
    let v_sg_n_base = nv!();
    b0.push_op(bop!(Mul, v_sn, c16), v_sg_n_base);

    // Per-SG TG-buffer offsets.
    let v_xs_sg_off = nv!();
    b0.push_op(bop!(Mul, v_sg_m_base, c36), v_xs_sg_off);
    let v_ws_sg_off = nv!();
    b0.push_op(bop!(Mul, v_sg_n_base, c36), v_ws_sg_off);

    // X coop-load mapping: 128 lanes × 8 contiguous K.
    let v_x_m_row = nv!();
    b0.push_op(bop!(Div, v_lane_in_tg, c4), v_x_m_row);
    let v_x_k_quad = nv!();
    b0.push_op(bop!(BitAnd, v_lane_in_tg, c3), v_x_k_quad);
    let v_x_k_base = nv!();
    b0.push_op(bop!(Mul, v_x_k_quad, c8), v_x_k_base);

    // x_m_base = tgid_y * 32, w_n_base = tgid_x * 32.
    let v_x_m_base = nv!();
    b0.push_op(bop!(Mul, v_tgid_y, c32), v_x_m_base);
    let v_w_n_base = nv!();
    b0.push_op(bop!(Mul, v_tgid_x, c32), v_w_n_base);

    // packs_per_row = k / 8.
    let v_packs_per_row = nv!();
    b0.push_op(bop!(Div, v_k, c8), v_packs_per_row);

    // wn_plus_wr = w_n_base + w_row (w_row == x_m_row).
    let v_wn_wr = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_x_m_row), v_wn_wr);

    // sb_base = wn_plus_wr * gs_per_row.
    let v_sb_base = nv!();
    b0.push_op(bop!(Mul, v_wn_wr, v_gs), v_sb_base);

    // w_pack_row_base = wn_plus_wr * packs_per_row.
    let v_w_pack_row_base = nv!();
    b0.push_op(bop!(Mul, v_wn_wr, v_packs_per_row), v_w_pack_row_base);

    // out bases.
    let v_out_m_base = nv!();
    b0.push_op(bop!(Add, v_x_m_base, v_sg_m_base), v_out_m_base);
    let v_out_n_base = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_sg_n_base), v_out_n_base);

    // sg_scratch_off = simd_group * 256.
    let v_sg_scratch_off = nv!();
    b0.push_op(bop!(Mul, v_sg, c256), v_sg_scratch_off);

    // o_row, o_col_base for the 16×16 store loop.
    let v_o_row = nv!();
    b0.push_op(bop!(Div, v_lane, c2), v_o_row);
    let v_lane_mod2 = nv!();
    b0.push_op(bop!(BitAnd, v_lane, c1), v_lane_mod2);
    let v_o_col_base = nv!();
    b0.push_op(bop!(Mul, v_lane_mod2, c8), v_o_col_base);

    // CoopTile setup + zero.
    b0.push_op_no_result(Op::CoopTileSetup {
        name: "gemm".into(),
        m: 16,
        n: 16,
        k: 32,
        ta: false,
        tb: true,
        tc: false,
        acc_mode: CoopTileAccMode::MultiplyAccumulate,
        exec_scope: CoopTileScope::SimdGroup,
        act_dtype: dt,
        acc_dtype: DType::F32,
        direct_inputs: false,
        a_is_tg: false,
        a_ei: 0,
        a_eo: 0,
        b_is_tg: false,
        b_ei: 0,
        b_eo: 0,
    });
    b0.push_op_no_result(Op::CoopTileZero { name: "gemm".into() });

    // K-loop.
    b0.push_op_no_result(Op::Loop {
        var: VarId::new(0),
        start: c0,
        end: v_k,
        step: c32,
        body: BlockId::new(1),
    });

    // Store ct_c, barrier, then write loop.
    b0.push_op_no_result(Op::CoopTileStoreC {
        name: "gemm".into(),
        ptr_name: "OutScratch".into(),
        ptr_offset: Some(v_sg_scratch_off),
        is_tg: true,
        dtype: DType::F32,
        ei: 16,
        eo: 16,
    });
    b0.push_op_no_result(Op::Barrier);
    b0.push_op_no_result(Op::Loop {
        var: VarId::new(1),
        start: c0,
        end: c8,
        step: c1,
        body: BlockId::new(2),
    });

    // -----------------------------------------------------------------------
    // Block 1 — K-loop body: X staging + W fp4-dequant + CoopTile ops.
    // -----------------------------------------------------------------------
    let mut b1 = Block::new(BlockId::new(1));
    let v_kb = ValueId::new(0xC000_0000);

    // x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base.
    let v_xm_mr = nv!();
    b1.push_op(bop!(Add, v_x_m_base, v_x_m_row), v_xm_mr);
    let v_xm_k = nv!();
    b1.push_op(bop!(Mul, v_xm_mr, v_k), v_xm_k);
    let v_xm_kb = nv!();
    b1.push_op(bop!(Add, v_xm_k, v_kb), v_xm_kb);
    let v_x_rdb = nv!();
    b1.push_op(bop!(Add, v_xm_kb, v_x_k_base), v_x_rdb);

    // x_ws_base = x_m_row * TG_LD + x_k_base.
    let v_mr_tgld = nv!();
    b1.push_op(bop!(Mul, v_x_m_row, c36), v_mr_tgld);
    let v_x_wsb = nv!();
    b1.push_op(bop!(Add, v_mr_tgld, v_x_k_base), v_x_wsb);

    // Unrolled X load/store: 8 elements per lane.
    for ei in 0u32..8 {
        let vc = nv!();
        b1.push_op(Op::Const { value: ei as i64 }, vc);
        let vdi = nv!();
        b1.push_op(bop!(Add, v_x_rdb, vc), vdi);
        let vxv = nv!();
        b1.push_op(
            Op::Load {
                src: "x".into(),
                indices: vec![IndexExpr::Value(vdi)],
                mask: None,
                other: None,
            },
            vxv,
        );
        let vti = nv!();
        b1.push_op(bop!(Add, v_x_wsb, vc), vti);
        b1.push_op_no_result(Op::ThreadgroupStore { name: "Xs".into(), index: vti, value: vxv });
    }

    // W fp4-dequant: 1 pack per lane covers 8 codes.
    // pack_k_off = kb / 8 + pack_in_row.
    let v_kb8 = nv!();
    b1.push_op(bop!(Div, v_kb, c8), v_kb8);
    let v_pkoff = nv!();
    b1.push_op(bop!(Add, v_kb8, v_x_k_quad), v_pkoff);
    let v_pdev = nv!();
    b1.push_op(bop!(Add, v_w_pack_row_base, v_pkoff), v_pdev);
    let v_pack = nv!();
    b1.push_op(
        Op::Load {
            src: "w".into(),
            indices: vec![IndexExpr::Value(v_pdev)],
            mask: None,
            other: None,
        },
        v_pack,
    );

    // k_off = kb + pack_in_row * 8.
    let v_pir8 = nv!();
    b1.push_op(bop!(Mul, v_x_k_quad, c8), v_pir8);
    let v_koff = nv!();
    b1.push_op(bop!(Add, v_kb, v_pir8), v_koff);

    // g = k_off / GROUP_SIZE (=32).
    let v_g = nv!();
    b1.push_op(bop!(Div, v_koff, c32), v_g);
    let v_sbg = nv!();
    b1.push_op(bop!(Add, v_sb_base, v_g), v_sbg);
    let v_scl_raw = nv!();
    b1.push_op(
        Op::Load {
            src: "scales".into(),
            indices: vec![IndexExpr::Value(v_sbg)],
            mask: None,
            other: None,
        },
        v_scl_raw,
    );
    let v_scl = nv!();
    b1.push_op(Op::Cast { value: v_scl_raw, dtype: DType::F32 }, v_scl);

    // ws_base = w_row * TG_LD + pack_in_row * 8.
    let v_wr36 = nv!();
    b1.push_op(bop!(Mul, v_x_m_row, c36), v_wr36);
    let v_wsb = nv!();
    b1.push_op(bop!(Add, v_wr36, v_pir8), v_wsb);

    // FP4 dequant via integer arithmetic — avoids needing scalar f32
    // const ops in hand-built IR. The codebook
    // `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` equals `two_m_int / 2` where
    // `two_m_int` is the integer "value × 2":
    //
    //   subnormal (exp=0): two_m_int = mantissa            ∈ {0, 1}
    //   normal    (exp≥1): two_m_int = (2 + mantissa) · 2^(exp − 1)
    //                                  ∈ {2, 3, 4, 6, 8, 12}
    //
    // After the integer compute we cast once to fp32 and divide by 2,
    // folding the / 2 into the multiply by scale (scale * sign / 2)
    // for fewer ops per code.

    // Final divisor (= 2). Used once per code to fold the / 2 into the
    // scale-sign product. Float div by 2 is a free bit-shift on the
    // exponent on every GPU; emitting it as a real div keeps the IR
    // arithmetic ungated.
    let v_two_f = nv!();
    b1.push_op(Op::Cast { value: c2, dtype: DType::F32 }, v_two_f);

    // Unrolled per-code dequant: 8 fp4 codes per pack.
    for (ni, shift) in [0u32, 4, 8, 12, 16, 20, 24, 28].iter().enumerate() {
        let vc_sh = nv!();
        b1.push_op(Op::Const { value: *shift as i64 }, vc_sh);
        let v_shr = nv!();
        b1.push_op(bop!(Shr, v_pack, vc_sh), v_shr);
        let v_code = nv!();
        b1.push_op(bop!(BitAnd, v_shr, c15), v_code); // code ∈ 0..16

        // Split code: sign = (code >> 3) & 1, mag_bits = code & 7.
        let v_code_high = nv!();
        b1.push_op(bop!(Shr, v_code, c3), v_code_high);
        let v_sign_bit = nv!();
        b1.push_op(bop!(BitAnd, v_code_high, c1), v_sign_bit); // 0 or 1
        let v_mag_bits = nv!();
        b1.push_op(bop!(BitAnd, v_code, c7), v_mag_bits); // 0..7

        // exp = mag_bits >> 1, mantissa = mag_bits & 1.
        let v_exp = nv!();
        b1.push_op(bop!(Shr, v_mag_bits, c1), v_exp); // 0..3
        let v_mant = nv!();
        b1.push_op(bop!(BitAnd, v_mag_bits, c1), v_mant); // 0 or 1

        // is_subnormal = (exp == 0).
        let v_is_subnormal = nv!();
        b1.push_op(Op::BinOp { op: BinOpKind::CmpEq, lhs: v_exp, rhs: c0 }, v_is_subnormal);

        // safe_exp = is_subnormal ? 1 : exp — keeps (exp − 1) ≥ 0 for the
        // shift below. The subnormal branch never reads pow2.
        let v_safe_exp = nv!();
        b1.push_op(Op::Select { cond: v_is_subnormal, on_true: c1, on_false: v_exp }, v_safe_exp);
        let v_shift_amt = nv!();
        b1.push_op(bop!(Sub, v_safe_exp, c1), v_shift_amt); // 0..2
        let v_pow2 = nv!();
        b1.push_op(bop!(Shl, c1, v_shift_amt), v_pow2); // ∈ {1, 2, 4}

        // normal_two_m = (mantissa + 2) * pow2 — integer "value × 2"
        // for the normal branch (exp ≥ 1).
        let v_mant_plus_2 = nv!();
        b1.push_op(bop!(Add, v_mant, c2), v_mant_plus_2);
        let v_normal_two_m = nv!();
        b1.push_op(bop!(Mul, v_mant_plus_2, v_pow2), v_normal_two_m);

        // two_m_int = is_subnormal ? mantissa : normal_two_m.
        let v_two_m_int = nv!();
        b1.push_op(
            Op::Select { cond: v_is_subnormal, on_true: v_mant, on_false: v_normal_two_m },
            v_two_m_int,
        );
        let v_two_m_f = nv!();
        b1.push_op(Op::Cast { value: v_two_m_int, dtype: DType::F32 }, v_two_m_f);

        // sign_f = 1.0 − 2.0·sign_bit ∈ {+1, −1}. Compute in fp32 (not
        // unsigned int) — `1u32 − 2u32` would underflow when sign_bit=1.
        let v_one_f = nv!();
        b1.push_op(Op::Cast { value: c1, dtype: DType::F32 }, v_one_f);
        let v_sign_bit_f = nv!();
        b1.push_op(Op::Cast { value: v_sign_bit, dtype: DType::F32 }, v_sign_bit_f);
        let v_two_sign_f = nv!();
        b1.push_op(bop!(Mul, v_two_f, v_sign_bit_f), v_two_sign_f);
        let v_sign_f = nv!();
        b1.push_op(bop!(Sub, v_one_f, v_two_sign_f), v_sign_f);

        // value = scale · sign_f · two_m_f / 2.
        let v_scl_sign = nv!();
        b1.push_op(bop!(Mul, v_scl, v_sign_f), v_scl_sign);
        let v_scl_sign_half = nv!();
        b1.push_op(bop!(Div, v_scl_sign, v_two_f), v_scl_sign_half);
        let v_val = nv!();
        b1.push_op(bop!(Mul, v_scl_sign_half, v_two_m_f), v_val);

        // Cast to T and store into Ws.
        let v_wv = nv!();
        b1.push_op(Op::Cast { value: v_val, dtype: dt }, v_wv);
        let vc_i = nv!();
        b1.push_op(Op::Const { value: ni as i64 }, vc_i);
        let v_wi = nv!();
        b1.push_op(bop!(Add, v_wsb, vc_i), v_wi);
        b1.push_op_no_result(Op::ThreadgroupStore { name: "Ws".into(), index: v_wi, value: v_wv });
    }

    b1.push_op_no_result(Op::Barrier);

    // Per-SG cooperative-tensor load from TG staging buffers.
    b1.push_op_no_result(Op::CoopTileLoadA {
        name: "gemm".into(),
        ptr_name: "Xs".into(),
        ptr_offset: Some(v_xs_sg_off),
        is_tg: true,
        dtype: dt,
        ei: 36,
        eo: 16,
        direct: false,
    });
    b1.push_op_no_result(Op::CoopTileLoadB {
        name: "gemm".into(),
        ptr_name: "Ws".into(),
        ptr_offset: Some(v_ws_sg_off),
        is_tg: true,
        dtype: dt,
        ei: 36,
        eo: 16,
        direct: false,
    });
    b1.push_op_no_result(Op::CoopTileRun { name: "gemm".into(), direct: false });
    b1.push_op_no_result(Op::Barrier);

    // -----------------------------------------------------------------------
    // Block 2 — write loop: cast fp32 scratch → T and store to `out`.
    // 32 lanes × 8 elems = 256 = 16×16.
    // -----------------------------------------------------------------------
    let mut b2 = Block::new(BlockId::new(2));
    let v_wi_loop = ValueId::new(0xC000_0001);

    let v_col = nv!();
    b2.push_op(bop!(Add, v_o_col_base, v_wi_loop), v_col);

    let v_r16 = nv!();
    b2.push_op(bop!(Mul, v_o_row, c16), v_r16);
    let v_sloc = nv!();
    b2.push_op(bop!(Add, v_r16, v_col), v_sloc);
    let v_sidx = nv!();
    b2.push_op(bop!(Add, v_sg_scratch_off, v_sloc), v_sidx);

    let v_f32 = nv!();
    b2.push_op(Op::ThreadgroupLoad { name: "OutScratch".into(), index: v_sidx }, v_f32);
    let v_vt = nv!();
    b2.push_op(Op::Cast { value: v_f32, dtype: dt }, v_vt);

    let v_orow = nv!();
    b2.push_op(bop!(Add, v_out_m_base, v_o_row), v_orow);
    let v_ocol = nv!();
    b2.push_op(bop!(Add, v_out_n_base, v_col), v_ocol);
    let v_orn = nv!();
    b2.push_op(bop!(Mul, v_orow, v_n), v_orn);
    let v_oidx = nv!();
    b2.push_op(bop!(Add, v_orn, v_ocol), v_oidx);

    b2.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(v_oidx)],
        value: v_vt,
        mask: None,
    });

    // -----------------------------------------------------------------------
    // Assemble kernel.
    // -----------------------------------------------------------------------
    let mut all_blocks = FxHashMap::default();
    all_blocks.insert(BlockId::new(0), b0.clone());
    all_blocks.insert(BlockId::new(1), b1);
    all_blocks.insert(BlockId::new(2), b2);

    k.body = b0;
    k.blocks = all_blocks;
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
            // Body must use CoopTile* ops, not InlineMsl.
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileZero { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileStoreC { .. })));
            assert!(!k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            // K-loop body (block 1) must have CoopTileLoadA/B and Run.
            let b1 = &k.blocks[&BlockId::new(1)];
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileLoadA { .. })));
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileLoadB { .. })));
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    fn dump_generated_msl() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = kernel_ir_for(DType::F16);
        k.name = "mt_fp_qmm_nax_f16".to_string();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, _t_name) in [(DType::F32, "float"), (DType::F16, "half")] {
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
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_fp_qmm_nax_{suffix}")));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
            assert!(msl.contains("gemm_ct_c"), "ct_c missing:\n{msl}");
        }
    }
}

//! `mt_qmm_nax` — production int4 quantized matmul via `mpp::tensor_ops::matmul2d`.
//!
//! This is the MPP (MetalPerformancePrimitives) counterpart of
//! `mt_qmm_mma` (the simdgroup-ladder variant). It mirrors the same algorithm — int4 weights dequantized
//! into threadgroup memory once per K-block, then a per-simdgroup matmul
//! against the fp T X-tile — but replaces the manual 8×8 `simdgroup_matmul`
//! ladder with one cooperative `matmul2d` per SG per K-block.
//!
//! Expressed entirely via `CoopTile*` IR ops — no `Op::InlineMsl`.
//! Geometry mirrors `mt_qmm_mma` (the simdgroup-ladder variant):
//!
//!   tpg = 128 = 4 SG × 32 lanes (WM = WN = 2)
//!   BM = BN = BK = 32 → 32×32 output tile (1024 outputs/TG)
//!   Grid: [N/32, M/32, 1]
//!   Per SG: one 16×16×32 `matmul2d` per K-block (acc-mode multiply_accumulate)
//!
//! Per-K-block layout (cooperative, all 128 lanes):
//!   1. X-tile coop-load → Xs[BM × TG_LD=36] (skewed for bank-conflict avoidance)
//!   2. W-tile coop-dequant int4 → Ws[BN × TG_LD=36] in fp T
//!   3. threadgroup_barrier
//!   4. Each SG calls `gemm_op.run(ct_a, ct_b, ct_c)` where ct_c persists across
//!      K-blocks (matmul2d_descriptor::mode::multiply_accumulate)
//!   5. threadgroup_barrier
//!
//! After all K-blocks, each SG stores its 16×16 fp32 ct_c into a per-SG
//! slot of `OutScratch`, then all 32 lanes coop-write it to `out` (cast to T).

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
/// Group size baked in at 64 (Qwen3.6-A3B default).
pub const GROUP_SIZE: u32 = 64;

/// Build the per-dtype [`Kernel`] IR for `mt_qmm_nax`.
///
/// Param layout (lock-step with `run_qmm_mma_mpp` in the correctness test):
///   buffer(0) = w          const device uint  *
///   buffer(1) = scales     const device {T}   *
///   buffer(2) = biases     const device {T}   *
///   buffer(3) = x          const device {T}   *
///   buffer(4) = out        device       {T}   *
///   buffer(5) = k          constant     uint  &
///   buffer(6) = n          constant     uint  &
///   buffer(7) = gs_per_row constant     uint  &
///
/// Dispatch geometry: grid `[n/32, m/32, 1]`, threadgroup `[128, 1, 1]`.
#[allow(unused_assignments)] // final nv!() bumps vid past last read — by design
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_qmm_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_qmm_nax");
    k.mode = KernelMode::Reduction;

    // All params use flat 1D indexing — rank-1 shapes satisfy type_check.
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
        name: "biases".into(),
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
    // ValueId counter shared across all three blocks so every variable name
    // is unique and the v{n} fallback names don't collide across blocks.
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

    // TG arrays (hoisted to function scope by codegen).
    b0.push_op_no_result(Op::ThreadgroupAlloc { dtype: dt, size: BM * TG_LD, name: "Xs".into() });
    b0.push_op_no_result(Op::ThreadgroupAlloc { dtype: dt, size: BN * TG_LD, name: "Ws".into() });
    // fp32 staging for ct_c.store() — 4 SG × 16×16 floats.
    b0.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: DType::F32,
        size: 4 * 16 * 16,
        name: "OutScratch".into(),
    });

    // Program IDs (axis=1 triggers tgid_y alias in Reduction mode).
    let v_tgid_y = nv!();
    b0.push_op(Op::ProgramId { axis: 1 }, v_tgid_y); // v0
    let v_tgid_x = nv!();
    b0.push_op(Op::ProgramId { axis: 0 }, v_tgid_x); // v1

    // Built-in lane/SG indices.
    let v_sg = nv!();
    b0.push_op(load0!("simd_id"), v_sg); // v2 simd_group
    let v_lane = nv!();
    b0.push_op(load0!("simd_lane"), v_lane); // v3 simd_lane

    // Constants.
    let c0 = nv!();
    b0.push_op(Op::Const { value: 0 }, c0); // v4
    let c1 = nv!();
    b0.push_op(Op::Const { value: 1 }, c1); // v5
    let c2 = nv!();
    b0.push_op(Op::Const { value: 2 }, c2); // v6
    let c3 = nv!();
    b0.push_op(Op::Const { value: 3 }, c3); // v7
    let c4 = nv!();
    b0.push_op(Op::Const { value: 4 }, c4); // v8
    let c8 = nv!();
    b0.push_op(Op::Const { value: 8 }, c8); // v9
    let c15 = nv!();
    b0.push_op(Op::Const { value: 15 }, c15); // v10
    let c16 = nv!();
    b0.push_op(Op::Const { value: 16 }, c16); // v11
    let c32 = nv!();
    b0.push_op(Op::Const { value: 32 }, c32); // v12
    let c36 = nv!();
    b0.push_op(Op::Const { value: 36 }, c36); // v13 TG_LD
    let c64 = nv!();
    b0.push_op(Op::Const { value: 64 }, c64); // v14 GROUP_SIZE
    let c256 = nv!();
    b0.push_op(Op::Const { value: 256 }, c256); // v15 per-SG scratch

    // Constexpr loads.
    let v_k = nv!();
    b0.push_op(load0!("k"), v_k); // v16
    let v_n = nv!();
    b0.push_op(load0!("n"), v_n); // v17
    let v_gs = nv!();
    b0.push_op(load0!("gs_per_row"), v_gs); // v18

    // lane_in_tg = simd_group * 32 + simd_lane.
    let v_sg32 = nv!();
    b0.push_op(bop!(Mul, v_sg, c32), v_sg32); // v19
    let v_lane_in_tg = nv!();
    b0.push_op(bop!(Add, v_sg32, v_lane), v_lane_in_tg); // v20

    // sm = simd_group / 2, sn = simd_group & 1.
    let v_sm = nv!();
    b0.push_op(bop!(Div, v_sg, c2), v_sm); // v21
    let v_sn = nv!();
    b0.push_op(bop!(BitAnd, v_sg, c1), v_sn); // v22

    // sg_m_base = sm * 16, sg_n_base = sn * 16.
    let v_sg_m_base = nv!();
    b0.push_op(bop!(Mul, v_sm, c16), v_sg_m_base); // v23
    let v_sg_n_base = nv!();
    b0.push_op(bop!(Mul, v_sn, c16), v_sg_n_base); // v24

    // Per-SG TG-buffer offsets: xs_sg_off = sg_m_base * TG_LD, ws_sg_off = sg_n_base * TG_LD.
    let v_xs_sg_off = nv!();
    b0.push_op(bop!(Mul, v_sg_m_base, c36), v_xs_sg_off); // v25
    let v_ws_sg_off = nv!();
    b0.push_op(bop!(Mul, v_sg_n_base, c36), v_ws_sg_off); // v26

    // x_m_row = lane_in_tg / 4  (= w_row; both derived from lane_in_tg/4).
    // x_k_quad = lane_in_tg & 3 (= pack_in_row).
    // x_k_base = x_k_quad * 8.
    let v_x_m_row = nv!();
    b0.push_op(bop!(Div, v_lane_in_tg, c4), v_x_m_row); // v27
    let v_x_k_quad = nv!();
    b0.push_op(bop!(BitAnd, v_lane_in_tg, c3), v_x_k_quad); // v28
    let v_x_k_base = nv!();
    b0.push_op(bop!(Mul, v_x_k_quad, c8), v_x_k_base); // v29

    // x_m_base = tgid_y * 32, w_n_base = tgid_x * 32.
    let v_x_m_base = nv!();
    b0.push_op(bop!(Mul, v_tgid_y, c32), v_x_m_base); // v30
    let v_w_n_base = nv!();
    b0.push_op(bop!(Mul, v_tgid_x, c32), v_w_n_base); // v31

    // packs_per_row = k / 8.
    let v_packs_per_row = nv!();
    b0.push_op(bop!(Div, v_k, c8), v_packs_per_row); // v32

    // wn_plus_wr = w_n_base + w_row (w_row == x_m_row == v27).
    let v_wn_wr = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_x_m_row), v_wn_wr); // v33

    // sb_base = wn_plus_wr * gs_per_row.
    let v_sb_base = nv!();
    b0.push_op(bop!(Mul, v_wn_wr, v_gs), v_sb_base); // v34

    // w_pack_row_base = wn_plus_wr * packs_per_row.
    let v_w_pack_row_base = nv!();
    b0.push_op(bop!(Mul, v_wn_wr, v_packs_per_row), v_w_pack_row_base); // v35

    // out_m_base = x_m_base + sg_m_base, out_n_base = w_n_base + sg_n_base.
    let v_out_m_base = nv!();
    b0.push_op(bop!(Add, v_x_m_base, v_sg_m_base), v_out_m_base); // v36
    let v_out_n_base = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_sg_n_base), v_out_n_base); // v37

    // sg_scratch_off = simd_group * 256 (per-SG offset into OutScratch).
    let v_sg_scratch_off = nv!();
    b0.push_op(bop!(Mul, v_sg, c256), v_sg_scratch_off); // v38

    // o_row = simd_lane / 2, o_col_base = (simd_lane & 1) * 8.
    let v_o_row = nv!();
    b0.push_op(bop!(Div, v_lane, c2), v_o_row); // v39
    let v_lane_mod2 = nv!();
    b0.push_op(bop!(BitAnd, v_lane, c1), v_lane_mod2); // v40
    let v_o_col_base = nv!();
    b0.push_op(bop!(Mul, v_lane_mod2, c8), v_o_col_base); // v41

    // CoopTile setup: descriptor (16×16×32, ta=false, tb=true) + ct objects.
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
    // Zero accumulator once before the K-loop.
    b0.push_op_no_result(Op::CoopTileZero { name: "gemm".into() });

    // K-loop: kb = 0..k step BK=32.  VarId(0) → i_0 in Block 1.
    b0.push_op_no_result(Op::Loop {
        var: VarId::new(0),
        start: c0,
        end: v_k,
        step: c32,
        body: BlockId::new(1),
    });

    // Store ct_c to per-SG slot of OutScratch.
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

    // Write loop: 32 lanes × 8 elems → 256 = 16×16. VarId(1) → i_1 in Block 2.
    b0.push_op_no_result(Op::Loop {
        var: VarId::new(1),
        start: c0,
        end: c8,
        step: c1,
        body: BlockId::new(2),
    });

    // -----------------------------------------------------------------------
    // Block 1 — K-loop body: X staging + W dequant + CoopTile ops.
    // -----------------------------------------------------------------------
    let mut b1 = Block::new(BlockId::new(1));
    let v_kb = ValueId::new(0xC000_0000); // loop var i_0

    // x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base.
    let v_xm_mr = nv!();
    b1.push_op(bop!(Add, v_x_m_base, v_x_m_row), v_xm_mr); // v42
    let v_xm_k = nv!();
    b1.push_op(bop!(Mul, v_xm_mr, v_k), v_xm_k); // v43
    let v_xm_kb = nv!();
    b1.push_op(bop!(Add, v_xm_k, v_kb), v_xm_kb); // v44
    let v_x_rdb = nv!();
    b1.push_op(bop!(Add, v_xm_kb, v_x_k_base), v_x_rdb); // v45

    // x_ws_base = x_m_row * TG_LD + x_k_base.
    let v_mr_tgld = nv!();
    b1.push_op(bop!(Mul, v_x_m_row, c36), v_mr_tgld); // v46
    let v_x_wsb = nv!();
    b1.push_op(bop!(Add, v_mr_tgld, v_x_k_base), v_x_wsb); // v47

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

    // W dequant: 1 pack per lane covers 8 nibbles.
    // pack_k_off = kb / 8 + pack_in_row.
    let v_kb8 = nv!();
    b1.push_op(bop!(Div, v_kb, c8), v_kb8); // v80
    let v_pkoff = nv!();
    b1.push_op(bop!(Add, v_kb8, v_x_k_quad), v_pkoff); // v81
    let v_pdev = nv!();
    b1.push_op(bop!(Add, v_w_pack_row_base, v_pkoff), v_pdev); // v82
    let v_pack = nv!();
    b1.push_op(
        Op::Load {
            src: "w".into(),
            indices: vec![IndexExpr::Value(v_pdev)],
            mask: None,
            other: None,
        },
        v_pack,
    ); // v83
    let v_pir8 = nv!();
    b1.push_op(bop!(Mul, v_x_k_quad, c8), v_pir8); // v84
    let v_koff = nv!();
    b1.push_op(bop!(Add, v_kb, v_pir8), v_koff); // v85
    let v_g = nv!();
    b1.push_op(bop!(Div, v_koff, c64), v_g); // v86
    let v_sbg = nv!();
    b1.push_op(bop!(Add, v_sb_base, v_g), v_sbg); // v87
    let v_scl_raw = nv!();
    b1.push_op(
        Op::Load {
            src: "scales".into(),
            indices: vec![IndexExpr::Value(v_sbg)],
            mask: None,
            other: None,
        },
        v_scl_raw,
    ); // v88
    let v_scl = nv!();
    b1.push_op(Op::Cast { value: v_scl_raw, dtype: DType::F32 }, v_scl); // v89
    let v_bia_raw = nv!();
    b1.push_op(
        Op::Load {
            src: "biases".into(),
            indices: vec![IndexExpr::Value(v_sbg)],
            mask: None,
            other: None,
        },
        v_bia_raw,
    ); // v90
    let v_bia = nv!();
    b1.push_op(Op::Cast { value: v_bia_raw, dtype: DType::F32 }, v_bia); // v91
    // ws_base = w_row * TG_LD + pack_in_row * 8.
    let v_wr36 = nv!();
    b1.push_op(bop!(Mul, v_x_m_row, c36), v_wr36); // v92
    let v_wsb = nv!();
    b1.push_op(bop!(Add, v_wr36, v_pir8), v_wsb); // v93

    // Unrolled nibble extraction: shift+mask pattern (avoids float constants).
    for (ni, shift) in [0u32, 4, 8, 12, 16, 20, 24, 28].iter().enumerate() {
        let vc_sh = nv!();
        b1.push_op(Op::Const { value: *shift as i64 }, vc_sh);
        let v_shr = nv!();
        b1.push_op(bop!(Shr, v_pack, vc_sh), v_shr);
        let v_nib = nv!();
        b1.push_op(bop!(BitAnd, v_shr, c15), v_nib);
        let v_q = nv!();
        b1.push_op(Op::Cast { value: v_nib, dtype: DType::F32 }, v_q);
        let v_sq = nv!();
        b1.push_op(bop!(Mul, v_scl, v_q), v_sq);
        let v_sqb = nv!();
        b1.push_op(bop!(Add, v_sq, v_bia), v_sqb);
        let v_wv = nv!();
        b1.push_op(Op::Cast { value: v_sqb, dtype: dt }, v_wv);
        let vc_i = nv!();
        b1.push_op(Op::Const { value: ni as i64 }, vc_i);
        let v_wi = nv!();
        b1.push_op(bop!(Add, v_wsb, vc_i), v_wi);
        b1.push_op_no_result(Op::ThreadgroupStore { name: "Ws".into(), index: v_wi, value: v_wv });
    }

    b1.push_op_no_result(Op::Barrier);

    // Per-SG cooperative-tensor load from TG staging buffers.
    // extents<TG_LD=36, 16>: stride-1 along the K dimension (inner), stride TG_LD along M/N (outer).
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
    let v_wi_loop = ValueId::new(0xC000_0001); // loop var i_1

    // col = o_col_base + i.
    let v_col = nv!();
    b2.push_op(bop!(Add, v_o_col_base, v_wi_loop), v_col);

    // scratch index = sg_scratch_off + o_row * 16 + col.
    let v_r16 = nv!();
    b2.push_op(bop!(Mul, v_o_row, c16), v_r16);
    let v_sloc = nv!();
    b2.push_op(bop!(Add, v_r16, v_col), v_sloc);
    let v_sidx = nv!();
    b2.push_op(bop!(Add, v_sg_scratch_off, v_sloc), v_sidx);

    // Load fp32 from scratch, cast to T.
    let v_f32 = nv!();
    b2.push_op(Op::ThreadgroupLoad { name: "OutScratch".into(), index: v_sidx }, v_f32);
    let v_vt = nv!();
    b2.push_op(Op::Cast { value: v_f32, dtype: dt }, v_vt);

    // Compute flat output index = (out_m_base + o_row) * n + (out_n_base + col).
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
            assert_eq!(k.name, "mt_qmm_nax");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.params[0].name, "w");
            assert_eq!(k.params[1].name, "scales");
            assert_eq!(k.params[2].name, "biases");
            assert_eq!(k.params[3].name, "x");
            assert_eq!(k.params[4].name, "out");
            assert!(k.params[4].is_output);
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
        k.name = "mt_qmm_nax_f16".to_string();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, _t_name) in [(DType::F32, "float"), (DType::F16, "half")] {
            let mut k = kernel_ir_for(dt);
            // Per-dtype naming convention used by the `tile emit` subcommand.
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            k.name = format!("mt_qmm_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_qmm_nax_{suffix}")));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
            assert!(msl.contains("gemm_ct_c"), "ct_c missing:\n{msl}");
        }
    }
}

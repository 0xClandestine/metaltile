//! `mt_steel_gemm_splitk_nax` — split-K GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the two-kernel split-K GEMM. Gated
//! behind the `nax` Cargo feature (Metal 4 / macOS 26+).
//!
//! Split-K partitions the K dimension across the grid z-axis so a
//! skinny-M / skinny-N matmul with a very large K still saturates the
//! GPU. It is a **two-kernel** dispatch:
//!
//!   1. `mt_steel_gemm_splitk_nax` — each K-split computes a partial
//!      `[M, N]` product over its slice of K via cooperative `matmul2d`
//!      and writes it (fp32) to a `[n_splits, M, N]` partials buffer.
//!   2. `mt_steel_gemm_splitk_accum_nax` — reduces the `n_splits`
//!      partial `[M, N]` matrices into the final `[M, N]` output.
//!
//! Both kernels are expressed via DSL IR ops — no `Op::InlineMsl`. The
//! split-K kernel is exactly `mt_steel_gemm_fused_nax` with a 3-D grid:
//! `tgid_z` selects the K-split and the K-loop walks only this split's
//! `[k_start, k_end)` range. The accumulator is fp32 so the cross-split
//! sum keeps full precision for f16 inputs — the partials tensor is f32
//! regardless of the operand dtype.
//!
//! Geometry mirrors `mt_steel_gemm_fused_nax`:
//!
//!   tpg = 128 = 4 SG × 32 lanes (WM = WN = 2)
//!   BM = BN = BK = 32 → 32×32 output tile
//!   Grid: [N/32, M/32, n_splits]
//!
//! ## DISPATCH INVARIANTS — split-K kernel
//!
//! - **TPG: 128 threads** (4 SG × 32 lanes). Fixed.
//! - **Grid: `[n/32, m/32, n_splits]`** — `tgid_z` = K-split index.
//! - **`m % 32 == 0`, `n % 32 == 0`, `k % 32 == 0`**; callers pad.
//! - **`k_per_split % 32 == 0`, `n_splits * k_per_split >= k`** — the
//!   K-loop is clamped to `k` so the last split may legally over-run.
//! - **`partials` is fp32, length `n_splits * m * n`**, `[split, M, N]`.
//! - **`KernelMode::Reduction`** so `tgid_*` lower to threadgroup indices.
//!
//! ## DISPATCH INVARIANTS — accum kernel
//!
//! - **One thread per `[M, N]` output element** — grid `[m*n, 1, 1]`.
//! - **`partials` length `n_splits * m * n` (fp32)**, `out` length `m*n`.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_splitk_nax_gpu_correctness.rs`.

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

/// Tile geometry — keep in lock-step with the codegen-emitted MSL.
pub const BM: u32 = 32;
pub const BN: u32 = 32;
pub const BK: u32 = 32;
/// Threads per group (4 SG × 32 lanes).
pub const TPG: u32 = 128;
/// Threadgroup-mem row skew. Stride = BK + 4 = 36.
pub const TG_SKEW: u32 = 4;
pub const TG_LD: u32 = BK + TG_SKEW; // 36

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_splitk_nax` — the
/// split-K partial GEMM (pass 1).
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = a            const device {T}    *
///   buffer(1) = b            const device {T}    *
///   buffer(2) = partials     device       float  *
///   buffer(3) = m            constant     uint   &
///   buffer(4) = k            constant     uint   &
///   buffer(5) = n            constant     uint   &
///   buffer(6) = k_per_split  constant     uint   &
///
/// Dispatch geometry: grid `[n/32, m/32, n_splits]`, threadgroup `[128, 1, 1]`.
#[allow(unused_assignments)] // final nv!() bumps vid past last read — by design
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
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "b".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    // Partials buffer is always fp32 — the accumulator dtype.
    k.params.push(Param {
        name: "partials".into(),
        dtype: DType::F32,
        shape: Shape::new([Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("m"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("k_per_split"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any, Dim::Any]));

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
    // Block 0 — preamble: program IDs, K-split range, per-lane indices, setup.
    // -----------------------------------------------------------------------
    let mut b0 = Block::new(BlockId::new(0));

    b0.push_op_no_result(Op::ThreadgroupAlloc { dtype: dt, size: BM * TG_LD, name: "Xs".into() });
    b0.push_op_no_result(Op::ThreadgroupAlloc { dtype: dt, size: BN * TG_LD, name: "Ws".into() });
    b0.push_op_no_result(Op::ThreadgroupAlloc {
        dtype: DType::F32,
        size: 4 * 16 * 16,
        name: "OutScratch".into(),
    });

    let v_tgid_y = nv!();
    b0.push_op(Op::ProgramId { axis: 1 }, v_tgid_y);
    let v_tgid_x = nv!();
    b0.push_op(Op::ProgramId { axis: 0 }, v_tgid_x);
    let v_split = nv!();
    b0.push_op(Op::ProgramId { axis: 2 }, v_split);

    let v_sg = nv!();
    b0.push_op(load0!("simd_id"), v_sg);
    let v_lane = nv!();
    b0.push_op(load0!("simd_lane"), v_lane);

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
    let c8 = nv!();
    b0.push_op(Op::Const { value: 8 }, c8);
    let c16 = nv!();
    b0.push_op(Op::Const { value: 16 }, c16);
    let c32 = nv!();
    b0.push_op(Op::Const { value: 32 }, c32);
    let c36 = nv!();
    b0.push_op(Op::Const { value: 36 }, c36); // TG_LD
    let c256 = nv!();
    b0.push_op(Op::Const { value: 256 }, c256); // per-SG scratch stride

    let v_m = nv!();
    b0.push_op(load0!("m"), v_m);
    let v_k = nv!();
    b0.push_op(load0!("k"), v_k);
    let v_n = nv!();
    b0.push_op(load0!("n"), v_n);
    let v_kps = nv!();
    b0.push_op(load0!("k_per_split"), v_kps);

    // This split's K-range: k_start = split * k_per_split,
    // k_end = min(k_start + k_per_split, k).
    let v_k_start = nv!();
    b0.push_op(bop!(Mul, v_split, v_kps), v_k_start);
    let v_k_end_raw = nv!();
    b0.push_op(bop!(Add, v_k_start, v_kps), v_k_end_raw);
    let v_k_end = nv!();
    b0.push_op(bop!(Min, v_k_end_raw, v_k), v_k_end);

    // part_base = split * m * n.
    let v_mn = nv!();
    b0.push_op(bop!(Mul, v_m, v_n), v_mn);
    let v_part_base = nv!();
    b0.push_op(bop!(Mul, v_split, v_mn), v_part_base);

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

    let v_sg_m_base = nv!();
    b0.push_op(bop!(Mul, v_sm, c16), v_sg_m_base);
    let v_sg_n_base = nv!();
    b0.push_op(bop!(Mul, v_sn, c16), v_sg_n_base);

    let v_xs_sg_off = nv!();
    b0.push_op(bop!(Mul, v_sg_m_base, c36), v_xs_sg_off);
    let v_ws_sg_off = nv!();
    b0.push_op(bop!(Mul, v_sg_n_base, c36), v_ws_sg_off);

    let v_x_m_row = nv!();
    b0.push_op(bop!(Div, v_lane_in_tg, c4), v_x_m_row);
    let v_x_k_quad = nv!();
    b0.push_op(bop!(BitAnd, v_lane_in_tg, c3), v_x_k_quad);
    let v_x_k_base = nv!();
    b0.push_op(bop!(Mul, v_x_k_quad, c8), v_x_k_base);

    let v_x_m_base = nv!();
    b0.push_op(bop!(Mul, v_tgid_y, c32), v_x_m_base);
    let v_w_n_base = nv!();
    b0.push_op(bop!(Mul, v_tgid_x, c32), v_w_n_base);

    let v_b_n = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_x_m_row), v_b_n);

    let v_mr_tgld = nv!();
    b0.push_op(bop!(Mul, v_x_m_row, c36), v_mr_tgld);
    let v_x_ws_base = nv!();
    b0.push_op(bop!(Add, v_mr_tgld, v_x_k_base), v_x_ws_base);

    let v_out_m_base = nv!();
    b0.push_op(bop!(Add, v_x_m_base, v_sg_m_base), v_out_m_base);
    let v_out_n_base = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_sg_n_base), v_out_n_base);

    let v_sg_scratch_off = nv!();
    b0.push_op(bop!(Mul, v_sg, c256), v_sg_scratch_off);

    let v_o_row = nv!();
    b0.push_op(bop!(Div, v_lane, c2), v_o_row);
    let v_lane_mod2 = nv!();
    b0.push_op(bop!(BitAnd, v_lane, c1), v_lane_mod2);
    let v_o_col_base = nv!();
    b0.push_op(bop!(Mul, v_lane_mod2, c8), v_o_col_base);

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

    // K-loop: kb = k_start .. k_end step BK=32. VarId(0) → i_0 in Block 1.
    b0.push_op_no_result(Op::Loop {
        var: VarId::new(0),
        start: v_k_start,
        end: v_k_end,
        step: c32,
        body: BlockId::new(1),
    });

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
    // Block 1 — K-loop body: A coop-load + transposed B coop-load.
    // -----------------------------------------------------------------------
    let mut b1 = Block::new(BlockId::new(1));
    let v_kb = ValueId::new(0xC000_0000); // loop var i_0

    // a_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base.
    let v_xm_mr = nv!();
    b1.push_op(bop!(Add, v_x_m_base, v_x_m_row), v_xm_mr);
    let v_xm_k = nv!();
    b1.push_op(bop!(Mul, v_xm_mr, v_k), v_xm_k);
    let v_xm_kb = nv!();
    b1.push_op(bop!(Add, v_xm_k, v_kb), v_xm_kb);
    let v_a_rdb = nv!();
    b1.push_op(bop!(Add, v_xm_kb, v_x_k_base), v_a_rdb);

    for ei in 0u32..8 {
        let vc = nv!();
        b1.push_op(Op::Const { value: ei as i64 }, vc);
        let vdi = nv!();
        b1.push_op(bop!(Add, v_a_rdb, vc), vdi);
        let vav = nv!();
        b1.push_op(
            Op::Load {
                src: "a".into(),
                indices: vec![IndexExpr::Value(vdi)],
                mask: None,
                other: None,
            },
            vav,
        );
        let vti = nv!();
        b1.push_op(bop!(Add, v_x_ws_base, vc), vti);
        b1.push_op_no_result(Op::ThreadgroupStore { name: "Xs".into(), index: vti, value: vav });
    }

    let v_b_k_base = nv!();
    b1.push_op(bop!(Add, v_kb, v_x_k_base), v_b_k_base);
    for ei in 0u32..8 {
        let vc = nv!();
        b1.push_op(Op::Const { value: ei as i64 }, vc);
        let v_bk = nv!();
        b1.push_op(bop!(Add, v_b_k_base, vc), v_bk);
        let v_bkn = nv!();
        b1.push_op(bop!(Mul, v_bk, v_n), v_bkn);
        let v_bidx = nv!();
        b1.push_op(bop!(Add, v_bkn, v_b_n), v_bidx);
        let v_bv = nv!();
        b1.push_op(
            Op::Load {
                src: "b".into(),
                indices: vec![IndexExpr::Value(v_bidx)],
                mask: None,
                other: None,
            },
            v_bv,
        );
        let v_wi = nv!();
        b1.push_op(bop!(Add, v_x_ws_base, vc), v_wi);
        b1.push_op_no_result(Op::ThreadgroupStore { name: "Ws".into(), index: v_wi, value: v_bv });
    }

    b1.push_op_no_result(Op::Barrier);

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
    // Block 2 — write loop: store fp32 scratch into this split's partial
    // slab `partials[part_base + (m_row)*n + n_col]` (no dtype narrowing —
    // partials is fp32).
    // -----------------------------------------------------------------------
    let mut b2 = Block::new(BlockId::new(2));
    let v_wi_loop = ValueId::new(0xC000_0001); // loop var i_1

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

    // partials index = part_base + (out_m_base + o_row) * n + (out_n_base + col).
    let v_orow = nv!();
    b2.push_op(bop!(Add, v_out_m_base, v_o_row), v_orow);
    let v_ocol = nv!();
    b2.push_op(bop!(Add, v_out_n_base, v_col), v_ocol);
    let v_orn = nv!();
    b2.push_op(bop!(Mul, v_orow, v_n), v_orn);
    let v_mn_idx = nv!();
    b2.push_op(bop!(Add, v_orn, v_ocol), v_mn_idx);
    let v_pidx = nv!();
    b2.push_op(bop!(Add, v_part_base, v_mn_idx), v_pidx);

    b2.push_op_no_result(Op::Store {
        dst: "partials".into(),
        indices: vec![IndexExpr::Value(v_pidx)],
        value: v_f32,
        mask: None,
    });

    let mut all_blocks = FxHashMap::default();
    all_blocks.insert(BlockId::new(0), b0.clone());
    all_blocks.insert(BlockId::new(1), b1);
    all_blocks.insert(BlockId::new(2), b2);

    k.body = b0;
    k.blocks = all_blocks;
    k
}

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_splitk_accum_nax` —
/// the split-K partial-sum reduction (pass 2).
///
/// `out[idx] = Σ_{s<n_splits} partials[s*m*n + idx]`, one thread per output
/// element. Expressed with a `StrideReduce` over the partials slab — no
/// `Op::InlineMsl`.
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = partials   const device float *
///   buffer(1) = out        device       {T}   *
///   buffer(2) = m          constant     uint  &
///   buffer(3) = n          constant     uint  &
///   buffer(4) = n_splits   constant     uint  &
///
/// Dispatch geometry: grid `[m * n, 1, 1]`, threadgroup `[1, 1, 1]`.
#[allow(unused_assignments)]
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

    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("m"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n_splits"),
        dtype: DType::U32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

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

    // Single block: idx = tgid_x; acc over n_splits stride-(m*n) hops.
    let mut b0 = Block::new(BlockId::new(0));

    let v_idx = nv!();
    b0.push_op(Op::ProgramId { axis: 0 }, v_idx);
    let v_m = nv!();
    b0.push_op(load0!("m"), v_m);
    let v_n = nv!();
    b0.push_op(load0!("n"), v_n);
    let v_ns = nv!();
    b0.push_op(load0!("n_splits"), v_ns);

    // total = m * n  (the per-split slab stride).
    let v_total = nv!();
    b0.push_op(bop!(Mul, v_m, v_n), v_total);

    // StackAlloc a 1-elem fp32 accumulator; seed it with split 0's value.
    b0.push_op_no_result(Op::StackAlloc { dtype: DType::F32, size: 1, name: "acc".into() });
    let c0 = nv!();
    b0.push_op(Op::Const { value: 0 }, c0);
    let v_seed = nv!();
    b0.push_op(
        Op::Load {
            src: "partials".into(),
            indices: vec![IndexExpr::Value(v_idx)],
            mask: None,
            other: None,
        },
        v_seed,
    );
    b0.push_op_no_result(Op::StackStore { name: "acc".into(), index: c0, value: v_seed });

    let c1 = nv!();
    b0.push_op(Op::Const { value: 1 }, c1);
    // Loop s = 1 .. n_splits: acc += partials[s*total + idx]. VarId(0) → i_0.
    b0.push_op_no_result(Op::Loop {
        var: VarId::new(0),
        start: c1,
        end: v_ns,
        step: c1,
        body: BlockId::new(1),
    });

    // out[idx] = (T) acc.
    let v_acc_final = nv!();
    b0.push_op(Op::StackLoad { name: "acc".into(), index: c0 }, v_acc_final);
    let v_out_val = nv!();
    b0.push_op(Op::Cast { value: v_acc_final, dtype: dt }, v_out_val);
    b0.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(v_idx)],
        value: v_out_val,
        mask: None,
    });

    // Block 1 — loop body: acc += partials[s * total + idx].
    let mut b1 = Block::new(BlockId::new(1));
    let v_s = ValueId::new(0xC000_0000); // loop var i_0
    let v_s_total = nv!();
    b1.push_op(bop!(Mul, v_s, v_total), v_s_total);
    let v_pidx = nv!();
    b1.push_op(bop!(Add, v_s_total, v_idx), v_pidx);
    let v_p = nv!();
    b1.push_op(
        Op::Load {
            src: "partials".into(),
            indices: vec![IndexExpr::Value(v_pidx)],
            mask: None,
            other: None,
        },
        v_p,
    );
    let v_cur = nv!();
    b1.push_op(Op::StackLoad { name: "acc".into(), index: c0 }, v_cur);
    let v_new = nv!();
    b1.push_op(bop!(Add, v_cur, v_p), v_new);
    b1.push_op_no_result(Op::StackStore { name: "acc".into(), index: c0, value: v_new });

    let mut all_blocks = FxHashMap::default();
    all_blocks.insert(BlockId::new(0), b0.clone());
    all_blocks.insert(BlockId::new(1), b1);
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
            assert_eq!(k.name, "mt_steel_gemm_splitk_nax");
            assert_eq!(k.params.len(), 3);
            assert_eq!(k.params[2].name, "partials");
            assert!(k.params[2].is_output);
            assert_eq!(k.params[2].dtype, DType::F32);
            assert_eq!(k.constexprs.len(), 4);

            assert!(!k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            let b1 = k.blocks.get(&BlockId::new(1)).expect("block 1");
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    fn accum_kernel_ir_constructs_for_f32_and_f16() {
        for dt in [DType::F32, DType::F16] {
            let k = accum_kernel_ir_for(dt);
            assert_eq!(k.name, "mt_steel_gemm_splitk_accum_nax");
            assert_eq!(k.params.len(), 2);
            assert_eq!(k.params[0].name, "partials");
            assert_eq!(k.params[1].name, "out");
            assert!(k.params[1].is_output);
            assert_eq!(k.constexprs.len(), 3);
            assert!(!k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
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
            assert!(!msl.contains("InlineMsl"));
        }
    }
}

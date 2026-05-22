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
//! Expressed entirely via the `CoopTile*` IR ops — no `Op::InlineMsl`. It
//! is exactly `mt_steel_gemm_fused_nax` with two extra integer loads
//! before the address arithmetic — the gather index of an output row is a
//! per-row scalar, the B-matrix index a per-N-block scalar. No new codegen
//! primitive is required; the redirection is ordinary arithmetic.
//!
//! Geometry mirrors `mt_steel_gemm_fused_nax`:
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

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_gather_nax`.
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
#[allow(unused_assignments)] // final nv!() bumps vid past last read — by design
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
        shape: Shape::new([Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // ValueId counter shared across all blocks. Mirrors `steel_gemm_fused_nax`.
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
    // Block 0 — preamble: program IDs, per-lane indices, gather loads, setup.
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

    let v_k = nv!();
    b0.push_op(load0!("k"), v_k);
    let v_n = nv!();
    b0.push_op(load0!("n"), v_n);

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

    // x_m_row = lane_in_tg / 4 (== w_n_row), x_k_quad = lane_in_tg & 3,
    // x_k_base = x_k_quad * 8 (== w_k_base).
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

    // ── Gather: a_src_row = lhs_indices[x_m_base + x_m_row] ──
    let v_lhs_arg = nv!();
    b0.push_op(bop!(Add, v_x_m_base, v_x_m_row), v_lhs_arg);
    let v_a_src_row = nv!();
    b0.push_op(
        Op::Load {
            src: "lhs_indices".into(),
            indices: vec![IndexExpr::Value(v_lhs_arg)],
            mask: None,
            other: None,
        },
        v_a_src_row,
    );

    // ── Gather: b_base = rhs_indices[tgid_x] * k * n ──
    let v_b_mat = nv!();
    b0.push_op(
        Op::Load {
            src: "rhs_indices".into(),
            indices: vec![IndexExpr::Value(v_tgid_x)],
            mask: None,
            other: None,
        },
        v_b_mat,
    );
    let v_b_mat_k = nv!();
    b0.push_op(bop!(Mul, v_b_mat, v_k), v_b_mat_k);
    let v_b_base = nv!();
    b0.push_op(bop!(Mul, v_b_mat_k, v_n), v_b_base);

    // b_n = w_n_base + w_n_row (w_n_row == x_m_row).
    let v_b_n = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_x_m_row), v_b_n);

    // x_ws_base = x_m_row * TG_LD + x_k_base — shared by the Xs/Ws tiles.
    let v_mr_tgld = nv!();
    b0.push_op(bop!(Mul, v_x_m_row, c36), v_mr_tgld);
    let v_x_ws_base = nv!();
    b0.push_op(bop!(Add, v_mr_tgld, v_x_k_base), v_x_ws_base);

    // out_m_base = x_m_base + sg_m_base, out_n_base = w_n_base + sg_n_base.
    let v_out_m_base = nv!();
    b0.push_op(bop!(Add, v_x_m_base, v_sg_m_base), v_out_m_base);
    let v_out_n_base = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_sg_n_base), v_out_n_base);

    // sg_scratch_off = simd_group * 256.
    let v_sg_scratch_off = nv!();
    b0.push_op(bop!(Mul, v_sg, c256), v_sg_scratch_off);

    // o_row = simd_lane / 2, o_col_base = (simd_lane & 1) * 8.
    let v_o_row = nv!();
    b0.push_op(bop!(Div, v_lane, c2), v_o_row);
    let v_lane_mod2 = nv!();
    b0.push_op(bop!(BitAnd, v_lane, c1), v_lane_mod2);
    let v_o_col_base = nv!();
    b0.push_op(bop!(Mul, v_lane_mod2, c8), v_o_col_base);

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
    b0.push_op_no_result(Op::CoopTileZero { name: "gemm".into() });

    // K-loop: kb = 0..k step BK=32. VarId(0) → i_0 in Block 1.
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
    // Block 1 — K-loop body: gathered-A coop-load + gathered-B coop-load.
    // -----------------------------------------------------------------------
    let mut b1 = Block::new(BlockId::new(1));
    let v_kb = ValueId::new(0xC000_0000); // loop var i_0

    // a_row_dev_base = a_src_row * k + kb + x_k_base.
    let v_xm_k = nv!();
    b1.push_op(bop!(Mul, v_a_src_row, v_k), v_xm_k);
    let v_xm_kb = nv!();
    b1.push_op(bop!(Add, v_xm_k, v_kb), v_xm_kb);
    let v_a_rdb = nv!();
    b1.push_op(bop!(Add, v_xm_kb, v_x_k_base), v_a_rdb);

    // Unrolled gathered-A load/store: 8 contiguous K-elems per lane.
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

    // Transposed gathered-B load: Ws[n_row, k] = b[b_base + (kb+k_base+i)*n + b_n].
    let v_b_k_base = nv!();
    b1.push_op(bop!(Add, v_kb, v_x_k_base), v_b_k_base);
    for ei in 0u32..8 {
        let vc = nv!();
        b1.push_op(Op::Const { value: ei as i64 }, vc);
        let v_bk = nv!();
        b1.push_op(bop!(Add, v_b_k_base, vc), v_bk);
        let v_bkn = nv!();
        b1.push_op(bop!(Mul, v_bk, v_n), v_bkn);
        let v_bkn_bn = nv!();
        b1.push_op(bop!(Add, v_bkn, v_b_n), v_bkn_bn);
        let v_bidx = nv!();
        b1.push_op(bop!(Add, v_b_base, v_bkn_bn), v_bidx);
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
    // Block 2 — write loop: cast fp32 scratch → T and store to `out`.
    // Destination row is the *contiguous* output row, not the gathered row.
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
            assert_eq!(k.name, "mt_steel_gemm_gather_nax");
            assert_eq!(k.params.len(), 5);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[1].name, "b");
            assert_eq!(k.params[2].name, "lhs_indices");
            assert_eq!(k.params[3].name, "rhs_indices");
            assert_eq!(k.params[4].name, "out");
            assert!(k.params[4].is_output);
            assert_eq!(k.constexprs.len(), 2);

            // Body must use CoopTile* ops, not InlineMsl.
            assert!(!k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileStoreC { .. })));

            let b1 = k.blocks.get(&BlockId::new(1)).expect("block 1");
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileLoadA { .. })));
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileLoadB { .. })));
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    /// Codegen sanity — MPP header + descriptor + the gather index loads.
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

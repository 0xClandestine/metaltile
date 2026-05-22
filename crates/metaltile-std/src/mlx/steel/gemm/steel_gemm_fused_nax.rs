//! `mt_steel_gemm_fused_nax` — plain fused GEMM via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the `nn` (non-transposed) steel-gemm
//! `C = A · B` where `A: [M, K]`, `B: [K, N]`, `C: [M, N]`, all row-major.
//! Gated behind the `nax` Cargo feature — the kernel requires the Metal 4
//! `MetalPerformancePrimitives` framework (macOS 26+).
//!
//! Expressed entirely via the `CoopTile*` IR ops — no `Op::InlineMsl`.
//! Codegen lowers `CoopTileSetup` / `Zero` / `LoadA` / `LoadB` / `Run` /
//! `StoreC` to the `mpp::tensor_ops::matmul2d` cooperative-tensor calls and
//! emits the framework include. This is the cooperative-tensor counterpart
//! of `steel_gemm_fused`; it mirrors `mt_qmm_mma_mpp`'s machinery exactly —
//! the only difference is that the B operand is already dense `T` (no int4
//! nibble-dequant): the W coop-dequant step of `mt_qmm_mma_mpp` is replaced
//! by a plain transposed coop-load of `B[K, N]`.
//!
//! Geometry mirrors `mt_qmm_mma_mpp`:
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
//! - **`KernelMode::Reduction`** so `tgid_*` lowers to the threadgroup
//!   index, not the global thread index.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_gemm_fused_nax_gpu_correctness.rs`.

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
/// Threadgroup-mem row skew — 4 elems of padding past BK to scatter
/// 32-bank conflicts on the column reads done inside `matmul2d`'s frag
/// load. Stride = BK + 4 = 36.
pub const TG_SKEW: u32 = 4;
pub const TG_LD: u32 = BK + TG_SKEW; // 36

/// Build the per-dtype [`Kernel`] IR for `mt_steel_gemm_fused_nax`.
///
/// Param layout (lock-step with the correctness test):
///   buffer(0) = a    const device {T}  *
///   buffer(1) = b    const device {T}  *
///   buffer(2) = out  device       {T}  *
///   buffer(3) = k    constant     uint &
///   buffer(4) = n    constant     uint &
///
/// Dispatch geometry: grid `[n/32, m/32, 1]`, threadgroup `[128, 1, 1]`.
#[allow(unused_assignments)] // final nv!() bumps vid past last read — by design
pub fn kernel_ir_for(dt: DType) -> Kernel {
    assert!(
        matches!(dt, DType::F32 | DType::F16),
        "mt_steel_gemm_fused_nax only supports F32 / F16, got {:?}",
        dt
    );
    let mut k = Kernel::new("mt_steel_gemm_fused_nax");
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
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("k"), dtype: DType::U32, value: None });
    k.constexprs.push(ConstExprDecl { name: ConstExpr::new("n"), dtype: DType::U32, value: None });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Any]));

    // ValueId counter shared across all blocks so v{n} fallback names are
    // unique. Mirrors the `mt_qmm_mma_mpp` construction.
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

    // TG arrays (hoisted to function scope by codegen). Xs holds A in
    // (m, k) row-major; Ws holds B transposed into (n, k) row-major so the
    // `tb=true` matmul reads it as the K×N operand.
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
    b0.push_op(Op::ProgramId { axis: 1 }, v_tgid_y);
    let v_tgid_x = nv!();
    b0.push_op(Op::ProgramId { axis: 0 }, v_tgid_x);

    // Built-in lane/SG indices.
    let v_sg = nv!();
    b0.push_op(load0!("simd_id"), v_sg);
    let v_lane = nv!();
    b0.push_op(load0!("simd_lane"), v_lane);

    // Constants.
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

    // Constexpr loads.
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

    // Per-SG TG-buffer offsets: xs_sg_off = sg_m_base * TG_LD,
    // ws_sg_off = sg_n_base * TG_LD.
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

    // b_n = w_n_base + w_n_row (w_n_row == x_m_row): the N column this lane
    // gathers from device B.
    let v_b_n = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_x_m_row), v_b_n);

    // x_ws_base = x_m_row * TG_LD + x_k_base — the TG-tile write offset,
    // shared by the A tile (Xs) and the transposed B tile (Ws).
    let v_mr_tgld = nv!();
    b0.push_op(bop!(Mul, v_x_m_row, c36), v_mr_tgld);
    let v_x_ws_base = nv!();
    b0.push_op(bop!(Add, v_mr_tgld, v_x_k_base), v_x_ws_base);

    // out_m_base = x_m_base + sg_m_base, out_n_base = w_n_base + sg_n_base.
    let v_out_m_base = nv!();
    b0.push_op(bop!(Add, v_x_m_base, v_sg_m_base), v_out_m_base);
    let v_out_n_base = nv!();
    b0.push_op(bop!(Add, v_w_n_base, v_sg_n_base), v_out_n_base);

    // sg_scratch_off = simd_group * 256 (per-SG offset into OutScratch).
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

    // Unrolled A load/store: 8 contiguous K-elems per lane.
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

    // Transposed B load: Ws[n_row, k] = B[kb + k_base + i, b_n].
    // b_k_base = kb + x_k_base; device index = (b_k_base + i) * n + b_n.
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

    // Per-SG cooperative-tensor load from the TG staging tiles.
    // extents<TG_LD=36, 16>: stride-1 along K (inner), TG_LD along M/N (outer).
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

    // Flat output index = (out_m_base + o_row) * n + (out_n_base + col).
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
            assert_eq!(k.name, "mt_steel_gemm_fused_nax");
            assert_eq!(k.params.len(), 3);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[1].name, "b");
            assert_eq!(k.params[2].name, "out");
            assert!(k.params[2].is_output);
            assert_eq!(k.constexprs.len(), 2);
            assert_eq!(k.constexprs[0].name.name(), "k");
            assert_eq!(k.constexprs[1].name.name(), "n");

            // Body must use CoopTile* ops, not InlineMsl.
            assert!(!k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileZero { .. })));
            assert!(k.body.ops.iter().any(|op| matches!(op, Op::CoopTileStoreC { .. })));

            // K-loop body (block 1) must carry CoopTileLoadA/B + Run.
            let b1 = k.blocks.get(&BlockId::new(1)).expect("block 1");
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileLoadA { .. })));
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileLoadB { .. })));
            assert!(b1.ops.iter().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    #[test]
    #[ignore]
    fn dump_msl() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = kernel_ir_for(DType::F32);
        k.name = "mt_steel_gemm_fused_nax_f32".into();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===MSL===\n{msl}\n===END===");
    }

    /// Codegen sanity — MPP header + descriptor + the 32×32 geometry.
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
            k.name = format!("mt_steel_gemm_fused_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(
                msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                "MPP include missing from generated MSL:\n{msl}"
            );
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_steel_gemm_fused_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name} Xs")));
            assert!(msl.contains(&format!("threadgroup {t_name} Ws")));
            assert!(msl.contains("tgid_y"), "tgid_y must be bound:\n{msl}");
        }
    }
}

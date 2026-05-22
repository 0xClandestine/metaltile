//! Non-power-of-2 Hadamard transform — `hadamard_m` factor M ∈ {12, 20, 28}.
//!
//! This is the second stage in MLX's `hadamard_mn_contiguous` pipeline, which
//! computes `y = H_{M·N} · x` by factoring it as `(H_M ⊗ I_N) · (I_M ⊗ H_N)`.
//! The metaltile-std version ships a **standalone** kernel for the pure M-element
//! Hadamard of any batch of M-vectors, suitable for testing and for use when the
//! batch structure has already been prepared by the power-of-2 first stage.
//!
//! ## Algorithm
//!
//! One threadgroup processes one vector of M elements:
//! 1. All M threads load their element into threadgroup memory and barrier.
//! 2. Each thread `t` accumulates `out[t] = Σ_j H_M[t][j] · buf[j]`.
//! 3. The ±1 entries of each row are encoded as a compile-time bitmask
//!    constant: bit j set = H[t][j] = +1, bit j clear = H[t][j] = −1.
//! 4. Result is scaled by `scale` and stored.
//!
//! Expressed via DSL IR ops — no `Op::InlineMsl`. The per-thread sign table
//! is a `StackAlloc` u32 array seeded with the M compile-time row masks; the
//! thread reads its own row with a dynamic `StackLoad(signs, t)`. The
//! M-element accumulation is fully unrolled (M ≤ 28).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [n_rows, 1, 1]`, `tg = [M, 1, 1]`.
//! - One threadgroup per M-element vector; `tpg = M` (12, 20, or 28).
//! - `M < 32` is safe because the kernel uses a plain threadgroup-barrier
//!   accumulate (no `simd_*` intrinsics); `simd_lane` doubles as the
//!   thread-in-threadgroup index since one partial simdgroup covers the TG.
//! - `n_rows * M` must equal the total element count of the input tensor.
//!
//! Correctness pinned by `tests/hadamard_m_gpu_correctness.rs`.
//!
//! ## Sign-bit encoding
//!
//! From Sloane's table (<http://neilsloane.com/hadamard/>), mirroring
//! `mlx/backend/common/hadamard.h`. Each entry `signs[t]` is a 32-bit
//! integer where bit j = 1 means H_M[t][j] = +1 (otherwise −1).
//!
//! Verified for orthogonality: H · H^T = M · I.

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{
        BinOpKind,
        Block,
        BlockId,
        ConstExprDecl,
        IndexExpr,
        Kernel,
        KernelMode,
        Op,
        Param,
        ParamKind,
        ValueId,
    },
    shape::{Dim, Shape},
};

// ── H_12 sign-bit encoding ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h` `h12` string.
// Verified: H_12 · H_12^T = 12 · I.
// Encoding: bit j of signs[t] = 1  ⟺  H_12[t][j] = +1.
const H12_SIGNS: [u32; 12] = [4093, 1364, 3127, 1681, 223, 2629, 883, 2329, 3523, 1129, 1807, 421];

// ── H_20 sign-bit encoding ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h` `h20` string.
// Verified: H_20 · H_20^T = 20 · I.
const H20_SIGNS: [u32; 20] = [
    445473, 859202, 702596, 389384, 747024, 641086, 234589, 469147, 938263, 828943, 984492, 953176,
    889521, 762211, 508614, 34194, 68357, 135722, 270452, 540873,
];

// ── H_28 sign-bit encoding ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h` `h28` string.
// Verified: H_28 · H_28^T = 28 · I.
const H28_SIGNS: [u32; 28] = [
    53043585, 106070914, 210061060, 153783816, 41229328, 80377888, 160739520, 79265980, 156451192,
    44483185, 88966243, 177932359, 87445519, 172810270, 125848794, 251697461, 237056618, 207758549,
    149162411, 31986518, 63972909, 3206502, 4315853, 8631579, 17246902, 34477548, 68954969,
    135812787,
];

/// Build the kernel IR for `mt_hadamard_m{M}` with M ∈ {12, 20, 28}.
///
/// The caller selects M at build time. Dispatch:
///   `grid = [n_rows, 1, 1]`, `tpg = [M, 1, 1]`, `KernelMode::Reduction`.
/// where `n_rows = total_elements / M`.
///
/// Constexpr `scale: f32` is passed as a 4-byte LE buffer under key `"scale"`.
#[allow(unused_assignments)] // final nv!() bumps vid past last read — by design
pub fn kernel_ir_for(m: u32, dt: DType) -> Kernel {
    assert!(matches!(m, 12 | 20 | 28), "mt_hadamard_m only supports M ∈ {{12, 20, 28}}, got {m}");
    assert!(
        matches!(dt, DType::F32 | DType::F16 | DType::BF16),
        "mt_hadamard_m only supports F32/F16/BF16, got {dt:?}"
    );

    let signs: &[u32] = match m {
        12 => &H12_SIGNS,
        20 => &H20_SIGNS,
        28 => &H28_SIGNS,
        _ => unreachable!(),
    };

    let name = format!("mt_hadamard_m{m}");
    let mut k = Kernel::new(&name);
    k.mode = KernelMode::Reduction;

    // inp: read-only M-element vectors (batch × M).
    k.params.push(Param {
        name: "inp".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    // out: write-only, same shape.
    k.params.push(Param {
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    // scale: f32 constexpr.
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("scale"),
        dtype: DType::F32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any]));

    let mut vid: u32 = 0;
    macro_rules! nv {
        () => {{
            let id = ValueId::new(vid);
            vid += 1;
            id
        }};
    }
    macro_rules! bop {
        ($op:ident, $l:expr, $r:expr) => {
            Op::BinOp { op: BinOpKind::$op, lhs: $l, rhs: $r }
        };
    }

    let mut b = Block::new(BlockId::new(0));

    // Thread-private sign table + threadgroup staging buffer.
    b.push_op_no_result(Op::StackAlloc { dtype: DType::U32, size: m, name: "signs".into() });
    b.push_op_no_result(Op::ThreadgroupAlloc { dtype: DType::F32, size: m, name: "buf".into() });

    let c1 = nv!();
    b.push_op(Op::Const { value: 1 }, c1);
    let v_one_f = nv!();
    b.push_op(Op::Cast { value: c1, dtype: DType::F32 }, v_one_f);

    // Per-index constants j ∈ 0..M — reused for the shift amount, the buf
    // index, and the signs-table slot.
    let mut cj: Vec<ValueId> = Vec::with_capacity(m as usize);
    for j in 0..m {
        let c = nv!();
        b.push_op(Op::Const { value: j as i64 }, c);
        cj.push(c);
    }

    // Seed the per-thread sign table with the M compile-time row masks.
    for (j, &c) in cj.iter().enumerate() {
        let cs = nv!();
        b.push_op(Op::Const { value: signs[j] as i64 }, cs);
        b.push_op_no_result(Op::StackStore { name: "signs".into(), index: c, value: cs });
    }

    // row = tgid_x, t = thread-in-threadgroup (== simd_lane for M < 32).
    let v_row = nv!();
    b.push_op(Op::ProgramId { axis: 0 }, v_row);
    let v_t = nv!();
    b.push_op(Op::Load { src: "simd_lane".into(), indices: vec![], mask: None, other: None }, v_t);

    // base = row * M, global element index = base + t.
    let v_m = nv!();
    b.push_op(Op::Const { value: m as i64 }, v_m);
    let v_base = nv!();
    b.push_op(bop!(Mul, v_row, v_m), v_base);
    let v_tg = nv!();
    b.push_op(bop!(Add, v_base, v_t), v_tg);

    // Phase 1: load this thread's element into the TG buffer (promote to f32).
    let v_inp = nv!();
    b.push_op(
        Op::Load {
            src: "inp".into(),
            indices: vec![IndexExpr::Value(v_tg)],
            mask: None,
            other: None,
        },
        v_inp,
    );
    let v_inp_f = nv!();
    b.push_op(Op::Cast { value: v_inp, dtype: DType::F32 }, v_inp_f);
    b.push_op_no_result(Op::ThreadgroupStore { name: "buf".into(), index: v_t, value: v_inp_f });
    b.push_op_no_result(Op::Barrier);

    // Phase 2: acc = Σ_j H_M[t][j] · buf[j], fully unrolled (M ≤ 28).
    // sign(t,j) = ((signs[t] >> j) & 1) ? +1 : -1 = 2·bit − 1 in f32.
    let v_signs_t = nv!();
    b.push_op(Op::StackLoad { name: "signs".into(), index: v_t }, v_signs_t);

    let mut acc: Option<ValueId> = None;
    for &c in &cj {
        let v_shifted = nv!();
        b.push_op(bop!(Shr, v_signs_t, c), v_shifted);
        let v_bit = nv!();
        b.push_op(bop!(BitAnd, v_shifted, c1), v_bit);
        let v_bitf = nv!();
        b.push_op(Op::Cast { value: v_bit, dtype: DType::F32 }, v_bitf);
        // sign = bitf + bitf − 1.0  (∈ {−1, +1}).
        let v_two_bitf = nv!();
        b.push_op(bop!(Add, v_bitf, v_bitf), v_two_bitf);
        let v_sign = nv!();
        b.push_op(bop!(Sub, v_two_bitf, v_one_f), v_sign);
        let v_bufj = nv!();
        b.push_op(Op::ThreadgroupLoad { name: "buf".into(), index: c }, v_bufj);
        let v_term = nv!();
        b.push_op(bop!(Mul, v_sign, v_bufj), v_term);
        acc = Some(match acc {
            None => v_term,
            Some(prev) => {
                let v_acc = nv!();
                b.push_op(bop!(Add, prev, v_term), v_acc);
                v_acc
            },
        });
    }
    let v_acc = acc.expect("M >= 1");

    // Phase 3: scale and store.
    let v_scale = nv!();
    b.push_op(Op::Load { src: "scale".into(), indices: vec![], mask: None, other: None }, v_scale);
    let v_scaled = nv!();
    b.push_op(bop!(Mul, v_acc, v_scale), v_scaled);
    let v_out = nv!();
    b.push_op(Op::Cast { value: v_scaled, dtype: dt }, v_out);
    b.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(v_tg)],
        value: v_out,
        mask: None,
    });

    k.body = b;
    k.sync_entry_block();
    k
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops mirror the H_m matrix math
mod tests {
    use super::*;

    #[test]
    fn kernel_ir_constructs_for_all_m_and_dtypes() {
        for m in [12u32, 20, 28] {
            for dt in [DType::F32, DType::F16, DType::BF16] {
                let k = kernel_ir_for(m, dt);
                assert_eq!(k.name, format!("mt_hadamard_m{m}"));
                assert_eq!(k.params.len(), 2);
                assert_eq!(k.params[0].name, "inp");
                assert!(!k.params[0].is_output);
                assert_eq!(k.params[1].name, "out");
                assert!(k.params[1].is_output);
                assert_eq!(k.constexprs.len(), 1);
                assert_eq!(k.constexprs[0].name.name(), "scale");
                // Body is DSL IR — no InlineMsl — with the sign table as a
                // StackAlloc seeded by StackStore.
                assert!(!k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
                assert!(k.body.ops.iter().any(|op| matches!(op, Op::StackAlloc { .. })));
                assert!(k.body.ops.iter().any(|op| matches!(op, Op::StackLoad { .. })));
            }
        }
    }

    #[test]
    #[should_panic(expected = "only supports M")]
    fn kernel_ir_rejects_invalid_m() { let _ = kernel_ir_for(16, DType::F32); }

    /// Codegen sanity — the generated MSL builds and carries the sign table.
    #[test]
    fn codegen_emits_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for m in [12u32, 20, 28] {
            let mut k = kernel_ir_for(m, DType::F32);
            k.name = format!("mt_hadamard_m{m}_f32");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(msl.contains(&format!("kernel void mt_hadamard_m{m}_f32")));
            assert!(!msl.contains("InlineMsl"));
        }
    }

    /// Verify H_12 is orthogonal: H · H^T = 12 · I.
    #[test]
    fn h12_is_orthogonal() {
        let m = 12usize;
        for i in 0..m {
            for j in 0..m {
                let dot: i32 = (0..m)
                    .map(|k| {
                        let si = if (H12_SIGNS[i] >> k) & 1 == 1 { 1i32 } else { -1 };
                        let sj = if (H12_SIGNS[j] >> k) & 1 == 1 { 1i32 } else { -1 };
                        si * sj
                    })
                    .sum();
                let expected = if i == j { m as i32 } else { 0 };
                assert_eq!(dot, expected, "H12[{i}]·H12[{j}] = {dot}, expected {expected}");
            }
        }
    }

    /// Verify H_20 is orthogonal: H · H^T = 20 · I.
    #[test]
    fn h20_is_orthogonal() {
        let m = 20usize;
        for i in 0..m {
            for j in 0..m {
                let dot: i32 = (0..m)
                    .map(|k| {
                        let si = if (H20_SIGNS[i] >> k) & 1 == 1 { 1i32 } else { -1 };
                        let sj = if (H20_SIGNS[j] >> k) & 1 == 1 { 1i32 } else { -1 };
                        si * sj
                    })
                    .sum();
                let expected = if i == j { m as i32 } else { 0 };
                assert_eq!(dot, expected, "H20[{i}]·H20[{j}] = {dot}, expected {expected}");
            }
        }
    }

    /// Verify H_28 is orthogonal: H · H^T = 28 · I.
    #[test]
    fn h28_is_orthogonal() {
        let m = 28usize;
        for i in 0..m {
            for j in 0..m {
                let dot: i32 = (0..m)
                    .map(|k| {
                        let si = if (H28_SIGNS[i] >> k) & 1 == 1 { 1i32 } else { -1 };
                        let sj = if (H28_SIGNS[j] >> k) & 1 == 1 { 1i32 } else { -1 };
                        si * sj
                    })
                    .sum();
                let expected = if i == j { m as i32 } else { 0 };
                assert_eq!(dot, expected, "H28[{i}]·H28[{j}] = {dot}, expected {expected}");
            }
        }
    }
}

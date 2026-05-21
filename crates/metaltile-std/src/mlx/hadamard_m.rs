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
//! Built as `Op::InlineMsl` rather than `#[kernel]` DSL because the DSL has no
//! mechanism to index into a compile-time per-thread constant array with a
//! dynamic thread index. The MSL uses `constant uint signs[M]` which the GPU
//! broadcasts efficiently.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [n_rows, 1, 1]`, `tg = [M, 1, 1]`.
//! - One threadgroup per M-element vector; `tpg = M` (12, 20, or 28).
//! - `M < 32` is safe because the kernel uses a plain threadgroup-barrier
//!   accumulate (no `simd_*` intrinsics).
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

use std::collections::BTreeMap;

use metaltile_core::{
    constexpr::ConstExpr,
    dtype::DType,
    ir::{Block, BlockId, ConstExprDecl, Kernel, KernelMode, Op, Param, ParamKind, ValueId},
    shape::{Dim, Shape},
};

// ── H_12 sign-bit encoding ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h` `h12` string.
// Verified: H_12 · H_12^T = 12 · I.
// Encoding: bit j of signs[t] = 1  ⟺  H_12[t][j] = +1.
//
//   row  0: +-++++++++++  → 4093
//   row  1: --+-+-+-+-+-  → 1364
//   row  2: +++-++----++  → 3127
//   row  3: +---+--+-++-  → 1681
//   row  4: +++++-++----  →  223  (Note: bit 0 = '+', only 0..11 matter)
//   row  5: +-+---+--+-+  → 2629
//   row  6: ++--+++-++--  →  883
//   row  7: +--++---+--+  → 2329
//   row  8: ++----+++-++  → 3523
//   row  9: +--+-++---+-  → 1129
//   row 10: ++++----+++-  → 1807
//   row 11: +-+--+-++---  →  421
const H12_SIGNS: [u32; 12] = [4093, 1364, 3127, 1681, 223, 2629, 883, 2329, 3523, 1129, 1807, 421];

// ── H_20 sign-bit encoding ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h` `h20` string.
// Verified: H_20 · H_20^T = 20 · I.
//
//   row  0: +----+----++--++-++-  → 445473
//   row  1: -+----+---+++---+-++  → 859202
//   row  2: --+----+---+++-+-+-+  → 702596
//   row  3: ---+----+---+++++-+-  → 389384
//   row  4: ----+----++--++-++-+  → 747024
//   row  5: -+++++-----+--+++--+  → 641086
//   row  6: +-+++-+---+-+--+++--  → 234589
//   row  7: ++-++--+---+-+--+++-  → 469147
//   row  8: +++-+---+---+-+--+++  → 938263
//   row  9: ++++-----++--+-+--++  → 828943
//   row 10: --++-+-++-+-----++++  → 984492
//   row 11: ---++-+-++-+---+-+++  → 953176
//   row 12: +---++-+-+--+--++-++  → 889521
//   row 13: ++---++-+----+-+++-+  → 762211
//   row 14: -++---++-+----+++++-  → 508614
//   row 15: -+--+--++-+----+----  →  34194
//   row 16: +-+-----++-+----+---  →  68357
//   row 17: -+-+-+---+--+----+--  → 135722
//   row 18: --+-+++------+----+-  → 270452
//   row 19: +--+--++------+----+  → 540873
const H20_SIGNS: [u32; 20] = [
    445473, 859202, 702596, 389384, 747024, 641086, 234589, 469147, 938263, 828943,
    984492, 953176, 889521, 762211, 508614, 34194, 68357, 135722, 270452, 540873,
];

// ── H_28 sign-bit encoding ─────────────────────────────────────────────────
//
// Derived from `mlx/backend/common/hadamard.h` `h28` string.
// Verified: H_28 · H_28^T = 28 · I.
//
//   row  0: +------++----++-+--+-+--++--  →  53043585
//   row  1: -+-----+++-----+-+--+-+--++-  → 106070914
//   row  2: --+-----+++---+-+-+----+--++  → 210061060
//   row  3: ---+-----+++---+-+-+-+--+--+  → 153783816
//   row  4: ----+-----+++---+-+-+++--+--  →  41229328
//   row  5: -----+-----++++--+-+--++--+-  →  80377888
//   row  6: ------++----++-+--+-+--++--+  → 160739520
//   row  7: --++++-+-------++--+++-+--+-  →  79265980
//   row  8: ---++++-+-----+-++--+-+-+--+  → 156451192
//   row  9: +---+++--+----++-++--+-+-+--  →  44483185
//   row 10: ++---++---+----++-++--+-+-+-  →  88966243
//   row 11: +++---+----+----++-++--+-+-+  → 177932359
//   row 12: ++++--------+-+--++-++--+-+-  →  87445519
//   row 13: -++++--------+++--++--+--+-+  → 172810270
//   row 14: -+-++-++--++--+--------++++-  → 125848794
//   row 15: +-+-++--+--++--+--------++++  → 251697461
//   row 16: -+-+-++--+--++--+----+---+++  → 237056618
//   row 17: +-+-+-++--+--+---+---++---++  → 207758549
//   row 18: ++-+-+-++--+------+--+++---+  → 149162411
//   row 19: -++-+-+-++--+------+-++++---  →  31986518
//   row 20: +-++-+---++--+------+-++++--  →  63972909
//   row 21: -++--++-+-++-+++----++------  →   3206502
//   row 22: +-++--++-+-++-+++-----+-----  →   4315853
//   row 23: ++-++---+-+-++-+++-----+----  →   8631579
//   row 24: -++-++-+-+-+-+--+++-----+---  →  17246902
//   row 25: --++-++++-+-+----+++-----+--  →  34477548
//   row 26: +--++-+-++-+-+----+++-----+-  →  68954969
//   row 27: ++--++-+-++-+-+----++------+  → 135812787
const H28_SIGNS: [u32; 28] = [
    53043585, 106070914, 210061060, 153783816, 41229328, 80377888, 160739520, 79265980,
    156451192, 44483185, 88966243, 177932359, 87445519, 172810270, 125848794, 251697461,
    237056618, 207758549, 149162411, 31986518, 63972909, 3206502, 4315853, 8631579,
    17246902, 34477548, 68954969, 135812787,
];

// ── MSL template ───────────────────────────────────────────────────────────
//
// The MSL source takes three template parameters filled in by `kernel_ir_for`:
//   {T}    → MSL type (float / half / bfloat)
//   {M}    → matrix size (12, 20, or 28) — a compile-time constant
//   {SIGNS} → comma-separated list of `M` u32 sign bitmasks
//
// Algorithm:
//   1. All M threads load their element into threadgroup float buf[M].
//   2. Barrier.
//   3. Each thread t accumulates acc = Σ_j sign(t,j) * buf[j]
//      where sign(t,j) = ((signs[t] >> j) & 1) ? +1 : -1.
//   4. Store (T)(acc * scale).
//
// The `constant uint signs[M]` array lives in constant address space, so
// each GPU lane reads its own row at index `tid` without a shuffle or TG op.
const MSL_TEMPLATE: &str = r#"// hadamard_m body — M={M}, one threadgroup per M-vector.
// signs[t]: bit j = 1 → H_M[t][j] = +1, bit j = 0 → H_M[t][j] = -1.
constant uint signs[{M}] = { {SIGNS} };

const uint row = tgid_x;          // threadgroup row index (0-based)
const uint t   = tid;             // thread index within the threadgroup (0..M-1)
const uint base = row * {M}u;

// Phase 1: load element into shared memory (promote to float for accuracy).
threadgroup float buf[{M}];
buf[t] = (float)(inp[base + t]);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase 2: accumulate H_M[t][*] · buf[*].
float acc = 0.0f;
for (uint j = 0u; j < {M}u; j++) {
    float sign = ((signs[t] >> j) & 1u) ? 1.0f : -1.0f;
    acc += sign * buf[j];
}

// Phase 3: scale and store.
out[base + t] = ({T})(acc * scale);
"#;

/// Substitute the three template placeholders in `MSL_TEMPLATE`.
fn build_msl(m: u32, signs: &[u32], dt: DType) -> String {
    let t_str = match dt {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        _ => unreachable!("hadamard_m only supports F32/F16/BF16"),
    };
    let signs_str: Vec<String> = signs.iter().map(|v| v.to_string()).collect();
    MSL_TEMPLATE
        .replace("{T}", t_str)
        .replace("{M}", &m.to_string())
        .replace("{SIGNS}", &signs_str.join(", "))
}

/// Build the kernel IR for `mt_hadamard_m<T>` with M ∈ {12, 20, 28}.
///
/// The caller selects M at build time. Dispatch:
///   `grid = [n_rows, 1, 1]`, `tpg = [M, 1, 1]`, `KernelMode::Reduction`.
/// where `n_rows = total_elements / M`.
///
/// Constexpr `scale: f32` is passed as a 4-byte LE buffer under key `"scale"`.
pub fn kernel_ir_for(m: u32, dt: DType) -> Kernel {
    assert!(
        matches!(m, 12 | 20 | 28),
        "mt_hadamard_m only supports M ∈ {{12, 20, 28}}, got {m}"
    );
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
        shape: Shape::new([Dim::Any, Dim::Known(m as usize)]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    // out: write-only, same shape.
    k.params.push(Param {
        name: "out".into(),
        dtype: dt,
        shape: Shape::new([Dim::Any, Dim::Known(m as usize)]),
        is_output: true,
        kind: ParamKind::Tensor,
    });

    // scale: f32 constexpr.
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("scale"),
        dtype: DType::F32,
        value: None,
    });

    k.return_shapes.push(Shape::new([Dim::Any, Dim::Known(m as usize)]));

    // Build body: Op::Load{tgid_x} to trigger the tgid_x alias, then InlineMsl.
    // Reduction mode unconditionally emits `tgid_x` for the reduction axis,
    // but the InlineMsl body also uses `tid` (thread_position_in_threadgroup).
    // The Load{tgid_x} op triggers the alias in the codegen preamble.
    let mut body = Block::new(BlockId::new(0));
    body.push_op(
        Op::Load {
            src: "tgid_x".to_string(),
            indices: Vec::new(),
            mask: None,
            other: None,
        },
        ValueId::new(0),
    );
    body.push_op_no_result(Op::InlineMsl {
        source: build_msl(m, signs, dt),
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
                assert!(k.body.ops.iter().any(|op| matches!(op, Op::InlineMsl { .. })));
            }
        }
    }

    #[test]
    #[should_panic(expected = "only supports M")]
    fn kernel_ir_rejects_invalid_m() {
        let _ = kernel_ir_for(16, DType::F32);
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

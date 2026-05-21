//! GPU correctness for the `steel_gemm_fused` kernel family —
//! `mlx::steel::gemm::steel_gemm_fused`.
//!
//! The kernel computes a plain row-major `C = A · B` (`nn` layout):
//!   A: [M, K]  B: [K, N]  C: [M, N]
//! using Apple 8×8 simdgroup-matrix MMA fragments. This file pins it
//! against a straight triple-loop fp32 CPU reference.
//!
//! ## Transpose-mode coverage
//!
//! The kernel itself only does the `nn` case. The four steel-gemm
//! transpose modes (`nn` / `nt` / `tn` / `tt`) are exercised by
//! pre-transposing A and/or B on the host so the kernel always
//! receives row-major `[M, K]` / `[K, N]` operands — then comparing
//! against the matching naive matmul. This validates that the kernel
//! produces the right answer for the data layouts each mode feeds it.
//! A prior revision loaded the B fragment with a transposed lane
//! convention and silently shipped `Bᵀ`-shaped output for every mode;
//! this test is what catches that class of bug.
//!
//! macOS-gated: needs a real Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::mlx::steel::gemm::steel_gemm_fused::{
    mt_steel_gemm_32x32x16_2x2,
    mt_steel_gemm_32x64x16_1x2,
    mt_steel_gemm_64x64x16_1x2,
    mt_steel_gemm_64x64x16_2x2,
};

/// Naive fp32 reference: `out[m, n] = sum_k a[m, k] * b[k, n]`.
/// `a` is row-major `[m_dim, k_dim]`, `b` is row-major `[k_dim, n_dim]`.
fn naive_matmul(a: &[f32], b: &[f32], m_dim: usize, k_dim: usize, n_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m_dim * n_dim];
    for mi in 0..m_dim {
        for ni in 0..n_dim {
            let mut acc = 0.0f32;
            for ki in 0..k_dim {
                acc += a[mi * k_dim + ki] * b[ki * n_dim + ni];
            }
            out[mi * n_dim + ni] = acc;
        }
    }
    out
}

/// Transpose a row-major `[rows, cols]` matrix into `[cols, rows]`.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Dispatch one `steel_gemm_fused` instantiation. `a` is `[m, k]`,
/// `b` is `[k, n]`, both row-major; returns `[m, n]`.
#[allow(clippy::too_many_arguments)]
fn run_steel_gemm(
    kernel_ir: fn(DType) -> Kernel,
    a: &[f32],
    b: &[f32],
    dt: Dt,
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    tpg: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("a".into(), pack_bytes(a, dt));
    buffers.insert("b".into(), pack_bytes(b, dt));
    buffers.insert("out".into(), vec![0u8; m * n * dt.bytes()]);
    // `m` / `n` / `k` are `#[constexpr]` params — passed as buffers.
    buffers.insert("m".into(), (m as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    // SimdGroup2D so `program_id<0/1>` map to the threadgroup index
    // (tid.x / tid.y) — see the kernel's DISPATCH INVARIANTS block.
    kernel.mode = KernelMode::SimdGroup2D;

    // Grid: one threadgroup per BM×BN output block. tpg = WM*WN*32.
    let grid = [n / bn, m / bm, 1];
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, [tpg, 1, 1])
        .expect("dispatch_with_grid");
    unpack_bytes(result.outputs.get("out").expect("out buffer"), dt)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.3e} at {at} (expected {:.6}, got {:.6})",
        expected[at],
        actual[at],
    );
}

// ── Block-shape descriptor ──────────────────────────────────────────────
struct Shape {
    name: &'static str,
    kernel_ir: fn(DType) -> Kernel,
    bm: usize,
    bn: usize,
    tpg: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "64x64x16_2x2",
        kernel_ir: mt_steel_gemm_64x64x16_2x2::kernel_ir_for,
        bm: 64,
        bn: 64,
        tpg: 128,
    },
    Shape {
        name: "32x32x16_2x2",
        kernel_ir: mt_steel_gemm_32x32x16_2x2::kernel_ir_for,
        bm: 32,
        bn: 32,
        tpg: 128,
    },
    Shape {
        name: "64x64x16_1x2",
        kernel_ir: mt_steel_gemm_64x64x16_1x2::kernel_ir_for,
        bm: 64,
        bn: 64,
        tpg: 64,
    },
    Shape {
        name: "32x64x16_1x2",
        kernel_ir: mt_steel_gemm_32x64x16_1x2::kernel_ir_for,
        bm: 32,
        bn: 64,
        tpg: 64,
    },
];

/// Run all four transpose modes for one block shape at one dtype.
///
/// The matmul is `C[M,N] = op_a(A) · op_b(B)`. The kernel only does
/// `nn`, so for each mode the host pre-transposes whatever operand the
/// mode flags and feeds the kernel row-major `[M,K]` / `[K,N]`.
fn check_shape_all_modes(shape: &Shape, dt: Dt, tol: f32) {
    let _g = gpu_lock();
    // Pick M / N as 2× the block dims so the 2-D grid has >1 block on
    // each axis (exercises `program_id<0/1>` block indexing). K spans
    // several BK=16 steps to exercise the K-accumulation loop.
    let m = shape.bm * 2;
    let n = shape.bn * 2;
    let k = 48; // 3 BK steps

    // Base operands in their *logical* `nn` orientation:
    //   a_nn : [M, K]   b_nn : [K, N]
    let a_nn = ramp(m * k, 19, 7.0);
    let b_nn = ramp(k * n, 23, 9.0);

    // Per-dtype rounding so the CPU oracle sees the same load-cast
    // quantisation the kernel does.
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| dt.round(x)).collect() };
    let a_r = round(&a_nn);
    let b_r = round(&b_nn);

    // nn — C = A · B. A:[M,K], B:[K,N], no transpose.
    {
        let expected = naive_matmul(&a_r, &b_r, m, k, n);
        let actual = run_steel_gemm(
            shape.kernel_ir,
            &a_nn,
            &b_nn,
            dt,
            m,
            n,
            k,
            shape.bm,
            shape.bn,
            shape.tpg,
        );
        assert_close(&actual, &expected, tol, &format!("{} nn {:?}", shape.name, dt.to_dtype()));
    }

    // nt — C = A · Bᵀ. Logical B is stored [N,K]; kernel needs [K,N],
    // so the host transposes the [N,K] store back into [K,N].
    {
        let b_nk = transpose(&b_r, k, n); // [N, K] — the "as-stored" Bᵀ operand
        let b_kn = transpose(&b_nk, n, k); // back to [K, N] for the kernel
        let expected = naive_matmul(&a_r, &b_kn, m, k, n);
        let actual = run_steel_gemm(
            shape.kernel_ir,
            &a_nn,
            &b_kn,
            dt,
            m,
            n,
            k,
            shape.bm,
            shape.bn,
            shape.tpg,
        );
        assert_close(&actual, &expected, tol, &format!("{} nt {:?}", shape.name, dt.to_dtype()));
    }

    // tn — C = Aᵀ · B. Logical A is stored [K,M]; kernel needs [M,K].
    {
        let a_km = transpose(&a_r, m, k); // [K, M] — the "as-stored" Aᵀ operand
        let a_mk = transpose(&a_km, k, m); // back to [M, K] for the kernel
        let expected = naive_matmul(&a_mk, &b_r, m, k, n);
        let actual = run_steel_gemm(
            shape.kernel_ir,
            &a_mk,
            &b_nn,
            dt,
            m,
            n,
            k,
            shape.bm,
            shape.bn,
            shape.tpg,
        );
        assert_close(&actual, &expected, tol, &format!("{} tn {:?}", shape.name, dt.to_dtype()));
    }

    // tt — C = Aᵀ · Bᵀ. Both operands stored transposed; host folds
    // them back to the kernel's row-major `[M,K]` / `[K,N]`.
    {
        let a_km = transpose(&a_r, m, k);
        let a_mk = transpose(&a_km, k, m);
        let b_nk = transpose(&b_r, k, n);
        let b_kn = transpose(&b_nk, n, k);
        let expected = naive_matmul(&a_mk, &b_kn, m, k, n);
        let actual = run_steel_gemm(
            shape.kernel_ir,
            &a_mk,
            &b_kn,
            dt,
            m,
            n,
            k,
            shape.bm,
            shape.bn,
            shape.tpg,
        );
        assert_close(&actual, &expected, tol, &format!("{} tt {:?}", shape.name, dt.to_dtype()));
    }
}

// ── f32 — every block shape, all four transpose modes ───────────────────
#[test]
fn steel_gemm_64x64x16_2x2_all_modes_f32() { check_shape_all_modes(&SHAPES[0], Dt::F32, 2e-3); }

#[test]
fn steel_gemm_32x32x16_2x2_all_modes_f32() { check_shape_all_modes(&SHAPES[1], Dt::F32, 2e-3); }

#[test]
fn steel_gemm_64x64x16_1x2_all_modes_f32() { check_shape_all_modes(&SHAPES[2], Dt::F32, 2e-3); }

#[test]
fn steel_gemm_32x64x16_1x2_all_modes_f32() { check_shape_all_modes(&SHAPES[3], Dt::F32, 2e-3); }

// ── f16 / bf16 — the canonical 64×64×16 / 2×2 shape ─────────────────────
// One block shape per low-precision dtype is enough: the transpose-mode
// logic is dtype-agnostic, and the f32 tests already cover all four
// shapes. f16 has a 10-bit mantissa, bf16 a 7-bit one — tolerances
// scale with the K-reduction width (48 terms here).
#[test]
fn steel_gemm_64x64x16_2x2_all_modes_f16() { check_shape_all_modes(&SHAPES[0], Dt::F16, 8e-2); }

#[test]
fn steel_gemm_64x64x16_2x2_all_modes_bf16() { check_shape_all_modes(&SHAPES[0], Dt::Bf16, 5e-1); }

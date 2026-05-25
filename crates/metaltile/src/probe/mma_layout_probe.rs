//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Empirical probe of Apple Metal's `simdgroup_matrix<f32, 8, 8>` layout.
//!
//! Computes C = A · B with A = identity (8×8) and B = label matrix
//! where `B[r, c] = r * 8 + c`. The MMA result should equal B if the
//! lane-element layout I assume matches Apple's actual convention.
//!
//! Each lane sets its frag elements per the **standard A/C convention**
//! (elem 0 at (fm, fn0), elem 1 at (fm, fn1)) and per the **B convention
//! per MLX GEMM** (elem 0 at (fn0, fm), elem 1 at (fn1, fm) — that is,
//! lane (fm, fn0) holds B[fn0, fm]).
//!
//! Output: 64 fp32 = the 8×8 C matrix, stored row-major as
//! `out[lane.fm * 8 + lane.fn0/1]`.
//!
//! Run:
//!   cargo test --release -p metaltile-std --test mma_layout_probe -- --nocapture

use crate::kernel;

#[kernel]
pub fn mt_mma_probe_a_identity_b_gemm(out: Tensor<f32>) {
    let lane = simd_lane;
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    let a = simdgroup_alloc::<f32, 8, 8>();
    let b = simdgroup_alloc::<f32, 8, 8>();
    let c = simdgroup_alloc::<f32, 8, 8>();

    // C init to 0
    simdgroup_elem_store(c, 0, 0.0f32);
    simdgroup_elem_store(c, 1, 0.0f32);

    // A = identity. Convention A: elem 0 at (fm, fn0), elem 1 at (fm, fn1).
    // A[r, c] = 1 if r == c else 0.
    let a0 = select(fm == fn0, 1.0f32, 0.0f32);
    let a1 = select(fm == fn1, 1.0f32, 0.0f32);
    simdgroup_elem_store(a, 0, a0);
    simdgroup_elem_store(a, 1, a1);

    // B = label matrix. GEMM convention: elem 0 at (fn0, fm), elem 1 at (fn1, fm).
    // B[r, c] = r * 8 + c. So B at frag-pos (fn0, fm) = fn0*8+fm.
    let b0 = (fn0 * 8u32 + fm).cast::<f32>();
    let b1 = (fn1 * 8u32 + fm).cast::<f32>();
    simdgroup_elem_store(b, 0, b0);
    simdgroup_elem_store(b, 1, b1);

    simdgroup_matmul(a, b, c);

    // Write C.elem[0/1] to out[fm*8 + fn0/fn1] per A/C convention.
    let c0 = simdgroup_elem_load(c, 0);
    let c1 = simdgroup_elem_load(c, 1);
    store(out[fm * 8u32 + fn0], c0);
    store(out[fm * 8u32 + fn1], c1);
}

#[kernel]
pub fn mt_mma_probe_a_identity_b_identity(out: Tensor<f32>) {
    let lane = simd_lane;
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    let a = simdgroup_alloc::<f32, 8, 8>();
    let b = simdgroup_alloc::<f32, 8, 8>();
    let c = simdgroup_alloc::<f32, 8, 8>();

    simdgroup_elem_store(c, 0, 0.0f32);
    simdgroup_elem_store(c, 1, 0.0f32);

    let a0 = select(fm == fn0, 1.0f32, 0.0f32);
    let a1 = select(fm == fn1, 1.0f32, 0.0f32);
    simdgroup_elem_store(a, 0, a0);
    simdgroup_elem_store(a, 1, a1);

    // B convention: elem 0 at (fm, fn0), elem 1 at (fm, fn1). Same as A.
    // B[r, c] = r*8+c. So B at (fm, fn0) = fm*8+fn0.
    let b0 = (fm * 8u32 + fn0).cast::<f32>();
    let b1 = (fm * 8u32 + fn1).cast::<f32>();
    simdgroup_elem_store(b, 0, b0);
    simdgroup_elem_store(b, 1, b1);

    simdgroup_matmul(a, b, c);

    let c0 = simdgroup_elem_load(c, 0);
    let c1 = simdgroup_elem_load(c, 1);
    store(out[fm * 8u32 + fn0], c0);
    store(out[fm * 8u32 + fn1], c1);
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Generic `argmax<T>` with u32 index output — FFAI's decode-form
//! greedy-sampler workhorse.
//!
//! Adapted from `mt_argmax_f32` (in `mlx/arg_reduce.rs`) but generic
//! over input dtype and emitting a `u32` index rather than a float-cast
//! version. Decode-form samplers (greedy token pick) need an integer
//! token id; the f32-output upstream variant doesn't fit that contract.
//!
//! Tie-breaking: strict `>` on values, smallest index on ties — matches
//! NumPy / PyTorch / MLX `argmax` semantics.
//!
//! Codegen-only — there's no MLX argmax template with the same
//! u32-output signature. Correctness validated in FFAI integration
//! tests against reference decoder output.

use metaltile::kernel;

// Tree-reduction strides: 128 → 64 → 32 → 16 → 8 → 4 → 2.
// Each iteration: threads with `lid < stride` merge the upper half into
// the lower half (take higher value; on ties take smaller index — NumPy
// argmax semantics).  Final stride-1 merge writes the result directly
// to `out[0]` and is kept inline below.
//
// Originally hand-unrolled via a `macro_rules! argmax_step!` invoked
// 7×; the proc-macro does not expand inner declarative macros, so the
// expansion silently produced no IR.  A DSL `for` loop over the seven
// stages yields identical MSL and survives the proc-macro intact.

#[kernel]
pub fn ffai_argmax<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
    let lid = tid;
    let mut best_val = neg_infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>();
            let better = v > best_val;
            if better {
                best_val = v;
                best_idx = pos;
            }
        }
    }
    threadgroup_store("tg_vals", lid, best_val);
    threadgroup_store("tg_idxs", lid, best_idx);
    threadgroup_barrier();
    // 7-stage power-of-two halving reduction over the 256-thread group.
    for _stage in range(0u32, 7u32, 1u32) {
        let stride = 128u32 >> _stage;
        if lid < stride {
            let ov = threadgroup_load("tg_vals", lid + stride);
            let oi = threadgroup_load("tg_idxs", lid + stride);
            let tv = threadgroup_load("tg_vals", lid);
            let ti = threadgroup_load("tg_idxs", lid);
            let bet = (ov > tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }
    // Final stride-1 merge writes result directly to output.
    if lid == 0u32 {
        let ov = threadgroup_load("tg_vals", 1u32);
        let oi = threadgroup_load("tg_idxs", 1u32);
        let tv = threadgroup_load("tg_vals", 0u32);
        let ti = threadgroup_load("tg_idxs", 0u32);
        let bet = (ov > tv) | ((ov == tv) & (oi < ti));
        let final_idx = select(bet, oi, ti);
        store(out[0], final_idx);
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn pack_u32_single(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn argmax(vals: &[f32]) -> u32 {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for (i, &v) in vals.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    fn make_argmax_setup(logits: Vec<f32>, expected_idx: u32, dt: DType) -> TestSetup {
        let n = logits.len();
        let mut kernel = ffai_argmax::kernel_ir_for(dt);
        kernel.mode = metaltile::core::ir::KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("out", vec![0u8; 4], DType::U32))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_u32_single(expected_idx), DType::U32))
            .grid_3d(1, 1, 1, [256, 1, 1])
    }

    #[test_kernel(name = "ffai/argmax_peak_f32", dtypes = [f32], tol = 0.0)]
    fn test_argmax_peak_f32(dt: DType) -> TestSetup {
        let mut logits = vec![0.0_f32; 1024];
        logits[777] = 100.0;
        make_argmax_setup(logits, 777, dt)
    }

    #[test_kernel(name = "ffai/argmax_ties_f32", dtypes = [f32], tol = 0.0)]
    fn test_argmax_ties_f32(dt: DType) -> TestSetup {
        let mut logits = vec![0.0_f32; 512];
        logits[42] = 10.0;
        logits[300] = 10.0;
        logits[500] = 10.0;
        make_argmax_setup(logits, 42, dt)
    }

    #[test_kernel(name = "ffai/argmax_random_f32", dtypes = [f32], tol = 0.0)]
    fn test_argmax_random_f32(dt: DType) -> TestSetup {
        let mut logits: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        logits[731] = 5.0;
        make_argmax_setup(logits, 731, dt)
    }

    #[test_kernel(name = "ffai/argmax_random_f16", dtypes = [f16], tol = 0.0)]
    fn test_argmax_random_f16(dt: DType) -> TestSetup {
        let mut logits_f32: Vec<f32> =
            (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        logits_f32[731] = 5.0;
        let logits: Vec<f32> =
            logits_f32.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let expected = argmax(&logits);
        make_argmax_setup(logits, expected, dt)
    }

    #[test_kernel(name = "ffai/argmax_random_bf16", dtypes = [bf16], tol = 0.0)]
    fn test_argmax_random_bf16(dt: DType) -> TestSetup {
        let mut logits_f32: Vec<f32> =
            (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        logits_f32[731] = 5.0;
        let logits: Vec<f32> =
            logits_f32.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let expected = argmax(&logits);
        make_argmax_setup(logits, expected, dt)
    }
}

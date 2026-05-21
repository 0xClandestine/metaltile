//! ArgReduce benchmarks — #[kernel] DSL vs MLX metal/arg_reduce.metal
//!
//! Two reductions over a flat input: `argmax` and `argmin`. Both
//! emit the winning index as a `u32` — MLX's `arg_reduce_general`
//! writes `out[out_idx] = best.index` (a `uint32_t`) regardless of
//! the input dtype, so a `u32` index buffer round-trips every index
//! exactly (a `T`-cast would lose large indices in f16 / bf16).
//!
//! Tie-breaking: strict comparison on values, smallest index on ties
//! (NumPy / PyTorch / MLX `arg_reduce` semantics).
//!
//! Both kernels are generic over `T` — f32 / f16 / bf16 all flow
//! through the same `#[kernel] fn`; values are widened to f32 for the
//! comparison so f16 / bf16 inputs reduce without precision loss.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [1, 1, 1]`, `tg = [256, 1, 1]`.
//! - TPG is fixed at 256 (8 simdgroups × 32 lanes) — the 7-stage
//!   halving tree below assumes exactly 256 threads.
//!
//! Correctness pinned by `tests/arg_reduce_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

// Tree-reduction strides: 128 → 64 → 32 → 16 → 8 → 4 → 2, then a final
// inline stride-1 merge. Each iteration merges the upper half into the
// lower half: take the winning value; on ties take the smaller index.
//
// Expressed as a DSL `for` loop over the seven stages rather than a
// hand-unrolled `macro_rules!` chain — the proc-macro does not expand
// inner declarative macros, so an unrolled inner macro would silently
// emit no IR (see docs/developing.md kernel-authoring hazards). The
// `for` loop yields identical MSL and survives the proc-macro intact.

#[bench_kernel(
    op="arg_reduce",
    subop="argmax",
    class=ArgReduce,
    n=1048576,
    check_n=4096,
    tpg=256,
    tol=0.5,
    mlx="argmax_{tn}",
    metal_file="arg_reduce.metal",
)]
#[kernel]
pub fn mt_argmax<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
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
            // argmax: take higher value; on ties take smaller index.
            let bet = (ov > tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }

    // Final stride-1 merge writes the result directly to output.
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

#[bench_kernel(
    op="arg_reduce",
    subop="argmin",
    class=ArgReduce,
    n=1048576,
    check_n=4096,
    tpg=256,
    tol=0.5,
    mlx="argmin_{tn}",
    metal_file="arg_reduce.metal",
)]
#[kernel]
pub fn mt_argmin<T>(inp: Tensor<T>, out: Tensor<u32>, #[constexpr] n: u32) {
    let lid = tid;
    // argmin seeds with +infinity so any finite value wins.
    let mut best_val = infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>();
            let better = v < best_val;
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
            // argmin: take lower value; on ties take smaller index.
            let bet = (ov < tv) | ((ov == tv) & (oi < ti));
            threadgroup_store("tg_vals", lid, select(bet, ov, tv));
            threadgroup_store("tg_idxs", lid, select(bet, oi, ti));
        }
        threadgroup_barrier();
    }

    // Final stride-1 merge writes the result directly to output.
    if lid == 0u32 {
        let ov = threadgroup_load("tg_vals", 1u32);
        let oi = threadgroup_load("tg_idxs", 1u32);
        let tv = threadgroup_load("tg_vals", 0u32);
        let ti = threadgroup_load("tg_idxs", 0u32);
        let bet = (ov < tv) | ((ov == tv) & (oi < ti));
        let final_idx = select(bet, oi, ti);
        store(out[0], final_idx);
    }
}

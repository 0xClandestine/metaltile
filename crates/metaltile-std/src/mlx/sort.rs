//! Sort benchmark — #[kernel] DSL vs MLX metal/sort.metal

use metaltile::{bench_kernel, kernel};
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[bench_kernel(
    op="sort",
    subop="sort",
    class=Sort,
    b=1024,
    n=1024,
    tpg=256,
    tol=0.0,
    mlx="c_block_sort_{tn}_{tn}_bn256_tn4",
    metal_file="sort.metal",
)]
#[kernel]
pub fn mt_sort<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let block_id = program_id::<0>();
    let t = tid;
    threadgroup_alloc("shared", 1024, T);
    let base = block_id * n;
    threadgroup_store("shared", t * 4u32, load(inp[base + t * 4u32]));
    threadgroup_store("shared", t * 4u32 + 1u32, load(inp[base + t * 4u32 + 1u32]));
    threadgroup_store("shared", t * 4u32 + 2u32, load(inp[base + t * 4u32 + 2u32]));
    threadgroup_store("shared", t * 4u32 + 3u32, load(inp[base + t * 4u32 + 3u32]));
    threadgroup_barrier();
    for _k in range(1u32, 11u32, 1u32) {
        for _jb in range(0u32, _k, 1u32) {
            let flip = _k - _jb - 1u32;
            if flip >= 7u32 {
                threadgroup_barrier();
            }
            for _e in range(0u32, 4u32, 1u32) {
                let gi = t * 4u32 + _e;
                let partner = gi ^ (1u32 << flip);
                if gi < partner {
                    let a = threadgroup_load("shared", gi);
                    let b = threadgroup_load("shared", partner);
                    let dir = (gi >> _k) & 1u32;
                    let want_swap = select(dir == 0u32, a > b, a < b);
                    threadgroup_store("shared", gi, select(want_swap, b, a));
                    threadgroup_store("shared", partner, select(want_swap, a, b));
                }
            }
        }
    }
    threadgroup_barrier();
    store(out[base + t * 4u32], threadgroup_load("shared", t * 4u32));
    store(out[base + t * 4u32 + 1u32], threadgroup_load("shared", t * 4u32 + 1u32));
    store(out[base + t * 4u32 + 2u32], threadgroup_load("shared", t * 4u32 + 2u32));
    store(out[base + t * 4u32 + 3u32], threadgroup_load("shared", t * 4u32 + 3u32));
}

// ─── mt_merge_pass ───────────────────────────────────────────────────────
//
// One pass of a bottom-up parallel merge sort — the multi-block path
// `mt_sort` (single 1024-element block) cannot reach. To sort an array
// larger than one threadgroup:
//
//   1. `mt_sort` sorts each 1024-element block independently → the
//      input is a sequence of sorted runs of length `run_len = 1024`.
//   2. `mt_merge_pass` is dispatched `ceil(log2(n / run_len))` times,
//      doubling `run_len` each call (1024 → 2048 → 4096 → …) until one
//      run spans the whole array. The host ping-pongs two buffers
//      between passes (`inp` ↔ `out`).
//
// This is the standard merge-sort recurrence; the per-pass kernel is a
// **rank merge** — embarrassingly parallel, one thread per element,
// no cross-thread cooperation (Grid3D, not Reduction):
//
//   For element `g` sitting at local position `i` inside its run, the
//   merged position is `i + (count of partner-run elements that sort
//   before it)`. The partner run is the sibling half of the merge
//   group. The count is found by a binary search over the partner run
//   (`log2(run_len)` steps, a constexpr-bounded loop). Stable: ties
//   between the two runs keep the left (lower-index) run's element
//   first, matching `Vec::sort` / MLX's stable merge.
//
// Inputs:
//   inp — [n] sequence of sorted runs, each `run_len` long
//   out — [n] the same elements, with each pair of runs merged
//
// Constexpr:
//   n        — total element count (multiple of `2 * run_len`)
//   run_len  — current sorted-run length (input). Output runs are
//              `2 * run_len`.
//   log2_run — `log2(run_len)`, the binary-search iteration count.
//              Passed explicitly so the search loop bound is a
//              compile-time constant (the DSL has no `while`).
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D** — one thread per array element, no cooperation.
//   `program_id::<0>()` is the flat element index `g`.
// - **Grid: `[n, 1, 1]`, TPG: `[1, 1, 1]`** (or any `grid·tpg == n`).
// - **`n` must be a multiple of `2 * run_len`.** A partial trailing
//   merge group is not handled — pad the input up to a power-of-two
//   multiple of `run_len` (caller's job; matches MLX `sort`).
// - **`log2_run == log2(run_len)` exactly**, and `run_len` must be a
//   power of two. A wrong `log2_run` either under-searches (wrong
//   rank) or reads past the partner run.
#[kernel]
pub fn mt_merge_pass<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] run_len: u32,
    #[constexpr] log2_run: u32,
) {
    let g = program_id::<0>();

    let merged_len = run_len * 2u32;
    // Which merge group (pair of runs) this element belongs to, and the
    // group's base offset in the flat array.
    let group = g / merged_len;
    let group_base = group * merged_len;
    // Position of `g` within its merge group, and which half (run) it
    // sits in: half 0 = left run, half 1 = right run.
    let in_group = g - group_base;
    let half = in_group / run_len;
    // Local index within the element's own run.
    let local = in_group - half * run_len;

    let my_val = load(inp[g]);

    // Binary-search the partner run for `count` — the number of partner
    // elements that sort before `my_val`. For a left-run element this is
    // the count strictly-less-than; for a right-run element it also
    // counts partner elements EQUAL to `my_val`, so the left run's
    // equal elements are placed first (a stable merge). `count` ends in
    // `[0, run_len]`.
    //
    // The search keeps a window `[lo, lo + span)` of *candidate counts*
    // (not array indices): `count` is the largest `c` such that the
    // first `c` partner elements all sort before `my_val`. Probing
    // candidate count `c` inspects partner element `c - 1` (the last of
    // the first `c`). Starting `lo = 0`, `span = run_len`, halving
    // `log2_run` times collapses the window to a single value — the
    // exact `count`, with every probe index in `[0, run_len)`.
    //
    // The partner run starts at `group_base + (1 - half) * run_len`.
    let partner_base = group_base + (1u32 - half) * run_len;
    let mut lo = 0u32;
    let mut span = run_len;
    for _s in range(0u32, log2_run, 1u32) {
        span = span / 2u32;
        // Candidate count `mid` corresponds to partner element `mid - 1`.
        let mid = lo + span;
        let probe = load(inp[partner_base + mid - 1u32]);
        // If partner element `mid - 1` sorts before `my_val`, then at
        // least `mid` partner elements do — advance the window's floor.
        let advance = select(half == 0u32, probe < my_val, probe <= my_val);
        lo = select(advance, mid, lo);
    }

    let dest = group_base + local + lo;
    store(out[dest], my_val);
}

inventory::submit! {
    BenchSpec {
        op: "sort",
        subop: "merge_pass",
        kernel_name: "mt_merge_pass",
        kernel_ir: mt_merge_pass::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0, // pure data movement — no numerical drift
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Grid3D),
    }
}

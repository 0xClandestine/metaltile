//! Scan benchmark — #[kernel] DSL vs MLX metal/scan.metal
//!
//! Two scan shapes over a `[rows, n]` input, scanned along the last
//! axis:
//!   - **inclusive** — `mt_scan`: `out[i] = Σ_{j≤i} inp[j]`.
//!   - **exclusive** — `mt_scan_exclusive`: `out[i] = Σ_{j<i} inp[j]`
//!     (`out[0] = 0`). MLX's `contig_scan_*` family carries an
//!     `exclusive` template flag for the same split.
//!
//! Both kernels share the identical two-level (per-simdgroup then
//! cross-simdgroup) prefix-sum machinery. The only difference is the
//! store stage: the inclusive kernel emits `base_prefix + s_k` (sum up
//! to and including element k), the exclusive kernel emits the prefix
//! that *precedes* element k — `base_prefix` for element 0, then
//! `base_prefix + v0 / s1 / s2`. `base_prefix` (= `cur_prefix +
//! warp_excl + thread_excl`) is already the exclusive prefix of every
//! element before this thread's 4-element group, so the exclusive
//! variant needs no extra reduction — just a one-slot store shift.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [1, rows, 1]`, `tg = [tpg, 1, 1]`.
//! - `tpg` a multiple of 32 (one full simdgroup); `n_simd ≤ 8` so the
//!   `sgs` threadgroup buffer (9 slots) covers every simdgroup plus the
//!   running-prefix slot at index `n_simd`.

use metaltile::{bench_kernel, kernel};
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

static SCAN_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];

#[bench_kernel(
    op="scan",
    subop="scan",
    class=Scan,
    shapes=&SCAN_SHAPES,
    tpg=256,
    tol=1e-3,
    mlx="contig_scan_inclusive_sum_{tn}_{tn}",
    metal_file="scan.metal",
)]
#[kernel]
pub fn mt_scan<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let lane = simd_lane;
    let sg = simd_id;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    if lid == 0 {
        threadgroup_store("sgs", ns, 0);
    }
    threadgroup_barrier();
    let zero_f = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), zero_f);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), zero_f);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), zero_f);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), zero_f);
        let s1 = v0 + v1;
        let s2 = s1 + v2;
        let s3 = s2 + v3;
        let thread_excl = simd_scan_exclusive(s3);
        if lane == 31 {
            threadgroup_store("sgs", sg, thread_excl + s3);
        }
        threadgroup_barrier();
        if sg == 0 {
            let wt = select(lane < ns, threadgroup_load("sgs", lane), zero_f);
            let wt_excl = simd_scan_exclusive(wt);
            if lane < ns {
                threadgroup_store("sgs", lane, wt_excl);
            }
        }
        threadgroup_barrier();
        let cur_prefix = threadgroup_load("sgs", ns);
        let warp_excl = threadgroup_load("sgs", sg);
        let base_prefix = cur_prefix + warp_excl + thread_excl;
        if base < n {
            store(out[row_off + base], (base_prefix + v0).cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], (base_prefix + s1).cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], (base_prefix + s2).cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], (base_prefix + s3).cast::<T>());
        }
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, base_prefix + s3);
        }
        threadgroup_barrier();
    }
}

// ── Exclusive scan ───────────────────────────────────────────────────────
//
// Identical machinery to `mt_scan`; only the store stage differs — each
// output position receives the running sum of every *strictly prior*
// element. `base_prefix` is the exclusive prefix before this thread's
// 4-element group, so element k stores `base_prefix + (sum of v0..v_{k-1})`.
//
// `BenchDispatch::Generic` because the `run_scan` bench runner hard-codes
// the inclusive-sum oracle; correctness is pinned by
// `tests/scan_exclusive_gpu_correctness.rs` instead.

#[kernel]
pub fn mt_scan_exclusive<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let lane = simd_lane;
    let sg = simd_id;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    if lid == 0 {
        threadgroup_store("sgs", ns, 0);
    }
    threadgroup_barrier();
    let zero_f = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), zero_f);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), zero_f);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), zero_f);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), zero_f);
        let s1 = v0 + v1;
        let s2 = s1 + v2;
        let s3 = s2 + v3;
        let thread_excl = simd_scan_exclusive(s3);
        if lane == 31 {
            threadgroup_store("sgs", sg, thread_excl + s3);
        }
        threadgroup_barrier();
        if sg == 0 {
            let wt = select(lane < ns, threadgroup_load("sgs", lane), zero_f);
            let wt_excl = simd_scan_exclusive(wt);
            if lane < ns {
                threadgroup_store("sgs", lane, wt_excl);
            }
        }
        threadgroup_barrier();
        let cur_prefix = threadgroup_load("sgs", ns);
        let warp_excl = threadgroup_load("sgs", sg);
        let base_prefix = cur_prefix + warp_excl + thread_excl;
        // Exclusive store: element k gets the sum of everything before it.
        // element 0 → base_prefix, element 1 → base_prefix + v0, etc.
        if base < n {
            store(out[row_off + base], base_prefix.cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], (base_prefix + v0).cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], (base_prefix + s1).cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], (base_prefix + s2).cast::<T>());
        }
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, base_prefix + s3);
        }
        threadgroup_barrier();
    }
}

inventory::submit! {
    BenchSpec {
        op: "scan",
        subop: "scan_exclusive",
        kernel_name: "mt_scan_exclusive",
        kernel_ir: mt_scan_exclusive::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

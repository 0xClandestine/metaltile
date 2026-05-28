//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile::kernel;

static SCAN_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];

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

// ── Multi-op scan variants (prod / max / min) ────────────────────────────
//
// The same two-level (per-simdgroup + cross-simdgroup) prefix-scan
// machinery from `mt_scan` / `mt_scan_exclusive`, parameterised by a
// different binary operation and identity element:
//
//   | kernel                  | op  | identity |
//   |-------------------------|-----|----------|
//   | mt_scan_prod            | ×   | 1.0      |
//   | mt_scan_prod_exclusive  | ×   | 1.0      |
//   | mt_scan_max             | max | -∞       |
//   | mt_scan_max_exclusive   | max | -∞       |
//   | mt_scan_min             | min | +∞       |
//   | mt_scan_min_exclusive   | min | +∞       |
//
// The exclusive variant stores the running prefix *before* the current
// element — identical in structure to `mt_scan_exclusive`.
//
// MLX's `contig_scan_*` family carries `op ∈ {sum, prod, max, min}` and
// an `exclusive` flag that cover all eight kernels (four ops × two
// inclusive/exclusive variants).
//
// ## Implementation strategy
//
// The DSL provides `simd_scan_exclusive` only for sum (hardware
// `simd_prefix_exclusive_sum`). For prod/max/min, both the within-SG
// prefix and the cross-SG prefix are implemented via a `"tgs"` threadgroup
// buffer of `lsize` f32 slots:
//
//   1. Each thread writes its chunk scalar (product / max / min of its 4
//      values) to `tgs[lid]`.
//   2. After a barrier, each thread reads the sequential prefix of
//      `tgs[0..lid]` — `ns ≤ 8` simdgroups × up to 32 lanes = ≤ 256
//      reads/thread, which is cheap for these ns sizes.
//   3. The cross-SG running prefix (carried between `_r` iterations via
//      `sgs[ns]`) is read and composed with the per-thread prefix.
//
// This avoids adding new simd intrinsics to the DSL while remaining
// correct for all values, including zeros and negatives.
//
// DISPATCH INVARIANTS (same as mt_scan):
//  - Reduction mode, `grid = [1, rows, 1]`, `tg = [tpg, 1, 1]`.
//  - `tpg` a multiple of 32; `n_simd ≤ 8` (so `lsize ≤ 256`).
//  - `tgs` buffer is `lsize` f32 slots; `sgs` is 9 f32 slots.

// ── Inclusive product scan ───────────────────────────────────────────────
//
// `out[i] = v[0] * v[1] * … * v[i]`  (inclusive prefix product per row).
// Identity element is 1.0; out-of-range loads are padded with 1.0.

#[kernel]
pub fn mt_scan_prod<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let ns = n_simd;
    let row_off = row * n;
    // sgs[ns] holds the running product across iterations; initialise to 1.
    threadgroup_alloc("sgs", 9);
    // tgs[lid] holds each thread's chunk scalar for sequential prefix reads.
    threadgroup_alloc("tgs", 256);
    if lid == 0 {
        threadgroup_store("sgs", ns, 1.0f32);
    }
    threadgroup_barrier();
    let one_f = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), one_f);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), one_f);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), one_f);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), one_f);
        // Thread-local inclusive prefix products.
        let p1 = v0 * v1;
        let p2 = p1 * v2;
        let p3 = p2 * v3;
        // Store this thread's chunk total for the prefix-read step.
        threadgroup_store("tgs", lid, p3);
        threadgroup_barrier();
        // Compute this thread's exclusive prefix product over tgs[0..lid].
        let mut t_excl = one_f;
        for _i in range(0u32, lid, 1u32) {
            t_excl = t_excl * threadgroup_load("tgs", _i);
        }
        let cur_prefix = threadgroup_load("sgs", ns);
        let base_prefix = cur_prefix * t_excl;
        if base < n {
            store(out[row_off + base], (base_prefix * v0).cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], (base_prefix * p1).cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], (base_prefix * p2).cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], (base_prefix * p3).cast::<T>());
        }
        threadgroup_barrier();
        // Update the running cross-chunk prefix: last thread holds the
        // inclusive product of the whole chunk.
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, base_prefix * p3);
        }
        threadgroup_barrier();
    }
}

// ── Exclusive product scan ───────────────────────────────────────────────
//
// `out[0] = 1`,  `out[i] = v[0] * … * v[i-1]`  (exclusive prefix product).

#[kernel]
pub fn mt_scan_prod_exclusive<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    threadgroup_alloc("tgs", 256);
    if lid == 0 {
        threadgroup_store("sgs", ns, 1.0f32);
    }
    threadgroup_barrier();
    let one_f = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), one_f);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), one_f);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), one_f);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), one_f);
        let p1 = v0 * v1;
        let p2 = p1 * v2;
        let p3 = p2 * v3;
        threadgroup_store("tgs", lid, p3);
        threadgroup_barrier();
        let mut t_excl = one_f;
        for _i in range(0u32, lid, 1u32) {
            t_excl = t_excl * threadgroup_load("tgs", _i);
        }
        let cur_prefix = threadgroup_load("sgs", ns);
        let base_prefix = cur_prefix * t_excl;
        // Exclusive: element k stores prefix of everything before it.
        if base < n {
            store(out[row_off + base], base_prefix.cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], (base_prefix * v0).cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], (base_prefix * p1).cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], (base_prefix * p2).cast::<T>());
        }
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, base_prefix * p3);
        }
        threadgroup_barrier();
    }
}

// ── Inclusive max scan ───────────────────────────────────────────────────
//
// `out[i] = max(v[0], …, v[i])`  (running maximum per row).
// Identity element is -∞; out-of-range loads are padded with -∞.

#[kernel]
pub fn mt_scan_max<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    threadgroup_alloc("tgs", 256);
    if lid == 0 {
        threadgroup_store("sgs", ns, neg_infinity());
    }
    threadgroup_barrier();
    let neginf = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), neginf);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), neginf);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), neginf);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), neginf);
        // Thread-local inclusive prefix maxima.
        let m1 = select(v0 > v1, v0, v1);
        let m2 = select(m1 > v2, m1, v2);
        let m3 = select(m2 > v3, m2, v3);
        // Store chunk max for the prefix-read step.
        threadgroup_store("tgs", lid, m3);
        threadgroup_barrier();
        // Exclusive prefix max over tgs[0..lid].
        let mut t_excl = neginf;
        for _i in range(0u32, lid, 1u32) {
            let v = threadgroup_load("tgs", _i);
            t_excl = select(v > t_excl, v, t_excl);
        }
        let cur_prefix = threadgroup_load("sgs", ns);
        // base_prefix = max of all elements before this thread's chunk.
        let base_prefix = select(cur_prefix > t_excl, cur_prefix, t_excl);
        // Inclusive: element k stores max of base_prefix and v[0..k].
        let out0 = select(base_prefix > v0, base_prefix, v0);
        let out1 = select(out0 > v1, out0, v1);
        let out2 = select(out1 > v2, out1, v2);
        let out3 = select(out2 > v3, out2, v3);
        if base < n {
            store(out[row_off + base], out0.cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], out1.cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], out2.cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], out3.cast::<T>());
        }
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, out3);
        }
        threadgroup_barrier();
    }
}

// ── Exclusive max scan ───────────────────────────────────────────────────
//
// `out[0] = -∞`,  `out[i] = max(v[0], …, v[i-1])`  (exclusive max prefix).

#[kernel]
pub fn mt_scan_max_exclusive<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    threadgroup_alloc("tgs", 256);
    if lid == 0 {
        threadgroup_store("sgs", ns, neg_infinity());
    }
    threadgroup_barrier();
    let neginf = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), neginf);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), neginf);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), neginf);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), neginf);
        let m1 = select(v0 > v1, v0, v1);
        let m2 = select(m1 > v2, m1, v2);
        let m3 = select(m2 > v3, m2, v3);
        threadgroup_store("tgs", lid, m3);
        threadgroup_barrier();
        let mut t_excl = neginf;
        for _i in range(0u32, lid, 1u32) {
            let v = threadgroup_load("tgs", _i);
            t_excl = select(v > t_excl, v, t_excl);
        }
        let cur_prefix = threadgroup_load("sgs", ns);
        let base_prefix = select(cur_prefix > t_excl, cur_prefix, t_excl);
        // Exclusive: element k stores max of base_prefix and v[0..k-1].
        let ep1 = select(base_prefix > v0, base_prefix, v0);
        let ep2 = select(ep1 > v1, ep1, v1);
        let ep3 = select(ep2 > v2, ep2, v2);
        if base < n {
            store(out[row_off + base], base_prefix.cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], ep1.cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], ep2.cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], ep3.cast::<T>());
        }
        threadgroup_barrier();
        // Running prefix = inclusive max of the whole chunk.
        let chunk_max = select(base_prefix > m3, base_prefix, m3);
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, chunk_max);
        }
        threadgroup_barrier();
    }
}

// ── Inclusive min scan ───────────────────────────────────────────────────
//
// `out[i] = min(v[0], …, v[i])`  (running minimum per row).
// Identity element is +∞; out-of-range loads are padded with +∞.

#[kernel]
pub fn mt_scan_min<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    threadgroup_alloc("tgs", 256);
    if lid == 0 {
        threadgroup_store("sgs", ns, infinity());
    }
    threadgroup_barrier();
    let posinf = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), posinf);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), posinf);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), posinf);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), posinf);
        // Thread-local inclusive prefix minima.
        let m1 = select(v0 < v1, v0, v1);
        let m2 = select(m1 < v2, m1, v2);
        let m3 = select(m2 < v3, m2, v3);
        threadgroup_store("tgs", lid, m3);
        threadgroup_barrier();
        let mut t_excl = posinf;
        for _i in range(0u32, lid, 1u32) {
            let v = threadgroup_load("tgs", _i);
            t_excl = select(v < t_excl, v, t_excl);
        }
        let cur_prefix = threadgroup_load("sgs", ns);
        let base_prefix = select(cur_prefix < t_excl, cur_prefix, t_excl);
        let out0 = select(base_prefix < v0, base_prefix, v0);
        let out1 = select(out0 < v1, out0, v1);
        let out2 = select(out1 < v2, out1, v2);
        let out3 = select(out2 < v3, out2, v3);
        if base < n {
            store(out[row_off + base], out0.cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], out1.cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], out2.cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], out3.cast::<T>());
        }
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, out3);
        }
        threadgroup_barrier();
    }
}

// ── Exclusive min scan ───────────────────────────────────────────────────
//
// `out[0] = +∞`,  `out[i] = min(v[0], …, v[i-1])`  (exclusive min prefix).

#[kernel]
pub fn mt_scan_min_exclusive<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    threadgroup_alloc("tgs", 256);
    if lid == 0 {
        threadgroup_store("sgs", ns, infinity());
    }
    threadgroup_barrier();
    let posinf = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]).cast::<f32>(), posinf);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]).cast::<f32>(), posinf);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]).cast::<f32>(), posinf);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]).cast::<f32>(), posinf);
        let m1 = select(v0 < v1, v0, v1);
        let m2 = select(m1 < v2, m1, v2);
        let m3 = select(m2 < v3, m2, v3);
        threadgroup_store("tgs", lid, m3);
        threadgroup_barrier();
        let mut t_excl = posinf;
        for _i in range(0u32, lid, 1u32) {
            let v = threadgroup_load("tgs", _i);
            t_excl = select(v < t_excl, v, t_excl);
        }
        let cur_prefix = threadgroup_load("sgs", ns);
        let base_prefix = select(cur_prefix < t_excl, cur_prefix, t_excl);
        // Exclusive: element k stores min of base_prefix and v[0..k-1].
        let ep1 = select(base_prefix < v0, base_prefix, v0);
        let ep2 = select(ep1 < v1, ep1, v1);
        let ep3 = select(ep2 < v2, ep2, v2);
        if base < n {
            store(out[row_off + base], base_prefix.cast::<T>());
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], ep1.cast::<T>());
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], ep2.cast::<T>());
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], ep3.cast::<T>());
        }
        threadgroup_barrier();
        let chunk_min = select(base_prefix < m3, base_prefix, m3);
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, chunk_min);
        }
        threadgroup_barrier();
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }

    // ── CPU oracles ──────────────────────────────────────────────────────

    fn cpu_scan_sum_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..n {
                out[r * n + c] = acc;
                acc += inp[r * n + c];
            }
        }
        out
    }

    fn cpu_scan_prod(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = 1.0f32;
            for c in 0..n {
                acc *= inp[r * n + c];
                out[r * n + c] = acc;
            }
        }
        out
    }

    fn cpu_scan_prod_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = 1.0f32;
            for c in 0..n {
                out[r * n + c] = acc;
                acc *= inp[r * n + c];
            }
        }
        out
    }

    fn cpu_scan_max(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = f32::NEG_INFINITY;
            for c in 0..n {
                if inp[r * n + c] > acc {
                    acc = inp[r * n + c];
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    fn cpu_scan_max_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = f32::NEG_INFINITY;
            for c in 0..n {
                out[r * n + c] = acc;
                if inp[r * n + c] > acc {
                    acc = inp[r * n + c];
                }
            }
        }
        out
    }

    fn cpu_scan_min(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = f32::INFINITY;
            for c in 0..n {
                if inp[r * n + c] < acc {
                    acc = inp[r * n + c];
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    fn cpu_scan_min_exclusive(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = f32::INFINITY;
            for c in 0..n {
                out[r * n + c] = acc;
                if inp[r * n + c] < acc {
                    acc = inp[r * n + c];
                }
            }
        }
        out
    }

    // ── Helper to build a scan TestSetup ─────────────────────────────────

    fn make_scan_setup(
        inp: Vec<f32>,
        expected: Vec<f32>,
        rows: usize,
        n: usize,
        kernel: metaltile_core::ir::Kernel,
    ) -> TestSetup {
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp), DType::F32))
            .input(TestBuffer::from_vec("out", pack_f32(&vec![0.0f32; rows * n]), DType::F32))
            .input(TestBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected), DType::F32))
            .grid_3d(1, rows as u32, 1, [256, 1, 1])
    }

    fn reduction_kernel(mut k: metaltile_core::ir::Kernel) -> metaltile_core::ir::Kernel {
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        k
    }

    // ── Exclusive sum ─────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/scan_exclusive_aligned", dtypes = [f32], tol = 2e-3)]
    fn test_scan_exclusive_aligned(dt: DType) -> TestSetup {
        let (rows, n) = (3usize, 2048usize);
        let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();
        let expected = cpu_scan_sum_exclusive(&inp, rows, n);
        make_scan_setup(
            inp,
            expected,
            rows,
            n,
            reduction_kernel(mt_scan_exclusive::kernel_ir_for(dt)),
        )
    }

    #[test_kernel(name = "mlx/scan_exclusive_ragged", dtypes = [f32], tol = 3e-3)]
    fn test_scan_exclusive_ragged(dt: DType) -> TestSetup {
        let (rows, n) = (2usize, 3000usize);
        let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let expected = cpu_scan_sum_exclusive(&inp, rows, n);
        make_scan_setup(
            inp,
            expected,
            rows,
            n,
            reduction_kernel(mt_scan_exclusive::kernel_ir_for(dt)),
        )
    }

    // ── Product scan ──────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/scan_prod_aligned", dtypes = [f32], tol = 0.01)]
    fn test_scan_prod_aligned(dt: DType) -> TestSetup {
        let (rows, n) = (3, 1024);
        let inp: Vec<f32> = (0..rows * n).map(|i| 0.9 + (i % 5) as f32 * 0.05).collect();
        let expected = cpu_scan_prod(&inp, rows, n);
        make_scan_setup(inp, expected, rows, n, reduction_kernel(mt_scan_prod::kernel_ir_for(dt)))
    }

    #[test_kernel(name = "mlx/scan_prod_exclusive", dtypes = [f32], tol = 0.01)]
    fn test_scan_prod_exclusive(dt: DType) -> TestSetup {
        let (rows, n) = (2, 512);
        let inp: Vec<f32> = (0..rows * n).map(|i| 0.9 + (i % 7) as f32 * 0.03).collect();
        let expected = cpu_scan_prod_exclusive(&inp, rows, n);
        make_scan_setup(
            inp,
            expected,
            rows,
            n,
            reduction_kernel(mt_scan_prod_exclusive::kernel_ir_for(dt)),
        )
    }

    // ── Max scan ──────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/scan_max_aligned", dtypes = [f32], tol = 1e-4)]
    fn test_scan_max_aligned(dt: DType) -> TestSetup {
        let (rows, n) = (3, 1024);
        let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 37 + 11) % 200) as f32 - 100.0).collect();
        let expected = cpu_scan_max(&inp, rows, n);
        make_scan_setup(inp, expected, rows, n, reduction_kernel(mt_scan_max::kernel_ir_for(dt)))
    }

    #[test_kernel(name = "mlx/scan_max_exclusive", dtypes = [f32], tol = 1e-4)]
    fn test_scan_max_exclusive(dt: DType) -> TestSetup {
        let (rows, n) = (2, 512);
        let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 53 + 7) % 200) as f32 - 100.0).collect();
        let expected = cpu_scan_max_exclusive(&inp, rows, n);
        make_scan_setup(
            inp,
            expected,
            rows,
            n,
            reduction_kernel(mt_scan_max_exclusive::kernel_ir_for(dt)),
        )
    }

    // ── Min scan ──────────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/scan_min_aligned", dtypes = [f32], tol = 1e-4)]
    fn test_scan_min_aligned(dt: DType) -> TestSetup {
        let (rows, n) = (3, 1024);
        let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 37 + 11) % 200) as f32 - 100.0).collect();
        let expected = cpu_scan_min(&inp, rows, n);
        make_scan_setup(inp, expected, rows, n, reduction_kernel(mt_scan_min::kernel_ir_for(dt)))
    }

    #[test_kernel(name = "mlx/scan_min_exclusive", dtypes = [f32], tol = 1e-4)]
    fn test_scan_min_exclusive(dt: DType) -> TestSetup {
        let (rows, n) = (2, 512);
        let inp: Vec<f32> = (0..rows * n).map(|i| ((i * 53 + 7) % 200) as f32 - 100.0).collect();
        let expected = cpu_scan_min_exclusive(&inp, rows, n);
        make_scan_setup(
            inp,
            expected,
            rows,
            n,
            reduction_kernel(mt_scan_min_exclusive::kernel_ir_for(dt)),
        )
    }
}

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile_core::{DType, bench::BenchSetup};

    use super::*;

    #[bench(name = "scan/scan", dtypes = [f32, f16, bf16])]
    fn bench_mt_scan(dt: DType) -> BenchSetup {
        crate::mlx::benches::bench_row_op_y(mt_scan::kernel_ir_for(dt), dt, 1024, 4096, 256)
    }
}

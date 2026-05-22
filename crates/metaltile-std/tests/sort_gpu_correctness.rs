//! GPU correctness for `mlx::sort` — single-block bitonic sort plus the
//! multi-block bottom-up merge path.
//!
//! `mt_sort<T>` sorts each block of `n=1024` elements in-place using a
//! bitonic sort network in shared memory. `mt_merge<T>` then runs one
//! bottom-up merge pass: it merges adjacent sorted runs of length `run`
//! into sorted runs of length `2*run`. Running `mt_sort` followed by
//! `log2(n_blocks)` `mt_merge` passes fully sorts an array of
//! `n_blocks * 1024` elements.
//!
//! ## DISPATCH INVARIANTS (mt_sort)
//! - **TPG: 256 threads** (each thread processes 4 elements).
//! - **n = TPG * 4 = 1024** (elements per block — hardcoded in the kernel).
//! - **Grid: 1 threadgroup per block** (1D, program_id<0> = block index).
//!
//! ## DISPATCH INVARIANTS (mt_merge)
//! - **Grid3D / Elementwise**: one thread per output element over the
//!   whole `n`-element array; `grid_x = ceil(n / tpg)`.
//! - `run` = current sorted-run length; `log_steps` must satisfy
//!   `2^log_steps >= 2*run` so the co-rank binary search converges.
//! - Input holds sorted runs of length `run`; output is a separate
//!   buffer (caller ping-pongs).
//!
//! CPU oracle: `Vec::sort_unstable_by` — defines the expected order.
//! Multi-block dispatch: grid_x = number of independent blocks.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::sort::{mt_merge, mt_sort};

/// Dispatch `mt_sort` over `n_blocks` independent blocks of `N=1024` elements.
fn run_sort(inp: &[f32], dt: Dt, n_blocks: usize) -> Vec<f32> {
    // N per block must equal 1024 (TPG=256, 4 elems/thread).
    const N: usize = 1024;
    assert_eq!(inp.len(), n_blocks * N, "input must be exactly n_blocks * 1024 elements");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; inp.len()], dt));
    buffers.insert("n".into(), (N as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_sort::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // 1 threadgroup per block, 256 threads per threadgroup.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_blocks, 1, 1], [256, 1, 1])
        .expect("sort dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_blocks * N);
    out
}

/// CPU oracle for a single block: sort in ascending order.
fn cpu_sort_block(block: &[f32]) -> Vec<f32> {
    let mut v: Vec<f32> = block.to_vec();
    v.sort_unstable_by(f32::total_cmp);
    v
}

#[test]
fn sort_single_block_matches_cpu_f32() {
    let _g = gpu_lock();
    const N: usize = 1024;
    // Reverse-sorted input is the worst case for many sort algorithms.
    let inp: Vec<f32> = (0..N).rev().map(|i| i as f32 * 0.1).collect();
    let expected = cpu_sort_block(&inp);
    let actual = run_sort(&inp, Dt::F32, 1);

    assert!(actual.iter().any(|&v| v != 0.0), "sort output all zeros — empty kernel body?");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "sort mismatch at [{i}]: expected {e:.4}, got {a:.4}");
    }
}

#[test]
fn sort_single_block_random_f32() {
    let _g = gpu_lock();
    const N: usize = 1024;
    // Pseudo-random pattern via the ramp helper — avoids all-equal or monotone.
    let inp: Vec<f32> = (0..N).map(|i| ((i * 37 + 13) % 100) as f32 * 0.1 - 5.0).collect();
    let expected = cpu_sort_block(&inp);
    let actual = run_sort(&inp, Dt::F32, 1);

    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "sort random mismatch at [{i}]: expected {e:.4}, got {a:.4}");
    }
}

#[test]
fn sort_two_independent_blocks_f32() {
    let _g = gpu_lock();
    const N: usize = 1024;
    // Two blocks with different input patterns — verify per-block independence.
    let block0: Vec<f32> = (0..N).rev().map(|i| i as f32).collect();
    let block1: Vec<f32> = (0..N).map(|i| ((i * 53 + 7) % 1000) as f32 * 0.01).collect();
    let inp: Vec<f32> = block0.iter().chain(block1.iter()).copied().collect();

    let expected0 = cpu_sort_block(&block0);
    let expected1 = cpu_sort_block(&block1);

    let actual = run_sort(&inp, Dt::F32, 2);
    let (actual0, actual1) = actual.split_at(N);

    for (i, (e, a)) in expected0.iter().zip(actual0.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "sort block0 mismatch at [{i}]");
    }
    for (i, (e, a)) in expected1.iter().zip(actual1.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "sort block1 mismatch at [{i}]");
    }
}

#[test]
fn sort_single_block_f16() {
    let _g = gpu_lock();
    const N: usize = 1024;
    // Values representable exactly in f16 — avoids rounding confusion.
    let inp: Vec<f32> = (0..N).map(|i| Dt::F16.round(((N - 1 - i) as f32) * 0.1)).collect();
    let expected = cpu_sort_block(&inp);
    let actual = run_sort(&inp, Dt::F16, 1);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-3, "sort f16 mismatch at [{i}]: expected {e:.4}, got {a:.4}");
    }
}

#[test]
fn sort_output_is_non_decreasing_f32() {
    let _g = gpu_lock();
    const N: usize = 1024;
    let inp: Vec<f32> = (0..N).map(|i| ((i * 97 + 31) % 200) as f32 - 100.0).collect();
    let actual = run_sort(&inp, Dt::F32, 1);
    for window in actual.windows(2) {
        assert!(window[0] <= window[1], "sort output not non-decreasing at {:?}", window);
    }
}

// ── Multi-block merge path ───────────────────────────────────────────────

/// Elements per block — `mt_sort`'s hardcoded TPG*4.
const BLOCK: usize = 1024;

/// Run one `mt_merge` pass over `inp` (which holds sorted runs of
/// length `run`), producing runs of length `2*run`. `n` is the logical
/// element count; the input may be padded with `+∞`-equivalent
/// sentinels past `n` (we pad with a large value so the partial-run
/// clamp logic is exercised even on padded buffers).
fn run_merge_pass(ctx: &Context, inp: &[f32], dt: Dt, n: usize, run: usize) -> Vec<f32> {
    // `log_steps` must satisfy 2^log_steps >= 2*run. The binary search
    // range is at most `run`, so any value with 2^log_steps >= 2*run is
    // safe; 20 covers runs up to 512K elements.
    const LOG_STEPS: u32 = 20;
    assert!((1u64 << LOG_STEPS) >= (2 * run) as u64, "log_steps too small for run={run}");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; inp.len()], dt));
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("run".into(), (run as u32).to_le_bytes().to_vec());
    buffers.insert("log_steps".into(), LOG_STEPS.to_le_bytes().to_vec());

    let mut kernel = mt_merge::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // One thread per output element; any TPG works for Grid3D.
    const TPG: usize = 256;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n.div_ceil(TPG), 1, 1], [
            TPG, 1, 1,
        ])
        .expect("merge dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(inp.len());
    out
}

/// Full multi-block sort: per-block bitonic sort, then `ceil(log2)`
/// merge passes that double the run length until it covers all of `n`.
/// `inp.len()` must be a multiple of `BLOCK` (the per-block sort needs
/// whole 1024-element blocks); `n <= inp.len()` is the logical length —
/// values in `[n, inp.len())` are padding and must compare *greater*
/// than every real element so they sink to the tail.
fn run_multiblock_sort(inp: &[f32], dt: Dt, n: usize) -> Vec<f32> {
    assert_eq!(inp.len() % BLOCK, 0, "input must be a whole number of 1024-element blocks");
    let n_blocks = inp.len() / BLOCK;
    let ctx = Context::new().expect("Context::new on macOS");

    // Stage 1: sort each 1024-element block independently.
    let mut data = run_sort(inp, dt, n_blocks);

    // Stage 2: merge passes. After pass k the sorted-run length is
    // `BLOCK * 2^(k+1)`; stop once one run covers the whole array.
    let mut run = BLOCK;
    while run < inp.len() {
        data = run_merge_pass(&ctx, &data, dt, n, run);
        run *= 2;
    }
    data.truncate(n);
    data
}

/// CPU oracle for the full array: ascending sort.
fn cpu_sort_full(v: &[f32]) -> Vec<f32> {
    let mut s = v.to_vec();
    s.sort_unstable_by(f32::total_cmp);
    s
}

/// Build a `n_blocks * 1024` reverse-sorted input — worst case for the
/// merge path (every element must move across every run boundary).
fn reverse_input(n_blocks: usize) -> Vec<f32> {
    let total = n_blocks * BLOCK;
    (0..total).rev().map(|i| i as f32).collect()
}

#[test]
fn sort_two_blocks_merge_matches_cpu_f32() {
    let _g = gpu_lock();
    let inp = reverse_input(2);
    let expected = cpu_sort_full(&inp);
    let actual = run_multiblock_sort(&inp, Dt::F32, inp.len());
    assert!(actual.iter().any(|&v| v != 0.0), "merge output all zeros — empty kernel body?");
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-6, "2-block merge mismatch at [{i}]: expected {e}, got {a}");
    }
}

#[test]
fn sort_four_blocks_merge_matches_cpu_f32() {
    let _g = gpu_lock();
    let inp = reverse_input(4);
    let expected = cpu_sort_full(&inp);
    let actual = run_multiblock_sort(&inp, Dt::F32, inp.len());
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-6, "4-block merge mismatch at [{i}]: expected {e}, got {a}");
    }
}

#[test]
fn sort_eight_blocks_merge_matches_cpu_f32() {
    let _g = gpu_lock();
    // Pseudo-random pattern across 8 blocks — exercises 3 merge passes
    // with elements interleaved across many run boundaries.
    let inp: Vec<f32> = (0..8 * BLOCK)
        .map(|i| ((i * 2_654_435_761usize) % 1_000_003) as f32 * 0.001 - 500.0)
        .collect();
    let expected = cpu_sort_full(&inp);
    let actual = run_multiblock_sort(&inp, Dt::F32, inp.len());
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-6, "8-block merge mismatch at [{i}]: expected {e}, got {a}");
    }
}

#[test]
fn sort_multiblock_output_is_non_decreasing_f32() {
    let _g = gpu_lock();
    let inp = reverse_input(4);
    let actual = run_multiblock_sort(&inp, Dt::F32, inp.len());
    for w in actual.windows(2) {
        assert!(w[0] <= w[1], "multi-block sort output not non-decreasing: {:?}", w);
    }
}

#[test]
fn sort_eight_blocks_merge_f16() {
    let _g = gpu_lock();
    // f16-exact values so the oracle and kernel agree bit-for-bit.
    let inp: Vec<f32> =
        (0..8 * BLOCK).map(|i| Dt::F16.round(((8 * BLOCK - 1 - i) as f32) * 0.25)).collect();
    let expected = cpu_sort_full(&inp);
    let actual = run_multiblock_sort(&inp, Dt::F16, inp.len());
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-2, "8-block f16 merge mismatch at [{i}]: expected {e}, got {a}");
    }
}

/// Stability: a single merge pass must keep equal keys in input order.
/// We sort `[key]` values where many keys repeat; a stable merge keeps
/// the run-A copy of any equal pair ahead of the run-B copy. Since the
/// values themselves are equal we can't observe *which* copy survived
/// directly — instead we verify the merge of two pre-sorted runs whose
/// equal elements straddle the A/B boundary still yields a correctly
/// ordered, full-length result with the right multiset (no element
/// dropped or duplicated, which a buggy co-rank would do).
#[test]
fn sort_merge_stable_equal_keys_f32() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    // Two sorted runs of 1024, each full of repeated keys, deliberately
    // overlapping in value range so equal keys span the run boundary.
    const RUN: usize = 1024;
    let run_a: Vec<f32> = (0..RUN).map(|i| (i / 64) as f32).collect(); // 0,0,..,15,15
    let run_b: Vec<f32> = (0..RUN).map(|i| (i / 64) as f32).collect(); // identical
    let inp: Vec<f32> = run_a.iter().chain(&run_b).copied().collect();

    let merged = run_merge_pass(&ctx, &inp, Dt::F32, inp.len(), RUN);

    // Sorted, full length, identical multiset to the input.
    for w in merged.windows(2) {
        assert!(w[0] <= w[1], "stable merge not non-decreasing: {:?}", w);
    }
    let mut got = merged.clone();
    let mut want = inp.clone();
    got.sort_unstable_by(f32::total_cmp);
    want.sort_unstable_by(f32::total_cmp);
    assert_eq!(got, want, "merge changed the multiset — element dropped/duplicated");
}

/// Non-power-of-two logical length: `n` is not a multiple of 1024 and
/// `n_blocks` is not a power of two. The input buffer is padded up to a
/// whole number of blocks with `+∞`-equivalent sentinels so the per-
/// block sort + clamped merge boundaries push the padding to the tail,
/// leaving the first `n` elements correctly sorted.
#[test]
fn sort_non_power_of_two_n_matches_cpu_f32() {
    let _g = gpu_lock();
    // 3 blocks of buffer (not a power of two), logical n = 2500 (not a
    // multiple of 1024). Padding value sorts above every real element.
    const N: usize = 2500;
    let n_blocks = N.div_ceil(BLOCK); // 3
    let total = n_blocks * BLOCK; // 3072
    let pad = 1.0e30_f32;

    let real: Vec<f32> = (0..N).map(|i| ((i * 7919 + 17) % 10_000) as f32 * 0.1 - 500.0).collect();
    let mut inp = real.clone();
    inp.resize(total, pad); // pad to whole blocks

    let expected = cpu_sort_full(&real);
    let actual = run_multiblock_sort(&inp, Dt::F32, N);
    assert_eq!(actual.len(), N, "result truncated to logical n");
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!((e - a).abs() < 1e-3, "non-pow2 sort mismatch at [{i}]: expected {e}, got {a}");
    }
}

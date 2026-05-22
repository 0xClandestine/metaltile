//! GPU correctness for `mlx::sort` — single-block bitonic sort.
//!
//! `mt_sort<T>` sorts each block of `n=1024` elements in-place using
//! a bitonic sort network in shared memory.
//!
//! ## DISPATCH INVARIANTS (mt_sort)
//! - **TPG: 256 threads** (each thread processes 4 elements).
//! - **n = TPG * 4 = 1024** (elements per block — hardcoded in the kernel).
//! - **Grid: 1 threadgroup per block** (1D, program_id<0> = block index).
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
use metaltile_std::mlx::sort::{mt_merge_pass, mt_sort};

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
        .dispatch_with_grid(
            &kernel,
            &buffers,
            &BTreeMap::new(),
            [n_blocks, 1, 1],
            [256, 1, 1],
        )
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
        assert!(
            (e - a).abs() < 1e-6,
            "sort mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
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
        assert!(
            (e - a).abs() < 1e-6,
            "sort random mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
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
    let inp: Vec<f32> = (0..N)
        .map(|i| Dt::F16.round(((N - 1 - i) as f32) * 0.1))
        .collect();
    let expected = cpu_sort_block(&inp);
    let actual = run_sort(&inp, Dt::F16, 1);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (e - a).abs() < 1e-3,
            "sort f16 mismatch at [{i}]: expected {e:.4}, got {a:.4}"
        );
    }
}

// ─── mt_merge_pass — multi-block merge sort ──────────────────────────────

/// Run one `mt_merge_pass` over `inp`: merge adjacent sorted runs of
/// `run_len` into sorted runs of `2 * run_len`. `n` is the total length.
fn run_merge_pass(inp: &[f32], dt: Dt, n: usize, run_len: usize) -> Vec<f32> {
    assert_eq!(inp.len(), n);
    assert!(n % (2 * run_len) == 0, "n must be a multiple of 2*run_len");
    let log2_run = (run_len as u32).trailing_zeros();
    assert_eq!(1usize << log2_run, run_len, "run_len must be a power of two");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
    buffers.insert("run_len".into(), (run_len as u32).to_le_bytes().to_vec());
    buffers.insert("log2_run".into(), log2_run.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_merge_pass::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per array element.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n, 1, 1], [1, 1, 1])
        .expect("merge_pass dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n);
    out
}

/// Full multi-block sort: `mt_sort` per 1024-block, then `mt_merge_pass`
/// doubling the run length until one run spans the whole array.
fn run_multiblock_sort(inp: &[f32], dt: Dt) -> Vec<f32> {
    const BLOCK: usize = 1024;
    let n = inp.len();
    assert!(n % BLOCK == 0, "n must be a multiple of 1024");
    let n_blocks = n / BLOCK;
    assert!(n_blocks.is_power_of_two(), "block count must be a power of two");

    // Stage 1 — sort each 1024-element block.
    let mut buf = run_sort(inp, dt, n_blocks);

    // Stage 2 — bottom-up merge: run_len 1024 → 2048 → … → n.
    let mut run_len = BLOCK;
    while run_len < n {
        buf = run_merge_pass(&buf, dt, n, run_len);
        run_len *= 2;
    }
    buf
}

#[test]
fn merge_pass_merges_two_sorted_runs_f32() {
    let _g = gpu_lock();
    // Two pre-sorted runs of 1024 with interleaving value ranges so the
    // merge genuinely interleaves elements from both.
    const RUN: usize = 1024;
    let mut run0: Vec<f32> = (0..RUN).map(|i| (i * 2) as f32).collect(); // evens
    let mut run1: Vec<f32> = (0..RUN).map(|i| (i * 2 + 1) as f32).collect(); // odds
    run0.sort_by(f32::total_cmp);
    run1.sort_by(f32::total_cmp);
    let inp: Vec<f32> = run0.iter().chain(run1.iter()).copied().collect();

    let mut expected = inp.clone();
    expected.sort_by(f32::total_cmp);

    let actual = run_merge_pass(&inp, Dt::F32, 2 * RUN, RUN);
    assert!(actual.iter().any(|&v| v != 0.0), "merge output all zeros — empty body?");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "merge mismatch at [{i}]: expected {e}, got {a}");
    }
}

#[test]
fn merge_pass_handles_equal_values_stably_f32() {
    let _g = gpu_lock();
    // Both runs contain the value 5.0 — a stable merge keeps the left
    // run's copies first. Output correctness only needs non-decreasing
    // order, but exercising ties catches an off-by-one in the rank
    // search (strict-less vs less-equal).
    const RUN: usize = 1024;
    let run0: Vec<f32> = (0..RUN).map(|i| if i < RUN / 2 { 5.0 } else { 9.0 }).collect();
    let run1: Vec<f32> = (0..RUN).map(|i| if i < RUN / 2 { 1.0 } else { 5.0 }).collect();
    let inp: Vec<f32> = run0.iter().chain(run1.iter()).copied().collect();

    let actual = run_merge_pass(&inp, Dt::F32, 2 * RUN, RUN);
    for window in actual.windows(2) {
        assert!(window[0] <= window[1], "merge not non-decreasing: {:?}", window);
    }
    // Multiset must be preserved exactly.
    let mut got = actual.clone();
    let mut want = inp.clone();
    got.sort_by(f32::total_cmp);
    want.sort_by(f32::total_cmp);
    assert_eq!(got, want, "merge changed the multiset of elements");
}

#[test]
fn multiblock_sort_4_blocks_matches_cpu_f32() {
    let _g = gpu_lock();
    // 4096 elements = 4 blocks → 2 merge passes (1024→2048→4096).
    const N: usize = 4096;
    let inp: Vec<f32> = (0..N).map(|i| ((i * 1103515245 + 12345) % 100000) as f32 * 0.01).collect();
    let mut expected = inp.clone();
    expected.sort_by(f32::total_cmp);

    let actual = run_multiblock_sort(&inp, Dt::F32);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "multiblock sort mismatch at [{i}]: {e} vs {a}");
    }
}

#[test]
fn multiblock_sort_8_blocks_reverse_input_f32() {
    let _g = gpu_lock();
    // 8192 elements, fully reverse-sorted — 3 merge passes. Worst case.
    const N: usize = 8192;
    let inp: Vec<f32> = (0..N).rev().map(|i| i as f32 * 0.5).collect();
    let mut expected = inp.clone();
    expected.sort_by(f32::total_cmp);

    let actual = run_multiblock_sort(&inp, Dt::F32);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-6, "multiblock reverse sort mismatch at [{i}]: {e} vs {a}");
    }
}

#[test]
fn multiblock_sort_2_blocks_f16() {
    let _g = gpu_lock();
    const N: usize = 2048;
    let inp: Vec<f32> = (0..N).map(|i| Dt::F16.round(((N - 1 - i) as f32) * 0.05)).collect();
    let mut expected = inp.clone();
    expected.sort_by(f32::total_cmp);

    let actual = run_multiblock_sort(&inp, Dt::F16);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-3, "multiblock f16 sort mismatch at [{i}]: {e} vs {a}");
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

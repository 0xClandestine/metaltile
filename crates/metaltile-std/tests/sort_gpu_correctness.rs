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
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::sort::mt_sort;

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

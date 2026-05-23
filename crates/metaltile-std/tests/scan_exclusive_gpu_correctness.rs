//! GPU correctness for `mlx::scan::mt_scan_exclusive` — the exclusive
//! prefix sum along the last axis of a `[rows, n]` input.
//!
//! Reference is the trivial sequential exclusive scan: `out[0] = 0`,
//! `out[i] = out[i-1] + inp[i-1]`. The kernel uses a two-level
//! (per-simdgroup / cross-simdgroup) prefix-sum, so this independent
//! sequential oracle is a genuine cross-check of the parallel form.
//!
//! Both `n` divisible by the per-iteration chunk and a ragged `n` are
//! exercised so the trailing `base < n` bounds guards are covered.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::scan::mt_scan_exclusive;

const TPG: usize = 256;

/// Sequential exclusive scan per row — the algorithm-independent oracle.
fn cpu_exclusive_scan(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
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

fn run(inp: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("inp".into(), pack_bytes(inp, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0; rows * n], Dt::F32));
    b.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_scan_exclusive::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    // Reduction mode: one threadgroup per row (program_id<1> = row).
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [1, rows, 1], [TPG, 1, 1])
        .expect("mt_scan_exclusive dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(rows * n);
    out
}

#[test]
fn exclusive_scan_matches_cpu_aligned_n() {
    let _g = gpu_lock();
    // n = 2048 is a whole number of per-iteration chunks (chunk = 256*4).
    let (rows, n) = (3usize, 2048usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();
    let expected = cpu_exclusive_scan(&inp, rows, n);
    let out = run(&inp, rows, n);

    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 2e-3, "elem {i}: got {a}, want {e}");
    }
    // out[0] of every row must be exactly the identity element.
    for r in 0..rows {
        assert_eq!(out[r * n], 0.0, "row {r} exclusive scan must start at 0");
    }
}

#[test]
fn exclusive_scan_matches_cpu_ragged_n() {
    let _g = gpu_lock();
    // n = 3000 is NOT a multiple of the 1024-element chunk — exercises the
    // trailing `base < n` bounds guards in the kernel.
    let (rows, n) = (2usize, 3000usize);
    let inp: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let expected = cpu_exclusive_scan(&inp, rows, n);
    let out = run(&inp, rows, n);

    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 3e-3, "elem {i}: got {a}, want {e}");
    }
}

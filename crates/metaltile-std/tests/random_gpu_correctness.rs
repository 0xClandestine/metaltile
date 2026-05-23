//! GPU correctness for `mlx::random` — XOR-shift hash to u32.
//!
//! Verifies `mt_random_hash`: each output element is the XOR-shift
//! hash of its 1-based thread index. The CPU oracle replicates the
//! exact same hash — output is deterministic and bit-exact.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{gpu_lock, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::random::mt_random_hash;

/// CPU replica of the XOR-shift hash used by the kernel:
/// `s = gid + 1; s ^= s << 13; s ^= s >> 17; s ^= s << 5`
fn cpu_random_hash(n: usize) -> Vec<u32> {
    (0..n)
        .map(|gid| {
            let mut s = gid as u32 + 1;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

fn run_random_hash(n: usize) -> Vec<u32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("out".into(), vec![0u8; n * 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    // The kernel is Grid3D (one thread per output element).
    // `mt_random_hash` is a non-generic `#[kernel]` — `kernel_ir_for`
    // takes no dtype argument.
    let mut kernel = mt_random_hash::kernel_ir_for();
    kernel.mode = KernelMode::Grid3D;
    let tpg = 1024usize;
    let groups = n.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("random_hash dispatch");
    let out_bytes = result.outputs.get("out").expect("out");
    unpack_u32_bytes(out_bytes).into_iter().take(n).collect()
}

#[test]
fn random_hash_matches_cpu_oracle() {
    let _g = gpu_lock();
    let n = 1024usize;
    let expected = cpu_random_hash(n);
    let actual = run_random_hash(n);
    assert_eq!(actual.len(), n, "output length mismatch");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(*a, *e, "random_hash mismatch at [{i}]: expected {e:#010x}, got {a:#010x}");
    }
}

#[test]
fn random_hash_output_not_all_zeros() {
    let _g = gpu_lock();
    // An empty kernel body would produce all-zeros; this catches it.
    let actual = run_random_hash(256);
    assert!(actual.iter().any(|&v| v != 0), "random_hash output is all zeros");
}

#[test]
fn random_hash_large_n() {
    let _g = gpu_lock();
    // 1M elements — tests multi-threadgroup dispatch.
    let n = 1 << 20;
    let expected = cpu_random_hash(n);
    let actual = run_random_hash(n);
    assert_eq!(actual.len(), n);
    // Only spot-check a few elements to keep the test fast.
    for i in [0, 1, 1023, n / 2, n - 1] {
        assert_eq!(actual[i], expected[i], "random_hash large_n mismatch at [{i}]");
    }
}

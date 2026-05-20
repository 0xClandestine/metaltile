//! GPU correctness for `ffai::gather` — bare-tensor embedding-table gather.
//!
//! Layout:
//!   table:   [vocab, dim]
//!   indices: [n_tokens]
//!   out:     [n_tokens, dim]
//!
//! For each output element `(token, d)`, the kernel copies
//! `table[indices[token], d]` to `out[token, d]`. One thread per output
//! element via Grid3D. The quantized variant lives in `dequant_gather`
//! and already has coverage; this file closes the bare-tensor gap.
//!
//! Regression class guarded: index-formula regressions silently smear
//! the embedding table — wrong `token` decomposition → cross-token bleed,
//! wrong `d` decomposition → cross-dim bleed. Either would only surface
//! as garbage decode in FFAI integration.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gather::gather;

fn run_gather(table: &[f32], indices: &[u32], dt: Dt, n_tokens: usize, dim: usize) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("table".into(), pack_bytes(table, dt));
    buffers.insert("indices".into(), pack_u32_bytes(indices));
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; n_tokens * dim], dt));
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = gather::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: program_id::<0>() = thread index. Total threads = n_tokens * dim.
    let total = n_tokens * dim;
    let tpg = 256usize;
    let groups = total.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("gather dispatch");

    let out_bytes = result.outputs.get("out").expect("out");
    let mut out = unpack_bytes(out_bytes, dt);
    out.truncate(n_tokens * dim);
    out
}

#[test]
fn gather_copies_correct_rows_f32() {
    let _g = gpu_lock();
    // Small table where every row has a recognizable value pattern:
    // table[r, d] = r * 1000 + d. A wrong token decomposition would
    // cross-contaminate immediately.
    let vocab = 17usize;
    let dim = 8usize;
    let n_tokens = 6usize;

    let table: Vec<f32> = (0..vocab * dim).map(|i| ((i / dim) * 1000 + (i % dim)) as f32).collect();
    let indices: Vec<u32> = vec![3, 0, 11, 7, 11, 16];

    let actual = run_gather(&table, &indices, Dt::F32, n_tokens, dim);

    for (token_i, &id) in indices.iter().enumerate() {
        for d in 0..dim {
            let expected = (id as usize * 1000 + d) as f32;
            let got = actual[token_i * dim + d];
            assert!(
                (got - expected).abs() < 1e-6,
                "token {token_i} (id={id}) d={d}: expected {expected}, got {got}",
            );
        }
    }
}

#[test]
fn gather_qwen_realistic_shape_f16() {
    let _g = gpu_lock();
    // Qwen-class embedding dim. f16 — tests the load-cast/store-cast
    // path; bit-exact since the kernel is pure copy.
    let vocab = 1024usize;
    let dim = 5120usize;
    let n_tokens = 4usize;

    // Smooth value pattern to keep the f16 round-trip lossless within
    // f16's range; index decomposition still pinned by the row offsets.
    let table: Vec<f32> =
        (0..vocab * dim).map(|i| Dt::F16.round((i % 257) as f32 * 0.01 - 1.0)).collect();
    let indices: Vec<u32> = vec![511, 0, 1023, 137];

    let actual = run_gather(&table, &indices, Dt::F16, n_tokens, dim);

    for (token_i, &id) in indices.iter().enumerate() {
        for d in 0..dim {
            let expected = table[id as usize * dim + d];
            let got = actual[token_i * dim + d];
            assert!(
                (got - expected).abs() < 1e-5,
                "token {token_i} (id={id}) d={d}: expected {expected}, got {got}",
            );
        }
    }
}

#[test]
fn gather_qwen_realistic_shape_bf16() {
    let _g = gpu_lock();
    let vocab = 512usize;
    let dim = 4096usize;
    let n_tokens = 3usize;

    let table: Vec<f32> =
        (0..vocab * dim).map(|i| Dt::Bf16.round((i % 257) as f32 * 0.01 - 1.0)).collect();
    let indices: Vec<u32> = vec![137, 0, 511];

    let actual = run_gather(&table, &indices, Dt::Bf16, n_tokens, dim);

    for (token_i, &id) in indices.iter().enumerate() {
        for d in 0..dim {
            let expected = table[id as usize * dim + d];
            let got = actual[token_i * dim + d];
            assert!(
                (got - expected).abs() < 1e-3,
                "token {token_i} (id={id}) d={d}: expected {expected}, got {got}",
            );
        }
    }
}

#[test]
fn gather_repeated_indices_share_row_f32() {
    let _g = gpu_lock();
    // Repeated indices must produce identical output rows. Catches a
    // regression where token decomposition would race-corrupt or
    // accidentally use the iteration index instead of indices[token].
    let vocab = 8usize;
    let dim = 16usize;
    let n_tokens = 5usize;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let indices: Vec<u32> = vec![3, 3, 3, 7, 7];

    let actual = run_gather(&table, &indices, Dt::F32, n_tokens, dim);

    // tokens 0, 1, 2 share row 3
    for d in 0..dim {
        let expected = table[3 * dim + d];
        assert!((actual[d] - expected).abs() < 1e-6);
        assert!((actual[dim + d] - expected).abs() < 1e-6);
        assert!((actual[2 * dim + d] - expected).abs() < 1e-6);
    }
    // tokens 3, 4 share row 7
    for d in 0..dim {
        let expected = table[7 * dim + d];
        assert!((actual[3 * dim + d] - expected).abs() < 1e-6);
        assert!((actual[4 * dim + d] - expected).abs() < 1e-6);
    }
}

#[test]
fn gather_boundary_indices_f32() {
    let _g = gpu_lock();
    // Index 0 and (vocab-1) must work — exercises the lowest and
    // highest valid offsets.
    let vocab = 64usize;
    let dim = 32usize;
    let n_tokens = 2usize;

    let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
    let indices: Vec<u32> = vec![0, (vocab - 1) as u32];

    let actual = run_gather(&table, &indices, Dt::F32, n_tokens, dim);

    for (d, value) in actual.iter().take(dim).enumerate() {
        assert!((value - d as f32).abs() < 1e-6, "id=0 d={d}");
    }
    for d in 0..dim {
        let expected = ((vocab - 1) * dim + d) as f32;
        assert!((actual[dim + d] - expected).abs() < 1e-6, "id=vocab-1 d={d}");
    }
}

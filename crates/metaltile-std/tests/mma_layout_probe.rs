//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Empirical probe of Apple Metal `simdgroup_matrix<f32, 8, 8>` lane layout.
//!
//! Dispatches MMA(identity, label) → C. Prints the 8×8 C matrix.
//! If C[r, c] == r*8+c, the probe's B-load convention matches Apple's
//! actual MMA semantics. Pattern of permutation reveals the actual layout.
//!
//! Run: `cargo test --release -p metaltile-std --test mma_layout_probe -- --nocapture`

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

use metaltile::core::ir::Kernel;
use metaltile_runtime::Context;
use metaltile_std::probe::mma_layout_probe::{
    mt_mma_probe_a_identity_b_gemm,
    mt_mma_probe_a_identity_b_identity,
};

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn run_probe(name: &str, kernel: Kernel) {
    let ctx = Context::new().expect("Context::new");

    let mut buffers = BTreeMap::new();
    buffers.insert("out".into(), vec![0u8; 64 * 4]);

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [32, 1, 1])
        .expect("dispatch");

    let out_bytes = result.outputs.get("out").expect("out buffer");
    let out = bytes_to_f32(out_bytes);

    println!();
    println!("──── {} ────", name);
    println!("(expected: row r col c → value r*8+c)");
    for r in 0..8 {
        print!("  row {}: ", r);
        for c in 0..8 {
            print!("{:6.1} ", out[r * 8 + c]);
        }
        println!();
    }
}

#[test]
fn probe_a_identity_b_gemm_convention() {
    run_probe(
        "A=identity, B=gemm-transposed (lane(fm,fn) holds B[fn,fm])",
        mt_mma_probe_a_identity_b_gemm::kernel_ir_for(),
    );
}

#[test]
fn probe_a_identity_b_identity_convention() {
    run_probe(
        "A=identity, B=identity (lane(fm,fn) holds B[fm,fn])",
        mt_mma_probe_a_identity_b_identity::kernel_ir_for(),
    );
}

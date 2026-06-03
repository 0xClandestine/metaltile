//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end check that the runtime dispatch verifier rejects a GPU-pinning
//! threadgroup geometry *before* it reaches the (non-preemptive) GPU.
//!
//! `mt_rms_norm` is a Reduction-mode kernel that derives `n_simd = lsize / 32`
//! and folds the row with a slow-path reduce. Dispatched with fewer than 32
//! threads per threadgroup, `n_simd == 0` and the reduction loop would spin
//! forever, hanging the device. The verifier must turn that into a clean
//! `Err` instead. (This is safe to run: the rejection happens before any GPU
//! command is encoded, and the codegen `if (n_simd == 0u) return;` escape hatch
//! backstops it even if the verifier were bypassed.)
//!
//! macOS-gated, shares the global `gpu_lock`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use metaltile::core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::rms_norm::mt_rms_norm;

/// Buffers for an `n`-wide single-row `mt_rms_norm` dispatch (constexpr `n`
/// is passed as a 4-byte buffer, matching the harness's dispatch convention).
fn rms_norm_buffers(n: usize) -> BTreeMap<String, Vec<u8>> {
    let mut b = BTreeMap::new();
    b.insert("x".into(), vec![0u8; n * 4]);
    b.insert("w".into(), vec![0u8; n * 4]);
    b.insert("out".into(), vec![0u8; n * 4]);
    b.insert("eps_buf".into(), 1.0e-5f32.to_le_bytes().to_vec());
    b.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    b
}

fn rms_norm_kernel() -> metaltile::core::ir::Kernel {
    let mut k = mt_rms_norm::kernel_ir_for(DType::F32);
    k.mode = KernelMode::Reduction;
    k
}

#[test]
fn rejects_sub_simdgroup_dispatch_of_n_simd_kernel() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let k = rms_norm_kernel();
    // n = TPG * 4, so a valid dispatch uses TPG = 128. Here we deliberately
    // pass 16 threads/threadgroup → n_simd = 16 / 32 = 0 → pinning geometry.
    let buffers = rms_norm_buffers(512);
    let err = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [16, 1, 1])
        .expect_err("a sub-simdgroup dispatch of an n_simd kernel must be rejected, not run");
    let msg = format!("{err}");
    assert!(
        msg.contains("n_simd") || msg.contains("multiple of 32"),
        "expected a pinning-geometry rejection, got: {msg}"
    );
}

#[test]
fn accepts_valid_full_simdgroup_dispatch() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let k = rms_norm_kernel();
    // n = 512 → TPG = 128 (a positive multiple of 32). The verifier must let
    // this through and the kernel must dispatch successfully.
    let buffers = rms_norm_buffers(512);
    ctx.dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [128, 1, 1])
        .expect("a full-simdgroup dispatch must be accepted");
}

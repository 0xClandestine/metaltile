//! GPU correctness for `mlx::arg_reduce` — the generic-`T`
//! `mt_argmax` / `mt_argmin` reductions. Both emit the winning index
//! as a `u32`, mirroring MLX `arg_reduce_general`'s `uint32_t` output
//! (the index is dtype-independent — a `T`-cast would lose large
//! indices in f16 / bf16).
//!
//! Distinct from `arg_reduce_gpu_correctness.rs`, which covers the
//! FFAI `ffai_argmax` u32-output decode-sampler variant; this file
//! exercises the MLX-wired `argmax` + `argmin` pair.
//!
//! Tests pin:
//!   - argmax / argmin against a CPU oracle on random values
//!   - ties take the smallest index (the semantic contract)
//!   - all three dtypes (f32 / f16 / bf16)
//!   - a strided cover (n far larger than TPG·N_iters granularity)
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_u32_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::mlx::arg_reduce::{mt_argmax, mt_argmin};

/// Dispatch one arg-reduce kernel; returns the winning `u32` index.
fn run(kernel_ir: fn(DType) -> metaltile_core::ir::Kernel, vals: &[f32], dt: Dt) -> u32 {
    let n = vals.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(vals, dt));
    // Output is one u32 index (4 bytes — the runtime's minimum).
    buffers.insert("out".into(), vec![0u8; 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Reduction dispatch: grid=[1,1,1] tg=[256,1,1]. TPG=256 satisfies
    // the ≥32 + multiple-of-32 contract (docs/developing.md).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [256, 1, 1])
        .expect("arg_reduce dispatch");

    unpack_u32_bytes(result.outputs.get("out").expect("out"))[0]
}

fn cpu_argmax(vals: &[f32]) -> u32 {
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (i, &v) in vals.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn cpu_argmin(vals: &[f32]) -> u32 {
    let mut best_val = f32::INFINITY;
    let mut best_idx = 0u32;
    for (i, &v) in vals.iter().enumerate() {
        if v < best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[test]
fn argmax_random_f32() {
    let _g = gpu_lock();
    let mut vals: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    vals[517] = 9.0;
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::F32), 517);
}

#[test]
fn argmin_random_f32() {
    let _g = gpu_lock();
    let mut vals: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    vals[842] = -9.0;
    assert_eq!(run(mt_argmin::kernel_ir_for, &vals, Dt::F32), 842);
}

#[test]
fn argmax_argmin_against_oracle_f32() {
    let _g = gpu_lock();
    let vals: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.0071).cos() * 3.0).collect();
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::F32), cpu_argmax(&vals));
    assert_eq!(run(mt_argmin::kernel_ir_for, &vals, Dt::F32), cpu_argmin(&vals));
}

#[test]
fn argmax_ties_take_smallest_index_f32() {
    let _g = gpu_lock();
    // Positions 4..7 tie at the max; argmax must return idx 4.
    let vals: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, 5.0, 5.0, 5.0, 5.0];
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::F32), 4);
}

#[test]
fn argmin_ties_take_smallest_index_f32() {
    let _g = gpu_lock();
    // Positions 0..3 tie at the min; argmin must return idx 0.
    let vals: Vec<f32> = vec![-5.0, -5.0, -5.0, -5.0, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(run(mt_argmin::kernel_ir_for, &vals, Dt::F32), 0);
}

#[test]
fn argmax_strided_cover_f32() {
    let _g = gpu_lock();
    // n far larger than TPG so each lane walks several strided chunks.
    let n = 65_536usize;
    let mut vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0009).sin() * 0.4).collect();
    let peak = 40_021usize;
    vals[peak] = 12.0;
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::F32), peak as u32);
}

#[test]
fn argmax_random_f16() {
    let _g = gpu_lock();
    let mut vals_f32: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    vals_f32[517] = 9.0;
    let vals: Vec<f32> = vals_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::F16), 517);
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::F16), cpu_argmax(&vals));
}

#[test]
fn argmin_random_f16() {
    let _g = gpu_lock();
    let mut vals_f32: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    vals_f32[842] = -9.0;
    let vals: Vec<f32> = vals_f32.iter().map(|&v| Dt::F16.round(v)).collect();
    assert_eq!(run(mt_argmin::kernel_ir_for, &vals, Dt::F16), 842);
}

#[test]
fn argmax_random_bf16() {
    let _g = gpu_lock();
    let mut vals_f32: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    vals_f32[517] = 9.0;
    let vals: Vec<f32> = vals_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::Bf16), 517);
    assert_eq!(run(mt_argmax::kernel_ir_for, &vals, Dt::Bf16), cpu_argmax(&vals));
}

#[test]
fn argmin_random_bf16() {
    let _g = gpu_lock();
    let mut vals_f32: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    vals_f32[842] = -9.0;
    let vals: Vec<f32> = vals_f32.iter().map(|&v| Dt::Bf16.round(v)).collect();
    assert_eq!(run(mt_argmin::kernel_ir_for, &vals, Dt::Bf16), 842);
}

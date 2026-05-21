//! GPU correctness for `mlx::rope::mt_rope<T>`.
//!
//! `mt_rope` is a **strided, 3-D grid** kernel — it processes a
//! `[batch, seq_len, n_heads, head_dim]` tensor by dispatching a 3-D grid
//! where:
//!
//! - `program_id::<0>()` = frequency-index dimension  (`px` ∈ `0..gx`)
//! - `program_id::<1>()` = sequence position          (`py` ∈ `0..seq_len`)
//! - `program_id::<2>()` = head-group index            (`pz` ∈ `0..gz`)
//!
//! with `gx = head_dim / (2 * n_per_group)`, `gz = n_heads / n_per_group`.
//!
//! Each thread applies the rotary-position embedding to **4 heads** (the inner
//! `for i in range(0, 4, 1)` loop) in the freq-pair `(px, px + gx)`.
//!
//! Constexprs:
//! - `h_stride: u32` — stride between heads (`head_dim`)
//! - `seq_stride: u32` — stride between sequence positions (`n_heads * head_dim`)
//! - `grid_x: u32` — `gx` (used for `idx2 = idx1 + gx`)
//! - `base: f32` — `log2(theta_base)` (e.g. `log2(10000) ≈ 13.288`)
//!
//! Formula per (head, px, py):
//! ```
//! d_norm    = px / gx
//! inv_freq  = exp2(-(d_norm * base))   // = theta^(-2px/head_dim)
//! theta     = py * inv_freq
//! rx1 = x1*cos(theta) - x2*sin(theta)
//! rx2 = x1*sin(theta) + x2*cos(theta)
//! ```
//!
//! Only f16 is ported (the kernel is typed `<T>` but the bench and MLX
//! reference only use fp16; the test covers f32 as well for numerical
//! verification since f32 avoids half-precision round-trip noise).
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::rope::mt_rope;

/// CPU oracle — mirrors `mt_rope`'s exact arithmetic.
///
/// `tensor` layout: `[seq_len, n_heads, head_dim]` (row-major).
/// Returns the rotated tensor in f32.
fn oracle_rope_f32(
    tensor: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    n_per_group: usize,
    theta_base: f32,
) -> Vec<f32> {
    let gx = head_dim / (2 * n_per_group);
    let gz = n_heads / n_per_group;
    let h_stride = head_dim;
    let seq_stride = n_heads * head_dim;
    let base = theta_base.log2();

    let mut out = tensor.to_vec();

    // Mirror the kernel's 3-D dispatch:
    //   px ∈ [0, gx), py ∈ [0, seq_len), pz ∈ [0, gz)
    for pz in 0..gz {
        let head_base_group = pz * n_per_group; // pz * 4 in the kernel (n_per_group=4)
        for py in 0..seq_len {
            for px in 0..gx {
                let px_f = px as f32;
                let gx_f = gx as f32;
                let d_norm = px_f / gx_f;
                let inv_freq = (-d_norm * base).exp2();
                let theta = py as f32 * inv_freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Inner loop: for i in range(0, 4, 1) → i.e. n_per_group heads
                for i in 0..n_per_group {
                    let head = head_base_group + i;
                    let idx1 = py * seq_stride + head * h_stride + px;
                    let idx2 = idx1 + gx;
                    let x1 = tensor[idx1];
                    let x2 = tensor[idx2];
                    out[idx1] = x1 * cos_t - x2 * sin_t;
                    out[idx2] = x1 * sin_t + x2 * cos_t;
                }
            }
        }
    }
    out
}

/// Dispatch `mt_rope<T>` for a `[seq_len, n_heads, head_dim]` tensor.
///
/// The bench spec uses `b=1, h=32, l=512, d=128, n_per_group=4`; this test
/// uses smaller shapes to keep memory and execution time low.
///
/// Constexprs are passed as LE-encoded bytes in the buffer map.
#[allow(clippy::too_many_arguments)]
fn run_rope(
    tensor: &[f32],
    dt: Dt,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    n_per_group: usize,
    theta_base: f32,
) -> Vec<f32> {
    let gx = head_dim / (2 * n_per_group);
    let gz = n_heads / n_per_group;
    let h_stride = head_dim as u32;
    let seq_stride = (n_heads * head_dim) as u32;
    let grid_x = gx as u32;
    let base = theta_base.log2(); // kernel's constexpr `base` = log2(theta)

    let n = tensor.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(tensor, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
    // Constexprs (all u32 except base which is f32):
    buffers.insert("h_stride".into(), h_stride.to_le_bytes().to_vec());
    buffers.insert("seq_stride".into(), seq_stride.to_le_bytes().to_vec());
    buffers.insert("grid_x".into(), grid_x.to_le_bytes().to_vec());
    // f32 constexpr — pass the raw bits as a 4-byte buffer.
    buffers.insert("base".into(), base.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_rope::kernel_ir_for(dt.to_dtype());
    // The macro emits KernelMode::Grid3D for this kernel; dispatch uses 3-D
    // grid where each grid dimension is one of {gx, seq_len, gz}.
    kernel.mode = KernelMode::Grid3D;

    // 3-D Grid3D: one thread per (px, py, pz) cell.  Each cell handles
    // n_per_group heads. TPG = [1,1,1] is safe since we keep total dispatch
    // threads = gx * seq_len * gz, each independent.
    let result = ctx
        .dispatch_with_grid(
            &kernel,
            &buffers,
            &BTreeMap::new(),
            [gx, seq_len, gz],
            [1, 1, 1],
        )
        .expect("rope dispatch");

    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

#[test]
fn rope_identity_at_position_zero_f32() {
    let _g = gpu_lock();
    // At py=0, theta=0, cos=1, sin=0 → output == input for all non-zero positions.
    // seq_len=1 means py=0 is the only position.
    let seq_len = 1;
    let n_heads = 8;
    let head_dim = 64;
    let n_per_group = 4;
    let theta_base = 10000.0_f32;
    let n = seq_len * n_heads * head_dim;
    let tensor: Vec<f32> = (0..n).map(|i| (i % 31) as f32 * 0.05 - 0.75).collect();

    let actual = run_rope(&tensor, Dt::F32, seq_len, n_heads, head_dim, n_per_group, theta_base);

    // At position 0, every idx1 and idx2 output should equal the input.
    let diff = max_abs_diff(&actual, &tensor);
    assert!(diff < 1e-5, "rope identity at pos=0: max |diff| = {diff:.2e}");
}

#[test]
fn rope_matches_oracle_f32() {
    let _g = gpu_lock();
    let seq_len = 4;
    let n_heads = 8;
    let head_dim = 64;
    let n_per_group = 4;
    let theta_base = 10000.0_f32;
    let n = seq_len * n_heads * head_dim;
    let tensor: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();

    let expected =
        oracle_rope_f32(&tensor, seq_len, n_heads, head_dim, n_per_group, theta_base);
    let actual = run_rope(&tensor, Dt::F32, seq_len, n_heads, head_dim, n_per_group, theta_base);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "rope f32: max |diff| = {diff:.2e} > 1e-5");
}

#[test]
fn rope_matches_oracle_f16() {
    let _g = gpu_lock();
    let seq_len = 4;
    let n_heads = 8;
    let head_dim = 64;
    let n_per_group = 4;
    let theta_base = 10000.0_f32;
    let n = seq_len * n_heads * head_dim;
    // Round through f16 so oracle uses same load precision.
    let tensor: Vec<f32> = (0..n).map(|i| Dt::F16.round(((i % 41) as f32 - 20.0) * 0.05)).collect();

    let expected =
        oracle_rope_f32(&tensor, seq_len, n_heads, head_dim, n_per_group, theta_base);
    let actual = run_rope(&tensor, Dt::F16, seq_len, n_heads, head_dim, n_per_group, theta_base);

    let diff = max_abs_diff(&actual, &expected);
    // f16 precision: trig functions on half inputs lose ~3 ULPs.
    assert!(diff < 1e-2, "rope f16: max |diff| = {diff:.2e} > 1e-2");
}

#[test]
fn rope_preserves_norm_f32() {
    let _g = gpu_lock();
    // RoPE is an isometric rotation: ||(rx1, rx2)|| == ||(x1, x2)||.
    let seq_len = 2;
    let n_heads = 4;
    let head_dim = 32;
    let n_per_group = 4;
    let theta_base = 10000.0_f32;
    let n = seq_len * n_heads * head_dim;
    let tensor: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.07).sin()).collect();

    let actual = run_rope(&tensor, Dt::F32, seq_len, n_heads, head_dim, n_per_group, theta_base);

    let gx = head_dim / (2 * n_per_group);
    let seq_stride = n_heads * head_dim;
    let h_stride = head_dim;
    for py in 0..seq_len {
        for head in 0..n_heads {
            let base = py * seq_stride + head * h_stride;
            for px in 0..gx {
                let i1 = base + px;
                let i2 = i1 + gx;
                let in_sq = tensor[i1] * tensor[i1] + tensor[i2] * tensor[i2];
                let out_sq = actual[i1] * actual[i1] + actual[i2] * actual[i2];
                let diff = (in_sq - out_sq).abs();
                assert!(
                    diff < 1e-5,
                    "rope norm not preserved at (py={py}, head={head}, px={px}): |diff| = {diff:.2e}",
                );
            }
        }
    }
}

#[test]
fn rope_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let seq_len = 1;
    let n_heads = 8;
    let head_dim = 64;
    let n_per_group = 4;
    let theta_base = 10000.0_f32;
    let n = seq_len * n_heads * head_dim;
    let tensor: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
    let actual = run_rope(&tensor, Dt::F32, seq_len, n_heads, head_dim, n_per_group, theta_base);
    assert!(actual.iter().any(|&v| v != 0.0), "rope output is all zeros (empty kernel body?)");
}

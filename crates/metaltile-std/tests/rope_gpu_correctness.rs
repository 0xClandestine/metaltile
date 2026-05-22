//! GPU correctness for `mlx::rope::mt_rope` — rotate-half RoPE.
//!
//! Grid3D, 4 heads per z-program. For each `(px, py, pz)`:
//!
//!   d_norm   = px / grid_x
//!   inv_freq = exp2(-(d_norm * base))        // base = log2(theta_base)
//!   theta    = py * inv_freq
//!   cos_t, sin_t = cos(theta), sin(theta)
//!   for head in pz*4 .. pz*4+4:
//!     idx1 = py*seq_stride + head*h_stride + px
//!     idx2 = idx1 + grid_x
//!     out[idx1] = x1*cos_t - x2*sin_t
//!     out[idx2] = x1*sin_t + x2*cos_t
//!
//! Tensor layout is `[n_heads, seq_len, head_dim]`:
//!   h_stride   = seq_len * head_dim   (stride between heads)
//!   seq_stride = head_dim             (stride between sequence positions)
//!   grid_x     = head_dim / 2         (the rotate-half split point)
//!
//! `px` indexes the first half of `head_dim`; `idx2 = idx1 + grid_x`
//! reaches the paired element in the second half.
//!
//! Coverage rationale: `mt_rope` had no end-to-end GPU coverage. A wrong
//! index formula, a dropped cross-term in the rotation, or a silent
//! kernel-emptiness regression would only surface as model gibberish.
//!
//! Scenarios:
//!   - Identity at sequence position 0 (theta=0 → cos=1, sin=0)
//!   - Standard RoPE vs a CPU oracle (f32 / f16 / bf16)
//!   - Rotation preserves the L2 norm of every `(x1, x2)` pair
//!
//! macOS-gated. Shares the gpu_lock to serialise with sibling tests.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::rope::mt_rope;

/// CPU oracle. Matches the kernel's exact arithmetic — the `exp2` form
/// of the inverse-frequency, the fused rotate. `n_heads` must be a
/// multiple of 4 (the kernel folds 4 heads per z-program).
fn naive_rope(inp: &[f32], n_heads: u32, seq_len: u32, head_dim: u32, theta_base: f32) -> Vec<f32> {
    assert!(n_heads.is_multiple_of(4), "kernel folds 4 heads per z-program");
    assert!(head_dim.is_multiple_of(2), "rotate-half needs an even head_dim");
    let grid_x = head_dim / 2;
    let h_stride = seq_len * head_dim;
    let seq_stride = head_dim;
    let base = theta_base.log2();
    let mut out = vec![0.0f32; inp.len()];

    for pz in 0..n_heads / 4 {
        for py in 0..seq_len {
            for px in 0..grid_x {
                let d_norm = px as f32 / grid_x as f32;
                let inv_freq = (-(d_norm * base)).exp2();
                let theta = py as f32 * inv_freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                for i in 0..4 {
                    let head = pz * 4 + i;
                    let idx1 = (py * seq_stride + head * h_stride + px) as usize;
                    let idx2 = idx1 + grid_x as usize;
                    let x1 = inp[idx1];
                    let x2 = inp[idx2];
                    out[idx1] = x1 * cos_t - x2 * sin_t;
                    out[idx2] = x1 * sin_t + x2 * cos_t;
                }
            }
        }
    }
    out
}

/// Dispatch the kernel and read back the rotated tensor in `dt`.
fn run_rope(
    inp: &[f32],
    dt: Dt,
    n_heads: u32,
    seq_len: u32,
    head_dim: u32,
    theta_base: f32,
) -> Vec<f32> {
    assert!(n_heads.is_multiple_of(4) && head_dim.is_multiple_of(2));
    let grid_x = head_dim / 2;
    let h_stride = seq_len * head_dim;
    let seq_stride = head_dim;
    let base = theta_base.log2();
    let elem_count = inp.len();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; elem_count], dt));
    // Constexpr scalars are bound from `buffers` keyed by name (encoded).
    buffers.insert("h_stride".into(), h_stride.to_le_bytes().to_vec());
    buffers.insert("seq_stride".into(), seq_stride.to_le_bytes().to_vec());
    buffers.insert("grid_x".into(), grid_x.to_le_bytes().to_vec());
    buffers.insert("base".into(), base.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_rope::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Single threadgroup: program_id::<0> = px, <1> = py, <2> = pz.
    // Keep grid_x * seq_len * (n_heads/4) ≤ 1024 so one TG covers it.
    let threads = grid_x as usize * seq_len as usize * (n_heads / 4) as usize;
    assert!(threads <= 1024, "test dispatches a single TG — keep the thread count ≤ 1024");
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [
            grid_x as usize,
            seq_len as usize,
            (n_heads / 4) as usize,
        ])
        .expect("dispatch_with_grid");

    unpack_bytes(result.outputs.get("out").expect("out buffer"), dt)
}

#[test]
fn rope_identity_at_position_zero_f32() {
    let _g = gpu_lock();
    // seq_len = 1 → every token is at position 0 → theta = 0 →
    // cos = 1, sin = 0 → output equals input regardless of theta_base.
    // Pins the index formula — a wrong head/px mapping smears values.
    let n_heads = 8u32;
    let seq_len = 1u32;
    let head_dim = 16u32;
    let inp: Vec<f32> =
        (0..n_heads * seq_len * head_dim).map(|i| 0.1 + (i as f32 * 0.017).sin()).collect();

    let actual = run_rope(&inp, Dt::F32, n_heads, seq_len, head_dim, 10000.0);
    for (idx, (a, e)) in actual.iter().zip(inp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "identity at pos=0 broke at idx={idx}: got {a}, expected {e}"
        );
    }
}

#[test]
fn rope_matches_oracle_f32() {
    let _g = gpu_lock();
    // Realistic shape, multiple sequence positions and head-groups.
    // f32 → compare bit-tight against the CPU oracle (same exp2 path).
    let n_heads = 8u32;
    let seq_len = 6u32;
    let head_dim = 16u32;
    let theta_base = 10000.0f32;

    let inp: Vec<f32> =
        (0..n_heads * seq_len * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let expected = naive_rope(&inp, n_heads, seq_len, head_dim, theta_base);
    let actual = run_rope(&inp, Dt::F32, n_heads, seq_len, head_dim, theta_base);

    assert!(actual.iter().any(|&v| v != 0.0), "rope: all-zero output (empty body?)");
    let mut max_diff = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 5e-5, "rope f32: max |diff| = {max_diff:.2e} exceeds 5e-5");
}

#[test]
fn rope_matches_oracle_f16() {
    let _g = gpu_lock();
    let n_heads = 8u32;
    let seq_len = 6u32;
    let head_dim = 16u32;
    let theta_base = 10000.0f32;

    let inp: Vec<f32> =
        (0..n_heads * seq_len * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    // Round source through f16 so the oracle uses the same load
    // precision as the kernel's initial cast.
    let inp_rounded: Vec<f32> = inp.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = naive_rope(&inp_rounded, n_heads, seq_len, head_dim, theta_base);
    let actual = run_rope(&inp, Dt::F16, n_heads, seq_len, head_dim, theta_base);

    let mut max_rel = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1e-3));
    }
    assert!(max_rel < 5e-3, "rope f16: max rel = {max_rel:.2e} > 5e-3");
}

#[test]
fn rope_matches_oracle_bf16() {
    let _g = gpu_lock();
    let n_heads = 8u32;
    let seq_len = 6u32;
    let head_dim = 16u32;
    let theta_base = 10000.0f32;

    let inp: Vec<f32> =
        (0..n_heads * seq_len * head_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let inp_rounded: Vec<f32> = inp.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = naive_rope(&inp_rounded, n_heads, seq_len, head_dim, theta_base);
    let actual = run_rope(&inp, Dt::Bf16, n_heads, seq_len, head_dim, theta_base);

    let mut max_rel = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_rel = max_rel.max((a - e).abs() / e.abs().max(1e-3));
    }
    // bf16 has a 7-bit mantissa — wider tolerance than f16.
    assert!(max_rel < 2e-2, "rope bf16: max rel = {max_rel:.2e} > 2e-2");
}

#[test]
fn rope_preserves_pair_norm_f32() {
    let _g = gpu_lock();
    // A rotation preserves the L2 norm of every rotated pair
    // `(x[idx1], x[idx2])`. Pins that neither cross-term was dropped —
    // a regression like `rx2 = x1*sin + x2*sin` would not preserve norm.
    let n_heads = 8u32;
    let seq_len = 5u32;
    let head_dim = 16u32;
    let theta_base = 10000.0f32;

    let inp: Vec<f32> =
        (0..n_heads * seq_len * head_dim).map(|i| 0.5 + (i as f32 * 0.071).cos()).collect();
    let actual = run_rope(&inp, Dt::F32, n_heads, seq_len, head_dim, theta_base);

    let grid_x = (head_dim / 2) as usize;
    let h_stride = (seq_len * head_dim) as usize;
    let seq_stride = head_dim as usize;
    for head in 0..n_heads as usize {
        for s in 0..seq_len as usize {
            for px in 0..grid_x {
                let idx1 = s * seq_stride + head * h_stride + px;
                let idx2 = idx1 + grid_x;
                let in_sq = inp[idx1] * inp[idx1] + inp[idx2] * inp[idx2];
                let out_sq = actual[idx1] * actual[idx1] + actual[idx2] * actual[idx2];
                assert!(
                    (in_sq - out_sq).abs() < 1e-4,
                    "norm not preserved at (head={head}, s={s}, px={px}): \
                     in² = {in_sq:.6}, out² = {out_sq:.6}",
                );
            }
        }
    }
}

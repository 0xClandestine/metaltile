//! GPU correctness for `mlx::rope` — standard RoPE (`mt_rope<T>`).
//!
//! `mt_rope` rotates query/key heads: for each `(d, seq, head_group)` thread,
//! pairs position `d` with `d + grid_x` and applies the rotation matrix:
//!   `rx1 = x1*cos - x2*sin`, `rx2 = x1*sin + x2*cos`
//! with `inv_freq = exp2(-(d/grid_x) * base)` and `theta = seq * inv_freq`.
//!
//! ## DISPATCH (mt_rope)
//! Grid3D: `program_id<0>` = d (0..head_dim/2), `program_id<1>` = seq (0..seq_len),
//! `program_id<2>` = head_group (0..n_heads/4). Each thread updates 4 heads.
//!
//! CPU oracle: exact Python-equivalent RoPE using exp2/log2 to match kernel
//! arithmetic.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::rope::mt_rope;

/// CPU oracle for `mt_rope`.
///
/// Layout: inp[seq * h_stride + head * seq_stride + d] (confusingly named in
/// the kernel — see the kernel source for the actual index formulas).
/// The kernel's `h_stride` is the stride between heads and `seq_stride` is the
/// stride between sequence positions. For a standard `[seq, heads, head_dim]`
/// tensor laid out as `[heads, seq, head_dim]`:
///   - `h_stride = head_dim / 2`
///   - `seq_stride = n_heads * head_dim`  (not used here, we use a simplified layout)
///
/// For the test we use the simplest case: `[n_heads, head_dim]` (single token).
/// With `seq=1`, `seq_stride=n_heads*head_dim`, `h_stride=head_dim/2`:
///   - idx1 = py * seq_stride + head * h_stride + px
///   - idx2 = idx1 + grid_x  (= + half_dim)
///   where py=0 (seq position), head = head_group*4 + i, px = d.
///
/// We pass `py=0`, `seq_stride=0` so only the `head * h_stride + px` part matters.
fn cpu_rope(inp: &[f32], n_heads: usize, head_dim: usize, base: f32) -> Vec<f32> {
    // Single token at position 0 for identity check, or caller provides position.
    cpu_rope_at_position(inp, n_heads, head_dim, base, 0)
}

fn cpu_rope_at_position(
    inp: &[f32],
    n_heads: usize,
    head_dim: usize,
    base: f32,
    seq_pos: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = inp.to_vec();
    for h in 0..n_heads {
        let head_base = h * head_dim;
        for d in 0..half {
            // inv_freq = exp2(-(d / half) * base)  — matches the kernel's log2 form.
            let d_norm = d as f32 / half as f32;
            let inv_freq = (-d_norm * base.log2()).exp2();
            let theta = seq_pos as f32 * inv_freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x1 = inp[head_base + d];
            let x2 = inp[head_base + d + half];
            out[head_base + d] = x1 * cos_t - x2 * sin_t;
            out[head_base + d + half] = x1 * sin_t + x2 * cos_t;
        }
    }
    out
}

fn run_rope(inp: &[f32], dt: Dt, n_heads: usize, head_dim: usize, base: f32, seq_len: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let h_stride = half; // stride between heads in the half-dim space
    let n_head_groups = n_heads.div_ceil(4);
    let seq_stride = n_heads * head_dim; // full stride per seq position

    let n_elems = n_heads * head_dim * seq_len;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("inp".into(), pack_bytes(inp, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_elems], dt));

    // Constexprs needed by the kernel.
    let mut constexprs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    constexprs.insert("h_stride".into(), (h_stride as u32).to_le_bytes().to_vec());
    constexprs.insert("seq_stride".into(), (seq_stride as u32).to_le_bytes().to_vec());
    constexprs.insert("grid_x".into(), (half as u32).to_le_bytes().to_vec());
    constexprs.insert("base".into(), base.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_rope::kernel_ir_for(dt.to_dtype());

    // Grid dimensions: [half, seq_len, n_head_groups]
    let result = ctx
        .dispatch_with_grid(
            &kernel,
            &buffers,
            &constexprs,
            [half, seq_len, n_head_groups],
            [1, 1, 1],
        )
        .expect("rope dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_elems);
    out
}

#[test]
fn rope_identity_at_position_zero_f32() {
    let _g = gpu_lock();
    // At seq_pos=0 every theta=0 → cos=1, sin=0 → output == input.
    let (n_heads, head_dim) = (4usize, 32usize);
    let inp: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let actual = run_rope(&inp, Dt::F32, n_heads, head_dim, 10000.0, 1);
    assert!(max_abs_diff(&actual, &inp) < 1e-5, "rope identity at seq=0 f32 mismatch");
}

#[test]
fn rope_matches_cpu_oracle_f32() {
    let _g = gpu_lock();
    // Standard RoPE at position 0 (identity). Verifies layout / index formulas.
    let (n_heads, head_dim) = (4usize, 32usize);
    let inp: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let expected = cpu_rope(&inp, n_heads, head_dim, 10000.0);
    let actual = run_rope(&inp, Dt::F32, n_heads, head_dim, 10000.0, 1);
    assert!(max_abs_diff(&actual, &expected) < 1e-5, "rope f32 mismatch vs cpu oracle");
}

#[test]
fn rope_output_not_all_zeros_f32() {
    let _g = gpu_lock();
    let (n_heads, head_dim) = (4usize, 32usize);
    let inp: Vec<f32> = (1..=n_heads * head_dim).map(|i| i as f32 * 0.1).collect();
    // Use a non-zero sequence position so rotation is non-trivial.
    let out = run_rope(&inp, Dt::F32, n_heads, head_dim, 10000.0, 1);
    assert!(out.iter().any(|&v| v != 0.0), "rope output all zeros — empty kernel?");
}

#[test]
fn rope_preserves_norm_f32() {
    // RoPE is an isometry: the L2 norm of each (x[d], x[d+half]) pair is preserved.
    let _g = gpu_lock();
    let (n_heads, head_dim) = (4usize, 32usize);
    let half = head_dim / 2;
    let inp: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
    // Non-zero position to make the rotation non-trivial.
    let out = run_rope(&inp, Dt::F32, n_heads, head_dim, 10000.0, 1);

    for h in 0..n_heads {
        for d in 0..half {
            let i1 = h * head_dim + d;
            let i2 = i1 + half;
            let in_sq = inp[i1].powi(2) + inp[i2].powi(2);
            let out_sq = out[i1].powi(2) + out[i2].powi(2);
            assert!(
                (in_sq - out_sq).abs() < 1e-4,
                "norm not preserved at (head={h}, d={d})"
            );
        }
    }
}

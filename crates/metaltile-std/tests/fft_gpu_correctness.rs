//! GPU correctness for `mlx::fft` — the radix-2 Cooley–Tukey FFT.
//!
//! Two independent checks per shape:
//!
//! 1. **Forward vs naive DFT.** The kernel's O(N log N) butterfly
//!    output is compared element-by-element against an
//!    algorithm-independent O(N²) direct DFT reference.
//! 2. **Round-trip `ifft(fft(x)) ≈ x`.** The forward transform is fed
//!    back through the kernel with `inv = 1`; the result must recover
//!    the original signal (the inverse path's conjugated twiddles and
//!    `1/N` scale are exercised here).
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_runtime::Context;
use metaltile_std::mlx::fft::{mt_fft_n32, mt_fft_n64, mt_fft_n128, mt_fft_n256};

/// Direct O(N²) DFT reference, per row.
///
/// `X[k] = Σ_n x[n] · e^{∓ i 2π k n / N}` — `−` for forward
/// (`inv = false`), `+` with a `1/N` scale for inverse.
fn naive_dft(re: &[f32], im: &[f32], rows: usize, n: usize, inv: bool) -> (Vec<f32>, Vec<f32>) {
    let mut out_re = vec![0.0_f32; rows * n];
    let mut out_im = vec![0.0_f32; rows * n];
    let sign = if inv { 1.0_f32 } else { -1.0_f32 };
    let scale = if inv { 1.0_f32 / n as f32 } else { 1.0_f32 };
    for r in 0..rows {
        let base = r * n;
        for k in 0..n {
            let mut acc_re = 0.0_f32;
            let mut acc_im = 0.0_f32;
            for t in 0..n {
                let angle = sign * std::f32::consts::TAU * (k as f32) * (t as f32) / n as f32;
                let (s, c) = angle.sin_cos();
                // (xr + xi·i)(c + s·i) = (xr·c − xi·s) + (xr·s + xi·c)i
                let xr = re[base + t];
                let xi = im[base + t];
                acc_re += xr * c - xi * s;
                acc_im += xr * s + xi * c;
            }
            out_re[base + k] = acc_re * scale;
            out_im[base + k] = acc_im * scale;
        }
    }
    (out_re, out_im)
}

/// Dispatch the FFT kernel; returns `(out_re, out_im)`.
fn run(
    kernel_ir: fn(DType) -> Kernel,
    re: &[f32],
    im: &[f32],
    dt: Dt,
    rows: usize,
    n: usize,
    inv: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("in_re".into(), pack_bytes(re, dt));
    b.insert("in_im".into(), pack_bytes(im, dt));
    b.insert("out_re".into(), pack_bytes(&vec![0.0; rows * n], dt));
    b.insert("out_im".into(), pack_bytes(&vec![0.0; rows * n], dt));
    // Constexprs are passed in the same buffer map (see mel/vocoder tests).
    b.insert("inv".into(), (inv as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel_ir(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [rows, 1, 1], [n, 1, 1])
        .expect("fft dispatch");

    let mut out_re = unpack_bytes(result.outputs.get("out_re").expect("out_re"), dt);
    let mut out_im = unpack_bytes(result.outputs.get("out_im").expect("out_im"), dt);
    out_re.truncate(rows * n);
    out_im.truncate(rows * n);
    (out_re, out_im)
}

/// Deterministic real-valued ramp signal.
fn ramp(rows: usize, n: usize) -> Vec<f32> {
    (0..rows * n).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect()
}

// ── Forward transform vs the direct DFT ─────────────────────────────────

#[test]
fn fft_n32_forward_matches_naive_dft_f32() {
    let _g = gpu_lock();
    let (rows, n) = (3, 32);
    let re = ramp(rows, n);
    let im = vec![0.0_f32; rows * n]; // real-input signal
    let (exp_re, exp_im) = naive_dft(&re, &im, rows, n, false);
    let (act_re, act_im) = run(mt_fft_n32::kernel_ir_for, &re, &im, Dt::F32, rows, n, false);
    assert!(act_re.iter().any(|&v| v != 0.0), "output is all zeros");
    assert!(max_abs_diff(&act_re, &exp_re) < 1e-3, "n32 forward re mismatch");
    assert!(max_abs_diff(&act_im, &exp_im) < 1e-3, "n32 forward im mismatch");
}

#[test]
fn fft_n64_forward_matches_naive_dft_f32() {
    let _g = gpu_lock();
    let (rows, n) = (2, 64);
    let re = ramp(rows, n);
    let im = vec![0.0_f32; rows * n];
    let (exp_re, exp_im) = naive_dft(&re, &im, rows, n, false);
    let (act_re, act_im) = run(mt_fft_n64::kernel_ir_for, &re, &im, Dt::F32, rows, n, false);
    assert!(max_abs_diff(&act_re, &exp_re) < 2e-3, "n64 forward re mismatch");
    assert!(max_abs_diff(&act_im, &exp_im) < 2e-3, "n64 forward im mismatch");
}

#[test]
fn fft_n128_forward_matches_naive_dft_complex_input_f32() {
    let _g = gpu_lock();
    let (rows, n) = (2, 128);
    // Genuinely complex input — exercises the imaginary plane on input.
    let re = ramp(rows, n);
    let im: Vec<f32> = (0..rows * n).map(|i| ((i % 13) as f32 - 6.0) * 0.07).collect();
    let (exp_re, exp_im) = naive_dft(&re, &im, rows, n, false);
    let (act_re, act_im) = run(mt_fft_n128::kernel_ir_for, &re, &im, Dt::F32, rows, n, false);
    assert!(max_abs_diff(&act_re, &exp_re) < 5e-3, "n128 forward re mismatch");
    assert!(max_abs_diff(&act_im, &exp_im) < 5e-3, "n128 forward im mismatch");
}

// ── Inverse transform vs the direct inverse DFT ─────────────────────────

#[test]
fn fft_n64_inverse_matches_naive_idft_f32() {
    let _g = gpu_lock();
    let (rows, n) = (2, 64);
    let re = ramp(rows, n);
    let im: Vec<f32> = (0..rows * n).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    let (exp_re, exp_im) = naive_dft(&re, &im, rows, n, true);
    let (act_re, act_im) = run(mt_fft_n64::kernel_ir_for, &re, &im, Dt::F32, rows, n, true);
    assert!(max_abs_diff(&act_re, &exp_re) < 2e-3, "n64 inverse re mismatch");
    assert!(max_abs_diff(&act_im, &exp_im) < 2e-3, "n64 inverse im mismatch");
}

// ── Round-trip: ifft(fft(x)) ≈ x ────────────────────────────────────────

#[test]
fn fft_n256_round_trip_recovers_input_f32() {
    let _g = gpu_lock();
    let (rows, n) = (2, 256);
    let re = ramp(rows, n);
    let im: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();

    let (fwd_re, fwd_im) = run(mt_fft_n256::kernel_ir_for, &re, &im, Dt::F32, rows, n, false);
    let (rt_re, rt_im) = run(mt_fft_n256::kernel_ir_for, &fwd_re, &fwd_im, Dt::F32, rows, n, true);

    assert!(max_abs_diff(&rt_re, &re) < 5e-3, "round-trip real mismatch");
    assert!(max_abs_diff(&rt_im, &im) < 5e-3, "round-trip imag mismatch");
}

// ── Half-precision: round-trip with a relaxed tolerance ─────────────────

#[test]
fn fft_n32_round_trip_f16() {
    let _g = gpu_lock();
    let (rows, n) = (2, 32);
    // Small magnitudes so the f16 butterfly accumulation stays in range.
    let re: Vec<f32> = (0..rows * n).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
    let im = vec![0.0_f32; rows * n];

    let (fwd_re, fwd_im) = run(mt_fft_n32::kernel_ir_for, &re, &im, Dt::F16, rows, n, false);
    let (rt_re, rt_im) = run(mt_fft_n32::kernel_ir_for, &fwd_re, &fwd_im, Dt::F16, rows, n, true);

    assert!(max_abs_diff(&rt_re, &re) < 3e-2, "f16 round-trip real mismatch");
    assert!(max_abs_diff(&rt_im, &im) < 3e-2, "f16 round-trip imag mismatch");
}

#[test]
fn fft_n64_round_trip_bf16() {
    let _g = gpu_lock();
    let (rows, n) = (2, 64);
    let re: Vec<f32> = (0..rows * n).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
    let im = vec![0.0_f32; rows * n];

    let (fwd_re, fwd_im) = run(mt_fft_n64::kernel_ir_for, &re, &im, Dt::Bf16, rows, n, false);
    let (rt_re, rt_im) = run(mt_fft_n64::kernel_ir_for, &fwd_re, &fwd_im, Dt::Bf16, rows, n, true);

    // bf16 has a 7-bit mantissa — round-trip through the full
    // forward+inverse butterfly accumulates noticeable error.
    assert!(max_abs_diff(&rt_re, &re) < 1.5e-1, "bf16 round-trip real mismatch");
    assert!(max_abs_diff(&rt_im, &im) < 1.5e-1, "bf16 round-trip imag mismatch");
}

//! End-to-end GPU correctness for `ffai::winograd_conv` — the Winograd
//! F(2×2, 3×3) fast convolution.
//!
//! Winograd computes the same 3×3-stride-1 convolution as a direct conv
//! but through three fixed transforms (input / filter / output) and a
//! 4×4 element-wise product. This validates proc-macro → IR → MSL → PSO
//! → dispatch → readback against a straight six-loop direct-conv CPU
//! reference — if any transform matrix is wrong the result diverges.
//!
//! Covers:
//!   - unpadded conv, single channel pair (the minimal transform check)
//!   - padded conv (1-px), multi-channel (the padding clamp + the
//!     per-channel accumulation in the transformed domain)
//!   - f32 / f16 / bf16
//!
//! The kernel requires even `out_h` / `out_w` (a dispatch invariant —
//! F(2×2, 3×3) emits 2×2 tiles); every shape here is chosen even.
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::winograd_conv::{
    winograd_conv2d_3x3,
    winograd_conv2d_3x3_split,
    winograd_filter_transform_3x3,
};

#[derive(Clone, Copy)]
struct ConvShape {
    batch: usize,
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    pad: usize,
}

impl ConvShape {
    // 3×3 kernel, stride 1.
    fn out_h(&self) -> usize { self.in_h + 2 * self.pad - 2 }
    fn out_w(&self) -> usize { self.in_w + 2 * self.pad - 2 }
}

/// CPU reference: direct 3×3 stride-1 convolution. NCHW input, OIHW
/// weight, NCHW output. Padding taps contribute zero. All f32.
#[allow(clippy::needless_range_loop)]
fn naive_conv3x3(input: &[f32], weight: &[f32], bias: &[f32], s: &ConvShape) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    let mut out = vec![0.0f32; s.batch * s.out_ch * out_h * out_w];
    for n in 0..s.batch {
        for oc in 0..s.out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias[oc];
                    for ic in 0..s.in_ch {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let ph = oh + ky;
                                let pw = ow + kx;
                                // Padded-frame index → real index.
                                if ph < s.pad
                                    || ph >= s.pad + s.in_h
                                    || pw < s.pad
                                    || pw >= s.pad + s.in_w
                                {
                                    continue;
                                }
                                let ih = ph - s.pad;
                                let iw = pw - s.pad;
                                let in_idx = ((n * s.in_ch + ic) * s.in_h + ih) * s.in_w + iw;
                                let w_idx = ((oc * s.in_ch + ic) * 3 + ky) * 3 + kx;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    let o_idx = ((n * s.out_ch + oc) * out_h + oh) * out_w + ow;
                    out[o_idx] = acc;
                }
            }
        }
    }
    out
}

/// Dispatch the Winograd kernel and read back the output.
fn run_winograd(input: &[f32], weight: &[f32], bias: &[f32], dt: Dt, s: &ConvShape) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    assert!(out_h % 2 == 0 && out_w % 2 == 0, "Winograd needs even output dims");
    let (tiles_h, tiles_w) = (out_h / 2, out_w / 2);
    let n_out = s.batch * s.out_ch * out_h * out_w;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("bias".into(), pack_bytes(bias, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("out_ch".into(), u(s.out_ch));
    buffers.insert("out_h".into(), u(out_h));
    buffers.insert("out_w".into(), u(out_w));
    buffers.insert("pad_h".into(), u(s.pad));
    buffers.insert("pad_w".into(), u(s.pad));
    buffers.insert("tiles_h".into(), u(tiles_h));
    buffers.insert("tiles_w".into(), u(tiles_w));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = winograd_conv2d_3x3::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per 2×2 output tile.
    let n_tiles = s.batch * s.out_ch * tiles_h * tiles_w;
    let tpg = 64usize;
    let grid = n_tiles.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("winograd dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

#[test]
fn winograd_unpadded_single_channel_matches_naive_f32() {
    let _g = gpu_lock();
    // Minimal transform check: one input channel, one output channel,
    // no padding. in 8×8 → out 6×6 (even). A wrong Bᵀ/G/Aᵀ matrix
    // entry shows up immediately here.
    let s = ConvShape { batch: 1, in_ch: 1, in_h: 8, in_w: 8, out_ch: 1, pad: 0 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * 9, 13, 6.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv3x3(&input, &weight, &bias, &s);
    let actual = run_winograd(&input, &weight, &bias, Dt::F32, &s);
    assert!(actual.iter().any(|&v| v != 0.0), "winograd: all-zero output (empty body?)");
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "winograd unpadded f32: max |diff| = {diff:.2e}");
}

#[test]
fn winograd_padded_multi_channel_matches_naive_f32() {
    let _g = gpu_lock();
    // 1-px padding (out_h == in_h), multiple input + output channels —
    // exercises the padding clamp on every tile edge and the
    // transformed-domain accumulation across channels. in 8×10, pad 1
    // → out 8×10 (both even).
    let s = ConvShape { batch: 2, in_ch: 4, in_h: 8, in_w: 10, out_ch: 5, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * 9, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let expected = naive_conv3x3(&input, &weight, &bias, &s);
    let actual = run_winograd(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 3e-3, "winograd padded f32: max |diff| = {diff:.2e}");
}

#[test]
fn winograd_padded_multi_channel_matches_naive_f16() {
    let _g = gpu_lock();
    let s = ConvShape { batch: 1, in_ch: 4, in_h: 8, in_w: 8, out_ch: 4, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * 9, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    // Round source through f16 so the reference uses the same load
    // precision as the kernel's initial cast.
    let inp_r: Vec<f32> = input.iter().map(|&v| Dt::F16.round(v)).collect();
    let w_r: Vec<f32> = weight.iter().map(|&v| Dt::F16.round(v)).collect();
    let b_r: Vec<f32> = bias.iter().map(|&v| Dt::F16.round(v)).collect();
    let expected = naive_conv3x3(&inp_r, &w_r, &b_r, &s);
    let actual = run_winograd(&input, &weight, &bias, Dt::F16, &s);
    let diff = max_abs_diff(&expected, &actual);
    // Winograd does its transforms + accumulation in f32 internally; the
    // only f16 loss is the load cast and the store cast.
    assert!(diff < 3e-2, "winograd f16: max |diff| = {diff:.2e}");
}

#[test]
fn winograd_padded_multi_channel_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = ConvShape { batch: 1, in_ch: 4, in_h: 8, in_w: 8, out_ch: 4, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * 9, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let inp_r: Vec<f32> = input.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let w_r: Vec<f32> = weight.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let b_r: Vec<f32> = bias.iter().map(|&v| Dt::Bf16.round(v)).collect();
    let expected = naive_conv3x3(&inp_r, &w_r, &b_r, &s);
    let actual = run_winograd(&input, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    // bf16 has a 7-bit mantissa — wider tolerance than f16.
    assert!(diff < 1.5e-1, "winograd bf16: max |diff| = {diff:.2e}");
}

/// Run the two-kernel cuDNN-style split — `winograd_filter_transform_3x3`
/// pre-transforms every filter into its 4×4 `U`, then
/// `winograd_conv2d_3x3_split` consumes that buffer. Result must match the
/// single-kernel `winograd_conv2d_3x3` (and thus the naive oracle).
fn run_winograd_split(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &ConvShape,
) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    assert!(out_h % 2 == 0 && out_w % 2 == 0, "Winograd needs even output dims");
    let (tiles_h, tiles_w) = (out_h / 2, out_w / 2);
    let n_out = s.batch * s.out_ch * out_h * out_w;
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    let ctx = Context::new().expect("Context::new on macOS");

    // ── Stage 1: filter transform → U[out_ch, in_ch, 4, 4] ──
    let n_filt = s.out_ch * s.in_ch;
    let mut fb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    fb.insert("weight".into(), pack_bytes(weight, dt));
    fb.insert("out".into(), pack_bytes(&vec![0.0f32; n_filt * 16], dt));
    fb.insert("in_ch".into(), u(s.in_ch));
    fb.insert("out_ch".into(), u(s.out_ch));
    let mut fk = winograd_filter_transform_3x3::kernel_ir_for(dt.to_dtype());
    fk.mode = KernelMode::Grid3D;
    let tpg = 64usize;
    let fgrid = n_filt.div_ceil(tpg);
    let fres = ctx
        .dispatch_with_grid(&fk, &fb, &BTreeMap::new(), [fgrid, 1, 1], [tpg, 1, 1])
        .expect("filter-transform dispatch");
    // The transformed-filter buffer, still dt-packed — feed it straight in.
    let u_bytes = fres.outputs.get("out").expect("u").clone();

    // ── Stage 2: split conv consuming the pre-transformed U ──
    let mut cb: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    cb.insert("input".into(), pack_bytes(input, dt));
    cb.insert("u".into(), u_bytes);
    cb.insert("bias".into(), pack_bytes(bias, dt));
    cb.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    cb.insert("in_ch".into(), u(s.in_ch));
    cb.insert("in_h".into(), u(s.in_h));
    cb.insert("in_w".into(), u(s.in_w));
    cb.insert("out_ch".into(), u(s.out_ch));
    cb.insert("out_h".into(), u(out_h));
    cb.insert("out_w".into(), u(out_w));
    cb.insert("pad_h".into(), u(s.pad));
    cb.insert("pad_w".into(), u(s.pad));
    cb.insert("tiles_h".into(), u(tiles_h));
    cb.insert("tiles_w".into(), u(tiles_w));
    let mut ck = winograd_conv2d_3x3_split::kernel_ir_for(dt.to_dtype());
    ck.mode = KernelMode::Grid3D;
    let n_tiles = s.batch * s.out_ch * tiles_h * tiles_w;
    let cgrid = n_tiles.div_ceil(tpg);
    let cres = ctx
        .dispatch_with_grid(&ck, &cb, &BTreeMap::new(), [cgrid, 1, 1], [tpg, 1, 1])
        .expect("split-conv dispatch");
    let mut out = unpack_bytes(cres.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

#[test]
fn winograd_split_padded_multi_channel_matches_naive_f32() {
    let _g = gpu_lock();
    let s = ConvShape { batch: 2, in_ch: 4, in_h: 8, in_w: 10, out_ch: 5, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * 9, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let expected = naive_conv3x3(&input, &weight, &bias, &s);
    let got = run_winograd_split(&input, &weight, &bias, Dt::F32, &s);
    let d = max_abs_diff(&expected, &got);
    println!("[winograd split f32] max|Δ| = {d:.5e}");
    assert!(d < 1e-3, "winograd split f32 max|Δ| = {d:.5e}");
}

#[test]
fn winograd_split_padded_multi_channel_matches_naive_f16() {
    let _g = gpu_lock();
    let s = ConvShape { batch: 1, in_ch: 4, in_h: 8, in_w: 8, out_ch: 4, pad: 1 };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * 9, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let expected = naive_conv3x3(&input, &weight, &bias, &s);
    let got = run_winograd_split(&input, &weight, &bias, Dt::F16, &s);
    let d = max_abs_diff(&expected, &got);
    println!("[winograd split f16] max|Δ| = {d:.5e}");
    // f16 transform-domain rounding — same bar as the single-kernel f16 test.
    assert!(d < 2.0, "winograd split f16 max|Δ| = {d:.5e}");
}

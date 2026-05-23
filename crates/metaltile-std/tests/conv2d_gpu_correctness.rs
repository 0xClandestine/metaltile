//! End-to-end GPU correctness for `ffai::conv2d` — the vision
//! patch-embedding 2D convolution (im2col + GEMM done as a direct conv).
//!
//! Validates proc-macro → IR → MSL → PSO → dispatch → readback against
//! a straight six-loop CPU reference. Covers:
//!   - the fixed-patch variants (`conv2d_patch14`, `conv2d_patch16`):
//!     non-overlapping patch conv, the real VLM stem shape
//!   - the generic variant (`conv2d_generic`): a small overlapping conv
//!     with padding, exercising the runtime kh/kw/stride/pad constexprs
//!     and the in-kernel padding clamp
//!   - f32 / f16 / bf16
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::conv2d::{conv2d_generic, conv2d_grouped, conv2d_patch14, conv2d_patch16};

#[derive(Clone, Copy)]
struct ConvShape {
    batch: usize,
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
}

impl ConvShape {
    fn out_h(&self) -> usize { (self.in_h + 2 * self.pad_h - self.kh) / self.stride_h + 1 }
    fn out_w(&self) -> usize { (self.in_w + 2 * self.pad_w - self.kw) / self.stride_w + 1 }
}

/// CPU reference: NCHW input, OIHW weight, NCHW output. Padding taps
/// (input index outside the real image) contribute zero. All f32.
// `oc` indexes `bias` but is also a coordinate used to derive every
// other index, so the explicit range loop is the readable form.
#[allow(clippy::needless_range_loop)]
fn naive_conv2d(input: &[f32], weight: &[f32], bias: &[f32], s: &ConvShape) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    let mut out = vec![0.0f32; s.batch * s.out_ch * out_h * out_w];
    for n in 0..s.batch {
        for oc in 0..s.out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias[oc];
                    for ic in 0..s.in_ch {
                        for ky in 0..s.kh {
                            for kx in 0..s.kw {
                                // Padded-frame index → real index.
                                let ph = oh * s.stride_h + ky;
                                let pw = ow * s.stride_w + kx;
                                if ph < s.pad_h
                                    || ph >= s.pad_h + s.in_h
                                    || pw < s.pad_w
                                    || pw >= s.pad_w + s.in_w
                                {
                                    continue;
                                }
                                let ih = ph - s.pad_h;
                                let iw = pw - s.pad_w;
                                let in_idx = ((n * s.in_ch + ic) * s.in_h + ih) * s.in_w + iw;
                                let w_idx = ((oc * s.in_ch + ic) * s.kh + ky) * s.kw + kx;
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

/// Dispatch a conv2d variant by kernel name and read back the output.
fn run_conv2d(
    kernel_name: &str,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &ConvShape,
) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    let n_out = s.batch * s.out_ch * out_h * out_w;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("bias".into(), pack_bytes(bias, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("batch".into(), u(s.batch));
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("out_ch".into(), u(s.out_ch));
    buffers.insert("out_h".into(), u(out_h));
    buffers.insert("out_w".into(), u(out_w));
    buffers.insert("kh".into(), u(s.kh));
    buffers.insert("kw".into(), u(s.kw));
    buffers.insert("stride_h".into(), u(s.stride_h));
    buffers.insert("stride_w".into(), u(s.stride_w));
    buffers.insert("pad_h".into(), u(s.pad_h));
    buffers.insert("pad_w".into(), u(s.pad_w));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = match kernel_name {
        "conv2d_patch14" => conv2d_patch14::kernel_ir_for(dt.to_dtype()),
        "conv2d_patch16" => conv2d_patch16::kernel_ir_for(dt.to_dtype()),
        "conv2d_generic" => conv2d_generic::kernel_ir_for(dt.to_dtype()),
        other => panic!("unknown conv2d kernel {other}"),
    };
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per output element. Tile by 256 threads/TG.
    let tpg = 256usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("conv2d dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

#[test]
fn conv2d_patch14_matches_naive_f32() {
    let _g = gpu_lock();
    // SigLIP / Qwen-VL stem: 3-channel image, 14×14 patch, stride 14,
    // projecting to a small out_ch so the CPU reference stays instant.
    let s = ConvShape {
        batch: 1,
        in_ch: 3,
        in_h: 28,
        in_w: 42,
        out_ch: 8,
        kh: 14,
        kw: 14,
        stride_h: 14,
        stride_w: 14,
        pad_h: 0,
        pad_w: 0,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let expected = naive_conv2d(&input, &weight, &bias, &s);
    let actual = run_conv2d("conv2d_patch14", &input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "conv2d_patch14 f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_patch16_matches_naive_f32() {
    let _g = gpu_lock();
    // CLIP / Gemma-VL stem: 16×16 patch, stride 16.
    let s = ConvShape {
        batch: 1,
        in_ch: 3,
        in_h: 32,
        in_w: 48,
        out_ch: 6,
        kh: 16,
        kw: 16,
        stride_h: 16,
        stride_w: 16,
        pad_h: 0,
        pad_w: 0,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 29, 14.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 31, 15.0);
    let bias = ramp(s.out_ch, 5, 2.0);
    let expected = naive_conv2d(&input, &weight, &bias, &s);
    let actual = run_conv2d("conv2d_patch16", &input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "conv2d_patch16 f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_generic_with_padding_matches_naive_f32() {
    let _g = gpu_lock();
    // Overlapping 3×3 stride-1 conv with 1-px padding — exercises the
    // runtime constexprs and the in-kernel padding clamp on every edge.
    let s = ConvShape {
        batch: 2,
        in_ch: 4,
        in_h: 9,
        in_w: 11,
        out_ch: 5,
        kh: 3,
        kw: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv2d(&input, &weight, &bias, &s);
    let actual = run_conv2d("conv2d_generic", &input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv2d_generic f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_generic_strided_matches_naive_f32() {
    let _g = gpu_lock();
    // Strided 5×5 conv, no padding — a non-patch wide-stride config.
    let s = ConvShape {
        batch: 1,
        in_ch: 2,
        in_h: 20,
        in_w: 24,
        out_ch: 4,
        kh: 5,
        kw: 5,
        stride_h: 3,
        stride_w: 2,
        pad_h: 0,
        pad_w: 0,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 19, 9.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 13, 6.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv2d(&input, &weight, &bias, &s);
    let actual = run_conv2d("conv2d_generic", &input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv2d_generic strided f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_patch14_matches_naive_f16() {
    let _g = gpu_lock();
    let s = ConvShape {
        batch: 1,
        in_ch: 3,
        in_h: 28,
        in_w: 28,
        out_ch: 4,
        kh: 14,
        kw: 14,
        stride_h: 14,
        stride_w: 14,
        pad_h: 0,
        pad_w: 0,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_conv2d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv2d("conv2d_patch14", &input, &weight, &bias, Dt::F16, &s);
    let diff = max_abs_diff(&expected, &actual);
    // 588-term reduction (3·14·14) in f16 — wider tolerance.
    assert!(diff < 2e-1, "conv2d_patch14 f16: max |diff| = {diff:.2e}");
}

// ── Grouped / dilated conv: conv2d_grouped ───────────────────────────

#[derive(Clone, Copy)]
struct GroupedShape {
    batch: usize,
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
}

impl GroupedShape {
    fn eff_kh(&self) -> usize { (self.kh - 1) * self.dilation_h + 1 }
    fn eff_kw(&self) -> usize { (self.kw - 1) * self.dilation_w + 1 }
    fn out_h(&self) -> usize { (self.in_h + 2 * self.pad_h - self.eff_kh()) / self.stride_h + 1 }
    fn out_w(&self) -> usize { (self.in_w + 2 * self.pad_w - self.eff_kw()) / self.stride_w + 1 }
    fn icpg(&self) -> usize { self.in_ch / self.groups }
    fn ocpg(&self) -> usize { self.out_ch / self.groups }
}

/// CPU reference for grouped + dilated conv. NCHW input, OIHW weight
/// where the I dimension is `in_ch / groups`. Padding taps contribute
/// zero; dilation scales the tap offsets. All f32.
#[allow(clippy::needless_range_loop)]
fn naive_conv2d_grouped(input: &[f32], weight: &[f32], bias: &[f32], s: &GroupedShape) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
    let (icpg, ocpg) = (s.icpg(), s.ocpg());
    let mut out = vec![0.0f32; s.batch * s.out_ch * out_h * out_w];
    for n in 0..s.batch {
        for oc in 0..s.out_ch {
            let group = oc / ocpg;
            let ic_base = group * icpg;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias[oc];
                    for wic in 0..icpg {
                        let real_ic = ic_base + wic;
                        for ky in 0..s.kh {
                            for kx in 0..s.kw {
                                let ph = oh * s.stride_h + ky * s.dilation_h;
                                let pw = ow * s.stride_w + kx * s.dilation_w;
                                if ph < s.pad_h
                                    || ph >= s.pad_h + s.in_h
                                    || pw < s.pad_w
                                    || pw >= s.pad_w + s.in_w
                                {
                                    continue;
                                }
                                let ih = ph - s.pad_h;
                                let iw = pw - s.pad_w;
                                let in_idx = ((n * s.in_ch + real_ic) * s.in_h + ih) * s.in_w + iw;
                                let w_idx = ((oc * icpg + wic) * s.kh + ky) * s.kw + kx;
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

fn run_conv2d_grouped(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &GroupedShape,
) -> Vec<f32> {
    let (out_h, out_w) = (s.out_h(), s.out_w());
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
    buffers.insert("kh".into(), u(s.kh));
    buffers.insert("kw".into(), u(s.kw));
    buffers.insert("stride_h".into(), u(s.stride_h));
    buffers.insert("stride_w".into(), u(s.stride_w));
    buffers.insert("pad_h".into(), u(s.pad_h));
    buffers.insert("pad_w".into(), u(s.pad_w));
    buffers.insert("dilation_h".into(), u(s.dilation_h));
    buffers.insert("dilation_w".into(), u(s.dilation_w));
    buffers.insert("icpg".into(), u(s.icpg()));
    buffers.insert("ocpg".into(), u(s.ocpg()));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = conv2d_grouped::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("conv2d_grouped dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

#[test]
fn conv2d_grouped_dilated_matches_naive_f32() {
    let _g = gpu_lock();
    // Dilation-2 3×3 conv, groups=1 (pure atrous test).
    let s = GroupedShape {
        batch: 1,
        in_ch: 3,
        in_h: 16,
        in_w: 18,
        out_ch: 6,
        kh: 3,
        kw: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 2,
        pad_w: 2,
        dilation_h: 2,
        dilation_w: 2,
        groups: 1,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 31, 15.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kh * s.kw, 23, 11.0);
    let bias = ramp(s.out_ch, 5, 2.0);
    let expected = naive_conv2d_grouped(&input, &weight, &bias, &s);
    let actual = run_conv2d_grouped(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv2d_grouped dilated f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_grouped_depthwise_matches_naive_f32() {
    let _g = gpu_lock();
    // Depthwise conv: groups == in_ch == out_ch, one channel per group.
    let s = GroupedShape {
        batch: 2,
        in_ch: 8,
        in_h: 12,
        in_w: 14,
        out_ch: 8,
        kh: 3,
        kw: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 8,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 19, 9.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kh * s.kw, 13, 6.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv2d_grouped(&input, &weight, &bias, &s);
    let actual = run_conv2d_grouped(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv2d_grouped depthwise f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_grouped_groups_and_dilation_matches_naive_f32() {
    let _g = gpu_lock();
    // groups=2 + dilation=2 + stride=2 — every degree of freedom at once.
    let s = GroupedShape {
        batch: 1,
        in_ch: 6,
        in_h: 20,
        in_w: 22,
        out_ch: 8,
        kh: 3,
        kw: 3,
        stride_h: 2,
        stride_w: 2,
        pad_h: 2,
        pad_w: 2,
        dilation_h: 2,
        dilation_w: 2,
        groups: 2,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 29, 14.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv2d_grouped(&input, &weight, &bias, &s);
    let actual = run_conv2d_grouped(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv2d_grouped groups+dilation f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_grouped_depthwise_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = GroupedShape {
        batch: 1,
        in_ch: 4,
        in_h: 10,
        in_w: 10,
        out_ch: 4,
        kh: 3,
        kw: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 4,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_conv2d_grouped(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv2d_grouped(&input, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-1, "conv2d_grouped depthwise bf16: max |diff| = {diff:.2e}");
}

#[test]
fn conv2d_generic_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = ConvShape {
        batch: 1,
        in_ch: 4,
        in_h: 9,
        in_w: 9,
        out_ch: 3,
        kh: 3,
        kw: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
    };
    let input = ramp(s.batch * s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_conv2d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv2d("conv2d_generic", &input, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-1, "conv2d_generic bf16: max |diff| = {diff:.2e}");
}

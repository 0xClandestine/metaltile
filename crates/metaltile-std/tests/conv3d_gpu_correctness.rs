//! End-to-end GPU correctness for `ffai::conv3d` — the volumetric 3D
//! convolution (NCDHW direct conv, the `steel_conv 3D` audit row).
//!
//! Validates proc-macro → IR → MSL → PSO → dispatch → readback against
//! a straight nine-loop CPU reference. Covers:
//!   - `conv3d_generic`: strided / padded dense 3D conv, exercising the
//!     in-kernel padding clamp on the depth, row, and column axes
//!   - `conv3d_grouped`: dilation (atrous) + grouped channels, including
//!     the depthwise case (`groups == in_ch`)
//!   - f32 / f16 / bf16
//!
//! macOS-gated: needs an actual Metal device to dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::conv3d::{conv3d_generic, conv3d_grouped};

/// A 3D conv configuration. `dilation`/`groups` default to 1 for the
/// dense `conv3d_generic` tests; the grouped tests set them explicitly.
#[derive(Clone, Copy)]
struct Conv3dShape {
    batch: usize,
    in_ch: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
}

impl Conv3dShape {
    fn eff_kd(&self) -> usize { (self.kd - 1) * self.dilation_d + 1 }
    fn eff_kh(&self) -> usize { (self.kh - 1) * self.dilation_h + 1 }
    fn eff_kw(&self) -> usize { (self.kw - 1) * self.dilation_w + 1 }
    fn out_d(&self) -> usize { (self.in_d + 2 * self.pad_d - self.eff_kd()) / self.stride_d + 1 }
    fn out_h(&self) -> usize { (self.in_h + 2 * self.pad_h - self.eff_kh()) / self.stride_h + 1 }
    fn out_w(&self) -> usize { (self.in_w + 2 * self.pad_w - self.eff_kw()) / self.stride_w + 1 }
    fn icpg(&self) -> usize { self.in_ch / self.groups }
    fn ocpg(&self) -> usize { self.out_ch / self.groups }
}

/// CPU reference: NCDHW input, OIDHW weight (I dimension = `in_ch /
/// groups`), NCDHW output. Padding taps (input index outside the real
/// volume) contribute zero; dilation scales the tap offsets. All f32.
// `oc` indexes `bias` but is also a coordinate used to derive every
// other index, so the explicit range loop is the readable form.
#[allow(clippy::needless_range_loop)]
fn naive_conv3d(input: &[f32], weight: &[f32], bias: &[f32], s: &Conv3dShape) -> Vec<f32> {
    let (out_d, out_h, out_w) = (s.out_d(), s.out_h(), s.out_w());
    let (icpg, ocpg) = (s.icpg(), s.ocpg());
    let mut out = vec![0.0f32; s.batch * s.out_ch * out_d * out_h * out_w];
    for n in 0..s.batch {
        for oc in 0..s.out_ch {
            let group = oc / ocpg;
            let ic_base = group * icpg;
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[oc];
                        for wic in 0..icpg {
                            let real_ic = ic_base + wic;
                            for kz in 0..s.kd {
                                for ky in 0..s.kh {
                                    for kx in 0..s.kw {
                                        // Padded-frame coords → real coords.
                                        let pd = od * s.stride_d + kz * s.dilation_d;
                                        let ph = oh * s.stride_h + ky * s.dilation_h;
                                        let pw = ow * s.stride_w + kx * s.dilation_w;
                                        if pd < s.pad_d
                                            || pd >= s.pad_d + s.in_d
                                            || ph < s.pad_h
                                            || ph >= s.pad_h + s.in_h
                                            || pw < s.pad_w
                                            || pw >= s.pad_w + s.in_w
                                        {
                                            continue;
                                        }
                                        let id = pd - s.pad_d;
                                        let ih = ph - s.pad_h;
                                        let iw = pw - s.pad_w;
                                        let in_idx =
                                            (((n * s.in_ch + real_ic) * s.in_d + id) * s.in_h + ih)
                                                * s.in_w
                                                + iw;
                                        let w_idx = (((oc * icpg + wic) * s.kd + kz) * s.kh + ky)
                                            * s.kw
                                            + kx;
                                        acc += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        let o_idx = (((n * s.out_ch + oc) * out_d + od) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
    }
    out
}

/// Common buffer-map setup shared by both kernel dispatch paths.
fn base_buffers(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &Conv3dShape,
) -> BTreeMap<String, Vec<u8>> {
    let (out_d, out_h, out_w) = (s.out_d(), s.out_h(), s.out_w());
    let n_out = s.batch * s.out_ch * out_d * out_h * out_w;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("input".into(), pack_bytes(input, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("bias".into(), pack_bytes(bias, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_d".into(), u(s.in_d));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("out_ch".into(), u(s.out_ch));
    buffers.insert("out_d".into(), u(out_d));
    buffers.insert("out_h".into(), u(out_h));
    buffers.insert("out_w".into(), u(out_w));
    buffers.insert("kd".into(), u(s.kd));
    buffers.insert("kh".into(), u(s.kh));
    buffers.insert("kw".into(), u(s.kw));
    buffers.insert("stride_d".into(), u(s.stride_d));
    buffers.insert("stride_h".into(), u(s.stride_h));
    buffers.insert("stride_w".into(), u(s.stride_w));
    buffers.insert("pad_d".into(), u(s.pad_d));
    buffers.insert("pad_h".into(), u(s.pad_h));
    buffers.insert("pad_w".into(), u(s.pad_w));
    buffers
}

/// Dispatch `conv3d_generic` (dense path) and read back the output.
fn run_conv3d_generic(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &Conv3dShape,
) -> Vec<f32> {
    let (out_d, out_h, out_w) = (s.out_d(), s.out_h(), s.out_w());
    let n_out = s.batch * s.out_ch * out_d * out_h * out_w;
    let buffers = base_buffers(input, weight, bias, dt, s);

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = conv3d_generic::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: one thread per output element. Tile by 256 threads/TG.
    let tpg = 256usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("conv3d_generic dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

/// Dispatch `conv3d_grouped` (dilation + groups) and read back.
fn run_conv3d_grouped(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &Conv3dShape,
) -> Vec<f32> {
    let (out_d, out_h, out_w) = (s.out_d(), s.out_h(), s.out_w());
    let n_out = s.batch * s.out_ch * out_d * out_h * out_w;
    let mut buffers = base_buffers(input, weight, bias, dt, s);
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("dilation_d".into(), u(s.dilation_d));
    buffers.insert("dilation_h".into(), u(s.dilation_h));
    buffers.insert("dilation_w".into(), u(s.dilation_w));
    buffers.insert("icpg".into(), u(s.icpg()));
    buffers.insert("ocpg".into(), u(s.ocpg()));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = conv3d_grouped::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("conv3d_grouped dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

/// A dense (groups=1, dilation=1) shape for the `conv3d_generic` tests.
fn dense_shape(
    batch: usize,
    in_ch: usize,
    spatial: (usize, usize, usize),
    out_ch: usize,
    kernel: (usize, usize, usize),
    stride: (usize, usize, usize),
    pad: (usize, usize, usize),
) -> Conv3dShape {
    Conv3dShape {
        batch,
        in_ch,
        in_d: spatial.0,
        in_h: spatial.1,
        in_w: spatial.2,
        out_ch,
        kd: kernel.0,
        kh: kernel.1,
        kw: kernel.2,
        stride_d: stride.0,
        stride_h: stride.1,
        stride_w: stride.2,
        pad_d: pad.0,
        pad_h: pad.1,
        pad_w: pad.2,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    }
}

#[test]
fn conv3d_generic_with_padding_matches_naive_f32() {
    let _g = gpu_lock();
    // Overlapping 3×3×3 stride-1 conv with 1-voxel padding — exercises
    // the in-kernel padding clamp on every depth/row/col edge.
    let s = dense_shape(2, 3, (7, 9, 8), 5, (3, 3, 3), (1, 1, 1), (1, 1, 1));
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv3d(&input, &weight, &bias, &s);
    let actual = run_conv3d_generic(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv3d_generic f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_generic_strided_matches_naive_f32() {
    let _g = gpu_lock();
    // Strided 2×3×3 conv, no padding — anisotropic kernel + stride, the
    // shape of a video-VLM (time, height, width) patch stem.
    let s = dense_shape(1, 4, (12, 16, 14), 6, (2, 3, 3), (2, 2, 2), (0, 0, 0));
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 19, 9.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 13, 6.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv3d(&input, &weight, &bias, &s);
    let actual = run_conv3d_generic(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv3d_generic strided f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_generic_patch_stem_matches_naive_f32() {
    let _g = gpu_lock();
    // Non-overlapping volumetric patch conv: kernel == stride. The 3D
    // analogue of a vision-patch stem.
    let s = dense_shape(1, 3, (8, 16, 16), 8, (2, 4, 4), (2, 4, 4), (0, 0, 0));
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 41, 20.0);
    let bias = ramp(s.out_ch, 7, 3.0);
    let expected = naive_conv3d(&input, &weight, &bias, &s);
    let actual = run_conv3d_generic(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "conv3d_generic patch stem f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_generic_matches_naive_f16() {
    let _g = gpu_lock();
    let s = dense_shape(1, 3, (6, 8, 8), 4, (3, 3, 3), (1, 1, 1), (1, 1, 1));
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_conv3d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv3d_generic(&input, &weight, &bias, Dt::F16, &s);
    let diff = max_abs_diff(&expected, &actual);
    // 81-term reduction (3·3·3·3) in f16 — wider tolerance.
    assert!(diff < 1e-1, "conv3d_generic f16: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_generic_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = dense_shape(1, 4, (6, 7, 7), 3, (3, 3, 3), (1, 1, 1), (1, 1, 1));
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 29, 14.0);
    let weight = ramp(s.out_ch * s.in_ch * s.kd * s.kh * s.kw, 19, 9.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_conv3d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv3d_generic(&input, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-1, "conv3d_generic bf16: max |diff| = {diff:.2e}");
}

// ── Grouped / dilated 3D conv: conv3d_grouped ────────────────────────

#[test]
fn conv3d_grouped_dilated_matches_naive_f32() {
    let _g = gpu_lock();
    // Dilation-2 3×3×3 conv, groups=1 (pure atrous test).
    let mut s = dense_shape(1, 3, (14, 16, 16), 6, (3, 3, 3), (1, 1, 1), (2, 2, 2));
    s.dilation_d = 2;
    s.dilation_h = 2;
    s.dilation_w = 2;
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 31, 15.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kd * s.kh * s.kw, 23, 11.0);
    let bias = ramp(s.out_ch, 5, 2.0);
    let expected = naive_conv3d(&input, &weight, &bias, &s);
    let actual = run_conv3d_grouped(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv3d_grouped dilated f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_grouped_depthwise_matches_naive_f32() {
    let _g = gpu_lock();
    // Depthwise 3D conv: groups == in_ch == out_ch, one channel/group.
    let mut s = dense_shape(2, 6, (8, 10, 10), 6, (3, 3, 3), (1, 1, 1), (1, 1, 1));
    s.groups = 6;
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 19, 9.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kd * s.kh * s.kw, 13, 6.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv3d(&input, &weight, &bias, &s);
    let actual = run_conv3d_grouped(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv3d_grouped depthwise f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_grouped_groups_and_dilation_matches_naive_f32() {
    let _g = gpu_lock();
    // groups=2 + dilation=2 + stride=2 — every degree of freedom at once.
    let mut s = dense_shape(1, 6, (16, 18, 18), 8, (3, 3, 3), (2, 2, 2), (2, 2, 2));
    s.dilation_d = 2;
    s.dilation_h = 2;
    s.dilation_w = 2;
    s.groups = 2;
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 29, 14.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kd * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let expected = naive_conv3d(&input, &weight, &bias, &s);
    let actual = run_conv3d_grouped(&input, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-3, "conv3d_grouped groups+dilation f32: max |diff| = {diff:.2e}");
}

#[test]
fn conv3d_grouped_depthwise_matches_naive_bf16() {
    let _g = gpu_lock();
    let mut s = dense_shape(1, 4, (6, 8, 8), 4, (3, 3, 3), (1, 1, 1), (1, 1, 1));
    s.groups = 4;
    let input = ramp(s.batch * s.in_ch * s.in_d * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.out_ch * s.icpg() * s.kd * s.kh * s.kw, 17, 8.0);
    let bias = ramp(s.out_ch, 3, 1.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_conv3d(&round(&input), &round(&weight), &round(&bias), &s);
    let actual = run_conv3d_grouped(&input, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-1, "conv3d_grouped depthwise bf16: max |diff| = {diff:.2e}");
}

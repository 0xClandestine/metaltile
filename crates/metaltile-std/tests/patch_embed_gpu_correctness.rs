//! End-to-end GPU correctness for `ffai::patch_embed` — fused image
//! unfold + linear projection for vision-transformer patch embedding.
//!
//! Validates the fused unfold/projection against a CPU reference that
//! does the two steps separately (explicit unfold to `[num_patches,
//! patch_dim]`, then a GEMM against the `[hidden, patch_dim]` weight).
//! Covers:
//!   - the SigLIP / Qwen-VL patch shape (14×14, 3-channel)
//!   - the CLIP / Gemma-VL patch shape (16×16)
//!   - f32 / f16 / bf16
//!
//! macOS-gated: needs an actual Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, ramp, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::patch_embed::patch_embed;

#[derive(Clone, Copy)]
struct PatchShape {
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    patch_h: usize,
    patch_w: usize,
    hidden: usize,
}

impl PatchShape {
    fn patches_h(&self) -> usize { self.in_h / self.patch_h }
    fn patches_w(&self) -> usize { self.in_w / self.patch_w }
    fn num_patches(&self) -> usize { self.patches_h() * self.patches_w() }
    fn patch_dim(&self) -> usize { self.in_ch * self.patch_h * self.patch_w }
}

/// CPU reference: explicit unfold then projection. `image` is NCHW
/// (single image), `weight` is flat `[hidden, patch_dim]`, output is
/// `[num_patches, hidden]`. All f32.
fn naive_patch_embed(image: &[f32], weight: &[f32], bias: &[f32], s: &PatchShape) -> Vec<f32> {
    let patch_dim = s.patch_dim();
    let input_plane = s.in_h * s.in_w;

    // Step 1: unfold the image into [num_patches, patch_dim].
    let mut unfolded = vec![0.0f32; s.num_patches() * patch_dim];
    for ph in 0..s.patches_h() {
        for pw in 0..s.patches_w() {
            let patch = ph * s.patches_w() + pw;
            for ic in 0..s.in_ch {
                for py in 0..s.patch_h {
                    for px in 0..s.patch_w {
                        let img_y = ph * s.patch_h + py;
                        let img_x = pw * s.patch_w + px;
                        let img_idx = ic * input_plane + img_y * s.in_w + img_x;
                        let col = ic * s.patch_h * s.patch_w + py * s.patch_w + px;
                        unfolded[patch * patch_dim + col] = image[img_idx];
                    }
                }
            }
        }
    }

    // Step 2: project — out[patch, h] = bias[h] + sum_c unfolded·weight.
    let mut out = vec![0.0f32; s.num_patches() * s.hidden];
    for patch in 0..s.num_patches() {
        for h in 0..s.hidden {
            let mut acc = bias[h];
            for c in 0..patch_dim {
                acc += unfolded[patch * patch_dim + c] * weight[h * patch_dim + c];
            }
            out[patch * s.hidden + h] = acc;
        }
    }
    out
}

fn run_patch_embed(
    image: &[f32],
    weight: &[f32],
    bias: &[f32],
    dt: Dt,
    s: &PatchShape,
) -> Vec<f32> {
    let n_out = s.num_patches() * s.hidden;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("image".into(), pack_bytes(image, dt));
    buffers.insert("weight".into(), pack_bytes(weight, dt));
    buffers.insert("bias".into(), pack_bytes(bias, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n_out], dt));
    let u = |v: usize| (v as u32).to_le_bytes().to_vec();
    buffers.insert("in_ch".into(), u(s.in_ch));
    buffers.insert("in_h".into(), u(s.in_h));
    buffers.insert("in_w".into(), u(s.in_w));
    buffers.insert("patch_h".into(), u(s.patch_h));
    buffers.insert("patch_w".into(), u(s.patch_w));
    buffers.insert("hidden".into(), u(s.hidden));

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = patch_embed::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Grid3D;

    let tpg = 256usize;
    let grid = n_out.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid, 1, 1], [tpg, 1, 1])
        .expect("patch_embed dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(n_out);
    out
}

#[test]
fn patch_embed_patch14_matches_naive_f32() {
    let _g = gpu_lock();
    // SigLIP / Qwen-VL: 14×14 patch, 3 channels, projecting a small grid.
    let s = PatchShape { in_ch: 3, in_h: 28, in_w: 42, patch_h: 14, patch_w: 14, hidden: 32 };
    let image = ramp(s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.hidden * s.patch_dim(), 41, 20.0);
    let bias = ramp(s.hidden, 11, 5.0);
    let expected = naive_patch_embed(&image, &weight, &bias, &s);
    let actual = run_patch_embed(&image, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "patch_embed patch14 f32: max |diff| = {diff:.2e}");
}

#[test]
fn patch_embed_patch16_matches_naive_f32() {
    let _g = gpu_lock();
    // CLIP / Gemma-VL: 16×16 patch.
    let s = PatchShape { in_ch: 3, in_h: 32, in_w: 48, patch_h: 16, patch_w: 16, hidden: 24 };
    let image = ramp(s.in_ch * s.in_h * s.in_w, 29, 14.0);
    let weight = ramp(s.hidden * s.patch_dim(), 31, 15.0);
    let bias = ramp(s.hidden, 7, 3.0);
    let expected = naive_patch_embed(&image, &weight, &bias, &s);
    let actual = run_patch_embed(&image, &weight, &bias, Dt::F32, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-3, "patch_embed patch16 f32: max |diff| = {diff:.2e}");
}

#[test]
fn patch_embed_matches_naive_f16() {
    let _g = gpu_lock();
    let s = PatchShape { in_ch: 3, in_h: 28, in_w: 28, patch_h: 14, patch_w: 14, hidden: 16 };
    let image = ramp(s.in_ch * s.in_h * s.in_w, 37, 18.0);
    let weight = ramp(s.hidden * s.patch_dim(), 41, 20.0);
    let bias = ramp(s.hidden, 11, 5.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::F16.round(x)).collect() };
    let expected = naive_patch_embed(&round(&image), &round(&weight), &round(&bias), &s);
    let actual = run_patch_embed(&image, &weight, &bias, Dt::F16, &s);
    let diff = max_abs_diff(&expected, &actual);
    // 588-term reduction (3·14·14) in f16.
    assert!(diff < 2e-1, "patch_embed f16: max |diff| = {diff:.2e}");
}

#[test]
fn patch_embed_matches_naive_bf16() {
    let _g = gpu_lock();
    let s = PatchShape { in_ch: 2, in_h: 16, in_w: 16, patch_h: 8, patch_w: 8, hidden: 12 };
    let image = ramp(s.in_ch * s.in_h * s.in_w, 23, 11.0);
    let weight = ramp(s.hidden * s.patch_dim(), 17, 8.0);
    let bias = ramp(s.hidden, 5, 2.0);
    let round = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|&x| Dt::Bf16.round(x)).collect() };
    let expected = naive_patch_embed(&round(&image), &round(&weight), &round(&bias), &s);
    let actual = run_patch_embed(&image, &weight, &bias, Dt::Bf16, &s);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 2e-1, "patch_embed bf16: max |diff| = {diff:.2e}");
}

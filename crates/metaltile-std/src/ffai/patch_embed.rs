//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused image-unfold + linear-projection patch embedding for vision
//! transformers.
//!
//! A ViT patch embedding takes an image, cuts it into non-overlapping
//! `patch_h × patch_w` tiles, flattens each tile into a
//! `in_ch · patch_h · patch_w` vector, and linearly projects every
//! vector into the model's hidden dimension. Done as two ops it
//! materialises the unfolded `[num_patches, patch_dim]` tensor in global
//! memory — pure bandwidth waste, since each unfolded value is read
//! exactly once by the projection GEMM. This kernel fuses the unfold and
//! the projection: each thread gathers its patch's pixels straight from
//! the image and dots them with one weight row, no intermediate buffer.
//!
//! It differs from `conv2d` in layout, not arithmetic — `conv2d` keeps
//! the NCHW image convention and writes NCHW output; `patch_embed` takes
//! the same NCHW image but treats the weight as a flat linear matrix
//! `[hidden, patch_dim]` and writes transformer-token output
//! `[num_patches, hidden]`, which is what a ViT block consumes directly.
//!
//! Layouts (NCHW image, flat linear weight):
//!
//!   image    [in_ch, in_h, in_w]                   T   (single image)
//!   weight   [hidden, in_ch * patch_h * patch_w]   T
//!   bias     [hidden]                               T
//!   out      [num_patches, hidden]                  T
//!
//!   patches_h  = in_h / patch_h
//!   patches_w  = in_w / patch_w
//!   num_patches = patches_h * patches_w
//!   patch_dim  = in_ch * patch_h * patch_w
//!
//! One thread per output element `(patch, h)` where `patch` indexes the
//! flattened patch grid (row-major over the `patches_h × patches_w`
//! grid) and `h` indexes the hidden dimension. The thread walks the
//! patch's `in_ch × patch_h × patch_w` pixels, dotting each with the
//! matching weight column, accumulating in fp32. Generic over T.
//!
//! Patch order matches PyTorch `unfold` / `nn.Conv2d` flattening:
//! the weight column for `(ic, py, px)` is at
//! `ic*patch_h*patch_w + py*patch_w + px`.
//!
//! Codegen-only. Correctness validated by `patch_embed_gpu_correctness`.

use metaltile::kernel;

#[kernel]
pub fn patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
) {
    // Flat output index → (patch, h). One thread per output element.
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    // Top-left pixel of this patch in the image.
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let w_row_base = h * patch_dim;
    let mut acc = load(bias[h]).cast::<f32>();
    // Walk the patch's in_ch × patch_h × patch_w pixels, dotting each
    // with the corresponding weight column. The patch grid divides the
    // image exactly (caller precondition), so every read is in-bounds —
    // no padding / clamp logic needed.
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic_base = ic * input_plane;
        let w_ic_base = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic_base + (py0 + py) * in_w;
            let w_row = w_row_base + w_ic_base + py * patch_w;
            for px in range(0u32, patch_w, 1u32) {
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let wt = load(weight[w_row + px]).cast::<f32>();
                acc = acc + pix * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };
    use metaltile::test_kernel;

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn pack_u32_scalar(v: usize) -> Vec<u8> { (v as u32).to_le_bytes().to_vec() }

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    /// CPU reference: explicit unfold then projection.
    fn naive_patch_embed(
        image: &[f32],
        weight: &[f32],
        bias: &[f32],
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let patches_h = in_h / patch_h;
        let patches_w = in_w / patch_w;
        let num_patches = patches_h * patches_w;
        let patch_dim = in_ch * patch_h * patch_w;
        let input_plane = in_h * in_w;
        let mut out = vec![0.0f32; num_patches * hidden];
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                let patch = ph * patches_w + pw;
                for h in 0..hidden {
                    let mut acc = bias[h];
                    for ic in 0..in_ch {
                        for py in 0..patch_h {
                            for px in 0..patch_w {
                                let img_y = ph * patch_h + py;
                                let img_x = pw * patch_w + px;
                                let img_idx = ic * input_plane + img_y * in_w + img_x;
                                let col = ic * patch_h * patch_w + py * patch_w + px;
                                acc += image[img_idx] * weight[h * patch_dim + col];
                            }
                        }
                    }
                    out[patch * hidden + h] = acc;
                }
            }
        }
        out
    }

    #[test_kernel(name = "patch_embed/patch14_f32", dtypes = [f32], tol = 2e-3)]
    fn test_patch14_f32(dt: DType) -> TestSetup {
        let (in_ch, in_h, in_w, patch_h, patch_w, hidden) = (3, 28, 42, 14, 14, 32);
        let n_out = (in_h / patch_h) * (in_w / patch_w) * hidden;
        let image = ramp(in_ch * in_h * in_w, 37, 18.0);
        let weight = ramp(hidden * in_ch * patch_h * patch_w, 41, 20.0);
        let bias = ramp(hidden, 11, 5.0);
        let expected =
            naive_patch_embed(&image, &weight, &bias, in_ch, in_h, in_w, patch_h, patch_w, hidden);

        let mut k = patch_embed::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("image", pack(&image, dt), dt))
            .input(TestBuffer::from_vec("weight", pack(&weight, dt), dt))
            .input(TestBuffer::from_vec("bias", pack(&bias, dt), dt))
            .input(TestBuffer::from_vec("in_ch", pack_u32_scalar(in_ch), DType::U32))
            .input(TestBuffer::from_vec("in_h", pack_u32_scalar(in_h), DType::U32))
            .input(TestBuffer::from_vec("in_w", pack_u32_scalar(in_w), DType::U32))
            .input(TestBuffer::from_vec("patch_h", pack_u32_scalar(patch_h), DType::U32))
            .input(TestBuffer::from_vec("patch_w", pack_u32_scalar(patch_w), DType::U32))
            .input(TestBuffer::from_vec("hidden", pack_u32_scalar(hidden), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[test_kernel(name = "patch_embed/patch16_f32", dtypes = [f32], tol = 2e-3)]
    fn test_patch16_f32(dt: DType) -> TestSetup {
        let (in_ch, in_h, in_w, patch_h, patch_w, hidden) = (3, 32, 48, 16, 16, 24);
        let n_out = (in_h / patch_h) * (in_w / patch_w) * hidden;
        let image = ramp(in_ch * in_h * in_w, 29, 14.0);
        let weight = ramp(hidden * in_ch * patch_h * patch_w, 31, 15.0);
        let bias = ramp(hidden, 7, 3.0);
        let expected =
            naive_patch_embed(&image, &weight, &bias, in_ch, in_h, in_w, patch_h, patch_w, hidden);

        let mut k = patch_embed::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("image", pack(&image, dt), dt))
            .input(TestBuffer::from_vec("weight", pack(&weight, dt), dt))
            .input(TestBuffer::from_vec("bias", pack(&bias, dt), dt))
            .input(TestBuffer::from_vec("in_ch", pack_u32_scalar(in_ch), DType::U32))
            .input(TestBuffer::from_vec("in_h", pack_u32_scalar(in_h), DType::U32))
            .input(TestBuffer::from_vec("in_w", pack_u32_scalar(in_w), DType::U32))
            .input(TestBuffer::from_vec("patch_h", pack_u32_scalar(patch_h), DType::U32))
            .input(TestBuffer::from_vec("patch_w", pack_u32_scalar(patch_w), DType::U32))
            .input(TestBuffer::from_vec("hidden", pack_u32_scalar(hidden), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[test_kernel(name = "patch_embed/f16", dtypes = [f16], tol = 2e-1)]
    fn test_patch_embed_f16(dt: DType) -> TestSetup {
        let (in_ch, in_h, in_w, patch_h, patch_w, hidden) = (3, 28, 28, 14, 14, 16);
        let n_out = (in_h / patch_h) * (in_w / patch_w) * hidden;
        let image_r: Vec<f32> =
            ramp(in_ch * in_h * in_w, 37, 18.0).iter().map(|&x| round(x, dt)).collect();
        let weight_r: Vec<f32> = ramp(hidden * in_ch * patch_h * patch_w, 41, 20.0)
            .iter()
            .map(|&x| round(x, dt))
            .collect();
        let bias_r: Vec<f32> = ramp(hidden, 11, 5.0).iter().map(|&x| round(x, dt)).collect();
        let expected = naive_patch_embed(
            &image_r, &weight_r, &bias_r, in_ch, in_h, in_w, patch_h, patch_w, hidden,
        );

        let mut k = patch_embed::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("image", pack(&image_r, dt), dt))
            .input(TestBuffer::from_vec("weight", pack(&weight_r, dt), dt))
            .input(TestBuffer::from_vec("bias", pack(&bias_r, dt), dt))
            .input(TestBuffer::from_vec("in_ch", pack_u32_scalar(in_ch), DType::U32))
            .input(TestBuffer::from_vec("in_h", pack_u32_scalar(in_h), DType::U32))
            .input(TestBuffer::from_vec("in_w", pack_u32_scalar(in_w), DType::U32))
            .input(TestBuffer::from_vec("patch_h", pack_u32_scalar(patch_h), DType::U32))
            .input(TestBuffer::from_vec("patch_w", pack_u32_scalar(patch_w), DType::U32))
            .input(TestBuffer::from_vec("hidden", pack_u32_scalar(hidden), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[test_kernel(name = "patch_embed/bf16", dtypes = [bf16], tol = 2e-1)]
    fn test_patch_embed_bf16(dt: DType) -> TestSetup {
        let (in_ch, in_h, in_w, patch_h, patch_w, hidden) = (2, 16, 16, 8, 8, 12);
        let n_out = (in_h / patch_h) * (in_w / patch_w) * hidden;
        let image_r: Vec<f32> =
            ramp(in_ch * in_h * in_w, 23, 11.0).iter().map(|&x| round(x, dt)).collect();
        let weight_r: Vec<f32> = ramp(hidden * in_ch * patch_h * patch_w, 17, 8.0)
            .iter()
            .map(|&x| round(x, dt))
            .collect();
        let bias_r: Vec<f32> = ramp(hidden, 5, 2.0).iter().map(|&x| round(x, dt)).collect();
        let expected = naive_patch_embed(
            &image_r, &weight_r, &bias_r, in_ch, in_h, in_w, patch_h, patch_w, hidden,
        );

        let mut k = patch_embed::kernel_ir_for(dt);
        k.mode = metaltile::core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("image", pack(&image_r, dt), dt))
            .input(TestBuffer::from_vec("weight", pack(&weight_r, dt), dt))
            .input(TestBuffer::from_vec("bias", pack(&bias_r, dt), dt))
            .input(TestBuffer::from_vec("in_ch", pack_u32_scalar(in_ch), DType::U32))
            .input(TestBuffer::from_vec("in_h", pack_u32_scalar(in_h), DType::U32))
            .input(TestBuffer::from_vec("in_w", pack_u32_scalar(in_w), DType::U32))
            .input(TestBuffer::from_vec("patch_h", pack_u32_scalar(patch_h), DType::U32))
            .input(TestBuffer::from_vec("patch_w", pack_u32_scalar(patch_w), DType::U32))
            .input(TestBuffer::from_vec("hidden", pack_u32_scalar(hidden), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }
}

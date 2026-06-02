//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized patch embedding** — the weight-quantized counterpart
//! of `ffai/patch_embed.rs`. Patch embedding is a linear projection
//! (`out[patch, h] = bias[h] + Σ_col image_patch[col] · W[h, col]`, `W` is
//! `[hidden, patch_dim]`), so its projection weight is a genuine quantizable
//! parameter — quantized along the `patch_dim` contraction in the spec formats
//! (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8 + legacy fp4/fp8 + symmetric int8).
//!
//! Only the weight is quantized (the per-channel `bias` stays `T` — it is tiny
//! and precision-sensitive). Geometry is identical to the dense `patch_embed`:
//! **Grid3D**, one thread per output element (`program_id::<0>()` = flat
//! `patch·hidden + h`). The per-`col` weight decode reuses the DSL decode
//! intrinsics; `patch_dim` is a multiple of `block_size` (4-bit `block_size` a
//! multiple of 8). fp8_e4m3 reuses the nvfp8 kernel. Codegen-only; correctness
//! pinned by the in-source `#[test_kernel]`s vs a `quant::format::dequant` oracle.

use metaltile::kernel;

/// mxfp4 quantized patch embed — E2M1 weight (block 32), E8M0 pow-2 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp4_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let w_packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_row_pack = h * w_packs_per_row;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale =
                    exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
                acc = acc + (e2m1_decode(nib) * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp4 quantized patch embed — E2M1 weight (block 16), E4M3 micro-scale × global.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp4_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let w_packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_row_pack = h * w_packs_per_row;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale =
                    e4m3_decode(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global;
                acc = acc + (e2m1_decode(nib) * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp4 quantized patch embed — E2M1 weight (group 32), per-group FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let w_packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_row_pack = h * w_packs_per_row;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + (e2m1_decode(nib) * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E4M3) quantized patch embed — 8-bit weight (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e4m3_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let n_blocks = patch_dim / block_size;
    let w_row = h * patch_dim;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
                let scale =
                    exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// mxfp8 (E5M2) quantized patch embed — 8-bit weight (block 32), E8M0 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxfp8_e5m2_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let n_blocks = patch_dim / block_size;
    let w_row = h * patch_dim;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
                let scale =
                    exp2(load(scales[w_row_blk + col / block_size]).cast::<f32>() - 127.0f32);
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp8 (E5M2) quantized patch embed — 8-bit weight (group 32), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let n_blocks = patch_dim / block_size;
    let w_row = h * patch_dim;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let elem = e5m2_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// nvfp8 quantized patch embed — E4M3 weight (block 16), per-block FP32 scale.
/// Also serves **fp8_e4m3** (same 8-bit-E4M3 + f32-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let n_blocks = patch_dim / block_size;
    let w_row = h * patch_dim;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let elem = e4m3_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Symmetric int8 quantized patch embed — 8-bit codes (group 64), FP32 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let h = idx % hidden;
    let patch = idx / hidden;
    let patches_w = in_w / patch_w;
    let py0 = (patch / patches_w) * patch_h;
    let px0 = (patch - (patch / patches_w) * patches_w) * patch_w;
    let input_plane = in_h * in_w;
    let patch_dim = in_ch * patch_h * patch_w;
    let n_blocks = patch_dim / block_size;
    let w_row = h * patch_dim;
    let w_row_blk = h * n_blocks;
    let mut acc = load(bias[h]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let img_ic = ic * input_plane;
        let col_ic = ic * patch_h * patch_w;
        for py in range(0u32, patch_h, 1u32) {
            let img_row = img_ic + (py0 + py) * in_w;
            for px in range(0u32, patch_w, 1u32) {
                let col = col_ic + py * patch_w + px;
                let pix = load(image[img_row + px0 + px]).cast::<f32>();
                let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
                let scale = load(scales[w_row_blk + col / block_size]);
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn patch_setup(
        kernel: Kernel,
        fmt: QFormat,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> TestSetup {
        let patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        let n_out = patches * hidden;
        let image_f = ramp(in_ch * in_h * in_w, 13, 6.0);
        let bias_f = ramp(hidden, 5, 2.0);
        // Quantize the [hidden, patch_dim] projection weight via the shared codec.
        let w_f = ramp(hidden * patch_dim, 11, 4.0);
        let p = crate::quant::format::pack(fmt, &w_f, hidden, patch_dim);
        let wdq = crate::quant::format::dequant(fmt, &p, hidden, patch_dim);
        let image = unpack_f32(&pack_f32(&image_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        // Oracle: explicit unfold + projection over the dequantized weight.
        let patches_w = in_w / patch_w;
        let input_plane = in_h * in_w;
        let mut expected = vec![0.0f32; n_out];
        for patch in 0..patches {
            let py0 = (patch / patches_w) * patch_h;
            let px0 = (patch % patches_w) * patch_w;
            for hh in 0..hidden {
                let mut acc = bias[hh];
                for ic in 0..in_ch {
                    for py in 0..patch_h {
                        for px in 0..patch_w {
                            let col = ic * patch_h * patch_w + py * patch_w + px;
                            let pix = image[ic * input_plane + (py0 + py) * in_w + (px0 + px)];
                            acc += pix * wdq[hh * patch_dim + col];
                        }
                    }
                }
                expected[patch * hidden + hh] = acc;
            }
        }
        let weight_dt = if fmt.element_bits() == 4 { DType::U32 } else { DType::U8 };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("image", pack_f32(&image_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // in_ch=4, patch 8×8 → patch_dim 256 (÷ 16/32/64); 16×16 image → 4 patches.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxfp4_patch_embed::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_nvfp4_patch_embed::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_fp4_patch_embed::kernel_ir_for(dt), QFormat::Fp4, 4, 16, 16, 8, 8, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxfp8_e4m3_patch_embed::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxfp8_e5m2_patch_embed::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_fp8_e5m2_patch_embed::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_nvfp8_patch_embed::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_nvfp8_patch_embed::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_int8_patch_embed::kernel_ir_for(dt), QFormat::Int8, 4, 16, 16, 8, 8, 64, dt)
    }
}

/// Decode-shape benches: ViT-class patch embed (3×224×224 image, 16×16 patches,
/// hidden 768 → patch_dim 768). Grid3D, one thread per output element.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn patch_bench(
        kernel: Kernel,
        fmt: QFormat,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> BenchSetup {
        let patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        let n_out = patches * hidden;
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (hidden * patch_dim / 8, DType::U32)
        } else {
            (hidden * patch_dim, DType::U8)
        };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let n_blocks = hidden * (patch_dim / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + in_ch * in_h * in_w * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("image", in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", hidden, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            .flops(2 * n_out as u64 * patch_dim as u64)
            .with_shape_label(format!("{} patches={patches} h={hidden} pd={patch_dim}", fmt.name()))
    }

    macro_rules! patch_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr, $name:literal) => {
            #[bench(name = $name, dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                patch_bench($kernel(dt), $fmt, 3, 224, 224, 16, 16, 768, dt)
            }
        };
    }
    patch_bench_fmt!(
        bench_mxfp4,
        mt_mxfp4_patch_embed::kernel_ir_for,
        QFormat::Mxfp4,
        "ffai/patch_embed_block/mxfp4"
    );
    patch_bench_fmt!(
        bench_nvfp4,
        mt_nvfp4_patch_embed::kernel_ir_for,
        QFormat::Nvfp4,
        "ffai/patch_embed_block/nvfp4"
    );
    patch_bench_fmt!(
        bench_fp4,
        mt_fp4_patch_embed::kernel_ir_for,
        QFormat::Fp4,
        "ffai/patch_embed_block/fp4"
    );
    patch_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_patch_embed::kernel_ir_for,
        QFormat::Mxfp8E4,
        "ffai/patch_embed_block/mxfp8_e4m3"
    );
    patch_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_patch_embed::kernel_ir_for,
        QFormat::Mxfp8E5,
        "ffai/patch_embed_block/mxfp8_e5m2"
    );
    patch_bench_fmt!(
        bench_fp8_e5m2,
        mt_fp8_e5m2_patch_embed::kernel_ir_for,
        QFormat::Fp8E5m2,
        "ffai/patch_embed_block/fp8_e5m2"
    );
    patch_bench_fmt!(
        bench_nvfp8,
        mt_nvfp8_patch_embed::kernel_ir_for,
        QFormat::Nvfp8,
        "ffai/patch_embed_block/nvfp8"
    );
    patch_bench_fmt!(
        bench_int8,
        mt_int8_patch_embed::kernel_ir_for,
        QFormat::Int8,
        "ffai/patch_embed_block/int8"
    );
}

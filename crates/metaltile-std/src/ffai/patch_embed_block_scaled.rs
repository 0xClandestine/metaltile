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

// ── FP16-scale twins (nvfp8 / fp4 / fp8_e5m2) ────────────────────────────────
// Identical element decode + weight indexing + Grid3D geometry to their
// FP32-scaled twin above; only the scale axis changes — read as a native `half`
// (`Tensor<f16>`) and cast to f32. The GPU half load matches the host
// `f16_scale_decode`, so the dequant-vs-oracle equality is preserved.

/// nvfp8 (FP16 scale) quantized patch embed — E4M3 weight (block 16), per-block
/// FP16 scale. Also serves **fp8_e4m3_f16** (same 8-bit-E4M3 + f16-scale shape).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_nvfp8_f16_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp4 (FP16 scale) quantized patch embed — E2M1 weight (group 32),
/// per-group FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp4_f16_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u32>,
    scales: Tensor<f16>,
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
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + (e2m1_decode(nib) * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// Legacy fp8 (E5M2, FP16 scale) quantized patch embed — 8-bit weight (group 32),
/// per-group FP16 scale.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_fp8_e5m2_f16_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ── Symmetric sub-byte integer patch embeds (int2/3/4/5/6 + MXINT2..6) ───────
// The projection weight `W[hidden, patch_dim]` is stored as signed N-bit
// two's-complement codes, tight-bit-packed LSB-first into u32 words. Each weight
// row (`patch_dim` codes) is word-aligned: row `h` begins at word
// `h · (patch_dim · BITS / 32)` (`patch_dim` is a multiple of 32, so every width
// keeps each row whole-word). For the `col`-th element of a row, the code lives at
// bit `col · BITS` within that row's bit-stream. Decode mirrors the proven
// `block_scaled_dequant` / `block_scaled_matmul` `int_*` macros: extract the low
// N bits with a straddle-aware two-word read, sign-extend in float (subtract 2^N
// when the top bit is set; `$half`/`$full` are 2^(N-1) / 2^N), then multiply by
// the block scale and the image pixel. Geometry is unchanged from the rest of the
// family (Grid3D, one thread per output element). `$half`/`$full` are passed as
// literals to keep the constexpr math out of the DSL shift operands.

/// FP32-scaled symmetric int patch embed (int2/3/4/5/6): per-element bit-stream
/// weight code × per-group FP32 scale, dotted with the unfolded image patch.
/// `w_row_word` indexes the row's tight bit-stream (`patch_dim · bits / 32` u32
/// words per row); `col` is the element index within the row.
macro_rules! int_patch_embed_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
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
            let words_per_row = patch_dim * $bits / 32u32;
            let n_blocks = patch_dim / block_size;
            let w_row_word = h * words_per_row;
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
                        let bit_off = col * $bits;
                        let word_idx = bit_off / 32u32;
                        let bit_in_w = bit_off & 31u32;
                        let bits_in_w0 = 32u32 - bit_in_w;
                        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                        let spill = $bits - lo_bits;
                        let w0 = load(weight[w_row_word + word_idx]);
                        let w1 = load(
                            weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                        );
                        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                        let q = lo | hi;
                        let qf = q.cast::<f32>();
                        let val = select(q >= $half, qf - $full, qf); // sign-extend
                        let scale = load(scales[w_row_blk + col / block_size]);
                        acc = acc + (val * scale) * pix;
                    }
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_patch_embed_f32!(mt_int2_patch_embed, 2u32, 2u32, 4.0f32);
int_patch_embed_f32!(mt_int3_patch_embed, 3u32, 4u32, 8.0f32);
int_patch_embed_f32!(mt_int4_patch_embed, 4u32, 8u32, 16.0f32);
int_patch_embed_f32!(mt_int5_patch_embed, 5u32, 16u32, 32.0f32);
int_patch_embed_f32!(mt_int6_patch_embed, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int patch embed (MXINT2/3/4/5/6): per-element bit-stream
/// weight code × pow-2 (E8M0) block scale `2^(bits-127)`, dotted with the image
/// patch. Same straddle-aware decode and per-row word alignment as
/// `int_patch_embed_f32`; only the scale axis differs (one u8 exponent per block
/// instead of a raw f32).
macro_rules! int_patch_embed_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
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
            let words_per_row = patch_dim * $bits / 32u32;
            let n_blocks = patch_dim / block_size;
            let w_row_word = h * words_per_row;
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
                        let bit_off = col * $bits;
                        let word_idx = bit_off / 32u32;
                        let bit_in_w = bit_off & 31u32;
                        let bits_in_w0 = 32u32 - bit_in_w;
                        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                        let spill = $bits - lo_bits;
                        let w0 = load(weight[w_row_word + word_idx]);
                        let w1 = load(
                            weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                        );
                        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                        let q = lo | hi;
                        let qf = q.cast::<f32>();
                        let val = select(q >= $half, qf - $full, qf); // sign-extend
                        let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                        let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                        acc = acc + (val * scale) * pix;
                    }
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_patch_embed_e8m0!(mt_mxint2_patch_embed, 2u32, 2u32, 4.0f32);
int_patch_embed_e8m0!(mt_mxint3_patch_embed, 3u32, 4u32, 8.0f32);
int_patch_embed_e8m0!(mt_mxint4_patch_embed, 4u32, 8u32, 16.0f32);
int_patch_embed_e8m0!(mt_mxint5_patch_embed, 5u32, 16u32, 32.0f32);
int_patch_embed_e8m0!(mt_mxint6_patch_embed, 6u32, 32u32, 64.0f32);

/// FP16-scaled symmetric int patch embed (int2/3/4/5/6): identical per-element
/// bit-stream decode and per-row word alignment as `int_patch_embed_f32`; only
/// the scale axis differs — read as a native `half` (`Tensor<f16>`) and cast to
/// f32 (one FP16 scale per group).
macro_rules! int_patch_embed_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            image: Tensor<T>,
            weight: Tensor<u32>,
            scales: Tensor<f16>,
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
            let words_per_row = patch_dim * $bits / 32u32;
            let n_blocks = patch_dim / block_size;
            let w_row_word = h * words_per_row;
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
                        let bit_off = col * $bits;
                        let word_idx = bit_off / 32u32;
                        let bit_in_w = bit_off & 31u32;
                        let bits_in_w0 = 32u32 - bit_in_w;
                        let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                        let spill = $bits - lo_bits;
                        let w0 = load(weight[w_row_word + word_idx]);
                        let w1 = load(
                            weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                        );
                        let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                        let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                        let q = lo | hi;
                        let qf = q.cast::<f32>();
                        let val = select(q >= $half, qf - $full, qf); // sign-extend
                        let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                        acc = acc + (val * scale) * pix;
                    }
                }
            }
            store(out[idx], acc.cast::<T>());
        }
    };
}
int_patch_embed_f16!(mt_int2_f16_patch_embed, 2u32, 2u32, 4.0f32);
int_patch_embed_f16!(mt_int3_f16_patch_embed, 3u32, 4u32, 8.0f32);
int_patch_embed_f16!(mt_int4_f16_patch_embed, 4u32, 8u32, 16.0f32);
int_patch_embed_f16!(mt_int5_f16_patch_embed, 5u32, 16u32, 32.0f32);
int_patch_embed_f16!(mt_int6_f16_patch_embed, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) quantized patch embed — 8-bit symmetric codes (byte layout,
/// group 64), per-group FP16 scale. Same byte-strided decode as `mt_int8_patch_embed`;
/// only the scale is read as a `half` and cast to f32.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_int8_f16_patch_embed<T>(
    image: Tensor<T>,
    weight: Tensor<u8>,
    scales: Tensor<f16>,
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
                let scale = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                acc = acc + (elem * scale) * pix;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

/// MXINT8 quantized patch embed — 8-bit symmetric codes (byte layout, block 32),
/// E8M0 pow-2 block scale `2^(bits-127)`. Element-strided like the 8-bit float
/// formats (one byte per code), decode is `int8_decode → val · scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_mxint8_patch_embed<T>(
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
                let elem = int8_decode(load(weight[w_row + col]).cast::<u32>());
                let sbits = load(scales[w_row_blk + col / block_size]).cast::<f32>();
                let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
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
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new formats pick up the right
        // buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
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

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). patch_dim 256 is a multiple of 32,
    // so each weight row's bit-stream is word-aligned for every width. The kernel
    // and oracle share the codec, so the GPU output tracks the unfold-then-project
    // reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_int2_patch_embed::kernel_ir_for(dt), QFormat::Int2, 4, 16, 16, 8, 8, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_int3_patch_embed::kernel_ir_for(dt), QFormat::Int3, 4, 16, 16, 8, 8, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_int4_patch_embed::kernel_ir_for(dt), QFormat::Int4, 4, 16, 16, 8, 8, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_int5_patch_embed::kernel_ir_for(dt), QFormat::Int5, 4, 16, 16, 8, 8, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_patch_embed(dt: DType) -> TestSetup {
        patch_setup(mt_int6_patch_embed::kernel_ir_for(dt), QFormat::Int6, 4, 16, 16, 8, 8, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxint2_patch_embed::kernel_ir_for(dt),
            QFormat::Mxint2,
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
    fn test_mxint3_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxint3_patch_embed::kernel_ir_for(dt),
            QFormat::Mxint3,
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
    fn test_mxint4_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxint4_patch_embed::kernel_ir_for(dt),
            QFormat::Mxint4,
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
    fn test_mxint5_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxint5_patch_embed::kernel_ir_for(dt),
            QFormat::Mxint5,
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
    fn test_mxint6_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxint6_patch_embed::kernel_ir_for(dt),
            QFormat::Mxint6,
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
    fn test_mxint8_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_mxint8_patch_embed::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }

    // FP16-scale twins of the FP32-scaled formats. Same element packing + Grid3D
    // geometry as their twin; only the scale buffer is f16. `fp8_e4m3_f16` reuses
    // the `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape). patch_dim 256 stays
    // a multiple of 32, so each weight row's bit-stream is word-aligned.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_nvfp8_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_nvfp8_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
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
    fn test_fp4_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_fp4_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Fp4F16,
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
    fn test_fp8_e5m2_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_fp8_e5m2_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
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
    fn test_int2_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_int2_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Int2F16,
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
    fn test_int3_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_int3_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Int3F16,
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
    fn test_int4_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_int4_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Int4F16,
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
    fn test_int5_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_int5_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Int5F16,
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
    fn test_int6_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_int6_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Int6F16,
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
    fn test_int8_f16_patch_embed(dt: DType) -> TestSetup {
        patch_setup(
            mt_int8_f16_patch_embed::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            16,
            16,
            8,
            8,
            64,
            dt,
        )
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
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble packs
        // + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let n_codes = hidden * patch_dim;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_codes, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n_codes, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
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
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                patch_bench($kernel(dt), $fmt, 3, 224, 224, 16, 16, 768, dt)
            }
        };
    }
    patch_bench_fmt!(bench_mxfp4, mt_mxfp4_patch_embed::kernel_ir_for, QFormat::Mxfp4);
    patch_bench_fmt!(bench_nvfp4, mt_nvfp4_patch_embed::kernel_ir_for, QFormat::Nvfp4);
    patch_bench_fmt!(bench_fp4, mt_fp4_patch_embed::kernel_ir_for, QFormat::Fp4);
    patch_bench_fmt!(bench_mxfp8_e4m3, mt_mxfp8_e4m3_patch_embed::kernel_ir_for, QFormat::Mxfp8E4);
    patch_bench_fmt!(bench_mxfp8_e5m2, mt_mxfp8_e5m2_patch_embed::kernel_ir_for, QFormat::Mxfp8E5);
    patch_bench_fmt!(bench_fp8_e5m2, mt_fp8_e5m2_patch_embed::kernel_ir_for, QFormat::Fp8E5m2);
    patch_bench_fmt!(bench_nvfp8, mt_nvfp8_patch_embed::kernel_ir_for, QFormat::Nvfp8);
    patch_bench_fmt!(bench_int8, mt_int8_patch_embed::kernel_ir_for, QFormat::Int8);
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). Same Grid3D geometry as the rest of the family.
    patch_bench_fmt!(bench_int2, mt_int2_patch_embed::kernel_ir_for, QFormat::Int2);
    patch_bench_fmt!(bench_int3, mt_int3_patch_embed::kernel_ir_for, QFormat::Int3);
    patch_bench_fmt!(bench_int4, mt_int4_patch_embed::kernel_ir_for, QFormat::Int4);
    patch_bench_fmt!(bench_int5, mt_int5_patch_embed::kernel_ir_for, QFormat::Int5);
    patch_bench_fmt!(bench_int6, mt_int6_patch_embed::kernel_ir_for, QFormat::Int6);
    patch_bench_fmt!(bench_mxint2, mt_mxint2_patch_embed::kernel_ir_for, QFormat::Mxint2);
    patch_bench_fmt!(bench_mxint3, mt_mxint3_patch_embed::kernel_ir_for, QFormat::Mxint3);
    patch_bench_fmt!(bench_mxint4, mt_mxint4_patch_embed::kernel_ir_for, QFormat::Mxint4);
    patch_bench_fmt!(bench_mxint5, mt_mxint5_patch_embed::kernel_ir_for, QFormat::Mxint5);
    patch_bench_fmt!(bench_mxint6, mt_mxint6_patch_embed::kernel_ir_for, QFormat::Mxint6);
    patch_bench_fmt!(bench_mxint8, mt_mxint8_patch_embed::kernel_ir_for, QFormat::Mxint8);
    // FP16-scale twins (nvfp8 / fp4 / fp8_e5m2 / int2..6 / int8 scales as f16).
    // Same Grid3D geometry as the rest of the family. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (8-bit E4M3 + f16 scale, block 32).
    patch_bench_fmt!(bench_nvfp8_f16, mt_nvfp8_f16_patch_embed::kernel_ir_for, QFormat::Nvfp8F16);
    patch_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_patch_embed::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    patch_bench_fmt!(bench_fp4_f16, mt_fp4_f16_patch_embed::kernel_ir_for, QFormat::Fp4F16);
    patch_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_patch_embed::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    patch_bench_fmt!(bench_int2_f16, mt_int2_f16_patch_embed::kernel_ir_for, QFormat::Int2F16);
    patch_bench_fmt!(bench_int3_f16, mt_int3_f16_patch_embed::kernel_ir_for, QFormat::Int3F16);
    patch_bench_fmt!(bench_int4_f16, mt_int4_f16_patch_embed::kernel_ir_for, QFormat::Int4F16);
    patch_bench_fmt!(bench_int5_f16, mt_int5_f16_patch_embed::kernel_ir_for, QFormat::Int5F16);
    patch_bench_fmt!(bench_int6_f16, mt_int6_f16_patch_embed::kernel_ir_for, QFormat::Int6F16);
    patch_bench_fmt!(bench_int8_f16, mt_int8_f16_patch_embed::kernel_ir_for, QFormat::Int8F16);
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **dequantizing gather** (quantized embedding tables) for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! For each output element `(token, d)`: gather row `indices[token]` of a
//! `[vocab, hidden]` block-scaled table, decode the element at column `d`
//! (E2M1 nibble / E4M3 / E5M2 byte) × its block scale, store to
//! `out[token, d]`. The block-scaled counterpart of `ffai/dequant_gather.rs`
//! (which handles int2–int8 affine). Pure dequant — no reduction.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Grid3D**, one thread per output element via
//!   `grid_1d(n_tokens·hidden, 256)`. `idx = program_id::<0>()`,
//!   `token = idx / hidden`, `d = idx − token·hidden`.
//! - `hidden` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[vocab, hidden/8]` u32 (4-bit) or `[vocab, hidden]` u8 (8-bit);
//!   scales `[vocab, hidden/block_size]` (u8 E8M0/E4M3 or f32 nvfp8);
//!   indices `[n_tokens]` u32; out `[n_tokens, hidden]`. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// mxfp4 dequantizing gather — E2M1 (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp4_dequant_gather<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let words_per_row = hidden / 8u32;
    let blocks_per_row = hidden / block_size;
    let packed = load(weight[token_id * words_per_row + d / 8u32]);
    let nib = (packed >> ((d % 8u32) * 4u32)) & 0xFu32;
    let val = e2m1_decode(nib);
    let sbits = load(scales[token_id * blocks_per_row + d / block_size]).cast::<f32>();
    let scale = exp2(sbits - 127.0f32);
    store(out[idx], (val * scale).cast::<T>());
}

/// nvfp4 dequantizing gather — E2M1 (block 16), E4M3 micro-scale × global.
#[kernel]
pub fn mt_nvfp4_dequant_gather<T>(
    weight: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let words_per_row = hidden / 8u32;
    let blocks_per_row = hidden / block_size;
    let packed = load(weight[token_id * words_per_row + d / 8u32]);
    let nib = (packed >> ((d % 8u32) * 4u32)) & 0xFu32;
    let val = e2m1_decode(nib);
    let scale = e4m3_decode(load(scales[token_id * blocks_per_row + d / block_size]).cast::<u32>())
        * global;
    store(out[idx], (val * scale).cast::<T>());
}

/// mxfp8 (E4M3) dequantizing gather — 8-bit (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp8_e4m3_dequant_gather<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let blocks_per_row = hidden / block_size;
    let elem = e4m3_decode(load(weight[token_id * hidden + d]).cast::<u32>());
    let sbits = load(scales[token_id * blocks_per_row + d / block_size]).cast::<f32>();
    let scale = exp2(sbits - 127.0f32);
    store(out[idx], (elem * scale).cast::<T>());
}

/// mxfp8 (E5M2) dequantizing gather — 8-bit (block 32), E8M0 pow-2 scale.
#[kernel]
pub fn mt_mxfp8_e5m2_dequant_gather<T>(
    weight: Tensor<u8>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let blocks_per_row = hidden / block_size;
    let elem = e5m2_decode(load(weight[token_id * hidden + d]).cast::<u32>());
    let sbits = load(scales[token_id * blocks_per_row + d / block_size]).cast::<f32>();
    let scale = exp2(sbits - 127.0f32);
    store(out[idx], (elem * scale).cast::<T>());
}

/// nvfp8 dequantizing gather — E4M3 (block 16), per-block FP32 scale.
#[kernel]
pub fn mt_nvfp8_dequant_gather<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let blocks_per_row = hidden / block_size;
    let elem = e4m3_decode(load(weight[token_id * hidden + d]).cast::<u32>());
    let scale = load(scales[token_id * blocks_per_row + d / block_size]);
    store(out[idx], (elem * scale).cast::<T>());
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 gathers ────────────────
// These share the block-scaled gather framework but store a raw per-group FP32
// scale (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit E4M3
// + f32 scale), so it reuses `mt_nvfp8_dequant_gather`; only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 dequantizing gather — E2M1 (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp4_dequant_gather<T>(
    weight: Tensor<u32>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let words_per_row = hidden / 8u32;
    let blocks_per_row = hidden / block_size;
    let packed = load(weight[token_id * words_per_row + d / 8u32]);
    let nib = (packed >> ((d % 8u32) * 4u32)) & 0xFu32;
    let val = e2m1_decode(nib);
    let scale = load(scales[token_id * blocks_per_row + d / block_size]);
    store(out[idx], (val * scale).cast::<T>());
}

/// Legacy fp8 (E5M2) dequantizing gather — 8-bit (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_dequant_gather<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let blocks_per_row = hidden / block_size;
    let elem = e5m2_decode(load(weight[token_id * hidden + d]).cast::<u32>());
    let scale = load(scales[token_id * blocks_per_row + d / block_size]);
    store(out[idx], (elem * scale).cast::<T>());
}

/// Symmetric int8 dequantizing gather — 8-bit codes (group 64), per-group FP32
/// scale (affine, scale-only). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_dequant_gather<T>(
    weight: Tensor<u8>,
    scales: Tensor<f32>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let blocks_per_row = hidden / block_size;
    let elem = int8_decode(load(weight[token_id * hidden + d]).cast::<u32>());
    let scale = load(scales[token_id * blocks_per_row + d / block_size]);
    store(out[idx], (elem * scale).cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{quant::format::QFormat, utils::pack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Deterministic `[vocab, hidden]` embedding table with mixed signs.
    fn table(vocab: usize, hidden: usize) -> Vec<f32> {
        (0..vocab * hidden)
            .map(|i| {
                let r = (i / hidden) as f32;
                let c = (i % hidden) as f32;
                let mag = (0.3 + r * 0.15) * (0.1 + (c % 13.0) * 0.2);
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect()
    }

    fn gather_setup(kernel: Kernel, fmt: QFormat, hidden: usize, dt: DType) -> TestSetup {
        let vocab = 8usize;
        let w = table(vocab, hidden);
        let p = crate::quant::format::pack(fmt, &w, vocab, hidden);
        let wdq = crate::quant::format::dequant(fmt, &p, vocab, hidden);
        // Non-monotonic gather that repeats row 4 — surfaces token→row bugs.
        let indices: Vec<u32> = vec![3, 0, 7, 1, 4, 4];
        let n_tokens = indices.len();
        let mut expected = vec![0.0f32; n_tokens * hidden];
        for (token, &tid) in indices.iter().enumerate() {
            for d in 0..hidden {
                expected[token * hidden + d] = wdq[tid as usize * hidden + d];
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
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", n_tokens * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_tokens * hidden, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_mxfp4_dequant_gather::kernel_ir_for(dt), QFormat::Mxfp4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_nvfp4_dequant_gather::kernel_ir_for(dt), QFormat::Nvfp4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_mxfp8_e4m3_dequant_gather::kernel_ir_for(dt), QFormat::Mxfp8E4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_mxfp8_e5m2_dequant_gather::kernel_ir_for(dt), QFormat::Mxfp8E5, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_nvfp8_dequant_gather::kernel_ir_for(dt), QFormat::Nvfp8, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // hidden=256 is 4×64, so the int8 group of 64 divides evenly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_fp4_dequant_gather::kernel_ir_for(dt), QFormat::Fp4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_nvfp8_dequant_gather::kernel_ir_for(dt), QFormat::Fp8E4m3, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_fp8_e5m2_dequant_gather::kernel_ir_for(dt), QFormat::Fp8E5m2, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_dequant_gather(dt: DType) -> TestSetup {
        gather_setup(mt_int8_dequant_gather::kernel_ir_for(dt), QFormat::Int8, 256, dt)
    }
}

/// Decode-shape benches: gather `n_tokens` rows of a `vocab × hidden`
/// block-scaled embedding table. Grid3D, one thread per output element.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn gb(kernel: Kernel, fmt: QFormat, hidden: usize, dt: DType) -> BenchSetup {
        let vocab = 4096usize;
        let n_tokens = 32usize;
        let blocks_per_row = hidden / fmt.block_size();
        let (codes_len, codes_dt) = if fmt.element_bits() == 4 {
            (vocab * hidden / 8, DType::U32)
        } else {
            (vocab * hidden, DType::U8)
        };
        let scales_dt = if matches!(
            fmt,
            QFormat::Nvfp8 | QFormat::Fp4 | QFormat::Fp8E4m3 | QFormat::Fp8E5m2 | QFormat::Int8
        ) {
            DType::F32
        } else {
            DType::U8
        };
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", vocab * blocks_per_row, scales_dt))
            .buffer(BenchBuffer::random("indices", n_tokens, DType::U32))
            .buffer(BenchBuffer::zeros("out", n_tokens * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_tokens * hidden, 256)
            .bytes_moved((n_tokens * hidden * dt.size_bytes()) as u64)
            .with_shape_label(format!("{} tok={n_tokens} h={hidden}", fmt.name()))
    }

    #[bench(name = "ffai/dequant_gather_block/mxfp4", dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_gather(dt: DType) -> BenchSetup {
        gb(mt_mxfp4_dequant_gather::kernel_ir_for(dt), QFormat::Mxfp4, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/nvfp4", dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_gather(dt: DType) -> BenchSetup {
        gb(mt_nvfp4_dequant_gather::kernel_ir_for(dt), QFormat::Nvfp4, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/mxfp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_gather(dt: DType) -> BenchSetup {
        gb(mt_mxfp8_e4m3_dequant_gather::kernel_ir_for(dt), QFormat::Mxfp8E4, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/mxfp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_gather(dt: DType) -> BenchSetup {
        gb(mt_mxfp8_e5m2_dequant_gather::kernel_ir_for(dt), QFormat::Mxfp8E5, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/nvfp8", dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_gather(dt: DType) -> BenchSetup {
        gb(mt_nvfp8_dequant_gather::kernel_ir_for(dt), QFormat::Nvfp8, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/fp4", dtypes = [f32, f16, bf16])]
    fn bench_fp4_gather(dt: DType) -> BenchSetup {
        gb(mt_fp4_dequant_gather::kernel_ir_for(dt), QFormat::Fp4, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_gather(dt: DType) -> BenchSetup {
        gb(mt_nvfp8_dequant_gather::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_gather(dt: DType) -> BenchSetup {
        gb(mt_fp8_e5m2_dequant_gather::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, dt)
    }
    #[bench(name = "ffai/dequant_gather_block/int8", dtypes = [f32, f16, bf16])]
    fn bench_int8_gather(dt: DType) -> BenchSetup {
        gb(mt_int8_dequant_gather::kernel_ir_for(dt), QFormat::Int8, 4096, dt)
    }
}

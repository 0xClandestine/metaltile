//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Standalone dequant kernels for the spec-conformant block-scaled formats
//! (nvfp4 / mxfp4 / mxfp8 / nvfp8 — see `specs/BENCH_METRICS_SPEC.md` Appendix B).
//!
//! Each kernel reads packed element codes + per-block scales and writes the
//! reconstructed `[rows, cols]` matrix. The decode math mirrors
//! [`crate::quant::codec`] exactly, so the GPU output is checked against the
//! host [`crate::quant::format::dequant`] oracle — same reference, no drift.
//!
//! ## DISPATCH INVARIANTS (all kernels here)
//!
//! - **Mode: Grid3D (elementwise), one thread per output element.** Pure
//!   per-element decode with no cross-thread cooperation — `program_id::<0>()`
//!   is the global thread index. Dispatch `grid = [ceil(n/256), 1, 1]`,
//!   `tpg = [256, 1, 1]`; the `if i < n` guard covers the tail. Being Grid3D
//!   (not Reduction) it is *not* exposed to the `n_simd == 0` freeze hazard.
//! - **`block_size`** is the format's K-block (16 or 32) and must divide `cols`.
//! - 4-bit codes pack 8 nibbles per `u32` (little-endian: element `i` → word
//!   `i/8`, nibble shift `(i & 7) * 4`). 8-bit codes are one `uchar` each.

use metaltile::kernel;

/// mxfp4 — E2M1 elements (block 32), E8M0 pow-2 block scale.
/// `scales[b]` is the biased exponent; effective scale `2^(bits - 127)`.
#[kernel]
pub fn mt_mxfp4_dequant<T>(
    codes: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let word = load(codes[i / 8u32]);
        let nib = (word >> ((i & 7u32) * 4u32)) & 0xFu32;
        let val = e2m1_decode(nib);
        let sbits = load(scales[i / block_size]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127), exact for integer bits
        store(out[i], (val * scale).cast::<T>());
    }
}

/// nvfp4 — E2M1 elements (block 16), E4M3 micro-scale × a global FP32.
/// `scales[b]` is an E4M3 code; effective block scale `e4m3(scales[b]) * global`.
#[kernel]
pub fn mt_nvfp4_dequant<T>(
    codes: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let i = program_id::<0>();
    if i < n {
        let word = load(codes[i / 8u32]);
        let nib = (word >> ((i & 7u32) * 4u32)) & 0xFu32;
        let elem = e2m1_decode(nib);
        // E4M3 micro-scale × global.
        let block_scale = e4m3_decode(load(scales[i / block_size]).cast::<u32>()) * global;
        store(out[i], (elem * block_scale).cast::<T>());
    }
}

/// mxfp8 (E4M3) — E4M3 elements (block 32), E8M0 pow-2 block scale.
#[kernel]
pub fn mt_mxfp8_e4m3_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = e4m3_decode(load(codes[i]).cast::<u32>());
        let sbits = load(scales[i / block_size]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// mxfp8 (E5M2) — E5M2 elements (block 32), E8M0 pow-2 block scale.
#[kernel]
pub fn mt_mxfp8_e5m2_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = e5m2_decode(load(codes[i]).cast::<u32>());
        let sbits = load(scales[i / block_size]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// nvfp8 — E4M3 elements (block 16), per-block FP32 scale (loaded directly).
#[kernel]
pub fn mt_nvfp8_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = e4m3_decode(load(codes[i]).cast::<u32>());
        let scale = load(scales[i / block_size]);
        store(out[i], (elem * scale).cast::<T>());
    }
}

// ── Legacy float-scale (fp4 / fp8) + symmetric int8 dequant ────────────────
// These share the per-element decode framework but store a raw per-group FP32
// scale (no E8M0/E4M3/global). fp8_e4m3 has the same shape as nvfp8 (8-bit
// E4M3 + f32 scale), so it reuses `mt_nvfp8_dequant` — only fp4 (4-bit E2M1),
// fp8_e5m2 (8-bit E5M2), and int8 (8-bit symmetric) need their own decode here.

/// Legacy fp4 — E2M1 elements (group 32), per-group FP32 scale (loaded directly).
#[kernel]
pub fn mt_fp4_dequant<T>(
    codes: Tensor<u32>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let word = load(codes[i / 8u32]);
        let nib = (word >> ((i & 7u32) * 4u32)) & 0xFu32;
        let elem = e2m1_decode(nib);
        let scale = load(scales[i / block_size]);
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// Legacy fp8 (E5M2) — E5M2 elements (group 32), per-group FP32 scale.
#[kernel]
pub fn mt_fp8_e5m2_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = e5m2_decode(load(codes[i]).cast::<u32>());
        let scale = load(scales[i / block_size]);
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// Symmetric int8 — 8-bit codes (group 64), per-group FP32 scale (scale-only
/// affine). Decode is sign-extend → `code · scale`.
#[kernel]
pub fn mt_int8_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = int8_decode(load(codes[i]).cast::<u32>());
        let scale = load(scales[i / block_size]);
        store(out[i], (elem * scale).cast::<T>());
    }
}

// ── Symmetric sub-byte integers (int2/3/4/5/6 + MXINT2..6) ──────────────────
// The element is a signed N-bit two's-complement code, tight-bit-packed LSB-first
// into u32 words (element i at bit `i·bits`). Decode = extract the low N bits
// (straddle-aware two-word read, mirroring `ffai/dequant_gemv.rs`), sign-extend
// in float (subtract 2^N when the top bit is set; `$half`/`$full` are 2^(N-1) /
// 2^N), then multiply by the block scale. A 4-bit stream is byte-identical to
// the nibble layout, so int4 rides the same path. `$half`/`$full` are passed as
// literals to keep the constexpr math out of the DSL shift operands.

/// FP32-scaled symmetric int (int2/3/4/5/6): bit-stream code × per-group FP32.
macro_rules! int_dequant_f32 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            codes: Tensor<u32>,
            scales: Tensor<f32>,
            out: Tensor<T>,
            #[constexpr] n: u32,
            #[constexpr] block_size: u32,
        ) {
            let i = program_id::<0>();
            if i < n {
                let bit_off = i * $bits;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                let spill = $bits - lo_bits;
                let w0 = load(codes[word_idx]);
                let w1 = load(codes[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                let q = lo | hi;
                let qf = q.cast::<f32>();
                let val = select(q >= $half, qf - $full, qf); // sign-extend
                let scale = load(scales[i / block_size]);
                store(out[i], (val * scale).cast::<T>());
            }
        }
    };
}
int_dequant_f32!(mt_int2_dequant, 2u32, 2u32, 4.0f32);
int_dequant_f32!(mt_int3_dequant, 3u32, 4u32, 8.0f32);
int_dequant_f32!(mt_int4_dequant, 4u32, 8u32, 16.0f32);
int_dequant_f32!(mt_int5_dequant, 5u32, 16u32, 32.0f32);
int_dequant_f32!(mt_int6_dequant, 6u32, 32u32, 64.0f32);

/// E8M0-scaled symmetric int (MXINT2/3/4/5/6): bit-stream code × pow-2 block scale.
macro_rules! int_dequant_e8m0 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            codes: Tensor<u32>,
            scales: Tensor<u8>,
            out: Tensor<T>,
            #[constexpr] n: u32,
            #[constexpr] block_size: u32,
        ) {
            let i = program_id::<0>();
            if i < n {
                let bit_off = i * $bits;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                let spill = $bits - lo_bits;
                let w0 = load(codes[word_idx]);
                let w1 = load(codes[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                let q = lo | hi;
                let qf = q.cast::<f32>();
                let val = select(q >= $half, qf - $full, qf); // sign-extend
                let sbits = load(scales[i / block_size]).cast::<f32>();
                let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127)
                store(out[i], (val * scale).cast::<T>());
            }
        }
    };
}
int_dequant_e8m0!(mt_mxint2_dequant, 2u32, 2u32, 4.0f32);
int_dequant_e8m0!(mt_mxint3_dequant, 3u32, 4u32, 8.0f32);
int_dequant_e8m0!(mt_mxint4_dequant, 4u32, 8u32, 16.0f32);
int_dequant_e8m0!(mt_mxint5_dequant, 5u32, 16u32, 32.0f32);
int_dequant_e8m0!(mt_mxint6_dequant, 6u32, 32u32, 64.0f32);

/// MXINT8 — 8-bit codes (byte layout, block 32), E8M0 pow-2 block scale.
#[kernel]
pub fn mt_mxint8_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = int8_decode(load(codes[i]).cast::<u32>());
        let sbits = load(scales[i / block_size]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32);
        store(out[i], (elem * scale).cast::<T>());
    }
}

// ── FP16-scale twins ─────────────────────────────────────────────────────────
// Identical element decode to their FP32-scaled twin; only the scale is read as a
// native `half` (`Tensor<f16>`) and cast to f32. The GPU half load matches the
// host `f16_scale_decode`, so the oracle still holds exactly.

/// nvfp8 (FP16 scale) — E4M3 elements (block 16), per-block FP16 scale. Also
/// serves `fp8_e4m3_f16` (same 8-bit-E4M3 + scale shape).
#[kernel]
pub fn mt_nvfp8_f16_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = e4m3_decode(load(codes[i]).cast::<u32>());
        let scale = load(scales[i / block_size]).cast::<f32>();
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// fp4 (FP16 scale) — E2M1 elements (group 32), per-group FP16 scale.
#[kernel]
pub fn mt_fp4_f16_dequant<T>(
    codes: Tensor<u32>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let word = load(codes[i / 8u32]);
        let nib = (word >> ((i & 7u32) * 4u32)) & 0xFu32;
        let elem = e2m1_decode(nib);
        let scale = load(scales[i / block_size]).cast::<f32>();
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// fp8 (E5M2, FP16 scale) — E5M2 elements (group 32), per-group FP16 scale.
#[kernel]
pub fn mt_fp8_e5m2_f16_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = e5m2_decode(load(codes[i]).cast::<u32>());
        let scale = load(scales[i / block_size]).cast::<f32>();
        store(out[i], (elem * scale).cast::<T>());
    }
}

/// Symmetric int (FP16 scale, sub-byte int2/3/4/5/6) — bit-stream code × FP16 scale.
macro_rules! int_dequant_f16 {
    ($name:ident, $bits:literal, $half:literal, $full:literal) => {
        #[kernel]
        pub fn $name<T>(
            codes: Tensor<u32>,
            scales: Tensor<f16>,
            out: Tensor<T>,
            #[constexpr] n: u32,
            #[constexpr] block_size: u32,
        ) {
            let i = program_id::<0>();
            if i < n {
                let bit_off = i * $bits;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                let spill = $bits - lo_bits;
                let w0 = load(codes[word_idx]);
                let w1 = load(codes[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                let q = lo | hi;
                let qf = q.cast::<f32>();
                let val = select(q >= $half, qf - $full, qf); // sign-extend
                let scale = load(scales[i / block_size]).cast::<f32>();
                store(out[i], (val * scale).cast::<T>());
            }
        }
    };
}
int_dequant_f16!(mt_int2_f16_dequant, 2u32, 2u32, 4.0f32);
int_dequant_f16!(mt_int3_f16_dequant, 3u32, 4u32, 8.0f32);
int_dequant_f16!(mt_int4_f16_dequant, 4u32, 8u32, 16.0f32);
int_dequant_f16!(mt_int5_f16_dequant, 5u32, 16u32, 32.0f32);
int_dequant_f16!(mt_int6_f16_dequant, 6u32, 32u32, 64.0f32);

/// int8 (FP16 scale) — 8-bit codes (byte layout, group 64), per-group FP16 scale.
#[kernel]
pub fn mt_int8_f16_dequant<T>(
    codes: Tensor<u8>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let elem = int8_decode(load(codes[i]).cast::<u32>());
        let scale = load(scales[i / block_size]).cast::<f32>();
        store(out[i], (elem * scale).cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{quant::format::QFormat, utils::pack_f32};

    /// Deterministic f32 weights with magnitude varying along K (so per-block
    /// scales differ) and mixed signs.
    fn weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f32;
                let c = (i % cols) as f32;
                let mag = (1.0 + r * 0.5) * (0.1 + (c % 11.0) * 0.25);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// Shared setup: pack `[rows, cols]` weights in `fmt`, dispatch the dequant
    /// kernel, and expect the host oracle's reconstruction. Kernel and oracle
    /// share `quant::codec`, so the match is near-exact.
    fn dequant_setup(
        kernel: Kernel,
        fmt: QFormat,
        rows: usize,
        cols: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(rows, cols);
        let p = crate::quant::format::pack(fmt, &w, rows, cols);
        let oracle = crate::quant::format::dequant(fmt, &p, rows, cols);
        let n = rows * cols;
        const TPG: u32 = 256;
        // Sub-byte codes (int2-6, E2M1) bind as bit-stream u32 words; 8-bit codes
        // as one uchar each. FP32→f32, FP16→f16 scales bind by value; E8M0/E4M3
        // scales as one byte.
        let codes_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .input(TestBuffer::from_vec("codes", p.codes, codes_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        // nvfp4 is two-level: the per-tensor global FP32 is a constexpr.
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&oracle, dt), dt)).grid_3d(
            (n as u32).div_ceil(TPG),
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // cols 64 is divisible by both block sizes (16 and 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_mxfp4_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxfp4_dequant::kernel_ir_for(dt), QFormat::Mxfp4, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_nvfp4_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_nvfp4_dequant::kernel_ir_for(dt), QFormat::Nvfp4, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxfp8_e4m3_dequant::kernel_ir_for(dt), QFormat::Mxfp8E4, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxfp8_e5m2_dequant::kernel_ir_for(dt), QFormat::Mxfp8E5, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_nvfp8_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_nvfp8_dequant::kernel_ir_for(dt), QFormat::Nvfp8, 4, 64, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here. cols
    // 64 is divisible by all group sizes here (16 / 32 / 64).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_fp4_dequant::kernel_ir_for(dt), QFormat::Fp4, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_nvfp8_dequant::kernel_ir_for(dt), QFormat::Fp8E4m3, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_fp8_e5m2_dequant::kernel_ir_for(dt), QFormat::Fp8E5m2, 4, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int8_dequant::kernel_ir_for(dt), QFormat::Int8, 4, 64, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale). The
    // kernel and oracle share the codec, so the GPU output matches the oracle to
    // float precision regardless of how coarse the quantization is. cols 64 is
    // divisible by both group sizes (int 64, mxint 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int2_dequant::kernel_ir_for(dt), QFormat::Int2, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int3_dequant::kernel_ir_for(dt), QFormat::Int3, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int4_dequant::kernel_ir_for(dt), QFormat::Int4, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int5_dequant::kernel_ir_for(dt), QFormat::Int5, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int6_dequant::kernel_ir_for(dt), QFormat::Int6, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxint2_dequant::kernel_ir_for(dt), QFormat::Mxint2, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxint3_dequant::kernel_ir_for(dt), QFormat::Mxint3, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxint4_dequant::kernel_ir_for(dt), QFormat::Mxint4, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxint5_dequant::kernel_ir_for(dt), QFormat::Mxint5, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxint6_dequant::kernel_ir_for(dt), QFormat::Mxint6, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxint8_dequant::kernel_ir_for(dt), QFormat::Mxint8, 4, 64, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_nvfp8_f16_dequant::kernel_ir_for(dt), QFormat::Nvfp8F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_nvfp8_f16_dequant::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_fp4_f16_dequant::kernel_ir_for(dt), QFormat::Fp4F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_fp8_e5m2_f16_dequant::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int2_f16_dequant::kernel_ir_for(dt), QFormat::Int2F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int3_f16_dequant::kernel_ir_for(dt), QFormat::Int3F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int4_f16_dequant::kernel_ir_for(dt), QFormat::Int4F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int5_f16_dequant::kernel_ir_for(dt), QFormat::Int5F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int6_f16_dequant::kernel_ir_for(dt), QFormat::Int6F16, 4, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_int8_f16_dequant::kernel_ir_for(dt), QFormat::Int8F16, 4, 64, dt)
    }
}

/// Bulk-dequant (memory-bound, one thread per output element) benches at a
/// canonical `[rows, cols]` weight tile so the latency + bandwidth columns rank
/// the precisions side by side (the spec's "which precision is fastest" goal).
/// Throughput is data-independent, so the packed code/scale buffers are random.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    fn dequant_bench(
        kernel: Kernel,
        fmt: QFormat,
        rows: usize,
        cols: usize,
        dt: DType,
    ) -> BenchSetup {
        let n = rows * cols;
        let n_blocks = rows * (cols / fmt.block_size());
        // 8-bit codes are one uchar each; sub-byte codes tight-bit-pack into u32
        // words (with a guard word for straddling 3/5/6-bit reads).
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + n * dt.size_bytes();
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("codes", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n, 256)
            .bytes_moved(bytes as u64)
            .with_shape_label(format!("{} m={rows} k={cols}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxfp4_dequant::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_nvfp4_dequant::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxfp8_e4m3_dequant::kernel_ir_for(dt), QFormat::Mxfp8E4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxfp8_e5m2_dequant::kernel_ir_for(dt), QFormat::Mxfp8E5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_nvfp8_dequant::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode in their own.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_fp4_dequant::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_nvfp8_dequant::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_fp8_e5m2_dequant::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int8_dequant::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int2_dequant::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int3_dequant::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int4_dequant::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int5_dequant::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int6_dequant::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxint2_dequant::kernel_ir_for(dt), QFormat::Mxint2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxint3_dequant::kernel_ir_for(dt), QFormat::Mxint3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxint4_dequant::kernel_ir_for(dt), QFormat::Mxint4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxint5_dequant::kernel_ir_for(dt), QFormat::Mxint5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxint6_dequant::kernel_ir_for(dt), QFormat::Mxint6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_mxint8_dequant::kernel_ir_for(dt), QFormat::Mxint8, 4096, 4096, dt)
    }
    // FP16-scale twins (fp8_e4m3_f16 reuses the nvfp8_f16 kernel).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_nvfp8_f16_dequant::kernel_ir_for(dt), QFormat::Nvfp8F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_nvfp8_f16_dequant::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_fp4_f16_dequant::kernel_ir_for(dt), QFormat::Fp4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(
            mt_fp8_e5m2_f16_dequant::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int2_f16_dequant::kernel_ir_for(dt), QFormat::Int2F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int3_f16_dequant::kernel_ir_for(dt), QFormat::Int3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int4_f16_dequant::kernel_ir_for(dt), QFormat::Int4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int5_f16_dequant::kernel_ir_for(dt), QFormat::Int5F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int6_f16_dequant::kernel_ir_for(dt), QFormat::Int6F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_dequant(dt: DType) -> BenchSetup {
        dequant_bench(mt_int8_f16_dequant::kernel_ir_for(dt), QFormat::Int8F16, 4096, 4096, dt)
    }
}

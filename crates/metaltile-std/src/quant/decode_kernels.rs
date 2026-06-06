//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled decode primitives as `#[kernel]` callees.
//!
//! These are the kernel-side mirrors of `quant::codec`. Each function is a
//! `#[kernel]` primitive that takes a raw integer code and produces a decoded
//! `f32` value. Callers use cross-kernel call syntax inside `#[kernel]` bodies:
//!
//! ```ignore
//! let val = mt_decode_e2m1(nib);
//! ```
//!
//! `KernelInlinePass` splices the body inline so there is no memory round-trip.
//! The bit-level encodings match `quant::codec` exactly.

use metaltile::kernel;

/// Decode a 4-bit E2M1 (fp4) code → f32.
///
/// Codebook magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, sign in bit 3.
/// Operand `inp` is the 4-bit code in the low nibble of a `u32`.
#[kernel]
pub fn mt_decode_e2m1(inp: Tensor<u32>, out: Tensor<f32>) {
    let code = load(inp[0u32]);
    let m = code & 7u32;
    let mag = select(
        m < 1u32,
        0.0f32,
        select(
            m < 2u32,
            0.5f32,
            select(
                m < 3u32,
                1.0f32,
                select(
                    m < 4u32,
                    1.5f32,
                    select(
                        m < 5u32,
                        2.0f32,
                        select(m < 6u32, 3.0f32, select(m < 7u32, 4.0f32, 6.0f32)),
                    ),
                ),
            ),
        ),
    );
    store(out[0u32], select((code & 8u32) != 0u32, -mag, mag));
}

/// Decode an 8-bit E4M3 (fp8) code → f32.
///
/// Format: 1 sign · 4 exp (bias 7) · 3 mantissa. Max ±448, no infinity.
/// Mirrors `quant::codec::e4m3_decode`.
#[kernel]
pub fn mt_decode_e4m3(inp: Tensor<u32>, out: Tensor<f32>) {
    let c = load(inp[0u32]);
    let e = (c >> 3u32) & 15u32;
    let m = c & 7u32;
    let sub = m.cast::<f32>() * 0.001953125f32;
    let norm = (1.0f32 + m.cast::<f32>() * 0.125f32) * exp2((e.cast::<f32>()) - 7.0f32);
    let mag = select(e < 1u32, sub, norm);
    store(out[0u32], select((c >> 7u32) != 0u32, -mag, mag));
}

/// Decode an 8-bit E5M2 (fp8) code → f32.
///
/// Format: 1 sign · 5 exp (bias 15) · 2 mantissa. Mirrors IEEE half high byte.
/// Mirrors `quant::codec::e5m2_decode`.
#[kernel]
pub fn mt_decode_e5m2(inp: Tensor<u32>, out: Tensor<f32>) {
    let c = load(inp[0u32]);
    let e = (c >> 2u32) & 31u32;
    let m = c & 3u32;
    let sub = m.cast::<f32>() * 0.0000152587890625f32;
    let norm = (1.0f32 + m.cast::<f32>() * 0.25f32) * exp2((e.cast::<f32>()) - 15.0f32);
    let mag = select(e < 1u32, sub, norm);
    store(out[0u32], select((c >> 7u32) != 0u32, -mag, mag));
}

/// Decode a symmetric int8 code → f32.
///
/// Reinterprets the low 8 bits as a signed `i8` (sign-extends bit 7).
/// Mirrors `quant::codec::int8_decode`.
#[kernel]
pub fn mt_decode_int8(inp: Tensor<u32>, out: Tensor<f32>) {
    let bits = load(inp[0u32]);
    // Cast to i32 before the right-shift so MSL uses arithmetic (sign-extending)
    // shift, matching `u8 as i8 as f32` in quant::codec::int8_decode.
    store(out[0u32], ((bits.cast::<i32>() << 24i32) >> 24i32).cast::<f32>());
}

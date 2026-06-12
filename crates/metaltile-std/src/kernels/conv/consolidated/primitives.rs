//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Decode primitives for block-scaled quantised-weight convolutions.
//!
//! Each function is a `#[kernel]` primitive that takes a raw integer code and
//! writes the decoded f32 value to an output slot. Callers use the cross-kernel
//! call syntax (`let v = mt_decode_e2m1(nib)`); `KernelInlinePass` splices the body
//! inline so there is no memory round-trip and no extra preamble helper needed.
//!
//! The bit-level encodings match `crates/metaltile-std/src/quant/codec.rs`
//! exactly — `codec.rs` is the Rust-side source of truth; these are the
//! kernel-side mirrors.

use metaltile::kernel;

/// TODO: This should be moved to a shared quant primitive, keeping here for my convenience while drafting.

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
    let norm = (1.0f32 + m.cast::<f32>() * 0.125f32) * exp2(e.cast::<f32>() - 7.0f32);
    let mag = select(e < 1u32, sub, norm);
    store(out[0u32], select(((c >> 7u32) & 1u32) != 0u32, -mag, mag));
}

/// Decode an 8-bit E5M2 (fp8) code → f32.
///
/// Format: 1 sign · 5 exp (bias 15) · 2 mantissa — the high byte of an IEEE
/// half. Mirrors `quant::codec::e5m2_decode`.
#[kernel]
pub fn mt_decode_e5m2(inp: Tensor<u32>, out: Tensor<f32>) {
    let c = load(inp[0u32]);
    let e = (c >> 2u32) & 31u32;
    let m = c & 3u32;
    let sub = m.cast::<f32>() * 0.0000152587890625f32;
    let norm = (1.0f32 + m.cast::<f32>() * 0.25f32) * exp2(e.cast::<f32>() - 15.0f32);
    let mag = select(e < 1u32, sub, norm);
    store(out[0u32], select(((c >> 7u32) & 1u32) != 0u32, -mag, mag));
}

/// Decode a symmetric-int8 code → f32.
///
/// Reinterprets the low 8 bits of `inp` as a signed byte (two's complement
/// sign-extension via `<< 24 / >> 24`). Mirrors `quant::codec::int8_decode`.
#[kernel]
pub fn mt_decode_int8(inp: Tensor<u32>, out: Tensor<f32>) {
    let c = load(inp[0u32]);
    let extended = ((c << 24u32).cast::<i32>() >> 24i32).cast::<f32>();
    store(out[0u32], extended);
}

/// Decode an 8-bit E8M0 exponent byte → f32 power-of-two scale.
///
/// Maps raw byte `b` (passed as u32 low-byte) to `2^(b − 127)`.
/// This is the MX-spec block-scale encoding: values 0..255 map to
/// `[2^−127, 2^128]`. Used by mxfp4, mxfp8, mxint2–8 formats.
#[kernel]
pub fn mt_decode_e8m0(inp: Tensor<u32>, out: Tensor<f32>) {
    let raw = load(inp[0u32]);
    store(out[0u32], exp2(raw.cast::<f32>() - 127.0f32));
}

/// Straddle-aware extraction of an unsigned N-bit code from a two-word buffer.
///
/// Extracts BITS-wide code at `bit_in_w` within `w0`, continuing into `w1`
/// when the code straddles a 32-bit word boundary (`spill > 0`).
///
///   lo   = bits [0, lo_bits) from `w0 >> bit_in_w`
///   hi   = bits [lo_bits, lo_bits+spill) from low bits of `w1`
///   result = lo | (hi << lo_bits)
///
/// Pre-conditions (caller must satisfy):
///   `bit_in_w = (col * BITS) & 31`
///   `lo_bits  = min(32 - bit_in_w, BITS)`
///   `spill    = BITS - lo_bits`
///   `w1` should equal `w0` when `spill == 0` (safe clamped load).
#[kernel]
pub fn mt_unpack_nbit(
    w0_inp: Tensor<u32>,
    w1_inp: Tensor<u32>,
    bit_in_w_inp: Tensor<u32>,
    lo_bits_inp: Tensor<u32>,
    spill_inp: Tensor<u32>,
    out: Tensor<u32>,
) {
    let w0 = load(w0_inp[0u32]);
    let w1 = load(w1_inp[0u32]);
    let biw = load(bit_in_w_inp[0u32]);
    let lb = load(lo_bits_inp[0u32]);
    let sp = load(spill_inp[0u32]);
    let lo = (w0 >> biw) & ((1u32 << lb) - 1u32);
    let hi = (w1 & ((1u32 << sp) - 1u32)) << lb;
    store(out[0u32], lo | hi);
}

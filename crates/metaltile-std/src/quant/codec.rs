//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Pure-Rust element + scale codecs for the block-scaled quant formats
//! (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! This is the **single source of truth** for the bit-level encoding: the
//! host-side packer ([`super::format`]) packs weights through these functions,
//! the CPU correctness oracle decodes through them, and the generated Metal
//! kernels emit the *same* math — so a kernel that disagrees with the oracle is
//! a real bug, not an oracle mismatch.
//!
//! Encodings mirror the pinned MLX `ekryski/mlx@alpha` headers
//! (`fp4.h` = E2M1, `fp8.h` = E4M3 + E8M0) where they exist; E5M2 (absent from
//! MLX) follows the OCP spec — conveniently the high byte of an IEEE half.
//!
//! - **E2M1** (fp4): 1 sign · 2 exp · 1 mant. Codebook magnitudes
//!   `{0, .5, 1, 1.5, 2, 3, 4, 6}`, sign in bit 3.
//! - **E4M3** (fp8): 1 sign · 4 exp (bias 7) · 3 mant. Max ±448, no inf.
//! - **E5M2** (fp8): 1 sign · 5 exp (bias 15) · 2 mant — IEEE-half's exponent
//!   field, i.e. the top 8 bits of an f16. Has inf/nan.
//! - **E8M0** (block scale): 8-bit biased power-of-two exponent. Value
//!   `2^(bits − 127)`; non-negative only.

// ── IEEE half (f16) bit helpers ────────────────────────────────────────────
// Shared by the E4M3 / E5M2 codecs (which route through the half encoding) and
// kept here so the module is self-contained.

/// Round an `f32` to IEEE half-precision bits (round-to-nearest-even).
fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 31) as u16) << 15;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant32 = x & 0x7F_FFFF;
    if exp <= 0 {
        return sign; // underflow → ±0 (subnormal flush; unused by our callers)
    }
    if exp >= 31 {
        return sign | 0x7C00; // overflow → ±inf
    }
    let mant16 = mant32 >> 13;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xFFF;
    let round_up = round_bit == 1 && (sticky != 0 || (mant16 & 1) == 1);
    let mant16 = (mant16 + u32::from(round_up)) as u16;
    if mant16 > 0x3FF {
        sign | (((exp + 1) as u16) << 10)
    } else {
        sign | ((exp as u16) << 10) | mant16
    }
}

/// Decode IEEE half-precision bits to `f32`.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp5 = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    if exp5 == 0 {
        // ±0 / subnormal: value = mantissa · 2^-24 (sign applied via bit 31).
        return f32::from_bits(sign)
            + (mantissa as f32) * 2.0f32.powi(-24) * if sign != 0 { -1.0 } else { 1.0 };
    }
    if exp5 == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)); // inf/nan
    }
    f32::from_bits(sign | ((exp5 + 112) << 23) | (mantissa << 13))
}

// ── E2M1 (fp4) ──────────────────────────────────────────────────────────────

/// The 8 E2M1 magnitudes indexed by the low 3 bits of the code.
pub const E2M1_CODEBOOK: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode a 4-bit E2M1 code (sign in bit 3, magnitude index in bits 0–2).
pub fn e2m1_decode(code: u8) -> f32 {
    let mag = E2M1_CODEBOOK[(code & 0x7) as usize];
    if code & 0x8 != 0 { -mag } else { mag }
}

/// Encode an `f32` to a 4-bit E2M1 code, rounding to the nearest codebook value
/// (midpoint thresholds, mirroring MLX `fp4.h`). NaN → `0x7` magnitude.
pub fn e2m1_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7;
    }
    let sign = if x.is_sign_negative() { 0x8 } else { 0x0 };
    let a = x.abs();
    // Midpoints between adjacent codebook magnitudes.
    let mag = if a > 5.0 {
        0x7 // 6.0
    } else if a >= 3.5 {
        0x6 // 4.0
    } else if a > 2.5 {
        0x5 // 3.0
    } else if a >= 1.75 {
        0x4 // 2.0
    } else if a > 1.25 {
        0x3 // 1.5
    } else if a >= 0.75 {
        0x2 // 1.0
    } else if a > 0.25 {
        0x1 // 0.5
    } else {
        0x0 // 0.0
    };
    mag | sign
}

// ── E4M3 (fp8) ────────────────────────────────────────────────────────────
// Port of MLX `fp8.h` (itself the PyTorch `Float8_e4m3fn` algorithm).

/// Decode an 8-bit E4M3 code to `f32`.
pub fn e4m3_decode(bits: u8) -> f32 {
    // 7 magnitude bits placed into a half pattern, then ×2^8 for the bias gap
    // (E4M3 bias 7 vs half bias 15). Sign in bit 7.
    let v: u16 = ((bits & 0x7f) as u16) << 7;
    let mag = f16_bits_to_f32(v) * 256.0;
    if bits & 0x80 != 0 { -mag } else { mag }
}

/// Encode an `f32` to an 8-bit E4M3 code with round-to-nearest-even and
/// saturation to ±448 (no infinities in E4M3).
pub fn e4m3_encode(f: f32) -> u8 {
    const FP8_MAX: u32 = 543 << 21;
    const DENORM_MASK: u32 = 141 << 23;
    let f_bits = f.to_bits();
    let sign = f_bits & 0x8000_0000;
    let mut f_bits = f_bits ^ sign;
    let bits = if f_bits >= FP8_MAX {
        // NaN or out-of-range → saturate to max magnitude.
        0x7e
    } else if f_bits < (121 << 23) {
        // Subnormal range: add the denorm bias then subtract it back out.
        f_bits = (f32::from_bits(f_bits) + f32::from_bits(DENORM_MASK)).to_bits();
        (f_bits.wrapping_sub(DENORM_MASK)) as u8
    } else {
        let mant_odd = (f_bits >> 20) & 1;
        f_bits = f_bits.wrapping_add(((7u32.wrapping_sub(127)) << 23).wrapping_add(0x7_FFFF));
        f_bits = f_bits.wrapping_add(mant_odd);
        (f_bits >> 20) as u8
    };
    bits | (sign >> 24) as u8
}

// ── E5M2 (fp8) ────────────────────────────────────────────────────────────
// E5M2 shares the half exponent field (5 bits, bias 15) with 2 mantissa bits —
// i.e. the high byte of an IEEE half.

/// Decode an 8-bit E5M2 code to `f32`.
pub fn e5m2_decode(bits: u8) -> f32 { f16_bits_to_f32((bits as u16) << 8) }

/// Largest finite E5M2 magnitude code (exp 30, mantissa 3 → 57344).
const E5M2_MAX_FINITE: u8 = 0x7b;

/// Encode an `f32` to an 8-bit E5M2 code: round to half, then round the half's
/// 10-bit mantissa to 2 bits (round-to-nearest-even) and take the high byte.
/// A **finite** input never becomes ±inf — it saturates to ±57344 (so a block
/// scale that rounds down can't turn a weight into inf and poison a matmul).
pub fn e5m2_encode(f: f32) -> u8 {
    let h = f32_to_f16_bits(f);
    let byte = if (h & 0x7c00) == 0x7c00 {
        // inf/nan in the half already: keep the high byte (mantissa in top bits).
        (h >> 8) as u8
    } else {
        // Round the 10-bit half mantissa down to 2 bits, nearest-even on bit 7.
        let round_bit = (h >> 7) & 1;
        let sticky = h & 0x7f;
        let lsb = (h >> 8) & 1;
        let round_up = round_bit == 1 && (sticky != 0 || lsb == 1);
        (h.wrapping_add(if round_up { 1 << 8 } else { 0 }) >> 8) as u8
    };
    // Saturate a finite input that landed in the inf/nan range (exp all ones).
    if f.is_finite() && (byte & 0x7f) >= 0x7c {
        return (byte & 0x80) | E5M2_MAX_FINITE;
    }
    byte
}

// ── E8M0 (block scale) ──────────────────────────────────────────────────────

/// Decode an 8-bit E8M0 scale to `f32`: `2^(bits − 127)`. `0xFF` is NaN
/// (returned as NaN); otherwise a pure power of two.
pub fn e8m0_decode(bits: u8) -> f32 {
    if bits == 0xff {
        return f32::NAN;
    }
    // Construct 2^(bits-127) directly in the f32 exponent field.
    f32::from_bits((bits as u32) << 23)
}

/// Encode a non-negative `f32` to an 8-bit E8M0 scale: round `log2(x)` to the
/// nearest integer, clamp to `[-127, 127]`, store biased by 127. Non-finite or
/// non-positive inputs map to `0xFF` / `0x00` respectively (mirrors MLX).
pub fn e8m0_encode(x: f32) -> u8 {
    if !x.is_finite() {
        return 0xff;
    }
    if x <= 0.0 {
        return 0x00;
    }
    let n = x.log2().round() as i32;
    let n = n.clamp(-127, 127);
    (n + 127) as u8
}

// ── int8 (symmetric affine element) ────────────────────────────────────────
// Unlike the fp formats, int8's "element" is the integer itself; the per-group
// FP32 scale is applied by the caller. Symmetric: codes in [-127, 127] (−128 is
// unused so |min| == |max|).

/// Decode a symmetric-int8 code (a `u8` reinterpreted as `i8`) to `f32`.
pub fn int8_decode(bits: u8) -> f32 { (bits as i8) as f32 }

/// Encode a scaled value (already divided by the group scale) to symmetric int8:
/// round-to-nearest, clamp to ±127, store as `u8`.
pub fn int8_encode(scaled: f32) -> u8 {
    let q = scaled.round().clamp(-127.0, 127.0) as i32;
    (q as i8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_round_trips_and_saturates() {
        for v in [-127i32, -64, -1, 0, 1, 63, 127] {
            assert_eq!(int8_decode(int8_encode(v as f32)), v as f32);
        }
        // Saturates to ±127 (−128 unused for symmetry).
        assert_eq!(int8_decode(int8_encode(200.0)), 127.0);
        assert_eq!(int8_decode(int8_encode(-200.0)), -127.0);
    }

    #[test]
    fn e2m1_codebook_round_trips_exactly() {
        // Every codebook value encodes to its own code and back.
        for (code, &mag) in E2M1_CODEBOOK.iter().enumerate() {
            assert_eq!(e2m1_decode(code as u8), mag, "code {code}");
            // Positive magnitude re-encodes to the same code.
            assert_eq!(e2m1_encode(mag) & 0x7, code as u8, "encode {mag}");
        }
        // Sign bit.
        assert_eq!(e2m1_decode(0x8 | 0x6), -4.0);
        assert_eq!(e2m1_encode(-3.0), 0x8 | 0x5);
    }

    #[test]
    fn e2m1_rounds_to_nearest_codebook_value() {
        assert_eq!(e2m1_decode(e2m1_encode(0.2)), 0.0); // < 0.25 → 0
        assert_eq!(e2m1_decode(e2m1_encode(0.6)), 0.5); // 0.25..0.75 → 0.5
        assert_eq!(e2m1_decode(e2m1_encode(2.4)), 2.0); // 1.75..2.5 → 2.0
        assert_eq!(e2m1_decode(e2m1_encode(100.0)), 6.0); // saturates to max mag
    }

    #[test]
    fn e8m0_is_pure_power_of_two() {
        for exp in -10i32..=10 {
            let x = 2.0f32.powi(exp);
            let code = e8m0_encode(x);
            assert_eq!(code, (exp + 127) as u8, "exp {exp}");
            assert_eq!(e8m0_decode(code), x, "decode exp {exp}");
        }
        // Rounds log2 to nearest: 1.4 ≈ 2^0.485 → 2^0 = 1.0.
        assert_eq!(e8m0_decode(e8m0_encode(1.4)), 1.0);
        // 1.6 ≈ 2^0.678 → 2^1 = 2.0.
        assert_eq!(e8m0_decode(e8m0_encode(1.6)), 2.0);
        assert_eq!(e8m0_encode(0.0), 0x00);
        assert!(e8m0_decode(0xff).is_nan());
    }

    #[test]
    fn e4m3_round_trips_representable_values_and_saturates() {
        // Exact powers of two and simple fractions are representable.
        for &v in &[0.0f32, 1.0, 2.0, 0.5, -1.0, 1.5, 8.0, 0.0625] {
            let r = e4m3_decode(e4m3_encode(v));
            assert!((r - v).abs() <= v.abs() * 0.07 + 1e-6, "e4m3 {v} → {r}");
        }
        // Saturates to ±448 (the E4M3 max), never inf.
        assert_eq!(e4m3_decode(e4m3_encode(1e6)), 448.0);
        assert_eq!(e4m3_decode(e4m3_encode(-1e6)), -448.0);
        assert!(e4m3_decode(e4m3_encode(448.0)).is_finite());
    }

    #[test]
    fn e5m2_round_trips_and_keeps_wide_range() {
        for &v in &[0.0f32, 1.0, 2.0, 0.5, -1.0, 1.5, 256.0, -0.25] {
            let r = e5m2_decode(e5m2_encode(v));
            // 2 mantissa bits ⇒ ~12.5% worst-case step.
            assert!((r - v).abs() <= v.abs() * 0.13 + 1e-6, "e5m2 {v} → {r}");
        }
        // E5M2's 5-bit exponent reaches the f16 max (57344) — far past E4M3.
        let big = e5m2_decode(e5m2_encode(40000.0));
        assert!(big > 30000.0, "e5m2 wide range: {big}");
    }

    #[test]
    fn e4m3_e5m2_zero_and_sign() {
        assert_eq!(e4m3_decode(e4m3_encode(0.0)), 0.0);
        assert_eq!(e5m2_decode(e5m2_encode(0.0)), 0.0);
        assert!(e4m3_decode(e4m3_encode(-2.0)) < 0.0);
        assert!(e5m2_decode(e5m2_encode(-2.0)) < 0.0);
    }
}

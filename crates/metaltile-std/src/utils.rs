//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Byte-packing helpers for authoring `#[bench]` / `#[test_kernel]` setups.
//!
//! These convert between `f32` values and the little-endian byte layout the
//! GPU expects for a given [`DType`], so a kernel author can write a CPU
//! oracle in `f32` and hand the runner the dtype-correct bytes:
//!
//! ```ignore
//! use metaltile::test::*;
//! use crate::utils::{pack_f32, scalar_bytes};
//!
//! let expected: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
//! TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)
//! ```

use half::{bf16, f16};
use metaltile_core::dtype::DType;

/// Pack a slice of `f32` values into little-endian bytes for `dt`.
///
/// `F32` is a straight memcpy; `F16`/`BF16` round each value to the target
/// precision (matching the load-cast the kernel performs on the GPU); integer
/// dtypes **value-cast** (`v as u32` etc., not a bit reinterpret), so a kernel
/// emitting integer output (e.g. argmax indices) round-trips through `f32`.
pub fn pack_f32(vals: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::F16 => vals.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect(),
        DType::BF16 => vals.iter().flat_map(|&v| bf16::from_f32(v).to_le_bytes()).collect(),
        DType::U32 => vals.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect(),
        DType::I32 => vals.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect(),
        DType::U8 => vals.iter().map(|&v| v as u8).collect(),
        DType::I8 => vals.iter().map(|&v| v as i8 as u8).collect(),
        _ => vals.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    }
}

/// Pack a single `f32` scalar into little-endian bytes for `dt`.
///
/// Convenience for the scalar `constant T&` inputs (e.g. arange's `start`/`step`).
pub fn scalar_bytes(v: f32, dt: DType) -> Vec<u8> { pack_f32(&[v], dt) }

/// Unpack little-endian `dt` bytes back into `f32` values.
///
/// Inverse of [`pack_f32`]; used by the test runner to read GPU output and
/// the expected buffer into a common `f32` representation for comparison.
pub fn unpack_f32(bytes: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F16 =>
            bytes.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        DType::BF16 =>
            bytes.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::U8 => bytes.iter().map(|&b| b as f32).collect(),
        DType::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
        _ => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trips_exactly() {
        let vals = [0.0, 1.5, -2.25, 1024.0];
        let bytes = pack_f32(&vals, DType::F32);
        assert_eq!(bytes.len(), vals.len() * 4);
        assert_eq!(unpack_f32(&bytes, DType::F32), vals);
    }

    #[test]
    fn f16_rounds_and_round_trips() {
        // 0.5 is exactly representable in f16, so the round-trip is exact.
        let vals = [0.0, 0.5, -0.5, 2.0];
        let bytes = pack_f32(&vals, DType::F16);
        assert_eq!(bytes.len(), vals.len() * 2);
        assert_eq!(unpack_f32(&bytes, DType::F16), vals);
    }

    #[test]
    fn scalar_bytes_matches_single_element_pack() {
        assert_eq!(scalar_bytes(3.5, DType::BF16), pack_f32(&[3.5], DType::BF16));
        assert_eq!(scalar_bytes(3.5, DType::F32), 3.5f32.to_le_bytes().to_vec());
    }

    #[test]
    fn bf16_rounds_and_round_trips_representable_values() {
        // 1.0/2.0/-0.5 are exactly representable in bf16, so they round-trip.
        let vals = [1.0, 2.0, -0.5, 0.0];
        let bytes = pack_f32(&vals, DType::BF16);
        assert_eq!(bytes.len(), vals.len() * 2);
        assert_eq!(unpack_f32(&bytes, DType::BF16), vals);
    }

    #[test]
    fn f16_rounds_lossy_values_to_nearest() {
        // 0.1 is not representable in f16; pack→unpack should round, not equal.
        let round = unpack_f32(&pack_f32(&[0.1], DType::F16), DType::F16)[0];
        assert!((round - 0.1).abs() < 1e-3 && round != 0.1);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            assert!(pack_f32(&[], dt).is_empty());
            assert!(unpack_f32(&[], dt).is_empty());
        }
    }

    #[test]
    fn unpack_drops_trailing_partial_element() {
        // 5 bytes is one full f32 (4) plus a stray byte; chunks_exact drops it.
        assert_eq!(unpack_f32(&[0, 0, 128, 63, 7], DType::F32), vec![1.0]);
        // 3 bytes is one full f16 (2) plus a stray byte.
        assert_eq!(unpack_f32(&pack_f32(&[2.0], DType::F16), DType::F16), vec![2.0]);
    }

    #[test]
    fn integer_dtypes_value_cast_round_trip() {
        let vals = [0.0, 1.0, 42.0, 255.0];
        for dt in [DType::U32, DType::I32] {
            assert_eq!(unpack_f32(&pack_f32(&vals, dt), dt), vals);
        }
        // Value cast, not a bit reinterpret: 1.0 → the integer 1.
        assert_eq!(pack_f32(&[1.0], DType::U32), 1u32.to_le_bytes().to_vec());
        // 8-bit dtypes narrow to one byte per element.
        assert_eq!(pack_f32(&[5.0, 250.0], DType::U8), vec![5, 250]);
        assert_eq!(unpack_f32(&[5, 250], DType::U8), vec![5.0, 250.0]);
    }

    #[test]
    fn scalar_bytes_width_matches_dtype() {
        assert_eq!(scalar_bytes(1.0, DType::F32).len(), 4);
        assert_eq!(scalar_bytes(1.0, DType::F16).len(), 2);
        assert_eq!(scalar_bytes(1.0, DType::BF16).len(), 2);
    }

    #[test]
    fn nan_and_inf_round_trip_through_float_dtypes() {
        // NaN != NaN, so check the predicate; ±inf survive f16/bf16 (both have
        // an inf encoding). Oracles that produce inf (e.g. a masked-out softmax
        // max of -inf) must round-trip faithfully.
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let out = unpack_f32(&pack_f32(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], dt), dt);
            assert!(out[0].is_nan(), "{dt:?}: NaN must round-trip");
            assert_eq!(out[1], f32::INFINITY, "{dt:?}: +inf must round-trip");
            assert_eq!(out[2], f32::NEG_INFINITY, "{dt:?}: -inf must round-trip");
        }
    }

    #[test]
    fn negative_zero_sign_is_preserved() {
        // -0.0 == 0.0 numerically, so compare the sign bit explicitly. A lost
        // sign would silently flip the result of e.g. copysign-based ops.
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let out = unpack_f32(&pack_f32(&[-0.0f32], dt), dt)[0];
            assert!(out.is_sign_negative() && out == 0.0, "{dt:?}: -0.0 sign must survive");
        }
    }

    #[test]
    fn f16_overflow_saturates_to_inf() {
        // 70000 exceeds f16's max finite (65504) → rounds to +inf, not a wrap.
        assert_eq!(unpack_f32(&pack_f32(&[70_000.0], DType::F16), DType::F16)[0], f32::INFINITY);
    }

    #[test]
    fn i8_round_trips_signed_range() {
        // I8 covers [-128, 127]; the sign must survive the u8 byte storage.
        let vals = [-128.0, -1.0, 0.0, 1.0, 127.0];
        assert_eq!(unpack_f32(&pack_f32(&vals, DType::I8), DType::I8), vals);
        // -1.0 stores as 0xFF (two's complement), not 0x01.
        assert_eq!(pack_f32(&[-1.0], DType::I8), vec![0xFF]);
    }
}

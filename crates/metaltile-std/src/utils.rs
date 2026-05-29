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
/// precision (matching the load-cast the kernel performs on the GPU). Any
/// other dtype falls back to the raw `f32` layout.
pub fn pack_f32(vals: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::F16 => vals.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect(),
        DType::BF16 => vals.iter().flat_map(|&v| bf16::from_f32(v).to_le_bytes()).collect(),
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
    fn non_float_dtype_falls_back_to_f32_layout() {
        let vals = [1.0, 2.0];
        assert_eq!(pack_f32(&vals, DType::I32), pack_f32(&vals, DType::F32));
        let bytes = pack_f32(&vals, DType::I32);
        assert_eq!(unpack_f32(&bytes, DType::U32), vals);
    }

    #[test]
    fn scalar_bytes_width_matches_dtype() {
        assert_eq!(scalar_bytes(1.0, DType::F32).len(), 4);
        assert_eq!(scalar_bytes(1.0, DType::F16).len(), 2);
        assert_eq!(scalar_bytes(1.0, DType::BF16).len(), 2);
    }
}

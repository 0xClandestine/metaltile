//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Shared helpers for `kernel_tests` and `kernel_benches` modules.

use metaltile::core::DType;

/// Pack `f32` values into the byte representation of `dt`.
pub fn pack_f32(vals: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
        DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
        DType::BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
        _ => panic!("unsupported dtype {dt:?}"),
    }
}

/// Pack a single `f32` scalar into the byte representation of `dt`.
///
/// Convenience wrapper around [`pack_f32`] for 1-element constant buffers
/// (e.g. the `start` and `step` scalars in `mt_arange`).
pub fn scalar_bytes(v: f32, dt: DType) -> Vec<u8> { pack_f32(&[v], dt) }

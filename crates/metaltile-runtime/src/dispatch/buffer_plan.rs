//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Buffer planning for Metal bindings.
//!
//! Computes the layout of Metal buffer slots for each kernel parameter:
//! data buffer index, byte length, and (for strided params) shape +
//! strides metadata.  Shared by [`super::single_dispatch`] and
//! [`super::chain_dispatch`].

use std::{borrow::Cow, collections::BTreeMap};

use metaltile_core::{
    ir::{Kernel, Param, ParamKind},
    shape::Dim,
};
use smallvec::SmallVec;

use crate::error::MetalTileError;

/// Inline capacity for shape/stride vectors. Covers tensor rank up to 6,
/// which fits every kernel currently in `metaltile-std` (the tallest is
/// rank‑5 conv3d weight tensors). Beyond 6 falls back to heap.
const INLINE_RANK: usize = 6;
type DimVec = SmallVec<[u32; INLINE_RANK]>;

// ---------------------------------------------------------------------------
// Buffer planning
// ---------------------------------------------------------------------------

/// Describes the Metal binding for one kernel parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParamBufferPlan {
    pub data_binding_index: usize,
    pub data_len: usize,
}

/// Number of Metal buffer slots a parameter consumes.
pub(crate) fn binding_slots(param: &Param) -> usize {
    if param.kind == ParamKind::Strided { 3 } else { 1 }
}

/// Compute the byte length of a parameter from its static shape, or
/// `None` when any dimension is unknown.
fn static_buffer_len(param: &Param) -> Result<Option<usize>, MetalTileError> {
    let Some(num_elements) = param.shape.num_elements() else {
        return Ok(None);
    };
    num_elements
        .checked_mul(param.dtype.size_bytes())
        .ok_or_else(|| {
            MetalTileError::Buffer(format!("buffer '{}' size overflows usize", param.name))
        })
        .map(Some)
}

/// Resolve the byte length needed for one parameter.
///
/// Uses the caller‑supplied buffer length when provided; falls back to
/// the static shape length.  Output parameters always use the larger
/// of the two so the caller can pre‑allocate space.
pub(crate) fn planned_data_len(
    param: &Param,
    buffers: &BTreeMap<String, Vec<u8>>,
) -> Result<usize, MetalTileError> {
    let provided_len = buffers.get(&param.name).map_or(0, Vec::len);
    let static_len = static_buffer_len(param)?;

    if let Some(expected_len) = static_len {
        if provided_len > 0 && provided_len < expected_len {
            return Err(MetalTileError::Buffer(format!(
                "buffer '{}' has {} bytes, expected at least {}",
                param.name, provided_len, expected_len
            )));
        }
        if param.is_output {
            return Ok(provided_len.max(expected_len));
        }
    }

    Ok(provided_len)
}

/// Build a binding plan for every parameter in the kernel.
pub(crate) fn build_param_buffer_plans(
    kernel: &Kernel,
    buffers: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<ParamBufferPlan>, MetalTileError> {
    let mut next_binding_index = 0usize;
    let mut plans = Vec::with_capacity(kernel.params.len());
    for param in &kernel.params {
        plans.push(ParamBufferPlan {
            data_binding_index: next_binding_index,
            data_len: planned_data_len(param, buffers)?,
        });
        next_binding_index += binding_slots(param);
    }
    Ok(plans)
}

// ---------------------------------------------------------------------------
// Strided metadata
// ---------------------------------------------------------------------------

/// Shape + strides byte data for a strided parameter.
pub(crate) type StridedMetadata<'a> = (Cow<'a, [u8]>, Cow<'a, [u8]>);

/// Pack `u32` values as little‑endian bytes.
pub(crate) fn encode_u32s(values: &[u32]) -> Vec<u8> {
    // Pre-allocate the exact byte count; flat_map's size_hint can't always
    // forward the inner `[u8; 4]` exact size, leaving `collect` to grow.
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Extract the known dimensions of a parameter's shape, or `None` if
/// any dimension is dynamic. Storage is stack-resident up to rank
/// [`INLINE_RANK`].
fn known_shape_dims(param: &Param) -> Result<Option<DimVec>, MetalTileError> {
    let mut dims: DimVec = SmallVec::with_capacity(param.shape.rank());
    for dim in param.shape.iter() {
        let Dim::Known(value) = dim else {
            return Ok(None);
        };
        dims.push(u32::try_from(*value).map_err(|_| {
            MetalTileError::Buffer(format!(
                "shape dimension for '{}' exceeds u32: {}",
                param.name, value
            ))
        })?);
    }
    Ok(Some(dims))
}

/// Compute row‑major strides for the given dimensions. Stack-resident
/// up to rank [`INLINE_RANK`].
fn row_major_strides(name: &str, dims: &[u32]) -> Result<DimVec, MetalTileError> {
    let mut strides: DimVec = SmallVec::from_elem(1u32, dims.len());
    let mut stride = 1u32;
    for (idx, &dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride.checked_mul(dim).ok_or_else(|| {
            MetalTileError::Buffer(format!("row-major strides for '{}' overflowed u32", name))
        })?;
    }
    Ok(strides)
}

/// Resolve shape and stride byte arrays for a strided parameter.
///
/// Looks for `{name}_shape` and `{name}_strides` in the buffer map.
/// When missing, synthesises them from the parameter's static shape.
/// Reuses a single `String` buffer for both lookups to avoid
/// allocations.
pub(crate) fn resolve_strided_metadata<'a>(
    param: &Param,
    buffers: &'a BTreeMap<String, Vec<u8>>,
) -> Result<StridedMetadata<'a>, MetalTileError> {
    let expected_len = param.shape.rank() * std::mem::size_of::<u32>();
    let defaults = known_shape_dims(param)?
        .map(|dims| {
            let strides = row_major_strides(&param.name, &dims)?;
            Ok::<(Vec<u8>, Vec<u8>), MetalTileError>((encode_u32s(&dims), encode_u32s(&strides)))
        })
        .transpose()?;

    // Single key buffer, reused across the two lookups: allocate once
    // (capacity = name + "_strides" — the longer suffix) and rewrite
    // the suffix in place.  Replaces two `format!` allocations per
    // strided param per dispatch.
    let mut key = String::with_capacity(param.name.len() + 8);
    key.push_str(&param.name);
    let prefix_len = key.len();
    key.push_str("_shape");

    let shape_data = match buffers.get(&key) {
        Some(bytes) => {
            if expected_len > 0 && bytes.len() < expected_len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer '{}' has {} bytes, expected at least {}",
                    key,
                    bytes.len(),
                    expected_len
                )));
            }
            Cow::Borrowed(bytes.as_slice())
        },
        None => {
            let Some((shape_bytes, _)) = defaults.as_ref() else {
                return Err(MetalTileError::Buffer(format!(
                    "missing required strided metadata buffer '{}'",
                    key
                )));
            };
            Cow::Owned(shape_bytes.clone())
        },
    };

    key.truncate(prefix_len);
    key.push_str("_strides");

    let strides_data = match buffers.get(&key) {
        Some(bytes) => {
            if expected_len > 0 && bytes.len() < expected_len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer '{}' has {} bytes, expected at least {}",
                    key,
                    bytes.len(),
                    expected_len
                )));
            }
            Cow::Borrowed(bytes.as_slice())
        },
        None => {
            let Some((_, strides_bytes)) = defaults.as_ref() else {
                return Err(MetalTileError::Buffer(format!(
                    "missing required strided metadata buffer '{}'",
                    key
                )));
            };
            Cow::Owned(strides_bytes.clone())
        },
    };

    Ok((shape_data, strides_data))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use metaltile_core::{
        dtype::DType,
        ir::{Kernel, ParamKind},
        shape::Shape,
    };

    use super::*;

    fn tensor_param(
        name: &str,
        dtype: DType,
        dims: &[usize],
        is_output: bool,
        kind: ParamKind,
    ) -> Param {
        Param {
            name: name.into(),
            dtype,
            shape: Shape::new(dims.iter().copied().map(metaltile_core::shape::Dim::Known)),
            is_output,
            kind,
        }
    }

    #[test]
    fn buffer_plans_follow_binding_indices_and_static_output_sizes() {
        let mut kernel = Kernel::new("binding_plan_test");
        kernel.params = vec![
            tensor_param("input", DType::F32, &[2, 2], false, ParamKind::Strided),
            tensor_param("out_a", DType::F32, &[4], true, ParamKind::Tensor),
            tensor_param("out_b", DType::F32, &[4], true, ParamKind::Tensor),
        ];

        let mut buffers = BTreeMap::new();
        buffers.insert("input".into(), vec![0u8; 16]);
        buffers.insert("out_b".into(), vec![0u8; 16]);

        let plans = build_param_buffer_plans(&kernel, &buffers).expect("buffer plans");
        assert_eq!(plans, vec![
            ParamBufferPlan { data_binding_index: 0, data_len: 16 },
            ParamBufferPlan { data_binding_index: 3, data_len: 16 },
            ParamBufferPlan { data_binding_index: 4, data_len: 16 },
        ]);
    }

    #[test]
    fn strided_metadata_defaults_to_row_major_shape_and_strides() {
        let param = tensor_param("input", DType::F32, &[2, 3, 4], false, ParamKind::Strided);
        let buffers = BTreeMap::new();

        let (shape_data, stride_data) =
            resolve_strided_metadata(&param, &buffers).expect("default strided metadata");

        assert_eq!(shape_data.as_ref(), encode_u32s(&[2, 3, 4]).as_slice());
        assert_eq!(stride_data.as_ref(), encode_u32s(&[12, 4, 1]).as_slice());
    }

    /// Side-by-side: the pre-PR Vec<u32> implementation of
    /// `known_shape_dims` + `row_major_strides`, kept ONLY in the test
    /// module so we can quantify the SmallVec win without revert-and-
    /// re-measure noise. Synced with the public impls — if they
    /// change, mirror the change here.
    fn known_shape_dims_vec(param: &Param) -> Option<Vec<u32>> {
        let mut dims = Vec::with_capacity(param.shape.rank());
        for dim in param.shape.iter() {
            let Dim::Known(value) = dim else {
                return None;
            };
            dims.push(*value as u32);
        }
        Some(dims)
    }

    fn row_major_strides_vec(dims: &[u32]) -> Vec<u32> {
        let mut strides = vec![1u32; dims.len()];
        let mut stride = 1u32;
        for (idx, &dim) in dims.iter().enumerate().rev() {
            strides[idx] = stride;
            stride *= dim;
        }
        strides
    }

    fn resolve_strided_metadata_vec(param: &Param) -> (Vec<u8>, Vec<u8>) {
        let dims = known_shape_dims_vec(param).expect("known dims");
        let strides = row_major_strides_vec(&dims);
        (encode_u32s(&dims), encode_u32s(&strides))
    }

    fn resolve_strided_metadata_smallvec(param: &Param) -> (Vec<u8>, Vec<u8>) {
        let dims = known_shape_dims(param).unwrap().expect("known dims");
        let strides = row_major_strides(&param.name, &dims).unwrap();
        (encode_u32s(&dims), encode_u32s(&strides))
    }

    #[test]
    #[ignore = "perf microbench"]
    fn perf_strided_metadata_vec_vs_smallvec() {
        // Modal rank=3 strided KV cache shape.
        let param =
            tensor_param("kv_cache", DType::F16, &[8, 4096, 128], false, ParamKind::Strided);

        const ITERS: usize = 5_000_000;
        for _ in 0..50_000 {
            std::hint::black_box(resolve_strided_metadata_vec(std::hint::black_box(&param)));
            std::hint::black_box(resolve_strided_metadata_smallvec(std::hint::black_box(&param)));
        }

        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(resolve_strided_metadata_vec(std::hint::black_box(&param)));
        }
        let vec_elapsed = t0.elapsed();
        let vec_ns_per = vec_elapsed.as_nanos() as f64 / ITERS as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(resolve_strided_metadata_smallvec(std::hint::black_box(&param)));
        }
        let sv_elapsed = t1.elapsed();
        let sv_ns_per = sv_elapsed.as_nanos() as f64 / ITERS as f64;

        println!();
        println!(
            "=== rank-3 strided metadata: 2× Vec<u32> intermediates vs 2× SmallVec ({ITERS} iters) ==="
        );
        println!("  Vec<u32>   (old): {vec_elapsed:>10.2?}  ({vec_ns_per:>5.1} ns/call)");
        println!("  SmallVec   (new): {sv_elapsed:>10.2?}  ({sv_ns_per:>5.1} ns/call)");
        let speedup = vec_ns_per / sv_ns_per;
        println!(
            "  → speedup        : {speedup:.2}× ({:+.1}%)",
            (1.0 - sv_ns_per / vec_ns_per) * 100.0
        );

        // Regression assertion: SmallVec must not regress within 5%.
        assert!(
            sv_ns_per * 1.05 <= vec_ns_per,
            "SmallVec resolve_strided_metadata ({sv_ns_per:.1} ns/call) should beat the Vec \
             baseline ({vec_ns_per:.1} ns/call)"
        );
    }

    #[test]
    #[ignore = "perf microbench"]
    fn perf_build_param_buffer_plans_modal_kernel() {
        // Modal FFAI kernel: 8 params, 2 strided. Bench reflects what
        // every single_dispatch.execute() does once per call.
        let mut kernel = Kernel::new("perf_modal_8param");
        kernel.params = vec![
            tensor_param("q", DType::F16, &[1024, 128], false, ParamKind::Strided),
            tensor_param("k", DType::F16, &[1024, 128], false, ParamKind::Strided),
            tensor_param("v", DType::F16, &[1024, 128], false, ParamKind::Tensor),
            tensor_param("mask", DType::F16, &[1024], false, ParamKind::Tensor),
            tensor_param("out", DType::F16, &[1024, 128], true, ParamKind::Tensor),
            tensor_param("scale_a", DType::F32, &[1], false, ParamKind::Tensor),
            tensor_param("scale_b", DType::F32, &[1], false, ParamKind::Tensor),
            tensor_param("bias", DType::F32, &[128], false, ParamKind::Tensor),
        ];
        let mut buffers = BTreeMap::new();
        buffers.insert("q".into(), vec![0u8; 1024 * 128 * 2]);
        buffers.insert("k".into(), vec![0u8; 1024 * 128 * 2]);

        const ITERS: usize = 1_000_000;
        for _ in 0..20_000 {
            std::hint::black_box(
                build_param_buffer_plans(&kernel, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(
                build_param_buffer_plans(&kernel, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        println!(
            "build_param_buffer_plans(8-param, 2-strided) × {ITERS}: {elapsed:?} ({ns_per_call:.1} \
             ns/call)"
        );
    }

    #[test]
    #[ignore = "perf microbench"]
    fn perf_resolve_strided_metadata() {
        let param = tensor_param(
            "long_strided_kv_cache",
            DType::F32,
            &[8, 4096, 128],
            false,
            ParamKind::Strided,
        );
        let buffers = BTreeMap::new();
        const ITERS: usize = 5_000_000;
        for _ in 0..50_000 {
            std::hint::black_box(
                resolve_strided_metadata(&param, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(
                resolve_strided_metadata(&param, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        println!("resolve_strided_metadata × {ITERS}: {elapsed:?} ({ns_per_call:.1} ns/call)");
    }
}

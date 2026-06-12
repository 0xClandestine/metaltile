//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Core / elementwise primitive kernels — the ops family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): binary / unary / ternary
//! elementwise, copy (+ strided), arange, random, reduce / arg_reduce, scan,
//! indexing, gather / scatter, hadamard, fence, logsumexp, clamp, axpy, and
//! vector add. Migrated from the legacy `mlx/` + `ffai/` split.
//!
//! `ffai/arg_reduce.rs` (`ffai_argmax`) was a byte-for-byte duplicate of
//! `mt_argmax` (in `arg_reduce.rs`) and was dropped, not moved.

pub mod arange;
pub mod arg_reduce;
pub mod axpy_scalar_inplace;
pub mod binary;
pub mod binary_two;
pub mod clamp_scalar;
pub mod copy;
// `fence.rs` is intentionally NOT declared: device/system memory fences and
// atomics have no `#[kernel]` DSL representation (the "1 out of scope" op in the
// audit). The file is kept for the implementation notes in its header.
pub mod gather;
pub mod gather_axis;
pub mod hadamard;
pub mod hadamard_m;
pub mod indexing;
pub mod logsumexp;
pub mod random;
pub mod reduce;
pub mod scan;
pub mod scatter_axis;
pub mod strided;
pub mod ternary;
pub mod unary;
pub mod vector_add;

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Normalization kernels — the norm family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). RMSNorm and its fused forms
//! (residual add, RoPE, gated, and the quantized RMSNorm→GEMV fusions),
//! LayerNorm, and 1-D adaptive instance norm. Kernels are named for the
//! operation; the quantized fusions carry the weight format as a name axis
//! (`mt_<format>_rms_norm_qgemv`).
//!
//! Migrated from the legacy `mlx/` (`rms_norm`, `layer_norm`) + `ffai/` split.
//! The `*_block_scaled_qgemv` files hold the per-format matrix and will fold
//! into the base `*_qgemv` files as a format axis in a later pass (plan §7).

pub mod adain1d;
pub mod gated_rms_norm_block_scaled_qgemv;
pub mod gated_rms_norm_qgemv;
pub mod gated_rmsnorm;
pub mod layer_norm;
pub mod rms_norm;
pub mod rms_norm_block_scaled_qgemv;
pub mod rms_norm_qgemv;
pub mod rms_norm_residual;
pub mod rms_norm_rope;

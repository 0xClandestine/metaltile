//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Rotary position embedding (RoPE) kernels — the rope family, migrated from
//! the legacy `mlx/` + `ffai/` split. Next consolidation steps (per
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): merge `rope_llama` +
//! `rope_llama_many`, fold `dsv4_partial_rope[_rows]` into one `partial_rope`,
//! and drop the `ffai_`/`dsv4_` name prefixes once the FFAI emit consumer is
//! updated.

pub mod base;
pub mod dsv4_partial_rope;
pub mod dsv4_partial_rope_rows;
pub mod rope_2d;
pub mod rope_llama;
pub mod rope_llama_many;
pub mod rope_yarn;

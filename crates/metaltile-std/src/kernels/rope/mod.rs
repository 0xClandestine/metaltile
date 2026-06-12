//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Rotary position embedding (RoPE) kernels — the rope family, migrated from
//! the legacy `mlx/` + `ffai/` split (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). All kernels carry the `mt_`
//! prefix; the decode + batched-prefill forms share a file:
//!   - `base`         — `mt_rope` (MLX `rope.metal` reference parity)
//!   - `rope_llama`   — `mt_rope_llama` (decode) + `mt_rope_llama_many` (prefill)
//!   - `partial_rope` — `mt_partial_rope` (decode) + `mt_partial_rope_rows`
//!   - `rope_2d`      — `mt_rope_2d` (vision M-RoPE)
//!   - `rope_yarn`    — `mt_rope_yarn`

pub mod base;
pub mod partial_rope;
pub mod rope_2d;
pub mod rope_llama;
pub mod rope_yarn;

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Rotary position embedding (RoPE) kernels — the rope family, migrated from
//! the legacy `mlx/` + `ffai/` split (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). All kernels carry the `mt_`
//! prefix. The position-batched kernels take a per-row/per-token position plus
//! an outer grid axis, so decode is just the single-row case (no separate
//! decode kernel):
//!   - `base`         — `mt_rope` (MLX `rope.metal` reference parity)
//!   - `rope_llama`   — `mt_rope_llama` (per-row `positions`; decode = T=1)
//!   - `partial_rope` — `mt_partial_rope` (DSv4 tail RoPE; decode = n_tokens=1)
//!   - `rope_2d`      — `mt_rope_2d` (vision M-RoPE)
//!   - `rope_yarn`    — `mt_rope_yarn`

pub mod base;
pub mod partial_rope;
pub mod rope_2d;
pub mod rope_llama;
pub mod rope_yarn;

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Kernel standard library, grouped by **operation family** — the consolidation
//! target (see `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). Families migrate here
//! one PR at a time from the legacy `mlx/` + `ffai/` folders; `pub fn` names are
//! preserved across the move so the kernel inventory / FFAI emit are unaffected.

pub mod rope;

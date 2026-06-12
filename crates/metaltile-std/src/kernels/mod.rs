//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Kernel standard library, grouped by **operation family** — the consolidation
//! target (see `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). Families migrate here
//! one PR at a time from the legacy `mlx/` + `ffai/` folders. Kernels are named
//! for the **operation / layout** they implement (`mt_<op>`), never for a model;
//! model-specific usage notes live in comments above the kernel. The FFAI emit
//! consumer is regenerated from the new inventory after each family lands.

pub mod convolution;
pub mod norm;
pub mod rope;
